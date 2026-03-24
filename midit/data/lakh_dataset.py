"""
LakhDataset: PyTorch Dataset for preprocessed MIDI sequences
Implements flexible chunking (bar-based or token-based) on-the-fly
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from typing import Dict, Optional
from omegaconf import DictConfig
from miditok import REMI, TokenizerConfig 
from midit.data.masking import BarLevelMasking, TokenLevelMasking


class LakhDataset(Dataset):
    """Dataset for Lakh MIDI preprocessed sequences.
    
    Loads full sequences and chunks them on-the-fly based on config.
    Supports both bar-based and token-based chunking strategies.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        split: str = 'train',
        apply_masking: bool = True,
    ):
        """Initialize dataset.

        Args:
            cfg: Full Hydra config (reads cfg.chunking.preprocessed_dir, cfg.chunking, cfg.masking)
            split: 'train' or 'val'
            apply_masking: Whether to apply masking (True for train, False for val)
        """
        data_dir = cfg.chunking.preprocessed_dir
        chunking_config = cfg.chunking
        masking_config = cfg.masking

        self.data_dir = Path(data_dir)
        self.chunking = chunking_config
        self.split = split
        self.apply_masking = apply_masking

        # Load all .pt files
        all_files = sorted(list(self.data_dir.glob("*.pt")))

        # Split deterministically with fixed seed
        import numpy as np
        np.random.seed(42)  # Fixed seed for reproducibility
        indices = np.arange(len(all_files))
        np.random.shuffle(indices)

        # Split: 80% train, 10% val, 10% test
        n_train = int(0.8 * len(all_files))
        n_val = int(0.1 * len(all_files))

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # Select files based on split
        if split == 'train':
            self.files = [all_files[i] for i in sorted(train_indices)]
        elif split == 'val':
            self.files = [all_files[i] for i in sorted(val_indices)]
        elif split == 'test':
            self.files = [all_files[i] for i in sorted(test_indices)]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")
        
        # ← NUOVO: Initialize tokenizer (for vocab)
        config = TokenizerConfig(
            pitch_range=(21, 109),
            beat_res={(0, 4): 8, (4, 12): 4},
            num_velocities=32,
            use_chords=False,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=True,
            special_tokens=["PAD", "BOS", "EOS", "MASK"],
            use_controls=False,
            delete_equal_successive_tempo_changes=True,
            delete_equal_successive_time_sig_changes=True,
        )
        
        self.tokenizer = REMI(config)
        
        # ← NUOVO: Get MASK token ID
        self.mask_token_id = None
        for token_name, token_id in self.tokenizer.vocab.items():
            if 'MASK' in token_name:
                self.mask_token_id = token_id
                break
        
        if self.mask_token_id is None:
            raise ValueError("MASK token not found in tokenizer vocab!")
        
        # ========================================
        # Pre-compute token type lists (OPTIMIZATION)
        # ========================================
        print(f"\n[INFO] Pre-computing vocab token lists...")
                
        # Build reverse vocab for fast lookup
        self.id_to_token = {v: k for k, v in self.tokenizer.vocab.items()}
                
        # Pre-compute sorted lists for each token type
        self.pitch_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                if k.startswith('Pitch_')])
        self.pitchdrum_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                    if k.startswith('PitchDrum_')])
        self.velocity_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                    if k.startswith('Velocity_')])
        self.duration_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                    if k.startswith('Duration_')])
        self.program_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                    if k.startswith('Program_')])
        self.position_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                    if k.startswith('Position_')])
        self.bar_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                if k.startswith('Bar_')])
        self.tempo_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                if k.startswith('Tempo_')])
        self.timesig_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                    if k.startswith('TimeSig_')])
        self.rest_tokens = sorted([k for k in self.tokenizer.vocab.keys() 
                                if k.startswith('Rest_')])

        # Build time signature mapping
        self.timesig_to_class = {token: idx for idx, token in enumerate(self.timesig_tokens)}

        # Precomputed O(1) lookup: token_name → (type, class_idx)
        # Replaces list.index() calls in _create_targets_dict (O(N) → O(1) per token)
        self.token_to_class = {}
        for idx, name in enumerate(self.pitch_tokens):
            self.token_to_class[name] = ('pitch', idx)
        for idx, name in enumerate(self.pitchdrum_tokens):
            self.token_to_class[name] = ('pitch', len(self.pitch_tokens) + idx)
        for idx, name in enumerate(self.velocity_tokens):
            self.token_to_class[name] = ('velocity', idx)
        for idx, name in enumerate(self.duration_tokens):
            self.token_to_class[name] = ('duration', idx)
        for idx, name in enumerate(self.program_tokens):
            self.token_to_class[name] = ('program', idx)
        for idx, name in enumerate(self.position_tokens):
            self.token_to_class[name] = ('position', idx)
        for idx, name in enumerate(self.bar_tokens):
            self.token_to_class[name] = ('bar', idx)
        for idx, name in enumerate(self.tempo_tokens):
            self.token_to_class[name] = ('tempo', idx)
        for idx, name in enumerate(self.timesig_tokens):
            self.token_to_class[name] = ('timesig', self.timesig_to_class[name])
        for idx, name in enumerate(self.rest_tokens):
            self.token_to_class[name] = ('rest', idx)

        print(f"[OK] Token lists pre-computed:")
        print(f"   Pitch tokens: {len(self.pitch_tokens)}")
        print(f"   Velocity tokens: {len(self.velocity_tokens)}")
        print(f"   Duration tokens: {len(self.duration_tokens)}")
        print(f"   Program tokens: {len(self.program_tokens)}")
        print(f"   Position tokens: {len(self.position_tokens)}")
        print(f"   Bar tokens: {len(self.bar_tokens)}")
        print(f"   Tempo tokens: {len(self.tempo_tokens)}")
        print(f"   TimeSig tokens: {len(self.timesig_tokens)}")
        print(f"   Rest tokens: {len(self.rest_tokens)}")
                
        # Store actual vocab sizes
        self.actual_vocab_sizes = {
            'pitch': len(self.pitch_tokens) + len(self.pitchdrum_tokens),
            'velocity': len(self.velocity_tokens),
            'duration': len(self.duration_tokens),
            'program': len(self.program_tokens),
            'position': len(self.position_tokens),
            'bar': len(self.bar_tokens),
            'tempo': len(self.tempo_tokens),
            'timesig': len(self.timesig_tokens),
            'rest': len(self.rest_tokens),
        }
                
        print(f"\n[INFO] Actual vocab sizes:")
        for token_type, size in self.actual_vocab_sizes.items():
            print(f"   {token_type:10s}: {size:>4d}")
        print()
        # Initialize masker based on config strategy
        if self.apply_masking:
            use_bar_level = getattr(masking_config, 'bar_level', True) if masking_config is not None else True

            if use_bar_level:
                self.masker = BarLevelMasking(
                    tokenizer_vocab=self.tokenizer.vocab,
                    mask_token_id=self.mask_token_id,
                    bar_mask_prob=getattr(masking_config, 'bar_mask_prob', 0.4) if masking_config is not None else 0.4,
                    type_mask_ratio=getattr(masking_config, 'type_mask_ratio', 0.8) if masking_config is not None else 0.8,
                )
            else:
                self.masker = TokenLevelMasking(
                    mask_token_id=self.mask_token_id,
                    mask_ratio=getattr(masking_config, 'mask_prob', 0.15),
                    mask_strategy=getattr(masking_config, 'mask_strategy', 'bert'),
                    vocab_size=len(self.tokenizer.vocab),
                )
        else:
            self.masker = None
        
        print(f"\n[OK] LakhDataset initialized")
        print(f"   Split: {split}")
        print(f"   Files: {len(self.files)}")
        print(f"   Chunking strategy: {self.chunking.strategy}")
        if self.chunking.strategy == 'bar_based':
            print(f"   Bars per chunk: {self.chunking.bars_per_chunk}")
        else:
            print(f"   Tokens per chunk: {self.chunking.tokens_per_chunk}")
        print(f"   Max tokens: {self.chunking.max_tokens}")
        print(f"   Apply masking: {self.apply_masking}")
        if self.apply_masking:
            print(f"   MASK token ID: {self.mask_token_id}")
        print()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        """Get one chunk from a song.
        
        Args:
            idx: File index
        
        Returns:
            Dictionary with:
                - input_ids: Token IDs (masked if training) (seq_len,)
                - original_tokens: Original token IDs (seq_len,)
                - mask_labels: 1 where masked, 0 otherwise (seq_len,)
                - targets: Dict of targets per token type
                - bar_positions: Bar boundaries in chunk
                - metadata: Source info
        """
        # Load full sequence
        try:
            data = torch.load(self.files[idx])
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return None
        
        # Chunk based on strategy
        if self.chunking.strategy == 'bar_based':
            chunk = self._chunk_by_bars(data)
        else:
            chunk = self._chunk_by_tokens(data)
        
        if chunk is None:
            return None
        
        # Original tokens
        original_tokens = chunk['midi_tokens']
        
        # Apply masking if training
        if self.apply_masking and self.masker is not None:
            masked_tokens, mask_labels = self.masker.apply_masking(
                tokens=original_tokens,
                bar_positions=chunk.get('bar_positions', None),
            )
        else:
            # No masking (validation)
            masked_tokens = original_tokens.clone()
            mask_labels = torch.zeros_like(original_tokens, dtype=torch.bool)
        
        # Handle outliers (truncate if too long)
        if len(masked_tokens) > self.chunking.max_tokens:
            masked_tokens = masked_tokens[:self.chunking.max_tokens]
            original_tokens = original_tokens[:self.chunking.max_tokens]
            mask_labels = mask_labels[:self.chunking.max_tokens]
        
        # Create targets dict for ALL tokens first
        all_targets = self._create_targets_dict(original_tokens)
        
        # DEBUG 1: After creating all_targets
        #print(f"\n🔍 DEBUG 1 - after _create_targets_dict():")
        #for key in all_targets.keys():
            #non_ignore = (all_targets[key] != -100).sum().item()
            #if non_ignore > 0:
                #print(f"   {key}: {non_ignore} non-ignore values")
        #print(f"   Total tokens: {len(original_tokens)}")
        #print(f"   Masked positions: {mask_labels.sum().item()}")
        
        # Create empty targets (all -100)
        targets = {}
        for key in all_targets.keys():
            targets[key] = torch.full_like(all_targets[key], -100, dtype=torch.long)
        
        # Copy ONLY masked positions from all_targets
        for key in all_targets.keys():
            targets[key][mask_labels] = all_targets[key][mask_labels]
        
        # DEBUG 2: After copying masked positions
        #print(f"\n🔍 DEBUG 2 - after copying masked positions:")
        #for key in targets.keys():
            #non_ignore = (targets[key] != -100).sum().item()
            #masked_count = mask_labels.sum().item()
            #if non_ignore > 0:
                #print(f"   {key}: {non_ignore} non-ignore (should match {masked_count} masked)")
            #else:
                # Check why this type has no values
                #all_target_masked = (all_targets[key][mask_labels] != -100).sum().item()
                #print(f"   {key}: 0 non-ignore (all_targets had {all_target_masked} at masked pos)")
        #print()
        
        return {
            'input_ids': masked_tokens,
            'original_tokens': original_tokens,
            'mask_labels': mask_labels,
            'targets': targets,
            'bar_positions': chunk.get('bar_positions', None),
            'metadata': chunk['metadata'],
        }
    
    def _chunk_by_bars(self, data: Dict) -> Optional[Dict]:
        """Extract chunk of N bars.
        
        Args:
            data: Full sequence data
        
        Returns:
            Chunk dictionary or None if not enough bars
        """
        tokens = data['tokens']
        bar_positions = data['bar_positions']
        metadata = data['metadata']
        
        num_bars = len(bar_positions) - 1  # -1 because last is end position
        bars_per_chunk = self.chunking.bars_per_chunk
        
        # Check if song has enough bars
        if num_bars < bars_per_chunk:
            return None
        
        # Random start bar
        max_start = num_bars - bars_per_chunk
        start_bar = random.randint(0, max_start)
        end_bar = start_bar + bars_per_chunk
        
        # Extract tokens in this bar range
        start_idx = bar_positions[start_bar]
        end_idx = bar_positions[end_bar]
        
        chunk_tokens = tokens[start_idx:end_idx]

        # Adjust bar positions to chunk-relative
        chunk_bar_positions = [
            pos - start_idx
            for pos in bar_positions[start_bar:end_bar+1]
        ]

        return {
            'midi_tokens': chunk_tokens,
            'bar_positions': chunk_bar_positions,
            'metadata': {
                'source_file': metadata['source_file'],
                'start_bar': start_bar,
                'end_bar': end_bar,
                'chunk_length': len(chunk_tokens),
            }
        }
    
    def _chunk_by_tokens(self, data: Dict) -> Optional[Dict]:
        """Extract chunk of N tokens (ignores bar boundaries).
        
        Args:
            data: Full sequence data
        
        Returns:
            Chunk dictionary or None if sequence too short
        """
        tokens = data['tokens']
        metadata = data['metadata']
        tokens_per_chunk = self.chunking.tokens_per_chunk
        
        # Check if sequence has enough tokens
        if len(tokens) < tokens_per_chunk:
            return None
        
        # Random start position
        max_start = len(tokens) - tokens_per_chunk
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + tokens_per_chunk
        
        chunk_tokens = tokens[start_idx:end_idx]

        return {
            'midi_tokens': chunk_tokens,
            'bar_positions': None,  # Not applicable for token-based
            'metadata': {
                'source_file': metadata['source_file'],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'chunk_length': len(chunk_tokens),
            }
        }
    
    def _create_targets_dict(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create targets dictionary for multi-head loss.
        
        Maps each token to its type and extracts the value using pre-computed lists.
        
        Args:
            tokens: Token IDs tensor [seq_len]
        
        Returns:
            Dictionary with 9 token type tensors, each [seq_len]
            Values are class indices for that type, or -100 (ignore)
        """
        seq_len = len(tokens)
        
        # Initialize all targets to -100 (ignore)
        targets = {
            'pitch': torch.full((seq_len,), -100, dtype=torch.long),
            'velocity': torch.full((seq_len,), -100, dtype=torch.long),
            'duration': torch.full((seq_len,), -100, dtype=torch.long),
            'program': torch.full((seq_len,), -100, dtype=torch.long),
            'position': torch.full((seq_len,), -100, dtype=torch.long),
            'bar': torch.full((seq_len,), -100, dtype=torch.long),
            'tempo': torch.full((seq_len,), -100, dtype=torch.long),
            'timesig': torch.full((seq_len,), -100, dtype=torch.long),
            'rest': torch.full((seq_len,), -100, dtype=torch.long),
        }
        
        # O(1) lookup per token via precomputed token_to_class dict
        for pos in range(seq_len):
            token_name = self.id_to_token.get(tokens[pos].item())
            if token_name is None:
                continue
            entry = self.token_to_class.get(token_name)
            if entry is not None:
                token_type, class_idx = entry
                targets[token_type][pos] = class_idx
        
        return targets


# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch):
    """Collate function for DataLoader.
    
    Handles variable-length sequences with padding.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary with padded sequences
    """
    # Filter out None values (skipped chunks)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences to max length
    padded_input_ids = []
    padded_original_tokens = []
    padded_mask_labels = []
    padded_targets = {key: [] for key in batch[0]['targets'].keys()}
    attention_masks = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        # Pad input_ids (use 0 as PAD)
        padded = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        padded_input_ids.append(padded)
        
        # Pad original_tokens
        padded_orig = torch.cat([
            item['original_tokens'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        padded_original_tokens.append(padded_orig)
        
        # Pad mask_labels
        padded_mask = torch.cat([
            item['mask_labels'],
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        padded_mask_labels.append(padded_mask)
        
        # Pad targets (use -100 for ignore_index)
        for key in padded_targets.keys():
            target_padded = torch.cat([
                item['targets'][key],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
            padded_targets[key].append(target_padded)
        
        # Create attention mask (1 = real token, 0 = padding)
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.bool),
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        attention_masks.append(mask)
    
    # Stack everything
    input_ids = torch.stack(padded_input_ids)
    original_tokens = torch.stack(padded_original_tokens)
    mask_labels = torch.stack(padded_mask_labels)
    attention_mask = torch.stack(attention_masks)
    
    targets = {}
    for key in padded_targets.keys():
        targets[key] = torch.stack(padded_targets[key])
    
    return {
        'input_ids': input_ids,              # Masked tokens
        'original_tokens': original_tokens,  # Ground truth
        'mask_labels': mask_labels,          # Where masked
        'targets': targets,                  # For loss
        'attention_mask': attention_mask,    # Padding mask
        'metadata': [item['metadata'] for item in batch],
    }