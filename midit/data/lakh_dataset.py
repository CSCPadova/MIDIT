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
from miditok import REMI, TokenizerConfig  # ← Aggiungi questo se non c'è
from midit.data.masking import BarLevelMasking


class LakhDataset(Dataset):
    """Dataset for Lakh MIDI preprocessed sequences.
    
    Loads full sequences and chunks them on-the-fly based on config.
    Supports both bar-based and token-based chunking strategies.
    """
    
    def __init__(
        self,
        data_dir: str,
        chunking_config: DictConfig,
        split: str = 'train',
        apply_masking: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing .pt files
            chunking_config: Chunking configuration from Hydra
            split: 'train' or 'val'
            apply_masking: Whether to apply masking (True for train, False for val)
        """
        self.data_dir = Path(data_dir)
        self.chunking = chunking_config
        self.split = split
        self.apply_masking = apply_masking
        
        # Load all .pt files
        self.files = sorted(list(self.data_dir.glob("*.pt")))
        
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")
        
        # ← NUOVO: Initialize tokenizer (for vocab)
        config = TokenizerConfig(
            pitch_range=(21, 109),
            beat_res={(0, 4): 8},
            num_velocities=32,
            use_chords=False,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=True,
            special_tokens=["PAD", "BOS", "EOS", "MASK"],
            use_controls=False,
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
        print(f"\n🔍 Pre-computing vocab token lists...")
                
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

        print(f"✅ Token lists pre-computed:")
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
                
        print(f"\n📊 Actual vocab sizes:")
        for token_type, size in self.actual_vocab_sizes.items():
            print(f"   {token_type:10s}: {size:>4d}")
        print()
        # ← NUOVO: Initialize masker
        if self.apply_masking:
            self.masker = BarLevelMasking(
                tokenizer_vocab=self.tokenizer.vocab,
                mask_token_id=self.mask_token_id,
                bar_mask_prob=0.15,
                type_mask_ratio=0.3,
            )
        else:
            self.masker = None
        
        print(f"\n✅ LakhDataset initialized")
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
        
        # Create targets dict
        targets = self._create_targets_dict(chunk_tokens)
        
        # Adjust bar positions to chunk-relative
        chunk_bar_positions = [
            pos - start_idx 
            for pos in bar_positions[start_bar:end_bar+1]
        ]
        
        return {
            'midi_tokens': chunk_tokens,
            'targets': targets,
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
        
        # Create targets dict
        targets = self._create_targets_dict(chunk_tokens)
        
        return {
            'midi_tokens': chunk_tokens,
            'targets': targets,
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
        
        # Process each token using pre-computed lists
        for pos in range(seq_len):
            token_id = tokens[pos].item()
            
            # Get token name from reverse vocab
            if token_id not in self.id_to_token:
                continue
            
            token_name = self.id_to_token[token_id]
            
            # ========================================
            # PITCH
            # ========================================
            if token_name.startswith('Pitch_'):
                try:
                    # Use pre-computed list
                    if token_name in self.pitch_tokens:
                        pitch_idx = self.pitch_tokens.index(token_name)
                        targets['pitch'][pos] = pitch_idx
                except (ValueError, IndexError):
                    pass
            
            elif token_name.startswith('PitchDrum_'):
                try:
                    # Drums also map to pitch space
                    if token_name in self.pitchdrum_tokens:
                        pitch_idx = self.pitchdrum_tokens.index(token_name)
                        # Offset by regular pitch tokens
                        targets['pitch'][pos] = len(self.pitch_tokens) + pitch_idx
                except (ValueError, IndexError):
                    pass
            
            # ========================================
            # VELOCITY
            # ========================================
            elif token_name.startswith('Velocity_'):
                try:
                    if token_name in self.velocity_tokens:
                        velocity_idx = self.velocity_tokens.index(token_name)
                        targets['velocity'][pos] = velocity_idx
                except (ValueError, IndexError):
                    pass
            
            # ========================================
            # DURATION
            # ========================================
            elif token_name.startswith('Duration_'):
                try:
                    # Use pre-computed list (NO MORE RECONSTRUCTION!)
                    if token_name in self.duration_tokens:
                        duration_idx = self.duration_tokens.index(token_name)
                        targets['duration'][pos] = duration_idx
                except (ValueError, IndexError):
                    pass
            
            # ========================================
            # PROGRAM
            # ========================================
            elif token_name.startswith('Program_'):
                try:
                    if token_name in self.program_tokens:
                        program_idx = self.program_tokens.index(token_name)
                        targets['program'][pos] = program_idx
                except (ValueError, IndexError):
                    pass
            
            # ========================================
            # POSITION
            # ========================================
            elif token_name.startswith('Position_'):
                try:
                    if token_name in self.position_tokens:
                        position_idx = self.position_tokens.index(token_name)
                        targets['position'][pos] = position_idx
                except (ValueError, IndexError):
                    pass
            
            # ========================================
            # BAR
            # ========================================
            elif token_name.startswith('Bar_'):
                try:
                    # Use pre-computed list (NOT always 0!)
                    if token_name in self.bar_tokens:
                        bar_idx = self.bar_tokens.index(token_name)
                        targets['bar'][pos] = bar_idx
                except (ValueError, IndexError):
                    pass
            
            # ========================================
            # TEMPO
            # ========================================
            elif token_name.startswith('Tempo_'):
                try:
                    # Use pre-computed list (NOT hard-coded formula!)
                    if token_name in self.tempo_tokens:
                        tempo_idx = self.tempo_tokens.index(token_name)
                        targets['tempo'][pos] = tempo_idx
                except (ValueError, IndexError):
                    pass
            
            # ========================================
            # TIME SIGNATURE
            # ========================================
            elif token_name.startswith('TimeSig_'):
                try:
                    # Use pre-computed mapping
                    if token_name in self.timesig_to_class:
                        targets['timesig'][pos] = self.timesig_to_class[token_name]
                except (ValueError, IndexError, KeyError):
                    pass
            
            # ========================================
            # REST
            # ========================================
            elif token_name.startswith('Rest_'):
                try:
                    # Use pre-computed list (NO MORE RECONSTRUCTION!)
                    if token_name in self.rest_tokens:
                        rest_idx = self.rest_tokens.index(token_name)
                        targets['rest'][pos] = rest_idx
                except (ValueError, IndexError):
                    pass
        
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