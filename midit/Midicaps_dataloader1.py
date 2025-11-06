"""DataLoader for MidiCaps dataset."""

import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from typing import Optional
import random


class MidiCapsDataset(Dataset):
    """Dataset for MidiCaps - text-to-MIDI pairs."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        midi_vocab_size: int = 512,
        text_vocab_size: int = 10000,
        fast_run: bool = False,
    ):
        """Initialize the MidiCaps dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('train', 'validation', 'test')
            max_midi_length: Maximum length of MIDI sequences
            max_text_length: Maximum length of text sequences
            midi_vocab_size: Size of MIDI vocabulary (not used with real data)
            text_vocab_size: Size of text vocabulary (not used with real data)
            fast_run: If True, use a small subset for fast testing
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.fast_run = fast_run
        
        # Load HuggingFace dataset
        from datasets import load_dataset
        from transformers import T5Tokenizer
        
        print(f"Loading MidiCaps dataset (split: {split})...")
        
        # Map split names (HF uses 'train', 'validation', 'test')
        hf_split = "train" if split in ["val", "validation"] else split
        
        if fast_run:
            # Small subset for testing
            self.dataset = load_dataset(
                "amaai-lab/MidiCaps",
                split=f"{hf_split}[:100]",
                cache_dir=data_dir
            )
        else:
            # Full dataset
            self.dataset = load_dataset(
                "amaai-lab/MidiCaps",
                split=f"{hf_split}[:5000]",
                cache_dir=data_dir
            )
        
        print(f"✓ Loaded {len(self.dataset)} examples")
        
        # Initialize tokenizer for text
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        
        self.length = len(self.dataset)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item
        
        Returns:
            Dictionary containing midi_tokens, text_tokens, and attention masks
        """
        # Load real example
        example = self.dataset[idx]
        
        # === MIDI Tokens (from metadata) ===
        # Generate synthetic tokens from musical metadata
        midi_data = []
        
        # Use tempo (convert to token)
        if 'tempo' in example and example['tempo']:
            midi_data.extend([100 + int(example['tempo']) % 400])
        
        # Use key signature
        if 'key' in example:
            key_token = abs(hash(str(example['key']))) % 128
            midi_data.extend([200 + key_token])
        # Time signature
        if 'time_signature' in example and example['time_signature']:
            ts = str(example['time_signature'])
            ts_token = abs(hash(ts)) % 64
            midi_data.append(500 + ts_token)
        #Duration
        if 'duration' in example and example['duration']:
            duration_seconds = int(example['duration'])
            duration_bin = min(duration_seconds // 60, 10)  # Max bin = 10
            midi_data.append(600 + duration_bin)
        
        # Use chord summary
        if 'chord_summary' in example and example['chord_summary']:
            chords = example['chord_summary'] 
            if isinstance(chords, list):
                for chord in chords[:10]:
                    if isinstance(chord, str):
                        chord_token = abs(hash(chord)) % 128
                        midi_data.append(300 + chord_token)   
        
        if 'instrument_summary' in example and example['instrument_summary']:
            instruments = example['instrument_summary']
            if isinstance(instruments, list):
                for inst in instruments[:5]:  # Max 5 instruments
                    if isinstance(inst, str):
                        inst_token = abs(hash(inst)) % 128
                        midi_data.append(700 + inst_token)
        
        if 'genre' in example and example['genre']:
            genres = example['genre']
            if isinstance(genres, list):
                for genre in genres[:3]:  # Max 3 genres
                    if isinstance(genre, str):
                        genre_token = abs(hash(genre)) % 64
                        midi_data.append(800 + genre_token)
        
        if 'mood' in example and example['mood']:
            moods = example['mood']
            if isinstance(moods, list):
                for mood in moods[:3]:  # Max 3 moods
                    if isinstance(mood, str):
                        mood_token = abs(hash(mood)) % 64
                        midi_data.append(900 + mood_token)

        # Pad with random notes if too short
        min_length = 100
        if len(midi_data) < min_length:
            pattern_length = min(20, len(midi_data))
            pattern = midi_data[:pattern_length] if pattern_length > 0 else [1]
    
         # Repeat pattern until we reach min_length
            while len(midi_data) < min_length:
        # Add pattern, but don't exceed min_length
                remaining = min_length - len(midi_data)
                midi_data.extend(pattern[:remaining])

        midi_tokens = torch.tensor(midi_data[:self.max_midi_length], dtype=torch.long)
        midi_length = len(midi_tokens)
        
        # Truncate if too long
        if midi_length > self.max_midi_length:
            midi_tokens = midi_tokens[:self.max_midi_length]
            midi_length = self.max_midi_length
        
        # Pad to max length
        padded_midi = torch.zeros(self.max_midi_length, dtype=torch.long)
        padded_midi[:midi_length] = midi_tokens
        
        # Create attention mask
        midi_attention_mask = torch.zeros(self.max_midi_length, dtype=torch.float)
        midi_attention_mask[:midi_length] = 1.0
        
        # === Text Caption ===
        caption = example['caption']
        
        # Tokenize with FLAN-T5
        text_encoding = self.tokenizer(
            caption,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        text_tokens = text_encoding['input_ids'].squeeze(0)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0).float()
        
        return {
            "midi_tokens": padded_midi,
            "text_tokens": text_tokens,
            "midi_attention_mask": midi_attention_mask,
            "text_attention_mask": text_attention_mask,
        }


class MidiCapsDataModule(L.LightningDataModule):
    """Lightning DataModule for MidiCaps dataset."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        midi_vocab_size: int = 512,
        text_vocab_size: int = 10000,
        val_split: float = 0.1,
        fast_run: bool = False,
    ):
        """Initialize the MidiCaps DataModule."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.midi_vocab_size = midi_vocab_size
        self.text_vocab_size = text_vocab_size
        self.val_split = val_split
        self.fast_run = fast_run
    
    def prepare_data(self):
        """Download and prepare data (called only once)."""
        from datasets import load_dataset
        
        print("Preparing MidiCaps dataset...")
        # This downloads/caches the dataset
        load_dataset("amaai-lab/MidiCaps", cache_dir=self.data_dir)
        print("✓ Dataset prepared!")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = MidiCapsDataset(
                data_dir=self.data_dir,
                split="train",
                max_midi_length=self.max_midi_length,
                max_text_length=self.max_text_length,
                midi_vocab_size=self.midi_vocab_size,
                text_vocab_size=self.text_vocab_size,
                fast_run=self.fast_run,
            )
            
            self.val_dataset = MidiCapsDataset(
                data_dir=self.data_dir,
                split="validation",
                max_midi_length=self.max_midi_length,
                max_text_length=self.max_text_length,
                midi_vocab_size=self.midi_vocab_size,
                text_vocab_size=self.text_vocab_size,
                fast_run=self.fast_run,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = MidiCapsDataset(
                data_dir=self.data_dir,
                split="test",
                max_midi_length=self.max_midi_length,
                max_text_length=self.max_text_length,
                midi_vocab_size=self.midi_vocab_size,
                text_vocab_size=self.text_vocab_size,
                fast_run=self.fast_run,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )