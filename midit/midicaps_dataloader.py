"""DataLoader for MidiCaps dataset."""

import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from typing import Optional
import random


class MidiCapsDataset(Dataset):
    """Dataset for MidiCaps - text-to-MIDI pairs.
    
    This is a simplified mock implementation. In production, this would load
    from the actual HuggingFace dataset.
    """
    
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
            midi_vocab_size: Size of MIDI vocabulary
            text_vocab_size: Size of text vocabulary
            fast_run: If True, use a small subset for fast testing
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.midi_vocab_size = midi_vocab_size
        self.text_vocab_size = text_vocab_size
        self.fast_run = fast_run
        
        # TODO: Replace with actual HuggingFace dataset loading
        # For now, create synthetic data for testing
        if fast_run:
            self.length = 100  # Small dataset for fast testing
        else:
            self.length = 10000 if split == "train" else 1000
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item
        
        Returns:
            Dictionary containing midi_tokens, text_tokens, and attention masks
        """
        # TODO: Replace with actual data loading
        # For now, generate synthetic data
        
        # Random MIDI sequence
        midi_length = random.randint(128, self.max_midi_length) if not self.fast_run else 128
        midi_tokens = torch.randint(0, self.midi_vocab_size, (midi_length,))
        
        # Pad to max length
        padded_midi = torch.zeros(self.max_midi_length, dtype=torch.long)
        padded_midi[:midi_length] = midi_tokens
        midi_attention_mask = torch.zeros(self.max_midi_length, dtype=torch.float)
        midi_attention_mask[:midi_length] = 1.0
        
        # Random text sequence
        text_length = random.randint(10, self.max_text_length) if not self.fast_run else 32
        text_tokens = torch.randint(0, self.text_vocab_size, (text_length,))
        
        # Pad to max length
        padded_text = torch.zeros(self.max_text_length, dtype=torch.long)
        padded_text[:text_length] = text_tokens
        text_attention_mask = torch.zeros(self.max_text_length, dtype=torch.float)
        text_attention_mask[:text_length] = 1.0
        
        return {
            "midi_tokens": padded_midi,
            "text_tokens": padded_text,
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
        """Initialize the MidiCaps DataModule.
        
        Args:
            data_dir: Directory containing the dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            max_midi_length: Maximum length of MIDI sequences
            max_text_length: Maximum length of text sequences
            midi_vocab_size: Size of MIDI vocabulary
            text_vocab_size: Size of text vocabulary
            val_split: Validation split ratio
            fast_run: If True, use small datasets for fast testing
        """
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
        """Download and prepare data.
        
        TODO: Implement actual HuggingFace dataset download
        from datasets import load_dataset
        load_dataset("amaai-lab/MidiCaps", cache_dir=self.data_dir)
        """
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
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
