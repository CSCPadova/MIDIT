"""
MidiCaps Real DataLoader - versione con lettura diretta dei file MIDI
Autore: Simone Zanetti & ChatGPT
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from typing import Optional
from datasets import load_dataset
from transformers import T5Tokenizer
import pretty_midi
import numpy as np

def midi_to_tokens(file_path: str, max_len: int = 1024):
    """Converte un file MIDI in una sequenza di token semplici."""
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
    except Exception as e:
        print(f"[Warning] Errore nel file {file_path}: {e}")
        return torch.zeros(max_len, dtype=torch.long)

    tokens = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            tokens.append(1000 + note.pitch)
            tokens.append(2000 + int(note.velocity // 8))  # 0-127 -> 0-15 bins
            shift = int(note.start * 100) % 500
            tokens.append(3000 + shift)

# Padding/truncation
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    return torch.tensor(tokens, dtype=torch.long)

class MidiCapsRealDataset(Dataset):
    """Dataset per MidiCaps - con conversione MIDI reale"""
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        fast_run: bool = False,
    ):
        super().__init__()
        print(f"🔹 Caricamento dataset MidiCaps reale ({split})...")
        self.data_dir = data_dir
        self.split = split
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.fast_run = fast_run

    # Carica dataset HuggingFace
        hf_split = "train" if split in ["val", "validation"] else split
        subset = "[:100]" if fast_run else "[:2000]"
        self.dataset = load_dataset(
            "amaai-lab/MidiCaps",
             split=f"{hf_split}{subset}",
             cache_dir=data_dir,
        )
        print(f"✓ {len(self.dataset)} esempi caricati")

        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        caption = example.get("caption", "")
        midi_path = example.get("midi_filename") or example.get("midi_file")

        # Se non c'è percorso MIDI valido → skip
        if not midi_path or not os.path.exists(midi_path):
            midi_tokens = torch.zeros(self.max_midi_length, dtype=torch.long)
        else:
            midi_tokens = midi_to_tokens(midi_path, self.max_midi_length)

        # Tokenize testo
        text_encoding = self.tokenizer(
            caption,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_tokens = text_encoding["input_ids"].squeeze(0)
        text_attention_mask = text_encoding["attention_mask"].squeeze(0).float()

        # Mask MIDI
        midi_mask = (midi_tokens != 0).float()

        return {
            "midi_tokens": midi_tokens,
            "midi_attention_mask": midi_mask,
            "text_tokens": text_tokens,
            "text_attention_mask": text_attention_mask,
        }
class MidiCapsRealDataModule(L.LightningDataModule):
    """Lightning DataModule per il dataset MIDI reale"""
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 2,
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        fast_run: bool = False,
):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.fast_run = fast_run
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MidiCapsRealDataset(
                self.data_dir, "train", self.max_midi_length, self.max_text_length, self.fast_run
        )
            self.val_dataset = MidiCapsRealDataset(
                self.data_dir, "validation", self.max_midi_length, self.max_text_length, self.fast_run
        )

        if stage == "test" or stage is None:
            self.test_dataset = MidiCapsRealDataset(
                self.data_dir, "test", self.max_midi_length, self.max_text_length, self.fast_run
        )

def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True,
    )

def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
    )

def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
    )

