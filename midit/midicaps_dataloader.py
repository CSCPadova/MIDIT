"""DataLoader for MidiCaps dataset with real MIDI tokenization."""

from random import sample
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from typing import Optional
from pathlib import Path
import pretty_midi
import json

# -----------------------
# Helper Functions
# -----------------------
def quantize_duration(duration_sec, max_bins=256, max_sec=2.0):
    """Quantize duration into bins."""
    bin_size = max_sec / max_bins
    bin_idx = min(int(duration_sec / bin_size), max_bins - 1)
    return 256 + bin_idx

def quantize_velocity(velocity, max_bins=128):
    """Map velocity 0-127 to token range 128-255."""
    return 128 + min(velocity, max_bins - 1)

def quantize_time_delta(delta_sec, max_bins=256, max_sec=2.0):
    """Quantize delta time between notes."""
    bin_size = max_sec / max_bins
    bin_idx = min(int(delta_sec / bin_size), max_bins - 1)
    return 512 + bin_idx

def extract_notes_from_midi(midi: pretty_midi.PrettyMIDI):
    """Extract notes from all instruments in a MIDI object."""
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
    notes.sort(key=lambda x: x['start'])
    return notes

def tokenize_notes(notes, max_tokens=None):
    """Convert notes to 4-token-per-note representation."""
    tokens = []
    prev_start = 0.0
    for note in notes:
        pitch = note['pitch']
        velocity = quantize_velocity(note['velocity'])
        duration = quantize_duration(note['end'] - note['start'])
        time_delta = quantize_time_delta(note['start'] - prev_start)
        tokens.extend([pitch, velocity, duration, time_delta])
        prev_start = note['start']
        
        if max_tokens and len(tokens) >= max_tokens:
            break
    
    return tokens

# ✅ CUSTOM COLLATE FUNCTION
def collate_fn(batch):
    """Custom collate function to handle variable-length metadata."""
    # Stack tensors (tutti hanno stessa dimensione)
    midi_tokens = torch.stack([item['midi_tokens'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    midi_attention_mask = torch.stack([item['midi_attention_mask'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    
    # Metadata: tieni come liste (non tensori!)
    tempo = [item['tempo'] for item in batch]
    key = [item['key'] for item in batch]
    time_signature = [item['time_signature'] for item in batch]
    duration = [item['duration'] for item in batch]
    genre = [item['genre'] for item in batch]
    mood = [item['mood'] for item in batch]
    instruments = [item['instruments'] for item in batch]
    chords = [item['chords'] for item in batch]
    
    return {
        'midi_tokens': midi_tokens,
        'text_tokens': text_tokens,
        'midi_attention_mask': midi_attention_mask,
        'text_attention_mask': text_attention_mask,
        'tempo': tempo,
        'key': key,
        'time_signature': time_signature,
        'duration': duration,
        'genre': genre,
        'mood': mood,
        'instruments': instruments,
        'chords': chords,
    }

# -----------------------
# Dataset
# -----------------------
class MidiCapsDataset(Dataset):
    """Dataset for MidiCaps - text-to-MIDI pairs with local files."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
    
        json_path = self.data_dir / f"{split}.json"
        print(f"📖 Loading {json_path}...")
    
        self.samples = []
        skipped = 0
    
        with open(json_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
            
                # ✅ FILTRA CARTELLA 1 (vuota) NON SERVESU COMPUTER ? CI SONO I FILE.
                #if sample['location'].startswith(('lmd_full/0/', 'lmd_full/1/')):
                #    skipped += 1
                #    continue
            
                self.samples.append(sample)
            
            # Stop se raggiungiamo max_samples
                if max_samples and len(self.samples) >= max_samples:
                    break
    
        print(f"✅ Loaded {len(self.samples)} samples (skipped {skipped} from folder 1)")
    
        from transformers import T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        midi_path = self.data_dir / sample['location']

        
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            notes = extract_notes_from_midi(midi)
            midi_data = tokenize_notes(notes, max_tokens=self.max_midi_length)
        except Exception as e:
            print(f"⚠️ Error loading {midi_path}: {e}")
            midi_data = []
        
        midi_length = len(midi_data)
        if midi_length > self.max_midi_length:
            midi_data = midi_data[:self.max_midi_length]
            midi_length = self.max_midi_length
        
        padded_midi = torch.zeros(self.max_midi_length, dtype=torch.long)
        if midi_length > 0:
            padded_midi[:midi_length] = torch.tensor(midi_data, dtype=torch.long)
        
        midi_attention_mask = torch.zeros(self.max_midi_length, dtype=torch.float)
        midi_attention_mask[:midi_length] = 1.0
        
        caption = sample['caption']
        
        text_encoding = self.tokenizer(
            caption,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_tokens = text_encoding['input_ids'].squeeze(0)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0).float()
        
        return {
            "midi_tokens": padded_midi,
            "text_tokens": text_tokens,
            "midi_attention_mask": midi_attention_mask,
            "text_attention_mask": text_attention_mask,
            "tempo": sample['tempo'],
            "key": sample['key'],
            "time_signature": sample['time_signature'],
            "duration": sample['duration'],
            "genre": sample.get('genre', []),
            "mood": sample.get('mood', []),
            "instruments": sample.get('instrument_summary', []),
            "chords": sample.get('all_chords', []),
        }

# -----------------------
# DataModule
# -----------------------
class MidiCapsDataModule(L.LightningDataModule):
    """Lightning DataModule for MidiCaps dataset."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        fast_run: bool = False,
        max_samples: int = None,  # ← AGGIUNGI
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.fast_run = fast_run
        self.max_samples = max_samples  # ← AGGIUNGI
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets with proper split"""
    
        if stage == "fit" or stage is None:
            if self.fast_run:
                # Fast run: usa pochi sample
                self.train_dataset = MidiCapsDataset(
                    data_dir=self.data_dir,
                    split="train",
                    max_midi_length=self.max_midi_length,
                    max_text_length=self.max_text_length,
                    max_samples=100,
                )
                self.val_dataset = MidiCapsDataset(
                    data_dir=self.data_dir,
                    split="train",
                    max_midi_length=self.max_midi_length,
                    max_text_length=self.max_text_length,
                    max_samples=50,
                )
            else:
                # Full training: split corretto 90/10
                import torch
                from torch.utils.data import random_split
                
                # Carica dataset completo
                full_dataset = MidiCapsDataset(
                    data_dir=self.data_dir,
                    split="train",
                    max_midi_length=self.max_midi_length,
                    max_text_length=self.max_text_length,
                    max_samples=self.max_samples,
                )
                
                # Split 90% train, 10% val
                total_size = len(full_dataset)
                train_size = int(0.9 * total_size)  # 151,547
                val_size = total_size - train_size  # 16,838
                
                # Random split con seed fisso per reproducibilità
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
                print(f"📊 Dataset split:")
                print(f"   Train: {len(self.train_dataset)} samples (90%)")
                print(f"   Val:   {len(self.val_dataset)} samples (10%)")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,  # ✅ AGGIUNGI QUESTO!
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,  # ✅ AGGIUNGI QUESTO!
        )


# -----------------------
# Test Script
# -----------------------
if __name__ == "__main__":
    print("🧪 Testing MidiCapsDataset\n")
    print("="*80)
    
    dataset = MidiCapsDataset(
        data_dir="data/midi_files",
        split="train",
        max_samples=5
    )
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i+1} ---")
        item = dataset[i]
        
        print(f"Caption tokens: {item['text_tokens'][:10]}...")
        print(f"MIDI tokens: {item['midi_tokens'][:20].tolist()}")
        print(f"MIDI length: {item['midi_attention_mask'].sum().int().item()}")
        print(f"\n✅ Metadata:")
        print(f"   Tempo: {item['tempo']} BPM")
        print(f"   Key: {item['key']}")
        print(f"   Duration: {item['duration']}s")
        print(f"   Genres: {', '.join(item['genre'])}")
        print(f"   Instruments: {', '.join(item['instruments'][:3])}")
    
    print("\n" + "="*80)
    print("✅ DataLoader ready for training!")
    
    # ✅ Test DataModule con collate_fn
    print("\n🧪 Testing DataModule with batching...")
    dm = MidiCapsDataModule(
        data_dir="data/midi_files",
        batch_size=2,
        num_workers=0,
        fast_run=True
    )
    dm.setup("fit")
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\n📦 Batch shapes:")
    print(f"   midi_tokens: {batch['midi_tokens'].shape}")
    print(f"   text_tokens: {batch['text_tokens'].shape}")
    print(f"   midi_attention_mask: {batch['midi_attention_mask'].shape}")
    
    print(f"\n📋 Batch metadata (lists):")
    print(f"   tempo: {batch['tempo']}")
    print(f"   key: {batch['key']}")
    print(f"   genre: {batch['genre']}")
    
    print("\n✅ All tests passed! 🎉")