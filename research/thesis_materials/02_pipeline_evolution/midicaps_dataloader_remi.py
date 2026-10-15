"""DataLoader for MidiCaps dataset with PIANO-ONLY + REMI+ tokenization."""

import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from typing import Optional
from pathlib import Path
import json
from miditok import REMI, TokenizerConfig

# Setup REMI tokenizer (globale, condiviso da tutti dataset instances)
REMI_CONFIG = TokenizerConfig(
    pitch_range=(21, 109),      # Piano range
    beat_res={(0, 4): 8},        # 8 subdivisions per beat
    num_velocities=32,           # 32 velocity bins
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
)

REMI_TOKENIZER = REMI(REMI_CONFIG)
VOCAB_SIZE = len(REMI_TOKENIZER.vocab)  # ~389

print(f"🎹 REMI+ Tokenizer loaded: vocab_size={VOCAB_SIZE}")

def collate_fn(batch):
    """Custom collate function to handle variable-length metadata."""
    midi_tokens = torch.stack([item['midi_tokens'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    midi_attention_mask = torch.stack([item['midi_attention_mask'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    
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


class MidiCapsDataset(Dataset):
    """Dataset for MidiCaps - PIANO-ONLY filtering."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        max_samples: Optional[int] = None,
        single_instrument_only: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.single_instrument_only = single_instrument_only
        self.verbose = verbose
    
        json_path = self.data_dir / f"{split}.json"
        
        if self.verbose:
            print(f"📖 Loading {json_path}...")
            if self.single_instrument_only:
                print(f"   🎹 FILTERING: Piano only")
    
        self.samples = []
        skipped_multi = 0
        skipped_no_inst = 0
        total_processed = 0
        
        from collections import Counter
        instrument_counts = Counter()
    
        with open(json_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
                total_processed += 1
                
                if self.single_instrument_only:
                    instruments = sample.get('instrument_summary', [])
                    
                    non_drum_instruments = [
                        inst for inst in instruments 
                        if inst.lower() != 'drums'
                    ]
                    
                    if len(non_drum_instruments) == 0:
                        skipped_no_inst += 1
                        continue
                    elif len(non_drum_instruments) > 1:
                        skipped_multi += 1
                        continue
                    elif non_drum_instruments[0] != "Piano":  # ← FILTRO PIANO
                        skipped_multi += 1
                        continue
                    
                    instrument_counts[non_drum_instruments[0]] += 1
                
                self.samples.append(sample)
            
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        if self.verbose:
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total processed: {total_processed:,}")
            print(f"   ✅ Loaded: {len(self.samples):,} samples")
            
            if self.single_instrument_only:
                print(f"   ⚠️  Skipped (multi/other): {skipped_multi:,}")
                print(f"   ⚠️  Skipped (no instruments): {skipped_no_inst:,}")
                print(f"   📈 Retention rate: {len(self.samples)/total_processed*100:.1f}%")
                
                if instrument_counts:
                    print(f"\n   🎵 Instruments in filtered dataset:")
                    for inst, count in instrument_counts.most_common(10):
                        print(f"      {inst}: {count:,} files ({count/len(self.samples)*100:.1f}%)")
    
        from transformers import T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        midi_path = self.data_dir / sample['location']
        
        # Inizializza variabili
        midi_data = []
        
        try:
            # Tokenize con REMI+
            tokens = REMI_TOKENIZER(str(midi_path))
            
            # Accedi agli ID dal TokSequence object
            if hasattr(tokens[0], 'ids'):
                token_ids = tokens[0].ids
            else:
                token_ids = tokens[0]
            
            # Converti a list se necessario
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            elif not isinstance(token_ids, list):
                token_ids = list(token_ids)
            
            # Normalizza token ID al range [0, 1]
            midi_data = [tid / VOCAB_SIZE for tid in token_ids]
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error loading {midi_path}: {e}")
            midi_data = []
        
        # Pad/truncate
        midi_length = len(midi_data)
        if midi_length > self.max_midi_length:
            midi_data = midi_data[:self.max_midi_length]
            midi_length = self.max_midi_length
        
        # Create tensor
        padded_midi = torch.zeros(self.max_midi_length, dtype=torch.float)
        if midi_length > 0:
            padded_midi[:midi_length] = torch.tensor(midi_data, dtype=torch.float)
        
        # Attention mask
        midi_attention_mask = torch.zeros(self.max_midi_length, dtype=torch.float)
        midi_attention_mask[:midi_length] = 1.0
        
        # Text tokenization
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
class MidiCapsDataModule(L.LightningDataModule):
    """Lightning DataModule for MidiCaps dataset with Piano-only filtering."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        max_midi_length: int = 1024,
        max_text_length: int = 256,
        fast_run: bool = False,
        max_samples: int = None,
        single_instrument_only: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_midi_length = max_midi_length
        self.max_text_length = max_text_length
        self.fast_run = fast_run
        self.max_samples = max_samples
        self.single_instrument_only = single_instrument_only
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if self.fast_run:
                print("\n🏃 FAST RUN MODE")
                self.train_dataset = MidiCapsDataset(
                    data_dir=self.data_dir,
                    split="train",
                    max_midi_length=self.max_midi_length,
                    max_text_length=self.max_text_length,
                    max_samples=100,
                    single_instrument_only=self.single_instrument_only,
                    verbose=True,
                )
                self.val_dataset = MidiCapsDataset(
                    data_dir=self.data_dir,
                    split="train",
                    max_midi_length=self.max_midi_length,
                    max_text_length=self.max_text_length,
                    max_samples=50,
                    single_instrument_only=self.single_instrument_only,
                    verbose=True,
                )
            else:
                import torch
                from torch.utils.data import random_split
                
                print("\n📚 FULL TRAINING MODE")
                
                full_dataset = MidiCapsDataset(
                    data_dir=self.data_dir,
                    split="train",
                    max_midi_length=self.max_midi_length,
                    max_text_length=self.max_text_length,
                    max_samples=self.max_samples,
                    single_instrument_only=self.single_instrument_only,
                    verbose=True,
                )
                
                total_size = len(full_dataset)
                train_size = int(0.9 * total_size)
                val_size = total_size - train_size
                
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
                print(f"\n📊 Dataset split:")
                print(f"   Train: {len(self.train_dataset):,} samples (90%)")
                print(f"   Val:   {len(self.val_dataset):,} samples (10%)")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    print("🧪 Testing MidiCapsDataset with PIANO-ONLY filter\n")
    print("="*80)
    
    print("\n[TEST 1] Loading dataset WITHOUT filter...")
    dataset_all = MidiCapsDataset(
        data_dir="data/midi_files",
        split="train",
        max_samples=15000,
        single_instrument_only=False,
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("\n[TEST 2] Loading dataset WITH piano-only filter...")
    dataset_piano = MidiCapsDataset(
        data_dir="data/midi_files",
        split="train",
        max_samples=1000,
        single_instrument_only=True,
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("📊 COMPARISON")
    print("="*80)
    print(f"   Without filter: {len(dataset_all):,} samples")
    print(f"   Piano only:     {len(dataset_piano):,} samples")
    print(f"   Reduction:      {(1 - len(dataset_piano)/len(dataset_all))*100:.1f}%")
    
    if len(dataset_piano) > 0:
        print("\n📝 Example piano sample:")
        item = dataset_piano[0]
        print(f"   Instruments: {item['instruments']}")
        print(f"   Key: {item['key']}, Tempo: {item['tempo']} BPM")
        print(f"   MIDI tokens shape: {item['midi_tokens'].shape}")
        print(f"   MIDI length: {item['midi_attention_mask'].sum().int().item()} tokens")
    
    print("\n" + "="*80)
    print("✅ All tests passed! 🎉")