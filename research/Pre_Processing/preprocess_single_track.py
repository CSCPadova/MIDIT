"""Re-preprocess Lakh dataset filtrando solo un singolo program (es. Piano).

Questo script:
1. Legge i MIDI originali dal Lakh dataset
2. Filtra solo le tracce con program specificato (default: 0 = Piano)
3. Tokenizza con REMI+ (stesso config del training)
4. Salva in formato compatibile con LakhDataset esistente
"""
import torch
from pathlib import Path
from miditok import REMI, TokenizerConfig
import symusic
from tqdm import tqdm
import argparse


def filter_midi_by_program(midi_path: Path, target_program: int = 0):
    """Carica MIDI e filtra solo tracce con program specificato.
    
    Args:
        midi_path: Path al file MIDI
        target_program: Program ID da mantenere (0 = Acoustic Grand Piano)
    
    Returns:
        symusic.Score filtrato o None se nessuna traccia match
    """
    try:
        score = symusic.Score(str(midi_path))
        
        # Filtra tracce per program
        filtered_tracks = []
        for track in score.tracks:
            if not track.is_drum and track.program == target_program:
                filtered_tracks.append(track)
        
        if not filtered_tracks:
            return None
        
        # Crea nuovo score con solo tracce filtrate
        new_score = symusic.Score(ticks_per_quarter=score.ticks_per_quarter)
        new_score.tracks = filtered_tracks
        new_score.tempos = score.tempos
        new_score.time_signatures = score.time_signatures
        
        # Verifica che ci siano note
        total_notes = sum(len(t.notes) for t in new_score.tracks)
        if total_notes == 0:
            return None
        
        return new_score
        
    except Exception as e:
        print(f"Error loading {midi_path.name}: {e}")
        return None


def tokenize_and_save(score, tokenizer, output_path: Path, source_file: str):
    """Tokenizza score e salva in formato dataset.
    
    Args:
        score: symusic.Score da tokenizzare
        tokenizer: REMI tokenizer
        output_path: Path dove salvare .pt
        source_file: Nome file sorgente per metadata
    """
    try:
        # Tokenizza (ritorna lista di token IDs)
        tokens = tokenizer(score)
        
        if len(tokens) == 0:
            return False
        
        # Converti a tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Trova bar positions (per bar-level masking)
        bar_positions = []
        for i, token_id in enumerate(tokens):
            token_name = tokenizer.vocab.token_to_event.get(token_id, "")
            if isinstance(token_name, str) and token_name.startswith('Bar_'):
                bar_positions.append(i)
        
        # Aggiungi posizione finale
        if len(bar_positions) > 0:
            bar_positions.append(len(tokens))
        else:
            # Fallback se no bar tokens
            bar_positions = [0, len(tokens)]
        
        bar_positions_tensor = torch.tensor(bar_positions, dtype=torch.long)
        
        # Salva in formato compatibile con LakhDataset
        data = {
            'tokens': tokens_tensor,
            'bar_positions': bar_positions_tensor,
            'metadata': {
                'source_file': source_file,
                'num_tokens': len(tokens),
                'num_bars': len(bar_positions) - 1,
            }
        }
        
        torch.save(data, output_path)
        return True
        
    except Exception as e:
        print(f"Error tokenizing {source_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess Lakh dataset for single program")
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Path to Lakh MIDI Clean directory')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed_phase1_piano',
                       help='Output directory for preprocessed files')
    parser.add_argument('--program', type=int, default=0,
                       help='Program ID to filter (0=Piano, 24=Guitar, 32=Bass, etc.)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Max files to process (for testing)')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🎹 SINGLE-TRACK PREPROCESSING - Program {args.program}")
    print("=" * 80)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer (SAME config as training!)
    print("\n🔤 Initializing tokenizer...")
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
    tokenizer = REMI(tokenizer_config=config)
    print(f"✅ Tokenizer ready (vocab size: {len(tokenizer.vocab)})")
    
    # Get all MIDI files
    print(f"\n📂 Scanning {input_dir}...")
    midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))
    
    if args.max_files:
        midi_files = midi_files[:args.max_files]
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Process files
    print(f"\n🎵 Processing (filtering Program {args.program})...")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for i, midi_path in enumerate(tqdm(midi_files, desc="Processing")):
        # Filter by program
        score = filter_midi_by_program(midi_path, target_program=args.program)
        
        if score is None:
            skipped += 1
            continue
        
        # Generate output filename
        output_filename = f"lakh_{processed:05d}_{midi_path.stem}.pt"
        output_path = output_dir / output_filename
        
        # Tokenize and save
        success = tokenize_and_save(score, tokenizer, output_path, midi_path.name)
        
        if success:
            processed += 1
        else:
            errors += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PREPROCESSING SUMMARY:")
    print("=" * 80)
    print(f"   Total input files:    {len(midi_files)}")
    print(f"   ✅ Processed:          {processed}")
    print(f"   ⚠️  Skipped (no match):  {skipped}")
    print(f"   ❌ Errors:             {errors}")
    print(f"   Success rate:         {processed/len(midi_files)*100:.1f}%")
    print(f"\n📁 Output: {output_dir}")
    print(f"   Files created: {len(list(output_dir.glob('*.pt')))}")
    print("=" * 80)
    
    # Sample check
    if processed > 0:
        print("\n🔍 Sample check (first file):")
        first_file = sorted(output_dir.glob("*.pt"))[0]
        data = torch.load(first_file)
        print(f"   File: {first_file.name}")
        print(f"   Tokens: {len(data['tokens'])}")
        print(f"   Bars: {data['metadata']['num_bars']}")
        print(f"   First 20 tokens: {data['tokens'][:20].tolist()}")


if __name__ == '__main__':
    main()
