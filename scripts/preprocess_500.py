"""Preprocess 500 MIDI files from Lakh dataset."""

import argparse
import torch
from pathlib import Path
from miditok import REMI, TokenizerConfig
import statistics
from tqdm import tqdm
from functools import partial
import multiprocessing as mp


def process_single_file(args):
    """Process a single MIDI file with critical filtering."""
    midi_path, tokenizer, id_to_token, output_dir, file_idx = args
    
    try:
        # Tokenize
        tokens = tokenizer(str(midi_path))
        
        # Handle format
        if isinstance(tokens, list):
            all_ids = []
            for tok_seq in tokens:
                all_ids.extend(tok_seq.ids)
        else:
            all_ids = tokens.ids
        
        # ========================================
        # CRITICAL FILTERS (avoid training crashes)
        # ========================================
        if len(all_ids) < 50:
            return {'success': False, 'error': 'too_short', 'file': str(midi_path.name)}
        
        if len(all_ids) > 50000:
            return {'success': False, 'error': 'too_long', 'file': str(midi_path.name)}
        
        # Extract bar positions (FAST with pre-built reverse vocab)
        bar_positions = []
        for idx, token_id in enumerate(all_ids):
            token_name = id_to_token.get(token_id, None)
            if token_name and "Bar" in token_name:
                bar_positions.append(idx)
        
        # Clean bar positions
        if not bar_positions or bar_positions[0] != 0:
            bar_positions.insert(0, 0)
        bar_positions.append(len(all_ids))
        
        # Filter: too few bars
        if len(bar_positions) < 3:
            return {'success': False, 'error': 'too_few_bars', 'file': str(midi_path.name)}
        
        # Extract metadata
        num_bars = len(bar_positions) - 1
        tokens_in_bars = []
        for j in range(len(bar_positions) - 1):
            tokens_in_bar = bar_positions[j+1] - bar_positions[j]
            tokens_in_bars.append(tokens_in_bar)
        
        # Get musical metadata (first 100 tokens)
        first_tempo = None
        first_timesig = None
        programs_used = set()
        
        for token_id in all_ids[:100]:
            token_name = id_to_token.get(token_id, None)
            if token_name:
                if token_name.startswith('Tempo_') and first_tempo is None:
                    first_tempo = token_name
                elif token_name.startswith('TimeSig_') and first_timesig is None:
                    first_timesig = token_name
                elif token_name.startswith('Program_'):
                    programs_used.add(token_name)
        
        metadata = {
            'source_file': str(midi_path.name),
            'num_tokens': len(all_ids),
            'num_bars': num_bars,
            'tokens_per_bar_mean': statistics.mean(tokens_in_bars) if tokens_in_bars else 0,
            'tokens_per_bar_min': min(tokens_in_bars) if tokens_in_bars else 0,
            'tokens_per_bar_max': max(tokens_in_bars) if tokens_in_bars else 0,
            'initial_tempo': first_tempo,
            'initial_timesig': first_timesig,
            'num_programs': len(programs_used),
        }
        
        # Descriptive filename
        source_id = midi_path.stem[:20]
        output_filename = f"lakh_{file_idx:05d}_{source_id}.pt"
        #output_filename = f"lakh_{file_idx:04d}_{source_id}.pt" 
        output_path = output_dir / output_filename
        
        # Save
        data = {
            'tokens': torch.tensor(all_ids, dtype=torch.long),
            'bar_positions': bar_positions,
            'metadata': metadata,
        }
        
        torch.save(data, output_path)
        
        return {
            'success': True, 
            'tokens': len(all_ids), 
            'bars': num_bars,
            'file': str(midi_path.name)
        }
        
    except Exception as e:
        return {
            'success': False, 
            'error': str(e), 
            'file': str(midi_path.name)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/lakh_clean')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed_phase1')
    parser.add_argument('--max_files', type=int, default=None)
    #parser.add_argument('--max_files', type=int, default=500) passo a tutto il dataset.
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎵 LAKH MIDI PREPROCESSING - FULL DATASET")
    print("="*70 + "\n")
    
    # Setup tokenizer
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
    
    tokenizer = REMI(config)
    print(f"✅ Tokenizer initialized (vocab: {len(tokenizer.vocab)})")
    
    # Pre-build reverse vocab (CRITICAL OPTIMIZATION)
    print(f"   Building reverse vocab lookup...")
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    print(f"   ✅ Reverse vocab ready\n")
    
    # Paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find files
    midi_files = list(input_dir.rglob("*.mid"))
    
    if args.max_files is not None:
        midi_files = midi_files[:args.max_files]
    
    #if args.max_files:
    #   midi_files = midi_files[:args.max_files]
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Files to process: {len(midi_files):,}")
    print(f"⚙️  Workers: {args.num_workers}")
    print(f"\n🔍 Filters enabled:")
    print(f"   - Min tokens: 50")
    print(f"   - Max tokens: 50,000")
    print(f"   - Min bars: 3\n")
    
    # Prepare args
    process_args = [
        (midi_path, tokenizer, id_to_token, output_dir, i)
        for i, midi_path in enumerate(midi_files)
    ]
    
    # Process
    print("🚀 Processing...\n")
    
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(midi_files),
                desc="Processing"
            ))
    else:
        results = [process_single_file(arg) for arg in tqdm(process_args, desc="Processing")]
    
    # Statistics
    success = sum(1 for r in results if r['success'])
    errors = sum(1 for r in results if not r['success'])
    
    # Error breakdown
    error_types = {}
    for r in results:
        if not r['success']:
            err = r.get('error', 'unknown')
            error_types[err] = error_types.get(err, 0) + 1
    
    print(f"\n{'='*70}")
    print("📊 STATISTICS")
    print(f"{'='*70}\n")
    print(f"✅ Success: {success:,}/{len(midi_files):,} ({100*success/len(midi_files):.1f}%)")
    print(f"❌ Filtered: {errors:,}/{len(midi_files):,} ({100*errors/len(midi_files):.1f}%)\n")
    
    if error_types:
        print(f"📊 Filter breakdown:")
        for err, count in sorted(error_types.items()):
            print(f"   - {err}: {count}")
        print()
    
    if success > 0:
        successful = [r for r in results if r['success']]
        total_tokens = sum(r['tokens'] for r in successful)
        total_bars = sum(r['bars'] for r in successful)
        
        print(f"📊 Processed data:")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Avg tokens/file: {total_tokens/success:,.0f}")
        print(f"   Total bars: {total_bars:,}")
        print(f"   Avg bars/file: {total_bars/success:.1f}\n")
        
        # Disk usage
        total_size = sum(f.stat().st_size for f in output_dir.glob("*.pt"))
        print(f"💾 Disk usage:")
        print(f"   Total: {total_size/1024/1024:.1f} MB")
        print(f"   Per file: {total_size/success/1024:.1f} KB\n")
    
    print(f"{'='*70}")
    print("✅ PREPROCESSING COMPLETE!")
    print(f"{'='*70}\n")
    
    print(f"📁 Output: {output_dir}")
    print(f"📊 Files created: {success:,}\n")
    
    if success > 0:
        print(f"💡 Next: Train on these {success} files with SLURM")
    else:
        print(f"⚠️  No files processed successfully - check filters!")


if __name__ == "__main__":
    main()