"""Filtra dataset preprocessato per singolo program."""
import torch
from pathlib import Path
from tqdm import tqdm
from miditok import REMI, TokenizerConfig


def analyze_program_tokens(tokens_tensor, tokenizer):
    """Analizza quali program sono presenti."""
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    
    program_counts = {}
    
    for token_id in tokens_tensor.tolist():
        if token_id not in id_to_token:
            continue
        
        token_name = id_to_token[token_id]
        
        if token_name.startswith('Program_'):
            try:
                prog_num = int(token_name.split('_')[1])
                program_counts[prog_num] = program_counts.get(prog_num, 0) + 1
            except:
                pass
    
    return program_counts


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/preprocessed_phase1')
    parser.add_argument('--output_dir', default='data/preprocessed_phase1_piano')
    parser.add_argument('--program', type=int, default=0)
    parser.add_argument('--min_program_ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"🎹 FILTER PREPROCESSED DATASET - Program {args.program}")
    print("=" * 80)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    print(f"✅ Tokenizer ready (vocab: {len(tokenizer.vocab)})")
    
    print(f"\n📂 Scanning {input_dir}...")
    files = sorted(input_dir.glob("*.pt"))
    print(f"Found {len(files)} preprocessed files")
    
    if len(files) == 0:
        print("❌ No files found!")
        return
    
    print(f"\n🎵 Analyzing & filtering...")
    
    kept = 0
    skipped = 0
    
    for file_path in tqdm(files, desc="Processing"):
        try:
            data = torch.load(file_path)
            tokens = data['tokens']
            
            program_counts = analyze_program_tokens(tokens, tokenizer)
            
            if not program_counts:
                skipped += 1
                continue
            
            target_count = program_counts.get(args.program, 0)
            total_program_tokens = sum(program_counts.values())
            
            if total_program_tokens == 0:
                skipped += 1
                continue
            
            ratio = target_count / total_program_tokens
            
            if ratio >= args.min_program_ratio:
                output_path = output_dir / file_path.name
                torch.save(data, output_path)
                kept += 1
            else:
                skipped += 1
        
        except Exception as e:
            print(f"\n⚠️ Error: {file_path.name}: {e}")
            skipped += 1
    
    print("\n" + "=" * 80)
    print("📊 FILTERING SUMMARY:")
    print("=" * 80)
    print(f"   Input files:       {len(files)}")
    print(f"   ✅ Kept:            {kept} ({kept/len(files)*100:.1f}%)")
    print(f"   ⚠️  Filtered out:    {skipped} ({skipped/len(files)*100:.1f}%)")
    print(f"\n📁 Output: {output_dir}")
    print(f"   Files: {len(list(output_dir.glob('*.pt')))}")
    print("=" * 80)
    
    if kept > 0:
        print("\n🔍 Sample check:")
        first_file = sorted(output_dir.glob("*.pt"))[0]
        data = torch.load(first_file)
        
        program_counts = analyze_program_tokens(data['tokens'], tokenizer)
        print(f"   File: {first_file.name}")
        print(f"   Programs: {program_counts}")
        
        target_count = program_counts.get(args.program, 0)
        total = sum(program_counts.values())
        print(f"   Program {args.program}: {target_count}/{total} = {target_count/total*100:.1f}%")


if __name__ == '__main__':
    main()
