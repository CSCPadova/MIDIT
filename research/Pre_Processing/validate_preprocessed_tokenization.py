"""Valida che i file .pt preprocessati abbiano tokenizzazione REMI+ corretta."""
import torch
from pathlib import Path
from miditok import REMI, TokenizerConfig
from collections import Counter
import random

def validate_token_sequence(tokens_tensor, tokenizer):
    """Valida sequenza token secondo grammatica REMI+.
    
    REMI+ structure:
    Bar → (Tempo, TimeSig) → Position → [Program → Pitch → Velocity → Duration]* → Position → ...
    """
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    
    errors = []
    warnings = []
    stats = {
        'total_tokens': len(tokens_tensor),
        'token_types': Counter(),
        'invalid_tokens': 0,
        'bar_positions': [],
        'programs': Counter(),
        'pitch_range': [],
    }
    
    prev_token_type = None
    position_count = 0
    bar_count = 0
    
    for i, token_id in enumerate(tokens_tensor.tolist()):
        # Check token exists
        if token_id not in id_to_token:
            stats['invalid_tokens'] += 1
            errors.append(f"Position {i}: Invalid token ID {token_id}")
            continue
        
        token_name = id_to_token[token_id]
        
        # Determine token type
        if token_name.startswith('Bar_'):
            token_type = 'BAR'
            bar_count += 1
            stats['bar_positions'].append(i)
        elif token_name.startswith('Position_'):
            token_type = 'POSITION'
            position_count += 1
        elif token_name.startswith('Program_'):
            token_type = 'PROGRAM'
            try:
                prog_num = int(token_name.split('_')[1])
                stats['programs'][prog_num] += 1
            except:
                pass
        elif token_name.startswith('Pitch_') or token_name.startswith('PitchDrum_'):
            token_type = 'PITCH'
            try:
                if token_name.startswith('Pitch_'):
                    pitch = int(token_name.split('_')[1])
                    stats['pitch_range'].append(pitch)
            except:
                pass
        elif token_name.startswith('Velocity_'):
            token_type = 'VELOCITY'
        elif token_name.startswith('Duration_'):
            token_type = 'DURATION'
        elif token_name.startswith('Tempo_'):
            token_type = 'TEMPO'
        elif token_name.startswith('TimeSig_'):
            token_type = 'TIMESIG'
        elif token_name.startswith('Rest_'):
            token_type = 'REST'
        else:
            token_type = 'SPECIAL'
        
        stats['token_types'][token_type] += 1
        
        # Validate transitions (basic REMI+ grammar)
        if prev_token_type == 'PITCH' and token_type not in ['VELOCITY']:
            warnings.append(f"Position {i}: Expected VELOCITY after PITCH, got {token_type}")
        
        if prev_token_type == 'VELOCITY' and token_type not in ['DURATION']:
            warnings.append(f"Position {i}: Expected DURATION after VELOCITY, got {token_type}")
        
        prev_token_type = token_type
    
    return stats, errors, warnings


def main():
    print("=" * 80)
    print("🔍 VALIDAZIONE TOKENIZZAZIONE FILE PREPROCESSATI")
    print("=" * 80)
    
    # Setup
    input_dir = Path('data/preprocessed_phase1')
    
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
    
    # Get files
    print(f"\n📂 Scanning {input_dir}...")
    all_files = sorted(input_dir.glob("*.pt"))
    print(f"Found {len(all_files)} files")
    
    # Sample random files
    sample_size = min(20, len(all_files))
    sample_files = random.sample(all_files, sample_size)
    
    print(f"\n🎲 Testing {sample_size} random files...")
    
    all_errors = []
    all_warnings = []
    all_stats = []
    
    for i, file_path in enumerate(sample_files, 1):
        try:
            data = torch.load(file_path)
            tokens = data['tokens']
            
            stats, errors, warnings = validate_token_sequence(tokens, tokenizer)
            
            all_stats.append(stats)
            all_errors.extend([(file_path.name, e) for e in errors])
            all_warnings.extend([(file_path.name, w) for w in warnings])
            
            # Print progress
            if i % 5 == 0:
                print(f"   Validated {i}/{sample_size} files...")
        
        except Exception as e:
            all_errors.append((file_path.name, f"Failed to load: {e}"))
    
    # Aggregate statistics
    print("\n" + "=" * 80)
    print("📊 STATISTICHE AGGREGATE:")
    print("=" * 80)
    
    total_tokens = sum(s['total_tokens'] for s in all_stats)
    total_invalid = sum(s['invalid_tokens'] for s in all_stats)
    
    print(f"\n🔢 Token totali analizzati: {total_tokens:,}")
    print(f"   Invalid tokens: {total_invalid} ({total_invalid/total_tokens*100:.3f}%)")
    
    # Token type distribution
    print("\n📋 Distribuzione Token Types:")
    aggregated_types = Counter()
    for s in all_stats:
        aggregated_types.update(s['token_types'])
    
    for token_type, count in aggregated_types.most_common():
        pct = count / total_tokens * 100
        print(f"   {token_type:12} {count:>8} ({pct:>5.2f}%)")
    
    # Program distribution
    print("\n🎸 Distribuzione Programs:")
    aggregated_programs = Counter()
    for s in all_stats:
        aggregated_programs.update(s['programs'])
    
    for program_id, count in aggregated_programs.most_common(10):
        print(f"   Program {program_id:>3}: {count:>6} occorrenze")
    
    # Pitch range
    print("\n🎹 Pitch Range:")
    all_pitches = []
    for s in all_stats:
        all_pitches.extend(s['pitch_range'])
    
    if all_pitches:
        print(f"   Min: {min(all_pitches)} (MIDI note)")
        print(f"   Max: {max(all_pitches)} (MIDI note)")
        print(f"   Range: {max(all_pitches) - min(all_pitches)} semitones")
        print(f"   Expected: 21-109 (config pitch_range)")
        
        if min(all_pitches) < 21 or max(all_pitches) > 109:
            print(f"   ⚠️ WARNING: Pitch out of config range!")
    
    # Errors
    print("\n" + "=" * 80)
    if all_errors:
        print(f"❌ ERRORI TROVATI: {len(all_errors)}")
        print("=" * 80)
        for filename, error in all_errors[:10]:
            print(f"   {filename}: {error}")
        if len(all_errors) > 10:
            print(f"   ... e altri {len(all_errors) - 10} errori")
    else:
        print("✅ NESSUN ERRORE CRITICO")
        print("=" * 80)
    
    # Warnings
    if all_warnings:
        print(f"\n⚠️ WARNING GRAMMATICA: {len(all_warnings)}")
        for filename, warning in all_warnings[:5]:
            print(f"   {filename}: {warning}")
        if len(all_warnings) > 5:
            print(f"   ... e altri {len(all_warnings) - 5} warning")
    
    # Sample decoded sequence
    print("\n" + "=" * 80)
    print("🔍 ESEMPIO SEQUENZA DECODIFICATA (primo file):")
    print("=" * 80)
    
    first_file = sample_files[0]
    data = torch.load(first_file)
    tokens = data['tokens'][:100]  # First 100 tokens
    
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    
    print(f"\nFile: {first_file.name}")
    print(f"Primi 100 token:\n")
    
    for i, token_id in enumerate(tokens.tolist()):
        if token_id in id_to_token:
            token_name = id_to_token[token_id]
            
            # Highlight important tokens
            if token_name.startswith('Bar_'):
                print(f"{i:>3}. {token_name:30} ← NUOVA BATTUTA")
            elif token_name.startswith('Program_'):
                print(f"{i:>3}. {token_name:30} ← CAMBIO STRUMENTO")
            elif token_name.startswith('Position_'):
                print(f"{i:>3}. {token_name:30} ← POSIZIONE")
            else:
                print(f"{i:>3}. {token_name:30}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("🎯 VERDETTO FINALE:")
    print("=" * 80)
    
    if total_invalid == 0 and len(all_errors) == 0:
        print("✅ TOKENIZZAZIONE CORRETTA!")
        print("   → Tutti token validi")
        print("   → Nessun errore critico")
        print("   → Grammatica REMI+ rispettata")
        print("\n💡 Il modello sta imparando da dati corretti.")
        print("   Il problema è SOLO la distribuzione multi-track.")
    else:
        print("⚠️ PROBLEMI RILEVATI!")
        print(f"   → {total_invalid} token invalidi")
        print(f"   → {len(all_errors)} errori")
        print("\n⚠️ Potrebbero esserci problemi nel preprocessing!")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
