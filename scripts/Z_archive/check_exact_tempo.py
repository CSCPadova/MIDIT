"""Check exact tempo in original MIDI file - FIXED VERSION."""

import miditoolkit
from pathlib import Path

print("\n" + "="*70)
print("🔬 EXACT TEMPO CHECK (FIXED)")
print("="*70 + "\n")

# Load original MIDI
midi_path = Path("data/lakh_clean/.38 Special/Caught Up In You.mid")

if not midi_path.exists():
    print(f"❌ File not found: {midi_path}")
    exit(1)

midi = miditoolkit.MidiFile(str(midi_path))

print(f"📁 File: {midi_path.name}\n")

# Tempo changes
print(f"📊 Tempo Information:\n")

if len(midi.tempo_changes) == 0:
    print("   ⚠️  No tempo changes in file!")
    print("   Using default: 120 BPM\n")
else:
    print(f"   Number of tempo changes: {len(midi.tempo_changes)}\n")
    
    for i, tempo_change in enumerate(midi.tempo_changes):
        time_tick = tempo_change.time
        tempo_value = tempo_change.tempo
        
        # Check if tempo_value is already BPM or in microseconds
        # If value < 1000, it's probably already BPM
        # If value > 10000, it's in microseconds per quarter
        
        if tempo_value < 1000:
            # Already BPM
            bpm = tempo_value
            us_per_quarter = 60_000_000 / bpm
            print(f"   Tempo change {i+1}:")
            print(f"      Time:  {time_tick} ticks")
            print(f"      BPM:   {bpm:.6f} (direct)")
            print(f"      μs/quarter: {us_per_quarter:.1f}")
        else:
            # Microseconds per quarter
            us_per_quarter = tempo_value
            bpm = 60_000_000 / us_per_quarter
            print(f"   Tempo change {i+1}:")
            print(f"      Time:  {time_tick} ticks")
            print(f"      μs/quarter: {us_per_quarter:.1f}")
            print(f"      BPM:   {bpm:.6f}")
        
        print()

# Check reconstructed
print(f"{'-'*70}\n")

reconstructed_path = Path("data/test_preprocessing/reconstructed.mid")

if reconstructed_path.exists():
    print(f"📁 Reconstructed: {reconstructed_path.name}\n")
    
    midi_recon = miditoolkit.MidiFile(str(reconstructed_path))
    
    if len(midi_recon.tempo_changes) == 0:
        print("   ⚠️  No tempo changes in reconstructed file!")
    else:
        print(f"   Number of tempo changes: {len(midi_recon.tempo_changes)}\n")
        
        for i, tempo_change in enumerate(midi_recon.tempo_changes):
            tempo_value = tempo_change.tempo
            
            if tempo_value < 1000:
                bpm = tempo_value
                us_per_quarter = 60_000_000 / bpm
                print(f"   Tempo change {i+1}:")
                print(f"      BPM:   {bpm:.6f} (direct)")
                print(f"      μs/quarter: {us_per_quarter:.1f}")
            else:
                us_per_quarter = tempo_value
                bpm = 60_000_000 / us_per_quarter
                print(f"   Tempo change {i+1}:")
                print(f"      μs/quarter: {us_per_quarter:.1f}")
                print(f"      BPM:   {bpm:.6f}")
            
            print()

# Comparison
print(f"{'='*70}")
print("📊 COMPARISON")
print(f"{'='*70}\n")

if len(midi.tempo_changes) > 0 and len(midi_recon.tempo_changes) > 0:
    # Get BPM from both
    orig_tempo = midi.tempo_changes[0].tempo
    recon_tempo = midi_recon.tempo_changes[0].tempo
    
    # Convert to BPM if needed
    if orig_tempo < 1000:
        orig_bpm = orig_tempo
    else:
        orig_bpm = 60_000_000 / orig_tempo
    
    if recon_tempo < 1000:
        recon_bpm = recon_tempo
    else:
        recon_bpm = 60_000_000 / recon_tempo
    
    diff_bpm = recon_bpm - orig_bpm
    diff_pct = 100 * diff_bpm / orig_bpm
    
    print(f"   Original BPM:      {orig_bpm:.6f}")
    print(f"   Reconstructed BPM: {recon_bpm:.6f}")
    print(f"   Difference:        {diff_bpm:+.6f} BPM ({diff_pct:+.3f}%)")
    
    if abs(diff_pct) < 0.5:
        print(f"\n   ✅ Difference < 0.5% - EXCELLENT")
    elif abs(diff_pct) < 1.0:
        print(f"\n   ⚠️  Difference < 1% - ACCEPTABLE")
    else:
        print(f"\n   ❌ Difference > 1% - NEEDS INVESTIGATION")
    
    # Additional check
    if abs(diff_bpm) < 0.1:
        print(f"   ✅ Absolute difference < 0.1 BPM - NEGLIGIBLE")
    elif abs(diff_bpm) < 1.0:
        print(f"   ⚠️  Absolute difference < 1 BPM - MINOR")
    else:
        print(f"   ⚠️  Absolute difference > 1 BPM - NOTICEABLE")

print()