"""
Validation: Round-trip test MIDI → tokens → MIDI
"""

from pathlib import Path
from miditok import REMI, TokenizerConfig
import symusic
from collections import Counter

def compare_midi(original, reconstructed, name="comparison"):
    """Confronta due MIDI files."""
    
    print(f"\n{'='*70}")
    print(f"📊 {name.upper()}")
    print(f"{'='*70}\n")
    
    # Track counts
    print("🎹 TRACKS:")
    print(f"   Original:      {len(original.tracks)} tracks")
    print(f"   Reconstructed: {len(reconstructed.tracks)} tracks")
    
    # Note counts
    orig_notes = sum(len(t.notes) for t in original.tracks)
    recon_notes = sum(len(t.notes) for t in reconstructed.tracks)
    
    print(f"\n🎵 NOTES:")
    print(f"   Original:      {orig_notes} notes")
    print(f"   Reconstructed: {recon_notes} notes")
    
    if orig_notes > 0:
        diff_pct = 100 * abs(orig_notes - recon_notes) / orig_notes
        print(f"   Difference:    {diff_pct:.1f}%")
        
        if diff_pct < 5:
            print(f"   ✅ GOOD: < 5% difference")
        elif diff_pct < 15:
            print(f"   ⚠️  OK: < 15% difference")
        else:
            print(f"   ❌ BAD: > 15% difference")
    
    # Pitch range
    if orig_notes > 0:
        orig_pitches = [note.pitch for t in original.tracks for note in t.notes]
        recon_pitches = [note.pitch for t in reconstructed.tracks for note in t.notes]
        
        print(f"\n🎹 PITCH RANGE:")
        print(f"   Original:      {min(orig_pitches)}-{max(orig_pitches)}")
        print(f"   Reconstructed: {min(recon_pitches)}-{max(recon_pitches)}")
    
    # Duration (in ticks)
    orig_duration = max((note.end for t in original.tracks for note in t.notes), default=0)
    recon_duration = max((note.end for t in reconstructed.tracks for note in t.notes), default=0)
    
    print(f"\n⏱️  DURATION:")
    print(f"   Original:      {orig_duration} ticks")
    print(f"   Reconstructed: {recon_duration} ticks")
    
    if orig_duration > 0:
        diff_pct = 100 * abs(orig_duration - recon_duration) / orig_duration
        print(f"   Difference:    {diff_pct:.1f}%")
        
        if diff_pct < 5:
            print(f"   ✅ GOOD")
        else:
            print(f"   ⚠️  Check manually")
    
    # Tempo
    print(f"\n🎵 TEMPO:")
    print(f"   Original:      {len(original.tempos)} tempo changes")
    print(f"   Reconstructed: {len(reconstructed.tempos)} tempo changes")
    
    # Time signatures
    print(f"\n📏 TIME SIGNATURES:")
    print(f"   Original:      {len(original.time_signatures)} changes")
    print(f"   Reconstructed: {len(reconstructed.time_signatures)} changes")
    
    print()

def validate_tokenization(midi_path, tokenizer, output_dir):
    """Test round-trip su 1 file."""
    
    print(f"\n{'='*70}")
    print(f"🔬 VALIDATION: {midi_path.name}")
    print(f"{'='*70}\n")
    
    # Load original
    print("Step 1: Loading original MIDI...")
    original = symusic.Score(str(midi_path))
    print(f"   ✅ {len(original.tracks)} tracks, {sum(len(t.notes) for t in original.tracks)} notes")
    
    # Filter drums
    print("\nStep 2: Filtering drums...")
    non_drum_tracks = [t for t in original.tracks if not t.is_drum]
    print(f"   ✅ {len(non_drum_tracks)} non-drum tracks")
    
    if len(non_drum_tracks) == 0:
        print("   ❌ No non-drum tracks, skipping")
        return False
    
    # Create filtered version
    filtered = symusic.Score(original.ticks_per_quarter)
    filtered.tracks = non_drum_tracks
    filtered.tempos = original.tempos
    filtered.time_signatures = original.time_signatures
    filtered.key_signatures = original.key_signatures
    
    # Tokenize
    print("\nStep 3: Tokenizing...")
    tokens = tokenizer(filtered)
    print(f"   ✅ {len(tokens.ids)} tokens generated")
    
    # Show token distribution
    token_strs = []
    for tid in tokens.ids:
        try:
            token_strs.append(tokenizer.vocab[tid])
        except:
            token_strs.append(f"ID:{tid}")
    
    token_types = Counter([t.split('_')[0] for t in token_strs])
    print(f"\n   Token type distribution:")
    for ttype, count in sorted(token_types.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / len(tokens.ids)
        print(f"      {ttype:15s}: {count:5d} ({pct:5.1f}%)")
    
    # Decode
    print("\nStep 4: Decoding tokens back to MIDI...")
    try:
        reconstructed = tokenizer.decode(tokens.ids)
        print(f"   ✅ Decoded to {len(reconstructed.tracks)} tracks")
    except Exception as e:
        print(f"   ❌ Decode failed: {e}")
        return False
    
    # Compare
    compare_midi(filtered, reconstructed, "Round-Trip Comparison")
    
    # Save files
    print(f"\n💾 Saving files for manual inspection...")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    orig_out = output_dir / f"original_{midi_path.stem}.mid"
    recon_out = output_dir / f"reconstructed_{midi_path.stem}.mid"
    
    filtered.dump_midi(str(orig_out))
    reconstructed.dump_midi(str(recon_out))
    
    print(f"   ✅ Original:      {orig_out}")
    print(f"   ✅ Reconstructed: {recon_out}")
    print(f"\n   🎧 Play both files and compare by ear!")
    
    return True

def main():
    """Run validation on multiple files."""
    
    print("\n" + "="*70)
    print("🔬 TOKENIZATION VALIDATION")
    print("="*70)
    
    # Setup tokenizer
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8},
        use_programs=True,
        use_time_signatures=True,
        use_tempos=True,
        use_chords=False,
        special_tokens=["PAD", "BOS", "EOS", "MASK"]
    )
    
    tokenizer = REMI(config)
    print(f"\n✅ Tokenizer: REMI, vocab size {len(tokenizer.vocab)}")
    
    # Test files
    lakh_files = list(Path("data/lakh_clean").rglob("*.mid"))[:5]
    midicaps_files = list(Path("data/midi_files/lmd_full").rglob("*.mid"))[:5]
    
    test_files = lakh_files + midicaps_files
    
    print(f"\n🎯 Testing {len(test_files)} files:")
    print(f"   - {len(lakh_files)} from Lakh Clean")
    print(f"   - {len(midicaps_files)} from MidiCaps")
    
    # Output directory
    output_dir = Path("data/validation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Validate each
    success_count = 0
    for i, midi_path in enumerate(test_files, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_files)}")
        print(f"{'#'*70}")
        
        if validate_tokenization(midi_path, tokenizer, output_dir):
            success_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    print(f"   Tested:     {len(test_files)} files")
    print(f"   Successful: {success_count}/{len(test_files)}")
    print(f"   Success rate: {100*success_count/len(test_files):.1f}%")
    
    if success_count == len(test_files):
        print(f"\n   ✅ ALL TESTS PASSED!")
        print(f"\n   🎧 NEXT: Listen to files in data/validation_output/")
        print(f"           Compare original vs reconstructed")
    else:
        print(f"\n   ⚠️  Some tests failed, check errors above")
    
    print()

if __name__ == "__main__":
    main()