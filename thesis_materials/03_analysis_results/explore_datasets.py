"""
Musical Data Exploration - Lakh Clean vs MidiCaps
Analizza proprietà musicali (note, duration, tempo, key, instruments, etc.)
"""

from pathlib import Path
from collections import Counter
import miditoolkit
import statistics

def analyze_midi_file(midi_path):
    """Estrae informazioni musicali da un MIDI."""
    try:
        midi = miditoolkit.MidiFile(midi_path)
        
        # Note count
        num_notes = sum(len(inst.notes) for inst in midi.instruments)
        
        # Duration (in beats)
        if num_notes > 0:
            max_tick = max(note.end for inst in midi.instruments for note in inst.notes)
            duration_beats = max_tick / midi.ticks_per_beat
        else:
            duration_beats = 0
        
        # Tracks
        num_tracks = len(midi.instruments)
        
        # Instruments
        instruments = [inst.program for inst in midi.instruments if not inst.is_drum]
        has_drums = any(inst.is_drum for inst in midi.instruments)
        
        # Tempo (BPM) - prende primo valore
        tempo_bpm = midi.tempo_changes[0].tempo if midi.tempo_changes else 120
        
        # Time Signature
        if midi.time_signature_changes:
            ts = midi.time_signature_changes[0]
            time_sig = (ts.numerator, ts.denominator)
        else:
            time_sig = (4, 4)  # default
        
        # Key Signature
        if midi.key_signature_changes:
            key_sig = midi.key_signature_changes[0].key_number
        else:
            key_sig = None
        
        # Ticks per quarter
        tpq = midi.ticks_per_beat
        
        return {
            'success': True,
            'filename': midi_path.name,
            'num_notes': num_notes,
            'duration_beats': duration_beats,
            'num_tracks': num_tracks,
            'instruments': instruments,
            'has_drums': has_drums,
            'tempo': tempo_bpm,
            'time_signature': time_sig,
            'key_signature': key_sig,
            'tpq': tpq
        }
        
    except Exception as e:
        return {
            'success': False,
            'filename': midi_path.name,
            'error': str(e)
        }

def print_statistics(values, label, unit=""):
    """Stampa statistiche descrittive."""
    if not values:
        print(f"   No data for {label}")
        return
    
    print(f"   Mean:   {statistics.mean(values):.1f}{unit}")
    print(f"   Median: {statistics.median(values):.1f}{unit}")
    print(f"   Min:    {min(values):.1f}{unit}")
    print(f"   Max:    {max(values):.1f}{unit}")
    print(f"   StdDev: {statistics.stdev(values):.1f}{unit}" if len(values) > 1 else "")

def explore_dataset(dataset_path, dataset_name, max_files=1000):
    """Esplora dataset MIDI - analisi musicale."""
    
    print(f"\n{'='*70}")
    print(f"🎵 {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    # Trova MIDI files
    midi_files = list(Path(dataset_path).rglob("*.mid"))
    total_files = len(midi_files)
    
    print(f"📁 Total files found: {total_files}")
    print(f"📊 Analyzing first {min(max_files, total_files)} files...\n")
    
    # Collezioni dati
    all_data = []
    note_counts = []
    durations = []
    track_counts = []
    all_instruments = []
    tempos = []
    time_sigs = []
    key_sigs = []
    tpq_values = []
    corrupted = []
    
    # Analizza files
    for i, midi_path in enumerate(midi_files[:max_files]):
        if (i + 1) % 200 == 0:
            print(f"   Progress: {i+1}/{min(max_files, total_files)}")
        
        result = analyze_midi_file(midi_path)
        
        if result['success']:
            all_data.append(result)
            note_counts.append(result['num_notes'])
            durations.append(result['duration_beats'])
            track_counts.append(result['num_tracks'])
            all_instruments.extend(result['instruments'])
            tempos.append(result['tempo'])
            time_sigs.append(result['time_signature'])
            if result['key_signature'] is not None:
                key_sigs.append(result['key_signature'])
            tpq_values.append(result['tpq'])
        else:
            corrupted.append(result)
    
    print(f"\n{'='*70}")
    print("📈 RESULTS")
    print(f"{'='*70}\n")
    
    # Success rate
    print(f"✅ Successfully analyzed: {len(all_data)}")
    print(f"❌ Corrupted/Error: {len(corrupted)}")
    
    if corrupted:
        print(f"\n⚠️  First 5 corrupted files:")
        for item in corrupted[:5]:
            print(f"   - {item['filename']}: {item['error'][:50]}")
    
    print(f"\n{'-'*70}\n")
    
    # === NOTE COUNTS ===
    print("🎼 NOTE COUNTS:")
    print_statistics(note_counts, "Notes", " notes")
    
    # Top 5 largest
    top_notes = sorted(all_data, key=lambda x: x['num_notes'], reverse=True)[:5]
    print(f"\n   📊 Top 5 files by note count:")
    for i, item in enumerate(top_notes, 1):
        print(f"   {i}. {item['filename'][:50]}: {item['num_notes']} notes")
    
    # Bottom 5 smallest
    bottom_notes = sorted(all_data, key=lambda x: x['num_notes'])[:5]
    print(f"\n   📊 Bottom 5 files by note count:")
    for i, item in enumerate(bottom_notes, 1):
        print(f"   {i}. {item['filename'][:50]}: {item['num_notes']} notes")
    
    print(f"\n{'-'*70}\n")
    
    # === DURATION ===
    print("⏱️  DURATION (in beats):")
    print_statistics(durations, "Duration", " beats")
    
    # Top 5 longest
    top_duration = sorted(all_data, key=lambda x: x['duration_beats'], reverse=True)[:5]
    print(f"\n   📊 Top 5 longest files:")
    for i, item in enumerate(top_duration, 1):
        print(f"   {i}. {item['filename'][:50]}: {item['duration_beats']:.0f} beats")
    
    print(f"\n{'-'*70}\n")
    
    # === TRACKS ===
    print("🎹 NUMBER OF TRACKS:")
    print_statistics(track_counts, "Tracks", " tracks")
    
    track_dist = Counter(track_counts)
    print(f"\n   📊 Track distribution:")
    for tracks, count in sorted(track_dist.items())[:10]:
        print(f"   {tracks} track(s): {count} files")
    
    print(f"\n{'-'*70}\n")
    
    # === INSTRUMENTS ===
    print("🎺 INSTRUMENTS (GM Programs):")
    instrument_counts = Counter(all_instruments)
    print(f"\n   📊 Top 15 most common instruments:")
    for program, count in instrument_counts.most_common(15):
        print(f"   Program {program:3d}: {count:4d} occurrences")
    
    print(f"\n{'-'*70}\n")
    
    # === TEMPO ===
    print("🎵 TEMPO (BPM):")
    print_statistics(tempos, "Tempo", " BPM")
    
    tempo_ranges = {
        "Very Slow (<60)": sum(1 for t in tempos if t < 60),
        "Slow (60-80)": sum(1 for t in tempos if 60 <= t < 80),
        "Moderate (80-120)": sum(1 for t in tempos if 80 <= t < 120),
        "Fast (120-160)": sum(1 for t in tempos if 120 <= t < 160),
        "Very Fast (>160)": sum(1 for t in tempos if t >= 160)
    }
    
    print(f"\n   📊 Tempo distribution:")
    for range_name, count in tempo_ranges.items():
        pct = 100 * count / len(tempos) if tempos else 0
        print(f"   {range_name:20s}: {count:4d} ({pct:.1f}%)")
    
    print(f"\n{'-'*70}\n")
    
    # === TIME SIGNATURE ===
    print("⏲️  TIME SIGNATURES:")
    ts_counts = Counter(time_sigs)
    print(f"\n   📊 Top 10 time signatures:")
    for ts, count in ts_counts.most_common(10):
        pct = 100 * count / len(time_sigs) if time_sigs else 0
        print(f"   {ts[0]}/{ts[1]:2d}: {count:4d} ({pct:.1f}%)")
    
    print(f"\n{'-'*70}\n")
    
    # === KEY SIGNATURE ===
    if key_sigs:
        print("🎼 KEY SIGNATURES:")
        key_counts = Counter(key_sigs)
        
        # Map key numbers to names
        key_names = {
            0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#",
            -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb"
        }
        
        print(f"\n   📊 Top 10 keys:")
        for key_num, count in key_counts.most_common(10):
            key_name = key_names.get(key_num, f"Key#{key_num}")
            pct = 100 * count / len(key_sigs) if key_sigs else 0
            print(f"   {key_name:5s}: {count:4d} ({pct:.1f}%)")
    
    print(f"\n{'-'*70}\n")
    
    # === TPQ ===
    print("🎚️  TICKS PER QUARTER (TPQ):")
    tpq_counts = Counter(tpq_values)
    print(f"\n   📊 TPQ distribution:")
    for tpq, count in sorted(tpq_counts.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / len(tpq_values) if tpq_values else 0
        print(f"   {tpq:4d} tpq: {count:4d} ({pct:.1f}%)")
    
    print()

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🔍 MIDI DATASET MUSICAL EXPLORATION")
    print("="*70)
    
    # Lakh Clean
    lakh_path = Path("data/lakh_clean")
    if lakh_path.exists():
        explore_dataset(lakh_path, "Lakh Clean Dataset", max_files=1000)
    else:
        print(f"\n⚠️  Lakh Clean not found at {lakh_path}")
    
    # MidiCaps
    midicaps_path = Path("data/midi_files/lmd_full")
    if midicaps_path.exists():
        explore_dataset(midicaps_path, "MidiCaps Dataset", max_files=1000)
    else:
        print(f"\n⚠️  MidiCaps not found at {midicaps_path}")
    
    print("\n" + "="*70)
    print("✅ EXPLORATION COMPLETE!")
    print("="*70 + "\n")