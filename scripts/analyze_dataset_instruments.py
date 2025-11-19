"""
Analizza dataset MidiCaps per contare file single vs multi-instrument.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def analyze_instruments(json_path):
    """Analizza distribuzione strumenti nel dataset."""
    
    single_instrument = []
    multi_instrument = []
    instrument_counts = Counter()
    instruments_per_file = []
    
    print(f"📖 Analyzing {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            
            # Get instruments (escludi drums)
            instruments = sample.get('instrument_summary', [])
            non_drum_instruments = [inst for inst in instruments if inst.lower() != 'drums']
            
            num_instruments = len(non_drum_instruments)
            instruments_per_file.append(num_instruments)
            
            # Count per instrument
            for inst in non_drum_instruments:
                instrument_counts[inst] += 1
            
            # Classify
            if num_instruments == 1:
                single_instrument.append({
                    'location': sample['location'],
                    'instrument': non_drum_instruments[0],
                    'caption': sample['caption'][:100] + '...',
                    'tempo': sample.get('tempo'),
                    'key': sample.get('key')
                })
            elif num_instruments > 1:
                multi_instrument.append({
                    'location': sample['location'],
                    'instruments': non_drum_instruments,
                    'num': num_instruments
                })
            
            # Progress
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i+1} samples...")
    
    return {
        'single': single_instrument,
        'multi': multi_instrument,
        'instrument_counts': instrument_counts,
        'instruments_per_file': instruments_per_file
    }


def print_statistics(results):
    """Stampa statistiche."""
    
    single = results['single']
    multi = results['multi']
    total = len(single) + len(multi)
    
    print("\n" + "="*80)
    print("📊 DATASET STATISTICS")
    print("="*80)
    
    print(f"\n📁 Total files: {total:,}")
    print(f"   ✅ Single-instrument: {len(single):,} ({len(single)/total*100:.1f}%)")
    print(f"   ⚠️  Multi-instrument:  {len(multi):,} ({len(multi)/total*100:.1f}%)")
    
    # Distribuzione numero strumenti
    from collections import Counter
    inst_dist = Counter(results['instruments_per_file'])
    
    print(f"\n🎵 Instruments per file distribution:")
    for num in sorted(inst_dist.keys())[:10]:  # Top 10
        count = inst_dist[num]
        print(f"   {num} instruments: {count:,} files ({count/total*100:.1f}%)")
    
    # Top instruments in single-instrument files
    print(f"\n🎹 Top 10 instruments (single-instrument files):")
    single_inst_counter = Counter([s['instrument'] for s in single])
    for inst, count in single_inst_counter.most_common(10):
        print(f"   {inst}: {count:,} files")
    
    # Esempi single-instrument
    print(f"\n📝 Example single-instrument files:")
    for i, sample in enumerate(single[:5]):
        print(f"\n   [{i+1}] {sample['instrument']}")
        print(f"       Key: {sample['key']}, Tempo: {sample['tempo']} BPM")
        print(f"       Caption: {sample['caption']}")
        print(f"       File: {sample['location']}")
    
    # Esempi multi-instrument
    print(f"\n📝 Example multi-instrument files:")
    for i, sample in enumerate(multi[:5]):
        print(f"\n   [{i+1}] {sample['num']} instruments: {', '.join(sample['instruments'][:3])}")
        if len(sample['instruments']) > 3:
            print(f"       ... and {len(sample['instruments'])-3} more")
        print(f"       File: {sample['location']}")


def save_single_instrument_list(single_files, output_path):
    """Salva lista file single-instrument per reference."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(single_files, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved single-instrument list to: {output_path}")


if __name__ == "__main__":
    # Path al dataset
    data_dir = Path("data/midi_files")  # ← Modifica se necessario
    json_path = data_dir / "train.json"
    
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print("   Please update data_dir in script")
        exit(1)
    
    # Analyze
    results = analyze_instruments(json_path)
    
    # Print stats
    print_statistics(results)
    
    # Save single-instrument list
    output_path = data_dir / "single_instrument_files.json"
    save_single_instrument_list(results['single'], output_path)
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)