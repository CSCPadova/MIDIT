from datasets import load_dataset

ds = load_dataset("amaai-lab/MidiCaps", split="train[:5]")

print("="*60)
print("DEBUG: Primi 5 samples")
print("="*60)

for i, sample in enumerate(ds):
    print(f"\nSample {i}:")
    print(f"  tempo: {sample.get('tempo', 'MISSING')}")
    print(f"  key: {sample.get('key', 'MISSING')}")
    print(f"  time_signature: {sample.get('time_signature', 'MISSING')}")
    print(f"  duration: {sample.get('duration', 'MISSING')}")
    print(f"  chord_summary: {sample.get('chord_summary', 'MISSING')[:3] if sample.get('chord_summary') else 'MISSING'}")
    print(f"  instrument_summary: {sample.get('instrument_summary', 'MISSING')[:3] if sample.get('instrument_summary') else 'MISSING'}")
    print(f"  genre: {sample.get('genre', 'MISSING')[:3] if sample.get('genre') else 'MISSING'}")
    print(f"  mood: {sample.get('mood', 'MISSING')[:3] if sample.get('mood') else 'MISSING'}")
