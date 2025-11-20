"""Esplora rappresentazione dati REMI+ nel dataloader."""

from midit.midicaps_dataloader_remi import MidiCapsDataset, REMI_TOKENIZER, VOCAB_SIZE
import torch

print("🔬 Esplorazione Dati REMI+\n")
print("="*80)

# Carica dataset piano-only
dataset = MidiCapsDataset(
    data_dir="data/midi_files",
    split="train",
    max_samples=100,  # Solo 100 per test veloce
    single_instrument_only=True,
    verbose=False,
)

print(f"\n✅ Dataset loaded: {len(dataset)} samples")
print(f"   Vocab size: {VOCAB_SIZE}")

# Inverti vocab per lookup ID → Nome
id_to_token = {v: k for k, v in REMI_TOKENIZER.vocab.items()}

# Carica un sample
print("\n" + "="*80)
print("📊 SAMPLE #0 - Analisi Dettagliata")
print("="*80)

sample = dataset[0]

print(f"\n1️⃣ METADATA:")
print(f"   Key: {sample['key']}")
print(f"   Tempo: {sample['tempo']} BPM")
print(f"   Time Signature: {sample['time_signature']}")
print(f"   Instruments: {sample['instruments']}")

print(f"\n2️⃣ TENSOR INFO:")
print(f"   Shape: {sample['midi_tokens'].shape}")
print(f"   Dtype: {sample['midi_tokens'].dtype}")
print(f"   Range: [{sample['midi_tokens'].min():.3f}, {sample['midi_tokens'].max():.3f}]")
print(f"   Valid tokens: {sample['midi_attention_mask'].sum().int().item()}")

print(f"\n3️⃣ PRIMI 50 TOKEN (Denormalizzati):")
print(f"   Format: [Index] Normalized → Token_ID → Token_Name\n")

valid_length = int(sample['midi_attention_mask'].sum().item())
tokens_normalized = sample['midi_tokens'][:valid_length]

for i in range(min(50, len(tokens_normalized))):
    norm_val = tokens_normalized[i].item()
    token_id = int(round(norm_val * VOCAB_SIZE))
    token_name = id_to_token.get(token_id, f"UNKNOWN_{token_id}")
    
    print(f"   [{i:2d}] {norm_val:.4f} → ID={token_id:3d} → {token_name}")

print(f"\n4️⃣ STATISTICHE TOKEN:")

# Denormalizza tutti token validi
all_token_ids = [int(round(t.item() * VOCAB_SIZE)) for t in tokens_normalized]

from collections import Counter
token_counts = Counter(all_token_ids)

print(f"   Token unici: {len(token_counts)}/{VOCAB_SIZE}")
print(f"   Token totali: {len(all_token_ids)}")

# Top 10 token più frequenti
print(f"\n   Top 10 token più usati:")
for token_id, count in token_counts.most_common(10):
    token_name = id_to_token.get(token_id, f"UNKNOWN_{token_id}")
    pct = count / len(all_token_ids) * 100
    print(f"      {token_name:20s}: {count:3d} ({pct:5.1f}%)")

# Conta per tipo
print(f"\n   Distribuzione per tipo:")
types = {
    'Bar': 0, 'TimeSig': 0, 'Tempo': 0, 'Position': 0,
    'Pitch': 0, 'Velocity': 0, 'Duration': 0, 'Rest': 0
}

for token_id in all_token_ids:
    token_name = id_to_token.get(token_id, "")
    for t in types.keys():
        if token_name.startswith(t):
            types[t] += 1
            break

for t, count in types.items():
    if count > 0:
        pct = count / len(all_token_ids) * 100
        print(f"      {t:12s}: {count:4d} ({pct:5.1f}%)")

print(f"\n5️⃣ CONFRONTO 4-TOKEN:")

# Stima note
num_notes_approx = types['Pitch']
tokens_4token = num_notes_approx * 4

print(f"   Note approssimative: {num_notes_approx}")
print(f"   Token REMI+: {len(all_token_ids)}")
print(f"   Token 4-token: {tokens_4token} (stimato)")
print(f"   Efficienza REMI+: {len(all_token_ids)/tokens_4token*100:.1f}% (meno = meglio)")

print("\n" + "="*80)
print("✅ Esplorazione completata!")
print("="*80)