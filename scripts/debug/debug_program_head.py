"""Debug: vedi cosa genera il program head."""
import torch
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig

model = MaskedTransformerREMI.load_from_checkpoint(
    "logs/checkpoints_phase1/phase1-epoch=44-val_loss=0.0982.ckpt"
)
model.eval()

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

# Genera
mask_id = 3
tokens = torch.full((1, 128), mask_id, dtype=torch.long)

with torch.no_grad():
    hidden_states = model(tokens)
    logits_dict = model.multi_head_output(hidden_states)
    
    program_logits = logits_dict['program']
    program_probs = torch.softmax(program_logits, dim=-1)
    program_sampled = torch.argmax(program_probs, dim=-1)[0]  # Greedy

print("🔍 PROGRAM HEAD ANALYSIS:")
print("=" * 80)

# Unique programs
unique_programs = torch.unique(program_sampled)
print(f"\nUnique program indices: {unique_programs.tolist()}")
print(f"Total unique: {len(unique_programs)}")

# Distribuzione
from collections import Counter
counter = Counter(program_sampled.tolist())
print(f"\nTop 10 most frequent:")
for prog_idx, count in counter.most_common(10):
    print(f"   Program index {prog_idx}: {count} occurrences ({count/128*100:.1f}%)")

# Get program token names
program_tokens = sorted([t for t in tokenizer.vocab.keys() if t.startswith('Program_')])
print(f"\n📚 Program vocab (first 20):")
for i, tok in enumerate(program_tokens[:20]):
    print(f"   Index {i}: {tok}")

print(f"\n💡 PROBLEMA:")
print(f"   Program head genera indici (0-{len(program_tokens)-1})")
print(f"   Ogni indice corrisponde a un token 'Program_X'")
print(f"   Non puoi filtrare per 'program_id=0' direttamente!")
