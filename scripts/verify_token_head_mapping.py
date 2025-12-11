"""
Verifica mapping token-to-head.
"""
import torch
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
from collections import Counter

# Load
print("🔧 Loading...")
model = MaskedTransformerREMI.load_from_checkpoint(
    "logs/checkpoints_phase1/phase1-epoch=44-val_loss=0.0982.ckpt"
)

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

print("\n" + "=" * 80)
print("🔬 TOKEN-HEAD MAPPING VERIFICATION")
print("=" * 80)

# Check embedding
print("\n📊 Model Embedding:")
print(f"   Vocab size: {model.token_embedding.num_embeddings}")
print(f"   Embed dim:  {model.token_embedding.embedding_dim}")

# Check multi-head output structure
print("\n🎯 Multi-Head Output:")
print(f"   Type: {type(model.multi_head_output)}")
print(f"   Attributes: {dir(model.multi_head_output)}")

# Get head dimensions
print("\n📐 Head Output Dimensions:")

# Access heads via vocab_sizes attribute
if hasattr(model.multi_head_output, 'vocab_sizes'):
    vocab_sizes = model.multi_head_output.vocab_sizes
    print(f"   Vocab sizes dict: {vocab_sizes}")
    
    for head_name, size in vocab_sizes.items():
        print(f"   {head_name:12} output_dim: {size}")
else:
    # Try to get from individual heads
    head_names = ['pitch', 'velocity', 'duration', 'program', 'position', 
                  'bar', 'tempo', 'timesig', 'rest']
    
    for head_name in head_names:
        head_attr = f"{head_name}_head"
        if hasattr(model.multi_head_output, head_attr):
            head = getattr(model.multi_head_output, head_attr)
            if hasattr(head, 'out_features'):
                print(f"   {head_name:12} output_dim: {head.out_features}")

# Count tokens by type in tokenizer
print("\n📚 Tokenizer Token Counts:")
token_counts = Counter()

for token_name in tokenizer.vocab.keys():
    if token_name.startswith('Pitch_') or token_name.startswith('PitchDrum_'):
        token_counts['pitch'] += 1
    elif token_name.startswith('Velocity_'):
        token_counts['velocity'] += 1
    elif token_name.startswith('Duration_'):
        token_counts['duration'] += 1
    elif token_name.startswith('Program_'):
        token_counts['program'] += 1
    elif token_name.startswith('Position_'):
        token_counts['position'] += 1
    elif token_name.startswith('Bar_'):
        token_counts['bar'] += 1
    elif token_name.startswith('Tempo_'):
        token_counts['tempo'] += 1
    elif token_name.startswith('TimeSig_'):
        token_counts['timesig'] += 1
    elif token_name.startswith('Rest_'):
        token_counts['rest'] += 1
    elif token_name in ['PAD', 'BOS', 'EOS', 'MASK']:
        token_counts['special'] += 1

for token_type, count in sorted(token_counts.items()):
    print(f"   {token_type:12} {count:>4} tokens")

# Compare
print("\n" + "=" * 80)
print("⚙️ COMPARISON:")
print("=" * 80)

if hasattr(model.multi_head_output, 'vocab_sizes'):
    vocab_sizes = model.multi_head_output.vocab_sizes
    
    print("\nHead vs Tokenizer:")
    mismatches = []
    
    for head_name in vocab_sizes.keys():
        model_size = vocab_sizes[head_name]
        tokenizer_size = token_counts.get(head_name, 0)
        
        match = "✅" if model_size == tokenizer_size else "❌"
        
        print(f"   {head_name:12} Model: {model_size:>4}  Tokenizer: {tokenizer_size:>4}  {match}")
        
        if model_size != tokenizer_size:
            mismatches.append((head_name, model_size, tokenizer_size))
    
    if mismatches:
        print("\n❌ CRITICAL MISMATCHES FOUND:")
        for head_name, model_sz, tok_sz in mismatches:
            print(f"   {head_name}: model expects {model_sz}, tokenizer has {tok_sz}")
            print(f"   → This will cause incorrect predictions!")
    else:
        print("\n✅ All head sizes match tokenizer!")
else:
    print("\n⚠️ Cannot access vocab_sizes directly")
    print("   Checking via model source code...")

# Additional check: look at actual Linear layers
print("\n🔍 Inspecting Linear Layers:")
for name, module in model.multi_head_output.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"   {name:30} in: {module.in_features:>4}  out: {module.out_features:>4}")

print("\n" + "=" * 80)
