"""Generate MIDI from text prompts using REMI+ trained model."""

import torch
from pathlib import Path
from midit.model.diffusion_transformer_v2_remi import MaskedDiffusionTransformerREMI
from midit.midicaps_dataloader_remi import REMI_TOKENIZER, VOCAB_SIZE
from transformers import T5Tokenizer
import pretty_midi

print("🎵 MIDI Generation with REMI+\n")
print("="*80)

# ============================================================================
# 1. SETUP
# ============================================================================

# Checkpoint path
checkpoint_path = "logs/checkpoints_remi/midit-remi-epoch=02-val_loss=0.04.ckpt"
# Output directory
output_dir = Path("generated_remi")
output_dir.mkdir(exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📍 Device: {device}")
print(f"📂 Checkpoint: {checkpoint_path}")
print(f"💾 Output: {output_dir}/")

# ============================================================================
# 2. LOAD MODEL
# ============================================================================

print(f"\n🔄 Loading model...")

model = MaskedDiffusionTransformerREMI.load_from_checkpoint(
    checkpoint_path,
    map_location=device,
)
model.eval()
model.to(device)

print(f"✅ Model loaded!")
print(f"   Vocab size: {model.midi_vocab_size}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 3. TEXT TOKENIZER
# ============================================================================

text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# ============================================================================
# 4. GENERATION FUNCTION
# ============================================================================

def generate_midi(prompt: str, filename: str, max_length: int = 1024, num_steps: int = 50):
    """Generate MIDI from text prompt.
    
    Args:
        prompt: Text description (e.g., "Piano in C major")
        filename: Output filename (without .mid)
        max_length: Max sequence length
        num_steps: Diffusion denoising steps
    """
    print(f"\n{'='*80}")
    print(f"🎹 Generating: {prompt}")
    print(f"{'='*80}")
    
    # Tokenize text
    text_encoding = text_tokenizer(
        prompt,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    text_tokens = text_encoding['input_ids'].to(device)
    text_mask = text_encoding['attention_mask'].float().to(device)
    
    print(f"📝 Prompt tokenized: {text_tokens.shape}")
    
    # Generate
    with torch.no_grad():
        midi_tokens = model.generate(
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            max_length=max_length,
            num_inference_steps=num_steps,
        )
    
    print(f"✅ Generated tokens: {midi_tokens.shape}")
    print(f"   Range: [{midi_tokens.min():.3f}, {midi_tokens.max():.3f}]")
    
    # Denormalize: [0,1] → token IDs [0-388]
    token_ids = (midi_tokens[0].cpu() * VOCAB_SIZE).int().tolist()
    
    # Filter out-of-range tokens
    token_ids = [tid for tid in token_ids if 0 <= tid < VOCAB_SIZE]
    
    print(f"🔢 Token IDs: {len(token_ids)} valid tokens")
    print(f"   First 10: {token_ids[:10]}")
    
    # Decode REMI+ → MIDI
    try:
        # Create TokSequence object (required by miditok)
        from miditok import TokSequence
        
        # REMI tokenizer expects list of lists (one per track)
        tok_sequence = TokSequence(ids=token_ids)
        
        # Decode to MIDI
        midi = REMI_TOKENIZER.decode([tok_sequence])
        
        # Save
        output_path = output_dir / f"{filename}.mid"
        midi.dump_midi(str(output_path))
        
        print(f"✅ MIDI saved: {output_path}")
        
        # Stats
        total_notes = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
        print(f"📊 Statistics:")
        print(f"   Total notes: {total_notes}")
        print(f"   Duration: {midi.get_end_time():.2f}s")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error decoding MIDI: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 5. TEST PROMPTS
# ============================================================================

prompts = [
    # Key tests
    ("Piano in C major", "c_major"),
    ("Piano in G major", "g_major"),
    ("Piano in D minor", "d_minor"),
    
    # Tempo tests
    ("Fast piano music at 140 BPM", "fast_140bpm"),
    ("Slow piano ballad at 60 BPM", "slow_60bpm"),
    
    # Mood tests
    ("Happy piano melody", "happy"),
    ("Melancholic piano piece", "melancholic"),
    
    # Combined
    ("Piano in C major at 120 BPM with happy mood", "c_major_120_happy"),
]

# ============================================================================
# 6. GENERATE ALL
# ============================================================================

print(f"\n{'='*80}")
print(f"🚀 Starting generation of {len(prompts)} prompts...")
print(f"{'='*80}\n")

results = []

for prompt, filename in prompts:
    path = generate_midi(prompt, filename, max_length=1024, num_steps=100)
    results.append((prompt, path))

# ============================================================================
# 7. SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print(f"✅ Generation Complete!")
print(f"{'='*80}\n")

print(f"📊 Summary:")
successful = sum(1 for _, path in results if path is not None)
print(f"   Generated: {successful}/{len(prompts)} MIDI files")
print(f"   Location: {output_dir.absolute()}/")

print(f"\n📂 Files:")
for prompt, path in results:
    status = "✅" if path else "❌"
    print(f"   {status} {prompt}")
    if path:
        print(f"      → {path}")

print(f"\n{'='*80}")
print(f"🎵 Listen to MIDI files in: {output_dir.absolute()}/")
print(f"{'='*80}")
