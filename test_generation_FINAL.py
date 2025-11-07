"""TEST FINALE con checkpoint CORRETTO"""
import torch
from midit.model.diffusion_transformer import MaskedDiffusionTransformer
from transformers import T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Device: {device}\n")

print("🔄 Loading CORRECT checkpoint (epoch=4, loss=0.24)...")
model = MaskedDiffusionTransformer.load_from_checkpoint(
    "logs/checkpoints/midit-epoch=04-val_loss=0.24.ckpt"
)
model.to(device).eval()

print(f"✅ Model loaded: d_model={model.d_model}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

test_prompts = [
    "A fast jazz song with piano at 140 BPM",
    "A slow classical piece with strings at 60 BPM",
    "An upbeat rock song with guitar at 180 BPM",
]

for prompt in test_prompts:
    print("="*70)
    print(f"📝 Prompt: {prompt}")
    
    text_enc = tokenizer(prompt, max_length=256, padding='max_length',
                         truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        tokens = model.generate(
            text_tokens=text_enc['input_ids'].to(device),
            text_attention_mask=text_enc['attention_mask'].to(device),
            max_length=100,
            num_inference_steps=50
        )
    
    tokens_cpu = tokens[0].cpu()
    non_zero = tokens_cpu[tokens_cpu != 0]
    
    print(f"\n📊 STATISTICS:")
    print(f"   First 30 tokens: {tokens_cpu[:30].tolist()}")
    print(f"   Unique tokens: {len(torch.unique(tokens_cpu))}/100")
    print(f"   Non-zero tokens: {len(non_zero)}")
    
    # Analisi per range
    tempo_tokens = [t.item() for t in non_zero if 100 <= t < 500]
    key_tokens = [t.item() for t in non_zero if 200 <= t < 328]
    chord_tokens = [t.item() for t in non_zero if 300 <= t < 428]
    ts_tokens = [t.item() for t in non_zero if 500 <= t < 564]
    dur_tokens = [t.item() for t in non_zero if 600 <= t < 610]
    inst_tokens = [t.item() for t in non_zero if 700 <= t < 828]
    genre_tokens = [t.item() for t in non_zero if 800 <= t < 864]
    mood_tokens = [t.item() for t in non_zero if 900 <= t < 964]
    
    print(f"\n📈 TOKEN DISTRIBUTION:")
    print(f"   Tempo (100-499):      {len(tempo_tokens)} tokens")
    print(f"   Key (200-327):        {len(key_tokens)} tokens")
    print(f"   Chords (300-427):     {len(chord_tokens)} tokens")
    print(f"   Time Sig (500-563):   {len(ts_tokens)} tokens")
    print(f"   Duration (600-609):   {len(dur_tokens)} tokens")
    print(f"   Instruments (700-827): {len(inst_tokens)} tokens")
    print(f"   Genre (800-863):      {len(genre_tokens)} tokens")
    print(f"   Mood (900-963):       {len(mood_tokens)} tokens")
    
    # Decode some values
    if tempo_tokens:
        decoded_tempo = tempo_tokens[0] - 100
        print(f"\n🎵 DECODED:")
        print(f"   Tempo: {decoded_tempo} BPM")
    
    print()

print("="*70)
print("✅ Test completato!")
