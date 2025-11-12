"""Verifica che generate() ora usi sampling probabilistico"""
import torch
from midit.model.diffusion_transformer import MaskedDiffusionTransformer
from transformers import T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MaskedDiffusionTransformer.load_from_checkpoint(
    "logs/checkpoints/midit-epoch=04-val_loss=0.24.ckpt"
)
model.to(device).eval()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
prompt = "A fast jazz song with piano at 140 BPM"

text_enc = tokenizer(prompt, max_length=256, padding='max_length',
                     truncation=True, return_tensors='pt')

print(f"📝 Prompt: {prompt}\n")
print("="*70)
print("TEST: 3 Generazioni con generate() PATCHATO")
print("="*70)

for i in range(3):
    with torch.no_grad():
        tokens = model.generate(
            text_tokens=text_enc['input_ids'].to(device),
            text_attention_mask=text_enc['attention_mask'].to(device),
            max_length=100,
            num_inference_steps=50
        )
    
    tokens_cpu = tokens[0].cpu()
    non_zero = tokens_cpu[tokens_cpu != 0]
    
    # Analisi
    unique = len(torch.unique(tokens_cpu))
    tempo_tokens = [t.item() for t in non_zero if 100 <= t < 500]
    
    tempos = sorted(set([t - 100 for t in tempo_tokens]))[:5]
    
    print(f"\nRun {i+1}:")
    print(f"  Unique: {unique}/100")
    print(f"  Tempos: {tempos}")
    print(f"  First 10: {tokens_cpu[:10].tolist()}")

print("\n" + "="*70)
print("✅ SUCCESSO se:")
print("   - Unique tokens: 40-60 (vs 11 prima)")
print("   - Tempos variano tra run")
print("   - First 10 diversi tra run")
print("="*70)
