"""Test sampling probabilistico - versione semplice"""
import torch
import torch.nn.functional as F
from midit.model.diffusion_transformer import MaskedDiffusionTransformer
from transformers import T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = MaskedDiffusionTransformer.load_from_checkpoint(
    "logs/checkpoints/midit-epoch=04-val_loss=0.24.ckpt"
)
model.to(device).eval()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
prompt = "A fast jazz song with piano at 140 BPM"

text_enc = tokenizer(prompt, max_length=256, padding='max_length',
                     truncation=True, return_tensors='pt')

print(f"📝 Testing: {prompt}\n")

# Test 1: Deterministico (attuale)
print("="*60)
print("TEST 1: DETERMINISTICO (argmax)")
print("="*60)

with torch.no_grad():
    tokens_det = model.generate(
        text_tokens=text_enc['input_ids'].to(device),
        text_attention_mask=text_enc['attention_mask'].to(device),
        max_length=100,
        num_inference_steps=50
    )

print(f"Unique tokens: {len(torch.unique(tokens_det[0]))}/100")
print(f"First 20: {tokens_det[0, :20].cpu().tolist()}\n")

# Test 2: Generazione multipla (stesso metodo)
print("="*60)
print("TEST 2: 3 GENERAZIONI DETERMINISTICHE")
print("="*60)

for i in range(3):
    with torch.no_grad():
        tokens = model.generate(
            text_tokens=text_enc['input_ids'].to(device),
            text_attention_mask=text_enc['attention_mask'].to(device),
            max_length=100,
            num_inference_steps=50
        )
    print(f"Run {i+1} - Unique: {len(torch.unique(tokens[0]))}/100")
    print(f"         First 10: {tokens[0, :10].cpu().tolist()}")

print("\n💡 Se i 3 run sono identici → sampling deterministico")
print("💡 Se diversi → c'è già randomness nel processo")
