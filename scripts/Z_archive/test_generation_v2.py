"""Test con più step"""
import torch
from midit.model.diffusion_transformer import MaskedDiffusionTransformer
from transformers import T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {device}")

model = MaskedDiffusionTransformer.load_from_checkpoint(
    "logs/checkpoints/midit-epoch=02-val_loss=0.42.ckpt"
)
model.to(device).eval()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Test prompts diversi
prompts = [
    "A fast jazz song with piano at 140 BPM",
    "A slow classical piece with strings",
    "An upbeat rock song with guitar at 180 BPM"
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"📝 Prompt: {prompt}")
    
    text_enc = tokenizer(prompt, max_length=256, padding='max_length', 
                         truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        tokens = model.generate(
            text_tokens=text_enc['input_ids'].to(device),
            text_attention_mask=text_enc['attention_mask'].to(device),
            max_length=100,
            num_inference_steps=50  # ← PIÙ STEP!
        )
    
    print(f"📊 First 30 tokens: {tokens[0, :30].cpu().tolist()}")
    print(f"📊 Unique tokens: {len(torch.unique(tokens))}/100")
