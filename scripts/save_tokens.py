import torch
import numpy as np
from midit.model.diffusion_transformer_v2 import MaskedDiffusionTransformer
from transformers import T5Tokenizer

checkpoint_path = "logs/checkpoints/midit-epoch=09-val_loss=0.02.ckpt"
print(f"📂 Loading checkpoint...")

model = MaskedDiffusionTransformer.load_from_checkpoint(checkpoint_path)
model.eval()
model.to('cpu')

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
prompts = [
    "A very slow and simple piano melody using only the white keys C D E F G A B from the C major scale, with gentle rhythm, no sharps or flats, peaceful and melodic, staying in middle register around C4 to C5",
    
    "Extremely relaxing slow piano music with long sustained notes, minimal movement, using only natural notes without any sharps or flats, very gentle and calm, soft dynamics, in the key of C major",
    
    "Fast repetitive drum pattern with short sharp notes, using low pitch percussion sounds below C4, rapid succession of hits, energetic and rhythmic, no melody just rhythm"
]

print("🎵 Generating tokens...\n")

for i, prompt in enumerate(prompts):
    text_encoding = tokenizer(prompt, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        generated = model.generate(
            text_tokens=text_encoding['input_ids'],
            text_attention_mask=text_encoding['attention_mask'],
            max_length=512,
            num_inference_steps=50
        )
    
    tokens = generated[0].cpu().numpy()
    filename = f"/tmp/tokens_{i}.npy"
    np.save(filename, tokens)
    print(f"✅ Saved: {filename} - {prompt}")

print("\n📦 Compress:")
print("tar -czf /tmp/tokens.tar.gz /tmp/tokens_*.npy")
