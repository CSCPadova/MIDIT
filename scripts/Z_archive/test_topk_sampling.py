"""Test Top-K sampling durante diffusion"""
import torch
import torch.nn.functional as F
from midit.model.diffusion_transformer import MaskedDiffusionTransformer
from transformers import T5Tokenizer

def generate_with_topk(
    model,
    text_tokens,
    text_attention_mask=None,
    max_length=100,
    num_inference_steps=50,
    top_k=50,
    temperature=1.0
):
    """Generate con Top-K sampling"""
    batch_size = text_tokens.shape[0]
    device = text_tokens.device
    
    # Start with random noise
    midi_tokens = torch.randint(
        0, model.midi_vocab_size, (batch_size, max_length), device=device
    )
    
    # Denoising loop
    timesteps = torch.linspace(
        model.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
    )
    
    for t in timesteps:
        t_batch = t.repeat(batch_size)
        
        # Predict logits
        predicted_logits = model(
            midi_tokens,
            text_tokens,
            t_batch,
            text_attention_mask,
        )
        
        # Apply temperature
        logits = predicted_logits / temperature
        
        # Top-K filtering
        if top_k > 0:
            # Keep only top k logits
            topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
            # Set others to -inf
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, topk_indices, topk_values)
        else:
            logits_filtered = logits
        
        # Sample
        probs = F.softmax(logits_filtered, dim=-1)
        midi_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)), 
            num_samples=1
        ).view(batch_size, max_length)
    
    return midi_tokens

# Setup
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

# Test different Top-K values
topk_values = [0, 20, 50, 100]  # 0 = no filtering (full softmax)

for topk in topk_values:
    print("="*70)
    print(f"🎲 TOP-K: {topk} {'(argmax-like)' if topk == 0 else ''}")
    print("="*70)
    
    with torch.no_grad():
        tokens = generate_with_topk(
            model,
            text_tokens=text_enc['input_ids'].to(device),
            text_attention_mask=text_enc['attention_mask'].to(device),
            max_length=100,
            num_inference_steps=50,
            top_k=topk,
            temperature=1.0
        )
    
    tokens_cpu = tokens[0].cpu()
    non_zero = tokens_cpu[tokens_cpu != 0]
    
    print(f"📊 Unique tokens: {len(torch.unique(tokens_cpu))}/100")
    print(f"📊 Non-zero: {len(non_zero)}")
    print(f"📊 First 20: {tokens_cpu[:20].tolist()}")
    
    # Tempo analysis
    tempo_tokens = [t.item() for t in non_zero if 100 <= t < 500]
    if tempo_tokens:
        unique_tempos = set(tempo_tokens)
        decoded = [t - 100 for t in unique_tempos]
        print(f"🎵 Tempos generated: {sorted(decoded)[:5]}")
    print()

print("✅ Test completato!")
print("\n💡 Se Top-K aumenta unique tokens → sampling aiuta!")
