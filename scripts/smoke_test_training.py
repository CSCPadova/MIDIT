"""Smoke test: Forward + Loss with real dataset and model."""

import torch
from omegaconf import DictConfig
from midit.data.lakh_dataset import LakhDataset, collate_fn
from midit.model.masked_transformer import MaskedTransformerREMI
from torch.utils.data import DataLoader

print("\n" + "="*70)
print("🔥 SMOKE TEST: Forward + Loss")
print("="*70 + "\n")

# Config
chunking_config = DictConfig({
    'strategy': 'bar_based',
    'bars_per_chunk': 8,
    'max_tokens': 8192,
})

# Load dataset
print("📦 Loading dataset...")
train_dataset = LakhDataset(
    data_dir="data/preprocessed_phase1",
    chunking_config=chunking_config,
    split='train',
    apply_masking=True,
)

print(f"✅ Dataset loaded: {len(train_dataset)} files\n")

# Create dataloader
print("🔄 Creating DataLoader...")
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
)
print(f"✅ DataLoader created\n")

# Get vocab sizes from dataset
vocab_sizes = train_dataset.actual_vocab_sizes.copy()

print("📊 Vocab sizes from dataset:")
for token_type, size in vocab_sizes.items():
    print(f"   {token_type:10s}: {size:>4d}")
print()

# Lambda weights
lambda_weights = {
    'pitch': 1.0,
    'velocity': 1.0,
    'duration': 1.0,
    'program': 1.0,
    'position': 4.5,
    'rest': 10.0,
    'bar': 10.0,
    'timesig': 10.0,
    'tempo': 20.0,
}

# Create model
print("🤖 Creating MaskedTransformerREMI...")
model = MaskedTransformerREMI(
    phase=1,
    vocab_sizes=vocab_sizes,  # ← From dataset!
    special_token_ids={'pad': 0, 'bos': 1, 'eos': 2, 'mask': 3},
    d_model=512,
    nhead=8,
    num_encoder_layers=6,  # Smaller for fast test
    dim_feedforward=2048,
    dropout=0.1,
    max_seq_length=8192,
    mask_prob=0.15,
    bar_level_masking=True,
    lambda_weights=lambda_weights,
    lr=3e-4,
)
print(f"✅ Model created\n")

# Get one batch
print("🔍 Loading one batch...")
batch = next(iter(train_loader))

if batch is None:
    print("❌ Batch is None!")
    exit(1)

midi_tokens = batch["input_ids"]
targets = batch["targets"]
midi_attention_mask = batch.get("attention_mask", None)
bar_positions = batch.get("bar_positions", None)

print(f"✅ Batch loaded:")
print(f"   Batch size: {midi_tokens.shape[0]}")
print(f"   Sequence length: {midi_tokens.shape[1]}")
print(f"   Target keys: {list(targets.keys())}")
print()

# Forward pass
print("🚀 Testing forward pass...")
try:
    with torch.no_grad():
        hidden_states = model(
            midi_tokens,
            text_tokens=None,
            text_attention_mask=None,
            midi_attention_mask=midi_attention_mask,
        )
    print(f"✅ Forward pass successful")
    print(f"   Hidden states shape: {hidden_states.shape}\n")
except Exception as e:
    print(f"❌ Forward pass FAILED: {e}\n")
    exit(1)

# Loss computation
print("🎯 Testing loss computation...")
try:
    # Apply masking
    masked_tokens, mask_indices = model.apply_bar_masking(midi_tokens, bar_positions)
    
    # Forward with masked
    hidden_states = model(
        masked_tokens,
        midi_attention_mask=midi_attention_mask,
    )
    
    # Compute loss
    losses = model.multi_head_output.compute_loss(
        hidden_states,
        targets,
        lambda_weights,
        mask=mask_indices,
    )
    
    print(f"✅ Loss computation successful\n")
    print(f"📊 Loss values:")
    for token_type, loss_value in losses.items():
        if token_type != 'total':
            print(f"   {token_type:10s}: {loss_value.item():>8.4f}")
    print(f"   {'total':10s}: {losses['total'].item():>8.4f}")
    print()
    
    # Check for NaN
    if torch.isnan(losses['total']):
        print("❌ LOSS IS NaN!")
        exit(1)
    
    print("✅ No NaN detected")
    
except Exception as e:
    print(f"❌ Loss computation FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ SMOKE TEST PASSED! Safe to launch SLURM.")
print("="*70 + "\n")