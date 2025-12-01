"""Test script to validate configuration loading."""

from omegaconf import OmegaConf
from pathlib import Path

def test_config():
    """Test loading and structure of config.yaml"""
    
    print("\n" + "="*70)
    print("🧪 TESTING CONFIG.YAML")
    print("="*70 + "\n")
    
    # Load config
    config_path = Path("conf/config.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        cfg = OmegaConf.load(config_path)
        print("✅ Config loaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Print main sections
    print("📊 Config structure:")
    print(f"   Model type: {cfg.model.type}")
    print(f"   Tokenizer: {cfg.tokenizer.type}")
    print(f"   Chunking strategy: {cfg.chunking.strategy}")
    print(f"   Bars per chunk: {cfg.chunking.bars_per_chunk}")
    print(f"   Max seq length: {cfg.model.max_seq_length}")
    print(f"   Encoder layers: {cfg.model.num_encoder_layers}")
    print(f"   Training phase: {cfg.train.phase}")
    print()
    
    # Validate chunking config
    print("🔧 Chunking configuration:")
    print(f"   Strategy: {cfg.chunking.strategy}")
    print(f"   Bars per chunk: {cfg.chunking.bars_per_chunk}")
    print(f"   Overlap bars: {cfg.chunking.overlap_bars}")
    print(f"   Tokens per chunk: {cfg.chunking.tokens_per_chunk}")
    print(f"   Max tokens: {cfg.chunking.max_tokens}")
    print(f"   Filter empty bars: {cfg.chunking.filter_empty_bars}")
    print()
    
    # Validate paths
    print("📁 Data paths:")
    print(f"   Raw data: {cfg.chunking.raw_data_dir}")
    print(f"   Preprocessed: {cfg.chunking.preprocessed_dir}")
    print(f"   Checkpoints: {cfg.paths.checkpoints_dir}")
    print()
    
    # Validate lambda weights
    print("💡 Lambda weights:")
    for token_type, weight in cfg.lambda_weights.items():
        print(f"   {token_type}: {weight}")
    print()
    
    # Validate vocab sizes
    print("📖 Vocab sizes:")
    total_vocab = 0
    for token_type, size in cfg.model.vocab_sizes.items():
        print(f"   {token_type}: {size}")
        total_vocab += size
    print(f"   Total: {total_vocab} (+ 4 special tokens)")
    print()
    
    print("="*70)
    print("✅ ALL CONFIG TESTS PASSED!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_config()
    exit(0 if success else 1)