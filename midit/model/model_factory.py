"""
Factory for constructing MaskedTransformerREMI with a verified vocabulary.

Motivation
----------
MaskedTransformerREMI.__init__ computes the token embedding size as:

    sum(vocab_sizes.values()) + 4   # per-head sizes + 4 special tokens

This equals len(tokenizer.vocab) only when:
  1. Token IDs are contiguous (no gaps), so len(vocab) is the correct upper bound.
  2. Every token class is counted in vocab_sizes, including PitchDrum tokens
     (which are merged into the 'pitch' head).

Historically, a post-init override in train_phase1.py patched a mismatch
silently, but that bypasses save_hyperparameters(), breaking load_from_checkpoint().

This factory runs two invariant checks before construction — failing loudly
rather than patching silently — and passes correct params to the constructor
so no post-init changes are needed.
"""

from typing import Dict

from omegaconf import DictConfig, OmegaConf
from miditok import REMI

from midit.model.masked_transformer import MaskedTransformerREMI


def build_model(
    cfg: DictConfig,
    tokenizer: REMI,
    vocab_sizes: Dict[str, int],
    total_training_steps: int,
    lr: float,
) -> MaskedTransformerREMI:
    """Construct MaskedTransformerREMI with verified vocabulary invariants.

    Args:
        cfg: Full Hydra config.
        tokenizer: Initialized MidiTok REMI tokenizer (from LakhDataset).
                   Used only for vocab validation — not stored in the model.
        vocab_sizes: Per-head vocabulary sizes (from LakhDataset.actual_vocab_sizes).
                     Must include 'pitch' as Pitch + PitchDrum combined.
        total_training_steps: Total optimizer steps for cosine LR scheduler.
                              Passed explicitly because DataLoaderWrapper causes
                              trainer.estimated_stepping_batches to return inf.
        lr: Peak learning rate.

    Returns:
        MaskedTransformerREMI constructed with correct parameters.
        save_hyperparameters() will capture the verified vocab_sizes, so
        load_from_checkpoint() reconstructs the model correctly.

    Raises:
        ValueError: If invariant (a) or (b) fails — construction is aborted
                    before any weights are allocated.
    """
    n_vocab = len(tokenizer.vocab)

    # ── Invariant (a): IDs are contiguous ────────────────────────────────────
    # If IDs were non-contiguous (gaps), len(vocab) would undercount the range
    # and nn.Embedding(len(vocab), d_model) would produce index-out-of-bounds
    # at forward time for tokens beyond the gap.
    max_id = max(tokenizer.vocab.values())
    if max_id + 1 != n_vocab:
        raise ValueError(
            f"Tokenizer vocab IDs are not contiguous.\n"
            f"  max(token_id) + 1 = {max_id + 1}\n"
            f"  len(tokenizer.vocab) = {n_vocab}\n"
            f"  Difference: {(max_id + 1) - n_vocab} gap(s) in ID space.\n"
            f"Embedding size would be too small — check the TokenizerConfig "
            f"for removed or non-consecutive token IDs."
        )

    # ── Invariant (b): Per-head sizes cover full vocab ────────────────────────
    # MaskedTransformerREMI computes embedding size as sum(vocab_sizes.values()) + 4.
    # This must equal len(tokenizer.vocab) or the embedding is wrong at construction.
    # The +4 accounts for PAD/BOS/EOS/MASK special tokens (IDs 0-3).
    computed = sum(vocab_sizes.values()) + 4
    if computed != n_vocab:
        raise ValueError(
            f"Per-head vocab sizes do not cover the full tokenizer vocabulary.\n"
            f"  sum(vocab_sizes.values()) + 4 = {computed}\n"
            f"  len(tokenizer.vocab)           = {n_vocab}\n"
            f"  Difference: {computed - n_vocab}\n"
            f"  Per-head sizes: {vocab_sizes}\n"
            f"Most likely cause: 'pitch' head size ({vocab_sizes.get('pitch')}) "
            f"does not include PitchDrum tokens. Expected pitch = Pitch + PitchDrum "
            f"(~89 + 62 = 151). Check that use_pitchdrum_tokens=True is set in "
            f"the runtime TokenizerConfig in lakh_dataset.py."
        )

    print(f"[OK] Vocab invariant (a): IDs contiguous — max_id+1 == len(vocab) == {n_vocab}")
    print(f"[OK] Vocab invariant (b): sum(vocab_sizes)+4 == len(vocab) == {n_vocab}")

    # ── Build model ───────────────────────────────────────────────────────────
    lambda_weights = OmegaConf.to_container(cfg.lambda_weights, resolve=True)
    class_weights = (
        OmegaConf.to_container(cfg.class_weights, resolve=True)
        if 'class_weights' in cfg else None
    )

    model = MaskedTransformerREMI(
        phase=1,
        vocab_sizes=vocab_sizes,
        special_token_ids={'pad': 0, 'bos': 1, 'eos': 2, 'mask': 3},
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_encoder_layers=cfg.model.num_encoder_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        max_seq_length=cfg.chunking.max_tokens,
        mask_prob=cfg.masking.mask_prob,
        bar_level_masking=cfg.masking.bar_level,
        lambda_weights=lambda_weights,
        class_weights=class_weights,
        lr=lr,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # No post-init override: invariant (b) guarantees that __init__'s own
    #   sum(vocab_sizes.values()) + 4
    # already equals len(tokenizer.vocab), so token_embedding is correctly sized.

    return model
