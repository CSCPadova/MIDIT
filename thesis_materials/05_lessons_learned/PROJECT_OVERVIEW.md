# MIDIT Project Overview

## Timeline

### November 2024: Initial Exploration
- Started with Diffusion Transformer approach
- 3 iterations: v1, v2, v2_remi
- Focus: Text-to-MIDI generation with MidiCaps

### December 2024: Pivot to MLM
- Switched to Masked Language Model (BERT-style)
- Reason: Better for discrete token sequences
- Phase 1: Pre-training on Lakh MIDI Dataset
- Phase 2: Fine-tuning on MidiCaps (planned)

### December 11, 2024: Critical Discovery ⭐
- **Problem:** Training loss 0.098 (good) but accuracy 22% (terrible)
- **Investigation:** Token frequency analysis on 1000 files
- **Discovery:** Extreme class imbalance (50,011× for pitch!)
- **Root Cause:** Unweighted cross-entropy → mode collapse
- **Solution:** Weighted loss + token-level masking + corrected lambdas

---

## Key Decisions

### Why Diffusion → MLM?
- Diffusion: Continuous latent space, complex training
- MLM: Discrete tokens, simpler, faster, proven for text
- BERT-style masking natural for REMI+ tokenization

### Why Lakh → MidiCaps Two-Phase?
- Phase 1: Learn MIDI structure without text (Lakh)
- Phase 2: Learn text-to-MIDI mapping (MidiCaps)
- Avoids overfitting on small captioned dataset

### Why Bar-Level → Token-Level Masking?
- Bar-level: Removes 15% of bars
- Problem: Tempo/TimeSig rare (0.14-1.7% dataset)
- 15% bars masked → model sees <0.15% tempo → 0% accuracy
- Token-level: Uniform 15% per token → model sees 85% tempo → learns!

### Why Weighted Loss?
- Without weights: Model predicts frequent tokens only
- Result: Low loss (frequent correct) but bad accuracy (rare ignored)
- Solution: Inverse frequency weighting balances learning

---

## Technical Achievements

### Dataset Pipeline
- 4 iterations of dataloader (v1 → current)
- Evolution: Basic MIDI → REMI+ → Quality filtering → MLM chunking
- Final: 12,484 Lakh files, chunk-based loading, masking

### Model Architecture
- MaskedTransformerREMI: Transformer-XL (12 layers, 8 heads)
- 42.5M parameters
- Multi-head output: 9 heads for REMI+ token types
- Segment length 512, recurrence mechanism

### Training Strategy
- Phase 1: 50 epochs, best checkpoint epoch 44
- Batch size 16, segment+recurrence 512
- Validation loss: 0.098
- But accuracy: 22% → discovered class imbalance!

---

## Lessons Learned

### 1. Loss vs Accuracy Mismatch
**Lesson:** Low loss ≠ good model
- Cross-entropy can be low by predicting frequent tokens
- Always check accuracy/reconstruction quality
- Per-class metrics reveal hidden problems

### 2. Class Imbalance in Music
**Lesson:** Music has extreme distributional biases
- 4/4 time signature dominates (94.4%)
- Drums (Program -1) dominate (32.1%)
- Standard ML techniques (weighted loss) apply!

### 3. Masking Strategy Matters
**Lesson:** Different masking for different goals
- Bar-level: Good for musical coherence
- Token-level: Good for rare token learning
- Phase 1 (pre-training): Token-level better
- Phase 2 (fine-tuning): Bar-level might be better

### 4. Systematic Debugging
**Lesson:** When something seems wrong, investigate systematically
- Step 1: Test reconstruction accuracy (not just loss)
- Step 2: Analyze per-head performance
- Step 3: Check data distributions
- Step 4: Identify root cause
- Step 5: Implement targeted solution

---

## Future Work

### Short Term (Phase 1 Optimization)
- Implement weighted loss (STEP 2)
- Re-train 100 epochs
- Expected: 22% → 60-65% accuracy
- Timeline: 1 week

### Medium Term (Phase 2)
- Fine-tune on MidiCaps with text conditioning
- Text-to-MIDI generation
- Evaluation: Caption relevance, musical quality

### Long Term
- Multi-track generation (currently single-track)
- Conditional generation (style, genre, etc.)
- Real-time generation
