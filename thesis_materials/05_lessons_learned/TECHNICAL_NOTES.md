# Technical Notes & Implementation Details

## REMI+ Tokenization

### Token Types (9 heads)
1. **Pitch** (151 classes): Pitch_0 to Pitch_127 + PitchDrum_0 to PitchDrum_127
2. **Velocity** (32 classes): Velocity_0 to Velocity_127 (quantized)
3. **Duration** (32 classes): Duration_0.0 to Duration_8.0 (beats)
4. **Program** (129 classes): Program_0 to Program_127 + Program_-1 (drums)
5. **Position** (96 classes): Position_0 to Position_63 (within bar)
6. **Bar** (1 class): Bar_None (bar boundary marker)
7. **Tempo** (32 classes): Tempo_30 to Tempo_250 (BPM)
8. **TimeSig** (9 classes): TimeSig_4/4, 3/4, 2/4, etc.
9. **Rest** (32 classes): Rest_0.0 to Rest_8.0 (beats)

### Sequence Structure
```
Linear sequence: [Bar, Position_0, Program_24, Pitch_60, Velocity_80, Duration_0.2, Position_8, ...]
NOT grouped by note: Each position has ONE token type
Multi-head predicts PER POSITION, not per note
```

### Why REMI+ > REMI?
- Position events > Time-Shift (easier to learn, no error accumulation)
- Explicit Duration > Note-Off (no dangling notes)
- Bar markers > Absolute time (metric structure)
- Tempo/Chord events (high-level control)

---

## Class Imbalance Analysis

### Frequency Statistics (1000 files sample)

**Pitch:**
- Total tokens: 5,556,438
- Most common: PitchDrum_42 (450,099 = 8.1%)
- Least common: Pitch_109 (9 = 0.00016%)
- Imbalance ratio: 50,011×
- 80% mass in top 42/150 tokens

**Program:**
- Total tokens: 5,579,802
- Most common: Program_-1 drums (1,791,945 = 32.1%)
- Least common: Program_97 (41)
- Imbalance ratio: 43,706×

**TimeSig:**
- Total tokens: 95,826
- Most common: TimeSig_4/4 (90,441 = 94.4%)
- Least common: TimeSig_3/8 (1)
- Imbalance ratio: 90,441×

**Tempo:**
- Total tokens: 7,668 (0.14% of dataset!)
- Most common: Tempo_107.74 (1,396)
- Bar-level masking removes 15% → model sees <0.12%
- Result: 0% accuracy (never learned)

### Weighted Loss Formula
```python
# Inverse square root frequency (better than 1/freq for stability)
weight_i = 1 / sqrt(freq_i / total)

# Normalize to mean=1.0 (preserves gradient scale)
weight_i = weight_i / mean(weights)
```

**Why sqrt?**
- 1/freq too aggressive (rare tokens dominate loss)
- sqrt balances frequent vs rare
- Standard in NLP (class imbalance)

---

## Lambda Weights (Loss Balancing)

### Original (Incorrect)
```yaml
position: 4.5  # Too high (structural, easy)
velocity: 1.0  # OK
bar: 10.0      # WAY too high (trivial, only 1 value)
timesig: 10.0  # Too high but rare so maybe ok
```

### Optimized (Frequency-Based)
```yaml
pitch: 1.0      # Important + difficult
velocity: 0.5   # Easy + repetitive
duration: 0.8   # Moderate
program: 1.0    # Important
position: 0.2   # Structural, easy
bar: 0.1        # Trivial (almost always correct)
tempo: 2.0      # Rare + critical
timesig: 2.0    # Rare + critical
rest: 0.5       # Moderate
```

**Rationale:**
- Higher for rare + important (tempo, timesig)
- Lower for easy + frequent (position, bar)
- Medium for core musical content (pitch, duration)

---

## Training Hyperparameters

### Model Architecture
```yaml
d_model: 768
n_layers: 12
n_heads: 8
d_ff: 3072 (4 × d_model)
dropout: 0.1
segment_len: 512
mem_len: 512 (recurrence)
```

### Training Config
```yaml
batch_size: 16
learning_rate: 3e-4
warmup_steps: 4000
max_epochs: 100 (increased from 50)
gradient_clip_val: 1.0
optimizer: AdamW
weight_decay: 0.01
```

### Masking
```yaml
mask_ratio: 0.15 (BERT standard)
strategy: token-level (optimized from bar-level)
distribution: 80% [MASK], 10% random, 10% unchanged
```

---

## Expected Results (After Optimization)

### Before (Unweighted Loss, Bar-Level Masking)
```
Pitch: 7.61% (mode collapse to frequent)
Velocity: 2.26%
Duration: 37.62%
Program: 2.73%
Position: 4.22%
Bar: 100% (mode collapse - always same)
Tempo: 0.00% (never learned)
TimeSig: 0.00% (never learned)
Rest: 45.69% (mode collapse)
Average: 22.24%
```

### After (Weighted Loss, Token-Level Masking)
```
Pitch: 60-70% (7-9× improvement)
Velocity: 50-60% (22× improvement)
Duration: 60-70% (1.6× improvement)
Program: 50-60% (18× improvement)
Position: 60-70% (14× improvement)
Bar: 95-100% (already good)
Tempo: 40-50% (from 0%!)
TimeSig: 70-80% (from 0%!)
Rest: 60-70% (1.3× improvement)
Average: 60-65% (2.7× improvement)
```

**Key improvements:**
- Rare tokens (tempo, timesig): 0% → 40-80%
- Mode collapse fixed: All heads learn
- Overall: 3× better reconstruction
