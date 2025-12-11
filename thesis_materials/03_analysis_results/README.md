# Analysis Results - MIDIT Project

## Critical Discovery: Class Imbalance (Dec 11, 2024) ⭐⭐⭐⭐⭐

### Token Frequency Analysis
**Script:** `analyze_token_frequencies.py`

**Key Findings:**
- **Pitch:** 50,011× imbalance (PitchDrum_42: 450,099 vs Pitch_109: 9)
- **TimeSig:** 90,441× imbalance (4/4 = 94.4% of dataset)
- **Program:** 43,706× imbalance (Program_-1 drums = 32.1%)
- **Tempo:** 1,396× imbalance + RARE (0.14% of dataset)

**Impact:**
- Model accuracy: 22% average
- Tempo: 0%, TimeSig: 0%, Pitch: 8%
- Model mode collapse to frequent tokens

**Solution:**
Weighted loss: weight_i = 1/sqrt(freq_i/total)

**Expected Improvement:**
22% → 60-65% accuracy (3× improvement)

**Thesis Usage:**
- Section 4.3: Class Imbalance Problem
- Figure 4.1: Token Distribution
- Table 4.1: Imbalance Statistics

---

## Analysis Scripts

1. **analyze_token_frequencies.py** - Token frequency per head
2. **explore_datasets.py** - Lakh vs MidiCaps comparison
3. **investigate.py** - Program -1 (drums) investigation
4. **compare_generation.py** - Baseline vs piano-only
5. **analyze_simpleV2.py** - REMI structure validation
6. **analyze_dataset_instruments.py** - Instrument distribution
