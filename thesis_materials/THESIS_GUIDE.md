# MIDIT Thesis Materials Guide

## 🎯 Main Story: Class Imbalance Discovery

**Date:** December 11, 2024
**Impact:** Project-defining discovery

### The Problem
Training Phase 1 showed:
- Validation loss: 0.098 (looks good!)
- Accuracy: 22% (terrible!)
- 8/9 heads failed completely

### The Investigation
Analyzed 1000 MIDI files, discovered:
- Pitch: 50,011× imbalance
- TimeSig: 90,441× imbalance (94.4% is 4/4)
- Tempo/TimeSig: 0% accuracy → never learned

### Root Cause
Cross-entropy without weights → model predicts frequent tokens only

### Solution
1. Weighted loss (inverse frequency)
2. Token-level masking (better rare token exposure)
3. Corrected lambda weights

### Expected Results
- Overall: 22% → 60-65% (3× improvement)
- Tempo: 0% → 40-50% (from zero!)
- TimeSig: 0% → 70-80% (from zero!)

---

## 📁 Materials Organization

### 01_approaches_explored/
- 3 diffusion model versions
- old_config/
- **Thesis:** Related Work, Methodology

### 02_pipeline_evolution/
- 4 dataloader versions (evolution)
- **Thesis:** Data Pipeline, Figure 3.2

### 03_analysis_results/
- 6 analysis scripts
- Token frequency analysis (KEY!)
- **Thesis:** Class Imbalance chapter (MAIN)

### 04_figures_data/
- token_frequencies.pt
- accuracy results
- **Thesis:** All figures/tables data

### 05_lessons_learned/
- Documentation
- Thesis outline

---

## 📊 Key Figures/Tables

### Critical (MUST HAVE):
1. **Figure 4.1:** Token Frequency Distribution
   - Barplot per head showing imbalance
   - Data: `04_figures_data/token_frequencies.pt`

2. **Table 4.1:** Class Imbalance Statistics
   - 50,011× pitch, 90,441× timesig, etc.

3. **Figure 4.2:** Accuracy Before/After
   - 22% → 60-65% improvement

### Supporting:
- Figure 3.1: Diffusion vs MLM Architecture
- Figure 3.2: Pipeline Evolution
- Table 3.1: Dataloader Comparison

---

## ✍️ Thesis Chapters

### Chapter 3: Methodology
- 3.1: Approach Evolution (diffusion → MLM)
- 3.2: Pipeline Evolution (dataloaders)
- 3.3: Dataset Selection

### Chapter 4: Implementation (MAIN CHAPTER)
- 4.1: Multi-Head Architecture
- 4.2: Masking Strategies
- **4.3: Class Imbalance Problem ⭐ (YOUR CONTRIBUTION)**
- 4.4: Weighted Loss Solution

### Chapter 5: Experiments
- 5.1: Phase 1 Results (before/after)
- 5.2: Ablation Studies

---

## 🎓 Your Contribution

**Main novelty:** Systematic identification and solution of class imbalance in REMI+ multi-head MIDI generation

**Evidence:**
- Quantified problem (50,011× imbalance)
- Diagnosed root cause (unweighted loss)
- Implemented solution (weighted loss + token masking)
- Expected 3× accuracy improvement

**This is your thesis story!** 📖
