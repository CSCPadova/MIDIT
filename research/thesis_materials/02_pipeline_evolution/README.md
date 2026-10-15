# Pipeline Evolution - MIDIT Project

## Dataloader Versions (Nov-Dec 2024)

### v1: midicaps_dataloader.py
- Basic MIDI loading
- Initial tokenization attempt
- Problems: [DA DOCUMENTARE]

### v2: midicaps_dataloader_remi.py
- REMI+ tokenization integrated
- PIANO-ONLY filtering
- Problems: [DA DOCUMENTARE]

### v3: midicaps_dataloader_v2.py
- Refinement
- Better handling of [DA DOCUMENTARE]

### v4: midicaps_dataloader_v2_filtered.py
- Quality filtering
- PIANO-ONLY focus
- Why: [DA DOCUMENTARE]

### Current: lakh_dataset.py
- MLM pre-training (Phase 1)
- Chunk-based loading
- Bar-level masking → Token-level masking (optimized)

### Thesis Usage:
- Section 3.2: Data Preprocessing Pipeline
- Figure 3.2: Pipeline Evolution Diagram
- Table 3.1: Dataloader Features Comparison
