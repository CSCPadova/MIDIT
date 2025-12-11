# Approaches Explored - MIDIT Project

## Diffusion Models (Nov-Dec 2024)

### Files:
- `diffusion_transformer.py` - Initial implementation
- `diffusion_transformer_v2.py` - Refinement
- `diffusion_transformer_v2_remi.py` - REMI tokenization integration

### Key Insights:
1. Why Diffusion?
   - [DA DOCUMENTARE] State-of-art for generation tasks
   - [DA DOCUMENTARE] Continuous latent space
   
2. Problems Encountered:
   - [DA DOCUMENTARE] Training instability?
   - [DA DOCUMENTARE] Slow generation?
   - [DA DOCUMENTARE] Quality issues?

3. Why Pivot to MLM?
   - [DA DOCUMENTARE] BERT-style masking simpler
   - [DA DOCUMENTARE] Better for discrete tokens
   - [DA DOCUMENTARE] Faster training/inference

### Thesis Usage:
- Section 2.2: Related Work - Diffusion Models
- Section 3.1: Methodology - Approach Evolution
- Figure 3.1: Architecture Comparison (Diffusion vs MLM)
