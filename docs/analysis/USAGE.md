> **Historical notes (Italian, early 2026), restored from a stash.** They describe the V1 tokenizer and a double-masking bug that has since been fixed; they do not describe the current code.

MIDIT è un sistema Transformer masked language model per la generazione di musica MIDI. L'obiettivo finale è generare musica MIDI a partire da descrizioni testuali (Text-to-MIDI), simile a BERT ma per musica.

Architettura in due fasi
Fase	Descrizione	Stato
Phase 1	Pre-training MLM su dataset Lakh MIDI	Completata (epoch 47)
Phase 2	Fine-tuning text-conditioned con MidiCaps	In sviluppo
Stack tecnologico
Framework: PyTorch + PyTorch Lightning
Tokenizer MIDI: miditok con formato REMI (~520 token vocab)
Text encoder: FLAN-T5 (congelato, per Phase 2)
Configurazione: Hydra + YAML
Logging: TensorBoard + WandB

Architettura modello (Phase 1)

MIDI tokens → Embedding (vocab ~520 → d_model=512)
            → Positional Embedding (sinusoidale)
            → Token Type Embedding (9 tipi)
            → Transformer Encoder (12 layer, 8 head, FFN=2048)
            → 9 teste di output separate:
               pitch (151) | velocity (32) | duration (128)
               program (129) | position (96) | bar (1)
               tempo (32) | timesig (9) | rest (32)
Pipeline dati
MIDI files Lakh → tokenizzati con REMI → .pt files salvati
LakhDataset carica i .pt, applica chunking (8 barre per chunk)
Masking on-the-fly (bar-level o token-level, 15%)
Batch con 9 dizionari target separati per tipo di token


1. DOPPIO MASKING — Bug Critico
Questo è il bug più grave. Il flusso attuale è:


Dataset (lakh_dataset.py:223-258)
  → applica masking → masked_tokens (con [MASK])
  → crea targets con valori solo nelle posizioni mascherate dal dataset
  → restituisce input_ids = masked_tokens

DataLoaderWrapper (train_phase1.py:30-35)
  → passa input_ids come midi_tokens al modello
  → NON passa mask_labels (dove erano le maschere del dataset)

training_step (masked_transformer.py:455)
  → riceve midi_tokens già mascherati
  → applica SECONDO masking indipendente → mask_indices (nuove posizioni)
  → calcola il loss usando mask_indices come denominatore
Effetto concreto: Il loss viene calcolato solo sull'intersezione tra le posizioni mascherate dal dataset e quelle mascherate dal modello. Con masking 15%+15%, solo ~2.25% dei token contribuisce, invece del 15% previsto. Segnale di training drasticamente ridotto.

Riepilogo per Priorità
Priorità	Problema	Impatto
CRITICO	Doppio masking (dataset + model)	Training signal ~7× più debole del previsto
ALTO	_create_targets_dict chiamata 2×	Spreco computazionale in ogni batch
MEDIO	bar_mask_prob / val_check_interval ignorati	Config non riflette la realtà
MEDIO	LR scheduler max_steps=10000 hardcoded	LR instabile su run lunghi
BASSO	config.yaml vocab_sizes e special tokens stantii	Confusione per chi legge
