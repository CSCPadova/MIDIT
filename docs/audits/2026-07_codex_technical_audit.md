# MIDIT — Technical Code Audit

Audit statico completato: nessun file modificato e nessun training avviato. Le affermazioni marcate **FACT** derivano dal codice letto; **INFERENCE** è una conseguenza diretta del comportamento; **HYPOTHESIS** richiede una verifica runtime.

## 1. `masked_transformer.py`

**Purpose & data flow — FACT.** Il modello riceve `midi_tokens: (B, L)`, somma token embedding e position embedding, applica `LayerNorm`, poi un `nn.TransformerEncoder` bidirezionale senza causal mask. Restituisce hidden states `(B, L, d_model)`, passati separatamente a `MultiHeadOutput`. (`midit/model/masked_transformer.py:371`)

Il masking non avviene nel forward: `training_step` e `validation_step` assumono che il dataset abbia già costruito `midi_tokens` mascherati, `mask_labels` e target. (`midit/model/masked_transformer.py:437`)

**Phase 2/text-conditioning — FACT.** Se e solo se `phase == 2` e arriva `text_encoder_config`, viene costruito FLAN-T5, le sue embedding sono concatenate come `[text | MIDI]`, processate dal medesimo encoder, e alla fine viene estratta solo la parte MIDI. (`midit/model/masked_transformer.py:213`, `:399`)

**FACT.** Nessuno degli entrypoint operativi costruisce il modello con `phase=2`, fornisce `text_encoder_config`, o produce `text_tokens`. `build_model()` forza esplicitamente `phase=1`; `train_infilling.py` ricarica il checkpoint senza convertire la fase. (`midit/model/model_factory.py:105`, `midit/train_infilling.py:142`)

**FACT.** La funzione interna `apply_bar_masking()` implementa 80/10/10 e bar masking, ma non è chiamata dai training step. È quindi codice inattivo nel flusso corrente. (`midit/model/masked_transformer.py:300`, `:457`)

**Strengths.**

- Encoder realmente bidirezionale, appropriato al problema di ricostruzione con contesto a sinistra e destra.
- Embedding, positional encoding e padding-mask sono passati correttamente all’encoder. (`midit/model/masked_transformer.py:392`)
- `model_factory.py` verifica che gli ID del tokenizer siano contigui e che le classi delle teste coprano l’intero vocabolario prima di allocare il modello: è un buon presidio contro checkpoint incompatibili. (`midit/model/model_factory.py:60`)

**Weaknesses / validity risks.**

- **INFERENCE — inerente al paradigma.** Le predizioni di tutti gli slot mascherati vengono prodotte in parallelo da un solo forward. Ogni hidden state usa il contesto globale, ma non c’è una fase iterativa che condizioni una predizione mascherata su quelle appena generate. Questo può produrre combinazioni locali incoerenti.
- **FACT.** `max_seq_length` è un limite rigido della `position_embedding`; input più lunghi causerebbero un indice fuori range. (`midit/model/masked_transformer.py:230`)
- **FACT.** I default interni di vocab e special token sono diversi dal tokenizer runtime; il flusso production è sicuro solo perché passa valori dal dataset/factory. (`midit/model/masked_transformer.py:115`, `midit/model/model_factory.py:105`)
- **FACT.** La Phase 2 è una capacità non integrata, non una fase sperimentale effettivamente supportata da config, dataset e training.

## 2. `multi_head_output.py`

**Purpose & data flow — FACT.** Nove proiezioni lineari indipendenti trasformano `(B, L, d_model)` in logits per `pitch`, `velocity`, `duration`, `program`, `position`, `bar`, `tempo`, `timesig`, `rest`. Pitch e PitchDrum condividono la testa pitch nel dataset e nell’inferenza. (`midit/model/multi_head_output.py:61`)

Per tipo \(t\), la loss effettiva è:

\[
L_t = \frac{\sum_i \mathrm{CE}(\mathrm{logits}_{t,i}, y_{t,i}) \cdot m_i}
{\max(1,\sum_i m_i \cdot [y_{t,i}\neq -100])}
\qquad
L = \sum_t \lambda_t L_t
\]

Questo è esattamente il codice: i target di tipo sbagliato sono `-100`, e il denominatore conta solo posizioni mascherate che appartengono a quel tipo. (`midit/model/multi_head_output.py:177`)

**Strengths.**

- Normalizzare per token mascherati del singolo tipo evita che tempo/time-signature, molto rari, siano diluiti dalla frequenza di pitch/position.
- Le lambda sono applicate dopo la media per tipo; la loro semantica è chiara e auditabile. (`midit/model/multi_head_output.py:199`)
- Le class weights, se disponibili, entrano nella CE prima della riduzione. (`midit/model/multi_head_output.py:181`)

**Weaknesses / risks.**

- **FACT.** Se un tipo non compare tra le posizioni mascherate, il suo contributo diventa zero grazie al denominatore clampato a uno. Questo è ragionevole, ma i log di loss per tipi rari hanno molti zeri e non sono una stima stabile della qualità. (`midit/model/multi_head_output.py:203`)
- **FACT.** Le nove teste non impongono vincoli sintattici tra tipi o tra posizioni. La testa pitch può per esempio produrre `PitchDrum_*` in uno slot il cui originale era `Pitch_*`; in inferenza entrambe le famiglie sono ammesse dalla stessa testa. (`midit/infer.py:69`)
- **FACT.** Le class weights vengono lette da un file relativo al current working directory e, se assente, il modello prosegue senza pesi. Inoltre sono buffer non persistenti, quindi non sono nel checkpoint. (`midit/model/masked_transformer.py:160`) Rischio: un fine-tuning/valutazione ricaricata altrove può avere una loss diversa senza fallire.

## 3. `lakh_dataset.py` e `masking.py`

**Purpose & data flow — FACT.**

- Carica file `.pt` con almeno `tokens`, `bar_positions`, `metadata`; ricava le finestre on-the-fly. (`midit/data/lakh_dataset.py:263`)
- Ricostruisce il tokenizer REMI+ direttamente nel codice; non legge le opzioni `tokenizer:` del YAML. (`midit/data/lakh_dataset.py:98`)
- Costruisce target di classe per ciascuna delle nove teste; Pitch e PitchDrum sono accorpati. (`midit/data/lakh_dataset.py:158`)
- Dopo il masking, conserva target solo nelle posizioni selezionate; tutte le altre diventano `-100`. (`midit/data/lakh_dataset.py:395`)

**Chunking — FACT.**

- `bar_based`: tutte le finestre consecutive di `bars_per_chunk` barre, stride 1; una stessa canzone produce quindi chunk molto correlati. (`midit/data/lakh_dataset.py:255`, `:426`)
- `token_based`: una finestra casuale per file a ogni accesso, senza confini di barra. (`midit/data/lakh_dataset.py:472`)
- Il collate usa `0` come PAD hardcoded e non restituisce `bar_positions`; questo non rompe il flusso attuale perché il masking è già avvenuto nel dataset. (`midit/data/lakh_dataset.py:551`)

**Split — FACT.** I file sono raggruppati per stem, togliendo solo il suffisso `_shift±N`; gli stem sono ordinati, mescolati con NumPy seed globale 42 e divisi 80/10/10. Questo evita leakage delle augmentation di pitch shift riconosciute. (`midit/data/lakh_dataset.py:49`)

**Masking — FACT.**

- `TokenLevelMasking`: seleziona `max(1, floor(L * mask_ratio))` posizioni uniformi e fa 80% `[MASK]`, 10% token casuale dell’intero vocabolario, 10% invariato. (`midit/data/masking.py:307`)
- `BarLevelMasking`: per ogni barra selezionata sceglie tipi presenti e maschera tutte le occorrenze di quei tipi. (`midit/data/masking.py:101`)
- `BarFillMasking`: sceglie una finestra contigua casuale di 1–`max_infill_bars` barre e maschera solo Pitch, PitchDrum, Velocity, Duration e Program; struttura, Position, tempo, time-signature e rest restano visibili. (`midit/data/masking.py:396`, `:523`)

**Validation determinism.**

- **FACT.** Entrambi gli script di training creano il validation dataset con `apply_masking=True`, non senza masking. (`midit/train_phase1.py:60`, `midit/train_infilling.py:79`)
- **FACT.** Nel Phase 1 token-level il validation masking usa il modulo globale `random` e non fissa un seed. (`midit/data/masking.py:317`)
- **FACT.** Nel BarFill validation il costruttore usa `random.Random(1234)`, ma il generatore è stateful e avanza a ogni campione; i dataloader mantengono inoltre worker persistenti. Quindi non è lo “stesso task identico” a ogni epoca, nonostante il commento lo dichiari. (`midit/data/lakh_dataset.py:223`, `midit/data/masking.py:445`, `midit/train_infilling.py:93`)
- **INFERENCE.** L’ordine dei worker può rendere anche la sequenza effettiva dei mask dipendente dal run; fissare un generatore stateful non equivale a derivare il mask da `(seed, sample_id, epoch)`.

**Risks / limits.**

- I chunk adiacenti della stessa canzone non attraversano train/val, ma il validation loss pesa le canzoni lunghe e le loro finestre sovrapposte più delle canzoni corte.
- Eccezioni in preload/chunk indexing sono silenziosamente saltate. Questo può ridurre il dataset reale rispetto a quello riportato e può falsare il calcolo manuale degli step. (`midit/data/lakh_dataset.py:266`)
- Il commento di `TokenLevelMasking` parla di `ceil`, il codice usa `int`, cioè floor. (`midit/data/masking.py:269`, `:311`)

## 4. Training Phase 1 e Phase 1b

**FACT.** `DataLoaderWrapper` rinomina `input_ids` in `midi_tokens`; non modifica target, mask o contenuto. (`midit/train_phase1.py:17`)

**Scheduler — FACT.** Entrambi gli script calcolano:

```text
effective_batches = min(limit_train_batches, len(train_loader_base))
steps_per_epoch = effective_batches // accumulate_grad_batches
total = steps_per_epoch * epochs
```

e passano `total` al modello per il cosine scheduler. (`midit/train_phase1.py:106`, `midit/train_infilling.py:110`)

**Valutazione del bug storico.**

- **FACT.** Per i valori attuali (`25000 // 4` in Phase 1, `50000 // 4` in Phase 1b), non c’è resto: la formula coincide con il valore atteso se il loader produce davvero tutti quei batch.
- **INFERENCE — fragilità.** Per un limite non divisibile per `accumulate_grad_batches`, il floor può sottostimare gli optimizer step se Lightning esegue lo step finale con accumulo parziale. Anche batch `None` saltati dal wrapper rendono il conteggio potenzialmente non coincidente.
- **Runtime check preciso:** eseguire soltanto un smoke test di un’epoca con `limit_train_batches=5`, `accumulate_grad_batches=4`, poi confrontare `trainer.global_step` con lo `total_training_steps` stampato; non serve un job di training reale.

**Checkpoint fine-tuning — FACT.** `train_infilling.py` usa `load_from_checkpoint()` senza parametri di override, quindi il costruttore riceve gli hyperparameter del checkpoint. Poi cambia soltanto gli attributi runtime `model.lr`, `model.warmup_steps` e `model.num_training_steps`. (`midit/train_infilling.py:142`)

| Parametro | Nel Phase 1b |
|---|---|
| `lr`, `warmup_steps`, `num_training_steps` | Sovrascritti e realmente usati dall’optimizer/scheduler |
| architettura, `vocab_sizes`, special IDs, `lambda_weights`, `weight_decay`, `mask_prob`, `bar_level_masking`, `phase`, `class_weights` | Riletti dal checkpoint; i corrispondenti valori nel YAML infilling non vengono riapplicati al modello |
| `cfg.masking.*`, chunking, batch, Trainer options | Attivi per costruire il dataset/Trainer del fine-tuning |

**Conseguenza — FACT.** Cambiare `lambda_weights` o `weight_decay` in `config_infilling.yaml` non cambia rispettivamente la loss o l’optimizer di quel fine-tuning; cambiare `train.lr` sì. (`midit/model/masked_transformer.py:539`)

## 5. Inferenza

**FACT.** `infer.py` ricostruisce lo stesso tokenizer hardcoded del dataset, ricrea lo split song-level val con seed 42 e seleziona un file originale per stem. Estrae sempre le prime 8 barre. (`midit/infer.py:36`, `:112`, `:151`)

**Masking e scaffold — FACT.** Per le barre selezionate, maschera soltanto content token; conserva Bar, Position, Tempo, TimeSig e Rest nel posto originale. (`midit/infer.py:178`)

**Oracle token-type selection — FACT.** Per ogni posizione mascherata, `reconstruct()` legge l’ID originale per sapere quale testa interrogare; poi fa argmax di quella testa e converte classe → token ID. (`midit/infer.py:248`)

Quindi l’esperimento fornisce al modello/evaluatore:

1. posizione e numero esatto di token mancanti;
2. token strutturali completi;
3. tipo reale di ogni slot mascherato.

**INFERENCE.** Questa è una valutazione di content reconstruction condizionata, non di infilling libero in cui il sistema debba decidere lunghezza, ordine e tipo di token mancanti. Va dichiarato esplicitamente in tesi.

**FACT.** L’argmax è deterministico a parità di checkpoint/input/modalità eval; non ci sono sampling, temperature o decoding iterativo nel percorso di produzione. (`midit/infer.py:267`)

**Risks.**

- `infer.py` non controlla che l’8-bar chunk sia sotto `max_seq_length`; `generate_samples_batch.py` scarta invece lunghezze `>=8000`, non `>8192`. (`scripts/generate_samples_batch.py:80`)
- `save_midi()` prova due API e può fallire restituendo `False`; il batch generator non controlla i return value e può contare come successo una seed con export fallito. (`midit/infer.py:365`, `scripts/generate_samples_batch.py:101`)
- `scripts/infer_from_midi.py --skip_filters` non bypassa davvero F1/F2/F3: chiama `_process_file(..., False)` e il preprocessore applica comunque i filtri; l’opzione sopprime solo l’eccezione successiva, ma il `.pt` potrebbe non esistere. (`scripts/infer_from_midi.py:48`, `scripts/Pre_Processing/preprocess_v2.py:106`)

## 6. Generazione batch e metriche

**Generazione — FACT.**

- Il checkpoint viene caricato una volta, poi il batch genera una directory per seed. (`scripts/generate_samples_batch.py:133`)
- Una seed indica un indice modulo il numero di file val; il fallback `seed + offset` può far convergere seed differenti sullo stesso brano. La provenienza non viene salvata in un manifest. (`scripts/generate_samples_batch.py:63`)
- Se `reconstructed.mid` esiste già, la seed viene saltata senza verificare checkpoint, data source o `mask_bars`. (`scripts/generate_samples_batch.py:58`)

**INFERENCE.** Se `n_seeds` supera il numero dei brani val o sono usati fallback, le medie possono pesare alcuni brani più volte. È verificabile confrontando i log di console o aggiungendo un manifest seed→source; il codice attuale non lo conserva.

### CP, GS, PCHE

**FACT.** `compute_infilling_metrics.py` calcola sui soli bar richiesti:

- **CP:** media per barra della cosine similarity tra chroma a 12 classi, pesata per durata; aggrega tutte le tracce. (`scripts/Evaluation_Infilling/compute_infilling_metrics.py:198`)
- **GS:** media per barra dello Jaccard fra grid binarie degli onset a sedicesimi; ignora pitch, velocity, durata e identità della traccia. (`scripts/Evaluation_Infilling/compute_infilling_metrics.py:224`)
- **PCHE:** valore assoluto della differenza fra le entropie Shannon dei pitch-class histogram aggregati su tutte le barre mascherate. (`scripts/Evaluation_Infilling/compute_infilling_metrics.py:255`)

**Risks.**

- **FACT.** PCHE non confronta gli histogram: due distribuzioni molto diverse possono avere uguale entropia e quindi PCHE=0.
- **FACT.** Se entrambe le barre sono vuote, `_jaccard` restituisce 1, ma il docstring dichiara che CP e GS penalizzano entrambe le barre vuote a zero. Il codice prevale sul commento. (`scripts/Evaluation_Infilling/compute_infilling_metrics.py:45`, `:166`)
- **INFERENCE.** CP alto può indicare pitch-class simili pur con ottave, strumenti, melody contour e armonia effettiva diversi; GS alto può riflettere solo gli onset.

### Crop automatico e metriche MusPy

**FACT.** Lo script batch individua una barra “mascherata” soltanto se original ha almeno una nota e `masked.mid` ha zero note. Se non rileva barre, calcola sul pezzo completo; questo fallback non è scritto come colonna nel CSV. (`scripts/Evaluation_Infilling/compute_muspy_metrics.py:88`, `:284`)

**Risk for validity — FACT.** Un fallback full-piece include le barre non mascherate, identiche fra original e reconstruction, e può gonfiare CP/GS. Senza flag nel CSV non è possibile escludere o quantificare tali righe a posteriori.

**FACT.** Le S-I metrics del batch sono calcolate sulla reconstruction croppata alle barre mascherate quando possibile: pitch-class entropy, groove consistency, scale consistency, pitch range, numero di pitch usati. (`scripts/Evaluation_Infilling/compute_muspy_metrics.py:231`)

### JS similarity

**FACT.** Il percorso `Matteo` usa per JS solo:

- `muspy_polyphony_rate`
- `muspy_groove_consistency`
- `muspy_scale_consistency`

e calcola \(100 \exp(-2 \cdot \overline{JS})\), con JS numerica tra gaussiane adattate da media e deviazione standard per metrica. (`scripts/Evaluation_Infilling/Matteo/js_similarity.py:26`, `:66`)

**FACT.** `compute_model_agg.py` prova a croppare la reconstruction alle barre mascherate, ma in caso di problema torna al file completo; `compute_reference_agg.py` calcola invece sempre sull’intero MIDI di riferimento. (`scripts/Evaluation_Infilling/Matteo/compute_model_agg.py:108`, `scripts/Evaluation_Infilling/Matteo/compute_reference_agg.py:80`)

**Risk for validity — FACT.** La distribuzione modello può essere di 1–3 barre, mentre la reference è un brano intero. Non è un confronto a scope equivalente.

**FACT.** Il calcolo di groove consistency non è uniforme: il modulo MusPy semplice usa `4`, l’aggregatore modello usa `resolution` del MIDI croppato. (`scripts/Evaluation_Infilling/Matteo/muspy_metrics.py:74`, `scripts/Evaluation_Infilling/Matteo/compute_model_agg.py:80`)

## 7. Configurazioni: chiavi attive e dead

Questa tabella si riferisce agli entrypoint operativi `train_phase1.py` e `train_infilling.py`; letture in script diagnostici non rendono una chiave operativamente attiva.

| Config key | Dove viene letta | Stato |
|---|---|---|
| `tokenizer.{type,pitch_range,beat_res,num_velocities,use_programs,one_token_stream_for_programs,use_pitchdrum_tokens,use_chords,use_rests,use_tempos,use_time_signatures,use_controls,special_tokens,delete_equal_successive_tempo_changes,delete_equal_successive_time_sig_changes,vocab_size}` | Nessuna lettura runtime; tokenizer hardcoded in `lakh_dataset.py:98–111` | **DEAD** |
| `chunking.strategy` | `lakh_dataset.py:255,313` | **ACTIVE** |
| `chunking.bars_per_chunk` | `lakh_dataset.py:257,441` | **ACTIVE** |
| `chunking.tokens_per_chunk` | `lakh_dataset.py:483` | **ACTIVE** |
| `chunking.max_tokens` | `lakh_dataset.py:378`; `model_factory.py:114` | **ACTIVE** |
| `chunking.min_tokens_per_bar`, `chunking.filter_empty_bars`, `chunking.raw_data_dir` | Nessuna lettura operativa | **DEAD** |
| `chunking.preprocessed_dir` | `lakh_dataset.py:37` | **ACTIVE** |
| `masking.mask_prob` | TokenLevel dataset; passato anche a funzione modello inattiva | **ACTIVE in Phase 1; inert per BarFill Phase 1b** |
| `masking.bar_level` | Routing legacy dataset `lakh_dataset.py:218–221` | **ACTIVE se non esiste `masking.strategy`** |
| `masking.bar_mask_prob`, `masking.type_mask_ratio` | BarLevelMasking `lakh_dataset.py:235–241` | **Conditionally active** |
| `masking.mask_strategy` | TokenLevel / BarFill `lakh_dataset.py:223–248` | **ACTIVE** |
| `masking.strategy` (solo infilling) | Routing `bar_fill` | **ACTIVE** |
| `masking.max_infill_bars` (solo infilling) | BarFill constructor | **ACTIVE** |
| Tutte le 9 `lambda_weights.*` | `model_factory.py:99` | **ACTIVE in Phase 1; checkpoint-controlled/inert in Phase 1b** |
| `model.type`, `model.vocab_size`, `model.max_seq_length` | Nessuna lettura operativa | **DEAD** |
| `model.{d_model,nhead,num_encoder_layers,dim_feedforward,dropout}` | `model_factory.py:109–113` | **ACTIVE in Phase 1; inert in Phase 1b checkpoint load** |
| `model.special_tokens.{pad,bos,eos,mask}` | Factory hardcodes `0,1,2,3` | **DEAD** |
| `train.phase`, `train.devices`, `train.monitor` | Nessuna lettura operativa; phase e monitor hardcoded, devices=1 | **DEAD** |
| `train.{batch_size,epochs,accelerator,lr,gradient_clip_val,precision,accumulate_grad_batches,limit_train_batches,limit_val_batches,save_top_k,log_every_n_steps,val_check_interval}` | train entrypoint | **ACTIVE** |
| `train.weight_decay` | `model_factory.py:120` | **ACTIVE Phase 1; inert Phase 1b** |
| `train.warmup_steps` | factory; poi override infilling | **ACTIVE** |
| `train.checkpoint_path` | `train_infilling.py:63` | **ACTIVE only Phase 1b** |
| `logging.use_wandb` | entrambi gli entrypoint | **ACTIVE** |
| `logging.wandb_project` | `train_phase1.py:172` | **ACTIVE Phase 1; DEAD Phase 1b** (nome hardcoded) |
| `paths.{lakh_dir,preprocessed_phase1,midicaps_dir,preprocessed_phase2,checkpoints_dir,logs_dir}` | Nessuna lettura operativa | **DEAD** |
| `paths.checkpoints_infilling` | Nessuna lettura; destinazione hardcoded da `logs_dir` | **DEAD** |
| `defaults` di `config_infilling.yaml` | Hydra composition | **ACTIVE** |

`conf/config.py` contiene dataclass non importate dagli entrypoint: non valida né rende attive le chiavi YAML. (`conf/config.py:7`)

## Data flow effettivo

```text
Raw MIDI
  └─ preprocess_v2.py / tokenizzazione REMI+ hardcoded
      └─ file .pt: tokens, bar_positions, metadata
          └─ split per song stem (seed NumPy 42)
              └─ chunking: finestre per barre oppure finestra token casuale
                  └─ masking nel Dataset
                      ├─ Phase 1: TokenLevel 80/10/10
                      └─ Phase 1b: BarFill, content-only su 1..N barre
                          └─ targets tipo-specifici, -100 fuori dal mask
                              └─ Transformer bidirezionale
                                  └─ 9 linear heads
                                      └─ somma CE per tipo pesata da lambda

Inference:
.pt / MIDI raw → 8 barre → structural scaffold conservato
  → content token [MASK] → encoder → argmax della testa scelta dall’ID originale
  → token REMI+ ricostruiti → MidiTok → reconstructed.mid
```

## Top 10 limitazioni, ordinate

1. **(a) Inerente al paradigma:** slot e lunghezza del gap sono fissi; il sistema non può generare contenuto di lunghezza variabile. (`midit/infer.py:248`)
2. **(a) Inerente al paradigma:** decoding parallelo non autoregressivo/iterativo; nessuna correzione delle dipendenze fra token generati.
3. **(b) Implementation choice:** inferenza usa il tipo originale come oracle per ogni slot; non misura la scelta del token type. (`midit/infer.py:258`)
4. **(b) Implementation choice:** nessun vincolo di grammatica REMI+ o di coerenza tipo/slot sulle nove teste.
5. **(c) Bug/fragilità:** validation masking non è realmente fisso tra epoche; Phase 1 è stocastico. (`midit/data/lakh_dataset.py:223`)
6. **(c) Bug/fragilità:** `config_infilling.yaml` sembra controllare più hyperparameter del modello di quanti ne controlli davvero; molti restano quelli del checkpoint. (`midit/train_infilling.py:142`)
7. **(c) Bug/fragilità:** scheduler manuale con floor per gradient accumulation non è robusto a resti o batch saltati. (`midit/train_phase1.py:106`)
8. **(c) Bug/fragilità:** fallback metriche full-piece non è tracciato nel CSV; può gonfiare CP/GS. (`scripts/Evaluation_Infilling/compute_muspy_metrics.py:119`)
9. **(b) Implementation choice:** JS confronta crop del modello e MIDI completi di reference. (`scripts/Evaluation_Infilling/Matteo/compute_model_agg.py:150`)
10. **(b) Implementation choice:** batch generation può duplicare brani e non salva il mapping seed→source. (`scripts/generate_samples_batch.py:63`)

## Discrepanze con la descrizione iniziale

- La descrizione di architettura Phase 1, 9 teste, BarFill content-only e argmax con scaffold è coerente con il codice.
- “15% token-level MLM” è approssimato: il codice seleziona esattamente `floor(0.15L)`, almeno uno, non una Bernoulli indipendente per token. (`midit/data/masking.py:311`)
- Phase 2 text-conditioned esiste nel file, ma non è un percorso addestrabile/attivabile dagli entrypoint presenti.
- “Deterministica” è corretta per argmax, ma non per validation masking Phase 1; e non è identica tra epoche neppure per BarFill validation.
- `vocab=550` è documentato, ma il codice non lo assume: costruisce il vocabolario runtime e lo valida. Il valore effettivo di un particolare ambiente/checkpoint non è stato verificato runtime.
- Il parametro count “~42.6M” non è verificabile dal solo sorgente senza materializzare il tokenizer e il modello; il codice lo stampa ma non conserva un valore di riferimento. (`midit/model/masked_transformer.py:275`)

## Hardening senza re-training

- Rendere validation masking index-deterministico: seed derivato da `(validation_seed, sample_index)` e non da un RNG che avanza; scrivere mask seed/indice nei log.
- Calcolare gli optimizer step con la semantica Lightning effettiva e aggiungere test per `limit % accumulate != 0`.
- In Phase 1b, stampare e salvare un diff esplicito fra hyperparameter del checkpoint e quelli applicati; fallire se una chiave YAML richiesta è inert.
- Eliminare le chiavi dead o farle diventare l’unica fonte del tokenizer, special ID, log/checkpoint paths.
- Salvare un manifest immutabile per ogni sample: checkpoint hash, source file, split, chunk, mask bars, token count, export status.
- Fallire o marcare esplicitamente le righe metriche con crop fallito; non usare full-piece fallback nelle medie masked-region.
- Uniformare `groove_consistency` e comparare regioni reference/model con durata e crop equivalenti.
- Correggere `--skip_filters` oppure rimuovere l’opzione.
- Aggiungere guardia `len(tokens) <= model.max_seq_length` in ogni entrypoint inferenza.

## Future work che richiede nuovo training/architettura

- Infilling a lunghezza variabile con span/length tokens o generazione iterativa.
- Decoding constrained dalla grammatica REMI+, incluso distinguere Pitch e PitchDrum.
- Ricostruzione multi-pass/Gibbs o decoder autoregressivo locale per coerenza fra token mascherati.
- Training/evaluation senza oracle token-type per una misura più vicina all’infilling reale.
- Se Phase 2 è un obiettivo di tesi, dataset text–MIDI, config, checkpoint policy e loop di training dedicati; oggi il solo encoder FLAN-T5 non costituisce un percorso sperimentale completo.
