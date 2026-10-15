# Probe anti-cheat — l'inferenza non può "barare"

**Branch:** `audit-fixes` · **Data:** 21 luglio
**Script:** [`scripts/audit_fixes/probe_no_cheat.py`](../../../scripts/verification/probe_no_cheat.py)

## Motivazione
Bisogna dimostrare **meccanicamente** che in inferenza il modello **non vede mai**
i token ground-truth nelle posizioni mascherate, e che le predizioni provengono
**solo** dai logit del modello. In altre parole: escludere che l'infilling sia un
copia-incolla mascherato. Tre livelli di prova:
- **PART A** — traccia statica del codice (dove nasce l'input, cosa vede il modello);
- **PART B** — probe runtime: agli slot mascherati l'input è il token MASK, non
  l'originale, e la predizione arriva dai logit;
- **PART C** — test di dipendenza dal contesto: se perturbo il contesto, le
  predizioni cambiano → il modello **condiziona sul contesto**, non ripete nulla.

Nessuna modifica ai file di pipeline; script usa-e-getta; nessun accesso in
scrittura a `outputs/samples_1000`. CPU sufficiente (no GPU).

---

## PART A — Traccia statica (evidenza file:linea)

### A.1 — L'input del modello ha MASK nelle posizioni mascherate, non l'originale
L'input è `masked_tokens`, costruito in `apply_bar_mask` (`infer.py:198-212`):
```
198  masked = tokens.clone()
...
210      if tok_name.startswith(NOTE_PREFIXES):
211          masked[pos] = MASK_ID            # slot di contenuto mascherato = MASK (id 3)
212          mask_labels[pos] = True
```
Quel tensore è l'input del modello (`infer.py:267-269`):
```
267  with torch.no_grad():
268      hidden = model(masked_tokens.unsqueeze(0).to(device))   # input = MASKED
269      logits = model.multi_head_output(hidden)
```
Non viene passata alcuna attention mask (`midi_attention_mask=None`): nessun leak
nemmeno da quel canale.

### A.2 — Gli id originali servono solo a: (a) selezione oracle della testa, (b) fallback
`original_tokens` è letto per-slot (`infer.py:279`) e usato **solo** per:
- **(a) selezione della testa (oracle):** `head_name = type_for_id.get(orig_id)` (`infer.py:286`);
- **(b) fallback out-of-range:** `reconstructed[pos_int] = orig_id` a `infer.py:292`
  (pos oltre i logit) e `infer.py:300` (class_idx ≥ len(names));
- più un **passthrough** dei soli special token (`infer.py:282-284`), che non si
  applica mai a uno slot di **contenuto** mascherato.

Tutte queste scritture toccano `reconstructed` (l'**output**), **mai**
`masked_tokens` (l'input) e **mai** una attention mask. L'input è fissato prima
del forward e non viene più modificato.

### A.3 — Catena completa: le predizioni vengono dai logit
`forward` → logit multi-head (`infer.py:268-269`) → per slot mascherato
`class_idx = logits[head_name][0, pos_int].argmax()` (`infer.py:295`) →
`pred_id = vocab.get(names[class_idx])` (`infer.py:303`) →
`reconstructed[pos_int] = pred_id` (`infer.py:305`) → detokenizzazione
(`save_midi`, `infer.py:365-382`). Gli **unici** punti in cui un id originale
finisce nell'output sono i due fallback (292, 300) e il passthrough special
(283). Tutto il resto è `argmax` sui logit.

**Conclusione PART A:** non esiste alcun percorso che copi gli id di input nelle
predizioni, oltre al fallback noto (che nel Task 5 conteremo separatamente).

---

## PART B — Probe runtime (input == MASK, predizioni dai logit)

Lo script riproduce **esattamente** la selezione seed→file e il masking della
pipeline riusando le funzioni di `infer.py` (`get_val_files`, `extract_chunk`,
`apply_bar_mask`) e l'offset-loop di `generate_samples_batch`. Poi fa un forward
e stampa, per i primi 10 slot di contenuto mascherati:

```
slot | input_id | MASK? | orig_id | pred_id | equal? | orig->pred (nomi) | source
```

- `input_id` deve essere `MASK_ID` (3) ovunque → se no, `*** ALERT: CHEAT POSSIBLE ***`.
- `source` mostra `logits_argmax` vs i fallback noti.
- **Atteso:** `input==MASK: 10/10`; `predicted==original` una **minoranza**
  (coerente col token-diff 46-73% già misurato).

## PART C — Dipendenza dal contesto (test economico e potente)

Stesso seed: si perturba **solo** il contenuto delle barre di **contesto** (token
di contenuto **fuori** dallo span token delle barre 3-5), mescolandone i valori
con seed fisso. Le barre mascherate e tutto lo scaffold (Position/Bar/TimeSig/…)
restano **byte-identici** (`masked-bar region unchanged: YES`). Poi si rifà il
forward e si contano quante delle 10 predizioni cambiano.

- **Atteso:** `predictions changed by context shuffle: > 0` → il modello
  condiziona sul contesto, non ripete nulla. Se `= 0` → `*** ALERT ***`.

Nota metodologica: si mescola il **contenuto** del contesto lasciando fisse tutte
le posizioni/struttura e la regione mascherata (perturbazione in-place). Reordinare
interi blocchi-barra sposterebbe gli indici e i position-embedding, sporcando il
test; per questo si è scelta la shuffle in-place del contenuto.

---

## Come eseguire (cluster)

Servono: (1) il **checkpoint** che ha generato `samples_1000`; (2) la stessa
`--data_dir` da cui sono stati generati i campioni (`preprocessed_v2_50k`),
altrimenti la selezione seed→file di seed_79 non combacia.

```bash
cd ~/MIDIT && git checkout audit-fixes && git pull
source .venv/bin/activate                 # CPU ok, no GPU necessaria
python scripts/audit_fixes/probe_no_cheat.py \
    --checkpoint /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt \
    --data_dir   /nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k \
    --seed 79 --n_slots 10 --mask_bars 3 4 5
```

## Cosa conferma il successo
- **PART B:** `input==MASK: 10/10`, `predicted==original` in minoranza, nessuna riga ALERT.
- **PART C:** `masked-bar region unchanged: YES` e `predictions changed by context shuffle > 0`.

Insieme provano meccanicamente che l'inferenza **non può barare**: il modello non
vede mai il ground-truth agli slot mascherati e le sue predizioni dipendono dal
contesto.

---

## Risultati runtime
_Da eseguire sul cluster — inserire qui l'output di PART B (tabella 10 slot) e
PART C (conteggio `changed`) una volta girato._
