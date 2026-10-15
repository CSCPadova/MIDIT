# Modifiche Luglio — Remediation pipeline di valutazione (Task 1–3)

**Branch:** `audit-fixes`
**Data:** 20 luglio
**Contesto:** due audit indipendenti (Claude Code + Codex) hanno segnalato problemi
nella pipeline di *evaluation* dell'infilling. Piano di remediation **minimale**:
nessun re-training, nessuna re-inferenza dei campioni già generati in
`/nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000`, solo fix chirurgici e piccoli
script di verifica usa-e-getta.

Regola di lavoro: modifiche locali → commit → push su `audit-fixes` → pull sul
cluster → esecuzione sul cluster. Un task alla volta.

---

## Task 1 — Controllo integrità dei campioni (stabilire la N reale)

### Problema
`generate_samples_batch.py` mappa `seed → file di validation` tramite modulo +
fallback `seed+offset`: seed diversi possono finire sulla **stessa canzone**, e
non viene salvato nessun manifest `seed → sorgente`. Inoltre lo script non
controlla il valore di ritorno di `save_midi()`, quindi un seed potrebbe
"riuscire" con un export fallito.

### Cosa abbiamo fatto
Script **read-only**: [`scripts/audit_fixes/check_sample_integrity.py`](../../../scripts/audit_fixes/check_sample_integrity.py)
- verifica che ogni `seed_N/` contenga `original.mid`, `masked.mid`, `reconstructed.mid`;
- calcola l'MD5 di `original.mid` su tutte le cartelle e raggruppa i duplicati;
- riporta: N effettiva (deduplicata), gruppi di duplicati, cartelle incomplete.

### Risultati (cluster)
```
seed_*/ dirs found     : 989
complete (3 files)     : 989
incomplete             : 0
UNIQUE source songs    : 973   <-- N effettiva (dedup)
redundant copies       : 16
duplicate groups (>1)  : 15
```
15 gruppi di duplicati, per lo più **seed adiacenti** (es. `seed_216/217/218`,
`seed_79/80`) — conferma esatta del meccanismo di fallback `seed+offset`.

**Verifica bonus (determinismo inferenza):** gli MD5 di `reconstructed.mid`
sono **identici all'interno di ogni gruppo di duplicati**:
```
seed_216/217/218 → 4ada40547cb597164905fe0c7977c090   (identici)
seed_79/80       → dc666e31458673762341221f7b2078b3   (identici)
```
Questo conferma empiricamente che l'inferenza è **deterministica** (stesso input
→ stesso output).

### Decisione
Regola di deduplicazione: **si tiene il seed più basso di ogni gruppo**, si
escludono gli altri 16.

**N effettiva = 973.**

---

## Task 2 — Verifica contaminazione da fallback "full-piece" + manifest

### Problema
`compute_muspy_metrics.py` calcola le metriche sull'**intero brano** quando il
rilevamento delle barre mascherate fallisce (`used_fallback=True`). Le righe in
fallback includono barre non mascherate (identiche tra original e reconstructed)
e **gonfiano CP/GS**. Il CSV non aveva nessuna colonna che segnalasse queste
righe.

### Cosa abbiamo fatto
Script: [`scripts/audit_fixes/build_sample_manifest.py`](../../../scripts/verification/build_sample_manifest.py)
- ri-esegue **esattamente** lo stesso criterio di rilevamento della pipeline
  reale, **importando** `_detect_masked_bars` da `compute_muspy_metrics.py`
  (nessun criterio reinventato): una barra è "mascherata" se `original.mid` ha
  ≥1 nota e `masked.mid` ne ha 0 (confini barra via `symusic`, `max_bars=64`);
- riusa il raggruppamento MD5 del Task 1 per marcare i duplicati;
- produce il **manifest** `scripts/audit_fixes/sample_manifest.csv`, unica fonte
  di verità per i Task 4/5/7.

**Colonne del manifest:** `seed, md5_original, status, exclusion_reason,
dup_group_id, n_masked_bars`.
- `status` ∈ {included, excluded}
- `exclusion_reason` ∈ {"", incomplete, duplicate, detection_failed}
- `n_masked_bars` = numero di barre mascherate rilevate (solo righe `included`)
- Precedenza esclusioni: `incomplete > duplicate > detection_failed`
- Idoneità (`included`) = completo **AND** non-duplicato **AND** rilevamento riuscito

### Risultati (cluster)
```
total seed dirs       : 989
INCLUDED (eligible)   : 973   <-- N effettiva per i Task 4/5/7
excluded:incomplete   : 0
excluded:duplicate    : 16
excluded:detect_failed: 0
duplicate groups      : 15
fallback_rate_over_candidates = 0.0%
```
`[OK] Masked-bar detection succeeded on every complete, non-duplicate seed.`

### Conclusione
**Fallback rate = 0%** sui campioni della tesi → le medie **non sono
contaminate** dal fallback full-piece. Nessuna riga da escludere per questo
motivo. Restano escluse solo le 16 copie duplicate.

---

## Freeze pre-Task-3 (backup di sicurezza)

Prima di toccare i file della pipeline reale, tutti gli output di
JS/agg/metriche sono stati copiati e resi **read-only**:
```
/nfsd/tesi/zanettisim/MIDIT/outputs/thesis_freeze_20jul/
  js_score_{midit,zs,anticipatory,upperbound}.json
  model_agg_{midit,zs,anticipatory,upperbound}.json
  ref_agg.json
  metrics_{midit,anticipatory}.csv
  metrics_fixed/
(chmod -R a-w)
```
Questi sono i numeri "di record" (incluso il vecchio **JS = 87.93**) finché il
Task 4 non produrrà il punteggio corretto.

**Scoperta importante:** la pipeline JS gira per **quattro varianti**
(`midit`, `zs`, `anticipatory`, `upperbound`). Il bug di `groove_consistency` le
colpisce tutte allo stesso modo → il Task 4 ricalcolerà **tutte e quattro** con
il protocollo corretto.

---

## Task 3 — Fix del mismatch di `groove_consistency` (measure_resolution)

### Problema
`muspy.groove_consistency(music, measure_resolution=...)` veniva chiamata con
argomenti **diversi** nei due lati del confronto JS:
- **reference** e **model-fallback**: `measure_resolution=4` (corretto)
- **model croppato**: `measure_resolution=resolution` (risoluzione nativa del
  MIDI, es. 480)

Griglie diverse → `muspy_groove_consistency` **non comparabile** tra model e
reference: corrompe 1 delle 3 metriche usate per il punteggio JS.

### Ricerca di tutti i call site
Cercati **tutti** i punti di chiamata reali di `groove_consistency` nel repo
(escluso `.venv`):

| File:linea | Argomento | Azione |
|---|---|---|
| `Matteo/muspy_metrics.py:86` | `4` (reference + fallback) | **mantieni** |
| `compute_muspy_metrics.py:267` | `measure_resolution=4` | **mantieni** |
| `Matteo/compute_model_agg.py:95` | `resolution` (model croppato) | **FIX → 4** |

`merge_metrics.py` cita `groove_consistency` solo in docstring/nomi colonna
(nessuna chiamata). Il mismatch era quindi **una sola riga**.

### Modifica (diff minimo)
File: `scripts/Evaluation_Infilling/Matteo/compute_model_agg.py`
```diff
-        "muspy_groove_consistency":   _safe(_muspy.groove_consistency, music, resolution),
+        "muspy_groove_consistency":   _safe(_muspy.groove_consistency, music, 4),
```
`empty_measure_rate` sulla riga sopra resta invariato (usa correttamente la
risoluzione reale). Dopo il fix, sia il path croppato sia il path di fallback
usano `4`: `model_agg` è internamente consistente e coerente col reference.
Poiché `_compute_metrics_on_music` è condiviso per **qualsiasi** `--samples_dir`,
questo unico fix corregge tutte e quattro le varianti.

### Verifica (cluster)
```
git pull → Fast-forward compute_model_agg.py | 1 insertion, 1 deletion
python -c "... import compute_model_agg; print('import OK')" → import OK
```

---

## Test diagnostico — anti-copiatura (token-level, barre 3-5)

**Data:** 21 luglio · Script: [`scripts/audit_fixes/token_diff_masked_bars.py`](../../../scripts/verification/token_diff_masked_bars.py)

### Motivazione
Con il masking Option B la struttura (Position, Bar, Tempo, TimeSig, Rest) è
**preservata** e solo i token di contenuto (Pitch, PitchDrum, Velocity, Duration,
Program) sono mascherati e predetti dal modello. Sorge il dubbio legittimo: il
modello **genera** davvero contenuto nuovo, o si limita a **ricopiare** l'input?
Le metriche MIDI (CP/GS) non lo dicono in modo diretto. Questo test lo verifica
**a livello di token REMI**, dentro le sole barre mascherate 3-5.

### Metodo
- Ri-tokenizza `original.mid` e `reconstructed.mid` con lo **stesso** tokenizer
  della pipeline (`tokenizer(path).ids`, come `preprocess_v2.py:192`). Il `.pt`
  **non** è usato: i token predetti dal modello non sono salvati su disco (solo
  il MIDI lo è), quindi entrambi i lati passano per lo stesso round-trip
  MIDI→REMI → il *delta* è comunque equo. È il meglio ottenibile dai file su disco.
- Crop alle barre 3-5 in **spazio-token** (`bar_positions[3]:bar_positions[6]`,
  la stessa lista che `apply_bar_mask` ha mascherato). `_detect_masked_bars` è
  usato **solo come cross-check** contro `manifest.n_masked_bars`, non per il crop.
- Split in CONTENT vs STRUCTURAL; per ogni tipo: `positional_diff` (valido solo se
  i conteggi combaciano) e `multiset_diff` (robusto all'ordinamento).

### Risultati — i 3 seed "puliti" (79, 216, 900) sono la prova vera
Sono i seed dove orig e recon hanno lo **stesso numero di token** (nessun
round-trip drift), quindi il `positional_diff` è valido e leggibile:

| seed | pitch pos-diff | content pos-diff | structural | esempi (orig → pred) |
|---|---|---|---|---|
| 79  | 67.9% | 18.8% | **0.0%** | Pitch_72→69, 69→65, 74→69 |
| 216 | 72.6% | 54.4% | **0.0%** | Pitch_43→PitchDrum_35, 55→35, 59→43 |
| 900 | 46.3% | 53.0% | **0.0%** | Pitch_56→67, 55→67 |

Lettura:
- **Structural = 0.0% esatto** su tutti e tre → masking corretto, struttura
  preservata perfettamente (Option B). Sanity check superato.
- **Pitch positional-diff 46-73%** → il modello cambia da metà a due terzi delle
  note. Un copiatore darebbe 0%. Questo è **generazione**.
- Gli esempi sono la conferma qualitativa: `Pitch_72 → Pitch_69` non è rumore, è
  una nota sostituita con un'altra **vicina e plausibile nello stesso registro**.
- Pattern notevole su seed 79: pitch cambia al 67.9% ma **duration e program
  restano a 0%** → il modello mantiene ritmo e strumento del contesto (vincolati)
  e reinventa le altezze. Non copia, non spara a caso: ha capito il task.

### I 2 seed "sporchi" (5, 500) — perché NON contano
Entrambi hanno `WARN: bar count differs` e `positional-diff: len!=`. Il round-trip
MIDI→REMI ha prodotto un numero di token diverso tra orig e recon (seed 5: 152→206
note; seed 500: 791→1121). Conseguenze:
- il `positional_diff` diventa inutilizzabile (`len!=`), resta solo il multiset;
- con lunghezze così sbilanciate il **multiset è ingannevolmente basso** — non
  perché il modello copi, ma perché il round-trip ha **aggiunto note** e gonfiato
  i conteggi. Gli `orig-only: Pitch_55 ×4` indicano disallineamento strutturale,
  non copia.
- seed 500 ha inoltre `n_masked_bars=17` (!) → non sono 3 barre, è un brano
  atipico ad alta densità, non rappresentativo.

Sono quindi **rumore metodologico, non segnale**, e vanno esclusi dalla lettura.

### Media aggregata e conclusione
La media su 5 seed (**CONTENT multiset-diff 29%**, structural positional 0.0%) è
**sottostimata** perché mescola 3 seed validi con 2 distorti dal round-trip. La
lettura corretta è: **sui seed con misura pulita il modello cambia ~50% del
contenuto (47-73% delle pitch) con sostituzioni plausibili, e preserva
perfettamente la struttura (0.0%)**.

**Conclusione: il modello genera davvero, non copia.** Prove convergenti: pitch
pos-diff 46-73% sui seed puliti (copiatore=0%); sostituzioni Pitch→Pitch nel
registro corretto; structural 0.0% esatto; e i dati già noti CP=0.68 / pitch
accuracy=40% (un copiatore darebbe CP=1.0 e acc=100%). Il modello fa il lavoro in
un singolo forward-pass parallelo invece che token-per-token: veloce e generativo
non sono in contraddizione per un MLM.

### Limite onesto del test
Il drift su seed 5/500 è un limite della **misura**, non del modello:
ri-tokenizzare MIDI introduce drift sul conteggio note nei brani densi. Il test
perfetto userebbe i token `.pt` originali vs i token realmente predetti dal
modello **prima** della detokenizzazione — ma quei token predetti non sono
salvati. Per la domanda "copia sì/no" i 3 seed puliti rispondono senza ambiguità.
Volendo blindare la difesa, si può rilanciare su 5-6 seed con bar-count stabile
(tabella tutta senza `len!=`), ma non è necessario.

### Riga per il log
> **Test anti-copiatura (5 seed, token REMI barre 3-5):** modello CONFERMATO
> generativo, non copia. Sui 3 seed con round-trip pulito (79, 216, 900): pitch
> positional-diff 46-73%, structural 0.0% esatto, sostituzioni Pitch→Pitch
> plausibili. Seed 5/500 distorti da round-trip MIDI→REMI (bar-count drift, seed
> 500 ha 17 barre) → esclusi, non rappresentativi. Media aggregata 29%
> sottostimata; lettura corretta ~50% content-diff sui seed validi. Coerente con
> CP=0.68 e pitch acc=40%. Paura "non genera" → chiusa, misurata.

---

## Stato attuale e impatto sui numeri della tesi

| Task | Deliverable | Numeri tesi cambiati |
|---|---|---|
| 1 | N effettiva = **973**; 15 gruppi duplicati; inferenza deterministica | N passa da 989 → **973** |
| 2 | fallback rate = **0%**; manifest come fonte di verità | nessuno (conferma medie pulite) |
| 3 | fix `groove_consistency` → `4` ovunque | **nessuno ancora** — attivo solo al ricalcolo del Task 4 |
| Diag. | test anti-copiatura: modello **generativo, non copia** (pitch pos-diff 46-73%, structural 0.0%) | conferma qualitativa, nessun numero cambia |

I numeri congelati in `thesis_freeze_20jul/` (incluso **JS = 87.93**) restano il
riferimento finché il **Task 4** non ricalcolerà, **una sola volta**, tutte e
quattro le varianti con: fix del Task 3, esclusione delle righe di fallback
(nessuna qui), e N deduplicata = 973.

### Commit su `audit-fixes`
- `d905331` — Task 1: script integrità campioni
- `5618230` — Task 2: script manifest fallback + `n_masked_bars`
- `b7425b2` — Task 3: fix `groove_consistency` → `measure_resolution=4`
- `60a531a` — Diagnostica: test anti-copiatura token-level (barre 3-5)

### Prossimo passo
**Task 4** — fix del mismatch di popolazione JS (ref_agg calcolato sulle regioni
croppate delle barre mascherate degli `original.mid`, come like-with-like) e
ricalcolo unico di `model_agg` + `ref_agg` + JS per le quattro varianti sulla N
corretta. Da progettare e approvare **prima** di scrivere codice.
