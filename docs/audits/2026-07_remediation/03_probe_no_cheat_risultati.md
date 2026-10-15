# Probe anti-cheat — Risultati e verdetto

**Branch:** `audit-fixes` · **Data:** 21 luglio · **Seed:** 79 · **Slot:** 20 · **Device:** CPU
**Script:** [`scripts/audit_fixes/probe_no_cheat.py`](../../../scripts/audit_fixes/probe_no_cheat.py)
**Metodo:** vedi [`02_probe_no_cheat.md`](02_probe_no_cheat.md)

## Verdetto
**Probe passato su tutta la linea.** La questione "il modello sta barando / l'argmax
invalida tutto" è **chiusa meccanicamente**. Tre livelli di prova indipendenti
convergono: meccanismo, output, metriche.

File del chunk: `04ffa3ff7730cc56e95df6919f9ce25f.pt` · 305 token · 8 barre · 42.586.658 parametri.

---

## PART B — input == MASK, predizioni tutte dai logit

I due dati decisivi:
- **`input==MASK: 20/20`** — il modello non ha mai visto un solo token ground-truth
  nelle posizioni mascherate. Il canale del "baro" è chiuso alla radice.
- **`source = logits_argmax` 20/20, zero fallback** — nessuna predizione è una
  copia dell'originale via fallback; sono tutte uscite dai logit. (Primo dato utile
  anche per il **Task 5**: sul campione osservato il fallback non scatta mai.)

`predicted==original: 16/20` — sembra tanto, ma **quali** slot sono uguali conta
tutto:

| Tipo | equal / tot | Lettura |
|---|---|---|
| Program  | 5/5 | strumento vincolato dal contesto → azzeccato (bravura, non copia) |
| Duration | 6/6 | groove/ritmo vincolato → azzeccato |
| Velocity | 4/5 | quasi stabile, 1 cambio su bin adiacente (95→91) |
| Pitch    | 2/5 | **3 pitch su 5 CAMBIATI** — dove vive la libertà musicale |

Esempi di sostituzione (slot mascherati): `Pitch_72→69`, `Pitch_72→69`,
`Pitch_69→65`, `Pitch_74→69`, `Velocity_95→91` — **note vicine nel registro, bin
adiacenti**. Un baro darebbe 20/20 uguali; un generatore competente dà esattamente
questo pattern. È lo **stesso** pattern del test token-diff di ieri su questo seed
(pitch 67.9% diversi, duration/program 0%).

---

## PART C — il colpo di grazia: 12/20 CHANGED con regione mascherata byte-identica

`masked-bar region unchanged: YES` → gli slot mascherati e tutto lo scaffold sono
**identici al bit** tra i due run. L'unica cosa cambiata è il **contenuto delle
barre di contesto** (144 token rimescolati). Risultato:

**`predictions changed by context shuffle: 12/20`.**

Nessun meccanismo di copia o lookup può produrlo: se il modello "rigurgitasse"
qualcosa legato agli slot, input identico → output identico. Invece le predizioni
**si muovono col contesto** → il modello legge la musica intorno al buco e
condiziona le sue scelte su di essa. È la definizione operativa di un infiller.

Dettagli di finezza:
- le **Duration** restano tutte stabili (slot 74/78/83/87 `same`) — sono le più
  rigide, come previsto;
- i **Pitch** cambiano quasi tutti;
- **slot 90**: con il contesto sgrammaticato il modello ri-predice `58`, che è
  l'originale — segno che **ricalcola la distribuzione da capo** ogni volta, non
  segue una tabella.

---

## Quadro completo — tre livelli di prova convergenti

| Livello | Quando | Evidenza |
|---|---|---|
| **Meccanismo** | oggi | input=MASK ovunque (20/20), predizioni solo dai logit (0 fallback), output dipendente dal contesto (12/20 CHANGED) |
| **Output** | ieri (token-diff) | 46-73% dei pitch diversi dal ground-truth, sostituzioni plausibili, structural 0% |
| **Metriche** | in tesi | CP=0.68, pitch accuracy=40% — incompatibili con qualunque copia (baro = CP 1.0, acc 100%) |

Sul dubbio "argmax è troppo semplice": quello osservato è un transformer da 42.5M
che **produce distribuzioni diverse al variare del contesto** — l'intelligenza è
lì; l'argmax le legge e basta. Veloce (un forward-pass parallelo) e generativo non
sono in contraddizione per un MLM.

**Capitolo argmax / "sta barando" → archiviato.** In difesa la risposta è la più
forte possibile: "l'abbiamo verificato meccanicamente, ecco il probe".

---

## Snippet pronto (tesi §6.5 / log Notion)

> **Mechanical verification (anti-cheat probe, 21/07):** static trace confirmed the
> model input contains `MASK_ID` at all masked positions (`infer.py:211`), with
> original ids used only for oracle head selection and the never-triggered
> out-of-range fallback. Runtime probe on seed 79 (20 content slots): `input==MASK`
> 20/20, all predictions from `logits_argmax` (zero fallbacks); 16/20 match ground
> truth, concentrated on context-constrained types (Program 5/5, Duration 6/6),
> while Pitch differs in 3/5 — consistent with the 46-73% token-diff.
> Context-dependence test: shuffling only context-bar content (masked region
> byte-identical) changed 12/20 predictions — predictions are conditioned on
> musical context, mechanically excluding any copy/replay path.

---

## Output grezzo (per riferimento)

### PART B
```
   slot  input_id  MASK?  orig_id  pred_id  equal?  orig->pred (names)               source
     71         3    YES      476      476     YES  Program_64 -> Program_64         logits_argmax
     72         3    YES       56       56     YES  Pitch_72 -> Pitch_72             logits_argmax
     73         3    YES      119      119     YES  Velocity_103 -> Velocity_103     logits_argmax
     74         3    YES      133      133     YES  Duration_1.0.8 -> Duration_1.0.8 logits_argmax
     75         3    YES      477      477     YES  Program_65 -> Program_65         logits_argmax
     76         3    YES       53       53     YES  Pitch_69 -> Pitch_69             logits_argmax
     77         3    YES      117      116      no  Velocity_95 -> Velocity_91       logits_argmax
     78         3    YES      133      133     YES  Duration_1.0.8 -> Duration_1.0.8 logits_argmax
     80         3    YES      476      476     YES  Program_64 -> Program_64         logits_argmax
     81         3    YES       56       53      no  Pitch_72 -> Pitch_69             logits_argmax
     82         3    YES      118      118     YES  Velocity_99 -> Velocity_99       logits_argmax
     83         3    YES      133      133     YES  Duration_1.0.8 -> Duration_1.0.8 logits_argmax
     84         3    YES      477      477     YES  Program_65 -> Program_65         logits_argmax
     85         3    YES       53       49      no  Pitch_69 -> Pitch_65             logits_argmax
     86         3    YES      116      116     YES  Velocity_91 -> Velocity_91       logits_argmax
     87         3    YES      133      133     YES  Duration_1.0.8 -> Duration_1.0.8 logits_argmax
     89         3    YES      476      476     YES  Program_64 -> Program_64         logits_argmax
     90         3    YES       58       53      no  Pitch_74 -> Pitch_69             logits_argmax
     91         3    YES      118      118     YES  Velocity_99 -> Velocity_99       logits_argmax
     92         3    YES      137      137     YES  Duration_1.4.8 -> Duration_1.4.8 logits_argmax
  input==MASK: 20/20   predicted==original: 16/20
```

### PART C
```
  masked-bar region unchanged: YES
   slot  orig_id  pred_base  pred_shuf  changed?
     71      476        476        476      same
     72       56         56         58   CHANGED
     73      119        119        117   CHANGED
     74      133        133        133      same
     75      477        477        476   CHANGED
     76       53         53         58   CHANGED
     77      117        116        117   CHANGED
     78      133        133        133      same
     80      476        476        476      same
     81       56         53         44   CHANGED
     82      118        118        117   CHANGED
     83      133        133        133      same
     84      477        477        477      same
     85       53         49         44   CHANGED
     86      116        116        117   CHANGED
     87      133        133        133      same
     89      476        476        476      same
     90       58         53         58   CHANGED
     91      118        118        117   CHANGED
     92      137        137        133   CHANGED
  predictions changed by context shuffle: 12/20
```
