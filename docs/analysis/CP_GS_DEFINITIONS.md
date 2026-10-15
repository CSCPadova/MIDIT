# CP and GS — exact definitions (read-only findings)

Both metrics are defined **only** in
`scripts/evaluation/compute_infilling_metrics.py`.
`compute_muspy_metrics.py` and `scripts/evaluation/decoding/decoding_ablation.py` **import**
`compute_CP` / `compute_GS` from there — there is no inline reimplementation.
All line numbers below are in `compute_infilling_metrics.py`. **FACT** = read in
code.

Both metrics are computed **per masked bar** (over the `bar_range` tick tuples)
and then **averaged** across bars (`compute_CP:214-221`, `compute_GS:243-252`).

---

## 1. CP (Content Preservation)

**a. Cosine of two chroma vectors? — YES.** Per bar, `compute_CP` builds
`chroma_o` and `chroma_i` and appends `_cosine(chroma_o, chroma_i)`
(`:217-219`), then returns the mean over bars (`:220`). `_cosine` is
`dot(a,b) / (‖a‖·‖b‖)` (`:138-143`).

**b. 12-dimensional, one bin per pitch class? — YES.**
`chroma = np.zeros(12)`, indexed by `note.pitch % 12` (`_chroma:129-131`).

**c. Weighting — DURATION-weighted, then L1-normalised (NOT note-count, NOT
binary).** `chroma[note.pitch % 12] += note.duration` (`:131`) — each pitch class
accumulates the **sum of note durations**, not a count and not a binary presence.
Then `chroma /= chroma.sum()` (L1 normalise, `:132-134`). So it is a
**duration-weighted bag-of-pitch-classes**. (Note: your options "note count" vs
"binary" are both wrong — it's duration.)

**d. Drums — INCLUDED, not excluded, not separate.** `_notes_in_range` iterates
`for track in score.tracks: for note in track.notes` with **no** `is_drum` /
program-128 filter (`:118-121`). Drum notes therefore contribute
`note.pitch % 12 += note.duration` to the same 12-bin chroma as pitched notes.
There is no separate handling.

**e. L2-normalised before cosine? — YES (inside `_cosine`).** `_cosine` divides by
the L2 norms `na, nb` (`:139-143`). The chroma is *also* L1-normalised in
`_chroma` (`:134`), but since cosine is scale-invariant that L1 step does not
change the result; the operative normalisation is the **L2** in `_cosine`.
**Edge case:** if either bar has no notes, its chroma is all-zeros → `_cosine`
returns **0.0** (`:141-142`); i.e. a both-empty or one-empty bar scores CP = 0.

**CP in one line:** `CP = mean_over_bars( cosine( duration-weighted 12-chroma of
original bar, of reconstructed bar ) )`, drums included, empty→0.

---

## 2. GS (Groove Similarity)

**a. Jaccard `|A∩B| / |A∪B|`? — YES.** `_jaccard` returns
`logical_and(a,b).sum() / logical_or(a,b).sum()` (`:171-174`). `compute_GS`
averages the per-bar Jaccard (`:248-252`).

**b. `1 − Hamming`? — NO.** It is Jaccard, not a normalised Hamming distance.

**c. Something else? — No.** Binary onset grids compared by Jaccard, averaged per
bar. **Edge case:** if the union is 0 (both bars have no onsets), `_jaccard`
returns **1.0** (`:172-173`) — a both-empty bar is treated as identical groove
(this differs from CP, which returns 0 for empty bars).

**d. Q (groove-vector dimensionality) — VARIABLE, = the number of 16th-note cells
in the bar; = 16 for a 4/4 bar regardless of tpq.**
`_onset_grid` uses `n_cells = max(1, round(bar_len / grid_ticks))` where
`bar_len = t_end - t_start` (`:153-154`), and `compute_GS` sets
`grid_ticks = max(1, round(tpq * 4 / grid_res))` with `grid_res = 16` (`:243`,
`:229`). So `grid_ticks = tpq/4` and for a 4/4 bar `bar_len = tpq*4`, giving
`n_cells = 16` — **independent of tpq** (both scale with tpq). For MIDIT
(tpq = 16): `grid_ticks = 4`, a 64-tick bar → **16 cells** (16th-note grid). It is
**not** fixed at 64 or 96; it is the bar's 16th-note count (variable for non-4/4
or partial bars). If the two bars' grids differ in length, `_jaccard` truncates
both to the shorter length (`:169-170`).

**e. Drums — INCLUDED.** `_onset_grid` collects onsets via `_notes_in_range`
(all tracks, no `is_drum` filter, `:156-162` + `:118-121`), so drum onsets are
part of the groove grid. The grid is **binary onset presence** (`grid[cell] =
True`, `:162`) — pitch, velocity, duration and track identity are ignored; only
whether *some* note onsets in each 16th-note cell.

**GS in one line:** `GS = mean_over_bars( Jaccard( binary 16th-note onset grid of
original bar, of reconstructed bar ) )`, drums included, both-empty→1.0.

---

## Summary table

| | CP | GS |
|---|---|---|
| Core op | cosine of 12-d chroma | Jaccard of binary onset grid |
| Vector dim | 12 (pitch classes) | Q = 16th-note cells/bar (=16 for 4/4, variable) |
| Weighting | note **duration**, L1-norm then L2 in cosine | **binary** onset presence |
| Drums | included (no filter) | included (no filter) |
| Normalisation | L2 (in `_cosine`); L1 pre-step is inert | none (set-based) |
| Empty-bar value | 0.0 | 1.0 |
| Averaging | mean over masked bars | mean over masked bars |
| Location | `compute_infilling_metrics.py:198-221` (`_chroma:127`, `_cosine:138`) | `compute_infilling_metrics.py:224-252` (`_onset_grid:146`, `_jaccard:166`) |
