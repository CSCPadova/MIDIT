# S-I metrics — how they are computed (read-only findings)

Source: `scripts/evaluation/compute_muspy_metrics.py` (the S-I metrics
live here) and `scripts/evaluation/compute_infilling_metrics.py`
(the P-I metrics CP/GS/PCHE; imported by the muspy script). No file modified.

---

## Q1 — `_compute_si_metrics`

**Yes, it exists.**
- **File / lines:** `scripts/evaluation/compute_muspy_metrics.py:231-279`.
- **Signature:** `_compute_si_metrics(path: Path, bar_range: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]`.
- **Input type:** a **filesystem path to a MIDI file** (not a `muspy.Music` or
  `symusic.Score`). Internally it loads the file with `muspy.read_midi(str(path))`
  (line 250) and, if `muspy` is unavailable, falls back to a `pretty_midi` path
  (`_si_fallback_pretty_midi`, lines 246-247 / 163-193).
- **Returns:** a **dict** keyed by the 5 S-I metric names (`SI_METRIC_NAMES`,
  lines 78-81): `pitch_class_entropy`, `groove_consistency`, `scale_consistency`,
  `pitch_range`, `n_pitches_used`. Values are floats, or `float('nan')` on failure.
- **Cropping:** it **crops when `bar_range` is provided**. If `bar_range is not
  None` it calls `_crop_music_to_range(music, bar_range)` (lines 256-257) before
  computing; if `bar_range is None` it computes on the **whole** file (documented
  at lines 236-241). `groove_consistency` is called with `measure_resolution=4`
  (line 267); the other four via the `metric_fns` loop (lines 259-277).

**For reuse:** the cleanest entry point for a new script is to call
`_compute_si_metrics(midi_path, bar_range=<tick ranges>)` directly — it does the
load + crop + all 5 muspy calls with the project's exact settings (groove res 4).

---

## Q2 — `pitch_class_entropy`

- Called as **`muspy.pitch_class_entropy(music)`** (registered in `metric_fns`,
  `compute_muspy_metrics.py:260`), where `music` is a `muspy.Music` (already
  cropped if a `bar_range` was passed).
- It takes a **full `muspy.Music` object**; to restrict it to a region you crop
  the `Music` first (via `_crop_music_to_range`) and pass the cropped object —
  there is no per-call tick-range argument.
- (There is also a self-contained `pretty_midi` reimplementation for the
  no-muspy fallback at `compute_muspy_metrics.py:181-188`.)

---

## Q3 — `n_pitches_used` and `pitch_range`

- **`n_pitches_used`:** `muspy.n_pitches_used(music)` (`compute_muspy_metrics.py:263`).
- **`pitch_range`:** `muspy.pitch_range(music)` (`compute_muspy_metrics.py:262`).
- Both take a **`muspy.Music`** object and, like `pitch_class_entropy`, operate on
  whatever notes are in that object — so they honour cropping only if the `Music`
  was cropped beforehand. Both are **extensive** (region-size-dependent): their
  values grow with the size of the cropped region, so the crop must be consistent
  across compared systems. Fallback (`pretty_midi`) equivalents exist at lines
  189-190.

---

## Q4 — Loading a MIDI file as `muspy.Music`

- The exact call is **`muspy.read_midi(str(path))`** (`compute_muspy_metrics.py:250`,
  and again at `compute_reference_agg` / `compute_model_agg` in the `Matteo/`
  pipeline). Not `muspy.load`.
- Guarded by a top-level `try: import muspy` → `MUSPY_AVAILABLE` flag
  (`compute_muspy_metrics.py:61-67`); if the import fails the code uses the
  `pretty_midi` fallback instead.

---

## Q5 — Cropping a `muspy.Music` to a tick range

**Yes — `_crop_music_to_range(music, bar_range)` exists.**
- **File / lines:** `scripts/evaluation/compute_muspy_metrics.py:196-228`.
- **Behaviour:** returns a **new `muspy.Music`** containing only notes whose onset
  `note.time` falls inside any `(t_start, t_end)` tuple in `bar_range`, and
  **re-times** surviving notes by subtracting `bar_range[0][0]` so the cropped
  region starts at tick 0 (needed for `groove_consistency`'s measure grid, lines
  205-228). It does **not** clip durations (documented: the 5 S-I metrics read
  only `note.time`/`note.pitch`).
- **So the MIDIs were NOT pre-cropped** — the full MIDI is loaded and cropped
  in-memory per masked-bar `bar_range`. The `bar_range` itself is a list of
  `(t_start, t_end)` **tick** tuples, produced by `_detect_masked_bars`
  (`compute_muspy_metrics.py:88-125`, from original-vs-masked note presence) or by
  slicing `_bar_boundaries(score)` at the masked bar indices (from
  `compute_infilling_metrics.py`).
- **Important caveat (already established in this project):** `_detect_masked_bars`
  builds `bar_range` in **ticks from the original's grid**. That is only safe when
  the reconstruction shares the same `ticks_per_quarter` (all MIDIT REMI+ outputs
  are tpq 16). For a like-with-like S-I on a fixed region, the robust choice is to
  build the `bar_range` on **each file's own** `_bar_boundaries` at the fixed
  masked bar indices (bars 3-5), rather than reusing the original's ticks across a
  different-tpq file.

---

## Q6 — Output CSV format

- Written by `main()` via `csv.DictWriter` with fieldnames **`CSV_COLUMNS`**
  (`compute_muspy_metrics.py:71-76`, writer at lines 453-457). The columns are:

  ```
  experiment, cp, gs, pche,
  pitch_class_entropy, groove_consistency, scale_consistency,
  pitch_range, n_pitches_used
  ```

- So **all five S-I metrics are present as column names**: `pitch_class_entropy`,
  `groove_consistency`, `scale_consistency`, `pitch_range`, `n_pitches_used`
  (alongside the three P-I metrics `cp`, `gs`, `pche` and an `experiment` id).
- Per-experiment rows come from `process_experiment` (lines 284-329), which for
  each `seed_*` dir detects masked bars, computes P-I (`_compute_pi_metrics`) and
  S-I (`_compute_si_metrics(recon_path, bar_range=bar_range)`), then `main()`
  writes the CSV and prints a per-experiment table + a markdown mean±std summary.

---

## Reuse recommendation for a new post-processing script

- Import from the flat module (as the project does): add
  `scripts/evaluation` to `sys.path`, then
  `from compute_muspy_metrics import _compute_si_metrics, _crop_music_to_range, _detect_masked_bars`.
- For each reconstruction MIDI, call
  `_compute_si_metrics(recon_path, bar_range=<masked-bar tick ranges>)` and read
  the 5 keys. Build `bar_range` from each file's own `_bar_boundaries` at bars
  3-5 (safe across tpq), or from `_detect_masked_bars` if a `masked.mid` is
  available and tpq matches.
- Write the same column names as `CSV_COLUMNS` so the output drops into the
  existing aggregation/reporting.
