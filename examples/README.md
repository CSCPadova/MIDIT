# Example MIDI files

Two short multi-track MIDI files for trying MIDIT without the dataset
(see the Quickstart in the [root README](../README.md)):

| File | Size | Bars | Notes |
|---|---|---|---|
| `0000799e8672292fe6f0fff08554ca40.mid` | 13.8 KB | 37 | used in the README Quickstart (`--start_bar 12`) |
| `00012722c199ae2a628ebb792ccc617a.mid` | 8.6 KB | 102 | |

Both pass the training quality filters (≥ 50 notes, ≥ 15 s, velocity diversity),
so `scripts/infer_from_midi.py` runs on them without `--skip_filters`.

## Source and license

The files are derived from the **Lakh MIDI Dataset** (`lmd_full`; the file names
are the dataset's MD5 identifiers), © Colin Raffel, distributed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

- Dataset page: <https://colinraffel.com/projects/lmd/>
- Colin Raffel, *Learning-Based Methods for Comparing Sequences, with Applications
  to Audio-to-MIDI Alignment and Matching*, PhD thesis, 2016.

**Changes:** the files are not the original dataset files. The drum tracks were
removed (tempo, time-signature and key-signature events kept) and the result was
re-saved with `symusic` by an early tokenizer-validation script
(`scripts/validate_tokenization.py`, commit `91ffcbb`). They are copies of
`thesis_materials/04_figures_data/validation_output/original_<id>.mid`.
