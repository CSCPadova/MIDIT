"""
scripts/audit_fixes/token_diff_masked_bars.py

THROWAWAY diagnostic (no GPU, read-only). Proves the model GENERATES content in
the masked bars rather than copying the input, by comparing REMI tokens (not raw
MIDI) inside bars 3-5.

Method (as agreed)
------------------
- Original tokens come from RE-TOKENIZING original.mid; reconstructed tokens from
  RE-TOKENIZING reconstructed.mid, both via `tokenizer(path).ids` — identical to
  the pipeline's tokenization (preprocess_v2.py:192,200-205). The .pt is NOT used:
  both sides go through the same MIDI->REMI round-trip so the delta is fair.
- Masked region = pipeline's TOKEN-space bars: bar_positions built by indexing
  "Bar" tokens (preprocess_v2.py:212-219) — the exact list apply_bar_mask masked
  (infer.py:202-207). Bars 3-5 = bar_positions[3]:bar_positions[6].
- _detect_masked_bars is imported and run only as a SANITY CROSS-CHECK: its
  masked-bar count is compared to manifest.n_masked_bars (drift guard); it does
  NOT define the cropped region.
- Tokens in the crop are split into CONTENT {pitch(+pitchdrum),velocity,duration,
  program} (model-predicted) and STRUCTURAL {position,bar,tempo,timesig,rest}
  (preserved). Structural diff should be ~0%.

Usage (cluster, main .venv, NO GPU):
    python scripts/audit_fixes/token_diff_masked_bars.py \
        --samples_dir /nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000 \
        --manifest    scripts/audit_fixes/sample_manifest.csv \
        --seeds 5 79 216 500 900
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_EVAL = _ROOT / "scripts" / "Evaluation_Infilling"
sys.path.insert(0, str(_ROOT))     # midit.infer
sys.path.insert(0, str(_EVAL))     # compute_muspy_metrics

from midit.infer import build_tokenizer, build_vocab_lookups   # noqa: E402
from compute_muspy_metrics import _detect_masked_bars           # noqa: E402

CONTENT_HEADS = ("pitch", "velocity", "duration", "program")
STRUCT_HEADS  = ("position", "bar", "tempo", "timesig", "rest")


# ── tokenization + bar boundaries (mirror preprocess_v2.py) ───────────────────

def tokenize_ids(tokenizer, midi_path: Path) -> list[int]:
    toks = tokenizer(str(midi_path))
    if isinstance(toks, list):
        ids: list[int] = []
        for s in toks:
            ids.extend(s.ids)
        return ids
    return list(toks.ids)


def bar_positions_from_ids(ids: list[int], id_to_token: dict) -> list[int]:
    """Token-index bar boundaries — identical logic to preprocess_v2.py:212-219."""
    bp = [idx for idx, tid in enumerate(ids)
          if (id_to_token.get(tid) and "Bar" in id_to_token.get(tid))]
    if not bp or bp[0] != 0:
        bp.insert(0, 0)
    bp.append(len(ids))
    return bp


def crop_to_bars(ids: list[int], bp: list[int], start_bar: int, end_bar: int):
    """Return ids for bars [start_bar, end_bar] inclusive, or None if out of range."""
    num_bars = len(bp) - 1
    if end_bar + 1 > num_bars:
        return None
    return ids[bp[start_bar]: bp[end_bar + 1]]


# ── token-type split + diffs ──────────────────────────────────────────────────

def split_by_type(crop_ids: list[int], type_for_id: dict):
    content: dict[str, list[int]] = defaultdict(list)
    struct:  dict[str, list[int]] = defaultdict(list)
    other:   list[int] = []
    for tid in crop_ids:
        head = type_for_id.get(tid)
        if head in CONTENT_HEADS:
            content[head].append(tid)
        elif head in STRUCT_HEADS:
            struct[head].append(tid)
        else:
            other.append(tid)   # special tokens (PAD/BOS/EOS/MASK) or unknown
    return content, struct, other


def positional_diff(a: list[int], b: list[int]):
    """% of positions that differ; None if lengths mismatch (not alignable)."""
    if len(a) != len(b):
        return None
    if not a:
        return 0.0
    return 100.0 * sum(1 for x, y in zip(a, b) if x != y) / len(a)


def multiset_diff(a: list[int], b: list[int]) -> float:
    """% of tokens not value-matched across the two bags (order-robust)."""
    ca, cb = Counter(a), Counter(b)
    matched = sum(min(ca[k], cb[k]) for k in (set(ca) | set(cb)))
    denom = max(len(a), len(b), 1)
    return 100.0 * (1 - matched / denom)


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"ERROR: manifest not found: {path}")
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["seed"]] = r
    return rows


# ── per-seed analysis ─────────────────────────────────────────────────────────

def analyze_seed(seed_int, samples_dir, manifest, tokenizer, lookups, mask_bars):
    name = f"seed_{seed_int}"
    d = samples_dir / name
    id_to_token = lookups["id_to_token"]
    type_for_id = lookups["type_for_id"]

    print("\n" + "=" * 74)
    print(f"  {name}")
    print("=" * 74)

    for fn in ("original.mid", "reconstructed.mid", "masked.mid"):
        if not (d / fn).exists():
            print(f"  SKIP: missing {fn}")
            return None

    mrow = manifest.get(name)
    if mrow is None:
        print(f"  WARN: {name} not in manifest")
    elif mrow["status"] != "included":
        print(f"  WARN: {name} status={mrow['status']} (reason={mrow['exclusion_reason']})")

    # sanity cross-check: tick-space detector vs manifest n_masked_bars
    try:
        bar_range, used_fallback = _detect_masked_bars(d / "original.mid", d / "masked.mid")
        n_detected = 0 if used_fallback else len(bar_range)
    except Exception as e:  # noqa: BLE001
        n_detected, used_fallback = -1, True
        print(f"  WARN: _detect_masked_bars raised: {type(e).__name__}: {e}")
    man_nmb = mrow.get("n_masked_bars", "") if mrow else ""
    flag = "OK" if str(n_detected) == str(man_nmb) else "MISMATCH"
    print(f"  cross-check: _detect_masked_bars={n_detected}  manifest.n_masked_bars={man_nmb}  [{flag}]")

    # tokenize both
    ids_o = tokenize_ids(tokenizer, d / "original.mid")
    ids_r = tokenize_ids(tokenizer, d / "reconstructed.mid")
    bp_o = bar_positions_from_ids(ids_o, id_to_token)
    bp_r = bar_positions_from_ids(ids_r, id_to_token)
    print(f"  tokens: orig={len(ids_o)} ({len(bp_o)-1} bars)  "
          f"recon={len(ids_r)} ({len(bp_r)-1} bars)")
    if (len(bp_o) - 1) != (len(bp_r) - 1):
        print("  WARN: bar count differs between orig and recon (round-trip drift)")

    start_bar, end_bar = min(mask_bars), max(mask_bars)
    crop_o = crop_to_bars(ids_o, bp_o, start_bar, end_bar)
    crop_r = crop_to_bars(ids_r, bp_r, start_bar, end_bar)
    if crop_o is None or crop_r is None:
        print(f"  SKIP: bars {start_bar}-{end_bar} out of range")
        return None

    cont_o, struct_o, _ = split_by_type(crop_o, type_for_id)
    cont_r, struct_r, _ = split_by_type(crop_r, type_for_id)

    # per-type table
    print(f"\n  {'type':<10} {'n_orig':>6} {'n_recon':>7} {'pos_diff%':>9} {'ms_diff%':>9}")
    print("  " + "-" * 46)

    def row(kind, o_map, r_map):
        results = {}
        for t in kind:
            a, b = o_map.get(t, []), r_map.get(t, [])
            pd = positional_diff(a, b)
            md = multiset_diff(a, b)
            pd_s = f"{pd:9.1f}" if pd is not None else f"{'len!=':>9}"
            print(f"  {t:<10} {len(a):>6} {len(b):>7} {pd_s} {md:9.1f}")
            results[t] = (len(a), len(b), pd, md)
        return results

    print("  [CONTENT — model-predicted]")
    cres = row(CONTENT_HEADS, cont_o, cont_r)
    print("  [STRUCTURAL — preserved, expect ~0%]")
    sres = row(STRUCT_HEADS, struct_o, struct_r)

    # overall content / structural (concatenated, fixed type order)
    all_c_o = [x for t in CONTENT_HEADS for x in cont_o.get(t, [])]
    all_c_r = [x for t in CONTENT_HEADS for x in cont_r.get(t, [])]
    all_s_o = [x for t in STRUCT_HEADS for x in struct_o.get(t, [])]
    all_s_r = [x for t in STRUCT_HEADS for x in struct_r.get(t, [])]
    c_pos = positional_diff(all_c_o, all_c_r)
    c_ms  = multiset_diff(all_c_o, all_c_r)
    s_pos = positional_diff(all_s_o, all_s_r)
    s_ms  = multiset_diff(all_s_o, all_s_r)
    print("  " + "-" * 46)
    print(f"  {'CONTENT':<10} {len(all_c_o):>6} {len(all_c_r):>7} "
          f"{(f'{c_pos:9.1f}' if c_pos is not None else '    len!=')} {c_ms:9.1f}")
    print(f"  {'STRUCT':<10} {len(all_s_o):>6} {len(all_s_r):>7} "
          f"{(f'{s_pos:9.1f}' if s_pos is not None else '    len!=')} {s_ms:9.1f}")

    # concrete before/after examples for pitch + duration (aligned pairs)
    for t in ("pitch", "duration"):
        a, b = cont_o.get(t, []), cont_r.get(t, [])
        print(f"\n  [{t} examples: orig -> pred]  (first differing pairs)")
        shown = 0
        if len(a) == len(b):
            for x, y in zip(a, b):
                if x != y:
                    print(f"    {id_to_token.get(x,'?'):<18} -> {id_to_token.get(y,'?')}")
                    shown += 1
                    if shown >= 4:
                        break
            if shown == 0:
                print("    (none differ — IDENTICAL, possible copy)")
        else:
            # lengths differ: show a few from each bag that don't match
            diff = list((Counter(a) - Counter(b)).elements())[:4]
            for x in diff:
                print(f"    orig-only: {id_to_token.get(x,'?')}")
            if not diff:
                print("    (multisets equal despite length note)")

    return {
        "content_pos": c_pos, "content_ms": c_ms,
        "struct_pos": s_pos, "struct_ms": s_ms,
        "per_type_ms": {t: cres[t][3] for t in CONTENT_HEADS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Task — token-level content-vs-copy diagnostic.")
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--manifest", default=str(_THIS.parent / "sample_manifest.csv"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 79, 216, 500, 900])
    ap.add_argument("--mask_bars", type=int, nargs="+", default=[3, 4, 5])
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    manifest = load_manifest(Path(args.manifest).expanduser().resolve())
    tokenizer = build_tokenizer()
    lookups = build_vocab_lookups(tokenizer)

    print(f"[token_diff] samples_dir={samples_dir}")
    print(f"[token_diff] seeds={args.seeds}  mask_bars={args.mask_bars}")

    results = []
    for s in args.seeds:
        r = analyze_seed(s, samples_dir, manifest, tokenizer, lookups, args.mask_bars)
        if r is not None:
            results.append(r)

    if not results:
        print("\nNo analyzable seeds.")
        return

    def avg(key, sub=None):
        vals = [(r["per_type_ms"][sub] if sub else r[key]) for r in results
                if (r["per_type_ms"][sub] if sub else r[key]) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 74)
    print(f"  AVERAGED OVER {len(results)} SEEDS")
    print("=" * 74)
    print(f"  CONTENT   multiset-diff : {avg('content_ms'):6.1f}%   "
          f"positional-diff: {avg('content_pos'):6.1f}%")
    print(f"  STRUCTURAL multiset-diff: {avg('struct_ms'):6.1f}%   "
          f"positional-diff: {avg('struct_pos'):6.1f}%")
    print("  per-content-type multiset-diff:")
    for t in CONTENT_HEADS:
        print(f"    {t:<10}: {avg('per_type_ms', t):6.1f}%")

    print("\n[INTERPRETATION GUIDE]")
    print("  content ~50-60%, structural ~0%  -> healthy generation")
    print("  content ~0%                      -> model is COPYING (bug)")
    print("  structural >>0%                  -> masking/round-trip problem")


if __name__ == "__main__":
    main()
