"""
scripts/verification/build_sample_manifest.py

THROWAWAY audit script (Task 2). Read-only w.r.t. samples_1000.
Writes ONLY scripts/verification/sample_manifest.csv (the single source of
truth for Tasks 4/5/7).

What it does
------------
1. Recomputes MD5(original.mid) per seed_N/ dir (same as Task 1) and groups
   duplicates. Keep-lowest-seed rule: the lowest seed in a duplicate group is
   kept; the others are excluded with reason "duplicate".
2. Re-runs the EXACT masked-bar detection used by the real pipeline, by
   importing `_detect_masked_bars` from
   scripts/evaluation/compute_muspy_metrics.py. Its criterion:
   a bar is "masked" iff original.mid has >=1 note there AND masked.mid has 0
   notes there (bar boundaries from symusic, max_bars=64). The function's
   second return value `used_fallback=True` means detection FAILED (would have
   fallen back to the full piece).
3. Emits scripts/verification/sample_manifest.csv with columns:
     seed, md5_original, status, exclusion_reason, dup_group_id, n_masked_bars
   status in {included, excluded}
   exclusion_reason in {"", incomplete, duplicate, detection_failed}
   n_masked_bars = number of detected masked bars for INCLUDED rows; empty ("")
     for excluded rows (detection not run, or fallback → no real count).
     Task 4 recomputes tick ranges via the same imported function and
     cross-checks its bar count against this column.
   Eligibility (status=included) = complete AND non-duplicate AND detection ok.

Exclusion precedence: incomplete > duplicate > detection_failed.

Usage (on cluster, in the eval env with symusic installed; NO GPU):
    python scripts/verification/build_sample_manifest.py \
        --samples_dir /nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

REQUIRED = ("original.mid", "masked.mid", "reconstructed.mid")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]                      # repo root
_EVAL_DIR = _ROOT / "scripts" / "evaluation"
_MANIFEST = _THIS.parent / "sample_manifest.csv"

# Import the REAL detection function so the criterion is identical.
sys.path.insert(0, str(_EVAL_DIR))
try:
    from compute_muspy_metrics import _detect_masked_bars  # noqa: E402
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        f"ERROR: could not import _detect_masked_bars from {_EVAL_DIR}\n"
        f"  {type(e).__name__}: {e}\n"
        "  Run this in the same environment used for metric computation "
        "(symusic + numpy must be importable)."
    )


def _seed_num(name: str) -> int:
    m = re.search(r"seed_(\d+)", name)
    return int(m.group(1)) if m else -1


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2 — build inclusion/exclusion manifest.")
    ap.add_argument("--samples_dir", required=True, help="Dir containing seed_N/ subdirs.")
    args = ap.parse_args()

    root = Path(args.samples_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"ERROR: {root} does not exist.")

    seed_dirs = sorted(
        [d for d in root.glob("seed_*") if d.is_dir()],
        key=lambda d: _seed_num(d.name),
    )
    if not seed_dirs:
        raise SystemExit(f"ERROR: no seed_*/ dirs under {root}.")

    # ── Pass 1: completeness + md5 grouping ───────────────────────────────────
    complete: dict[str, bool] = {}
    md5_of: dict[str, str] = {}
    md5_to_seeds: dict[str, list[str]] = defaultdict(list)

    for d in seed_dirs:
        name = d.name
        missing = [f for f in REQUIRED if not (d / f).exists()]
        complete[name] = not missing
        if not missing:
            digest = _md5(d / "original.mid")
            md5_of[name] = digest
            md5_to_seeds[digest].append(name)

    # dup_group_id for groups with >1 member (stable: ordered by lowest seed)
    dup_groups = {k: v for k, v in md5_to_seeds.items() if len(v) > 1}
    ordered_group_keys = sorted(
        dup_groups.keys(),
        key=lambda k: min(_seed_num(s) for s in dup_groups[k]),
    )
    md5_to_group_id: dict[str, str] = {}
    kept_seed_of_group: dict[str, str] = {}
    for i, k in enumerate(ordered_group_keys, start=1):
        gid = f"g{i:03d}"
        md5_to_group_id[k] = gid
        seeds_sorted = sorted(dup_groups[k], key=lambda s: _seed_num(s))
        kept_seed_of_group[k] = seeds_sorted[0]  # keep lowest seed

    # ── Pass 2: detection + classification ────────────────────────────────────
    rows: list[dict] = []
    n_detection_failed = 0
    failed_seeds: list[str] = []
    n_detection_error = 0

    for d in seed_dirs:
        name = d.name
        md5 = md5_of.get(name, "")
        gid = md5_to_group_id.get(md5, "") if md5 else ""

        status = "included"
        reason = ""
        n_masked = ""  # empty for excluded rows

        if not complete[name]:
            status, reason = "excluded", "incomplete"
        elif gid and kept_seed_of_group[md5] != name:
            status, reason = "excluded", "duplicate"
        else:
            # run detection (only reaches here for complete, kept/unique seeds)
            bar_range = []
            try:
                bar_range, used_fallback = _detect_masked_bars(
                    d / "original.mid", d / "masked.mid"
                )
            except Exception as e:  # noqa: BLE001
                used_fallback = True
                n_detection_error += 1
                print(f"  [{name}] detection raised: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            if used_fallback:
                status, reason = "excluded", "detection_failed"
                n_detection_failed += 1
                failed_seeds.append(name)
            else:
                n_masked = len(bar_range)  # real masked-bar count

        rows.append({
            "seed": name,
            "md5_original": md5,
            "status": status,
            "exclusion_reason": reason,
            "dup_group_id": gid,
            "n_masked_bars": n_masked,
        })

    # ── Write manifest ────────────────────────────────────────────────────────
    with open(_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seed", "md5_original", "status", "exclusion_reason",
                        "dup_group_id", "n_masked_bars"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ── Report ────────────────────────────────────────────────────────────────
    n_total = len(rows)
    n_included = sum(1 for r in rows if r["status"] == "included")
    n_incomplete = sum(1 for r in rows if r["exclusion_reason"] == "incomplete")
    n_duplicate = sum(1 for r in rows if r["exclusion_reason"] == "duplicate")

    print("\n" + "=" * 70)
    print("  TASK 2 — INCLUSION/EXCLUSION MANIFEST")
    print("=" * 70)
    print(f"  samples_dir           : {root}")
    print(f"  manifest              : {_MANIFEST}")
    print(f"  total seed dirs       : {n_total}")
    print("-" * 70)
    print(f"  INCLUDED (eligible)   : {n_included}   <-- effective N for Tasks 4/5/7")
    print(f"  excluded:incomplete   : {n_incomplete}")
    print(f"  excluded:duplicate    : {n_duplicate}")
    print(f"  excluded:detect_failed: {n_detection_failed}"
          f"   (of which raised errors: {n_detection_error})")
    print(f"  duplicate groups      : {len(dup_groups)}")
    print("=" * 70)

    if failed_seeds:
        print("\n[DETECTION-FAILED SEEDS] (would have used full-piece fallback)")
        for s in sorted(failed_seeds, key=lambda x: _seed_num(x)):
            print(f"  {s}")
    else:
        print("\n[OK] Masked-bar detection succeeded on every complete, non-duplicate seed.")

    fallback_rate = 100.0 * n_detection_failed / max(1, (n_total - n_incomplete - n_duplicate))
    print(f"\n[SUMMARY LINE]")
    print(
        f"  N_total={n_total}  N_included={n_included}  "
        f"excl_dup={n_duplicate}  excl_detect={n_detection_failed}  "
        f"excl_incomplete={n_incomplete}  "
        f"fallback_rate_over_candidates={fallback_rate:.1f}%"
    )


if __name__ == "__main__":
    main()
