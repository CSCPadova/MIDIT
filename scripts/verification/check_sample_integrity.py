"""
scripts/verification/check_sample_integrity.py

THROWAWAY audit script (Task 1). Read-only: never writes to samples_1000.

Checks the integrity of generated infilling samples:
  1. Each seed_N/ dir has original.mid, masked.mid, reconstructed.mid.
  2. MD5 of original.mid across all dirs -> duplicate groups (different seeds
     that landed on the same source song via the modulo + seed+offset fallback).
  3. Reports: effective N (complete dirs), duplicate groups, incomplete dirs.

Usage (on cluster):
    python scripts/verification/check_sample_integrity.py \
        --samples_dir /nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

REQUIRED = ("original.mid", "masked.mid", "reconstructed.mid")


def _seed_num(p: Path) -> int:
    m = re.search(r"seed_(\d+)", p.name)
    return int(m.group(1)) if m else -1


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1 — sample integrity check (read-only).")
    ap.add_argument("--samples_dir", required=True, help="Dir containing seed_N/ subdirs.")
    args = ap.parse_args()

    root = Path(args.samples_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"ERROR: {root} does not exist.")

    seed_dirs = sorted(
        [d for d in root.glob("seed_*") if d.is_dir()],
        key=_seed_num,
    )
    n_found = len(seed_dirs)
    if n_found == 0:
        raise SystemExit(f"ERROR: no seed_*/ dirs under {root}.")

    incomplete: list[tuple[str, list[str]]] = []
    complete: list[Path] = []

    for d in seed_dirs:
        missing = [f for f in REQUIRED if not (d / f).exists()]
        if missing:
            incomplete.append((d.name, missing))
        else:
            complete.append(d)

    # MD5 of original.mid over COMPLETE dirs only
    md5_to_seeds: dict[str, list[str]] = defaultdict(list)
    unreadable: list[str] = []
    for d in complete:
        try:
            digest = _md5(d / "original.mid")
        except Exception as e:  # noqa: BLE001
            unreadable.append(f"{d.name} ({e})")
            continue
        md5_to_seeds[digest].append(d.name)

    dup_groups = {k: v for k, v in md5_to_seeds.items() if len(v) > 1}
    n_unique_originals = len(md5_to_seeds)
    n_hashed = sum(len(v) for v in md5_to_seeds.values())
    n_redundant = n_hashed - n_unique_originals  # extra copies beyond one-per-song

    print("\n" + "=" * 70)
    print("  TASK 1 — SAMPLE INTEGRITY REPORT")
    print("=" * 70)
    print(f"  samples_dir            : {root}")
    print(f"  seed_*/ dirs found     : {n_found}")
    print(f"  complete (3 files)     : {len(complete)}")
    print(f"  incomplete             : {len(incomplete)}")
    print(f"  unreadable original.mid: {len(unreadable)}")
    print("-" * 70)
    print(f"  originals hashed       : {n_hashed}")
    print(f"  UNIQUE source songs    : {n_unique_originals}   <-- effective N (dedup)")
    print(f"  redundant copies       : {n_redundant}")
    print(f"  duplicate groups (>1)  : {len(dup_groups)}")
    print("=" * 70)

    if incomplete:
        print("\n[INCOMPLETE DIRS]")
        for name, missing in incomplete:
            print(f"  {name}: missing {', '.join(missing)}")

    if unreadable:
        print("\n[UNREADABLE original.mid]")
        for u in unreadable:
            print(f"  {u}")

    if dup_groups:
        print("\n[DUPLICATE GROUPS — same original.mid MD5]")
        # sort groups by size desc, then by first seed
        ordered = sorted(
            dup_groups.items(),
            key=lambda kv: (-len(kv[1]), _seed_num(Path(kv[1][0]))),
        )
        for digest, seeds in ordered:
            seeds_sorted = sorted(seeds, key=lambda s: _seed_num(Path(s)))
            print(f"  md5={digest[:12]}  x{len(seeds):>3}  {', '.join(seeds_sorted)}")
    else:
        print("\n[OK] No duplicate original.mid found — every seed is a distinct song.")

    print("\n[SUMMARY LINE]")
    print(
        f"  N_found={n_found}  N_complete={len(complete)}  "
        f"N_unique={n_unique_originals}  dup_groups={len(dup_groups)}  "
        f"incomplete={len(incomplete)}"
    )


if __name__ == "__main__":
    main()
