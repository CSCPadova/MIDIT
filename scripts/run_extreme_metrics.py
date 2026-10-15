"""
scripts/run_extreme_metrics.py

Run compute_muspy_metrics.py for all 6 extreme infilling configurations.

Usage:
    python scripts/run_extreme_metrics.py --base_dir <extreme_outputs> --out_dir <metrics_dir>

    Thesis run (local paths at the time):
        --base_dir C:/Users/simoz/OneDrive/Desktop/ResultsInfilling/Extreme/extreme
        --out_dir  C:/Users/simoz/OneDrive/Desktop/ResultsInfilling/Extreme/metrics
"""

import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS   = ['first1', 'first2', 'first3', 'last1', 'last2', 'last3']
SCRIPT    = Path('scripts/Evaluation_Infilling/compute_muspy_metrics.py')

parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
parser.add_argument('--base_dir', required=True,
                    help='Directory with one sub-directory per extreme config (first1 ... last3).')
parser.add_argument('--out_dir', required=True, help='Directory for the per-config CSVs.')
args = parser.parse_args()

BASE      = Path(args.base_dir)
OUT_DIR   = Path(args.out_dir)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input base : {BASE}")
print(f"Output dir : {OUT_DIR}")
print(f"Configs    : {CONFIGS}")
print()

failed = []
for cfg in CONFIGS:
    results_dir = BASE / cfg
    output_csv  = OUT_DIR / f'extreme_{cfg}.csv'

    print(f"{'='*60}")
    print(f"Config: {cfg}")
    print(f"  results_dir : {results_dir}")
    print(f"  output_csv  : {output_csv}")
    print()

    if not results_dir.exists():
        print(f"  ERROR: {results_dir} does not exist — skipping.")
        failed.append(cfg)
        continue

    cmd = [
        sys.executable,
        str(SCRIPT),
        '--results_dir', str(results_dir),
        '--output_csv',  str(output_csv),
    ]

    result = subprocess.run(cmd, text=True, capture_output=False)

    if result.returncode != 0:
        print(f"\n  ERROR: {cfg} exited with code {result.returncode}")
        failed.append(cfg)

    print()

print('='*60)
if failed:
    print(f"Failed configs: {failed}")
else:
    print(f"All {len(CONFIGS)} configs completed.")
print(f"CSVs saved to: {OUT_DIR}")
