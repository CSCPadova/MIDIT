"""
scripts/Evaluation_Infilling/Matteo/compute_model_agg.py

Compute model aggregate muspy statistics over generated infilling samples.
Output JSON is consumed by compute_js_similarity() as model_agg.

Mirrors compute_reference_agg.py but walks seed_N/reconstructed.mid
subdirs instead of a flat MIDI directory.

Metrics are computed on the masked bar region only (bars 3-4-5, detected
from original.mid vs masked.mid in the same seed_N/ folder). Falls back
to the full reconstructed.mid when original.mid is absent or bar
detection fails.

Usage:
    python scripts/Evaluation_Infilling/Matteo/compute_model_agg.py \
        --samples_dir /nfsd/.../outputs/samples_1000 \
        --output_json outputs/model_agg.json \
        --max_files   1000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# ── Crop support ─────────────────────────────────────────────────────────────
# _detect_masked_bars and _crop_music_to_range live in the parent directory.
# Guarded by try/except: if symusic or muspy is not installed the script
# falls back to computing metrics on the full reconstructed.mid.
_EVAL_DIR = _HERE.parent   # scripts/Evaluation_Infilling/
sys.path.insert(0, str(_EVAL_DIR))

_CROP_AVAILABLE = False
try:
    import muspy as _muspy
    from compute_muspy_metrics import _detect_masked_bars, _crop_music_to_range
    from muspy_metrics import _safe, _primary_key
    _CROP_AVAILABLE = True
except Exception:
    pass

from muspy_metrics import compute_muspy_metrics

try:
    from js_similarity import aggregate
except ImportError:
    def aggregate(per_file: dict) -> dict:
        if not per_file:
            return {}
        all_metrics = list(next(iter(per_file.values())).keys())
        agg: dict[str, dict] = {}
        for metric in all_metrics:
            values = [
                v[metric]
                for v in per_file.values()
                if v.get(metric) is not None
                and isinstance(v[metric], (int, float))
                and not isinstance(v[metric], bool)
            ]
            n = len(values)
            if n == 0:
                agg[metric] = {"mean": None, "std": None, "n": 0}
            elif n == 1:
                agg[metric] = {"mean": float(values[0]), "std": None, "n": 1}
            else:
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                agg[metric] = {"mean": float(mean), "std": math.sqrt(variance), "n": n}
        return agg


def _compute_metrics_on_music(music: "_muspy.Music") -> dict[str, float | None]:
    """Compute the same 12 muspy metrics as compute_muspy_metrics() on a
    pre-loaded (and optionally cropped) muspy.Music object, skipping read_midi."""
    resolution = music.resolution
    metrics: dict[str, float | None] = {
        "muspy_pitch_range":          _safe(_muspy.pitch_range, music),
        "muspy_n_pitches_used":       _safe(_muspy.n_pitches_used, music),
        "muspy_n_pitch_classes_used": _safe(_muspy.n_pitch_classes_used, music),
        "muspy_pitch_entropy":        _safe(_muspy.pitch_entropy, music),
        "muspy_pitch_class_entropy":  _safe(_muspy.pitch_class_entropy, music),
        "muspy_scale_consistency":    _safe(_muspy.scale_consistency, music),
        "muspy_polyphony":            _safe(_muspy.polyphony, music),
        "muspy_polyphony_rate":       _safe(_muspy.polyphony_rate, music),
        "muspy_empty_beat_rate":      _safe(_muspy.empty_beat_rate, music),
        "muspy_empty_measure_rate":   _safe(_muspy.empty_measure_rate, music, resolution),
        "muspy_groove_consistency":   _safe(_muspy.groove_consistency, music, resolution),
    }
    key = _primary_key(music)
    if key is None:
        metrics["muspy_pitch_in_scale_rate"] = None
    else:
        root, mode = key
        metrics["muspy_pitch_in_scale_rate"] = _safe(
            _muspy.pitch_in_scale_rate, music, root, mode
        )
    return metrics


def _get_metrics_for_seed(
    recon_path: Path,
) -> tuple[dict[str, float | None], bool]:
    """Compute muspy metrics for one seed, cropped to the masked bar region.

    Returns (metrics_dict, used_fallback).
    Falls back to the full reconstructed.mid when:
      - crop support is unavailable (symusic/muspy not installed)
      - original.mid or masked.mid is absent from seed_N/
      - bar detection fails or reports no masked bars
      - muspy read/crop raises an exception
    """
    if not _CROP_AVAILABLE:
        return compute_muspy_metrics(recon_path), True

    seed_dir    = recon_path.parent
    orig_path   = seed_dir / "original.mid"
    masked_path = seed_dir / "masked.mid"

    if not orig_path.exists() or not masked_path.exists():
        print(
            f"  Warning [{seed_dir.name}]: original.mid/masked.mid not found"
            f" — using full file", file=sys.stderr,
        )
        return compute_muspy_metrics(recon_path), True

    try:
        bar_range, used_fallback = _detect_masked_bars(orig_path, masked_path)
    except Exception as e:
        print(
            f"  Warning [{seed_dir.name}]: bar detection failed ({e})"
            f" — using full file", file=sys.stderr,
        )
        return compute_muspy_metrics(recon_path), True

    if used_fallback:
        print(
            f"  Warning [{seed_dir.name}]: no masked bars detected"
            f" — using full file", file=sys.stderr,
        )
        return compute_muspy_metrics(recon_path), True

    try:
        music = _muspy.read_midi(str(recon_path))
        music = _crop_music_to_range(music, bar_range)
        return _compute_metrics_on_music(music), False
    except Exception as e:
        print(
            f"  Warning [{seed_dir.name}]: crop/metrics failed ({e})"
            f" — using full file", file=sys.stderr,
        )
        return compute_muspy_metrics(recon_path), True


def _seed_number(path: Path) -> int:
    """Extract integer seed from 'seed_N/reconstructed.mid' path for sorting."""
    m = re.search(r"seed_(\d+)", path.parts[-2])
    return int(m.group(1)) if m else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute model aggregate muspy metrics over generated infilling samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples_dir", required=True,          help="Directory containing seed_N/ subdirs with reconstructed.mid.")
    parser.add_argument("--output_json", required=True,          help="Path to save the aggregate JSON.")
    parser.add_argument("--max_files",   type=int, default=1000, help="Max seeds to process.")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    if not samples_dir.exists():
        print(f"Error: --samples_dir '{samples_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(
        samples_dir.glob("seed_*/reconstructed.mid"),
        key=_seed_number,
    )
    n_found = len(all_files)

    if n_found == 0:
        print(f"Error: no seed_*/reconstructed.mid found under '{samples_dir}'.", file=sys.stderr)
        sys.exit(1)

    all_files = all_files[: args.max_files]
    n_total = len(all_files)

    crop_mode = "masked-bars crop" if _CROP_AVAILABLE else "full file (symusic/muspy unavailable)"

    print(f"\n[compute_model_agg]")
    print(f"  samples_dir : {samples_dir}")
    print(f"  output_json : {output_json}")
    print(f"  found       : {n_found} seed_*/reconstructed.mid")
    print(f"  processing  : {n_total} (max_files={args.max_files})")
    print(f"  mode        : {crop_mode}")
    print()

    per_file: dict[str, dict] = {}
    n_ok       = 0
    n_fail     = 0
    n_fallback = 0

    for i, path in enumerate(all_files):
        metrics, used_fallback = _get_metrics_for_seed(path)
        per_file[str(path)] = metrics

        if any(v is not None for v in metrics.values()):
            n_ok += 1
        else:
            n_fail += 1

        if used_fallback:
            n_fallback += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            print(f"  [{i + 1:>5}/{n_total}]  ok={n_ok}  all-None={n_fail}  fallback={n_fallback}")

    agg = aggregate(per_file)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_json}")
    print(f"Seeds: {n_total}  (ok={n_ok}, all-None={n_fail}, fallback={n_fallback})\n")
    print(f"{'Metric':<35} {'mean':>10}  {'std':>10}  {'n':>6}")
    print("-" * 68)
    for metric, stats in agg.items():
        mean = stats.get("mean")
        std  = stats.get("std")
        n    = stats.get("n", 0)
        mean_s = f"{mean:.4f}" if mean is not None else "None"
        std_s  = f"{std:.4f}"  if std  is not None else "None"
        print(f"  {metric:<33} {mean_s:>10}  {std_s:>10}  {n:>6}")


if __name__ == "__main__":
    main()
