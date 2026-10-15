"""Smoke tests: imports, Hydra config composition and CLI --help.

CPU only; no dataset or checkpoint required. Run with `uv run pytest`.
Tests for the evaluation scripts are skipped unless the `eval` dependency group
is installed (`uv sync --group eval`).
"""

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT / "conf"

HAS_EVAL_DEPS = all(importlib.util.find_spec(m) is not None for m in ("muspy", "scipy"))
needs_eval = pytest.mark.skipif(not HAS_EVAL_DEPS, reason="needs `uv sync --group eval`")


# =================================
# Imports
# =================================

@pytest.mark.parametrize("module", [
    "midit",
    "midit.infer",
    "midit.loss",
    "midit.model",
    "midit.model.model_factory",
    "midit.data.lakh_dataset",
    "midit.data.masking",
])
def test_import(module):
    importlib.import_module(module)


# =================================
# Hydra config composition
# =================================

def _compose(config_name, overrides=()):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


@pytest.mark.parametrize("config_name", ["config", "config_infilling", "config_infilling_v2"])
def test_compose_config(config_name):
    cfg = _compose(config_name)
    assert cfg.model.d_model == 512
    if config_name != "config":
        assert cfg.masking.strategy == "bar_fill"


def test_compose_readme_overrides():
    """The override keys used in README.md / conf/README.md must exist."""
    cfg = _compose("config", [
        "chunking.preprocessed_dir=data/preprocessed_v2_50k",
        "train.lr=3e-4",
        "lambda_weights.tempo=1.0",
        "lambda_weights.timesig=0.5",
        "lambda_weights.rest=1.0",
    ])
    assert cfg.train.lr == 3e-4
    assert cfg.lambda_weights.tempo == 1.0

    cfg = _compose("config_infilling", [
        "train.checkpoint_path=logs/checkpoints/midit-phase1-epoch14.ckpt",
        "chunking.preprocessed_dir=data/preprocessed_v2_50k",
        "train.lr=3e-5",
    ])
    assert cfg.train.checkpoint_path == "logs/checkpoints/midit-phase1-epoch14.ckpt"


# =================================
# CLI --help for the commands documented in README.md
# =================================

HELP_COMMANDS = [
    ["scripts/inference/infer_from_midi.py"],
    ["midit/infer.py"],
    ["-m", "midit"],
    ["scripts/inference/generate_samples_batch.py"],
    ["scripts/preprocessing/preprocess_v2.py"],
    ["midit/train_phase1.py"],
    ["midit/train_infilling.py"],
    ["scripts/evaluation/evaluate_phase1_accuracy_DIRECT.py"],
    ["scripts/audit_fixes/check_sample_integrity.py"],
    ["scripts/audit_fixes/probe_no_cheat.py"],
    pytest.param(["scripts/evaluation/compute_muspy_metrics.py"], marks=needs_eval),
    pytest.param(["scripts/evaluation/js/compute_reference_agg.py"], marks=needs_eval),
    pytest.param(["scripts/evaluation/js/compute_model_agg.py"], marks=needs_eval),
    pytest.param(["scripts/evaluation/js/compute_js_score.py"], marks=needs_eval),
]


@pytest.mark.parametrize("command", HELP_COMMANDS, ids=lambda c: " ".join(c))
def test_cli_help(command, tmp_path):
    # Run from an empty directory so nothing is written into the repository.
    args = [a if a.startswith("-") or a == "midit" else str(ROOT / a) for a in command]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, *args, "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage" in result.stdout.lower() or "powered by hydra" in result.stdout.lower()
    assert not any(tmp_path.iterdir()), "--help must not create files"
