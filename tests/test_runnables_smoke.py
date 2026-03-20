from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_help(module_name):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_benchmark_runnables_help():
    for module_name in [
        "runnables.run_continuous_benchmark",
        "runnables.run_discrete_benchmark",
        "runnables.run_sachs_benchmark",
        "runnables.run_sensitivity",
        "runnables.run_cardio_case_study",
        "runnables.run_cardio_bootstrap_evaluation",
        "runnables.run_symbolic_pruning_study",
    ]:
        _run_help(module_name)
