import importlib
import numpy as np

import pytest


def test_metrics_return_finite_values():
    try:
        metrics_module = importlib.import_module("utils.metrics")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Metric dependencies are unavailable in this environment: {exc}")

    HSIC = metrics_module.HSIC
    dHSIC = metrics_module.dHSIC
    mmd = metrics_module.mmd
    rf = metrics_module.rf
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 3))
    y = x + rng.normal(scale=0.1, size=(32, 3))
    score_mmd = mmd(x, y)
    score_rf = rf(x, y)
    score_hsic = HSIC(x[:, 0], x[:, 1:])
    score_dhsic = dHSIC(x[:, 0], x[:, 1], x[:, 2])
    assert np.isfinite(score_mmd)
    assert np.isfinite(score_rf)
    assert np.isfinite(score_hsic)
    assert np.isfinite(score_dhsic)
