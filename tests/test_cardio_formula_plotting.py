import importlib

import pytest


def test_get_delta_substitutes_feature_names():
    try:
        plotting = importlib.import_module("plotting.cardio_formula")
        pd = importlib.import_module("pandas")
        sp = importlib.import_module("sympy")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Cardio formula plotting helpers are unavailable in this environment: {exc}")

    x_frame = pd.DataFrame({"age": [1.0, 2.0], "bmi": [3.0, 4.0]})
    formula = 2 * sp.symbols("x_1") + 3 * sp.symbols("x_2") + sp.Float(1.0)

    delta_formula, delta_frames = plotting.get_delta(x_frame, formula)
    delta_frame = delta_frames[0]

    assert str(delta_formula) == "2*age + 3*bmi + 1.0"
    assert list(delta_frame.columns) == ["age", "bmi", "const"]
    assert delta_frame.loc[0, "age"] == pytest.approx(2.0)
    assert delta_frame.loc[0, "bmi"] == pytest.approx(9.0)
    assert delta_frame.loc[0, "const"] == pytest.approx(1.0)


def test_select_symbolic_formula_payload_is_reproducible():
    try:
        plotting = importlib.import_module("plotting.cardio_formula")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Cardio formula plotting helpers are unavailable in this environment: {exc}")

    payloads = [
        {"model_name": "kaam_mixed_symbolic", "seed": 0, "has_formulas": True, "formulas": {"a": 1}},
        {"model_name": "kaam_mixed_symbolic", "seed": 1, "has_formulas": True, "formulas": {"a": 2}},
        {"model_name": "kaam_mixed_pruned", "seed": 2, "has_formulas": False, "formulas": None},
    ]

    first = plotting.select_symbolic_formula_payload(payloads, rng_seed=2026)
    second = plotting.select_symbolic_formula_payload(payloads, rng_seed=2026)

    assert first["seed"] == second["seed"]
    assert first["model_name"] == "kaam_mixed_symbolic"
