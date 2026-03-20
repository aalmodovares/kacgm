import importlib
import pickle

import pytest


def test_cardio_bootstrap_helpers():
    try:
        cardio_utils = importlib.import_module("utils.cardio")
        pd = importlib.import_module("pandas")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Cardio bootstrap utilities are unavailable in this environment: {exc}")

    build_bootstrap_indices = cardio_utils.build_bootstrap_indices
    summarize_bootstrap_metrics = cardio_utils.summarize_bootstrap_metrics

    indices = build_bootstrap_indices(num_rows=5, n_bootstraps=3, bootstrap_seed=123)
    assert len(indices) == 3
    assert all(len(sample) == 5 for sample in indices)

    rows = pd.DataFrame(
        [
            {"model_name": "kan_mixed", "base_model": "kan_mixed", "variant": "Original", "metric": "mmd_obs_avg", "value": 0.10},
            {"model_name": "kan_mixed", "base_model": "kan_mixed", "variant": "Original", "metric": "mmd_obs_avg", "value": 0.20},
            {"model_name": "kan_mixed", "base_model": "kan_mixed", "variant": "Original", "metric": "mmd_obs_avg", "value": 0.30},
        ]
    )
    summary = summarize_bootstrap_metrics(rows)
    assert list(summary["model_name"]) == ["kan_mixed"]
    assert list(summary["metric"]) == ["mmd_obs_avg"]
    assert summary.loc[0, "n"] == 3
    assert summary.loc[0, "mean"] == pytest.approx(0.20)
    assert summary.loc[0, "q025"] <= summary.loc[0, "median"] <= summary.loc[0, "q975"]


def test_cardio_bootstrap_sample_bundle(tmp_path):
    try:
        runner = importlib.import_module("runnables.run_cardio_bootstrap_evaluation")
        pd = importlib.import_module("pandas")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Cardio bootstrap runnable is unavailable in this environment: {exc}")

    observational = pd.DataFrame({"age": [1.0, 2.0], "bmi": [3.0, 4.0]})
    interventional = [("age", 60, observational.copy())]
    runner._store_sample_bundle(tmp_path, "kaam_mixed_symbolic", 1, 2, observational, interventional)

    sample_dir = tmp_path / "by_bootstrap" / "kaam_mixed_symbolic" / "seed_1" / "bootstrap_2"
    assert (sample_dir / "observational.csv").exists()
    assert (sample_dir / "observational.pkl").exists()
    assert (sample_dir / "interventional.pkl").exists()
    assert (sample_dir / "interventional_manifest.csv").exists()

    with open(sample_dir / "observational.pkl", "rb") as handle:
        loaded_observational = pickle.load(handle)
    with open(sample_dir / "interventional.pkl", "rb") as handle:
        loaded_interventional = pickle.load(handle)

    assert list(loaded_observational.columns) == ["age", "bmi"]
    assert loaded_interventional[0][0] == "age"
    assert loaded_interventional[0][1] == 60


def test_cardio_bootstrap_seed_sample_bundle(tmp_path):
    try:
        runner = importlib.import_module("runnables.run_cardio_bootstrap_evaluation")
        pd = importlib.import_module("pandas")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Cardio bootstrap runnable is unavailable in this environment: {exc}")

    observational = pd.DataFrame({"age": [1.0, 2.0], "bmi": [3.0, 4.0]})
    interventional = [("age", 60, observational.copy())]
    formula_payload = {
        "model_name": "kaam_mixed_symbolic",
        "base_model": "kaam_mixed",
        "variant": "Symbolic",
        "seed": 3,
        "has_formulas": True,
        "formula_status": "symbolic_extracted",
        "n_formulas": 1,
        "formulas": {"systolic": "0.1*age + 0.2"},
    }
    runner._store_seed_sample_bundle(
        tmp_path,
        "kaam_mixed_symbolic",
        3,
        observational,
        interventional,
        formula_payload=formula_payload,
    )

    sample_dir = tmp_path / "by_seed" / "kaam_mixed_symbolic" / "seed_3"
    assert (sample_dir / "observational.csv").exists()
    assert (sample_dir / "interventional.pkl").exists()
    assert (sample_dir / "interventional_manifest.csv").exists()
    assert (sample_dir / "formula_payload.pkl").exists()
    assert (sample_dir / "formula_metadata.json").exists()


def test_cardio_bootstrap_hydrates_best_params_cache(tmp_path):
    try:
        runner = importlib.import_module("runnables.run_cardio_bootstrap_evaluation")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Cardio bootstrap runnable is unavailable in this environment: {exc}")

    cardio_data_dir = tmp_path / "cardio" / "data"
    cardio_bootstrap_data_dir = tmp_path / "cardio_bootstrap" / "data"
    cardio_data_dir.mkdir(parents=True)
    cardio_bootstrap_data_dir.mkdir(parents=True)

    source_file = cardio_data_dir / "best_params_kan_cardio.pkl"
    source_file.write_bytes(b"cache")

    copied_files = runner._hydrate_cardio_best_params_cache(
        cardio_bootstrap_data_dir,
        output_dir=tmp_path / "cardio_bootstrap",
    )

    target_file = cardio_bootstrap_data_dir / "best_params_kan_cardio.pkl"
    assert target_file.exists()
    assert target_file.read_bytes() == b"cache"
    assert target_file in copied_files
