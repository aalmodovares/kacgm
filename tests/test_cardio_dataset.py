import importlib

import pytest


def test_cardio_loader_and_preparation():
    try:
        cardio_module = importlib.import_module("datasets.cardio")
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Cardio dataset dependencies are unavailable in this environment: {exc}")

    get_cardio_graph = cardio_module.get_cardio_graph
    load_cardio_dataframe = cardio_module.load_cardio_dataframe
    prepare_cardio_data = cardio_module.prepare_cardio_data
    cardio = load_cardio_dataframe()
    graph = get_cardio_graph()
    prepared = prepare_cardio_data(data=cardio.head(200))
    assert not cardio.empty
    assert set(["age", "bmi", "diabetes", "systolic", "ischemia", "macv"]).issubset(cardio.columns)
    assert "age" in graph.nodes
    assert prepared.factual_train_d.shape[1] == prepared.factual_eval_d.shape[1]
    assert prepared.factual_train_dn.shape == prepared.factual_train_d.shape
