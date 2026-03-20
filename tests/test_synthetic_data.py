import importlib

import pytest


def test_graph_data_generate_returns_expected_shapes():
    try:
        graph_data = importlib.import_module("datasets.synthetic").graph_data
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        pytest.skip(f"Synthetic dataset dependencies are unavailable in this environment: {exc}")

    data, cf_data, graph, formula = graph_data(name="3-chain-linear").generate(num_samples=20, seed=0)
    assert len(data) == 20
    assert len(cf_data) == 6
    assert list(data.columns) == ["x1", "x2", "x3"]
    assert set(graph.nodes()) == {"x1", "x2", "x3"}
    assert "x2" in formula
