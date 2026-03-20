from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CardioPreparedData": ("datasets.cardio", "CardioPreparedData"),
    "ExperimentationModel": ("datasets.sachs", "ExperimentationModel"),
    "get_cardio_graph": ("datasets.cardio", "get_cardio_graph"),
    "get_structural_equations_sachs": ("datasets.sachs", "get_structural_equations_sachs"),
    "graph_data": ("datasets.synthetic", "graph_data"),
    "graph_sachs": ("datasets.sachs", "graph_sachs"),
    "infer_node_types": ("datasets.cardio", "infer_node_types"),
    "load_cardio_dataframe": ("datasets.cardio", "load_cardio_dataframe"),
    "noises_distr": ("datasets.sachs", "noises_distr"),
    "post_process_samples": ("datasets.cardio", "post_process_samples"),
    "prepare_cardio_data": ("datasets.cardio", "prepare_cardio_data"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'datasets' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

