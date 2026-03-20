from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "causalflow_model": ("models.flow", "causalflow_model"),
    "create_dbcm_model_from_graph": ("models.dbcm", "create_model_from_graph"),
    "create_model_from_graph": ("models.factory", "create_model_from_graph"),
    "default_flow_params": ("models.flow", "default_params"),
    "kan_auto_symbolic_formula": ("models.kan", "kan_auto_symbolic_formula"),
    "kan_model_mixed": ("models.kan", "kan_model_mixed"),
    "kan_predictor": ("models.kan", "kan_predictor"),
    "symbolic_kan_regressor": ("models.kan", "symbolic_kan_regressor"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'models' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
