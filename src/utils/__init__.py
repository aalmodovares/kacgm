from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ASSETS_ROOT": ("utils.paths", "ASSETS_ROOT"),
    "CONFIGS_ROOT": ("utils.paths", "CONFIGS_ROOT"),
    "ExperimentPaths": ("utils.paths", "ExperimentPaths"),
    "HSIC": ("utils.metrics", "HSIC"),
    "NOTEBOOKS_ROOT": ("utils.paths", "NOTEBOOKS_ROOT"),
    "OUTPUTS_ROOT": ("utils.paths", "OUTPUTS_ROOT"),
    "RAW_DATA_ROOT": ("utils.paths", "RAW_DATA_ROOT"),
    "REPO_ROOT": ("utils.paths", "REPO_ROOT"),
    "RUNNABLES_ROOT": ("utils.paths", "RUNNABLES_ROOT"),
    "SRC_ROOT": ("utils.paths", "SRC_ROOT"),
    "dHSIC": ("utils.metrics", "dHSIC"),
    "create_model_from_graph": ("models.factory", "create_model_from_graph"),
    "ensure_dir": ("utils.paths", "ensure_dir"),
    "evaluate_kaam": ("utils.evaluation", "evaluate_kaam"),
    "evaluate_kaam_mixed": ("utils.evaluation", "evaluate_kaam_mixed"),
    "evaluate_model": ("utils.evaluation", "evaluate_model"),
    "friedman_test": ("utils.stats", "friedman_test"),
    "get_best_hyperparams": ("utils.hyperparams", "get_best_hyperparams"),
    "get_experiment_paths": ("utils.paths", "get_experiment_paths"),
    "get_global_checkpoint_root": ("utils.paths", "get_global_checkpoint_root"),
    "get_kaam_hyperparameters": ("utils.hyperparams", "get_kaam_hyperparameters"),
    "get_p_values_from_table_data": ("utils.stats", "get_p_values_from_table_data"),
    "make_run_id": ("utils.paths", "make_run_id"),
    "mmd": ("utils.metrics", "mmd"),
    "resolve_path": ("utils.paths", "resolve_path"),
    "rf": ("utils.metrics", "rf"),
    "slugify": ("utils.paths", "slugify"),
    "symbolic_kan_regressor": ("models.kan", "symbolic_kan_regressor"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

