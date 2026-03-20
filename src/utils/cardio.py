from __future__ import annotations

import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


CARDIO_BASE_MODELS = ["kan_mixed", "kaam_mixed", "kan", "kaam", "anm", "dbcm", "flow"]
CARDIO_VARIANTS = ("Original", "Pruned", "Symbolic")
CARDIO_METRICS = ("mmd_obs_avg", "rf_acc_obs_avg", "hsic", "dhsic", "mae")

CARDIO_KAN_PARAM_GRID = {
    "hidden_dim": [0, 5],
    "batch_size": [-1],
    "grid": [1, 5, 10],
    "k": [1, 5, 10],
    "seed": [0],
    "lr": [0.01, 0.001, 0.0001],
    "early_stop": [True],
    "steps": [10000],
    "lamb": [1, 0.1, 0.001],
    "lamb_entropy": [0.1],
    "sparse_init": [False],
    "mult_kan": [False, True],
    "try_gpu": [False],
    "loss": ["mse"],
}

CARDIO_DBCM_PARAM_GRID = {
    "num_epochs": [100, 200, 500],
    "lr": [0.01, 0.001, 0.0001],
    "batch_size": [32, 64, 128],
    "hidden_dim": [32, 64, 128],
}

CARDIO_FLOW_PARAM_GRID = {
    "flow_type": ["CausalNSF", "CausalMAF"],
    "hidden_dims": [(32, 32), (64, 64), (64, 64, 64), (128, 128)],
    "base_lr": [1e-2, 1e-3, 1e-4, 1e-5],
    "early_stopping_patience": [30],
    "scheduler": [None, "plateau"],
    "batch_size": [256],
    "train_val_split": [(0.8, 0.2)],
    "max_epochs": [1000],
    "device": ["cpu"],
    "bins": [4, 8, 16],
}

_CARDIO_SPECIAL_MODELS = {
    "kan_mixed",
    "kaam_mixed",
    "kaam_mixed_pruned",
    "kaam_mixed_symbolic",
    "flow",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clone_model_params_for_seed(model_name, params, seed, checkpoint_root: str | Path | None = None):
    params = deepcopy(params)
    if model_name not in {"kan", "kaam", "kan_mixed", "kaam_mixed"}:
        return params

    checkpoint_base = Path(checkpoint_root) if checkpoint_root is not None else None
    for node, node_params in params.items():
        if not isinstance(node_params, dict):
            continue
        node_params["seed"] = int(seed)
        if checkpoint_base is not None:
            node_params["checkpoint_dir"] = str(checkpoint_base / str(node))
    return params


def get_residuals_anm(causal_model, data):
    from dowhy.gcm.util.general import is_categorical
    from dowhy.graph import get_ordered_predecessors, is_root_node

    residuals = {}
    for node in causal_model.graph.nodes:
        if is_root_node(causal_model.graph, node):
            residuals[node] = data[node].to_numpy()
        elif is_categorical(data[node].to_numpy()):
            continue
        else:
            parents_samples = data[get_ordered_predecessors(causal_model.graph, node)].to_numpy()
            residuals[node] = causal_model.causal_mechanism(node).estimate_noise(data[node].to_numpy(), parents_samples)
    return residuals


def mae_anm(causal_model, data, node_list=None, aggregation=None):
    node_list = list(causal_model.graph.nodes) if node_list is None else list(node_list)
    mae = {}
    for node in node_list:
        if causal_model.graph.in_degree(node) == 0:
            continue
        parents = sorted(causal_model.graph.predecessors(node))
        parents_samples = data[parents].to_numpy()
        y_true = data[node].to_numpy().flatten()
        mechanism = causal_model.causal_mechanism(node)
        if hasattr(mechanism, "classifier_model"):
            y_pred = mechanism.classifier_model.predict(parents_samples).flatten()
        else:
            y_pred = mechanism.prediction_model.predict(parents_samples).flatten()
        mae[node] = float(np.mean(np.abs(y_true - y_pred)))
    if aggregation == "mean":
        return float(np.mean(list(mae.values())))
    if aggregation == "sum":
        return float(np.sum(list(mae.values())))
    return mae


def get_residuals_flow(flow_model, data):
    import torch

    residuals_array = flow_model.flow().transform(
        torch.tensor(data[flow_model.topological_order].values, dtype=torch.float32)
    ).detach().numpy()
    residuals_df = pd.DataFrame(residuals_array, columns=flow_model.topological_order)
    return {node: residuals_df.loc[:, node].to_numpy() for node in flow_model.graph.nodes}


def resolve_cardio_best_params(
    model_names,
    graph_cardio,
    factual_train_d,
    factual_eval_d,
    factual_train_dn,
    load_best_params,
    verbose,
    data_dir,
    n_jobs,
    num_classes,
):
    from utils.hyperparams import get_best_hyperparams

    best_params = {}
    mixed_kan_params = deepcopy(CARDIO_KAN_PARAM_GRID)
    mixed_kan_params["node_types"] = [infer_node_types(factual_train_d, default_num_classes=2)[0]]
    mixed_kan_params["num_classes"] = [num_classes]

    for model_name in model_names:
        if model_name in best_params:
            continue
        if model_name in {"kan_mixed", "kaam_mixed"}:
            best_params[model_name] = get_best_hyperparams(
                model_name,
                "cardio",
                mixed_kan_params,
                graph_cardio,
                factual_train_d,
                factual_eval_d,
                load_best_params,
                verbose,
                data_dir,
                n_jobs,
                num_classes=num_classes,
            )
        elif model_name in {"kan", "kaam"}:
            best_params[model_name] = get_best_hyperparams(
                model_name,
                "cardio",
                CARDIO_KAN_PARAM_GRID,
                graph_cardio,
                factual_train_dn,
                factual_eval_d,
                load_best_params,
                verbose,
                data_dir,
                n_jobs,
                num_classes=num_classes,
            )
        elif model_name == "dbcm":
            best_params[model_name] = get_best_hyperparams(
                model_name,
                "cardio",
                CARDIO_DBCM_PARAM_GRID,
                graph_cardio,
                factual_train_dn,
                factual_eval_d,
                load_best_params,
                verbose,
                data_dir,
                n_jobs,
                num_classes=num_classes,
            )
        elif model_name == "flow":
            best_params[model_name] = get_best_hyperparams(
                model_name,
                "cardio",
                CARDIO_FLOW_PARAM_GRID,
                graph_cardio,
                factual_train_dn,
                factual_eval_d,
                load_best_params,
                verbose,
                data_dir,
                n_jobs,
                num_classes=num_classes,
            )
        else:
            best_params[model_name] = {}
    return best_params


def fit_cardio_model(
    model_name,
    graph_cardio,
    params,
    factual_train_d,
    factual_train_dn,
    seed=0,
    checkpoint_root: str | Path | None = None,
):
    from dowhy import gcm
    from dowhy.gcm.auto import AssignmentQuality

    from models.factory import create_model_from_graph
    from models.flow import causalflow_model
    from models.kan import kan_model_mixed

    set_global_seed(int(seed))
    params = clone_model_params_for_seed(model_name, params, seed, checkpoint_root=checkpoint_root)

    if model_name in {"kan_mixed", "kaam_mixed"}:
        model = kan_model_mixed(graph_cardio, deepcopy(params))
        start = time.time()
        model.fit(data=factual_train_d)
        training_time = time.time() - start
    elif model_name == "flow":
        model = causalflow_model(graph_cardio, deepcopy(params))
        start = time.time()
        model.fit(data=factual_train_dn)
        training_time = time.time() - start
    else:
        model = create_model_from_graph(graph_cardio, model_name, params)
        if model_name == "anm":
            gcm.auto.assign_causal_mechanisms(model, factual_train_dn, override_models=True, quality=AssignmentQuality.BETTER)
        start = time.time()
        gcm.fit(model, data=factual_train_dn)
        training_time = time.time() - start

    return model, float(training_time), params


def derive_kaam_mixed_variants(model, factual_train_d, seed=0):
    set_global_seed(int(seed))
    pruned_model = model.clone()
    pruned_model.draw_samples(1000, seed=seed)
    start = time.time()
    pruned_model.prune(factual_train_d)
    pruning_time = time.time() - start

    set_global_seed(int(seed))
    symbolic_model = pruned_model.clone()
    symbolic_model.draw_samples(1000, seed=seed)
    start = time.time()
    symbolic_model.to_symbolic(factual_train_d)
    symbolic_time = time.time() - start

    return {
        "kaam_mixed_pruned": (pruned_model, float(pruning_time)),
        "kaam_mixed_symbolic": (symbolic_model, float(symbolic_time)),
    }


def draw_cardio_observational_samples(model_name, model, num_samples, sample_seed):
    from dowhy import gcm

    set_global_seed(int(sample_seed))
    if model_name in _CARDIO_SPECIAL_MODELS:
        return model.draw_samples(num_samples=num_samples, seed=int(sample_seed))
    return gcm.draw_samples(model, num_samples=num_samples)


def draw_cardio_interventional_samples(model_name, model, intervention, num_samples, sample_seed):
    from dowhy import gcm

    set_global_seed(int(sample_seed))
    if model_name in _CARDIO_SPECIAL_MODELS:
        return model.interventional_samples(intervention, num_samples_to_draw=num_samples, seed=int(sample_seed))
    return gcm.interventional_samples(model, intervention, num_samples_to_draw=num_samples)


def evaluate_cardio_observational(
    model_name,
    model,
    factual_eval_d,
    node_info,
    sample_seed=42,
):
    from utils.metrics import HSIC, dHSIC, mmd, rf

    obs_samples = draw_cardio_observational_samples(model_name, model, len(factual_eval_d), sample_seed)
    obs_samples = obs_samples[factual_eval_d.columns]
    obs_samples = post_process_samples(obs_samples, node_info=node_info, denorm=False)

    metrics = {
        "mmd_obs_avg": float(mmd(factual_eval_d.to_numpy(), obs_samples.to_numpy())),
        "rf_acc_obs_avg": float(rf(factual_eval_d.to_numpy(), obs_samples.to_numpy(), seed=int(sample_seed))),
    }

    extras = {}
    if model_name in {"kan_mixed", "kaam_mixed", "kaam_mixed_pruned", "kaam_mixed_symbolic"}:
        residuals = model.get_residuals(factual_eval_d)
        metrics["mae"] = float(model.mae(factual_eval_d, aggregation="mean"))
        if model_name == "kaam_mixed_symbolic":
            extras["formulas"] = model.get_formulas(ex_round=3)
    elif model_name == "flow":
        residuals = get_residuals_flow(model, factual_eval_d)
    else:
        residuals = get_residuals_anm(model, factual_eval_d)
        if model_name != "dbcm":
            metrics["mae"] = float(mae_anm(model, factual_eval_d, aggregation="mean"))

    metrics["hsic"] = float(HSIC(residuals["systolic"], factual_eval_d[["age", "bmi"]].to_numpy()))
    metrics["dhsic"] = float(dHSIC(residuals["systolic"], residuals["age"], residuals["bmi"]))
    return metrics, obs_samples, residuals, extras


def collect_cardio_interventional_cache(model_name, model, factual_eval_d, node_info, sample_seed=42):
    interventions = []
    for node in factual_eval_d.columns:
        for value in node_info[node]["int_values"]:
            intervention = {node: (lambda _, fixed_value=value: fixed_value)}
            int_samples = draw_cardio_interventional_samples(
                model_name,
                model,
                intervention,
                len(factual_eval_d),
                sample_seed=sample_seed,
            )
            int_samples = int_samples[factual_eval_d.columns]
            int_samples = post_process_samples(int_samples, node_info=node_info, denorm=False)
            interventions.append((node, value, int_samples))
    return interventions


def build_bootstrap_indices(num_rows, n_bootstraps, bootstrap_seed):
    rng = np.random.default_rng(int(bootstrap_seed))
    return [rng.integers(0, num_rows, size=num_rows) for _ in range(int(n_bootstraps))]


def summarize_bootstrap_metrics(df):
    if df.empty:
        return df.copy()

    group_cols = ["model_name", "base_model", "variant", "metric"]
    summary = (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
            q025=lambda values: values.quantile(0.025),
            q975=lambda values: values.quantile(0.975),
            n="count",
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return summary


def metric_rows_from_result(model_name, seed, bootstrap_id, metrics):
    base_model, variant = model_name_to_variant(model_name)
    rows = []
    for metric_name, value in metrics.items():
        rows.append(
            {
                "model_name": model_name,
                "base_model": base_model,
                "variant": variant,
                "seed": int(seed),
                "bootstrap": int(bootstrap_id),
                "metric": metric_name,
                "value": float(value),
            }
        )
    return rows


def model_name_to_variant(model_name):
    if model_name.endswith("_pruned"):
        return model_name[: -len("_pruned")], "Pruned"
    if model_name.endswith("_symbolic"):
        return model_name[: -len("_symbolic")], "Symbolic"
    return model_name, "Original"


def save_bootstrap_indices(indices, target_path):
    target_path = Path(target_path)
    rows = []
    for bootstrap_id, values in enumerate(indices):
        for draw_index, source_index in enumerate(values):
            rows.append(
                {
                    "bootstrap": int(bootstrap_id),
                    "draw_index": int(draw_index),
                    "source_index": int(source_index),
                }
            )
    pd.DataFrame(rows).to_csv(target_path, index=False)


def infer_node_types(dataframe, discrete_threshold=5, default_num_classes=2):
    from datasets.cardio import infer_node_types as _infer_node_types

    return _infer_node_types(dataframe, discrete_threshold=discrete_threshold, default_num_classes=default_num_classes)


def post_process_samples(df, node_info, denorm=False):
    from datasets.cardio import post_process_samples as _post_process_samples

    return _post_process_samples(df, node_info=node_info, denorm=denorm)
