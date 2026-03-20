from __future__ import annotations

import itertools
import os
import pickle
from copy import deepcopy

import numpy as np
from dowhy import gcm
from joblib import Parallel, delayed

from models.factory import create_model_from_graph
from models.flow import causalflow_model
from models.kan import kan_model_mixed
from utils.metrics import mmd, rf
from utils.paths import get_global_checkpoint_root, make_run_id, slugify


def _strip_search_metadata(params, graph, model_name):
    if model_name in {"kan", "kan_mixed", "kaam", "kaam_mixed"}:
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                continue
            params[node].pop("mmd", None)
            params[node].pop("rf_acc", None)
            params[node].pop("checkpoint_dir", None)
    else:
        params.pop("mmd", None)
        params.pop("rf_acc", None)
        params.pop("checkpoint_dir", None)
    return params


def get_best_hyperparams(
    model_name,
    dataset,
    param_candidates,
    graph,
    factual_train,
    factual_eval,
    load_existent,
    verbose,
    data_dir,
    n_threads=1,
    delete_saved_models=True,
    noise=None,
    num_classes=None,
):
    del delete_saved_models
    data_dir = os.fspath(data_dir)

    if load_existent:
        fname = os.path.join(data_dir, f"best_params_{model_name}_{dataset}.pkl")
        if model_name == "kaam":
            fname = os.path.join(data_dir, f"best_params_kan_{dataset}.pkl")
        if model_name == "kaam_mixed":
            fname = os.path.join(data_dir, f"best_params_kan_mixed_{dataset}.pkl")
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                results = pickle.load(f)
            best_params = results["best_params"]
            if "kaam" in model_name:
                best_params = get_kaam_hyperparameters(results["results_all"], graph)
            best_params = _strip_search_metadata(best_params, graph, model_name)
            print(f"Loaded existing best parameters for {model_name}: {best_params} and dataset {dataset}")
            return best_params
        print(f"No existing parameters found for {model_name} and {dataset}, proceeding with grid search...")

    keys, values = zip(*param_candidates.items())
    all_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Starting grid search for {model_name} on dataset {dataset} with {len(all_params)} combinations...")

    checkpoint_root = get_global_checkpoint_root() / slugify(dataset) / slugify(model_name)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def evaluate_params(params):
        params = deepcopy(params)
        if model_name in {"kan", "kaam", "kan_mixed", "kaam_mixed"} and params["hidden_dim"] == 0 and params["mult_kan"]:
            return 0, 0

        candidate_checkpoint = checkpoint_root / make_run_id(f"{dataset}_{model_name}")
        candidate_checkpoint.mkdir(parents=True, exist_ok=True)

        if model_name in {"kan", "kan_mixed", "kaam", "kaam_mixed"}:
            per_node_params = {}
            for node in graph.nodes:
                if graph.in_degree(node) == 0:
                    continue
                per_node_params[node] = deepcopy(params)
                per_node_params[node]["checkpoint_dir"] = os.fspath(candidate_checkpoint / slugify(str(node)))
        else:
            per_node_params = deepcopy(params)

        if model_name in {"kan_mixed", "kaam_mixed", "flow"}:
            if model_name in {"kan_mixed", "kaam_mixed"}:
                model = kan_model_mixed(graph, deepcopy(per_node_params))
            else:
                model = causalflow_model(graph, deepcopy(per_node_params))
            model.fit(data=factual_train)
            np.random.seed(42)
            obs_samples = model.draw_samples(num_samples=len(factual_eval))
        else:
            model = create_model_from_graph(graph, model_name, params=deepcopy(per_node_params), noise=noise)
            gcm.fit(model, data=factual_train)
            np.random.seed(42)
            obs_samples = gcm.draw_samples(model, num_samples=len(factual_eval))

        obs_samples = obs_samples[factual_eval.columns]
        discrete_nodes = [node for node in factual_eval.columns if len(factual_eval[node].unique()) <= 5]
        if discrete_nodes:
            obs_samples[discrete_nodes] = obs_samples[discrete_nodes].round().astype(int)
            for dnode in discrete_nodes:
                obs_samples[dnode] = obs_samples[dnode].clip(0, num_classes[dnode] - 1)

        metric_mmd = {}
        metric_rf_acc = {}
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                continue
            metric_mmd[node] = mmd(factual_eval[node].to_numpy().reshape(-1, 1), obs_samples[node].to_numpy().reshape(-1, 1))
            metric_rf_acc[node] = rf(factual_eval[node].to_numpy().reshape(-1, 1), obs_samples[node].to_numpy().reshape(-1, 1))
        metric_mmd["all"] = mmd(factual_eval.to_numpy(), obs_samples.to_numpy())
        metric_rf_acc["all"] = rf(factual_eval.to_numpy(), obs_samples.to_numpy())
        params["mmd"] = metric_mmd
        params["rf_acc"] = metric_rf_acc
        if verbose:
            print(f"Params: {params}, MMD: {metric_mmd}, RF ACC: {metric_rf_acc}")
        return params, metric_mmd, metric_rf_acc

    results_all = Parallel(n_jobs=n_threads)(delayed(evaluate_params)(params) for params in all_params)
    results_all = [res for res in results_all if res[0] != 0]

    best_params = {}
    best_metric = {}
    if model_name in {"kan", "kan_mixed", "kaam", "kaam_mixed"}:
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                continue
            best_params_node = None
            best_metric_node = float("inf")
            for params, metric_mmd, metric_rf_acc in results_all:
                if metric_rf_acc[node] + metric_rf_acc["all"] < best_metric_node:
                    best_metric_node = metric_rf_acc[node] + metric_rf_acc["all"]
                    best_params_node = params
            best_params[node] = deepcopy(best_params_node)
            best_metric[node] = best_metric_node
    else:
        best_params_ = None
        best_metric_ = float("inf")
        for params, metric_mmd, metric_rf_acc in results_all:
            if metric_rf_acc["all"] < best_metric_:
                best_metric_ = metric_rf_acc["all"]
                best_params_ = params
        best_params = deepcopy(best_params_)
        best_metric["all"] = best_metric_

    results = {"results_all": results_all, "best_params": best_params, "best_metric": best_metric}
    with open(os.path.join(data_dir, f"best_params_{model_name}_{dataset}.pkl"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    if "kaam" in model_name:
        best_params = get_kaam_hyperparameters(results_all, graph)
    best_params = _strip_search_metadata(best_params, graph, model_name)
    print(f"Best parameters for {model_name}: {best_params} with MMD: {best_metric}")
    return best_params


def get_kaam_hyperparameters(kan_params, graph):
    best_kaam_params = {}
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            continue
        best_metric_node = float("inf")
        best_kaam_params_node = None
        for params, metric_mmd, metric_rf_acc in kan_params:
            if metric_rf_acc[node] + metric_rf_acc["all"] < best_metric_node and params["mult_kan"] is False and params["hidden_dim"] == 0:
                best_metric_node = metric_rf_acc[node] + metric_rf_acc["all"]
                best_kaam_params_node = deepcopy(params)
        best_kaam_params[node] = best_kaam_params_node
    return best_kaam_params
