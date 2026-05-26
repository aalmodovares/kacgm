from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product


DEFAULT_DATASET = "simpson-non-linear"
DEFAULT_N = 3000
DEFAULT_N_REALIZATIONS = 10
DEFAULT_BASE_SEED = 0
DEFAULT_GRAPH_CONDITIONS = (
    "correct",
    "missing_edge_x2_x3",
    "reversed_edge_x1_x2",
    "added_edge_x2_x4",
)
DEFAULT_INTERVENTIONS = (
    ("x1", -1.0),
    ("x1", 0.0),
    ("x1", 1.0),
    ("x2", -1.0),
    ("x2", 0.0),
    ("x2", 1.0),
)
DEFAULT_ATE_CONTRASTS = ((-1.0, 0.0), (0.0, 1.0), (-1.0, 1.0))
DEFAULT_OUTCOME_NODES = ("x3", "x4")
DEFAULT_KAN_PARAMS = {
    "x1": {
        "hidden_dim": 0,
        "batch_size": -1,
        "grid": 3,
        "k": 3,
        "seed": 0,
        "lr": 0.001,
        "early_stop": True,
        "steps": 10000,
        "lamb": 0.01,
        "lamb_entropy": 0.1,
        "sparse_init": False,
        "mult_kan": False,
        "try_gpu": False,
        "loss": "mse",
    },
    "x2": {
        "hidden_dim": 0,
        "batch_size": -1,
        "grid": 3,
        "k": 3,
        "seed": 0,
        "lr": 0.001,
        "early_stop": True,
        "steps": 10000,
        "lamb": 0.01,
        "lamb_entropy": 0.1,
        "sparse_init": False,
        "mult_kan": False,
        "try_gpu": False,
        "loss": "mse",
    },
    "x3": {
        "hidden_dim": 0,
        "batch_size": -1,
        "grid": 3,
        "k": 3,
        "seed": 0,
        "lr": 0.001,
        "early_stop": True,
        "steps": 10000,
        "lamb": 0.01,
        "lamb_entropy": 0.1,
        "sparse_init": False,
        "mult_kan": False,
        "try_gpu": False,
        "loss": "mse",
    },
    "x4": {
        "hidden_dim": 0,
        "batch_size": -1,
        "grid": 3,
        "k": 3,
        "seed": 0,
        "lr": 0.001,
        "early_stop": True,
        "steps": 10000,
        "lamb": 0.01,
        "lamb_entropy": 0.1,
        "sparse_init": False,
        "mult_kan": False,
        "try_gpu": False,
        "loss": "mse",
    },
}
DEFAULT_FLOW_PARAMS = {
    "flow_type": "CausalNSF",
    "hidden_dims": (32, 32, 32),
    "base_lr": 1e-3,
    "early_stopping_patience": 50,
    "scheduler": "plateau",
    "batch_size": 256,
    "train_val_split": (0.8, 0.2),
    "max_epochs": 2000,
    "device": "cpu",
    "bins": 8,
}
DEFAULT_ZUKO_NSF_PARAMS = {
    "transforms": 3,
    "hidden_features": (64, 64),
    "base_lr": 1e-3,
    "early_stopping_patience": 30,
    "scheduler": "plateau",
    "batch_size": 256,
    "train_val_split": (0.8, 0.2),
    "max_epochs": 1000,
    "device": "cpu",
    "bins": 8,
}


def _make_intervention(node, value):
    return {node: (lambda _, fixed_value=float(value): fixed_value)}


def _make_graph_condition(true_graph, graph_condition):
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(true_graph.nodes)
    graph.add_edges_from(true_graph.edges)
    if graph_condition == "correct":
        return graph
    if graph_condition == "missing_edge_x2_x3":
        graph.remove_edge("x2", "x3")
        return graph
    if graph_condition == "reversed_edge_x1_x2":
        graph.remove_edge("x1", "x2")
        graph.add_edge("x2", "x1")
        return graph
    if graph_condition == "added_edge_x2_x4":
        graph.add_edge("x2", "x4")
        return graph
    raise ValueError(f"Unknown graph condition: {graph_condition}")


def _metric_row(
    graph_condition,
    realization,
    model_name,
    metric,
    value,
    intervened_node=None,
    intervention_id=None,
    intervention_value=None,
    intervention_value_ref=None,
    outcome_node=None,
):
    return {
        "graph_condition": str(graph_condition),
        "realization": int(realization),
        "model": str(model_name),
        "metric": str(metric),
        "value": float(value),
        "intervened_node": None if intervened_node is None else str(intervened_node),
        "intervention_id": intervention_id,
        "intervention_value": intervention_value,
        "intervention_value_ref": intervention_value_ref,
        "outcome_node": None if outcome_node is None else str(outcome_node),
    }


def _draw_observational_samples(model_name, model, num_samples, sample_seed):
    import numpy as np
    from dowhy import gcm

    np.random.seed(int(sample_seed))
    if model_name == "flow":
        return model.draw_samples(num_samples=num_samples, seed=int(sample_seed))
    return gcm.draw_samples(model, num_samples=num_samples)


def _draw_interventional_samples(model_name, model, intervention, num_samples, sample_seed):
    import numpy as np
    from dowhy import gcm

    np.random.seed(int(sample_seed))
    if model_name == "flow":
        return model.interventional_samples(intervention, num_samples_to_draw=num_samples, seed=int(sample_seed))
    return gcm.interventional_samples(model, intervention, num_samples_to_draw=num_samples)


def _draw_counterfactual_samples(model_name, model, intervention, factual_eval, sample_seed):
    import numpy as np
    from dowhy import gcm

    np.random.seed(int(sample_seed))
    if model_name == "flow":
        return model.counterfactual_samples(intervention, factual_samples=factual_eval, seed=int(sample_seed))
    return gcm.counterfactual_samples(
        model,
        intervention,
        observed_data=factual_eval.copy(),
    )


def _compute_residuals(model_name, model, factual_eval):
    from utils.cardio import get_residuals_anm, get_residuals_flow

    if model_name == "flow":
        return get_residuals_flow(model, factual_eval)
    return get_residuals_anm(model, factual_eval)


def _add_independence_rows(rows, graph_condition, realization, model_name, model_graph, factual_eval, residuals):
    from utils.metrics import HSIC, dHSIC

    for target_node in DEFAULT_OUTCOME_NODES:
        parents = list(model_graph.predecessors(target_node))
        if not parents:
            continue
        rows.append(
            _metric_row(
                graph_condition=graph_condition,
                realization=realization,
                model_name=model_name,
                metric="hsic",
                value=HSIC(residuals[target_node], factual_eval[parents].to_numpy()),
                outcome_node=target_node,
            )
        )
        rows.append(
            _metric_row(
                graph_condition=graph_condition,
                realization=realization,
                model_name=model_name,
                metric="dhsic",
                value=dHSIC(residuals[target_node], *[residuals[parent] for parent in parents]),
                outcome_node=target_node,
            )
        )


def _fit_kan_model(graph, factual_train, kan_params):
    from dowhy import gcm

    from models.factory import create_model_from_graph

    model = create_model_from_graph(graph, "kan", deepcopy(kan_params))
    gcm.fit(model, data=factual_train)
    return model


def _fit_flow_model(graph, factual_train, flow_params):
    from models.flow import causalflow_model

    model = causalflow_model(graph, deepcopy(flow_params))
    model.fit(data=factual_train)
    return model


def _fit_zuko_nsf_model(factual_train, zuko_nsf_params):
    from models.zuko_flow import zuko_nsf_model

    model = zuko_nsf_model(deepcopy(zuko_nsf_params))
    model.fit(data=factual_train)
    return model


def run_one(graph_condition, realization, dataset, n, kan_params, flow_params, zuko_nsf_params, base_seed=0):
    import numpy as np

    from datasets.synthetic import graph_data
    from utils.metrics import mmd, rf

    seed = int(base_seed + realization)
    sample_seed = int(seed + 10_000)
    data_all, data_cf_all, true_graph, _ = graph_data(name=dataset).generate(
        num_samples=2 * n,
        seed=seed,
    )
    factual_train = data_all.iloc[:n].reset_index(drop=True)
    factual_eval = data_all.iloc[n : 2 * n].reset_index(drop=True)
    cf_eval = [frame.iloc[n : 2 * n].reset_index(drop=True) for frame in data_cf_all]
    model_graph = _make_graph_condition(true_graph, graph_condition)

    models = {
        "kan": _fit_kan_model(model_graph, factual_train, kan_params),
        "flow": _fit_flow_model(model_graph, factual_train, flow_params),
    }
    obs_only_models = {
        "nsf": _fit_zuko_nsf_model(factual_train, zuko_nsf_params),
    }

    rows = []
    for model_name, model in obs_only_models.items():
        obs_samples = model.draw_samples(num_samples=n, seed=sample_seed)
        obs_samples = obs_samples[factual_eval.columns]
        rows.append(
            _metric_row(
                graph_condition=graph_condition,
                realization=realization,
                model_name=model_name,
                metric="mmd_obs",
                value=mmd(factual_eval.to_numpy(), obs_samples.to_numpy()),
            )
        )
        rows.append(
            _metric_row(
                graph_condition=graph_condition,
                realization=realization,
                model_name=model_name,
                metric="rf_obs",
                value=rf(factual_eval.to_numpy(), obs_samples.to_numpy(), seed=sample_seed),
            )
        )

    for model_name, model in models.items():
        obs_samples = _draw_observational_samples(model_name, model, n, sample_seed)
        obs_samples = obs_samples[factual_eval.columns]
        rows.append(
            _metric_row(
                graph_condition=graph_condition,
                realization=realization,
                model_name=model_name,
                metric="mmd_obs",
                value=mmd(factual_eval.to_numpy(), obs_samples.to_numpy()),
            )
        )
        rows.append(
            _metric_row(
                graph_condition=graph_condition,
                realization=realization,
                model_name=model_name,
                metric="rf_obs",
                value=rf(factual_eval.to_numpy(), obs_samples.to_numpy(), seed=sample_seed),
            )
        )

        residuals = _compute_residuals(model_name, model, factual_eval)
        _add_independence_rows(rows, graph_condition, realization, model_name, model.graph, factual_eval, residuals)

        cf_predictions = {}
        for intervention_id, (intervened_node, intervention_value) in enumerate(DEFAULT_INTERVENTIONS):
            intervention = _make_intervention(intervened_node, intervention_value)
            int_samples = _draw_interventional_samples(model_name, model, intervention, n, sample_seed + intervention_id)
            int_samples = int_samples[factual_eval.columns]
            cf_pred = _draw_counterfactual_samples(model_name, model, intervention, factual_eval, sample_seed + intervention_id)
            cf_pred = cf_pred[factual_eval.columns]
            cf_true = cf_eval[intervention_id]
            cf_predictions[(intervened_node, intervention_value)] = cf_pred

            rows.append(
                _metric_row(
                    graph_condition=graph_condition,
                    realization=realization,
                    model_name=model_name,
                    metric="mmd_int",
                    value=mmd(cf_true.to_numpy(), int_samples.to_numpy()),
                    intervened_node=intervened_node,
                    intervention_id=intervention_id,
                    intervention_value=intervention_value,
                )
            )
            rows.append(
                _metric_row(
                    graph_condition=graph_condition,
                    realization=realization,
                    model_name=model_name,
                    metric="rf_int",
                    value=rf(cf_true.to_numpy(), int_samples.to_numpy(), seed=sample_seed + intervention_id),
                    intervened_node=intervened_node,
                    intervention_id=intervention_id,
                    intervention_value=intervention_value,
                )
            )
            for outcome_node in DEFAULT_OUTCOME_NODES:
                rows.append(
                    _metric_row(
                        graph_condition=graph_condition,
                        realization=realization,
                        model_name=model_name,
                        metric="cf_mae",
                        value=np.abs(cf_true[outcome_node].to_numpy() - cf_pred[outcome_node].to_numpy()).mean(),
                        intervened_node=intervened_node,
                        intervention_id=intervention_id,
                        intervention_value=intervention_value,
                        outcome_node=outcome_node,
                    )
                )

        for intervened_node in ("x1", "x2"):
            for outcome_node in DEFAULT_OUTCOME_NODES:
                for value_a, value_b in DEFAULT_ATE_CONTRASTS:
                    cf_true_a = cf_eval[DEFAULT_INTERVENTIONS.index((intervened_node, value_a))]
                    cf_true_b = cf_eval[DEFAULT_INTERVENTIONS.index((intervened_node, value_b))]
                    cf_pred_a = cf_predictions[(intervened_node, value_a)]
                    cf_pred_b = cf_predictions[(intervened_node, value_b)]
                    ate_true = float(cf_true_b[outcome_node].mean() - cf_true_a[outcome_node].mean())
                    ate_pred = float(cf_pred_b[outcome_node].mean() - cf_pred_a[outcome_node].mean())
                    rows.append(
                        _metric_row(
                            graph_condition=graph_condition,
                            realization=realization,
                            model_name=model_name,
                            metric="ate_error",
                            value=abs(ate_pred - ate_true),
                            intervened_node=intervened_node,
                            intervention_value=value_a,
                            intervention_value_ref=value_b,
                            outcome_node=outcome_node,
                        )
                    )
    return rows


def run_graph_misspecification_sensitivity(
    output_dir=None,
    dataset=DEFAULT_DATASET,
    n=DEFAULT_N,
    n_realizations=DEFAULT_N_REALIZATIONS,
    jobs=1,
    debug=False,
    base_seed=DEFAULT_BASE_SEED,
):
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed

    from utils.paths import get_experiment_paths

    paths = get_experiment_paths("graph_misspecification_sensitivity", output_dir=output_dir)
    kan_params = deepcopy(DEFAULT_KAN_PARAMS)
    flow_params = deepcopy(DEFAULT_FLOW_PARAMS)
    zuko_nsf_params = deepcopy(DEFAULT_ZUKO_NSF_PARAMS)

    if debug:
        for node_params in kan_params.values():
            node_params["steps"] = 2
        flow_params["max_epochs"] = 2
        zuko_nsf_params["max_epochs"] = 2
        n_realizations = min(int(n_realizations), 2)
        jobs = 1

    tasks = list(product(DEFAULT_GRAPH_CONDITIONS, range(int(n_realizations))))
    all_rows_nested = Parallel(n_jobs=jobs, backend="loky", verbose=10)(
        delayed(run_one)(
            graph_condition,
            realization,
            dataset,
            int(n),
            kan_params,
            flow_params,
            zuko_nsf_params,
            base_seed=base_seed,
        )
        for graph_condition, realization in tasks
    )
    rows = [row for chunk in all_rows_nested for row in chunk]
    df = (
        pd.DataFrame(rows)
        .sort_values(["metric", "graph_condition", "model", "realization"])
        .reset_index(drop=True)
    )
    df.to_csv(paths.data / "graph_misspecification_sensitivity_long.csv", index=False)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the graph misspecification sensitivity experiment.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where experiment outputs will be written.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Synthetic dataset name.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Train and evaluation samples per realization.")
    parser.add_argument("--n-realizations", type=int, default=DEFAULT_N_REALIZATIONS, help="Number of realizations.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs across graph-condition and realization pairs.")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="Base seed used to generate realizations.")
    parser.add_argument("--debug", action="store_true", help="Run a reduced debug version.")
    return parser


def main():
    args = build_parser().parse_args()
    run_graph_misspecification_sensitivity(
        output_dir=args.output_dir,
        dataset=args.dataset,
        n=args.n,
        n_realizations=args.n_realizations,
        jobs=args.jobs,
        debug=args.debug,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
