from __future__ import annotations

import argparse
import pickle
import time
from copy import deepcopy

DEFAULT_DATASETS = [
    "3-chain-linear",
    "3-chain-non-linear",
    "4-chain-linear",
    "5-chain-linear",
    "collider-linear",
    "fork-linear",
    "fork-non-linear",
    "simpson-non-linear",
    "simpson-symprod",
    "triangle-linear",
    "triangle-non-linear",
]
DEFAULT_N_VALUES = [1000, 100, 10]
DEFAULT_MODELS = ["kan_mixed", "kaam_mixed", "kan", "kaam", "anm", "dbcm", "flow"]
DEFAULT_KAN_PARAMS = {
    "hidden_dim": [0, 5],
    "batch_size": [-1],
    "grid": [1, 5, 10],
    "k": [1, 5, 10],
    "seed": [0],
    "lr": [0.01, 0.001, 0.0001],
    "early_stop": [True],
    "steps": [10000],
    "lamb": [0.1, 0.01, 0.001],
    "lamb_entropy": [0.1],
    "sparse_init": [False],
    "mult_kan": [False, True],
    "try_gpu": [False],
    "loss": ["mse"],
}
DEFAULT_DBCM_PARAMS = {
    "num_epochs": [100, 200, 500],
    "lr": [0.01, 0.001, 0.0001],
    "batch_size": [32, 64, 128],
    "hidden_dim": [32, 64, 128],
}
DEFAULT_FLOW_PARAMS = {
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
DEFAULT_INTERVENTIONS = [
    {"x1": lambda x: -1},
    {"x1": lambda x: 0},
    {"x1": lambda x: 1},
    {"x2": lambda x: -1},
    {"x2": lambda x: 0},
    {"x2": lambda x: 1},
]


def run_discrete_benchmark(output_dir=None, datasets=None, n_values=None, model_names=None, load_best_params=True, verbose=False, n_jobs=1):
    import numpy as np
    from dowhy import gcm
    from dowhy.gcm.auto import AssignmentQuality

    from datasets.synthetic import graph_data
    from models.factory import create_model_from_graph
    from models.flow import causalflow_model
    from models.kan import kan_model_mixed
    from utils.evaluation import evaluate_model
    from utils.hyperparams import get_best_hyperparams
    from utils.paths import get_experiment_paths

    paths = get_experiment_paths("discrete", output_dir=output_dir)
    datasets = datasets or DEFAULT_DATASETS
    n_values = n_values or DEFAULT_N_VALUES
    model_names = model_names or DEFAULT_MODELS

    for dataset in datasets:
        for n in n_values:
            print(f"Training and evaluating on discrete dataset {dataset} with n={n} samples")
            data_all, data_cf_all, graph, formula_gt = graph_data(name=dataset).generate(num_samples=2 * n, seed=0, discrete=True)
            factual_train_d = data_all.iloc[:n]
            factual_eval_d = data_all.iloc[n : 2 * n]
            cf_eval_d = [c.iloc[n : 2 * n] for c in data_cf_all]

            factual_train_d.to_csv(paths.samples / f"{dataset}_factual_train_d_n_{n}.csv", index=False)
            factual_eval_d.to_csv(paths.samples / f"{dataset}_factual_eval_d_n_{n}.csv", index=False)
            for ic, cf_frame in enumerate(cf_eval_d):
                cf_frame.to_csv(paths.samples / f"{dataset}_cf_eval_d_n_{n}_{ic}.csv", index=False)

            factual_train_dn = factual_train_d + np.random.normal(0, 0.01, factual_train_d.shape)
            factual_train_dn.to_csv(paths.samples / f"{dataset}_factual_train_dn_n_{n}.csv", index=False)

            node_types = {}
            num_classes = {}
            for node in factual_train_d.columns:
                if len(factual_train_d[node].unique()) <= 5:
                    node_types[node] = "discrete"
                    num_classes[node] = 3
                else:
                    node_types[node] = "continuous"

            mixed_kan_params = deepcopy(DEFAULT_KAN_PARAMS)
            mixed_kan_params["node_types"] = [node_types]
            mixed_kan_params["num_classes"] = [num_classes]

            for model_name in model_names:
                if model_name in {"kan_mixed", "kaam_mixed"}:
                    params = get_best_hyperparams(model_name, dataset, mixed_kan_params, graph, factual_train_d, factual_eval_d, load_best_params, verbose, paths.data, n_jobs, num_classes=num_classes)
                elif model_name in {"kan", "kaam"}:
                    params = get_best_hyperparams(model_name, dataset, DEFAULT_KAN_PARAMS, graph, factual_train_dn, factual_eval_d, load_best_params, verbose, paths.data, n_jobs, num_classes=num_classes)
                elif model_name == "dbcm":
                    params = get_best_hyperparams(model_name, dataset, DEFAULT_DBCM_PARAMS, graph, factual_train_dn, factual_eval_d, load_best_params, verbose, paths.data, n_jobs, num_classes=num_classes)
                elif model_name == "flow":
                    params = get_best_hyperparams(model_name, dataset, DEFAULT_FLOW_PARAMS, graph, factual_train_dn, factual_eval_d, load_best_params, verbose, paths.data, n_jobs, num_classes=num_classes)
                else:
                    params = {}

                if model_name in {"kan_mixed", "kaam_mixed"}:
                    model = kan_model_mixed(graph, deepcopy(params))
                    start = time.time()
                    model.fit(data=factual_train_d)
                    training_time = time.time() - start
                elif model_name == "flow":
                    model = causalflow_model(graph, deepcopy(params))
                    start = time.time()
                    model.fit(data=factual_train_dn)
                    training_time = time.time() - start
                else:
                    model = create_model_from_graph(graph, model_name, params)
                    if model_name == "anm":
                        gcm.auto.assign_causal_mechanisms(model, factual_train_dn, override_models=True, quality=AssignmentQuality.BETTER)
                    start = time.time()
                    gcm.fit(model, data=factual_train_dn)
                    training_time = time.time() - start

                results = evaluate_model(
                    model=model,
                    n=n,
                    factual_eval=factual_eval_d,
                    cf_eval=cf_eval_d,
                    dataset=dataset,
                    model_name=model_name,
                    r={"training_time": training_time, "model": model_name},
                    samples_dir=paths.samples,
                    images_dir=paths.images,
                    formula_gt=formula_gt,
                    kan_params=params,
                    verbose=verbose,
                    intervention=DEFAULT_INTERVENTIONS,
                    num_classes=num_classes,
                )
                with open(paths.data / f"{dataset}_results_n_{n}_{model_name}.pkl", "wb") as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the discrete synthetic benchmark.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where benchmark outputs will be written.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets to run.")
    parser.add_argument("--n-values", nargs="*", type=int, default=None, help="Sample sizes to evaluate.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to evaluate.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs used during hyperparameter search.")
    parser.add_argument("--recompute-best-params", action="store_true", help="Ignore cached best hyperparameters and recompute them.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main():
    args = build_parser().parse_args()
    run_discrete_benchmark(
        output_dir=args.output_dir,
        datasets=args.datasets,
        n_values=args.n_values,
        model_names=args.models,
        load_best_params=not args.recompute_best_params,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
