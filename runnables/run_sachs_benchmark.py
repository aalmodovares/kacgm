from __future__ import annotations

import argparse
import pickle
import time
from copy import deepcopy

DEFAULT_MODELS = ["kan", "kaam", "anm", "dbcm", "flow"]
DEFAULT_NOISE_SETTINGS = ["additive", "nonadditive"]
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


def build_interventions(factual_eval):
    import numpy as np

    interventions = []
    intervention_percentiles = {"PIP3": [25, 50, 75], "praf": [25, 50, 75], "pmek": [25, 50, 75]}
    for var, percentiles in intervention_percentiles.items():
        var_values = factual_eval[var].quantile(np.array(percentiles) / 100).values
        for value in var_values:
            interventions.append({var: (lambda _, fixed_value=float(value): fixed_value)})
    return interventions


def sample_sachs_dataset(n, seed, noise_setting):
    import numpy as np
    import torch

    from datasets.sachs import ExperimentationModel, get_structural_equations_sachs, graph_sachs, noises_distr

    np.random.seed(seed)
    torch.manual_seed(seed)
    structural_equations = get_structural_equations_sachs()
    generator = ExperimentationModel(graph_sachs, "sachs", structural_equations[noise_setting], noises_distr)
    factual_train, noise_train = generator.sample(n)
    factual_eval, noise_eval = generator.sample(n)
    interventions = build_interventions(factual_eval)
    cf_eval = [generator.get_counterfactuals(intervention, noise_eval) for intervention in interventions]
    return generator, factual_train, factual_eval, cf_eval, interventions


def resolve_best_params(paths, n, model_names, load_best_params, verbose, hp_jobs):
    from datasets.sachs import graph_sachs
    from utils.hyperparams import get_best_hyperparams

    best_params = {}
    _, factual_train_add, factual_eval_add, _, _ = sample_sachs_dataset(n=n, seed=0, noise_setting="additive")
    _, factual_train_nonadd, factual_eval_nonadd, _, _ = sample_sachs_dataset(n=n, seed=0, noise_setting="nonadditive")

    for model_name in model_names:
        if model_name in {"kan", "kaam"}:
            best_params[model_name] = get_best_hyperparams(model_name, "sachs", DEFAULT_KAN_PARAMS, graph_sachs, factual_train_add, factual_eval_add, load_best_params, verbose, paths.data, hp_jobs)
        elif model_name == "dbcm":
            best_params[model_name] = get_best_hyperparams(model_name, "sachs", DEFAULT_DBCM_PARAMS, graph_sachs, factual_train_add, factual_eval_add, load_best_params, verbose, paths.data, hp_jobs)
        elif model_name == "flow":
            best_params[("flow", "additive")] = get_best_hyperparams("flow", "sachs", DEFAULT_FLOW_PARAMS, graph_sachs, factual_train_add, factual_eval_add, load_best_params, verbose, paths.data, hp_jobs)
            best_params[("flow", "nonadditive")] = get_best_hyperparams("flow", "sachs_nonadditive", DEFAULT_FLOW_PARAMS, graph_sachs, factual_train_nonadd, factual_eval_nonadd, load_best_params, verbose, paths.data, hp_jobs)
    return best_params


def run_single_realization(paths, n, realization, noise_setting, model_names, params_lookup, verbose):
    from dowhy import gcm
    from dowhy.gcm.auto import AssignmentQuality

    from datasets.sachs import graph_sachs
    from models.factory import create_model_from_graph
    from models.flow import causalflow_model
    from utils.evaluation import evaluate_model

    _, factual_train, factual_eval, cf_eval, interventions = sample_sachs_dataset(n=n, seed=realization, noise_setting=noise_setting)
    dataset_name = f"sachs_real_{realization}_{noise_setting}"

    factual_train.to_csv(paths.samples / f"sachs_factual_train_n_{n}_{realization}_{noise_setting}.csv", index=False)
    factual_eval.to_csv(paths.samples / f"sachs_factual_eval_n_{n}_{realization}_{noise_setting}.csv", index=False)
    for ic, cf_frame in enumerate(cf_eval):
        cf_frame.to_csv(paths.samples / f"sachs_cf_eval_n_{n}_{ic}_{realization}_{noise_setting}.csv", index=False)

    for model_name in model_names:
        if model_name == "flow":
            params = deepcopy(params_lookup[("flow", noise_setting)])
            model = causalflow_model(graph_sachs, params)
            start = time.time()
            model.fit(data=factual_train)
            training_time = time.time() - start
        else:
            params = deepcopy(params_lookup.get(model_name, {}))
            model = create_model_from_graph(graph_sachs, model_name, params)
            if model_name == "anm":
                gcm.auto.assign_causal_mechanisms(model, factual_train, override_models=True, quality=AssignmentQuality.BETTER)
            start = time.time()
            gcm.fit(model, data=factual_train)
            training_time = time.time() - start

        results = evaluate_model(
            model=model,
            n=n,
            factual_eval=factual_eval,
            cf_eval=cf_eval,
            dataset=dataset_name,
            model_name=model_name,
            r={"training_time": training_time, "model": model_name},
            samples_dir=paths.samples,
            images_dir=paths.images,
            formula_gt=None,
            kan_params=params,
            verbose=verbose,
            intervention=interventions,
            seed=realization,
        )
        with open(paths.data / f"sachs_{noise_setting}_results_n_{n}_{model_name}_real_{realization}.pkl", "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_sachs_benchmark(output_dir=None, n=2000, n_realizations=10, model_names=None, noise_settings=None, load_best_params=True, verbose=False, hp_jobs=1, run_jobs=1):
    from joblib import Parallel, delayed
    from utils.paths import get_experiment_paths

    paths = get_experiment_paths("sachs", output_dir=output_dir)
    model_names = model_names or DEFAULT_MODELS
    noise_settings = noise_settings or DEFAULT_NOISE_SETTINGS

    params_lookup = resolve_best_params(paths, n, model_names, load_best_params, verbose, hp_jobs)
    Parallel(n_jobs=run_jobs, backend="loky")(
        delayed(run_single_realization)(paths, n, realization, noise_setting, model_names, params_lookup, verbose)
        for realization in range(n_realizations)
        for noise_setting in noise_settings
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run the Sachs benchmark.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where benchmark outputs will be written.")
    parser.add_argument("--n", type=int, default=2000, help="Number of train and evaluation samples per realization.")
    parser.add_argument("--n-realizations", type=int, default=10, help="Number of random realizations.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to evaluate.")
    parser.add_argument("--noise-settings", nargs="*", default=None, help="Subset of noise settings to evaluate.")
    parser.add_argument("--hp-jobs", type=int, default=1, help="Parallel jobs used during hyperparameter search.")
    parser.add_argument("--run-jobs", type=int, default=1, help="Parallel jobs used across realizations.")
    parser.add_argument("--recompute-best-params", action="store_true", help="Ignore cached best hyperparameters and recompute them.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main():
    args = build_parser().parse_args()
    run_sachs_benchmark(
        output_dir=args.output_dir,
        n=args.n,
        n_realizations=args.n_realizations,
        model_names=args.models,
        noise_settings=args.noise_settings,
        load_best_params=not args.recompute_best_params,
        verbose=args.verbose,
        hp_jobs=args.hp_jobs,
        run_jobs=args.run_jobs,
    )


if __name__ == "__main__":
    main()
