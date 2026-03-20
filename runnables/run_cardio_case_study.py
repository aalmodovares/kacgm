from __future__ import annotations

import argparse
import pickle


def run_cardio_case_study(output_dir=None, models=None, load_best_params=True, verbose=False, n_jobs=1, include_interventions=False):
    from datasets.cardio import get_cardio_graph, infer_node_types, prepare_cardio_data
    from utils.cardio import (
        CARDIO_BASE_MODELS,
        collect_cardio_interventional_cache,
        derive_kaam_mixed_variants,
        evaluate_cardio_observational,
        fit_cardio_model,
        resolve_cardio_best_params,
    )
    from utils.paths import get_experiment_paths

    paths = get_experiment_paths("cardio", output_dir=output_dir)
    model_names = list(models or CARDIO_BASE_MODELS)
    graph_cardio = get_cardio_graph()
    prepared = prepare_cardio_data()
    factual_train_d = prepared.factual_train_d
    factual_eval_d = prepared.factual_eval_d
    factual_train_dn = prepared.factual_train_dn
    node_info = prepared.node_info

    factual_train_d.to_csv(paths.data / "cardio_factual_train_d.csv", index=False)
    factual_eval_d.to_csv(paths.data / "cardio_factual_eval_d.csv", index=False)
    factual_train_dn.to_csv(paths.samples / "cardio_factual_train_dn.csv", index=False)

    _, num_classes = infer_node_types(factual_train_d, default_num_classes=2)
    best_params = resolve_cardio_best_params(
        model_names=model_names,
        graph_cardio=graph_cardio,
        factual_train_d=factual_train_d,
        factual_eval_d=factual_eval_d,
        factual_train_dn=factual_train_dn,
        load_best_params=load_best_params,
        verbose=verbose,
        data_dir=paths.data,
        n_jobs=n_jobs,
        num_classes=num_classes,
    )

    fitted_models = {}
    results = {}
    for model_name in model_names:
        checkpoint_root = paths.checkpoints / model_name / "seed_0"
        model, training_time, used_params = fit_cardio_model(
            model_name=model_name,
            graph_cardio=graph_cardio,
            params=best_params.get(model_name, {}),
            factual_train_d=factual_train_d,
            factual_train_dn=factual_train_dn,
            seed=0,
            checkpoint_root=checkpoint_root,
        )
        fitted_models[model_name] = model
        results[model_name] = {
            "training_time": training_time,
            "model": model_name,
            "seed": 0,
            "best_params": used_params,
        }

    if "kaam_mixed" in fitted_models:
        for variant_name, (variant_model, elapsed) in derive_kaam_mixed_variants(
            fitted_models["kaam_mixed"],
            factual_train_d,
            seed=0,
        ).items():
            fitted_models[variant_name] = variant_model
            results[variant_name] = {
                "training_time": elapsed,
                "model": variant_name,
                "seed": 0,
            }

    for model_name, model in fitted_models.items():
        metrics, obs_samples, residuals, extras = evaluate_cardio_observational(
            model_name=model_name,
            model=model,
            factual_eval_d=factual_eval_d,
            node_info=node_info,
            sample_seed=42,
        )
        results[model_name].update(metrics)

        obs_samples.to_csv(paths.samples / f"cardio_obs_samples_{model_name}.csv", index=False)
        with open(paths.samples / f"cardio_obs_samples_{model_name}.pkl", "wb") as handle:
            pickle.dump(obs_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(paths.samples / f"cardio_residuals_{model_name}.pkl", "wb") as handle:
            pickle.dump(residuals, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if "formulas" in extras:
            with open(paths.samples / "cardio_formulas_kaam.pkl", "wb") as handle:
                pickle.dump(extras["formulas"], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if include_interventions:
            interventional_cache = collect_cardio_interventional_cache(
                model_name=model_name,
                model=model,
                factual_eval_d=factual_eval_d,
                node_info=node_info,
                sample_seed=42,
            )
            with open(paths.data / f"cardio_interventional_eval_{model_name}.pkl", "wb") as handle:
                pickle.dump(interventional_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(paths.data / f"cardio_results_{model_name}.pkl", "wb") as handle:
            pickle.dump(results[model_name], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(paths.data / "cardio_results_all.pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the cardio case study.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where case study outputs will be written.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to evaluate.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs used during hyperparameter search.")
    parser.add_argument("--recompute-best-params", action="store_true", help="Ignore cached best hyperparameters and recompute them.")
    parser.add_argument("--include-interventions", action="store_true", help="Also generate the interventional sample cache used by the notebook.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main():
    args = build_parser().parse_args()
    run_cardio_case_study(
        output_dir=args.output_dir,
        models=args.models,
        load_best_params=not args.recompute_best_params,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        include_interventions=args.include_interventions,
    )


if __name__ == "__main__":
    main()
