from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path


def _resolve_sample_seed(seed, bootstrap_id, model_name):
    model_offset = sum(ord(char) for char in str(model_name)) % 10_000
    return int(1_000_000 + 10_000 * int(seed) + 100 * int(bootstrap_id) + model_offset)


def _resolve_cardio_best_params_source_dirs(output_dir=None):
    from utils.paths import OUTPUTS_ROOT, resolve_path

    candidate_dirs = []
    if output_dir is not None:
        resolved_output_dir = resolve_path(output_dir)
        candidate_dirs.append(resolved_output_dir.parent / "cardio" / "data")
    candidate_dirs.append(OUTPUTS_ROOT / "cardio" / "data")

    unique_dirs = []
    seen = set()
    for directory in candidate_dirs:
        directory = Path(directory)
        normalized = str(directory.resolve()) if directory.exists() else str(directory)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_dirs.append(directory)
    return unique_dirs


def _hydrate_cardio_best_params_cache(target_data_dir, output_dir=None):
    target_data_dir = Path(target_data_dir)
    copied_files = []
    for source_dir in _resolve_cardio_best_params_source_dirs(output_dir=output_dir):
        if not source_dir.exists():
            continue
        for source_file in source_dir.glob("best_params_*_cardio.pkl"):
            target_file = target_data_dir / source_file.name
            if target_file.exists():
                continue
            shutil.copy2(source_file, target_file)
            copied_files.append(target_file)
    return copied_files


def _format_value_token(value):
    token = f"{float(value):+.3f}"
    return token.replace("+", "pos").replace("-", "neg").replace(".", "p")


def _store_interventional_tables(sample_dir, interventional_samples):
    import pandas as pd

    manifest_rows = []
    for node, value, int_samples in interventional_samples:
        file_name = f"interventional__{node}__{_format_value_token(value)}.csv"
        int_samples.to_csv(sample_dir / file_name, index=False)
        manifest_rows.append(
            {
                "node": node,
                "value": float(value),
                "file_name": file_name,
                "num_rows": int(len(int_samples)),
            }
        )
    pd.DataFrame(
        manifest_rows,
        columns=["node", "value", "file_name", "num_rows"],
    ).to_csv(sample_dir / "interventional_manifest.csv", index=False)


def _store_formula_payload(sample_dir, formula_payload):
    if formula_payload is None:
        return

    metadata = {
        "model_name": formula_payload["model_name"],
        "base_model": formula_payload["base_model"],
        "variant": formula_payload["variant"],
        "seed": int(formula_payload["seed"]),
        "has_formulas": bool(formula_payload["has_formulas"]),
        "formula_status": formula_payload["formula_status"],
        "n_formulas": int(formula_payload["n_formulas"]),
    }
    (sample_dir / "formula_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with open(sample_dir / "formula_payload.pkl", "wb") as handle:
        pickle.dump(formula_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _store_sample_bundle(samples_root, model_name, seed, bootstrap_id, observational_samples, interventional_samples):
    sample_dir = Path(samples_root) / "by_bootstrap" / model_name / f"seed_{seed}" / f"bootstrap_{bootstrap_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    observational_samples.to_csv(sample_dir / "observational.csv", index=False)
    with open(sample_dir / "observational.pkl", "wb") as handle:
        pickle.dump(observational_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(sample_dir / "interventional.pkl", "wb") as handle:
        pickle.dump(interventional_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _store_interventional_tables(sample_dir, interventional_samples)


def _store_seed_sample_bundle(samples_root, model_name, seed, observational_samples, interventional_samples, formula_payload=None):
    sample_dir = Path(samples_root) / "by_seed" / model_name / f"seed_{seed}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    observational_samples.to_csv(sample_dir / "observational.csv", index=False)
    with open(sample_dir / "observational.pkl", "wb") as handle:
        pickle.dump(observational_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(sample_dir / "interventional.pkl", "wb") as handle:
        pickle.dump(interventional_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _store_interventional_tables(sample_dir, interventional_samples)
    _store_formula_payload(sample_dir, formula_payload)


def run_cardio_bootstrap_evaluation(
    output_dir=None,
    models=None,
    n_seeds=10,
    n_bootstraps=10,
    load_best_params=True,
    verbose=False,
    n_jobs=1,
    bootstrap_seed=2026,
    store_samples=False,
):
    import pandas as pd
    from joblib import Parallel, delayed

    from datasets.cardio import get_cardio_graph, infer_node_types, prepare_cardio_data
    from utils.cardio import (
        CARDIO_BASE_MODELS,
        build_bootstrap_indices,
        collect_cardio_interventional_cache,
        derive_kaam_mixed_variants,
        evaluate_cardio_observational,
        fit_cardio_model,
        metric_rows_from_result,
        resolve_cardio_best_params,
        save_bootstrap_indices,
        summarize_bootstrap_metrics,
    )
    from utils.paths import get_experiment_paths

    paths = get_experiment_paths("cardio_bootstrap", output_dir=output_dir)
    requested_models = list(models or CARDIO_BASE_MODELS)
    invalid_models = sorted(set(requested_models) - set(CARDIO_BASE_MODELS))
    if invalid_models:
        raise ValueError(f"Unsupported cardio bootstrap base models: {invalid_models}")

    graph_cardio = get_cardio_graph()
    prepared = prepare_cardio_data()
    factual_train_d = prepared.factual_train_d
    factual_eval_d = prepared.factual_eval_d
    factual_train_dn = prepared.factual_train_dn
    node_info = prepared.node_info

    factual_train_d.to_csv(paths.data / "cardio_bootstrap_train_d.csv", index=False)
    factual_eval_d.to_csv(paths.data / "cardio_bootstrap_eval_d.csv", index=False)
    factual_train_dn.to_csv(paths.samples / "cardio_bootstrap_train_dn.csv", index=False)
    factual_train_d.to_csv(paths.samples / "cardio_bootstrap_train_d.csv", index=False)
    factual_eval_d.to_csv(paths.samples / "cardio_bootstrap_eval_d.csv", index=False)
    _hydrate_cardio_best_params_cache(paths.data, output_dir=output_dir)

    _, num_classes = infer_node_types(factual_train_d, default_num_classes=2)
    best_params = resolve_cardio_best_params(
        model_names=requested_models,
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

    bootstrap_indices = build_bootstrap_indices(
        num_rows=len(factual_eval_d),
        n_bootstraps=n_bootstraps,
        bootstrap_seed=bootstrap_seed,
    )
    save_bootstrap_indices(bootstrap_indices, paths.data / "bootstrap_indices.csv")
    (paths.data / "bootstrap_config.json").write_text(
        json.dumps(
            {
                "n_seeds": int(n_seeds),
                "n_bootstraps": int(n_bootstraps),
                "bootstrap_seed": int(bootstrap_seed),
                "models": requested_models,
                "store_samples": bool(store_samples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def evaluate_base_model(model_name, seed):
        checkpoint_root = paths.checkpoints / model_name / f"seed_{seed}"
        model, training_time, used_params = fit_cardio_model(
            model_name=model_name,
            graph_cardio=graph_cardio,
            params=best_params.get(model_name, {}),
            factual_train_d=factual_train_d,
            factual_train_dn=factual_train_dn,
            seed=seed,
            checkpoint_root=checkpoint_root,
        )
        fitted_models = {model_name: (model, training_time)}
        if model_name == "kaam_mixed":
            fitted_models.update(derive_kaam_mixed_variants(model, factual_train_d, seed=seed))

        metric_rows = []
        fit_rows = []
        formula_payloads = []
        for eval_model_name, model_or_pair in fitted_models.items():
            if isinstance(model_or_pair, tuple):
                eval_model, elapsed = model_or_pair
            else:
                eval_model, elapsed = model_or_pair, training_time
            base_model = eval_model_name.replace("_pruned", "").replace("_symbolic", "")
            variant = "Pruned" if eval_model_name.endswith("_pruned") else "Symbolic" if eval_model_name.endswith("_symbolic") else "Original"
            fit_rows.append(
                {
                    "model_name": eval_model_name,
                    "base_model": base_model,
                    "variant": variant,
                    "seed": int(seed),
                    "training_time": float(elapsed),
                }
            )
            formula_payload = None
            if store_samples or eval_model_name in {"kaam_mixed_pruned", "kaam_mixed_symbolic"}:
                seed_sample_seed = _resolve_sample_seed(seed, -1, eval_model_name)
                _, seed_observational_samples, _, seed_extras = evaluate_cardio_observational(
                    model_name=eval_model_name,
                    model=eval_model,
                    factual_eval_d=factual_eval_d,
                    node_info=node_info,
                    sample_seed=seed_sample_seed,
                )
                if eval_model_name in {"kaam_mixed_pruned", "kaam_mixed_symbolic"}:
                    formulas = seed_extras.get("formulas")
                    formula_payload = {
                        "model_name": eval_model_name,
                        "base_model": base_model,
                        "variant": variant,
                        "seed": int(seed),
                        "has_formulas": formulas is not None,
                        "formula_status": "symbolic_extracted" if formulas is not None else "not_symbolic",
                        "n_formulas": 0 if formulas is None else len(formulas),
                        "formulas": formulas,
                    }
                    formula_payloads.append(formula_payload)

                if store_samples:
                    seed_interventional_samples = collect_cardio_interventional_cache(
                        model_name=eval_model_name,
                        model=eval_model,
                        factual_eval_d=factual_eval_d,
                        node_info=node_info,
                        sample_seed=seed_sample_seed,
                    )
                    _store_seed_sample_bundle(
                        paths.samples,
                        eval_model_name,
                        seed,
                        seed_observational_samples,
                        seed_interventional_samples,
                        formula_payload=formula_payload,
                    )

            for bootstrap_id, sampled_indices in enumerate(bootstrap_indices):
                bootstrap_eval = factual_eval_d.iloc[sampled_indices].reset_index(drop=True)
                sample_seed = _resolve_sample_seed(seed, bootstrap_id, eval_model_name)
                metrics, observational_samples, _, extras = evaluate_cardio_observational(
                    model_name=eval_model_name,
                    model=eval_model,
                    factual_eval_d=bootstrap_eval,
                    node_info=node_info,
                    sample_seed=sample_seed,
                )
                if store_samples:
                    interventional_samples = collect_cardio_interventional_cache(
                        model_name=eval_model_name,
                        model=eval_model,
                        factual_eval_d=bootstrap_eval,
                        node_info=node_info,
                        sample_seed=sample_seed,
                    )
                    _store_sample_bundle(
                        paths.samples,
                        eval_model_name,
                        seed,
                        bootstrap_id,
                        observational_samples,
                        interventional_samples,
                    )

                metric_rows.extend(metric_rows_from_result(eval_model_name, seed, bootstrap_id, metrics))

        return metric_rows, fit_rows, {"model_name": model_name, "seed": int(seed), "best_params": used_params}, formula_payloads

    tasks = [(model_name, seed) for model_name in requested_models for seed in range(int(n_seeds))]
    task_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(evaluate_base_model)(model_name, seed) for model_name, seed in tasks
    )

    metric_rows = [row for rows, _, _, _ in task_results for row in rows]
    fit_rows = [row for _, rows, _, _ in task_results for row in rows]
    param_rows = [row for _, _, row, _ in task_results]
    formula_payloads = [row for _, _, _, rows in task_results for row in rows]

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["metric", "model_name", "seed", "bootstrap"]
    ).reset_index(drop=True)
    fit_df = pd.DataFrame(fit_rows).sort_values(["model_name", "seed"]).reset_index(drop=True)
    summary_df = summarize_bootstrap_metrics(metrics_df)
    training_summary_df = (
        fit_df.groupby(["model_name", "base_model", "variant"], dropna=False)["training_time"]
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
        .sort_values(["model_name"])
        .reset_index(drop=True)
    )

    metrics_df.to_csv(paths.data / "cardio_bootstrap_metrics_long.csv", index=False)
    summary_df.to_csv(paths.data / "cardio_bootstrap_metrics_summary.csv", index=False)
    fit_df.to_csv(paths.data / "cardio_bootstrap_training_long.csv", index=False)
    training_summary_df.to_csv(paths.data / "cardio_bootstrap_training_summary.csv", index=False)
    with open(paths.data / "cardio_bootstrap_best_params.json", "w", encoding="utf-8") as handle:
        json.dump(param_rows, handle, indent=2, default=str)
    with open(paths.data / "cardio_bootstrap_symbolic_formulas.pkl", "wb") as handle:
        pickle.dump(formula_payloads, handle, protocol=pickle.HIGHEST_PROTOCOL)
    formula_metadata_columns = [
        "model_name",
        "base_model",
        "variant",
        "seed",
        "has_formulas",
        "formula_status",
        "n_formulas",
    ]
    formula_metadata = pd.DataFrame(
        [
            {
                "model_name": row["model_name"],
                "base_model": row["base_model"],
                "variant": row["variant"],
                "seed": row["seed"],
                "has_formulas": row["has_formulas"],
                "formula_status": row["formula_status"],
                "n_formulas": row["n_formulas"],
            }
            for row in formula_payloads
        ],
        columns=formula_metadata_columns,
    )
    formula_metadata.to_csv(paths.data / "cardio_bootstrap_symbolic_formulas_metadata.csv", index=False)
    formula_rows = []
    for row in formula_payloads:
        if not row["has_formulas"]:
            continue
        for node, formula in row["formulas"].items():
            formula_rows.append(
                {
                    "model_name": row["model_name"],
                    "base_model": row["base_model"],
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "node": node,
                    "formula": str(formula),
                }
            )
    pd.DataFrame(
        formula_rows,
        columns=["model_name", "base_model", "variant", "seed", "node", "formula"],
    ).to_csv(paths.data / "cardio_bootstrap_symbolic_formulas.csv", index=False)


def build_parser():
    parser = argparse.ArgumentParser(description="Run repeated bootstrap evaluation for the cardio case study.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where repeated cardio outputs will be written.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of base cardio models to evaluate.")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of training seeds per model.")
    parser.add_argument("--n-bootstraps", type=int, default=10, help="Number of bootstrap evaluations per seed.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs across model-seed tasks.")
    parser.add_argument("--bootstrap-seed", type=int, default=2026, help="Seed used to generate shared bootstrap resamples.")
    parser.add_argument("--store-samples", action="store_true", help="Also store observational and interventional samples for each model, seed, and bootstrap.")
    parser.add_argument("--recompute-best-params", action="store_true", help="Ignore cached best hyperparameters and recompute them.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main():
    args = build_parser().parse_args()
    run_cardio_bootstrap_evaluation(
        output_dir=args.output_dir,
        models=args.models,
        n_seeds=args.n_seeds,
        n_bootstraps=args.n_bootstraps,
        load_best_params=not args.recompute_best_params,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        bootstrap_seed=args.bootstrap_seed,
        store_samples=args.store_samples,
    )


if __name__ == "__main__":
    main()
