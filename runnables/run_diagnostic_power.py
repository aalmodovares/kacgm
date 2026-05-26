from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter

from runnables.run_sensitivity import DEFAULT_FLOW_PARAMS, DEFAULT_KAN_PARAMS


DEFAULT_EXPERIMENT_NAME = "diagnostic_power"
DEFAULT_DATASET = "triangle-sensitivity-nonlinear"
DEFAULT_ALPHA_GRID = (0.0, 0.5, 1.0)
DEFAULT_N_GRID = (10, 50, 100, 500, 1000, 3000, 6000)
DEFAULT_N_SEEDS = 100
DEFAULT_BASE_SEED = 0
DEFAULT_EFFECT_NODE = "x2"
DEFAULT_OUTCOME_NODE = "x3"
DEFAULT_ATE_LOW = 0.0
DEFAULT_ATE_HIGH = 1.0
DEFAULT_NUM_PERMUTATIONS = 99
DEFAULT_RF_ESTIMATORS = 100
DEFAULT_PVALUE_METHOD = "permutation"
DEFAULT_DIAGNOSTIC_ALPHA = 0.05
DEFAULT_PERMUTATION_N_JOBS = 0
EXPECTED_MODEL_NAMES = ("kan", "flow", "oracle")


def _alpha_slug(alpha_value):
    return str(alpha_value).replace("-", "m").replace(".", "p")


def _task_result_path(task_dir, alpha_value, n, seed):
    return Path(task_dir) / f"alpha_{_alpha_slug(alpha_value)}_n_{int(n)}_seed_{int(seed)}.csv"


def _make_intervention(node, value):
    return {node: (lambda _, fixed_value=float(value): fixed_value)}


def _generate_task_data(dataset, n, alpha_value, seed):
    from datasets.synthetic import graph_data

    data_all, data_cf_all, graph, _, u_all = graph_data(name=dataset).generate(
        num_samples=2 * int(n),
        seed=int(seed),
        alpha=float(alpha_value),
        return_u=True,
    )
    factual_train = data_all.iloc[:n].reset_index(drop=True)
    factual_eval = data_all.iloc[n : 2 * n].reset_index(drop=True)
    cf_eval = [frame.iloc[n : 2 * n].reset_index(drop=True) for frame in data_cf_all]
    u_eval = u_all.iloc[n : 2 * n].reset_index(drop=True)

    oracle_obs, _, _, _ = graph_data(name=dataset).generate(
        num_samples=int(n),
        seed=int(seed) + 500_000,
        alpha=float(alpha_value),
    )
    oracle_obs = oracle_obs.reset_index(drop=True)
    return factual_train, factual_eval, cf_eval, graph, u_eval, oracle_obs


def _fit_kan_model(graph, factual_train, kan_params, fit_seed, checkpoint_root):
    from dowhy import gcm

    from models.factory import create_model_from_graph
    from utils.cardio import clone_model_params_for_seed, set_global_seed

    set_global_seed(int(fit_seed))
    params = clone_model_params_for_seed("kan", kan_params, int(fit_seed), checkpoint_root=checkpoint_root)
    model = create_model_from_graph(graph, "kan", deepcopy(params))
    gcm.fit(model, data=factual_train)
    return model, params


def _fit_flow_model(graph, factual_train, flow_params, fit_seed):
    from models.flow import causalflow_model
    from utils.cardio import set_global_seed

    set_global_seed(int(fit_seed))
    model = causalflow_model(graph, deepcopy(flow_params))
    model.fit(data=factual_train)
    return model, deepcopy(flow_params)


def _draw_observational_samples(model_name, model, num_samples, sample_seed):
    from dowhy import gcm

    from utils.cardio import set_global_seed

    set_global_seed(int(sample_seed))
    if model_name == "flow":
        return model.draw_samples(num_samples=num_samples, seed=int(sample_seed))
    return gcm.draw_samples(model, num_samples=num_samples)


def _draw_counterfactual_samples(model_name, model, intervention, factual_eval, sample_seed):
    from dowhy import gcm

    from utils.cardio import set_global_seed

    set_global_seed(int(sample_seed))
    if model_name == "flow":
        return model.counterfactual_samples(intervention, factual_samples=factual_eval, seed=int(sample_seed))
    return gcm.counterfactual_samples(model, intervention, observed_data=factual_eval.copy())


def _get_model_residuals(model_name, model, factual_eval):
    from utils.cardio import get_residuals_anm, get_residuals_flow

    if model_name == "flow":
        return get_residuals_flow(model, factual_eval)
    return get_residuals_anm(model, factual_eval)


def _evaluate_model(
    model_name,
    model,
    dataset,
    alpha_value,
    n,
    seed,
    factual_eval,
    cf_eval,
    graph,
    pvalue_method,
    num_permutations,
    permutation_n_jobs,
    diagnostic_alpha,
    rf_estimators,
    fit_seed,
    sample_seed,
):
    import networkx as nx
    import numpy as np
    from dowhy.graph import get_ordered_predecessors

    from utils.diagnostics import c2st_random_forest_test, dhsic_test, holm_adjust_pvalues, hsic_test

    observational_samples = _draw_observational_samples(model_name, model, int(n), int(sample_seed))
    observational_samples = observational_samples[factual_eval.columns]
    c2st_result = c2st_random_forest_test(
        real_samples=factual_eval.to_numpy(),
        generated_samples=observational_samples.to_numpy(),
        seed=int(seed),
        n_estimators=int(rf_estimators),
    )

    intervention_zero = _make_intervention(DEFAULT_EFFECT_NODE, DEFAULT_ATE_LOW)
    intervention_one = _make_intervention(DEFAULT_EFFECT_NODE, DEFAULT_ATE_HIGH)
    cf_pred_zero = _draw_counterfactual_samples(model_name, model, intervention_zero, factual_eval, int(sample_seed) + 11)
    cf_pred_one = _draw_counterfactual_samples(model_name, model, intervention_one, factual_eval, int(sample_seed) + 29)
    cf_pred_zero = cf_pred_zero[factual_eval.columns]
    cf_pred_one = cf_pred_one[factual_eval.columns]

    cf_true_zero = cf_eval[4]
    cf_true_one = cf_eval[5]
    ate_true = float(cf_true_one[DEFAULT_OUTCOME_NODE].mean() - cf_true_zero[DEFAULT_OUTCOME_NODE].mean())
    ate_pred = float(cf_pred_one[DEFAULT_OUTCOME_NODE].mean() - cf_pred_zero[DEFAULT_OUTCOME_NODE].mean())
    cf_errors = np.concatenate(
        [
            cf_pred_zero[DEFAULT_OUTCOME_NODE].to_numpy() - cf_true_zero[DEFAULT_OUTCOME_NODE].to_numpy(),
            cf_pred_one[DEFAULT_OUTCOME_NODE].to_numpy() - cf_true_one[DEFAULT_OUTCOME_NODE].to_numpy(),
        ]
    )

    residuals = _get_model_residuals(model_name, model, factual_eval)
    node_order = list(nx.topological_sort(graph))
    hsic_parents = get_ordered_predecessors(graph, DEFAULT_OUTCOME_NODE)
    hsic_result = hsic_test(
        residuals[DEFAULT_OUTCOME_NODE],
        factual_eval[hsic_parents].to_numpy(),
        method=pvalue_method,
        num_permutations=int(num_permutations),
        seed=int(seed) + 101,
        n_jobs=int(permutation_n_jobs),
    )
    dhsic_result = dhsic_test(
        *[residuals[node] for node in node_order],
        method=pvalue_method,
        num_permutations=int(num_permutations),
        seed=int(seed) + 202,
        n_jobs=int(permutation_n_jobs),
    )

    raw_pvalues = np.array(
        [
            hsic_result["pvalue"],
            dhsic_result["pvalue"],
            c2st_result["pvalue"],
        ],
        dtype=float,
    )
    holm_reject, adjusted_pvalues = holm_adjust_pvalues(raw_pvalues, alpha=float(diagnostic_alpha))
    reject_hsic3 = int(hsic_result["pvalue"] <= float(diagnostic_alpha))
    reject_dhsic = int(dhsic_result["pvalue"] <= float(diagnostic_alpha))
    reject_c2st = int(c2st_result["pvalue"] <= float(diagnostic_alpha))
    reject_global = int(float(np.min(adjusted_pvalues)) <= float(diagnostic_alpha))

    return {
        "dataset": str(dataset),
        "alpha": float(alpha_value),
        "n": int(n),
        "seed": int(seed),
        "model_name": str(model_name),
        "data_seed": int(seed),
        "fit_seed": int(fit_seed),
        "sample_seed": int(sample_seed),
        "validation_size": int(len(factual_eval)),
        "pvalue_alpha": float(diagnostic_alpha),
        "pvalue_method": str(pvalue_method),
        "num_permutations": int(num_permutations),
        "permutation_n_jobs": int(permutation_n_jobs),
        "rf_n_estimators": int(rf_estimators),
        "c2st_train_fraction": 0.5,
        "c2st_test_fraction": 0.5,
        "hsic_target_node": DEFAULT_OUTCOME_NODE,
        "hsic_parent_nodes": "|".join(hsic_parents),
        "hsic_x3_stat": float(hsic_result["statistic"]),
        "hsic_x3_pvalue": float(hsic_result["pvalue"]),
        "hsic_x3_pvalue_method": str(hsic_result["method"]),
        "dhsic_nodes": "|".join(node_order),
        "dhsic_stat": float(dhsic_result["statistic"]),
        "dhsic_pvalue": float(dhsic_result["pvalue"]),
        "dhsic_pvalue_method": str(dhsic_result["method"]),
        "c2st_accuracy": float(c2st_result["accuracy"]),
        "c2st_pvalue": float(c2st_result["pvalue"]),
        "c2st_n_te": int(c2st_result["n_te"]),
        "c2st_k_correct": int(c2st_result["k_correct"]),
        "c2st_pvalue_method": str(c2st_result["method"]),
        "holm_hsic3_pvalue": float(adjusted_pvalues[0]),
        "holm_dhsic_pvalue": float(adjusted_pvalues[1]),
        "holm_c2st_pvalue": float(adjusted_pvalues[2]),
        "holm_reject_hsic3": int(holm_reject[0]),
        "holm_reject_dhsic": int(holm_reject[1]),
        "holm_reject_c2st": int(holm_reject[2]),
        "reject_hsic3": reject_hsic3,
        "reject_dhsic": reject_dhsic,
        "reject_c2st": reject_c2st,
        "reject_global": reject_global,
        "ate_intervened_node": DEFAULT_EFFECT_NODE,
        "ate_outcome_node": DEFAULT_OUTCOME_NODE,
        "ate_value_low": float(DEFAULT_ATE_LOW),
        "ate_value_high": float(DEFAULT_ATE_HIGH),
        "ate_error": float(abs(ate_pred - ate_true)),
        "cf_mae_intervened_node": DEFAULT_EFFECT_NODE,
        "cf_mae_outcome_node": DEFAULT_OUTCOME_NODE,
        "cf_mae_intervention_values": f"{DEFAULT_ATE_LOW}|{DEFAULT_ATE_HIGH}",
        "cf_mae": float(np.mean(np.abs(cf_errors))),
    }


def _evaluate_oracle(
    dataset,
    alpha_value,
    n,
    seed,
    factual_eval,
    cf_eval,
    graph,
    u_eval,
    oracle_obs,
    pvalue_method,
    num_permutations,
    permutation_n_jobs,
    diagnostic_alpha,
    rf_estimators,
):
    import networkx as nx
    import numpy as np
    from dowhy.graph import get_ordered_predecessors

    from utils.diagnostics import c2st_random_forest_test, dhsic_test, holm_adjust_pvalues, hsic_test

    node_order = list(nx.topological_sort(graph))
    hsic_parents = get_ordered_predecessors(graph, DEFAULT_OUTCOME_NODE)
    c2st_result = c2st_random_forest_test(
        real_samples=factual_eval.to_numpy(),
        generated_samples=oracle_obs[factual_eval.columns].to_numpy(),
        seed=int(seed),
        n_estimators=int(rf_estimators),
    )
    hsic_result = hsic_test(
        u_eval["u3"].to_numpy(),
        factual_eval[hsic_parents].to_numpy(),
        method=pvalue_method,
        num_permutations=int(num_permutations),
        seed=int(seed) + 303,
        n_jobs=int(permutation_n_jobs),
    )
    dhsic_result = dhsic_test(
        *[u_eval[f"u{index + 1}"].to_numpy() for index in range(len(node_order))],
        method=pvalue_method,
        num_permutations=int(num_permutations),
        seed=int(seed) + 404,
        n_jobs=int(permutation_n_jobs),
    )

    raw_pvalues = np.array(
        [
            hsic_result["pvalue"],
            dhsic_result["pvalue"],
            c2st_result["pvalue"],
        ],
        dtype=float,
    )
    holm_reject, adjusted_pvalues = holm_adjust_pvalues(raw_pvalues, alpha=float(diagnostic_alpha))

    return {
        "dataset": str(dataset),
        "alpha": float(alpha_value),
        "n": int(n),
        "seed": int(seed),
        "model_name": "oracle",
        "data_seed": int(seed),
        "fit_seed": int(seed),
        "sample_seed": int(seed) + 500_000,
        "validation_size": int(len(factual_eval)),
        "pvalue_alpha": float(diagnostic_alpha),
        "pvalue_method": str(pvalue_method),
        "num_permutations": int(num_permutations),
        "permutation_n_jobs": int(permutation_n_jobs),
        "rf_n_estimators": int(rf_estimators),
        "c2st_train_fraction": 0.5,
        "c2st_test_fraction": 0.5,
        "hsic_target_node": DEFAULT_OUTCOME_NODE,
        "hsic_parent_nodes": "|".join(hsic_parents),
        "hsic_x3_stat": float(hsic_result["statistic"]),
        "hsic_x3_pvalue": float(hsic_result["pvalue"]),
        "hsic_x3_pvalue_method": str(hsic_result["method"]),
        "dhsic_nodes": "|".join(node_order),
        "dhsic_stat": float(dhsic_result["statistic"]),
        "dhsic_pvalue": float(dhsic_result["pvalue"]),
        "dhsic_pvalue_method": str(dhsic_result["method"]),
        "c2st_accuracy": float(c2st_result["accuracy"]),
        "c2st_pvalue": float(c2st_result["pvalue"]),
        "c2st_n_te": int(c2st_result["n_te"]),
        "c2st_k_correct": int(c2st_result["k_correct"]),
        "c2st_pvalue_method": str(c2st_result["method"]),
        "holm_hsic3_pvalue": float(adjusted_pvalues[0]),
        "holm_dhsic_pvalue": float(adjusted_pvalues[1]),
        "holm_c2st_pvalue": float(adjusted_pvalues[2]),
        "holm_reject_hsic3": int(holm_reject[0]),
        "holm_reject_dhsic": int(holm_reject[1]),
        "holm_reject_c2st": int(holm_reject[2]),
        "reject_hsic3": int(hsic_result["pvalue"] <= float(diagnostic_alpha)),
        "reject_dhsic": int(dhsic_result["pvalue"] <= float(diagnostic_alpha)),
        "reject_c2st": int(c2st_result["pvalue"] <= float(diagnostic_alpha)),
        "reject_global": int(float(np.min(adjusted_pvalues)) <= float(diagnostic_alpha)),
        "ate_intervened_node": DEFAULT_EFFECT_NODE,
        "ate_outcome_node": DEFAULT_OUTCOME_NODE,
        "ate_value_low": float(DEFAULT_ATE_LOW),
        "ate_value_high": float(DEFAULT_ATE_HIGH),
        "ate_error": 0.0,
        "cf_mae_intervened_node": DEFAULT_EFFECT_NODE,
        "cf_mae_outcome_node": DEFAULT_OUTCOME_NODE,
        "cf_mae_intervention_values": f"{DEFAULT_ATE_LOW}|{DEFAULT_ATE_HIGH}",
        "cf_mae": 0.0,
    }


def _evaluate_task(
    alpha_value,
    n,
    seed,
    dataset,
    kan_params,
    flow_params,
    task_checkpoint_root,
    pvalue_method,
    num_permutations,
    permutation_n_jobs,
    diagnostic_alpha,
    rf_estimators,
):
    factual_train, factual_eval, cf_eval, graph, u_eval, oracle_obs = _generate_task_data(
        dataset,
        int(n),
        float(alpha_value),
        int(seed),
    )
    task_checkpoint_root = Path(task_checkpoint_root)
    kan_model, _ = _fit_kan_model(
        graph,
        factual_train,
        deepcopy(kan_params),
        fit_seed=int(seed) + 11,
        checkpoint_root=task_checkpoint_root / "kan",
    )
    flow_model, _ = _fit_flow_model(
        graph,
        factual_train,
        deepcopy(flow_params),
        fit_seed=int(seed) + 23,
    )

    rows = [
        _evaluate_model(
            model_name="kan",
            model=kan_model,
            dataset=dataset,
            alpha_value=alpha_value,
            n=n,
            seed=seed,
            factual_eval=factual_eval,
            cf_eval=cf_eval,
            graph=graph,
            pvalue_method=pvalue_method,
            num_permutations=num_permutations,
            permutation_n_jobs=permutation_n_jobs,
            diagnostic_alpha=diagnostic_alpha,
            rf_estimators=rf_estimators,
            fit_seed=int(seed) + 11,
            sample_seed=int(seed) + 1_011,
        ),
        _evaluate_model(
            model_name="flow",
            model=flow_model,
            dataset=dataset,
            alpha_value=alpha_value,
            n=n,
            seed=seed,
            factual_eval=factual_eval,
            cf_eval=cf_eval,
            graph=graph,
            pvalue_method=pvalue_method,
            num_permutations=num_permutations,
            permutation_n_jobs=permutation_n_jobs,
            diagnostic_alpha=diagnostic_alpha,
            rf_estimators=rf_estimators,
            fit_seed=int(seed) + 23,
            sample_seed=int(seed) + 2_023,
        ),
        _evaluate_oracle(
            dataset=dataset,
            alpha_value=alpha_value,
            n=n,
            seed=seed,
            factual_eval=factual_eval,
            cf_eval=cf_eval,
            graph=graph,
            u_eval=u_eval,
            oracle_obs=oracle_obs,
            pvalue_method=pvalue_method,
            num_permutations=num_permutations,
            permutation_n_jobs=permutation_n_jobs,
            diagnostic_alpha=diagnostic_alpha,
            rf_estimators=rf_estimators,
        ),
    ]
    return rows


def _backfill_oracle_row(
    existing_rows_df,
    alpha_value,
    n,
    seed,
    dataset,
    pvalue_method,
    num_permutations,
    permutation_n_jobs,
    diagnostic_alpha,
    rf_estimators,
):
    import pandas as pd

    _, factual_eval, cf_eval, graph, u_eval, oracle_obs = _generate_task_data(
        dataset,
        int(n),
        float(alpha_value),
        int(seed),
    )
    oracle_row = _evaluate_oracle(
        dataset=dataset,
        alpha_value=alpha_value,
        n=n,
        seed=seed,
        factual_eval=factual_eval,
        cf_eval=cf_eval,
        graph=graph,
        u_eval=u_eval,
        oracle_obs=oracle_obs,
        pvalue_method=pvalue_method,
        num_permutations=num_permutations,
        permutation_n_jobs=permutation_n_jobs,
        diagnostic_alpha=diagnostic_alpha,
        rf_estimators=rf_estimators,
    )
    combined_df = pd.concat([existing_rows_df, pd.DataFrame([oracle_row])], ignore_index=True, sort=False)
    combined_df = combined_df.sort_values(["alpha", "n", "seed", "model_name"]).reset_index(drop=True)
    return combined_df


def _load_or_run_task(
    alpha_value,
    n,
    seed,
    dataset,
    kan_params,
    flow_params,
    task_dir,
    pvalue_method,
    num_permutations,
    permutation_n_jobs,
    diagnostic_alpha,
    rf_estimators,
    overwrite=False,
):
    import pandas as pd

    task_path = _task_result_path(task_dir, alpha_value, n, seed)
    if task_path.exists() and not overwrite:
        cached_df = pd.read_csv(task_path)
        cached_models = set(cached_df["model_name"].dropna().astype(str).tolist()) if "model_name" in cached_df else set()
        missing_models = set(EXPECTED_MODEL_NAMES) - cached_models
        if not missing_models:
            return cached_df.to_dict(orient="records")
        if missing_models == {"oracle"}:
            backfilled_df = _backfill_oracle_row(
                existing_rows_df=cached_df,
                alpha_value=alpha_value,
                n=n,
                seed=seed,
                dataset=dataset,
                pvalue_method=pvalue_method,
                num_permutations=num_permutations,
                permutation_n_jobs=permutation_n_jobs,
                diagnostic_alpha=diagnostic_alpha,
                rf_estimators=rf_estimators,
            )
            tmp_path = task_path.with_suffix(".tmp")
            backfilled_df.to_csv(tmp_path, index=False)
            tmp_path.replace(task_path)
            return backfilled_df.to_dict(orient="records")

    rows = _evaluate_task(
        alpha_value=alpha_value,
        n=n,
        seed=seed,
        dataset=dataset,
        kan_params=kan_params,
        flow_params=flow_params,
        task_checkpoint_root=Path(task_dir).parent / "model_states" / task_path.stem,
        pvalue_method=pvalue_method,
        num_permutations=num_permutations,
        permutation_n_jobs=permutation_n_jobs,
        diagnostic_alpha=diagnostic_alpha,
        rf_estimators=rf_estimators,
    )
    df = pd.DataFrame(rows)
    tmp_path = task_path.with_suffix(".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(task_path)
    return rows

def _empirical_upper_tail_pvalue(observed_stat, null_stats):
    import numpy as np

    null_stats = np.asarray(null_stats, dtype=float)
    null_stats = null_stats[np.isfinite(null_stats)]
    if null_stats.size == 0:
        return float("nan")
    return float((1.0 + np.sum(null_stats >= float(observed_stat))) / (null_stats.size + 1.0))


def _empirical_upper_threshold(null_stats, alpha=0.05):
    import numpy as np

    null_stats = np.asarray(null_stats, dtype=float)
    null_stats = null_stats[np.isfinite(null_stats)]
    if null_stats.size == 0:
        return float("nan")

    sorted_stats = np.sort(null_stats)
    # Empirical (1-alpha)-quantile, using an upper/order-statistic convention.
    rank = int(np.ceil((1.0 - float(alpha)) * len(sorted_stats))) - 1
    rank = int(np.clip(rank, 0, len(sorted_stats) - 1))
    return float(sorted_stats[rank])


def _add_calibrated_diagnostics(results_df, diagnostic_alpha=0.05):
    """
    Adds calibrated p-values/rejection indicators using alpha=0 as the
    fitted-model baseline, separately for each (model_name, n).

    Uncalibrated columns are kept unchanged.

    Calibration rule:
      - For each diagnostic statistic T, model m, and sample size n,
        the alpha=0 runs define the empirical null distribution.
      - For alpha=0 rows, use leave-one-out calibration.
      - For alpha>0 rows, compare against all alpha=0 runs.
      - Larger statistics are treated as stronger evidence of misspecification.
    """
    import numpy as np

    from utils.diagnostics import holm_adjust_pvalues

    df = results_df.copy()

    diagnostic_specs = [
        ("hsic3", "hsic_x3_stat"),
        ("dhsic", "dhsic_stat"),
        ("c2st", "c2st_accuracy"),
    ]

    for diagnostic_name, _ in diagnostic_specs:
        df[f"calibrated_{diagnostic_name}_pvalue"] = np.nan
        df[f"calibrated_{diagnostic_name}_threshold"] = np.nan
        df[f"calibrated_{diagnostic_name}_n_null"] = 0
        df[f"reject_calibrated_{diagnostic_name}"] = 0

    df["holm_calibrated_hsic3_pvalue"] = np.nan
    df["holm_calibrated_dhsic_pvalue"] = np.nan
    df["holm_calibrated_c2st_pvalue"] = np.nan
    df["holm_reject_calibrated_hsic3"] = 0
    df["holm_reject_calibrated_dhsic"] = 0
    df["holm_reject_calibrated_c2st"] = 0
    df["reject_calibrated_global"] = 0

    for (model_name, n), group in df.groupby(["model_name", "n"], sort=False):
        group_index = group.index
        alpha0_index = group_index[np.isclose(df.loc[group_index, "alpha"].to_numpy(dtype=float), 0.0)]

        if len(alpha0_index) == 0:
            raise ValueError(
                f"Calibration requires alpha=0 rows for model_name={model_name}, n={n}."
            )

        for diagnostic_name, stat_col in diagnostic_specs:
            null_stats_all = df.loc[alpha0_index, stat_col].to_numpy(dtype=float)
            threshold = _empirical_upper_threshold(null_stats_all, alpha=float(diagnostic_alpha))

            for row_index in group_index:
                observed_stat = float(df.at[row_index, stat_col])

                if np.isclose(float(df.at[row_index, "alpha"]), 0.0):
                    # Leave-one-out calibration for alpha=0, to avoid calibrating
                    # a run against itself.
                    null_index = [idx for idx in alpha0_index if idx != row_index]
                    null_stats = df.loc[null_index, stat_col].to_numpy(dtype=float)
                else:
                    null_stats = null_stats_all

                calibrated_pvalue = _empirical_upper_tail_pvalue(observed_stat, null_stats)

                df.at[row_index, f"calibrated_{diagnostic_name}_pvalue"] = calibrated_pvalue
                df.at[row_index, f"calibrated_{diagnostic_name}_threshold"] = threshold
                df.at[row_index, f"calibrated_{diagnostic_name}_n_null"] = int(len(null_stats))
                df.at[row_index, f"reject_calibrated_{diagnostic_name}"] = int(
                    calibrated_pvalue <= float(diagnostic_alpha)
                )

    for row_index in df.index:
        calibrated_pvalues = np.array(
            [
                df.at[row_index, "calibrated_hsic3_pvalue"],
                df.at[row_index, "calibrated_dhsic_pvalue"],
                df.at[row_index, "calibrated_c2st_pvalue"],
            ],
            dtype=float,
        )

        if np.any(~np.isfinite(calibrated_pvalues)):
            continue

        holm_reject, adjusted_pvalues = holm_adjust_pvalues(
            calibrated_pvalues,
            alpha=float(diagnostic_alpha),
        )

        df.at[row_index, "holm_calibrated_hsic3_pvalue"] = float(adjusted_pvalues[0])
        df.at[row_index, "holm_calibrated_dhsic_pvalue"] = float(adjusted_pvalues[1])
        df.at[row_index, "holm_calibrated_c2st_pvalue"] = float(adjusted_pvalues[2])
        df.at[row_index, "holm_reject_calibrated_hsic3"] = int(holm_reject[0])
        df.at[row_index, "holm_reject_calibrated_dhsic"] = int(holm_reject[1])
        df.at[row_index, "holm_reject_calibrated_c2st"] = int(holm_reject[2])
        df.at[row_index, "reject_calibrated_global"] = int(
            float(np.min(adjusted_pvalues)) <= float(diagnostic_alpha)
        )

    return df


def _build_rejection_rate_summary(results_df):
    records = []

    rejection_specs = [
        ("uncalibrated", "hsic3", "reject_hsic3"),
        ("uncalibrated", "dhsic", "reject_dhsic"),
        ("uncalibrated", "c2st", "reject_c2st"),
        ("uncalibrated", "global", "reject_global"),
        ("calibrated", "hsic3", "reject_calibrated_hsic3"),
        ("calibrated", "dhsic", "reject_calibrated_dhsic"),
        ("calibrated", "c2st", "reject_calibrated_c2st"),
        ("calibrated", "global", "reject_calibrated_global"),
    ]

    group_cols = ["model_name", "alpha", "n"]

    for (model_name, alpha_value, n), group in results_df.groupby(group_cols, sort=True):
        for calibration, diagnostic, reject_col in rejection_specs:
            if reject_col not in group.columns:
                continue

            valid_values = group[reject_col].dropna().astype(int)
            n_runs = int(len(valid_values))
            n_rejections = int(valid_values.sum())
            rejection_rate = float(n_rejections / n_runs) if n_runs > 0 else float("nan")

            records.append(
                {
                    "model_name": model_name,
                    "alpha": float(alpha_value),
                    "n": int(n),
                    "calibration": calibration,
                    "diagnostic": diagnostic,
                    "reject_col": reject_col,
                    "n_runs": n_runs,
                    "n_rejections": n_rejections,
                    "rejection_rate": rejection_rate,
                }
            )

    import pandas as pd

    return pd.DataFrame(records)

def benchmark_diagnostic_power_task(
    dataset=DEFAULT_DATASET,
    alpha_value=1.0,
    n=3000,
    seed=DEFAULT_BASE_SEED,
    kan_params=None,
    flow_params=None,
    task_checkpoint_root=None,
    pvalue_method=DEFAULT_PVALUE_METHOD,
    num_permutations=DEFAULT_NUM_PERMUTATIONS,
    permutation_n_jobs=DEFAULT_PERMUTATION_N_JOBS,
    diagnostic_alpha=DEFAULT_DIAGNOSTIC_ALPHA,
    rf_estimators=DEFAULT_RF_ESTIMATORS,
):
    kan_params = deepcopy(DEFAULT_KAN_PARAMS if kan_params is None else kan_params)
    flow_params = deepcopy(DEFAULT_FLOW_PARAMS if flow_params is None else flow_params)
    task_checkpoint_root = Path("." if task_checkpoint_root is None else task_checkpoint_root)
    started = perf_counter()
    rows = _evaluate_task(
        alpha_value=alpha_value,
        n=n,
        seed=seed,
        dataset=dataset,
        kan_params=kan_params,
        flow_params=flow_params,
        task_checkpoint_root=task_checkpoint_root,
        pvalue_method=pvalue_method,
        num_permutations=num_permutations,
        permutation_n_jobs=permutation_n_jobs,
        diagnostic_alpha=diagnostic_alpha,
        rf_estimators=rf_estimators,
    )
    elapsed_seconds = perf_counter() - started
    return {
        "dataset": str(dataset),
        "alpha": float(alpha_value),
        "n": int(n),
        "seed": int(seed),
        "elapsed_seconds": float(elapsed_seconds),
        "num_rows": int(len(rows)),
        "pvalue_method": str(pvalue_method),
        "num_permutations": int(num_permutations),
        "permutation_n_jobs": int(permutation_n_jobs),
    }


def run_diagnostic_power(
    output_dir=None,
    dataset=DEFAULT_DATASET,
    alpha_grid=None,
    n_grid=None,
    n_seeds=DEFAULT_N_SEEDS,
    base_seed=DEFAULT_BASE_SEED,
    n_jobs=1,
    pvalue_method=DEFAULT_PVALUE_METHOD,
    num_permutations=DEFAULT_NUM_PERMUTATIONS,
    permutation_n_jobs=DEFAULT_PERMUTATION_N_JOBS,
    diagnostic_alpha=DEFAULT_DIAGNOSTIC_ALPHA,
    rf_estimators=DEFAULT_RF_ESTIMATORS,
    overwrite=False,
    benchmark_only=False,
    skip_benchmark=False,
    debug=False,
):
    import pandas as pd
    from joblib import Parallel, delayed

    from utils.paths import ensure_dir, get_experiment_paths

    resolved_pvalue_method = "permutation" if str(pvalue_method) == "auto" else str(pvalue_method)
    resolved_permutation_n_jobs = int(permutation_n_jobs)
    if resolved_permutation_n_jobs == 0:
        resolved_permutation_n_jobs = -1 if int(n_jobs) == 1 else 1

    paths = get_experiment_paths(DEFAULT_EXPERIMENT_NAME, output_dir=output_dir)
    task_dir = ensure_dir(paths.checkpoints / "task_results")
    benchmark_dir = ensure_dir(paths.checkpoints / "benchmark")

    alpha_grid = list(DEFAULT_ALPHA_GRID if alpha_grid is None else alpha_grid)
    n_grid = list(DEFAULT_N_GRID if n_grid is None else n_grid)
    kan_params = deepcopy(DEFAULT_KAN_PARAMS)
    flow_params = deepcopy(DEFAULT_FLOW_PARAMS)

    if debug:
        for node_params in kan_params.values():
            node_params["steps"] = 2
        flow_params["max_epochs"] = 2
        alpha_grid = [alpha_grid[0]]
        n_grid = [n_grid[0]]
        n_seeds = 1
        n_jobs = 1
        resolved_permutation_n_jobs = 1
        num_permutations = min(int(num_permutations), 9)
        skip_benchmark = True

    benchmark_record = None
    if not skip_benchmark:
        benchmark_record = benchmark_diagnostic_power_task(
            dataset=dataset,
            alpha_value=max(alpha_grid),
            n=max(n_grid),
            seed=int(base_seed),
            kan_params=kan_params,
            flow_params=flow_params,
            task_checkpoint_root=benchmark_dir,
            pvalue_method=resolved_pvalue_method,
            num_permutations=int(num_permutations),
            permutation_n_jobs=int(resolved_permutation_n_jobs),
            diagnostic_alpha=float(diagnostic_alpha),
            rf_estimators=int(rf_estimators),
        )
        pd.DataFrame([benchmark_record]).to_csv(paths.data / "diagnostic_power_benchmark.csv", index=False)
        if benchmark_only:
            return pd.DataFrame([benchmark_record])

    seeds = [int(base_seed) + offset for offset in range(int(n_seeds))]
    tasks = [(float(alpha_value), int(n), int(seed)) for alpha_value, n, seed in product(alpha_grid, n_grid, seeds)]
    rows_nested = Parallel(n_jobs=int(n_jobs), backend="loky", verbose=10)(
        delayed(_load_or_run_task)(
            alpha_value=alpha_value,
            n=n,
            seed=seed,
            dataset=dataset,
            kan_params=kan_params,
            flow_params=flow_params,
            task_dir=task_dir,
            pvalue_method=resolved_pvalue_method,
            num_permutations=int(num_permutations),
            permutation_n_jobs=int(resolved_permutation_n_jobs),
            diagnostic_alpha=float(diagnostic_alpha),
            rf_estimators=int(rf_estimators),
            overwrite=overwrite,
        )
        for alpha_value, n, seed in tasks
    )
    rows = [row for chunk in rows_nested for row in chunk]
    results_df = pd.DataFrame(rows).sort_values(["alpha", "n", "seed", "model_name"]).reset_index(drop=True)

    # Keep the raw/non-calibrated results and add calibrated diagnostics
    # computed from the alpha=0 fitted-model baseline.
    results_df = _add_calibrated_diagnostics(
        results_df,
        diagnostic_alpha=float(diagnostic_alpha),
    )

    results_df.to_csv(paths.data / "diagnostic_power_results.csv", index=False)

    rejection_summary_df = _build_rejection_rate_summary(results_df)
    rejection_summary_df.to_csv(paths.data / "diagnostic_power_rejection_rates.csv", index=False)

    return results_df


def build_parser():
    parser = argparse.ArgumentParser(description="Run the finite-sample diagnostic-power experiment.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where experiment outputs will be written.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Synthetic dataset name.")
    parser.add_argument("--alpha-grid", nargs="*", type=float, default=None, help="Explicit alpha grid values.")
    parser.add_argument("--n-grid", nargs="*", type=int, default=None, help="Explicit sample-size grid values.")
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS, help="Number of seeds per alpha and sample size.")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="First seed used in the grid.")
    parser.add_argument(
        "--n_jobs",
        "--jobs",
        dest="n_jobs",
        type=int,
        default=1,
        help="Parallel jobs across paired (alpha, n, seed) tasks.",
    )
    parser.add_argument(
        "--pvalue-method",
        type=str,
        default=DEFAULT_PVALUE_METHOD,
        choices=("auto", "permutation"),
        help="P-value method for HSIC and dHSIC.",
    )
    parser.add_argument("--num-permutations", type=int, default=DEFAULT_NUM_PERMUTATIONS, help="Number of permutations for independence-test p-values.")
    parser.add_argument("--permutation-n-jobs", type=int, default=DEFAULT_PERMUTATION_N_JOBS, help="Threaded jobs used inside permutation-based p-value evaluation. Use 0 for auto.")
    parser.add_argument("--diagnostic-alpha", type=float, default=DEFAULT_DIAGNOSTIC_ALPHA, help="Significance level used for rejection indicators.")
    parser.add_argument("--rf-estimators", type=int, default=DEFAULT_RF_ESTIMATORS, help="Number of trees in the C2ST random forest.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute task results even if cached files already exist.")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only the benchmark task at the largest requested sample size.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip the pre-run benchmark task.")
    parser.add_argument("--debug", action="store_true", help="Run a reduced debug version.")
    return parser


def main():
    args = build_parser().parse_args()
    run_diagnostic_power(
        output_dir=args.output_dir,
        dataset=args.dataset,
        alpha_grid=args.alpha_grid,
        n_grid=args.n_grid,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        n_jobs=args.n_jobs,
        pvalue_method=args.pvalue_method,
        num_permutations=args.num_permutations,
        permutation_n_jobs=args.permutation_n_jobs,
        diagnostic_alpha=args.diagnostic_alpha,
        rf_estimators=args.rf_estimators,
        overwrite=args.overwrite,
        benchmark_only=args.benchmark_only,
        skip_benchmark=args.skip_benchmark,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
