from __future__ import annotations

import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as multitest
from tabulate import tabulate


def friedman_test(all_data, comp_index, alpha, higher_is_better):
    all_data = np.asarray(all_data)
    n_datasets, n_methods = all_data.shape
    if n_methods < 2:
        raise ValueError("Friedman test requires at least two methods.")

    if higher_is_better:
        ranks = np.array([stats.rankdata(-row) for row in all_data])
    else:
        ranks = np.array([stats.rankdata(row) for row in all_data])

    average_ranks = np.mean(ranks, axis=0)
    statistic, p_value = stats.friedmanchisquare(*[all_data[:, i] for i in range(n_methods)])

    baseline = average_ranks[comp_index]
    z_values = []
    p_values = []
    for idx in range(n_methods):
        z = (average_ranks[idx] - baseline) / np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
        z_values.append(z)
        p_values.append(2 * (1 - stats.norm.cdf(abs(z))))

    reject_corr, p_values_corr, _, _ = multitest.multipletests(p_values, alpha=alpha, method="holm")
    reject_unc = np.array([p < alpha for p in p_values])

    return {
        "friedman_statistic": statistic,
        "friedman_p_value": p_value,
        "average_ranks": average_ranks,
        "z_values": np.asarray(z_values),
        "p_values_post_hoc_unc": np.asarray(p_values),
        "p_values_post_hoc_corr": np.asarray(p_values_corr),
        "reject_post_hoc_unc": reject_unc,
        "reject_post_hoc_corr": reject_corr,
    }


def get_p_values_from_table_data(
    data,
    alpha=0.05,
    higher_is_better=True,
    output_latex=True,
    list_of_methods=None,
    list_of_metrics=None,
):
    table = np.asarray(data)
    if table.ndim != 3:
        raise ValueError("Expected data with shape (datasets, methods, metrics).")

    n_datasets, n_methods, n_metrics = table.shape
    list_of_methods = list_of_methods or [f"method_{i}" for i in range(n_methods)]
    list_of_metrics = list_of_metrics or [f"metric_{i}" for i in range(n_metrics)]

    method_scores = table.mean(axis=2)
    comp_index = int(np.argmin(method_scores.mean(axis=0)) if not higher_is_better else np.argmax(method_scores.mean(axis=0)))
    results = friedman_test(method_scores, comp_index, alpha, higher_is_better)

    p_values_unc = results["p_values_post_hoc_unc"]
    p_values_corr = results["p_values_post_hoc_corr"]
    best_method = comp_index

    friedman_post_hoc_table_unc = ["Friedman post-hoc tests (all metrics, uncorrected)"]
    friedman_post_hoc_table_corr = ["Friedman post-hoc tests (all metrics, corrected)"]
    for j in range(n_methods):
        p_val_str_corr = f"{p_values_corr[j]:.3f}" if p_values_corr[j] >= 1e-3 else "<1e-3"
        if p_values_corr[j] >= alpha:
            p_val_str_corr += "*"
        if j == best_method:
            p_val_str_corr += " (baseline)"
        friedman_post_hoc_table_corr.append(p_val_str_corr)

        p_val_str_unc = f"{p_values_unc[j]:.3f}" if p_values_unc[j] >= 1e-3 else "<1e-3"
        if p_values_unc[j] >= alpha:
            p_val_str_unc += "*"
        if j == best_method:
            p_val_str_unc += " (baseline)"
        friedman_post_hoc_table_unc.append(p_val_str_unc)

    table_data = [friedman_post_hoc_table_unc, friedman_post_hoc_table_corr]
    if output_latex:
        print(tabulate(table_data, headers=["All metrics"] + list_of_methods, tablefmt="latex"))
    else:
        print(tabulate(table_data, headers=["All metrics"] + list_of_methods, tablefmt="grid"))

    return results
