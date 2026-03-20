from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product

DEFAULT_KAN_PARAMS = {
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


def run_one(alpha_value, realization, dataset, n, kan_params, flow_params, base_seed=0):
    import numpy as np
    import torch
    from dowhy import gcm
    from dowhy.graph import get_ordered_predecessors, is_root_node

    from datasets.synthetic import graph_data
    from models.factory import create_model_from_graph
    from models.flow import causalflow_model
    from utils.metrics import HSIC, dHSIC, mmd, rf

    seed = int(base_seed + 10_000 * realization + round(1_000 * float(alpha_value)))
    data_all, data_cf_all, graph, formula_gt, u_df = graph_data(name=dataset).generate(
        num_samples=2 * n,
        seed=seed,
        alpha=alpha_value,
        return_u=True,
    )

    factual_train = data_all.iloc[:n]
    factual_eval = data_all.iloc[n : 2 * n]
    cf_train = [c.iloc[:n] for c in data_cf_all]
    cf_eval = [c.iloc[n : 2 * n] for c in data_cf_all]
    u_eval = u_df.iloc[n : 2 * n]

    kan_model = create_model_from_graph(graph, "kan", deepcopy(kan_params))
    flow_model = causalflow_model(graph, deepcopy(flow_params))
    gcm.fit(kan_model, factual_train)
    flow_model.fit(factual_train)

    obs_kan = gcm.draw_samples(kan_model, num_samples=n)
    obs_flow = flow_model.draw_samples(num_samples=n)

    interventions = [{"x2": (lambda _: 0.0)}, {"x2": (lambda _: 1.0)}]
    int_kan = [gcm.interventional_samples(kan_model, intervention, num_samples_to_draw=n) for intervention in interventions]
    int_flow = [flow_model.interventional_samples(intervention, num_samples_to_draw=n) for intervention in interventions]
    cfactual_kan = [gcm.counterfactual_samples(kan_model, intervention, factual_eval) for intervention in interventions]
    cfactual_flow = [flow_model.counterfactual_samples(intervention, factual_eval) for intervention in interventions]

    int_kan_y = [samples.values[:, -1] for samples in int_kan]
    int_flow_y = [samples.values[:, -1] for samples in int_flow]
    cfactual_kan_y = [samples.values[:, -1] for samples in cfactual_kan]
    cfactual_flow_y = [samples.values[:, -1] for samples in cfactual_flow]
    cf_train_real = [cf_traini.values[:, -1] for cf_traini in cf_train[4:6]]
    cf_real = [cf_evali.values[:, -1] for cf_evali in cf_eval[4:6]]

    ate_kan = float(int_kan_y[1].mean() - int_kan_y[0].mean())
    ate_flow = float(int_flow_y[1].mean() - int_flow_y[0].mean())
    ate_train_real = float(cf_train_real[1].mean() - cf_train_real[0].mean())
    ate_real = float(cf_real[1].mean() - cf_real[0].mean())

    cf_error_kan = np.concatenate([cfactual_kan_y[i] - cf_real[i] for i in range(2)])
    cf_error_flow = np.concatenate([cfactual_flow_y[i] - cf_real[i] for i in range(2)])

    residuals = []
    for node in kan_model.graph.nodes:
        if is_root_node(kan_model.graph, node):
            residuals_i = factual_eval[node].to_numpy().reshape(-1, 1)
        else:
            parents_samples = factual_eval[get_ordered_predecessors(kan_model.graph, node)].to_numpy()
            residuals_i = kan_model.causal_mechanism(node).estimate_noise(factual_eval[node].to_numpy(), parents_samples)
        residuals.append(residuals_i)
    u_kan = np.stack(residuals).T.squeeze()
    u_flow = flow_model.flow().transform(torch.tensor(factual_eval.values, dtype=torch.float32)).detach().numpy()

    rows = []

    def add(metric, model, value):
        rows.append(
            {
                "alpha": float(alpha_value),
                "realization": int(realization),
                "metric": str(metric),
                "model": str(model),
                "value": float(value),
            }
        )

    add("mmd_obs", "orig", mmd(factual_train.values, factual_eval.values))
    add("mmd_obs", "kan", mmd(obs_kan.values, factual_eval.values))
    add("mmd_obs", "flow", mmd(obs_flow.values, factual_eval.values))
    add("rf_obs", "orig", rf(factual_train.values, factual_eval.values))
    add("rf_obs", "kan", rf(obs_kan.values, factual_eval.values))
    add("rf_obs", "flow", rf(obs_flow.values, factual_eval.values))
    add("mmd_int", "orig", mmd(cf_train[5].values, cf_eval[5].values))
    add("mmd_int", "kan", mmd(int_kan[1].values, cf_eval[5].values))
    add("mmd_int", "flow", mmd(int_flow[1].values, cf_eval[5].values))
    add("rf_int", "orig", rf(cf_train[5].values, cf_eval[5].values))
    add("rf_int", "kan", rf(int_kan[1].values, cf_eval[5].values))
    add("rf_int", "flow", rf(int_flow[1].values, cf_eval[5].values))
    add("ate_error", "orig", float(np.abs(ate_train_real - ate_real)))
    add("ate_error", "kan", float(np.abs(ate_kan - ate_real)))
    add("ate_error", "flow", float(np.abs(ate_flow - ate_real)))
    add("cf_mae", "kan", float(np.abs(cf_error_kan).mean()))
    add("cf_mae", "flow", float(np.abs(cf_error_flow).mean()))
    add("cf_mse", "kan", float((cf_error_kan ** 2).mean()))
    add("cf_mse", "flow", float((cf_error_flow ** 2).mean()))
    add("hsic", "orig", float(HSIC(u_eval.values[:, 2], factual_eval[["x1", "x2"]].values)))
    add("hsic", "kan", float(HSIC(u_kan[:, 2], factual_eval[["x1", "x2"]].values)))
    add("hsic", "flow", float(HSIC(u_flow[:, 2], factual_eval[["x1", "x2"]].values)))
    add("dhsic", "orig", float(dHSIC(u_eval.values[:, 0], u_eval.values[:, 1], u_eval.values[:, 2])))
    add("dhsic", "kan", float(dHSIC(u_kan[:, 0], u_kan[:, 1], u_kan[:, 2])))
    add("dhsic", "flow", float(dHSIC(u_flow[:, 0], u_flow[:, 1], u_flow[:, 2])))

    return rows


def create_plot(df_plot, figure_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    metrics_map = {
        "mmd_obs": "MMD obs",
        "rf_obs": "RF obs",
        "mmd_int": "MMD int",
        "rf_int": "RF int",
        "ate_error": "ATE error",
        "cf_mae": "CF MAE",
        "cf_mse": "CF MSE",
        "hsic": "HSIC",
        "dhsic": "dHSIC",
    }
    df_plot = df_plot.copy()
    df_plot["metric"] = df_plot["metric"].map(metrics_map)
    df_plot = df_plot[df_plot["metric"].isin(["RF obs", "RF int", "ATE error", "CF MAE", "HSIC", "dHSIC"])]
    df_plot["model"] = df_plot["model"].map({"kan": "KAN", "flow": "Flow", "orig": "Data"})
    df_plot["model"] = pd.Categorical(df_plot["model"], categories=["Data", "KAN", "Flow"], ordered=True)

    sns.set_theme(style="whitegrid")
    grid = sns.FacetGrid(df_plot, col="metric", col_wrap=3, sharey=False, height=3.5)
    grid.map_dataframe(sns.lineplot, x="alpha", y="value", hue="model", estimator="mean", errorbar="sd")
    grid.add_legend()
    grid.fig.tight_layout()
    grid.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def run_sensitivity(output_dir=None, dataset="triangle-sensitivity-nonlinear", n=3000, n_realizations=10, alpha_grid=None, jobs=1, debug=False):
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed

    from utils.paths import get_experiment_paths

    paths = get_experiment_paths("sensitivity", output_dir=output_dir)
    alpha_grid = np.linspace(0, 1, num=10) if alpha_grid is None else np.asarray(alpha_grid, dtype=float)
    kan_params = deepcopy(DEFAULT_KAN_PARAMS)
    flow_params = deepcopy(DEFAULT_FLOW_PARAMS)

    if debug:
        kan_params["x2"]["steps"] = 2
        kan_params["x3"]["steps"] = 2
        flow_params["max_epochs"] = 2
        alpha_grid = np.linspace(0, 1, num=2)
        n_realizations = min(n_realizations, 2)
        jobs = 1

    tasks = list(product(alpha_grid, range(n_realizations)))
    all_rows_nested = Parallel(n_jobs=jobs, backend="loky", verbose=10)(
        delayed(run_one)(alpha, realization, dataset, n, kan_params, flow_params) for alpha, realization in tasks
    )
    rows = [row for chunk in all_rows_nested for row in chunk]
    df = pd.DataFrame(rows).sort_values(["metric", "alpha", "model", "realization"]).reset_index(drop=True)
    df.to_csv(paths.data / "sensitivity_analysis_results.csv", index=False)
    create_plot(df, paths.figures / "sensitivity_analysis_results.png")


def build_parser():
    parser = argparse.ArgumentParser(description="Run the additivity sensitivity experiment.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where experiment outputs will be written.")
    parser.add_argument("--dataset", type=str, default="triangle-sensitivity-nonlinear", help="Synthetic dataset name.")
    parser.add_argument("--n", type=int, default=3000, help="Train and evaluation samples per realization.")
    parser.add_argument("--n-realizations", type=int, default=10, help="Number of realizations.")
    parser.add_argument("--alpha-grid", nargs="*", type=float, default=None, help="Explicit alpha grid values.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs across alpha/realization pairs.")
    parser.add_argument("--debug", action="store_true", help="Run a reduced debug version.")
    return parser


def main():
    args = build_parser().parse_args()
    run_sensitivity(
        output_dir=args.output_dir,
        dataset=args.dataset,
        n=args.n,
        n_realizations=args.n_realizations,
        alpha_grid=args.alpha_grid,
        jobs=args.jobs,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
