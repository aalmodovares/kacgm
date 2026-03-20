from __future__ import annotations

import argparse
import pickle
from copy import deepcopy

def run_symbolic_pruning_study(output_dir=None, n=5000, discrete=False, seed=1234, epochs=10000, include_orig_formula=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sympy as sp

    from datasets.synthetic import graph_data
    from models.kan import kan_model_mixed
    from utils.metrics import HSIC, mmd, rf
    from utils.paths import get_experiment_paths

    paths = get_experiment_paths("symbolic_pruning", output_dir=output_dir)
    x3_type = "discrete" if discrete else "continuous"
    kan_params = {
        "x2": {
            "node_types": {"x1": "continuous", "x2": "continuous", "x3": x3_type},
            "num_classes": {"x1": None, "x2": None, "x3": 2},
            "hidden_dim": 0,
            "batch_size": -1,
            "grid": 3,
            "k": 3,
            "seed": 0,
            "lr": 0.0005,
            "early_stop": True,
            "steps": epochs,
            "lamb": 0.01,
            "lamb_entropy": 0.1,
            "sparse_init": False,
            "mult_kan": False,
            "try_gpu": False,
            "loss": "mse",
            "verbose": 1,
        },
        "x3": {
            "node_types": {"x1": "continuous", "x2": "continuous", "x3": x3_type},
            "num_classes": {"x1": None, "x2": None, "x3": 2},
            "hidden_dim": 0,
            "batch_size": -1,
            "grid": 1,
            "k": 3,
            "seed": 0,
            "lr": 0.0005,
            "early_stop": True,
            "steps": epochs,
            "lamb": 0.01,
            "lamb_entropy": 0.1,
            "sparse_init": False,
            "mult_kan": False,
            "try_gpu": False,
            "loss": "mse",
            "verbose": 1,
        },
    }

    data_all, data_cf_all, graph, formula_gt = graph_data(name="triangle-non-linear-2").generate(num_samples=2 * n, seed=542, discrete=discrete, return_u=False)
    factual_train = data_all.iloc[:n]
    factual_eval = data_all.iloc[n : 2 * n]
    cf_train = [c.iloc[:n] for c in data_cf_all]
    cf_eval = [c.iloc[n : 2 * n] for c in data_cf_all]

    if discrete:
        for df in [factual_train, factual_eval] + cf_train + cf_eval:
            df.iloc[:, -1] = df.iloc[:, -1].apply(lambda value: 0 if value == -1 else 1)

    kan_model = kan_model_mixed(graph, deepcopy(kan_params))
    kan_model.fit(factual_train)

    pruned_model = kan_model.clone()
    pruned_model.draw_samples(1000, seed=seed)
    pruned_model.prune(factual_train)
    symbolic_model = pruned_model.clone()
    symbolic_model.draw_samples(1000, seed=seed)
    symbolic_model.to_symbolic(factual_train, method="ours")

    obs_kan = kan_model.draw_samples(num_samples=n)
    obs_kan_pruned = pruned_model.draw_samples(num_samples=n)
    obs_kan_symbolic = symbolic_model.draw_samples(num_samples=n)
    interventions = [{"x2": (lambda _: 0.0)}, {"x2": (lambda _: 1.0)}]
    int_kan = [kan_model.interventional_samples(intervention, num_samples_to_draw=n) for intervention in interventions]
    int_kan_pruned = [pruned_model.interventional_samples(intervention, num_samples_to_draw=n) for intervention in interventions]
    int_kan_symbolic = [symbolic_model.interventional_samples(intervention, num_samples_to_draw=n) for intervention in interventions]
    residuals_kan = kan_model.get_residuals(factual_eval)
    residuals_pruned = pruned_model.get_residuals(factual_eval)
    residuals_symbolic = symbolic_model.get_residuals(factual_eval)

    results = {
        "mmd_obs": {
            "kan": mmd(obs_kan.values, factual_eval.values),
            "pruned": mmd(obs_kan_pruned.values, factual_eval.values),
            "symbolic": mmd(obs_kan_symbolic.values, factual_eval.values),
        },
        "rf_obs": {
            "kan": rf(obs_kan.values, factual_eval.values),
            "pruned": rf(obs_kan_pruned.values, factual_eval.values),
            "symbolic": rf(obs_kan_symbolic.values, factual_eval.values),
        },
        "mmd_int": {
            "kan": mmd(int_kan[1].values, cf_eval[5].values),
            "pruned": mmd(int_kan_pruned[1].values, cf_eval[5].values),
            "symbolic": mmd(int_kan_symbolic[1].values, cf_eval[5].values),
        },
        "rf_int": {
            "kan": rf(int_kan[1].values, cf_eval[5].values),
            "pruned": rf(int_kan_pruned[1].values, cf_eval[5].values),
            "symbolic": rf(int_kan_symbolic[1].values, cf_eval[5].values),
        },
        "hsic": {
            "kan": HSIC(residuals_kan["x2"], factual_eval[["x1"]].values),
            "pruned": HSIC(residuals_pruned["x2"], factual_eval[["x1"]].values),
            "symbolic": HSIC(residuals_symbolic["x2"], factual_eval[["x1"]].values),
        },
        "formulas": symbolic_model.get_formulas(ex_round=3),
    }

    if include_orig_formula:
        symbolic_orig_model = pruned_model.clone()
        symbolic_orig_model.draw_samples(1000, seed=seed)
        symbolic_orig_model.to_symbolic(data=factual_train, method="orig")
        obs_kan_symbolic_orig = symbolic_orig_model.draw_samples(num_samples=n)
        int_kan_symbolic_orig = [symbolic_orig_model.interventional_samples(intervention, num_samples_to_draw=n) for intervention in interventions]
        residuals_symbolic_orig = symbolic_orig_model.get_residuals(factual_eval)
        results["mmd_obs"]["orig"] = mmd(obs_kan_symbolic_orig.values, factual_eval.values)
        results["rf_obs"]["orig"] = rf(obs_kan_symbolic_orig.values, factual_eval.values)
        results["mmd_int"]["orig"] = mmd(int_kan_symbolic_orig[1].values, cf_eval[5].values)
        results["rf_int"]["orig"] = rf(int_kan_symbolic_orig[1].values, cf_eval[5].values)
        results["hsic"]["orig"] = HSIC(residuals_symbolic_orig["x2"], factual_eval[["x1"]].values)
        results["formulas_orig"] = symbolic_orig_model.get_formulas(ex_round=3)

    with open(paths.data / "symbolic_pruning_results.pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    labels = ["KAN", "Pruned", "Symbolic"] + (["Orig"] if include_orig_formula else [])
    axes[0].plot(labels, [results["mmd_obs"].get(key.lower(), results["mmd_obs"].get(key)) for key in ["kan", "pruned", "symbolic"] + (["orig"] if include_orig_formula else [])])
    axes[1].plot(labels, [results["rf_obs"].get(key.lower(), results["rf_obs"].get(key)) for key in ["kan", "pruned", "symbolic"] + (["orig"] if include_orig_formula else [])])
    axes[2].plot(labels, [results["hsic"].get(key.lower(), results["hsic"].get(key)) for key in ["kan", "pruned", "symbolic"] + (["orig"] if include_orig_formula else [])])
    axes[3].plot(labels, [results["mmd_int"].get(key.lower(), results["mmd_int"].get(key)) for key in ["kan", "pruned", "symbolic"] + (["orig"] if include_orig_formula else [])])
    axes[4].plot(labels, [results["rf_int"].get(key.lower(), results["rf_int"].get(key)) for key in ["kan", "pruned", "symbolic"] + (["orig"] if include_orig_formula else [])])
    axes[0].set_title("Observational MMD")
    axes[1].set_title("Observational RF Accuracy")
    axes[2].set_title("HSIC between residuals and parents")
    axes[3].set_title("Interventional MMD")
    axes[4].set_title("Interventional RF Accuracy")
    fig.tight_layout()
    fig.savefig(paths.figures / "symbolic_pruning_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    formulas = results["formulas"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    x1_range = np.linspace(factual_eval["x1"].min(), factual_eval["x1"].max(), 100)
    x2_range = np.linspace(factual_eval["x2"].min(), factual_eval["x2"].max(), 100)
    fixed_x1 = np.ones(100) * factual_eval["x1"].mean()
    fixed_x2 = np.ones(100) * factual_eval["x2"].mean()
    df_x1 = pd.DataFrame({"x1": x1_range, "x2": fixed_x2})
    df_x2 = pd.DataFrame({"x1": fixed_x1, "x2": x2_range})
    fx3_x1 = symbolic_model.evaluate_symbolic(df_x1, "x3")["x3"]
    fx3_x2 = symbolic_model.evaluate_symbolic(df_x2, "x3")["x3"]
    kan_predictions_x1 = kan_model.predict_node(df_x1, "x3", proba=discrete, logits=discrete)
    kan_predictions_x2 = kan_model.predict_node(df_x2, "x3", proba=discrete, logits=discrete)
    prune_predictions_x1 = pruned_model.predict_node(df_x1, "x3", proba=discrete, logits=discrete)
    prune_predictions_x2 = pruned_model.predict_node(df_x2, "x3", proba=discrete, logits=discrete)
    if discrete:
        kan_predictions_x1 = kan_predictions_x1[:, 1]
        kan_predictions_x2 = kan_predictions_x2[:, 1]
        prune_predictions_x1 = prune_predictions_x1[:, 1]
        prune_predictions_x2 = prune_predictions_x2[:, 1]

    x1, x2 = sp.symbols("x1 x2")
    true_formula = 0.4 * x1 + 0.1 * x1 ** 2 + 0.7 * x2 - 0.5 * x2 ** 2
    true_f = sp.lambdify((x1, x2), true_formula, modules="numpy")
    fx3_x1_true = true_f(df_x1["x1"], df_x1["x2"])
    fx3_x2_true = true_f(df_x2["x1"], df_x2["x2"])

    axes[0].plot(x1_range, fx3_x1 - fx3_x1.mean(), label="Formula")
    axes[0].plot(x1_range, fx3_x1_true - fx3_x1_true.mean(), label="True formula", linestyle="dashdot")
    axes[0].plot(x1_range, kan_predictions_x1 - kan_predictions_x1.mean(), label="KAN")
    axes[0].plot(x1_range, prune_predictions_x1 - prune_predictions_x1.mean(), label="Pruned KAN", linestyle="dashed")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("logit(x3)" if discrete else "x3")
    axes[0].set_title("Symbolic formula for x3 vs x1")
    axes[0].legend()

    axes[1].plot(x2_range, fx3_x2 - fx3_x2.mean(), label="Formula")
    axes[1].plot(x2_range, fx3_x2_true - fx3_x2_true.mean(), label="True formula", linestyle="dashdot")
    axes[1].plot(x2_range, kan_predictions_x2 - kan_predictions_x2.mean(), label="KAN")
    axes[1].plot(x2_range, prune_predictions_x2 - prune_predictions_x2.mean(), label="Pruned KAN", linestyle="dashed")
    axes[1].set_xlabel("x2")
    axes[1].set_ylabel("logit(x3)" if discrete else "x3")
    axes[1].set_title("Symbolic formula for x3 vs x2")
    axes[1].legend()
    axes[0].text(0.05, 0.85, f"Formula: {formulas['x3']}", transform=axes[0].transAxes, fontsize=10, verticalalignment="top")
    axes[0].text(0.05, 0.65, f"True formula: {true_formula}", transform=axes[0].transAxes, fontsize=10, verticalalignment="top")
    fig.tight_layout()
    fig.savefig(paths.figures / "symbolic_pruning_formula.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the pruning and symbolic regression study.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where study outputs will be written.")
    parser.add_argument("--n", type=int, default=5000, help="Train and evaluation samples per split.")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed used for symbolic side calculations.")
    parser.add_argument("--epochs", type=int, default=10000, help="Training steps for KAN nodes.")
    parser.add_argument("--discrete", action="store_true", help="Use the discrete version of the study.")
    parser.add_argument("--skip-orig-formula", action="store_true", help="Skip the original symbolic baseline.")
    return parser


def main():
    args = build_parser().parse_args()
    run_symbolic_pruning_study(
        output_dir=args.output_dir,
        n=args.n,
        discrete=args.discrete,
        seed=args.seed,
        epochs=args.epochs,
        include_orig_formula=not args.skip_orig_formula,
    )


if __name__ == "__main__":
    main()
