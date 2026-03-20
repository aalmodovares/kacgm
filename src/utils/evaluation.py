from __future__ import annotations

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from dowhy import gcm

from models.kan import (
    kan_auto_symbolic_formula,
    kan_model_mixed,
    kan_predictor,
    symbolic_kan_regressor,
)
from utils.metrics import mmd, rf


def evaluate_model(
    model,
    n,
    factual_eval,
    cf_eval,
    dataset,
    model_name,
    r,
    samples_dir=None,
    images_dir=None,
    formula_gt=None,
    kan_params=None,
    verbose=False,
    intervention=None,
    num_classes=None,
    seed=None,
):
    if intervention is None:
        intervention = [{"x1": lambda x: 1}]

    results = deepcopy(r)
    discrete = False
    discrete_nodes = []
    for node in factual_eval.columns:
        if len(factual_eval[node].unique()) <= 5:
            discrete = True
            discrete_nodes.append(node)

    if seed is not None:
        np.random.seed(seed)

    if model_name in ["kan_mixed", "kaam_mixed", "flow"]:
        obs_samples = model.draw_samples(num_samples=n)
    else:
        obs_samples = gcm.draw_samples(model, num_samples=n)

    if discrete:
        obs_samples[discrete_nodes] = obs_samples[discrete_nodes].round().astype(int)
        for dnode in discrete_nodes:
            obs_samples[dnode] = obs_samples[dnode].clip(0, num_classes[dnode] - 1)
    obs_samples = obs_samples[factual_eval.columns]
    obs_samples.to_csv(os.path.join(samples_dir, f"{dataset}_obs_samples_n_{n}_{model_name}.csv"), index=False)

    results["mmd_obs_avg"] = mmd(factual_eval.to_numpy(), obs_samples.to_numpy())
    results["rf_acc_obs_avg"] = rf(factual_eval.to_numpy(), obs_samples.to_numpy())

    for node in factual_eval.columns:
        plt.hist([factual_eval[node], obs_samples[node]], label=["real", "sampled"], bins=20)
        plt.title(f"Node {node} distribution comparison")
        plt.legend(loc="best")
        plt.tight_layout()
        node_name = node.replace("/", "_")
        plt.savefig(os.path.join(images_dir, f"hist_{dataset}_eval_node_{node_name}_{n}_{model_name}.png"))
        plt.close()

        if model_name == "kaam" and str(node) not in {"pakts473", "p44/42"}:
            results[node] = evaluate_kaam(
                model,
                node,
                factual_eval,
                formula_gt,
                kan_params,
                verbose,
                save_name=os.path.join(images_dir, f"{model_name}_{dataset}_{node_name}_n_{n}"),
            )

        if model_name == "kaam_mixed" and node not in discrete_nodes and str(node) not in {"pakts473", "p44/42"}:
            results[node] = evaluate_kaam_mixed(
                model,
                node,
                factual_eval,
                formula_gt,
                kan_params,
                verbose,
                save_name=os.path.join(images_dir, f"{model_name}_{dataset}_{node_name}_n_{n}"),
            )

    mmd_int = []
    rf_acc_int = []
    mse_cf = []
    mae_cf = []
    for inter_index, inter in enumerate(intervention):
        if model_name in ["kan_mixed", "kaam_mixed", "flow"]:
            int_samples = model.interventional_samples(inter, num_samples_to_draw=n)
        else:
            int_samples = gcm.interventional_samples(model, inter, num_samples_to_draw=n)

        if not discrete:
            if model_name == "flow":
                cf_estimates = model.counterfactual_samples(inter, factual_samples=factual_eval)
            else:
                cf_estimates = gcm.counterfactual_samples(model, inter, observed_data=pd.DataFrame(factual_eval.values, columns=factual_eval.columns))
        else:
            int_samples[discrete_nodes] = int_samples[discrete_nodes].round().astype(int)
            for dnode in discrete_nodes:
                int_samples[dnode] = int_samples[dnode].clip(0, num_classes[dnode] - 1)

        int_samples = int_samples[cf_eval[inter_index].columns]
        if not discrete:
            cf_estimates = cf_estimates[factual_eval.columns]

        int_samples.to_csv(os.path.join(samples_dir, f"{dataset}_int_samples_n_{n}_{model_name}_{inter_index}.csv"), index=False)
        if not discrete:
            cf_estimates.to_csv(os.path.join(samples_dir, f"{dataset}_cf_estimates_n_{n}_{model_name}_{inter_index}.csv"), index=False)

        mmd_int.append(mmd(cf_eval[inter_index].to_numpy(), int_samples.to_numpy()))
        rf_acc_int.append(rf(cf_eval[inter_index].to_numpy(), int_samples.to_numpy()))
        if not discrete:
            mse_cf.append(np.mean(np.square(cf_eval[inter_index].to_numpy() - cf_estimates.to_numpy())))
            mae_cf.append(np.mean(np.abs(cf_eval[inter_index].to_numpy() - cf_estimates.to_numpy())))

    results["mmd_int_avg"] = sum(mmd_int) / len(mmd_int)
    results["mmd_int_all"] = mmd_int
    results["rf_acc_int_avg"] = sum(rf_acc_int) / len(rf_acc_int)
    results["rf_acc_int_all"] = rf_acc_int
    if not discrete:
        results["mse_cf_avg"] = sum(mse_cf) / len(mse_cf)
        results["mse_cf_all"] = mse_cf
        results["mae_cf_avg"] = sum(mae_cf) / len(mae_cf)
        results["mae_cf_all"] = mae_cf
    return results


def evaluate_kaam(model, node, factual, formula_gt, kan_params, verbose=True, save_name=None):
    results = {"mae_uniform": None, "mae_weighted": None, "formula": None, "formula_gt": None}

    if hasattr(model.causal_mechanism(node), "prediction_model"):
        if isinstance(model.causal_mechanism(node).prediction_model, kan_predictor):
            loss = model.causal_mechanism(node).prediction_model.hyperparameters["loss"]
            x_names = sorted(model.graph.predecessors(node))
            X = factual[x_names].to_numpy()
            Y = factual[[node]].to_numpy()
            kan_object = model.causal_mechanism(node).prediction_model.model
            if len(kan_object.width_in) != 2 or kan_params[node]["mult_kan"]:
                print("Warning: symbolic regression has only been tested for KAAM, we resort to standard KAN auto-symbolic...")
                formula = kan_auto_symbolic_formula(kan_object, X, x_names)[0]
            else:
                if verbose:
                    print("Fitting symbolic regressor...")
                symb_obj = symbolic_kan_regressor(x_names, [node], loss)
                symb_obj.fit(
                    model.causal_mechanism(node).prediction_model.model,
                    X,
                    Y,
                    val_split=0.2,
                    r2_threshold=0.95,
                    show_results=False,
                    save_dir=None,
                )
                formula = symb_obj.get_formula()[0]

            results["formula_gt"] = None if formula_gt is None else formula_gt[node + "_str"]
            results["formula"] = formula

            if verbose:
                print(f"Formula for {node}: {formula}")
                if formula_gt is not None:
                    print(f"Ground truth formula for {node}: {formula_gt[node + '_str']}")

            x_min = X.min(axis=0)
            x_max = X.max(axis=0)
            if np.amin(x_max - x_min) < 1e-6:
                return results

            def error_func(x):
                return np.abs(sympy.lambdify(x_names, formula)(*x.T) - formula_gt[node](*x.T))

            n_samples = len(X)
            if len(x_names) == 1:
                samples = np.random.uniform(x_min, x_max, n_samples).reshape((n_samples, len(x_names)))
            else:
                samples_1d = [np.random.uniform(x_min[i], x_max[i], int(n_samples ** (1 / len(x_names)))) for i in range(len(x_names))]
                mesh = np.meshgrid(*samples_1d)
                samples = np.array([m.flatten() for m in mesh]).T

            if formula_gt is not None:
                mae_uniform = np.mean(error_func(samples))
                results["mae_uniform"] = mae_uniform
                if verbose:
                    print(f"Numerical integral of the MAE of the symbolic formula (uniform sampling): {mae_uniform}")
                mae_weighted = np.mean(error_func(X))
                results["mae_weighted"] = mae_weighted
                if verbose:
                    print(f"Numerical integral of the MAE of the symbolic formula (weighted by factual samples): {mae_weighted}")

            if len(x_names) <= 2:
                if len(x_names) == 1:
                    plt.scatter(factual[x_names[0]], factual[node], label="real", alpha=0.5)
                    x_range = np.linspace(factual[x_names[0]].min(), factual[x_names[0]].max(), 100)
                    y_symb = sympy.lambdify(x_names, formula)(x_range)
                    plt.plot(x_range, y_symb, color="r", label="symbolic")
                    y_kan = model.causal_mechanism(node).prediction_model.predict(x_range.reshape(-1, 1))
                    plt.plot(x_range, y_kan, color="g", label="KAN")
                    if formula_gt is not None:
                        plt.plot(x_range, formula_gt[node](x_range), color="k", label="ground truth")
                    plt.xlabel(x_names[0])
                    plt.ylabel(node)
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                    ax.scatter(factual[x_names[0]], factual[x_names[1]], factual[node], label="real", alpha=1)
                    x1_range = np.linspace(factual[x_names[0]].min(), factual[x_names[0]].max(), 30)
                    x2_range = np.linspace(factual[x_names[1]].min(), factual[x_names[1]].max(), 30)
                    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
                    x_mesh = np.array([x1_mesh.flatten(), x2_mesh.flatten()]).T
                    y_symb = sympy.lambdify(x_names, formula)(x_mesh[:, 0], x_mesh[:, 1])
                    ax.plot_trisurf(x_mesh[:, 0], x_mesh[:, 1], y_symb, color="r", alpha=0.2, label="symbolic")
                    y_kan = model.causal_mechanism(node).prediction_model.predict(x_mesh).squeeze()
                    ax.plot_trisurf(x_mesh[:, 0], x_mesh[:, 1], y_kan, color="g", alpha=0.2, label="KAAM")
                    if formula_gt is not None:
                        y_gt = formula_gt[node](x_mesh[:, 0], x_mesh[:, 1])
                        ax.plot_trisurf(x_mesh[:, 0], x_mesh[:, 1], y_gt, color="k", alpha=0.2, label="ground truth")
                    ax.set_xlabel(x_names[0])
                    ax.set_ylabel(x_names[1])
                    ax.set_zlabel(node)
                plt.title(f"Node {node} prediction comparison")
                plt.legend(loc="best")
                plt.tight_layout()
                if save_name is not None:
                    plt.savefig(save_name + "_symbolic.png", dpi=300)
                plt.close()
    return results


def evaluate_kaam_mixed(model, node, factual, formula_gt, kan_params, verbose=True, save_name=None):
    results = {"mae_uniform": None, "mae_weighted": None, "formula": None, "formula_gt": None}

    if hasattr(model.models[node], "model"):
        loss = model.models[node].hyperparameters["loss"]
        x_names = sorted(model.graph.predecessors(node))
        X = factual[x_names].to_numpy()
        Y = factual[[node]].to_numpy()
        kan_object = model.models[node].model
        if len(kan_object.width_in) != 2 or kan_params[node]["mult_kan"]:
            print("Warning: symbolic regression has only been tested for KAAM, we resort to standard KAN auto-symbolic...")
            formula = kan_auto_symbolic_formula(kan_object, X, x_names)[0]
        else:
            if verbose:
                print("Fitting symbolic regressor...")
            symb_obj = symbolic_kan_regressor(x_names, [node], loss)
            symb_obj.fit(
                kan_object,
                X,
                Y,
                val_split=0.2,
                r2_threshold=0.95,
                show_results=False,
                save_dir=None,
            )
            formula = symb_obj.get_formula()[0]

        results["formula_gt"] = None if formula_gt is None else formula_gt[node + "_str"]
        results["formula"] = formula

        if verbose:
            print(f"Formula for {node}: {formula}")
            if formula_gt is not None:
                print(f"Ground truth formula for {node}: {formula_gt[node + '_str']}")

        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        if np.amax(x_max - x_min) < 1e-6:
            return results

        def error_func(x):
            return np.abs(sympy.lambdify(x_names, formula)(*x.T) - formula_gt[node](*x.T))

        n_samples = len(X)
        if len(x_names) == 1:
            samples = np.random.uniform(x_min, x_max, n_samples).reshape((n_samples, len(x_names)))
        else:
            samples_1d = [np.random.uniform(x_min[i], x_max[i], int(n_samples ** (1 / len(x_names)))) for i in range(len(x_names))]
            mesh = np.meshgrid(*samples_1d)
            samples = np.array([m.flatten() for m in mesh]).T

        if formula_gt is not None:
            mae_uniform = np.mean(error_func(samples))
            results["mae_uniform"] = mae_uniform
            if verbose:
                print(f"Numerical integral of the MAE of the symbolic formula (uniform sampling): {mae_uniform}")
            mae_weighted = np.mean(error_func(X))
            results["mae_weighted"] = mae_weighted
            if verbose:
                print(f"Numerical integral of the MAE of the symbolic formula (weighted by factual samples): {mae_weighted}")

        if len(x_names) <= 2:
            if len(x_names) == 1:
                plt.scatter(factual[x_names[0]], factual[node], label="real", alpha=0.5)
                x_range = np.linspace(factual[x_names[0]].min(), factual[x_names[0]].max(), 100)
                y_symb = sympy.lambdify(x_names, formula)(x_range)
                plt.plot(x_range, y_symb, color="r", label="symbolic")
                y_kan = model.models[node].predict(x_range.reshape(-1, 1))
                plt.plot(x_range, y_kan, color="g", label="KAN")
                if formula_gt is not None:
                    plt.plot(x_range, formula_gt[node](x_range), color="k", label="ground truth")
                plt.xlabel(x_names[0])
                plt.ylabel(node)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(factual[x_names[0]], factual[x_names[1]], factual[node], label="real", alpha=1)
                x1_range = np.linspace(factual[x_names[0]].min(), factual[x_names[0]].max(), 30)
                x2_range = np.linspace(factual[x_names[1]].min(), factual[x_names[1]].max(), 30)
                x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
                x_mesh = np.array([x1_mesh.flatten(), x2_mesh.flatten()]).T
                y_symb = sympy.lambdify(x_names, formula)(x_mesh[:, 0], x_mesh[:, 1])
                ax.plot_trisurf(x_mesh[:, 0], x_mesh[:, 1], y_symb, color="r", alpha=0.2, label="symbolic")
                y_kan = model.models[node].predict(x_mesh).squeeze()
                ax.plot_trisurf(x_mesh[:, 0], x_mesh[:, 1], y_kan, color="g", alpha=0.2, label="KAAM")
                if formula_gt is not None:
                    y_gt = formula_gt[node](x_mesh[:, 0], x_mesh[:, 1])
                    ax.plot_trisurf(x_mesh[:, 0], x_mesh[:, 1], y_gt, color="k", alpha=0.2, label="ground truth")
                ax.set_xlabel(x_names[0])
                ax.set_ylabel(x_names[1])
                ax.set_zlabel(node)
            plt.title(f"Node {node} prediction comparison")
            plt.legend(loc="best")
            plt.tight_layout()
            if save_name is not None:
                plt.savefig(save_name + "_symbolic.png", dpi=300)
            plt.close()
    return results
