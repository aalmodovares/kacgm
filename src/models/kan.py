import datetime
import os
from copy import deepcopy
from pathlib import Path
import time

import dowhy.gcm as gcm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sympy
import torch
from kan import *
from scipy.optimize import minimize as minimize
from scipy.stats import norm
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

class kan_model_mixed(object):
    """
    Class for mixed KAN model for SCMs with both discrete and continuous variables.
    We assume all noise is additive standard gaussian
    """
    def __init__(self, graph, params):
        self.graph = graph # Causal graph in networkx format
        #self.params = params  # KAN params for each node
        # Create all the models we will need
        self.nodes = list(graph.nodes)
        # Get the parents of each node
        self.parents = {node: list(graph.predecessors(node)) for node in self.nodes}
        example_node = params.keys().__iter__().__next__()
        # Get the type of data of each node and create the corresponding node
        self.node_types = params[example_node]['node_types']
        self.num_classes = params[example_node].get('num_classes', {})
        # Delete 'node_types' and 'num_classes' from params to avoid issues when creating the kan_predictor objects
        for node in params.keys():
            if 'node_types' in params[node]:
                del params[node]['node_types']
            if 'num_classes' in params[node]:
                del params[node]['num_classes']

        if 'verbose' not in params[example_node]:
            for node in params.keys():
                params[node]['verbose'] = 0

        self.verbose = params[example_node]['verbose']
        self.symbolic_method = None

        self.models = {}
        self.noise_models = {}
        for node in self.nodes:
            if len(self.parents[node]) == 0:
                self.models[node] = gcm.ScipyDistribution(norm)  # Root nodes are normal (will be fitted)
            else:
                if self.node_types[node] == 'discrete':
                    params_disc = deepcopy(params[node])
                    params_disc['loss'] = 'discrete'  # Discrete nodes use discrete losses
                    params_disc['num_classes'] = self.num_classes[node]
                    params_disc['verbose'] = self.verbose
                    self.models[node] = kan_predictor(**params_disc)
                else:
                    params_cont = deepcopy(params[node])
                    params_cont['loss'] = 'mse'  # Continuous nodes use mse loss
                    self.models[node] = kan_predictor(**params_cont)
                    self.noise_models[node] = gcm.ScipyDistribution(norm)

    def fit(self, data):
        if self.verbose>0:
            print("Fitting KAN model...")
        for node in self.nodes:
            if self.verbose>0:
                print('Fitting node {}...'.format(node))
            if len(self.parents[node]) > 0:
                X = data[self.parents[node]].to_numpy()
                Y = data[node].to_numpy()
                if len(Y.shape) == 1:
                    Y = Y.reshape(-1, 1)
                self.models[node].fit(X, Y)
                # Now, fit the noise model if the node is continuous
                if self.node_types[node] == 'continuous':
                    residuals = Y.flatten() - self.models[node].predict(X).flatten()
                    self.noise_models[node].fit(residuals)
            else:
                self.models[node].fit(data[node].to_numpy())


    def draw_samples(self, num_samples, seed=42):
        np.random.seed(seed)
        data = {node: np.zeros(num_samples) for node in self.nodes}
        for node in nx.topological_sort(self.graph):
            if len(self.parents[node]) == 0:
                data[node] = self.models[node].draw_samples(num_samples).flatten()
            else:
                X = np.array([data[parent] for parent in self.parents[node]]).T
                if self.node_types[node] == "continuous":
                    y_pred = self.models[node].predict(X).flatten()
                    data[node] = y_pred + self.noise_models[node].draw_samples(num_samples).flatten()
                else:
                    y_pred_proba = self.models[node].predict_probabilities(X)
                    # Sample from the predicted probabilities
                    data_aux = np.zeros(num_samples)
                    for i in range(num_samples):
                        # if the vector is Nx1, then is binary, and we should have Nx2, with 1-p, p
                        if y_pred_proba.shape[1] == 1:
                            y_pred_proba_new = np.zeros((y_pred_proba.shape[0], 2))
                            y_pred_proba_new[:, 0] = 1 - y_pred_proba[:, 0]
                            y_pred_proba_new[:, 1] = y_pred_proba[:, 0]
                            y_pred_proba = y_pred_proba_new

                        data_aux[i] = np.random.choice(self.models[node].hyperparameters['num_classes'], p=y_pred_proba[i])

                    data[node] = data_aux
        return pd.DataFrame(data)

    def interventional_samples(self, intervention, num_samples_to_draw, seed=42):
        np.random.seed(seed) # For reproducibility
        data = {node: np.zeros(num_samples_to_draw) for node in self.nodes}
        for node in nx.topological_sort(self.graph):
            if node in intervention:
                # Set the node to the intervention value
                data_parents = pd.DataFrame({parent: data[parent] for parent in self.parents[node]})
                if len(data_parents) == 0:
                    data_parents = pd.DataFrame(np.ones((num_samples_to_draw, 1)))  # This is to have some data to pass to the intervention, note that the intervention is supposed to be a constant!!
                intervention_data = data_parents.apply(intervention[node], axis=1)
                data[node] = intervention_data
            elif len(self.parents[node]) == 0:
                data[node] = self.models[node].draw_samples(num_samples_to_draw).flatten()
            else:
                X = np.array([data[parent] for parent in self.parents[node]]).T
                if self.node_types[node] == "continuous":
                    y_pred = self.models[node].predict(X).flatten()
                    # Add noise
                    data[node] = y_pred + self.noise_models[node].draw_samples(num_samples_to_draw).flatten()
                else:
                    y_pred_proba = self.models[node].predict_probabilities(X)
                    # Sample from the predicted probabilities
                    data_aux = np.zeros(num_samples_to_draw)
                    for i in range(num_samples_to_draw):
                        if y_pred_proba.shape[1] == 1:
                            y_pred_proba_new = np.zeros((y_pred_proba.shape[0], 2))
                            y_pred_proba_new[:, 0] = 1 - y_pred_proba[:, 0]
                            y_pred_proba_new[:, 1] = y_pred_proba[:, 0]
                            y_pred_proba = y_pred_proba_new

                        data_aux[i] = np.random.choice(self.models[node].hyperparameters['num_classes'], p=y_pred_proba[i])


                    data[node] = data_aux
        return pd.DataFrame(data)

    def prune(self, data = None, node_th=1e-2, edge_th=3e-2):
        # if data is not none, fit the residuals
        for node in self.nodes:
            if len(self.parents[node]) > 0:
                model = self.models[node].model
                model = model.prune_node(node_th, log_history=False)
                # self.prune_node(node_th, log_history=False)
                model.forward(model.cache_data)
                model.attribute()
                model.prune_edge(edge_th, log_history=False)

                # if data is not none, re-estimate residuals
                if data is not None and self.node_types[node] == 'continuous':
                    X = data[self.parents[node]].to_numpy()
                    Y = data[node].to_numpy()
                    self.noise_models[node] = gcm.ScipyDistribution(norm)
                    residuals = Y.flatten() - self.models[node].predict(X).flatten()
                    self.noise_models[node].fit(residuals)

    def to_symbolic(self, data, method='ours', r2_threshold=0.95):

        # method ours means using our symbolic kan regressor.
        # method: original (orig) means using the auto_symbolic method of the original kan_predictor
        self.symbolic_method = method
        for node in self.nodes:
            if len(self.parents[node]) > 0:
                X = data[self.parents[node]].to_numpy()
                Y = data[node].to_numpy()
                if len(Y.shape) == 1:
                    Y = Y.reshape(-1, 1)
                if method == 'ours':
                    # kan_object = self.models[node].model
                    loss = self.models[node].hyperparameters['loss']
                    model = self.models[node].model
                    symb_obj = symbolic_kan_regressor(self.parents[node], [node], loss)
                    symb_obj.fit(model, X, Y, val_split=0.2, r2_threshold=r2_threshold, show_results=False, save_dir=None)
                    self.models[node].model = symb_obj

                else:
                    self.models[node].model.auto_symbolic(r2_threshold=r2_threshold)

                # noise has to be re-estimated
                if self.node_types[node] == 'continuous':
                    self.noise_models[node] = gcm.ScipyDistribution(norm)
                    residuals = Y.flatten() - self.models[node].predict(X).flatten()
                    self.noise_models[node].fit(residuals)

    def mae(self, data, node_list =None, aggregation=None):
        # mean absolute error of the predictions
        '''

        aggregation: mean or sum, returns only one value across all nodes
        else, returns the MAE per node.
        '''

        if node_list is None:
            node_list = self.nodes

        if not isinstance(node_list, list):
            node_list = [node_list]

        mae = {}
        for node in node_list:
            if len(self.parents[node]) > 0:
                X = data[self.parents[node]].to_numpy()
                Y = data[node].to_numpy()
                if len(Y.shape) == 1:
                    Y = Y.reshape(-1, 1)
                y_pred = self.models[node].predict(X).flatten()
                mae[node] = mean_absolute_error(Y.flatten(), y_pred)

        if aggregation == 'mean':
            return np.mean(list(mae.values()))

        if aggregation == 'sum':
            return np.sum(list(mae.values()))

        return mae

    def get_formulas(self, ex_round=4):
        assert self.symbolic_method is not None, "You need to call to_symbolic before getting the formulas"

        formulas = {}
        for node in self.nodes:
            if len(self.parents[node]) > 0:
                if self.symbolic_method == 'ours':
                    formulas[node] = self.models[node].model.get_formula(ex_round=ex_round)[0]
                else:
                    expr = self.models[node].model.symbolic_formula()[0][0]
                    expr_round = expr
                    for a in sympy.preorder_traversal(expr):
                        if isinstance(a, sympy.Float):
                            expr_round = expr_round.subs(a, round(a, ex_round))
                    formulas[node] = expr_round
        return formulas

    def get_residuals(self, data, node_list=None):
        if node_list is None:
            # only continuous variables
            node_list = []
            for node in self.nodes:
                if self.node_types[node] == 'continuous':
                    node_list.append(node)

        residuals = {}
        for node in node_list:
            if len(self.parents[node]) > 0:
                X = data[self.parents[node]].to_numpy()
                Y = data[node].to_numpy()
                if len(Y.shape) == 1:
                    Y = Y.reshape(-1, 1)
                y_pred = self.models[node].predict(X).flatten()
                residuals[node] = Y.flatten() - y_pred
            else:
                residuals[node] = data[node].to_numpy().flatten()

        return  residuals

    def clone(self, map_location="cpu"):
        # 1) structure
        new = self.__class__.__new__(self.__class__)
        new.graph = nx.DiGraph(self.graph)  # copia del grafo (independiente)
        new.nodes = list(self.nodes)
        new.parents = deepcopy(self.parents)
        new.node_types = deepcopy(self.node_types)
        new.num_classes = deepcopy(self.num_classes)

        new.models = {}
        new.noise_models = {}

        # 2) copy
        for node in new.nodes:
            old_m = self.models[node]

            # Root nodes:  scipy
            if len(new.parents[node]) == 0:
                new.models[node] = deepcopy(old_m)
                continue

            # clone predictor
            if isinstance(old_m, kan_predictor):
                new_pred = old_m.clone(map_location=map_location)
                new.models[node] = new_pred
            else:
                # Fallback:
                new.models[node] = deepcopy(old_m)

            if node in getattr(self, "noise_models", {}):
                new.noise_models[node] = deepcopy(self.noise_models[node])

        return new

    def evaluate_symbolic(self, data, nodes):
        assert self.symbolic_method is not None, "You need to call to_symbolic before evaluating the formulas"
        if not isinstance(nodes, list):
            nodes = [nodes]
        values = {}
        for node in nodes:
            _data = data.copy()
            if self.symbolic_method == 'ours':
                formula = self.models[node].model.get_formula()[0]

            else:
                formula = self.models[node].model.symbolic_formula()[0][0]
                # numbers like x1 are x_1 in the formula of KAN, so we replace the columns of the dataframe
                _data.columns = [col.replace('x', 'x_') for col in data.columns]

            import sympy as sp
            syms = sorted(formula.free_symbols, key=lambda s: s.name)  # stable order
            f = sp.lambdify(syms, formula, 'numpy')
            vals = f(*[_data[s.name].to_numpy() for s in syms])
            values[node] = vals
        return values

    def predict_node(self, data, node, proba=False, logits=False):
        if logits:
            assert proba is True
        if len(self.parents[node]) == 0:
            return self.models[node].draw_samples(len(data)).flatten()
        else:
            X = data[self.parents[node]].to_numpy()
            if self.node_types[node] == "continuous":
                y_pred = self.models[node].predict(X).flatten()
                return y_pred
            else:
                y_pred_proba = self.models[node].predict_probabilities(X, logits=logits)
                # Get the class with highest probability

                if proba:
                    return y_pred_proba

                y_pred = np.argmax(y_pred_proba, axis=1)
                return y_pred





class kan_predictor(object):
    """
    Class for KAN model, it includes the definition of the model and functions to run the model
    """

    def __init__(self, hidden_dim=0, batch_size=500, grid=1, k=1, seed=0, lr=0.01, early_stop=True, steps=10000,
                    lamb=0.1, lamb_entropy=0.1, sparse_init=False, mult_kan=False, try_gpu=False, loss='mse', num_classes=None,
                 verbose=0, checkpoint_dir=None):

        self.hyperparameters = {'hidden_dim': hidden_dim, 'batch_size': batch_size, 'grid': grid, 'k': k, 'seed': seed,
                                'lr': lr, 'early_stop': early_stop, 'steps': steps, 'lamb': lamb, 'lamb_entropy': lamb_entropy,
                                'sparse_init': sparse_init, 'mult_kan': mult_kan, 'try_gpu': try_gpu, 'loss': loss, 'num_classes': num_classes,
                                'verbose': verbose, 'checkpoint_dir': checkpoint_dir}
        #self.classes = [0, 1]  # Note that this is only used by DoWhy for binary classification

    def seed_all(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def fit(self, X, Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        self.set_model(x_train, x_test, y_train, y_test)
        # Train the model
        #self.model.save_act = True
        #self.model.speed()
        results = self.custom_fit(self.dataset, batch=self.hyperparameters["batch_size"],
                                  steps=self.hyperparameters["steps"], lamb=self.hyperparameters["lamb"],
                                  lamb_entropy=self.hyperparameters["lamb_entropy"], lr=self.hyperparameters["lr"],
                                  early_stop=self.hyperparameters["early_stop"], patience=10,
                                  save_fig=False, verbose=self.hyperparameters['verbose'])

        return {'model': self.model, 'y_pred': self.predict(self.x_test), 'train_results': results}

    def get_params(self, deep=False):
        return self.hyperparameters

    def set_params(self, hidden_dim=0, batch_size=500, grid=1, k=1, seed=0, lr=0.01, early_stop=True, steps=10000,
                    lamb=0.1, lamb_entropy=0.1, sparse_init=False, mult_kan=False, try_gpu=False, loss='mse', num_classes=None,
                    verbose=0, checkpoint_dir=None):
        self.hyperparameters = {'hidden_dim': hidden_dim, 'batch_size': batch_size, 'grid': grid, 'k': k, 'seed': seed,
                                'lr': lr, 'early_stop': early_stop, 'steps': steps, 'lamb': lamb, 'lamb_entropy': lamb_entropy,
                                'sparse_init': sparse_init, 'mult_kan': mult_kan, 'try_gpu': try_gpu, 'loss': loss, 'num_classes': num_classes,
                                'verbose': verbose, 'checkpoint_dir': checkpoint_dir}
        return self

    def set_model(self, x_train, x_test, y_train, y_test):

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.hyperparameters['try_gpu'] else "cpu")
        print("Using device:", self.device)

        dt_now = str(datetime.datetime.now()).replace(" ", "_")
        save_folder = f"kan_{self.hyperparameters['hidden_dim']}_{self.hyperparameters['grid']}_{self.hyperparameters['k']}_{self.hyperparameters['lr']}_{self.hyperparameters['lamb']}_{self.hyperparameters['mult_kan']}_{dt_now}"

        self.seed_all(self.hyperparameters["seed"])

        n_targets = 1 if len(y_test.shape) == 1 else y_test.shape[1]
        if self.hyperparameters["loss"] == 'discrete':
            n_targets = self.hyperparameters["num_classes"]

        input_size = x_train.shape[1]

        if self.hyperparameters["hidden_dim"] == 0:
            self.width = [input_size, n_targets]
        elif isinstance(self.hyperparameters["hidden_dim"], int):
            self.width = [input_size, self.hyperparameters["hidden_dim"], n_targets]
        else:
            self.width = [input_size] + self.hyperparameters["hidden_dim"] + [n_targets]

        if self.hyperparameters['mult_kan']:  # Add multiplication nodes
            new_width = []
            for i in range(len(self.width)):
                if i == 0 or i == len(self.width) - 1:
                    new_width.append(self.width[i])
                else:
                    new_width.append([self.width[i], self.width[i]])
            self.width = new_width

        checkpoint_root = self.hyperparameters.get("checkpoint_dir")
        if checkpoint_root is None:
            checkpoint_root = Path(__file__).resolve().parents[2] / "outputs" / "checkpoints" / "kan"
        checkpoint_root = Path(checkpoint_root)
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        self.model = KAN(
            width=self.width,
            grid=self.hyperparameters["grid"],
            k=self.hyperparameters["k"],
            seed=0,
            device=self.device,
            sparse_init=self.hyperparameters["sparse_init"],
            ckpt_path=os.fspath(checkpoint_root / save_folder),
            auto_save=False,
        )

        self.x_train = torch.from_numpy(x_train).to(self.device).float()
        self.y_train = torch.from_numpy(y_train).to(self.device).float()
        self.x_test = torch.from_numpy(x_test).to(self.device).float()
        self.y_test = torch.from_numpy(y_test).to(self.device).float()
        if self.hyperparameters['loss'] == 'discrete':
            self.y_train = torch.squeeze(self.y_train.long())
            self.y_test = torch.squeeze(self.y_test.long())
        # Note that KAN interface uses "test" for what we call "val": we reverse here for consistency
        self.dataset = {'train_input': self.x_train, 'train_label': self.y_train,
                        'test_input': self.x_test, 'test_label': self.y_test}
        if self.hyperparameters['loss'] == 'mse':
            self.criterion = nn.MSELoss()
        elif self.hyperparameters['loss'] == 'discrete':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss {self.hyperparameters['loss']} not recognized")


    def predict(self, X):
        if not torch.is_tensor(X):
            X = torch.from_numpy(X.copy()).to(self.device).float()
        y_pred = self.model.forward(X)
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        if self.hyperparameters['loss'] == 'discrete':
            y_pred_proba = self.predict_probabilities(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def predict_probabilities(self, X, logits=False):
        assert self.hyperparameters['loss'] == 'discrete', "predict_probabilities is only available for discrete classification"
        if not torch.is_tensor(X):
            X = torch.from_numpy(X.copy()).to(self.device).float()
        y_pred_proba = self.model.forward(X)
        if not torch.is_tensor(y_pred_proba):
            y_pred_proba = torch.from_numpy(y_pred_proba).to(self.device).float()
        else:
            y_pred_proba = y_pred_proba.detach()
        if logits:
            return y_pred_proba.cpu().numpy()
        softmax = torch.nn.Softmax(dim=1)
        y_pred_proba = softmax(y_pred_proba).cpu().numpy()
        return y_pred_proba

    def custom_fit(self, dataset, steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0.,
                   lamb_coefdiff=0., update_grid=True, grid_update_num=10, lr=1., start_grid_update_step=-1,
                   stop_grid_update_step=50, batch=-1,
                   save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1,
                   img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n',
                   display_metrics=None, early_stop=True, patience=30, verbose=0):

        if lamb > 0. and not self.model.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')

        old_save_act, old_symbolic_enabled = self.model.disable_symbolic_in_fit(lamb)

        if verbose > 0:
            pbar = tqdm(range(steps), desc='description', ncols=100)
        else:
            pbar = range(steps)

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        optimizer = torch.optim.Adam(self.model.get_params(), lr=lr)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        results['train_metrics'] = []
        results['test_metrics'] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
        else:
            batch_size = batch
        batch_size_test = dataset['test_input'].shape[0]

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        best_loss = np.inf
        patience_counter = 0

        for _ in pbar:

            if _ == steps - 1 and old_save_act:
                self.model.save_act = True

            if save_fig and _ % save_fig_freq == 0:
                save_act = self.model.save_act
                self.model.save_act = True

            n_batches_train = len(dataset['train_input']) // batch_size

            for ibt in range(n_batches_train):

                batch_start = ibt * batch_size
                batch_end = min((ibt + 1) * batch_size, len(dataset['train_input']))
                train_id = np.arange(batch_start, batch_end)

                if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                    self.model.update_grid(dataset['train_input'][train_id])

                pred_train = self.model.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = self.criterion(pred_train, dataset['train_label'][train_id])
                if self.model.save_act:
                    if reg_metric == 'edge_backward':
                        self.model.attribute()
                    if reg_metric == 'node_backward':
                        self.model.node_attribute()
                    reg_ = self.model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred_test = self.model.forward(dataset['test_input'])
            test_loss = self.criterion(pred_test, dataset['test_label'])

            # For conveniency, we get train loss and reg on the last batch only

            results['train_loss'].append(train_loss.cpu().detach().numpy())
            results['test_loss'].append(test_loss.cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if _ % log == 0 and verbose > 0:
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (
                    torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(),
                    reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and _ % save_fig_freq == 0:
                self.model.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_),
                                beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
                self.model.save_act = save_act

            if early_stop:
                if results['test_loss'][-1] < best_loss:
                    best_loss = results['test_loss'][-1]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        print(f'Early stopping at step {_}')
                        break

        self.model.log_history('fit')
        # revert back to original state
        self.model.symbolic_enabled = old_symbolic_enabled
        return results

    def prune(self):
        self.model = self.model.prune()

    def clone(self, map_location="cpu"):
        new = self.__class__.__new__(self.__class__)
        new.hyperparameters = deepcopy(self.hyperparameters)

        if not hasattr(self, "model"):
            return new


        new.device = torch.device(map_location)

        if hasattr(self, "width"):
            new.width = deepcopy(self.width)

        new.model = KAN(
            width=deepcopy(self.model.width) if hasattr(self.model, "width") else deepcopy(self.width),
            grid=self.hyperparameters["grid"],
            k=self.hyperparameters["k"],
            seed=0,
            device=new.device,
            sparse_init=self.hyperparameters["sparse_init"],
            ckpt_path=getattr(self.model, "ckpt_path", None),
            auto_save=False
        )

        # Copiar pesos sin compartir referencias
        sd = self.model.state_dict()
        sd = {k: v.detach().clone().to(new.device) for k, v in sd.items()}
        new.model.load_state_dict(sd)

        # Copia del resto de atributos que influyen en inferencia
        # (si los usas después de entrenar)
        for attr in ["criterion"]:
            if hasattr(self, attr):
                setattr(new, attr, deepcopy(getattr(self, attr)))

        return new


def kan_auto_symbolic_formula(kan_object, X, x_names, ex_round=4, weight_simple=0.0):
    _ = kan_object.forward(torch.from_numpy(X.copy()).to(kan_object.device).float()).detach().cpu().numpy()  # Just to ensure that acts and post_splines are computed
    kan_object.auto_symbolic(weight_simple=weight_simple)  # Add the lib in here if desired, note that simple_weight is set to 0.0 to avoid the simple symbolic formula!
    expr = kan_object.symbolic_formula(var=x_names)[0]
    # Round the floats in the expression
    expr_round = []
    for i in range(len(expr)):
        e  = expr[i]
        for a in sympy.preorder_traversal(e):
            if isinstance(a, sympy.Float):
                e = e.subs(a, round(a, ex_round))
        expr_round.append(e)
    return expr_round


class symbolic_kan_regressor(object):  # Class implemented for symbolic regression with single-layer KAN objects
    def __init__(self, x_names=None, y_names=None, loss='mse'):
        self.x_names = x_names
        if isinstance(self.x_names, list):
            # replace all spaces in the names
            self.x_names = [name.replace(" ", "_") for name in x_names]
        self.y_names = y_names
        self.symbolic_functions = None
        self.epsilon = 1e-10 # Small value to prevent division by zero
        #self.complex_functions = ['triangle', 'sqrt', 'inv_sqrt', 'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'tanh', 'sgn', 'arccos', 'arctan', 'arctanh']  # Functions that require 4 parameters: a, b, c, d
        self.complex_functions = ['triangle', 'sqrt', 'inv_sqrt', 'exp', 'log', 'abs', 'sin', 'cos', 'tan', 'tanh', 'sgn', 'arctan']  # Functions that require 4 parameters: a, b, c, d (we drop arccos and arctanh due to numerical issues)
        self.loss = loss
        self.n_outputs = len(y_names)  # Assume that each output is scalar-valued

    def get_formula(self, ex_round=4):
        assert self.symbolic_functions is not None, "You must fit the model before getting the formula"
        # Create a sympy expression for each output

        n_outputs = max([j for _, _, _, j in self.symbolic_functions]) + 1
        n_inputs = max([i for _, _, i, _ in self.symbolic_functions]) + 1

        assert n_outputs * n_inputs == len(self.symbolic_functions), "The number of symbolic functions does not match the number of inputs and outputs"

        formulas = []
        for j in range(n_outputs):
            expr = 0
            for fun_name, params, i, j2 in self.symbolic_functions:
                if j2 == j:
                    x = sympy.symbols(self.x_names[i] if self.x_names is not None else f'x{i+1}')
                    if fun_name == 'polynomial':
                        expr += sum([p * x**k for k, p in enumerate(params)])
                    elif fun_name == 'inv_polynomial':
                        expr += sum([p * (1 / (x + self.epsilon))**k for k, p in enumerate(params)])
                    elif fun_name in self.complex_functions:
                        a, b, c, d = params[0], params[1], params[2], params[3]
                        if fun_name == 'triangle':
                            # Parameters are as follows: a (first slope), b (y offset of first slope), c (second slope), d (x offset of second slope)
                            # Compute the x value where the triangle changes slope
                            x_th = (d - b) / (a - c + self.epsilon)  # Avoid division by zero
                            expr += sympy.Piecewise((a * x + b, x < x_th), (c * x + d, True))
                        elif fun_name == 'exp':
                            expr += c * sympy.exp(a * x + b) + d
                        elif fun_name == 'sqrt':
                            expr += c * sympy.sqrt(sympy.Abs(a * x + b) + self.epsilon) + d
                        elif fun_name == 'inv_sqrt':
                            expr += c / (sympy.sqrt(sympy.Abs(a * x + b) + self.epsilon)) + d
                        elif fun_name == 'log':
                            expr += c * sympy.log(sympy.Abs(a * x + b) + self.epsilon) + d
                        elif fun_name == 'abs':
                            expr += c * sympy.Abs(a * x + b) + d
                        elif fun_name == 'sin':
                            expr += c * sympy.sin(a * x + b) + d
                        elif fun_name == 'cos':
                            expr += c * sympy.cos(a * x + b) + d
                        elif fun_name == 'tan':
                            expr += c * sympy.tan(a * x + b) + d
                        elif fun_name == 'tanh':
                            expr += c * sympy.tanh(a * x + b) + d
                        elif fun_name == 'sgn':
                            expr += c * sympy.sign(a * x + b) + d
                        elif fun_name == 'arccos':
                            #expr += c * sympy.acos(sympy.Max(sympy.Min(a * x + b, 1 - self.epsilon), -1 + self.epsilon)) + d
                            expr += c * sympy.acos(a * x + b) + d
                        elif fun_name == 'arctan':
                            expr += c * sympy.atan(a * x + b) + d
                        elif fun_name == 'arctanh':
                            #expr += c * sympy.atanh(sympy.Max(sympy.Min(a * x + b, 1 - self.epsilon), -1 + self.epsilon)) + d
                            expr += c * sympy.atanh(a * x + b) + d
                    else:
                        raise ValueError(f"Function {fun_name} not implemented")

            expr_round = expr
            for a in sympy.preorder_traversal(expr):
                if isinstance(a, sympy.Float):
                    expr_round = expr_round.subs(a, round(a, ex_round))
            formulas.append(expr_round)
        return formulas

    def predict_individual(self, x, fun_name, params):

        if fun_name == 'polynomial':
            p = np.polynomial.Polynomial(params)
            return p(x)
        elif fun_name == 'inv_polynomial':
            p = np.polynomial.Polynomial(params)
            return p(1 / (x + self.epsilon))  # Avoid division by zero
        elif fun_name in self.complex_functions:
            a, b, c, d = params[0], params[1], params[2], params[3]
            if fun_name == 'triangle':
                # Parameters are as follows: a (first slope), b (y offset of first slope), c (second slope), d (x offset of second slope)
                # Compute the x value where the triangle changes slope
                x_th = (d - b) / (a - c + self.epsilon)  # Avoid division by zero
                return np.where(x < x_th, a * x + b, c * x + d)
            elif fun_name == 'exp':
                return c * np.exp(a * x + b) + d
            elif fun_name == 'sqrt':
                return c * np.sqrt(np.abs(a * x + b) + self.epsilon) + d
            elif fun_name == 'inv_sqrt':
                return c / (np.sqrt(np.abs(a * x + b) + self.epsilon)) + d
            elif fun_name == 'log':
                return c * np.log(np.abs(a * x + b) + self.epsilon) + d
            elif fun_name == 'abs':
                return c * np.abs(a * x + b) + d
            elif fun_name == 'sin':
                return c * np.sin(a * x + b) + d
            elif fun_name == 'cos':
                return c * np.cos(a * x + b) + d
            elif fun_name == 'tan':
                return c * np.tan(a * x + b) + d
            elif fun_name == 'tanh':
                return c * np.tanh(a * x + b) + d
            elif fun_name == 'sgn':
                return c * np.sign(a * x + b) + d
            elif fun_name == 'arccos':
                return c * np.arccos(np.clip(a * x + b, -1 + self.epsilon, 1 - self.epsilon)) + d
            elif fun_name == 'arctan':
                return c * np.arctan(a * x + b) + d
            elif fun_name == 'arctanh':
                return c * np.arctanh(np.clip(a * x + b, -1 + self.epsilon, 1 - self.epsilon)) + d
        else:
            raise ValueError(f"Function {fun_name} not implemented")

    def predict(self, x):

        assert self.symbolic_functions is not None, "You must fit the model before predicting"
        y_pred = np.zeros((x.shape[0], self.n_outputs))
        for fun_name, params, i, j in self.symbolic_functions:
            x_i = x[:, i]
            y_pred[:, j] += self.predict_individual(x_i, fun_name, params)
        return y_pred

    def forward(self, x):
        return self.predict(x)

    def get_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) # Loss over the logits in case of binary!
        '''
        if self.loss == 'mse':
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        else:
            # Different metrics for classification!
            y_true = y_true.astype(int)
            y_proba_1 = np.squeeze(y_pred)
            if len(y_proba_1.shape) > 1:
                y_proba_1 = y_proba_1[:, 1]  # Get the probability of class 1
            mae = roc_auc_score(y_true, y_proba_1)  # Note that y_pred is the probabilities of the binary classes
            y_label = (y_proba_1 > 0.5).astype(int)
            r2 = f1_score(y_true, y_label)  # F1 score as a proxy for R2 in classification
        '''
        return mae, r2

    def fit(self, kan_object, X, Y, val_split=0.2, r2_threshold=0.95, show_results=False, min_param_val=-100, max_param_val=100, save_dir=None):

        assert len(kan_object.width_in) == 2, "This class only supports single-layer KAN objects"

        if self.loss == 'mse':
            assert kan_object.width_out[1] == self.n_outputs, "The number of outputs in the KAN object does not match the number of expected values for Y and the loss function " + str(self.loss)
            binary = False
        else:
            assert kan_object.width_out[1] == 2, "For classification, only binary classification is supported so far"
            binary = True

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=val_split, random_state=0)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        # We must ensure that acts and spline_postacts are computed
        y_pred_test_original = kan_object.forward(torch.from_numpy(x_test).to(kan_object.device).float()).detach().cpu().numpy()
        y_pred_train_original = kan_object.forward(torch.from_numpy(x_train).to(kan_object.device).float()).detach().cpu().numpy()

        self.symbolic_functions = []

        if self.x_names is None:
            self.x_names = [f'x{i+1}' for i in range(kan_object.width_in[0])]
        if self.y_names is None:
            self.y_names = [f'y{j+1}' for j in range(kan_object.width_out[1])]

        l = 0  # Only one layer
        t_init = time.time()
        n_outputs = 1 if binary else kan_object.width_out[l + 1]
        for i in range(kan_object.width_in[l]):  # i indexes the input features
            for j in range(n_outputs): # j indexes the output features

                best_r = -np.inf
                best_fun = -np.inf

                t_iter = time.time()

                x = kan_object.acts[l][:, i].detach().cpu().numpy()
                if np.std(x) < 1e-10:  # Very low std, add a bit of noise and warn
                    print(f"Warning: input {i} has very low std ({np.std(x):.4e}), adding noise to avoid numerical issues")
                    x += np.random.normal(0, 1e-5, size=x.shape)
                if binary:
                    y = kan_object.spline_postacts[l][:, 1, i].detach().cpu().numpy() - kan_object.spline_postacts[l][:, 0, i].detach().cpu().numpy() # MAke y the logit  of the difference!
                else:
                    y = kan_object.spline_postacts[l][:, j, i].detach().cpu().numpy()
                success = False

                def evaluate_params(params, fname):
                    y_pred = self.predict_individual(x, fname, params)
                    mae, r2 = self.get_metrics(y, y_pred)
                    if r2 >= r2_threshold:
                        print(f"Found {fname} of degree {degree} for input {i + 1}/{kan_object.width_in[l]} and output {j + 1}/{kan_object.width_out[l + 1]} with r2 {r2:.4f}. Total time: {time.time() - t_init:.2f}s, Iteration time: {time.time() - t_iter:.2f}s, Average time per output: {(time.time() - t_init) / ((i * kan_object.width_out[l + 1]) + j + 1):.2f}s")
                        self.symbolic_functions.append((fname, params, i, j))
                        return True, mae, r2
                    else:
                        return False, mae, r2

                # Start from simple to more complex functions
                for optimization_strategy in ['polynomial', 'inv_polynomial', 'complex']:
                    if success:
                        break

                    if optimization_strategy == 'polynomial':
                        opt_iter = range(5)
                    elif optimization_strategy == 'inv_polynomial':
                        opt_iter = range(1, 5)
                    else:
                        opt_iter = self.complex_functions

                    if optimization_strategy == 'polynomial':
                        fname = 'polynomial'
                        for degree in opt_iter:
                            params = np.polynomial.Polynomial.fit(x, y, degree).convert().coef
                            success, mae, r2 = evaluate_params(params, fname)
                            if r2 > best_r:
                                best_r = r2
                                best_fun = (fname, params, i, j)
                            if success:
                                break
                    elif optimization_strategy == 'inv_polynomial':
                        fname = 'inv_polynomial'
                        for degree in opt_iter:
                            x_inv = 1 / (x + self.epsilon)  # Avoid division by zero
                            params = np.polynomial.Polynomial.fit(x_inv, y, degree).convert().coef
                            success, mae, r2 = evaluate_params(params, fname)
                            if r2 > best_r:
                                best_r = r2
                                best_fun = (fname, params, i, j)
                            if success:
                                break
                    else:
                        for fname in opt_iter:
                            bounds_nm = [(None, None)] * 4

                            def error_function(params):
                                y_pred = self.predict_individual(x, fname, params)

                                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)): # Penalize invalid predictions
                                    return np.inf
                                mae, r2 = self.get_metrics(y, y_pred)
                                return mae

                            if fname == 'triangle': # Triangle is very sensitive to initial guess, so we first do a coarse grid search to find a good initial guess. If it does not work visually, try changing the grid_size and min/max_grid_val
                                grid_size = 5
                                min_grid_val = -1
                                max_grid_val = 1
                                a_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                b_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                c_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                d_vals = np.linspace(min_grid_val, max_grid_val, grid_size)
                                best_mae = np.inf
                                best_params_grid = None
                                for a in a_vals:
                                    for b in b_vals:
                                        for c in c_vals:
                                            for d in d_vals:
                                                params_grid = np.array([a, b, c, d])
                                                mae_grid = error_function(params_grid)
                                                if mae_grid < best_mae:
                                                    best_mae = mae_grid
                                                    best_params_grid = params_grid
                                initial_guess = best_params_grid
                            else:
                                initial_guess = np.array([1.0, 0.0, 1.0, 0.0])

                            result = minimize(error_function, initial_guess, method='Nelder-Mead', bounds=bounds_nm, options={"maxiter": 100000, "maxfev": 100000})

                            params_opt = result.x

                            success, mae, r2 = evaluate_params(params_opt, fname)
                            if r2 > best_r:
                                best_r = r2
                                best_fun = (fname, params_opt, i, j)
                            if success:
                                break
                if not success:
                    print(f"Could not find a good function for input {i + 1}/{kan_object.width_in[l]} and output {j + 1}/{kan_object.width_out[l + 1]}. Best r2 was {best_r:.4f} with function {best_fun[0]}. Total time: {time.time() - t_init:.2f}s, Iteration time: {time.time() - t_iter:.2f}s, Average time per output: {(time.time() - t_init) / ((i * kan_object.width_out[l + 1]) + j + 1):.2f}s")
                    self.symbolic_functions.append(best_fun)

                if show_results or save_dir is not None:
                    plt.scatter(x, y, label='KAN Spline', alpha=0.5)
                    x_lin = np.linspace(x.min(), x.max(), 100)
                    y_lin = self.predict_individual(x_lin, self.symbolic_functions[-1][0], self.symbolic_functions[-1][1])
                    if binary:
                        y_lin = 1 / (1 + np.exp(-y_lin))  # Apply sigmoid to get probabilities
                    plt.plot(x_lin, y_lin, 'r-', label='Symbolic Fit')
                    plt.xlabel(f"Activation of input {i + 1} (Feature: {self.x_names[i] if self.x_names is not None else i + 1})")
                    plt.ylabel(f"Spline post-activation of output {j + 1} (Target: {self.y_names[j] if self.y_names is not None else j + 1})")
                    plt.title(f"Symbolic Fit for input {self.x_names[i] if self.x_names is not None else i + 1} and output {self.y_names[j] if self.y_names is not None else j + 1}")
                    plt.legend()
                    if save_dir is not None:
                        plt.savefig(os.path.join(save_dir, f'symbolic_fit_input_{i+1}_output_{j+1}.png'), bbox_inches='tight', dpi=300)
                    if show_results:
                        plt.show()
                    plt.close()

        y_pred_symbolic_train = self.predict(x_train)
        y_pred_symbolic_test = self.predict(x_test)

        print("Symbolic regression results:")
        for i, target in enumerate(self.y_names):
            mae_train_original, r2_train_original = self.get_metrics(y_train[:, i], y_pred_train_original[:, i])
            mae_test_original, r2_test_original = self.get_metrics(y_test[:, i], y_pred_test_original[:, i])
            train_mae, train_r2 = self.get_metrics(y_train[:, i], y_pred_symbolic_train[:, i])
            test_mae, test_r2 = self.get_metrics(y_test[:, i], y_pred_symbolic_test[:, i])

            print(f"Target {target}:")
            print(f"  Train MAE: {train_mae:.4f} (Original: {mae_train_original:.4f}), Train R2: {train_r2:.4f} (Original: {r2_train_original:.4f})")
            print(f"  Test MAE: {test_mae:.4f} (Original: {mae_test_original:.4f}), Test R2: {test_r2:.4f} (Original: {r2_test_original:.4f})")

            if save_dir is not None:
                with open(os.path.join(save_dir, 'metrics.txt'), 'a') as f:
                    f.write(f"Target {target}:\n")
                    f.write(f"  Train MAE: {train_mae:.4f} (Original: {mae_train_original:.4f}), Train R2: {train_r2:.4f} (Original: {r2_train_original:.4f})\n")
                    f.write(f"  Test MAE: {test_mae:.4f} (Original: {mae_test_original:.4f}), Test R2: {test_r2:.4f} (Original: {r2_test_original:.4f})\n\n")
                    f.write("----------------------------------------\n")

            if show_results or save_dir is not None:
                plt.figure()
                plt.scatter(y_train[:, i], y_pred_symbolic_train[:, i], label='Train SYM', alpha=0.5)
                plt.scatter(y_test[:, i], y_pred_symbolic_test[:, i], label='Test SYM', alpha=0.5)
                plt.scatter(y_train[:, i], y_pred_train_original[:, i], label='Train Spline', alpha=0.5)
                plt.scatter(y_test[:, i], y_pred_test_original[:, i], label='Test Spline', alpha=0.5)
                plt.plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'k--', label='Ideal')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Symbolic Regression Predictions for {target}')
                plt.legend()
                if save_dir is not None:
                    plt.savefig(os.path.join(save_dir, f'symbolic_regression_predictions_{target}.png'), bbox_inches='tight', dpi=300)
                if show_results:
                    plt.show()
                plt.close()
