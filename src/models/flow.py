from causalflows.flows import CausalMAF, CausalNSF
import networkx as nx
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm
import pandas as pd
import numpy as np

default_params = {
    'flow_type': 'CausalNSF',
    'hidden_dims': [64, 64],
    'base_lr': 1e-3,
    'early_stopping_patience': 30,
    'scheduler': None,
    'batch_size': 256,
    'train_val_split': [0.8, 0.2],
    'max_epochs': 1000,
    'device': 'cpu',
    'bins': 8
}

class causalflow_model(object):

    def __init__(self, graph, params):# Graph is a nx graph, params is a dictionary

        self.graph = graph
        self.flow_type = params['flow_type']
        self.hidden_dims = params['hidden_dims']
        self.base_lr = params['base_lr']
        self.early_stopping_patience = params['early_stopping_patience']
        self.scheduler_name = params['scheduler']
        self.batch_size = params['batch_size']
        self.train_val_test_split = params['train_val_split']
        self.max_epochs = params['max_epochs']
        self.bins = params['bins']


        if torch.cuda.is_available() and params['device'] == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.num_features = len(graph.nodes)

        self.topological_order = list(nx.topological_sort(graph))
        self.adjacency = nx.to_numpy_array(graph, nodelist=self.topological_order).T
        # add ones to the diagonal
        self.adjacency += np.eye(self.num_features)

        if self.flow_type == 'CausalMAF':
            self.flow = CausalMAF(features = self.num_features,
                                  context=0,
                                  adjacency = self.adjacency,
                                  hidden_features = self.hidden_dims)
        elif self.flow_type == 'CausalNSF':
            self.flow = CausalNSF(features = self.num_features,
                                  context=0,
                                  adjacency = self.adjacency,
                                  hidden_features = self.hidden_dims,
                                  bins=self.bins)
        else:
            raise ValueError(f"Unsupported flow type: {self.flow_type}")

        self.optim = optim.Adam(self.flow.parameters(), lr=self.base_lr)
        if self.scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', patience=10, factor=0.90)
        elif self.scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=50, gamma=0.95)
        else:
            self.scheduler = None

    def fit(self, data):# Data is a pandas dataframe with the factual data (each column is a node)

        self.original_order = data.columns.tolist()

        data_adj = data[self.topological_order]

        # split data and make dataloaders
        n = len(data_adj)
        n_train = int(self.train_val_test_split[0] * n)
        n_val = int(self.train_val_test_split[1] * n)
        train_data = data_adj.iloc[:n_train]
        val_data = data_adj.iloc[n_train:]

        train_dataset = TensorDataset(torch.tensor(train_data.values, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(val_data.values, dtype=torch.float32))


        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)


        # define the train loop (not defined in flows)
        self.flow.to(self.device)
        history = {"train_nll": [], "val_nll": [] if val_loader is not None else None, 'lr': []}

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in tqdm(range(self.max_epochs)):
            self.flow.train()
            running, seen = 0.0, 0
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self.device, non_blocking=True)

                self.optim.zero_grad(set_to_none=True)
                loss = -self.flow().log_prob(x).mean()  # mean NLL over batch
                loss.backward()
                self.optim.step()

                running += float(loss.item()) * x.size(0)
                seen += x.size(0)

            train_nll = running / max(seen, 1)
            history["train_nll"].append(train_nll)

            if val_loader is not None:
                self.flow.eval()
                running, seen = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0] if isinstance(batch, (tuple, list)) else batch
                        x = x.to(self.device, non_blocking=True)

                        loss = -self.flow().log_prob(x).mean()  # mean NLL over batch

                        running += float(loss.item()) * x.size(0)
                        seen += x.size(0)

                val_nll = running / max(seen, 1)
                history["val_nll"].append(val_nll)

                # early stopping
                if val_nll < best_val_loss:
                    best_val_loss = val_nll
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

                if self.scheduler is not None:
                    if self.scheduler_name == 'plateau':
                        self.scheduler.step(val_nll)
                    else:
                        self.scheduler.step()

            lrs = [g["lr"] for g in self.optim.param_groups]
            history["lr"].append(lrs[0] if len(lrs) == 1 else lrs)

        return history

    def draw_samples(self, num_samples, seed=42):  # Num samples is an integer, returns a pandas dataframe where each column is a node
        # set the seed
        torch.manual_seed(seed)
        # draw samples
        samples = self.flow().sample((num_samples,))
        samples_df = pd.DataFrame(samples.detach().cpu().numpy(), columns=self.topological_order)
        samples_df = samples_df[self.original_order]
        return samples_df


    def interventional_samples(self, intervention, num_samples_to_draw, seed=42):  # Intervention is a dictionary such as {'x1': lambda x:2}, num_samples_to_draw ia an integer, returns a pandas dataframe
        # set the seed
        torch.manual_seed(seed)

        # get the intervened variables and their new values
        intervened_vars = list(intervention.keys())
        intervened_indices = [self.topological_order.index(var) for var in intervened_vars]
        intervened_values = [float(intervention[var](None)) for var in intervened_vars]
        # draw samples
        samples = self.flow().sample_interventional(index=intervened_indices, value=intervened_values, sample_shape=(num_samples_to_draw,))
        samples_df = pd.DataFrame(samples.detach().cpu().numpy(), columns=self.topological_order)
        samples_df = samples_df[self.original_order]
        return samples_df

    def counterfactual_samples(self, intervention, factual_samples, seed=42):  # Intervention is a dictionary such as {'x1': lambda x:2}, factual samples is a pandas df with the factual observed samples, returns a pandas dataframe
        # set the seed
        torch.manual_seed(seed)

        # get the intervened variables and their new values
        intervened_vars = list(intervention.keys())
        intervened_indices = [self.topological_order.index(var) for var in intervened_vars]
        intervened_values = [float(intervention[var](None)) for var in intervened_vars]
        # prepare factual samples
        factual_samples_adj = factual_samples[self.topological_order]
        factual_tensor = torch.tensor(factual_samples_adj.values, dtype=torch.float32).to(self.device)
        # draw samples

        cf_samples = self.flow().compute_counterfactual(factual=factual_tensor,
                                                      index=intervened_indices,
                                                      value=intervened_values)

        cf_samples_df = pd.DataFrame(cf_samples.detach().cpu().numpy(), columns=self.topological_order)
        cf_samples_df = cf_samples_df[self.original_order]
        return cf_samples_df



if __name__ == "__main__":

    params = {
        'flow_type': 'CausalNSF',
        'hidden_dims': [64, 64],
        'base_lr': 1e-3,
        'early_stopping_patience': 30,
        'scheduler': 'plateau',
        'batch_size': 256,
        'train_val_split': [0.8, 0.2],
        'max_epochs': 1000,
        'device': 'cpu',
        'bins': 4
    }
    # test the model

    e1 = torch.randn(1000)
    x1 = e1
    e2 = torch.randn(1000)
    x2 = 2 * x1 + e2
    e3 = torch.randn(1000)
    x3 = x1 + 0.5 * x2 + e3
    # data
    # data = pd.DataFrame({
    #     'x1': x1,
    #     'x3': x3,
    #     'x2': x2,
    # })
    #
    # exo = pd.DataFrame({
    #     'e1': e1,
    #     'e3': e3,
    #     'x2': x2,
    # })

    data = pd.DataFrame({
        'x1': x1,
        'x3': x3,
        'x2': x2,
    })

    exo = pd.DataFrame({
        'e1': e1,
        'e3': e3,
        'x2': x2,
    })

    print('topological order')


    # graph
    graph = nx.DiGraph()
    graph.add_edges_from([('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')])

    # model
    model = causalflow_model(graph, params)

    print('topological order')
    print(model.topological_order)


    print('adjacency')
    print(model.adjacency)
    # fit
    h = model.fit(data)

    # plot the training history
    import matplotlib.pyplot as plt
    plt.plot(h['train_nll'], label='train_nll')
    plt.plot(h['val_nll'], label='val_nll')
    plt.legend()
    plt.show()
    # draw samples
    samples = model.draw_samples(10)
    print('observational data generated')
    print(samples)
    # interventional samples
    intervention = {'x2': lambda x: 1.0}
    inter_samples = model.interventional_samples(intervention, 10)
    print('interventional data')
    print(inter_samples)
    # counterfactual samples
    factual_samples = data.iloc[:10]
    cf_samples = model.counterfactual_samples(intervention, factual_samples)
    print('cfactual data')
    print(cf_samples)

    print('factual data')
    print(pd.concat((data.iloc[:10], exo[:10]), axis=1))
