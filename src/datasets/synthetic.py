import numpy as np
import pandas as pd
import networkx as nx

def sigmoid(x):
    mask = x >= 0
    z = np.zeros_like(x)
    z[mask] = np.exp(-x[mask])
    z[~mask] = np.exp(x[~mask])

    x2 = np.zeros_like(x)
    x2[mask] = 1 / (1 + z[mask])
    x2[~mask] = z[~mask] / (1 + z[~mask])
    return x2


class graph_data(object):
    def __init__(self, name='chain-linear'):
        self.name = name

    def standardize(self, x):
        mean_x = np.mean(x)
        std_x = np.std(x)
        x_std = (x - mean_x) / std_x
        return x_std, mean_x, std_x

    def generate(self, num_samples=100, seed=None, discrete=False, alpha=None, return_u=False):
        # Note that most graphs are obtained from "Causal normalizing flows: from theory to practice"
        if seed is not None:
            np.random.seed(seed)

        if self.name == '3-chain-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 10 * x1 - u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.25 * x2 + 2 * u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)

            def cf_x1_3chainlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 10 * x1_cf - u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.25 * x2_cf + 2 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            def cf_x2_3chainlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 0.25 * x2_cf + 2 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            var_names = ['x1', 'x2', 'x3']
            graph = nx.DiGraph([('x1', 'x2'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3]).T, columns=var_names)
            cfs = [cf_x1_3chainlinear(-1), cf_x1_3chainlinear(0), cf_x1_3chainlinear(1), cf_x2_3chainlinear(-1), cf_x2_3chainlinear(0), cf_x2_3chainlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (10 * x1 - mean_x2) / std_x2, 'x3': lambda x2: (0.25 * x2 - mean_x3) / std_x3,
                       'x2_str': f'(10 * x1 - {mean_x2})/{std_x2}', 'x3_str': f'(0.25 * x2 - {mean_x3})/{std_x3}'}


        elif self.name == '3-chain-non-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = np.exp(x1 / 2.0) + u2 / 4.0
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = (x2 - 5) ** 3 / 15.0 + u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)

            def cf_x1_3chainnonlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = np.exp(x1_cf / 2.0) + u2 / 4.0
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = (x2_cf - 5) ** 3 / 15.0 + u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            def cf_x2_3chainnonlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = (x2_cf - 5) ** 3 / 15.0 + u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            var_names = ['x1', 'x2', 'x3']
            graph = nx.DiGraph([('x1', 'x2'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3]).T, columns=var_names)
            cfs = [cf_x1_3chainnonlinear(-1), cf_x1_3chainnonlinear(0), cf_x1_3chainnonlinear(1), cf_x2_3chainnonlinear(-1), cf_x2_3chainnonlinear(0), cf_x2_3chainnonlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (np.exp(x1 / 2.0) - mean_x2) / std_x2, 'x3': lambda x2: ((x2 - 5) ** 3 / 15.0 - mean_x3 ) / std_x3,
                       'x2_str': f'(exp(x1 / 2.0) - {mean_x2}) / {std_x3}', 'x3_str': f'((x2 - 5) ** 3 / 15.0 - {mean_x3}) / {std_x3}'}

        elif self.name == '4-chain-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            u4 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 5 * x1 - u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = -0.5 * x2 - 1.5 * u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)
            x4 = x3 + u4
            x4, mean_x4, std_x4 = self.standardize(x4)

            def cf_x1_4chainlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 5 * x1_cf - u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = -0.5 * x2_cf - 1.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = x3_cf + u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            def cf_x2_4chainlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = -0.5 * x2_cf - 1.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = x3_cf + u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T


            var_names = ['x1', 'x2', 'x3', 'x4']
            graph = nx.DiGraph([('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4')])
            data = pd.DataFrame(np.vstack([x1, x2, x3, x4]).T, columns=var_names)
            cfs = [cf_x1_4chainlinear(-1), cf_x1_4chainlinear(0), cf_x1_4chainlinear(1), cf_x2_4chainlinear(-1), cf_x2_4chainlinear(0), cf_x2_4chainlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (5 * x1 - mean_x2)/std_x2, 'x3': lambda x2: (-0.5 * x2 - mean_x3)/std_x3, 'x4': lambda x3: (x3 - mean_x4) / std_x4,
                       'x2_str': f'(5 * x1 - {mean_x2})/{std_x2}', 'x3_str': f'(-0.5 * x2 - {mean_x3})/{std_x3}', 'x4_str': f'(x3 - {mean_x4})/{std_x4}'}

        elif self.name == '5-chain-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            u4 = np.random.normal(0, 1, num_samples)
            u5 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 10 * x1 - u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.25 * x2 +2 * u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)
            x4 = x3 + u4
            x4, mean_x4, std_x4 = self.standardize(x4)
            x5 = - x4 + u5
            x5, mean_x5, std_x5 = self.standardize(x5)

            def cf_x1_5chainlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 10 * x1_cf - u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.25 * x2_cf + 2 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = x3_cf + u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                x5_cf = - x4_cf + u5
                x5_cf = (x5_cf - mean_x5) / std_x5
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf, x5_cf]).T

            def cf_x2_5chainlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 0.25 * x2_cf + 2 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = x3_cf + u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                x5_cf = - x4_cf + u5
                x5_cf = (x5_cf - mean_x5) / std_x5
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf, x5_cf]).T

            var_names = ['x1', 'x2', 'x3', 'x4', 'x5']
            graph = nx.DiGraph([('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4'), ('x4', 'x5')])
            data = pd.DataFrame(np.vstack([x1, x2, x3, x4, x5]).T, columns=var_names)
            cfs = [cf_x1_5chainlinear(-1), cf_x1_5chainlinear(0), cf_x1_5chainlinear(1), cf_x2_5chainlinear(-1), cf_x2_5chainlinear(0), cf_x2_5chainlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (10 * x1 - mean_x2) / std_x2, 'x3': lambda x2: (0.25 * x2 - mean_x3) / std_x3, 'x4': lambda x3: (x3 - mean_x4) / std_x4, 'x5': lambda x4: (-x4 - mean_x5) / std_x5,
                       'x2_str': f'(10 * x1 - {mean_x2}) / {std_x2}', 'x3_str': f'(0.25 * x2 - {mean_x3})/{std_x3}', 'x4_str': f'(x3 - {mean_x4})/{std_x4}', 'x5_str': f'(-x4 - {mean_x5})/{std_x5}'}

        elif self.name == 'collider-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 2 - u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.25 * x2 - 0.5 * x1 + 0.5 * u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)

            def cf_x1_colliderlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 2 - u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.25 * x2_cf - 0.5 * x1_cf + 0.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            def cf_x2_colliderlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 0.25 * x2_cf - 0.5 * x1_cf + 0.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            var_names = ['x1', 'x2', 'x3']
            graph = nx.DiGraph([('x1', 'x3'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3]).T, columns=var_names)
            cfs = [cf_x1_colliderlinear(-1), cf_x1_colliderlinear(0), cf_x1_colliderlinear(1), cf_x2_colliderlinear(-1), cf_x2_colliderlinear(0), cf_x2_colliderlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (2 - mean_x2) / std_x2, 'x3': lambda x1, x2: (0.25 * x2 - 0.5 * x1 - mean_x3)/std_x3,
                       'x2_str': f'(2-{mean_x2})/{std_x2}', 'x3_str': f'(0.25 * x2 - 0.5 * x1 - {mean_x3})/{std_x3}'}

        elif self.name == 'fork-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            u4 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 2 - u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.25 * x2 - 1.5 * x1 + 0.5 * u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)
            x4 = x3 + 0.25 * u4
            x4, mean_x4, std_x4 = self.standardize(x4)

            def cf_x1_forklinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 2 - u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.25 * x2_cf - 1.5 * x1_cf + 0.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = x3_cf + 0.25 * u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            def cf_x2_forklinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 0.25 * x2_cf - 1.5 * x1_cf + 0.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = x3_cf + 0.25 * u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            var_names = ['x1', 'x2', 'x3', 'x4']
            graph = nx.DiGraph([('x1', 'x3'), ('x2', 'x3'), ('x3', 'x4')])
            data = pd.DataFrame(np.vstack([x1, x2, x3, x4]).T, columns=var_names)
            cfs = [cf_x1_forklinear(-1), cf_x1_forklinear(0), cf_x1_forklinear(1), cf_x2_forklinear(-1), cf_x2_forklinear(0), cf_x2_forklinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (2 - mean_x2) /std_x2, 'x3': lambda x1, x2: (0.25 * x2 - 1.5 * x1 - mean_x3) / std_x3, 'x4': lambda x3: (x3 - mean_x4) / std_x4,
                       'x2_str': f'(2 - {mean_x2})/{std_x2}', 'x3_str': f'(0.25 * x2 - 1.5 * x1 - {mean_x3})/{std_x3}', 'x4_str': f'(x3-{mean_x4})/{std_x4}'}

        elif self.name == 'fork-non-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            u4 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = u2
            x3 = 4 / (1 + np.exp(- x1 - x2)) - x2 ** 2 + 0.5 * u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)
            x4 = 20 / (1 + np.exp(0.5 * x3 ** 2 - x3)) + u4
            x4, mean_x4, std_x4 = self.standardize(x4)

            def cf_x1_forknonlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = u2
                x3_cf = 4 / (1 + np.exp(- x1_cf - x2_cf)) - x2_cf ** 2 + 0.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = 20 / (1 + np.exp(0.5 * x3_cf ** 2 - x3_cf)) + u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            def cf_x2_forknonlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 4 / (1 + np.exp(- x1_cf - x2_cf)) - x2_cf ** 2 + 0.5 * u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = 20 / (1 + np.exp(0.5 * x3_cf ** 2 - x3_cf)) + u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            var_names = ['x1', 'x2', 'x3', 'x4']
            graph = nx.DiGraph([('x1', 'x3'), ('x2', 'x3'), ('x3', 'x4')])
            data = pd.DataFrame(np.vstack([x1, x2, x3, x4]).T, columns=var_names)
            cfs = [cf_x1_forknonlinear(-1), cf_x1_forknonlinear(0), cf_x1_forknonlinear(1), cf_x2_forknonlinear(-1), cf_x2_forknonlinear(0), cf_x2_forknonlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x3': lambda x1, x2: (4 / (1 + np.exp(- x1 - x2)) - x2 ** 2 - mean_x3) / std_x3,
                       'x4': lambda x3: (20 / (1 + np.exp(0.5 * x3 ** 2 - x3)) - mean_x4) / std_x4,
                       'x3_str': f'(4 / (1 + exp(- x1 - x2)) - x2 ** 2 - {mean_x3}) / {std_x3}',
                       'x4_str': f'(20 / (1 + exp(0.5 * x3 ** 2 - x3)) - {mean_x4}) / {std_x4}'}

        elif self.name == 'simpson-non-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            u4 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = self.s(1 - x1) + np.sqrt(3 / 20) * u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = np.tanh(2 * x2) + 1.5 * x1 - 1 + np.tanh(u3)
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)
            x4 = (x3 - 4) / 5 + 3 + u4 / np.sqrt(10)
            x4, mean_x4, std_x4 = self.standardize(x4)

            def cf_x1_simpsonnonlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = self.s(1 - x1_cf) + np.sqrt(3 / 20) * u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = np.tanh(2 * x2_cf) + 1.5 * x1_cf - 1 + np.tanh(u3)
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = (x3_cf - 4) / 5 + 3 + u4 / np.sqrt(10)
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            def cf_x2_simpsonnonlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = np.tanh(2 * x2_cf) + 1.5 * x1_cf - 1 + np.tanh(u3)
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = (x3_cf - 4) / 5 + 3 + u4 / np.sqrt(10)
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            var_names = ['x1', 'x2', 'x3', 'x4']
            graph = nx.DiGraph([('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3'), ('x3', 'x4')])
            data = pd.DataFrame(np.vstack([x1, x2, x3, x4]).T, columns=var_names)
            cfs = [cf_x1_simpsonnonlinear(-1), cf_x1_simpsonnonlinear(0), cf_x1_simpsonnonlinear(1), cf_x2_simpsonnonlinear(-1), cf_x2_simpsonnonlinear(0), cf_x2_simpsonnonlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (self.s(1 - x1) - mean_x2) / std_x2,
                       'x3': lambda x1, x2: (np.tanh(2 * x2) + 1.5 * x1 - 1 - mean_x3) / std_x3,
                       'x4': lambda x3: ((x3 - 4) / 5 + 3 - mean_x4) / std_x4,
                       'x2_str': f'(softplus(1 - x1) - {mean_x2}) / {std_x2}',
                       'x3_str': f'(tanh(2 * x2) + 1.5 * x1 - 1 - {mean_x3}) / {std_x3}',
                       'x4_str': f'((x3 - 4) / 5 + 3 - {mean_x4}) / {std_x4}'}

        elif self.name == 'simpson-symprod':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            u4 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 2 * np.tanh(2 * x1) + u2 /np.sqrt(10)
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.5 * x1 * x2 + u3 / np.sqrt(2)
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)
            x4 = np.tanh(1.5 * x1) + np.sqrt(3/10) * u4
            x4, mean_x4, std_x4 = self.standardize(x4)

            def cf_x1_simpsonsymprod(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 2 * np.tanh(2 * x1_cf) + u2 /np.sqrt(10)
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.5 * x1_cf * x2_cf + u3 / np.sqrt(2)
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = np.tanh(1.5 * x1_cf) + np.sqrt(3/10) * u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T

            def cf_x2_simpsonsymprod(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 0.5 * x1_cf * x2_cf + u3 / np.sqrt(2)
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                x4_cf = np.tanh(1.5 * x1_cf) + np.sqrt(3/10) * u4
                x4_cf = (x4_cf - mean_x4) / std_x4
                return np.vstack([x1_cf, x2_cf, x3_cf, x4_cf]).T


            var_names = ['x1', 'x2', 'x3', 'x4']
            graph = nx.DiGraph([('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3, x4]).T, columns=var_names)
            cfs = [cf_x1_simpsonsymprod(-1), cf_x1_simpsonsymprod(0), cf_x1_simpsonsymprod(1), cf_x2_simpsonsymprod(-1), cf_x2_simpsonsymprod(0), cf_x2_simpsonsymprod(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (2 * np.tanh(2 * x1) - mean_x2) / std_x2,
                       'x3': lambda x1, x2: (0.5 * x1 * x2 - mean_x3) / std_x3,
                       'x4': lambda x1: (np.tanh(1.5 * x1) - mean_x4) / std_x4,
                       'x2_str': f'(2 * tanh(2 * x1) - {mean_x2}) / {std_x2}',
                       'x3_str': f'(0.5 * x1 * x2 - {mean_x3}) / {std_x3}',
                       'x4_str': f'(tanh(1.5 * x1) - {mean_x4}) / {std_x4}'}

        elif self.name == 'triangle-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 10 * x1 - u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.5 * x2 + x1 + u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)

            def cf_x1_trianglelinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 10 * x1_cf - u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.5 * x2_cf + x1_cf + u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            def cf_x2_trianglelinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 0.5 * x2_cf + x1_cf + u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            var_names = ['x1', 'x2', 'x3']
            graph = nx.DiGraph([('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3]).T, columns=var_names)
            cfs = [cf_x1_trianglelinear(-1), cf_x1_trianglelinear(0), cf_x1_trianglelinear(1), cf_x2_trianglelinear(-1), cf_x2_trianglelinear(0), cf_x2_trianglelinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (10 * x1 - mean_x2) / std_x2, 'x3': lambda x1, x2: (0.5 * x2 + x1 - mean_x3) / std_x3,
                       'x2_str': f'(10 * x1 - {mean_x2}) / {std_x2}', 'x3_str': f'(0.5 * x2 + x1 - {mean_x3}) / {std_x3}'}

        elif self.name == 'triangle-non-linear':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            x1 = u1  # IMPORTANT: In the original code, here we had u1+1, we set it to u1 to be consistent with other graphs
            x2 = 2 * x1 ** 2 + u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 20 / (1 + np.exp(- x2 ** 2 + x1)) + u3
            x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)

            def cf_x1_trianglenonlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 2 * x1_cf ** 2 + u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 20 / (1 + np.exp(- x2_cf ** 2 + x1_cf)) + u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            def cf_x2_trianglenonlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 20 / (1 + np.exp(- x2_cf ** 2 + x1_cf)) + u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            var_names = ['x1', 'x2', 'x3']
            graph = nx.DiGraph([('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3]).T, columns=var_names)
            cfs = [cf_x1_trianglenonlinear(-1), cf_x1_trianglenonlinear(0), cf_x1_trianglenonlinear(1), cf_x2_trianglenonlinear(-1), cf_x2_trianglenonlinear(0), cf_x2_trianglenonlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (2 * x1 ** 2 - mean_x2) / std_x2, 'x3': lambda x1, x2: (20 / (1 + np.exp(- x2 ** 2 + x1)) - mean_x3) / std_x3,
                       'x2_str': f'(2 * x1 ** 2 - {mean_x2}) / {std_x2}', 'x3_str': f'(20 / (1 + exp(- x2 ** 2 + x1)) - {mean_x3}) / {std_x3}'}

        elif self.name == 'triangle-non-linear-2':
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = sigmoid(x1)**0.5 - x1 - u2
            # x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.4*x1 + 0.1*x1**2 + 0.7*x2 - 0.5*x2**2 +  1.0* u3
            # x3, mean_x3, std_x3 = self.standardize(x3)
            if discrete:
                x3 = self.discretize(x3)

            def cf_x1_trianglenonlinear2(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf =  sigmoid(x1_cf)**0.5 - x1 - u2
                # x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.4*x1_cf + 0.1*x1_cf**2 + 0.7*x2_cf - 0.5*x2_cf**2  +  1.0* u3
                # x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            def cf_x2_trianglenonlinear2(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf = 0.4*x1_cf + 0.1*x1_cf**2 + 0.7*x2_cf - 0.5*x2_cf**2 +  1.0* u3
                # x3_cf = (x3_cf - mean_x3) / std_x3
                if discrete:
                    x3_cf = self.discretize(x3_cf)
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            var_names = ['x1', 'x2', 'x3']
            graph = nx.DiGraph([('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3]).T, columns=var_names)
            cfs = [cf_x1_trianglenonlinear2(-1), cf_x1_trianglenonlinear2(0), cf_x1_trianglenonlinear2(1), cf_x2_trianglenonlinear2(-1), cf_x2_trianglenonlinear2(0), cf_x2_trianglenonlinear2(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: sigmoid(x1)**0.5 - x1, 'x3': lambda x1, x2: 0.4*x1 + 0.1*x1**2 + 0.7*x2 - 0.5*x2**2,
                       'x2_str': f'sigmoid(x1)**0.5 - x1', 'x3_str': f'0.4*x1 + 0.1*x1**2 + 0.7*x2 - 0.5*x2**2'}



        elif self.name == 'triangle-sensitivity-nonlinear':
            #test that alpha is not none
            u1 = np.random.normal(0, 1, num_samples)
            u2 = np.random.normal(0, 1, num_samples)
            u3 = np.random.normal(0, 1, num_samples)
            x1 = u1
            x2 = 1/2 * x1 + 0.1*x1**2 + 0.6*u2
            x2, mean_x2, std_x2 = self.standardize(x2)
            x3 = 0.4*x1 + 0.1*x1**2 + 0.7*x2 - 0.5*x2**2 + 0.125*x2**3 - 0.5*(alpha*x2)*np.tanh(u3+1) +  0.5*u3
            x3, mean_x3, std_x3 = self.standardize(x3)

            def cf_x1_trianglesensitivitynonlinear(x1_val):
                x1_cf = x1_val * np.ones_like(x1)
                x2_cf = 1/2 * x1_cf + 0.1*x1_cf**2 + 0.6*u2
                x2_cf = (x2_cf - mean_x2) / std_x2
                x3_cf = 0.4*x1_cf + 0.1*x1_cf**2 + 0.7*x2_cf - 0.5*x2_cf**2 + 0.125*x2_cf**3 - 0.5*(alpha*x2_cf)*np.tanh(u3+1) +  0.5*u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            def cf_x2_trianglesensitivitynonlinear(x2_val):
                x1_cf = u1
                x2_cf = x2_val * np.ones_like(x2)
                x3_cf  = 0.4*x1_cf + 0.1*x1_cf**2 + 0.7*x2_cf - 0.5*x2_cf**2 + 0.125*x2_cf**3 - 0.5*(alpha*x2_cf)*np.tanh(u3+1) +  0.5*u3
                x3_cf = (x3_cf - mean_x3) / std_x3
                return np.vstack([x1_cf, x2_cf, x3_cf]).T

            var_names = ['x1', 'x2', 'x3']
            graph = nx.DiGraph([('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')])
            data = pd.DataFrame(np.vstack([x1, x2, x3]).T, columns=var_names)
            # only intervene in values 1 and 0
            cfs = [cf_x1_trianglesensitivitynonlinear(-1), cf_x1_trianglesensitivitynonlinear(0), cf_x1_trianglesensitivitynonlinear(1), cf_x2_trianglesensitivitynonlinear(-1), cf_x2_trianglesensitivitynonlinear(0), cf_x2_trianglesensitivitynonlinear(1)]
            data_cf = [pd.DataFrame(c, columns=var_names) for c in cfs]
            formula = {'x2': lambda x1: (1/2 * x1 + 0.1*x1**2 - mean_x2) / std_x2, 'x3': lambda x1, x2: (0.4*x1 + 0.1*np.sqrt(abs(x1)) + np.sin(x2 + alpha*u3) - mean_x3) / std_x3,
                       'x2_str': f'(1/2 * x1 + 0.1*x1**2 - {mean_x2}) / {std_x2}', 'x3_str': f'(0.4*x1 + 0.1*sqrt(abs(x1)) + sin(x2 + alpha*u3) - {mean_x3}) / {std_x3}'}
            u_df = pd.DataFrame(np.vstack([u1, u2, u3]).T, columns=['u1', 'u2', 'u3'])

        else:
            raise ValueError(f"Unknown sem_name: {self.name}")

        if return_u:
            return data, data_cf, graph, formula, u_df
        return data, data_cf, graph, formula  # Note: we do not return the noiseless data for now, data_noiseless, it may be not needed after all

    def s(self, x): # softplus function
        return np.log(1 + np.exp(x))

    def discretize(self, x):  # Generate a discrete variable
        # Discretize in 3 values
        probs = np.zeros((x.shape[0], 3))
        for i in range(x.shape[0]):
            if x[i] < -1:
                probs[i, 0] = 0.8
                probs[i, 1] = 0.1
                probs[i, 2] = 0.1
            elif x[i] > 1:
                probs[i, 0] = 0.1
                probs[i, 1] = 0.1
                probs[i, 2] = 0.8
            else:
                probs[i, 0] = 0.1
                probs[i, 1] = 0.8
                probs[i, 2] = 0.1
        x_discretized = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_discretized[i] = np.random.choice(3, p=probs[i, :])
        return x_discretized

