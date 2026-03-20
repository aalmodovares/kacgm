# Sachs equations and noises, based on https://github.com/patrickrchao/DiffusionBasedCausalModels/blob/main/experiments/structural_equations.py
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm
from dowhy import gcm
from dowhy.gcm import EmpiricalDistribution, ScipyDistribution, StructuralCausalModel
from dowhy.gcm.causal_mechanisms import FunctionalCausalModel, StochasticModel
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import is_root_node, get_ordered_predecessors
from dowhy.gcm.whatif import _parent_samples_of, _evaluate_intervention

graph_sachs = nx.DiGraph([ ('plcg', 'PIP3'), ('plcg', 'PIP2'), ('PIP3', 'PIP2'),
                           ('PKC', 'PKA'), ('PKC', 'pjnk'), ('PKC', 'P38'),
                           ('PKC', 'praf'), ('PKC', 'pmek'),
                           ('PKA', 'p44/42'), ('PKA', 'pjnk'), ('PKA', 'praf'),
                           ('PKA', 'pmek'), ('PKA', 'p44/42'), ('PKA', 'pakts473'),
                           ('PKA', 'P38'), ('praf', 'pmek'), ('pmek', 'p44/42'), ('p44/42', 'pakts473')
                           ])  # Graph from the decaflow paper


# Set the seed for repeatability, always generate the same equations!
np.random.seed(42)

Normal = lambda mean, variance: ScipyDistribution(norm, loc=mean, scale = np.sqrt(variance))

def silu(x):
    sigmoid = np.where(x >= 0,  1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return np.multiply(x, sigmoid)

noises_distr = {}

for node in graph_sachs.nodes():
    noises_distr[node] = Normal(0, 1)

def get_structural_equations_sachs():
    weights_additive = {}

    for node in graph_sachs.nodes():
        in_deg = graph_sachs.in_degree(node)
        if in_deg > 0:
            if node+"_1" not in weights_additive:
                weights_additive[node+"_1"] = np.random.uniform(low=-1, high=1, size=(16 , in_deg))
                weights_additive[node+"_2"] = np.random.uniform(low=-1, high= 1, size=(1, 16))

    weights_nonadditive   = {}
    for node in graph_sachs.nodes():
        in_deg = graph_sachs.in_degree(node)
        if in_deg > 0:
            if node+"_1" not in weights_nonadditive:
                weights_nonadditive[node+"_1"] = np.random.uniform(low=-1, high=1, size=(16 , in_deg+1))
                weights_nonadditive[node+"_2"] = np.random.uniform(low=-1, high= 1, size=(1, 16))

    structural_eq = {'additive': {}, 'nonadditive': {}}
    for node in graph_sachs.nodes:
        if graph_sachs.in_degree(node) > 0:
            w1, w2 = weights_additive[node+"_1"], weights_additive[node+"_2"]
            def create_helper(mat1, mat2):
                def helper(u, *params):
                    full_params = np.concatenate(params, axis=1).T
                    return (mat2 @ silu(mat1 @ full_params) * 0.75)[0, :] + u.reshape(-1)
                return helper
            structural_eq['additive'][node] = create_helper(w1, w2)
        else:
            structural_eq['additive'][node] = lambda u: u


    for node in graph_sachs.nodes:
        if graph_sachs.in_degree(node) > 0:
            w1, w2 = weights_nonadditive[node+"_1"], weights_nonadditive[node+"_2"]
            def create_helper(mat1, mat2):
                def helper(*params):
                    full_params = np.concatenate(params, axis=1).T
                    return (mat2 @ silu(mat1 @ full_params) * 0.75)[0, :]

                return helper
            structural_eq['nonadditive'][node] = create_helper(w1, w2)
        else:
            structural_eq['nonadditive'][node] = lambda u: u

    return structural_eq




# Classes for the generator object
class GeneralNoiseModel(FunctionalCausalModel):
    def __init__(self, formula, noise_model):

        self._noise_model = noise_model
        self._formula = formula

    def evaluate(self, parent_samples, noise_samples):
        parent_samples, noise_samples = shape_into_2d(parent_samples, noise_samples)
        return self._formula(noise_samples,parent_samples)

    def clone(self):
        return GeneralNoiseModel(self._formula, noise_model=self._noise_model.clone())

    def fit(self, X, Y):
        pass

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        return self._noise_model.draw_samples(num_samples)

class RootGeneralNoiseModel(StochasticModel):
    def __init__(self, formula, noise_model):

        self._noise_model = noise_model
        self._formula = formula

    def evaluate(self, noise_samples: np.ndarray) -> np.ndarray:
        noise_samples = shape_into_2d(noise_samples)
        return self._formula(noise_samples)

    def clone(self):
        return RootGeneralNoiseModel(self._formula, noise_model=self._noise_model.clone())

    def fit(self, X):
        pass

    def draw_noise_samples(self, num_samples):
        return self._noise_model.draw_samples(num_samples)

    def draw_samples(self, num_samples):
        return self.evaluate(self.draw_noise_samples(num_samples))


class ExperimentationModel(StructuralCausalModel):
    """Wrapper for StructuralCausalModels.
        Accepts a graph, name, dictionary of structural equations, and dictionary of noise_distributions

        Allows for arbitrary sampling and counterfactual inference
    """
    def __init__(self, graph,  name,  structural_equations, noise_distributions):
        self.graph = graph

        self.structural_equations = structural_equations
        self.name = name
        self.noise_distributions = noise_distributions
        num_nodes = self.graph.number_of_nodes()
        if not (len(noise_distributions)  == len(structural_equations) == num_nodes):
            raise ValueError("Inconsistent number of nodes")

        self.model = StructuralCausalModel(graph)
        self._set_causal_mechanisms()
        temp = pd.DataFrame(np.zeros((1,num_nodes)))
        temp.columns = list(self.graph.nodes)
        gcm.fit(self.model, temp)

    def _set_causal_mechanisms(self):
        sorted_nodes = nx.topological_sort(self.graph)
        for node in sorted_nodes:
            if is_root_node(self.graph, node):
                self.model.set_causal_mechanism(node, RootGeneralNoiseModel(
                self.structural_equations[node],
                 self.noise_distributions[node]))
            else:
                spread = lambda func: (lambda noise, parents: func(noise,
                                    *np.hsplit(parents,list(range(1,parents.shape[1])))))
                spread_func = spread(self.structural_equations[node])
                self.model.set_causal_mechanism(node,
                            GeneralNoiseModel(spread_func,
                                              self.noise_distributions[node]))


    def sample(self, num_samples):
        data_samples, noise_samples = self._draw_data_and_noise_samples(num_samples)
        return data_samples, noise_samples

    def get_counterfactuals(self, interventions, noise_data):
        samples = counterfactual_samples(self.model, interventions=interventions, noise_data=noise_data)
        topologically_sorted_nodes = list(nx.topological_sort(self.graph))
        samples = pd.DataFrame(np.empty((noise_data.shape[0], len(topologically_sorted_nodes))), columns=topologically_sorted_nodes)
        for node in topologically_sorted_nodes:
            if is_root_node(self.graph, node):
                node_data = self.model.causal_mechanism(node).evaluate(noise_data[node].to_numpy())
            else:
                node_data = self.model.causal_mechanism(node).evaluate(
                    _parent_samples_of(node, self, samples), noise_data[node].to_numpy()
                )

            samples[node] = _evaluate_intervention(node, interventions, node_data.reshape(-1))

        # #Necessary since the default implementation assumes root nodes are equal to noise
        # for node in self.graph.nodes:
        #     if is_root_node(self.graph, node) and node not in interventions:
        #         samples[node] = self.model.causal_mechanism(node).evaluate(samples[node])
        return samples

    def _draw_data_and_noise_samples(self, num_samples):
        sorted_nodes = nx.topological_sort(self.graph)

        drawn_noise_samples = {}
        drawn_samples = {}

        for node in sorted_nodes:
            noise = self.model.causal_mechanism(node).draw_noise_samples(num_samples)
            drawn_noise_samples[node] = noise
            if is_root_node(self.graph, node):
                drawn_samples[node] = self.model.causal_mechanism(node).evaluate(noise)
            else:
                drawn_samples[node] = self.model.causal_mechanism(node).evaluate(
                    column_stack_selected_numpy_arrays(drawn_samples, get_ordered_predecessors(self.graph, node)),
                    noise)

        return convert_to_data_frame(drawn_samples), convert_to_data_frame(drawn_noise_samples)


    def data_from_noise(self, noise_data):
        """Necessary since the default implementation assumes root nodes are equal to noise
        modified from https://github.com/py-why/dowhy/blob/ead8d47102f0ac6db51d84432874c331fb84f3cb/dowhy/gcm/_noise.py
        """

        sorted_nodes = list(nx.topological_sort(self.graph))
        data = pd.DataFrame(np.empty((noise_data.shape[0], len(sorted_nodes))), columns=sorted_nodes)

        for node in sorted_nodes:
            if is_root_node(self.graph, node):
                data[node] = self.model.causal_mechanism(node).evaluate(noise_data[node].to_numpy())
            else:
                data[node] = self.model.causal_mechanism(node).evaluate(
                    data[get_ordered_predecessors(self.graph, node)].to_numpy(), noise_data[node].to_numpy()
                )
        return data



def column_stack_selected_numpy_arrays(dict_with_numpy_arrays, keys):
    return np.column_stack([dict_with_numpy_arrays[x] for x in keys])


def convert_to_data_frame(dict_with_numpy_arrays):
    return pd.DataFrame({k: convert_numpy_array_to_pandas_column(v) for (k, v) in dict_with_numpy_arrays.items()})


def convert_numpy_array_to_pandas_column(*args):
    """Prepares given numpy arrays to be used as column data in a pandas data frame. This means, for numpy arrays with
    one feature, a flatten version is returned for a better performance. For numpy arrays with multiple columns,
    the entries (row-wise) are returned in a list.
    Example:
       array([[1], [2]]) -> array([1, 2])
       array([[1, 2], [3, 4]]) -> list([[1, 2], [3, 4]])
       array([[1]]) -> array([1])
    """

    def shaping(X):
        X = X.squeeze()

        if X.ndim == 0:
            return np.array([X])

        if X.ndim > 1:
            return list(X)
        else:
            return X

    result = [shaping(x) for x in args]

    if len(result) == 1:
        return result[0]
    else:
        return result

