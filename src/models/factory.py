from __future__ import annotations

from copy import deepcopy

from dowhy import gcm
from scipy.stats import norm

from models.dbcm import create_model_from_graph as create_model_from_graph_dbcm
from models.kan import kan_predictor


def create_model_from_graph(g, model="kan", params=None, noise=None):
    params = {} if params is None else params
    if model in {"kan", "kaam"}:
        g = gcm.InvertibleStructuralCausalModel(g)
        for node in g.graph.nodes:
            if g.graph.in_degree[node] == 0:
                g.set_causal_mechanism(node, gcm.ScipyDistribution(norm))
            else:
                predictor = kan_predictor(**deepcopy(params[node]))
                noise_model = gcm.ScipyDistribution(norm) if noise is None else gcm.ScipyDistribution(noise)
                g.set_causal_mechanism(
                    node,
                    gcm.AdditiveNoiseModel(
                        prediction_model=predictor,
                        noise_model=noise_model,
                    ),
                )
    elif model == "dbcm":
        g = create_model_from_graph_dbcm(g, deepcopy(params))
    elif model == "anm":
        g = gcm.InvertibleStructuralCausalModel(g)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return g

