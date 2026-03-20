from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.paths import RAW_DATA_ROOT, resolve_path


CARDIO_FILENAME = "cardio.csv"
CARDIO_DROP_COLUMNS = ["randid", "diastolic pressure"]
CARDIO_RENAME_MAP = {
    "diabetes mellitus": "diabetes",
    "systolic pressure": "systolic",
    "cardiac ischemia": "ischemia",
    "major acute cardiovascular event": "macv",
}


@dataclass
class CardioPreparedData:
    factual_train_d: pd.DataFrame
    factual_eval_d: pd.DataFrame
    factual_train_dn: pd.DataFrame
    node_info: dict


def load_cardio_dataframe(path=None):
    csv_path = resolve_path(path) if path is not None else RAW_DATA_ROOT / CARDIO_FILENAME
    data_cardio = pd.read_csv(csv_path)
    data_cardio = data_cardio.drop(columns=CARDIO_DROP_COLUMNS)
    data_cardio = data_cardio.rename(columns=CARDIO_RENAME_MAP)
    data_cardio = data_cardio.dropna()
    return data_cardio


def get_cardio_graph():
    return nx.DiGraph(
        [
            ("age", "diabetes"),
            ("age", "systolic"),
            ("age", "ischemia"),
            ("age", "bmi"),
            ("bmi", "systolic"),
            ("bmi", "ischemia"),
            ("bmi", "diabetes"),
            ("systolic", "ischemia"),
            ("diabetes", "macv"),
            ("diabetes", "ischemia"),
            ("ischemia", "macv"),
        ]
    )


def prepare_cardio_data(data=None, split_eval=0.5, seed=42, discrete_threshold=5, jitter_std=0.01):
    if data is None:
        data = load_cardio_dataframe()
    else:
        data = data.copy(deep=True)
    np.random.seed(seed)
    factual_train_d, factual_eval_d = train_test_split(data, test_size=split_eval, random_state=seed)
    factual_train_d = factual_train_d.copy(deep=True)
    factual_eval_d = factual_eval_d.copy(deep=True)

    int_min_val = -3
    int_max_val = 3
    int_step = 0.1
    int_vector_values = np.arange(int_min_val, int_max_val, int_step)

    node_info = {}
    for node in factual_train_d.columns:
        node_info[node] = {}
        if len(factual_train_d[node].unique()) > discrete_threshold:
            node_info[node]["type"] = "continuous"
            factual_train_d[node] = factual_train_d[node].astype(np.float64)
            factual_eval_d[node] = factual_eval_d[node].astype(np.float64)
            node_info[node]["mean"] = factual_train_d[node].mean()
            node_info[node]["std"] = factual_train_d[node].std()
            factual_train_d.loc[:, node] = (factual_train_d[node] - node_info[node]["mean"]) / node_info[node]["std"]
            factual_eval_d.loc[:, node] = (factual_eval_d[node] - node_info[node]["mean"]) / node_info[node]["std"]
            node_info[node]["int_values"] = tuple(int_vector_values)
            node_info[node]["percentile_25"] = np.percentile(factual_eval_d[node], 25)
            node_info[node]["percentile_50"] = np.median(factual_eval_d[node])
            node_info[node]["percentile_75"] = np.percentile(factual_eval_d[node], 75)
        else:
            node_info[node]["type"] = "discrete"
            node_info[node]["int_values"] = (0, 1)

    factual_train_dn = factual_train_d + np.random.normal(0, jitter_std, factual_train_d.shape)
    return CardioPreparedData(
        factual_train_d=factual_train_d,
        factual_eval_d=factual_eval_d,
        factual_train_dn=factual_train_dn,
        node_info=node_info,
    )


def post_process_samples(df, node_info, denorm=False):
    df = df.copy()
    for node in df.columns:
        if node_info[node]["type"] == "discrete":
            df[node] = df[node].astype(int)
            df[node] = np.clip(df[node], min(node_info[node]["int_values"]), max(node_info[node]["int_values"]))
    if denorm:
        for node in df.columns:
            if node_info[node]["type"] == "continuous":
                df[node] = df[node] * node_info[node]["std"] + node_info[node]["mean"]
    return df


def infer_node_types(dataframe, discrete_threshold=5, default_num_classes=2):
    node_types = {}
    num_classes = {}
    for node in dataframe.columns:
        if len(dataframe[node].unique()) <= discrete_threshold:
            node_types[node] = "discrete"
            num_classes[node] = default_num_classes
        else:
            node_types[node] = "continuous"
    return node_types, num_classes
