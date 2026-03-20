from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from utils.paths import get_experiment_paths


def load_csv(experiment_name, relative_name, output_dir=None):
    paths = get_experiment_paths(experiment_name, output_dir=output_dir)
    return pd.read_csv(paths.data / relative_name)


def load_pickle(experiment_name, relative_name, output_dir=None):
    paths = get_experiment_paths(experiment_name, output_dir=output_dir)
    with open(paths.data / relative_name, "rb") as handle:
        return pickle.load(handle)


def load_sample_csv(experiment_name, relative_name, output_dir=None):
    paths = get_experiment_paths(experiment_name, output_dir=output_dir)
    return pd.read_csv(paths.samples / relative_name)


def resolve_experiment_dir(experiment_name, output_dir=None) -> Path:
    return get_experiment_paths(experiment_name, output_dir=output_dir).root

