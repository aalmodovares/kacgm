![KaCGM](assets/KaCGM.png)

# KaCGM: Kolmogorov-Arnold Causal Generative Models

Codebase for the experiments, case studies, and analysis notebooks associated with the KaCGM paper.

> [!IMPORTANT]
> Please, note that this is a refactored version of the first experimental code, so it can contain minimal errors that we haven´t found. If
> you find any issue, please, report it to us by opening an issue in the repository or contact me to
> <alejandro.almodovar@upm.es>
>  We will be happy to fix it as soon as possible.

## Repository Layout

```text
.
|-- assets/                  # Static figures used in the repository README and paper support material
|-- configs/                 # Default experiment configurations
|-- data/raw/                # Raw data that remains versioned
|-- notebooks/               # Analysis notebooks and paper plots
|-- outputs/                 # Generated samples, metrics, figures, checkpoints (git-ignored)
|-- runnables/               # Executable experiment entry points
|-- src/
|   |-- datasets/            # Dataset generators and case-study loaders
|   |-- models/              # KAN, flow, and diffusion-based causal models
|   |-- plotting/            # Notebook/result loading helpers
|   `-- utils/               # Paths, metrics, evaluation, hyperparameter search, stats
`-- tests/                   # Lightweight pytest coverage for the refactor
```

## Installation

Create a clean environment and install the project in editable mode:

```bash
conda create -n kacgm python==3.11
conda activate kacgm
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The editable install is required so the `datasets`, `models`, `plotting`, and `utils` packages resolve correctly in both the runnables and the notebooks.

## Running Experiments

Synthetic continuous benchmark:

```bash
python -m runnables.run_continuous_benchmark --n-jobs 8
```

Synthetic mixed/discrete benchmark:

```bash
python -m runnables.run_discrete_benchmark --n-jobs 8
```

Sachs semi-synthetic benchmark:

```bash
python -m runnables.run_sachs_benchmark --hp-jobs 8 --run-jobs 4
```

Cardio case study:

```bash
python -m runnables.run_cardio_case_study --n-jobs 8
```

Cardio case study including cached interventional samples for the notebook:

```bash
python -m runnables.run_cardio_case_study --n-jobs 8 --include-interventions
```

Repeated cardio bootstrap evaluation with 10 seeds and 10 bootstrap test resamples per seed:

```bash
python -m runnables.run_cardio_bootstrap_evaluation --n-seeds 10 --n-bootstraps 10 --n-jobs 8
```

Repeated cardio bootstrap evaluation while also storing observational and interventional sample bundles for every model, seed, and bootstrap replicate:

```bash
python -m runnables.run_cardio_bootstrap_evaluation --n-seeds 10 --n-bootstraps 10 --n-jobs 8 --store-samples
```

Sensitivity analysis:

```bash
python -m runnables.run_sensitivity --jobs 8
```

Pruning and symbolic regression study:

```bash
python -m runnables.run_symbolic_pruning_study
```

All experiment outputs are written under `outputs/<experiment>/` with separate `data/`, `samples/`, `images/`, `figures/`, and `checkpoints/` subfolders where applicable. These generated artifacts are intentionally ignored by git.

To run the repeated cardio analysis end to end:

```bash
python -m runnables.run_cardio_bootstrap_evaluation --n-seeds 10 --n-bootstraps 10 --n-jobs 8 --store-samples
jupyter lab notebooks/cardio_bootstrap_results.ipynb
```

The bootstrap cardio runnable writes:

- `outputs/cardio_bootstrap/data/cardio_bootstrap_metrics_long.csv` with one row per model, seed, bootstrap, and metric.
- `outputs/cardio_bootstrap/data/cardio_bootstrap_metrics_summary.csv` with the aggregated bootstrap summaries and empirical 2.5% and 97.5% quantiles used by the notebook confidence intervals.
- `outputs/cardio_bootstrap/data/cardio_bootstrap_symbolic_formulas_metadata.csv` with the KAAM pruned/symbolic formula extraction status by seed.
- `outputs/cardio_bootstrap/data/cardio_bootstrap_symbolic_formulas.csv` with the symbolic formulas extracted from `kaam_mixed_symbolic`.
- `outputs/cardio_bootstrap/samples/by_seed/<model>/seed_<seed>/` when `--store-samples` is enabled, including `observational.csv`, `observational.pkl`, `interventional.pkl`, `interventional_manifest.csv`, and the per-intervention CSV tables used for notebook inspection.
- `outputs/cardio_bootstrap/samples/by_bootstrap/<model>/seed_<seed>/bootstrap_<bootstrap>/` when `--store-samples` is enabled, including the bootstrap-specific observational/interventional sample bundle.

## Notebooks

The notebooks in `notebooks/` are intended for loading stored results and generating paper figures. Run the corresponding experiment first, then open:

- `notebooks/paper_plots.ipynb`
- `notebooks/sachs_analysis.ipynb`
- `notebooks/sachs_results_analysis.ipynb`
- `notebooks/cardio.ipynb`
- `notebooks/cardio_bootstrap_results.ipynb`
- `notebooks/sensitivity_plot.ipynb`

## Tests

Run the lightweight structural checks with:

```bash
pytest
```

## Notes

- The cardio dataset is not included in the repository, but you can find it in the original paper by
[Kyriaocu et al.](https://www.sciencedirect.com/science/article/pii/S0196064422005789). The dataset should be stored as
`data/raw/cardio.csv` for the runnables to work.
- `outputs/` is kept as a skeleton in the repository so all runnables share a predictable output layout.

## Interface
There is an online interface that you can check out to observe the causal effect of some interventions in
the semi-synthetic Sachs dataset and in the cardio case study. You can find it at [this link](https://huggingface.co/spaces/marElizo/kan4scminterface).

## Citation

If you find this code useful for your research, please consider citing the paper:

```bibtex
@misc{almodovar2026kolmogorovarnoldcausalgenerativemodels,
      title={Kolmogorov-Arnold causal generative models}, 
      author={Alejandro Almodóvar and Mar Elizo and Patricia A. Apellániz and Santiago Zazo and Juan Parras},
      year={2026},
      eprint={2603.20184},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.20184}, 
}
```

