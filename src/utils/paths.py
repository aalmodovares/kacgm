from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ASSETS_ROOT = REPO_ROOT / "assets"
CONFIGS_ROOT = REPO_ROOT / "configs"
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"
RUNNABLES_ROOT = REPO_ROOT / "runnables"
OUTPUTS_ROOT = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class ExperimentPaths:
    name: str
    root: Path
    data: Path
    samples: Path
    images: Path
    figures: Path
    checkpoints: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def make_run_id(prefix: str | None = None) -> str:
    suffix = uuid.uuid4().hex[:10]
    if prefix:
        return f"{slugify(prefix)}_{suffix}"
    return suffix


def get_experiment_paths(name: str, output_dir: str | Path | None = None) -> ExperimentPaths:
    root = resolve_path(output_dir) if output_dir is not None else OUTPUTS_ROOT / name
    root = ensure_dir(root)
    return ExperimentPaths(
        name=name,
        root=root,
        data=ensure_dir(root / "data"),
        samples=ensure_dir(root / "samples"),
        images=ensure_dir(root / "images"),
        figures=ensure_dir(root / "figures"),
        checkpoints=ensure_dir(root / "checkpoints"),
    )


def get_global_checkpoint_root() -> Path:
    return ensure_dir(OUTPUTS_ROOT / "checkpoints" / "kan")

