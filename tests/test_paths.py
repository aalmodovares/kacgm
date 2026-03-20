from utils.paths import OUTPUTS_ROOT, REPO_ROOT, get_experiment_paths


def test_get_experiment_paths_creates_expected_directories(tmp_path):
    paths = get_experiment_paths("unit_test", output_dir=tmp_path / "unit_test")
    assert paths.root.exists()
    assert paths.data.exists()
    assert paths.samples.exists()
    assert paths.images.exists()
    assert paths.figures.exists()
    assert paths.checkpoints.exists()
    assert REPO_ROOT.exists()
    assert OUTPUTS_ROOT.exists()
