from proofplane_v0.config import Settings
from proofplane_v0.experiments.runner import smoke


def test_synthetic_end_to_end(tmp_path):
    report = smoke(Settings(database=tmp_path / "db.sqlite", artifacts_dir=tmp_path / "artifacts"))
    assert report.exists()
    assert (report.parent / "results.jsonl").read_text().count("\n") == 3
    assert "Primary comparison" in report.read_text()
    assert "Datasets: synthetic" in report.read_text()
