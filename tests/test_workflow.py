from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.datasets.synthetic import SyntheticAdapter
from proofplane_v0.workflow import build_index, create_splits, enrich_split, prepare_cases


def test_persisted_synthetic_workflow(tmp_path):
    configured = Settings(
        data_dir=tmp_path / "data",
        database=tmp_path / "db.sqlite",
        artifacts_dir=tmp_path / "artifacts",
    )
    adapter = SyntheticAdapter()
    store = CorpusStore(configured.database)
    prepared = prepare_cases(adapter, store, configured, [], 3)
    assert len(prepared) == 3
    splits = create_splits(adapter, store, configured.split_seed)
    assert set(splits.values()) == {"train", "dev", "eval"}
    for split in ("train", "dev", "eval"):
        assert enrich_split(adapter, store, configured, split, 1, mock=True)
    assert build_index(store, configured.artifacts_dir / "primary_index.json") == 1
    assert "hidden_payload" not in (configured.artifacts_dir / "primary_index.json").read_text()
