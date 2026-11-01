from proofplane_v0.corpus.enrich import deterministic_hidden
from proofplane_v0.datasets.synthetic import SyntheticAdapter
from proofplane_v0.experiments.buckets import evaluation_bucket
from proofplane_v0.experiments.splits import assert_isolated, project_split


def test_project_split_isolation():
    cases = SyntheticAdapter().list_cases()
    splits = project_split(cases)
    assert_isolated(cases, splits)


def test_bucket_assignment():
    payloads = [deterministic_hidden(x) for x in SyntheticAdapter().list_cases()]
    assert evaluation_bucket(payloads[0], [payloads[0]]) == "seen_family"
    assert evaluation_bucket(payloads[1], [payloads[0]]) == "novel_ood"
