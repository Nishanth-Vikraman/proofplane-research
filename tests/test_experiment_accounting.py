from proofplane_v0.experiments.runner import success_at_k, unique_generated_path


def test_generated_paths_are_unique_even_when_model_repeats_filename():
    assert unique_generated_path("test_same.py", 0) == "proofplane_generated/test_0_same.py"
    assert unique_generated_path("test_same.py", 1) == "proofplane_generated/test_1_same.py"


def test_success_at_k_uses_candidate_rank_and_available_budget():
    assert success_at_k([False, True, False], [1, 2, 3, 5]) == {
        "1": False,
        "2": True,
        "3": True,
        "5": False,
    }
