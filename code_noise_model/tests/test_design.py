from noise_model.design import make_candidate_designs


def test_candidate_designs_are_200_trials_and_balanced():
    designs = make_candidate_designs(n_trials=200, seed=1)
    assert len(designs) == 5
    for design in designs:
        assert len(design.trials) == 200
        counts = design.trials["target_index"].value_counts()
        assert counts.min() == counts.max() == 40
