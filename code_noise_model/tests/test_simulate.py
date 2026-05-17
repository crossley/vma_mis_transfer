from noise_model.design import make_candidate_designs
from noise_model.models import MODEL_FAMILIES, default_params
from noise_model.simulate import simulate_subject


def test_simulate_all_families():
    design = make_candidate_designs(n_trials=25, seed=2)[0]
    for family in MODEL_FAMILIES:
        data = simulate_subject(design, family, default_params(family), seed=3)
        assert len(data) == 25
        assert data["y"].notna().all()
