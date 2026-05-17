from noise_model.design import make_candidate_designs
from noise_model.fit import fit_model
from noise_model.models import default_params
from noise_model.simulate import simulate_subject


def test_fit_smoke_output_only():
    design = make_candidate_designs(n_trials=30, seed=3)[0]
    data = simulate_subject(design, "output_only", default_params("output_only"), seed=4)
    fit = fit_model(data, "output_only", maxiter=10)
    assert fit.nll > 0
    assert "A" in fit.params
