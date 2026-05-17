"""Validate two-stage KF output-vs-process workflow with estimated mean dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.fit import fit_kf_binary_two_stage
from noise_model.models import default_params
from noise_model.search import generate_hybrid_smooth_designs
from noise_model.simulate import simulate_subject
from noise_model.validation import variable_parameter_sampler


OUTPUTS = ROOT / "outputs"
DESIGN_NAME = "hybrid_304"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = next(d for d in generate_hybrid_smooth_designs(n_trials=240, seed=3500) if d.name == DESIGN_NAME)
    rows = []
    n_reps = 36
    start = {**default_params("bias_term"), "nC_sd": 1.0}
    for true_family in ["output_only", "bias_term"]:
        for rep in range(n_reps):
            seed = 10100 + rep + (10000 if true_family == "bias_term" else 0)
            params = variable_parameter_sampler(true_family, rep, seed)
            data = simulate_subject(design, true_family, params, seed=seed, subject=rep)
            output_fit, process_fit, mean_params = fit_kf_binary_two_stage(
                data,
                start,
                maxiter_mean=100,
                maxiter_noise=80,
            )
            delta = output_fit.heldout_nll - process_fit.heldout_nll
            rows.append(
                {
                    "true_model": true_family,
                    "rep": rep,
                    "delta_process": delta,
                    "best_heldout": "bias_term" if delta > 0 else "output_only",
                    "output_heldout_nll": output_fit.heldout_nll,
                    "process_heldout_nll": process_fit.heldout_nll,
                    "fit_A": mean_params["A"],
                    "fit_B": mean_params["B"],
                    "fit_C": mean_params["C"],
                    "fit_output_sd": mean_params["output_sd"],
                    "fit_width": mean_params["width"],
                    "fit_process_sd": process_fit.params["nC_sd"],
                    "true_A": params["A"],
                    "true_B": params["B"],
                    "true_C": params["C"],
                    "true_output_sd": params["output_sd"],
                    "true_width": params["width"],
                    "true_process_sd": params.get("nC_sd", 0.0),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUTS / "kf_binary_two_stage_results.csv", index=False)

    matrix = pd.crosstab(results.true_model, results.best_heldout, normalize="index")
    matrix.to_csv(OUTPUTS / "kf_binary_two_stage_confusion_raw.csv")
    print("Raw heldout confusion")
    print(matrix.round(3).to_string())
    print("accuracy", round((results.true_model == results.best_heldout).mean(), 3))

    output = results[results.true_model == "output_only"]
    process = results[results.true_model == "bias_term"]
    threshold_rows = []
    for target in [0.05, 0.10, 0.20]:
        best = None
        for threshold in sorted(set(results.delta_process.round(9))):
            false_process = float((output.delta_process >= threshold).mean())
            if false_process <= target:
                process_recall = float((process.delta_process >= threshold).mean())
                row = {
                    "false_process_target": target,
                    "threshold": threshold,
                    "output_recall": 1.0 - false_process,
                    "process_recall": process_recall,
                    "false_process_rate": false_process,
                    "false_output_rate": 1.0 - process_recall,
                }
                if best is None or row["process_recall"] > best["process_recall"]:
                    best = row
        threshold_rows.append(best)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUTPUTS / "kf_binary_two_stage_thresholds.csv", index=False)
    print("\nConservative thresholds")
    print(thresholds.round(3).to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
