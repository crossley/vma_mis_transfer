"""Validate statsmodels KF/MLE output-only vs additive-process models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.search import generate_hybrid_smooth_designs
from noise_model.simulate import simulate_subject
from noise_model.statsmodels_kf import fit_statsmodels_kf_binary
from noise_model.validation import variable_parameter_sampler


OUTPUTS = ROOT / "outputs"
DESIGN_NAME = "hybrid_304"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = next(d for d in generate_hybrid_smooth_designs(n_trials=240, seed=3500) if d.name == DESIGN_NAME)
    rows = []
    n_reps = 20
    for true_family in ["output_only", "bias_term"]:
        for rep in range(n_reps):
            seed = 11100 + rep + (10000 if true_family == "bias_term" else 0)
            params = variable_parameter_sampler(true_family, rep, seed)
            data = simulate_subject(design, true_family, params, seed=seed, subject=rep)
            output_fit, process_fit = fit_statsmodels_kf_binary(data, maxiter=120)
            delta = output_fit.nll - process_fit.nll
            rows.append(
                {
                    "true_model": true_family,
                    "rep": rep,
                    "output_nll": output_fit.nll,
                    "process_nll": process_fit.nll,
                    "delta_process": delta,
                    "best_aic": "bias_term" if process_fit.aic < output_fit.aic else "output_only",
                    "best_bic": "bias_term" if process_fit.bic < output_fit.bic else "output_only",
                    "output_success": output_fit.success,
                    "process_success": process_fit.success,
                    **{f"output_{k}": v for k, v in output_fit.params.items()},
                    **{f"process_{k}": v for k, v in process_fit.params.items()},
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUTS / "statsmodels_kf_binary_results.csv", index=False)

    for criterion in ["aic", "bic"]:
        col = f"best_{criterion}"
        matrix = pd.crosstab(results.true_model, results[col], normalize="index")
        matrix.to_csv(OUTPUTS / f"statsmodels_kf_binary_confusion_{criterion}.csv")
        print(f"\n{criterion} confusion")
        print(matrix.round(3).to_string())
        print("accuracy", round((results.true_model == results[col]).mean(), 3))

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
    thresholds.to_csv(OUTPUTS / "statsmodels_kf_binary_thresholds.csv", index=False)
    print("\nconservative likelihood thresholds")
    print(thresholds.round(3).to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
