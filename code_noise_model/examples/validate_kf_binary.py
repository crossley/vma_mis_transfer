"""Validate simplified KF-style output-noise vs additive-process comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.design import TrialDesign
from noise_model.fit import fit_model
from noise_model.search import generate_hybrid_smooth_designs
from noise_model.simulate import simulate_subject
from noise_model.validation import variable_parameter_sampler


OUTPUTS = ROOT / "outputs"
DESIGN_NAME = "hybrid_304"
TRUE_FAMILIES = ("output_only", "bias_term")
FIT_FAMILIES = ("output_only", "bias_term")


def _load_design() -> TrialDesign:
    designs = generate_hybrid_smooth_designs(n_trials=240, seed=3500)
    return next(design for design in designs if design.name == DESIGN_NAME)


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = _load_design()
    rows = []
    param_rows = []
    n_reps = 24
    for true_family in TRUE_FAMILIES:
        for rep in range(n_reps):
            seed = 8100 + rep + (1000 if true_family == "bias_term" else 0)
            params = variable_parameter_sampler(true_family, rep, seed)
            data = simulate_subject(design, true_family, params, seed=seed, subject=rep)
            fits = [fit_model(data, family, maxiter=40, fast=True) for family in FIT_FAMILIES]
            best_heldout = min(fits, key=lambda fit: fit.heldout_nll).family
            best_bic = min(fits, key=lambda fit: fit.bic).family
            best_aic = min(fits, key=lambda fit: fit.aic).family
            for fit in fits:
                rows.append(
                    {
                        "design": design.name,
                        "true_model": true_family,
                        "rep": rep,
                        "fit_model": fit.family,
                        "heldout_nll": fit.heldout_nll,
                        "aic": fit.aic,
                        "bic": fit.bic,
                        "best_heldout": best_heldout,
                        "best_aic": best_aic,
                        "best_bic": best_bic,
                    }
                )
                if fit.family == true_family:
                    for key, value in fit.params.items():
                        param_rows.append(
                            {
                                "true_model": true_family,
                                "rep": rep,
                                "parameter": key,
                                "recovered": value,
                                "true": params.get(key),
                            }
                        )

    results = pd.DataFrame(rows)
    params = pd.DataFrame(param_rows)
    results.to_csv(OUTPUTS / "kf_binary_results.csv", index=False)
    params.to_csv(OUTPUTS / "kf_binary_parameter_recovery.csv", index=False)

    one = results[["true_model", "rep", "best_heldout", "best_aic", "best_bic"]].drop_duplicates()
    for criterion in ["heldout", "aic", "bic"]:
        col = f"best_{criterion}"
        matrix = pd.crosstab(one["true_model"], one[col], normalize="index")
        matrix.to_csv(OUTPUTS / f"kf_binary_confusion_{criterion}.csv")
        print(f"\n{criterion} confusion")
        print(matrix.round(3).to_string())
        print("accuracy", round((one["true_model"] == one[col]).mean(), 3))

    pivot = results.pivot_table(
        index=["true_model", "rep"],
        columns="fit_model",
        values="heldout_nll",
    ).reset_index()
    pivot["delta_process"] = pivot["output_only"] - pivot["bias_term"]
    threshold_rows = []
    output = pivot[pivot.true_model == "output_only"]
    process = pivot[pivot.true_model == "bias_term"]
    for target in [0.05, 0.10, 0.20]:
        best = None
        for threshold in sorted(set(pivot.delta_process.round(9))):
            false_process = float((output.delta_process >= threshold).mean())
            if false_process <= target:
                process_recall = float((process.delta_process >= threshold).mean())
                output_recall = 1.0 - false_process
                accuracy = float(
                    ((output.delta_process < threshold).sum() + (process.delta_process >= threshold).sum())
                    / len(pivot)
                )
                candidate = {
                    "false_process_target": target,
                    "threshold": threshold,
                    "accuracy": accuracy,
                    "output_recall": output_recall,
                    "process_recall": process_recall,
                    "false_process_rate": false_process,
                    "false_output_rate": 1.0 - process_recall,
                }
                if best is None or candidate["process_recall"] > best["process_recall"]:
                    best = candidate
        if best is not None:
            threshold_rows.append(best)
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(OUTPUTS / "kf_binary_conservative_thresholds.csv", index=False)
    print("\nconservative heldout likelihood thresholds")
    print(threshold_df.round(3).to_string(index=False))

    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
