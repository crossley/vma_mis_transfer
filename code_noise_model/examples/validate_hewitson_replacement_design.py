"""Validate the Hewitson replacement diagnostic block."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.design import make_hewitson_replacement_diagnostic_design
from noise_model.scalar_kf import fit_scalar_kf, simulate_scalar
from noise_model.validation import validate_design_with_parameter_variability


OUTPUTS = ROOT / "outputs"


def sample_scalar_params(family: str, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    params = {
        "A": float(np.clip(rng.normal(0.15, 0.04), 0.05, 0.30)),
        "B": float(np.clip(rng.normal(0.98, 0.025), 0.90, 1.02)),
        "C": float(np.clip(rng.normal(0.0, 0.5), -1.5, 1.5)),
        "output_sd": float(np.clip(rng.normal(3.0, 0.75), 1.5, 5.5)),
    }
    if family == "process":
        params["process_sd"] = float(np.clip(rng.lognormal(np.log(1.0), 0.35), 0.25, 2.5))
    return params


def conservative_delta_table(results: pd.DataFrame) -> pd.DataFrame:
    output = results[results.true_model == "output_only"]
    process = results[results.true_model == "process"]
    rows = []
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
        rows.append(best)
    return pd.DataFrame(rows)


def validate_primary_target_scalar(trials: pd.DataFrame, n_reps: int = 80) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = trials[(trials.target_deg == 0.0) & (trials.phase == "diagnostic")].reset_index(drop=True)
    rows = []
    for true_model in ["output_only", "process"]:
        for rep in range(n_reps):
            seed = 33100 + rep + (10000 if true_model == "process" else 0)
            params = sample_scalar_params(true_model, seed)
            data = simulate_scalar(primary, true_model, params, seed=seed)
            output_fit = fit_scalar_kf(data, process=False, maxiter=250)
            process_fit = fit_scalar_kf(data, process=True, maxiter=250)
            delta = output_fit.nll - process_fit.nll
            rows.append(
                {
                    "true_model": true_model,
                    "rep": rep,
                    "delta_process": delta,
                    "best": "process" if delta > 0.0 else "output_only",
                    "output_nll": output_fit.nll,
                    "process_nll": process_fit.nll,
                    **{f"true_{key}": value for key, value in params.items()},
                }
            )
    results = pd.DataFrame(rows)
    confusion = pd.crosstab(results.true_model, results.best, normalize="index")
    thresholds = conservative_delta_table(results)
    return results, confusion, thresholds


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = make_hewitson_replacement_diagnostic_design()
    design.trials.to_csv(OUTPUTS / "hewitson_replacement_diagnostic_trials.csv", index=False)

    scalar_results, scalar_confusion, scalar_thresholds = validate_primary_target_scalar(design.trials)
    scalar_results.to_csv(OUTPUTS / "hewitson_replacement_scalar_kf_results.csv", index=False)
    scalar_confusion.to_csv(OUTPUTS / "hewitson_replacement_scalar_kf_confusion.csv")
    scalar_thresholds.to_csv(OUTPUTS / "hewitson_replacement_scalar_kf_thresholds.csv", index=False)

    features, predictions, matrix, conservative, binary, subtype = validate_design_with_parameter_variability(
        design,
        n_reps=40,
        n_splits=10,
        seed=33200,
    )
    features.to_csv(OUTPUTS / "hewitson_replacement_validation_features.csv", index=False)
    predictions.to_csv(OUTPUTS / "hewitson_replacement_validation_predictions.csv", index=False)
    matrix.to_csv(OUTPUTS / "hewitson_replacement_validation_sixway_confusion.csv")
    conservative.to_csv(OUTPUTS / "hewitson_replacement_validation_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / "hewitson_replacement_validation_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / "hewitson_replacement_validation_subtype.csv", index=False)

    print("Trial counts")
    print(design.trials.groupby(["phase", "feedback_type"]).size().to_string())
    print("\nTarget counts")
    print(design.trials.target_deg.value_counts().sort_index().to_string())
    print("\nPrimary-target scalar KF confusion")
    print(scalar_confusion.round(3).to_string())
    print("\nPrimary-target conservative thresholds")
    print(scalar_thresholds.round(3).to_string(index=False))
    print("\nWhole-design model-free conservative recovery")
    print(conservative.to_string(index=False))
    print("\nWhole-design raw binary recovery")
    print(binary.to_string(index=False))


if __name__ == "__main__":
    main()
