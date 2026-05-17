"""Parameter recovery for the fully shuffled no-clamp diagnostic design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.design import (
    make_hewitson_replacement_drift_block_design,
    make_hewitson_replacement_no_clamp_shuffled_design,
    make_hewitson_replacement_process_focused_design,
)
from noise_model.scalar_kf import fit_scalar_kf, simulate_scalar


OUTPUTS = ROOT / "outputs"


def sample_params(family: str, seed: int) -> dict[str, float]:
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


def parameter_summary(results: pd.DataFrame, family: str, parameters: list[str]) -> pd.DataFrame:
    rows = []
    subset = results[(results.true_model == family) & (results.best == family)].copy()
    for parameter in parameters:
        true = subset[f"true_{parameter}"].to_numpy(dtype=float)
        recovered = subset[f"fit_{parameter}"].to_numpy(dtype=float)
        if len(true) < 3 or np.isclose(np.std(true), 0.0) or np.isclose(np.std(recovered), 0.0):
            corr = np.nan
        else:
            corr = float(np.corrcoef(true, recovered)[0, 1])
        error = recovered - true
        rows.append(
            {
                "true_model": family,
                "parameter": parameter,
                "n_correct": int(len(subset)),
                "true_mean": float(np.mean(true)) if len(true) else np.nan,
                "recovered_mean": float(np.mean(recovered)) if len(recovered) else np.nan,
                "bias": float(np.mean(error)) if len(error) else np.nan,
                "mae": float(np.mean(np.abs(error))) if len(error) else np.nan,
                "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                "correlation": corr,
            }
        )
    return pd.DataFrame(rows)


def evaluate_design(design, n_reps: int, seed_offset: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    primary = design.trials[(design.trials.phase == "diagnostic") & (design.trials.target_deg == 0.0)]
    rows = []
    for true_model in ["output_only", "process"]:
        for rep in range(n_reps):
            seed = seed_offset + rep + (10000 if true_model == "process" else 0)
            params = sample_params(true_model, seed)
            data = simulate_scalar(primary, true_model, params, seed=seed)
            output_fit = fit_scalar_kf(data, process=False, maxiter=180)
            process_fit = fit_scalar_kf(data, process=True, maxiter=180)
            best = "process" if output_fit.nll - process_fit.nll > 0.0 else "output_only"
            chosen = process_fit if best == "process" else output_fit
            row = {
                "design": design.name,
                "true_model": true_model,
                "rep": rep,
                "best": best,
                "delta_process": output_fit.nll - process_fit.nll,
                "output_nll": output_fit.nll,
                "process_nll": process_fit.nll,
            }
            for parameter, value in params.items():
                row[f"true_{parameter}"] = value
            for parameter, value in chosen.params.items():
                row[f"fit_{parameter}"] = value
            if "true_process_sd" not in row:
                row["true_process_sd"] = 0.0
            if "fit_process_sd" not in row:
                row["fit_process_sd"] = 0.0
            rows.append(row)

    results = pd.DataFrame(rows)
    confusion = pd.crosstab(results.true_model, results.best, normalize="index")
    summary = pd.concat(
        [
            parameter_summary(results, "output_only", ["A", "B", "C", "output_sd"]),
            parameter_summary(results, "process", ["A", "B", "C", "output_sd", "process_sd"]),
        ],
        ignore_index=True,
    )
    summary.insert(0, "design", design.name)
    counts = primary.feedback_type.value_counts()
    return results, confusion, summary, counts


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    designs = [
        make_hewitson_replacement_no_clamp_shuffled_design(),
        make_hewitson_replacement_process_focused_design(),
        make_hewitson_replacement_drift_block_design(),
    ]
    all_results = []
    all_summary = []
    for i, design in enumerate(designs):
        design.trials.to_csv(OUTPUTS / f"{design.name}_trials.csv", index=False)
        results, confusion, summary, counts = evaluate_design(design, n_reps=80, seed_offset=77100 + i * 30000)
        results.to_csv(OUTPUTS / f"{design.name}_parameter_recovery_results.csv", index=False)
        confusion.to_csv(OUTPUTS / f"{design.name}_parameter_recovery_confusion.csv")
        summary.to_csv(OUTPUTS / f"{design.name}_parameter_recovery_summary.csv", index=False)
        all_results.append(results)
        all_summary.append(summary)

        print(f"\n{design.name}")
        print("Primary target diagnostic trial count", int(counts.sum()))
        print(counts.to_string())
        print("\nModel recovery")
        print(confusion.round(3).to_string())
        print("\nParameter recovery among correctly identified datasets")
        cols = ["design", "true_model", "parameter", "n_correct", "bias", "mae", "rmse", "correlation"]
        print(summary[cols].round(3).to_string(index=False))

    pd.concat(all_results, ignore_index=True).to_csv(OUTPUTS / "parameter_recovery_comparison_results.csv", index=False)
    pd.concat(all_summary, ignore_index=True).to_csv(OUTPUTS / "parameter_recovery_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
