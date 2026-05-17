"""Constrained fast KF check for the Hewitson replacement diagnostic block."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.design import make_hewitson_replacement_diagnostic_design
from noise_model.scalar_kf import prepare_scalar_data, scalar_nll_prepared, simulate_scalar


OUTPUTS = ROOT / "outputs"
OUTPUT_GRID = np.linspace(1.0, 6.5, 24)
PROCESS_GRID = np.r_[0.001, np.linspace(0.25, 3.0, 20)]


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


def best_constrained_nll(data: pd.DataFrame, mean_params: dict[str, float], process: bool) -> float:
    prepared = prepare_scalar_data(data)
    best = np.inf
    if process:
        for output_sd in OUTPUT_GRID:
            for process_sd in PROCESS_GRID:
                params = {**mean_params, "output_sd": float(output_sd), "process_sd": float(process_sd)}
                best = min(best, scalar_nll_prepared(prepared, params, process=True))
    else:
        for output_sd in OUTPUT_GRID:
            params = {**mean_params, "output_sd": float(output_sd)}
            best = min(best, scalar_nll_prepared(prepared, params, process=False))
    return float(best)


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = make_hewitson_replacement_diagnostic_design()
    design.trials.to_csv(OUTPUTS / "hewitson_replacement_diagnostic_trials.csv", index=False)
    primary = design.trials[(design.trials.phase == "diagnostic") & (design.trials.target_deg == 0.0)]
    rows = []
    for true_model in ["output_only", "process"]:
        for rep in range(100):
            seed = 55100 + rep + (10000 if true_model == "process" else 0)
            params = sample_params(true_model, seed)
            data = simulate_scalar(primary, true_model, params, seed=seed)
            mean_params = {key: params[key] for key in ["A", "B", "C"]}
            output_nll = best_constrained_nll(data, mean_params, process=False)
            process_nll = best_constrained_nll(data, mean_params, process=True)
            delta = output_nll - process_nll
            rows.append({"true_model": true_model, "rep": rep, "delta_process": delta, "best": "process" if delta > 0 else "output_only"})
    results = pd.DataFrame(rows)
    confusion = pd.crosstab(results.true_model, results.best, normalize="index")
    threshold_rows = []
    output = results[results.true_model == "output_only"]
    process = results[results.true_model == "process"]
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
    results.to_csv(OUTPUTS / "hewitson_replacement_constrained_fast_results.csv", index=False)
    confusion.to_csv(OUTPUTS / "hewitson_replacement_constrained_fast_confusion.csv")
    thresholds.to_csv(OUTPUTS / "hewitson_replacement_constrained_fast_thresholds.csv", index=False)
    print("Primary-target trial count", len(primary))
    print("Primary-target feedback counts")
    print(primary.feedback_type.value_counts().to_string())
    print("\nConstrained KF confusion")
    print(confusion.round(3).to_string())
    print("\nConservative thresholds")
    print(thresholds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
