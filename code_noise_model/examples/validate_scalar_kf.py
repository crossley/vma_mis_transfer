"""Validate true scalar single-target KF output-vs-process fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.search import generate_hybrid_smooth_designs
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


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = next(d for d in generate_hybrid_smooth_designs(n_trials=240, seed=3500) if d.name == "hybrid_304")
    trials = design.trials.copy()
    trials["target_index"] = 0
    trials["target_deg"] = 0.0
    trials.to_csv(OUTPUTS / "scalar_hybrid_304_trials.csv", index=False)
    rows = []
    n_reps = 80
    for true_model in ["output_only", "process"]:
        for rep in range(n_reps):
            seed = 12100 + rep + (10000 if true_model == "process" else 0)
            params = sample_params(true_model, seed)
            data = simulate_scalar(trials, true_model, params, seed=seed)
            out = fit_scalar_kf(data, process=False, maxiter=250)
            proc = fit_scalar_kf(data, process=True, maxiter=250)
            delta = out.nll - proc.nll
            rows.append(
                {
                    "true_model": true_model,
                    "rep": rep,
                    "delta_process": delta,
                    "best": "process" if delta > 0 else "output_only",
                    "output_nll": out.nll,
                    "process_nll": proc.nll,
                    **{f"out_{k}": v for k, v in out.params.items()},
                    **{f"proc_{k}": v for k, v in proc.params.items()},
                    **{f"true_{k}": v for k, v in params.items()},
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUTS / "scalar_kf_results.csv", index=False)
    matrix = pd.crosstab(results.true_model, results.best, normalize="index")
    matrix.to_csv(OUTPUTS / "scalar_kf_confusion_raw.csv")
    print("Raw scalar KF confusion")
    print(matrix.round(3).to_string())
    print("accuracy", round((results.true_model == results.best).mean(), 3))

    output = results[results.true_model == "output_only"]
    process = results[results.true_model == "process"]
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
    thresholds.to_csv(OUTPUTS / "scalar_kf_thresholds.csv", index=False)
    print("\nConservative thresholds")
    print(thresholds.round(3).to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
