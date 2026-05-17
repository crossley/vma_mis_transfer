"""Compare no-clamp Hewitson replacement variants with scalar KF recovery."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.design import make_hewitson_replacement_no_clamp_design
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


def threshold_table(results: pd.DataFrame) -> pd.DataFrame:
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


def evaluate_design(design, n_reps: int, seed_offset: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = design.trials[(design.trials.phase == "diagnostic") & (design.trials.target_deg == 0.0)]
    rows = []
    for true_model in ["output_only", "process"]:
        for rep in range(n_reps):
            seed = seed_offset + rep + (10000 if true_model == "process" else 0)
            params = sample_params(true_model, seed)
            data = simulate_scalar(primary, true_model, params, seed=seed)
            output_fit = fit_scalar_kf(data, process=False, maxiter=120)
            process_fit = fit_scalar_kf(data, process=True, maxiter=120)
            delta = output_fit.nll - process_fit.nll
            rows.append(
                {
                    "design": design.name,
                    "true_model": true_model,
                    "rep": rep,
                    "delta_process": delta,
                    "best": "process" if delta > 0.0 else "output_only",
                }
            )
    results = pd.DataFrame(rows)
    confusion = pd.crosstab(results.true_model, results.best, normalize="index")
    thresholds = threshold_table(results)
    counts = primary.feedback_type.value_counts().rename_axis("feedback_type").reset_index(name="n")
    counts.insert(0, "design", design.name)
    return results, confusion, thresholds, counts


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    designs = [
        make_hewitson_replacement_no_clamp_design(high_primary_density=False),
        make_hewitson_replacement_no_clamp_design(high_primary_density=True),
    ]
    all_results = []
    all_thresholds = []
    all_counts = []
    for i, design in enumerate(designs):
        design.trials.to_csv(OUTPUTS / f"{design.name}_trials.csv", index=False)
        results, confusion, thresholds, counts = evaluate_design(design, n_reps=40, seed_offset=66100 + i * 1000)
        results.to_csv(OUTPUTS / f"{design.name}_scalar_results.csv", index=False)
        confusion.to_csv(OUTPUTS / f"{design.name}_scalar_confusion.csv")
        thresholds.insert(0, "design", design.name)
        confusion_out = confusion.copy()
        confusion_out.insert(0, "true_model", confusion_out.index)
        confusion_out.to_csv(OUTPUTS / f"{design.name}_scalar_confusion_flat.csv", index=False)
        all_results.append(results)
        all_thresholds.append(thresholds)
        all_counts.append(counts)
        print(f"\n{design.name}")
        print("primary target trial count", int(counts.n.sum()))
        print(counts[["feedback_type", "n"]].to_string(index=False))
        print(confusion.round(3).to_string())
        print(thresholds.round(3).to_string(index=False))
    pd.concat(all_results, ignore_index=True).to_csv(OUTPUTS / "no_clamp_scalar_results.csv", index=False)
    pd.concat(all_thresholds, ignore_index=True).to_csv(OUTPUTS / "no_clamp_scalar_thresholds.csv", index=False)
    pd.concat(all_counts, ignore_index=True).to_csv(OUTPUTS / "no_clamp_primary_counts.csv", index=False)


if __name__ == "__main__":
    main()
