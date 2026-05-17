"""Synthetic model-recovery and design-ranking workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .design import TrialDesign
from .fit import FitResult, fit_all_models
from .models import MODEL_FAMILIES, default_params
from .simulate import simulate_subject


def _best_family(fits: list[FitResult], criterion: str) -> str:
    if criterion == "heldout":
        return min(fits, key=lambda fit: fit.heldout_nll).family
    if criterion == "aic":
        return min(fits, key=lambda fit: fit.aic).family
    if criterion == "bic":
        return min(fits, key=lambda fit: fit.bic).family
    raise ValueError(f"Unknown criterion: {criterion}")


def evaluate_design(
    design: TrialDesign,
    n_reps: int = 4,
    seed: int = 10,
    maxiter: int = 120,
    fast: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate and refit all true model families for one design."""

    rows = []
    param_rows = []
    run = 0
    for true_family in MODEL_FAMILIES:
        true_params = default_params(true_family)
        for rep in range(n_reps):
            run_seed = seed + 1000 * run
            data = simulate_subject(design, true_family, true_params, run_seed, subject=rep)
            fits = fit_all_models(data, maxiter=maxiter, fast=fast)
            best_heldout = _best_family(fits, "heldout")
            best_aic = _best_family(fits, "aic")
            best_bic = _best_family(fits, "bic")
            for fit in fits:
                rows.append(
                    {
                        "design": design.name,
                        "true_model": true_family,
                        "rep": rep,
                        "fit_model": fit.family,
                        "nll": fit.nll,
                        "heldout_nll": fit.heldout_nll,
                        "aic": fit.aic,
                        "bic": fit.bic,
                        "best_heldout": best_heldout,
                        "best_aic": best_aic,
                        "best_bic": best_bic,
                    }
                )
                if fit.family == true_family:
                    for key, true_value in true_params.items():
                        if key in fit.params:
                            param_rows.append(
                                {
                                    "design": design.name,
                                    "true_model": true_family,
                                    "rep": rep,
                                    "parameter": key,
                                    "true_value": true_value,
                                    "recovered_value": fit.params[key],
                                }
                            )
            run += 1
    return pd.DataFrame(rows), pd.DataFrame(param_rows)


def confusion_matrix(results: pd.DataFrame, criterion: str = "heldout") -> pd.DataFrame:
    """Return normalized model-recovery matrix for one selection criterion."""

    best_col = f"best_{criterion}"
    one_per_run = results[["design", "true_model", "rep", best_col]].drop_duplicates()
    matrix = pd.crosstab(one_per_run["true_model"], one_per_run[best_col], normalize="index")
    for family in MODEL_FAMILIES:
        if family not in matrix.columns:
            matrix[family] = 0.0
    return matrix[list(MODEL_FAMILIES)]


def score_design(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize recovery accuracy for each design."""

    rows = []
    for design, group in results.groupby("design"):
        one_per_run = group[["true_model", "rep", "best_heldout", "best_aic", "best_bic"]].drop_duplicates()
        rows.append(
            {
                "design": design,
                "heldout_accuracy": np.mean(one_per_run.true_model == one_per_run.best_heldout),
                "aic_accuracy": np.mean(one_per_run.true_model == one_per_run.best_aic),
                "bic_accuracy": np.mean(one_per_run.true_model == one_per_run.best_bic),
                "mean_accuracy": np.mean(
                    [
                        np.mean(one_per_run.true_model == one_per_run.best_heldout),
                        np.mean(one_per_run.true_model == one_per_run.best_aic),
                        np.mean(one_per_run.true_model == one_per_run.best_bic),
                    ]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)


def evaluate_designs(
    designs: list[TrialDesign],
    n_reps: int = 4,
    seed: int = 10,
    maxiter: int = 120,
    fast: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate and rank a list of candidate designs."""

    result_frames = []
    param_frames = []
    for i, design in enumerate(designs):
        results, params = evaluate_design(
            design,
            n_reps=n_reps,
            seed=seed + i * 100,
            maxiter=maxiter,
            fast=fast,
        )
        result_frames.append(results)
        param_frames.append(params)
    all_results = pd.concat(result_frames, ignore_index=True)
    all_params = pd.concat(param_frames, ignore_index=True)
    ranking = score_design(all_results)
    return all_results, all_params, ranking
