"""Robust validation under synthetic participant parameter variability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .classifier_recovery import evaluate_design_classifier
from .diagnostics import conservative_threshold_table, output_vs_process_table, process_subtype_table
from .models import default_params


def variable_parameter_sampler(family: str, subject: int, seed: int) -> dict[str, float]:
    """Draw plausible subject-level parameters around representative defaults."""

    rng = np.random.default_rng(seed + subject * 1009)
    params = default_params(family)
    params["A"] = float(np.clip(rng.normal(0.15, 0.04), 0.05, 0.30))
    params["B"] = float(np.clip(rng.normal(0.98, 0.025), 0.90, 1.02))
    params["C"] = float(np.clip(rng.normal(0.0, 0.5), -1.5, 1.5))
    params["output_sd"] = float(np.clip(rng.normal(3.0, 0.75), 1.5, 5.5))
    params["width"] = float(np.clip(rng.normal(20.0, 6.0), 8.0, 40.0))
    if "process_sd" in params:
        params["process_sd"] = float(np.clip(rng.lognormal(np.log(1.0), 0.35), 0.25, 2.5))
    if "nA_sd" in params:
        params["nA_sd"] = float(np.clip(rng.lognormal(np.log(0.08), 0.35), 0.02, 0.25))
    if "nB_sd" in params:
        params["nB_sd"] = float(np.clip(rng.lognormal(np.log(0.025), 0.35), 0.005, 0.08))
    if "nC_sd" in params:
        params["nC_sd"] = float(np.clip(rng.lognormal(np.log(1.0), 0.35), 0.25, 2.5))
    return params


def validate_design_with_parameter_variability(
    design,
    n_reps: int = 160,
    n_splits: int = 30,
    seed: int = 900,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run model-free recovery for one design with variable synthetic subjects."""

    features, predictions, matrix = evaluate_design_classifier(
        design,
        n_reps=n_reps,
        n_splits=n_splits,
        seed=seed,
        parameter_sampler=variable_parameter_sampler,
    )
    conservative = conservative_threshold_table(predictions)
    binary = output_vs_process_table(predictions)
    subtype = process_subtype_table(predictions)
    return features, predictions, matrix, conservative, binary, subtype
