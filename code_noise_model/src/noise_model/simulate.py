"""Simulation for multidimensional implicit adaptation noise models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .design import TARGETS_DEG, TrialDesign
from .models import MODEL_FAMILIES


def generalization_matrix(width: float, targets: np.ndarray = TARGETS_DEG) -> np.ndarray:
    """Gaussian generalization matrix over target directions."""

    distance = targets[:, None] - targets[None, :]
    return np.exp(-(distance**2) / (2.0 * width**2))


def learning_error(feedback_type: str, rotation: float, clamp_error: float, y: float) -> float:
    """Return the scalar error that drives implicit updating."""

    if feedback_type == "none":
        return 0.0
    if feedback_type == "clamp":
        return float(clamp_error)
    return float(rotation - y)


def simulate_subject(
    design: TrialDesign,
    family: str,
    params: dict[str, float],
    seed: int,
    subject: int = 0,
) -> pd.DataFrame:
    """Simulate one synthetic participant for one model family."""

    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family}")

    rng = np.random.default_rng(seed)
    trials = design.trials.reset_index(drop=True)
    n_targets = len(TARGETS_DEG)
    gen = generalization_matrix(params["width"])
    x = np.zeros(n_targets)
    rows = []

    for _, trial in trials.iterrows():
        target_index = int(trial.target_index)
        x_before = x.copy()
        y = x[target_index] + rng.normal(0.0, params["output_sd"])
        error = learning_error(
            trial.feedback_type,
            trial.rotation,
            trial.clamp_error,
            y,
        )
        g = gen[:, target_index]

        A = params.get("A", 0.0)
        B = params.get("B", 0.0)
        C = params.get("C", 0.0)

        if family == "output_only":
            x = B * x + A * g * error + C
        elif family == "additive":
            x = B * x + A * g * error + C + rng.normal(0.0, params["process_sd"], n_targets)
        elif family == "error_term":
            nA = rng.normal(A, params["nA_sd"])
            x = B * x + nA * g * error + C
        elif family == "retention_term":
            nB = rng.normal(B, params["nB_sd"], n_targets)
            x = nB * x + A * g * error + C
        elif family == "bias_term":
            nC = rng.normal(C, params["nC_sd"], n_targets)
            x = B * x + A * g * error + nC
        elif family == "full_component":
            nA = rng.normal(A, params["nA_sd"])
            nB = rng.normal(B, params["nB_sd"], n_targets)
            nC = rng.normal(C, params["nC_sd"], n_targets)
            x = nB * x + nA * g * error + nC
        else:
            raise AssertionError("Unhandled family")

        rows.append(
            {
                "subject": subject,
                "true_model": family,
                "design": design.name,
                "trial": int(trial.trial),
                "phase": trial.phase,
                "target_index": target_index,
                "target_deg": float(trial.target_deg),
                "feedback_type": trial.feedback_type,
                "rotation": float(trial.rotation),
                "clamp_error": trial.clamp_error,
                "is_probe": bool(trial.is_probe),
                "x_before": float(x_before[target_index]),
                "y": float(y),
                "error": float(error),
                "x_after": float(x[target_index]),
            }
        )

    return pd.DataFrame(rows)
