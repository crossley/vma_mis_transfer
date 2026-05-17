"""Approximate likelihood fitting for candidate noise-model families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .design import TARGETS_DEG
from .models import MODEL_FAMILIES, default_params
from .simulate import generalization_matrix, learning_error


@dataclass(frozen=True)
class FitResult:
    """Result of fitting one model family to one dataset."""

    family: str
    success: bool
    nll: float
    aic: float
    bic: float
    train_nll: float
    heldout_nll: float
    params: dict[str, float]


BOUNDS = {
    "A": (-1.0, 1.0),
    "B": (0.0, 1.2),
    "C": (-5.0, 5.0),
    "output_sd": (0.2, 20.0),
    "width": (1.0, 90.0),
    "process_sd": (0.001, 20.0),
    "nA_sd": (0.001, 2.0),
    "nB_sd": (0.001, 0.5),
    "nC_sd": (0.001, 20.0),
}


def _params_from_vector(keys: list[str], values: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(keys, values, strict=True)}


def _transition_covariance(
    family: str,
    params: dict[str, float],
    mean: np.ndarray,
    g: np.ndarray,
    error: float,
) -> np.ndarray:
    n = len(mean)
    q = np.zeros((n, n))
    if family == "additive":
        q += np.eye(n) * params["process_sd"] ** 2
    elif family == "error_term":
        q += np.outer(g, g) * (params["nA_sd"] * error) ** 2
    elif family == "retention_term":
        q += np.diag((params["nB_sd"] * mean) ** 2)
    elif family == "bias_term":
        q += np.eye(n) * params["nC_sd"] ** 2
    elif family == "full_component":
        q += np.outer(g, g) * (params["nA_sd"] * error) ** 2
        q += np.diag((params["nB_sd"] * mean) ** 2)
        q += np.eye(n) * params["nC_sd"] ** 2
    return q


def sequence_nll(
    data: pd.DataFrame,
    family: str,
    params: dict[str, float],
    mask: np.ndarray | None = None,
) -> float:
    """Approximate negative log likelihood using Gaussian filtering."""

    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family}")

    rows = data.sort_values("trial").reset_index(drop=True)
    if mask is None:
        mask = np.ones(len(rows), dtype=bool)

    n_targets = len(TARGETS_DEG)
    mean = np.zeros(n_targets)
    cov = np.eye(n_targets) * 0.01
    gen = generalization_matrix(params["width"])
    nll = 0.0

    for i, row in rows.iterrows():
        idx = int(row.target_index)
        h = np.zeros(n_targets)
        h[idx] = 1.0
        pred_mean = float(h @ mean)
        pred_var = float(h @ cov @ h + params["output_sd"] ** 2)
        pred_var = max(pred_var, 1e-9)
        residual = float(row.y - pred_mean)

        if mask[i]:
            nll += 0.5 * (np.log(2.0 * np.pi * pred_var) + residual**2 / pred_var)

        kalman_gain = cov @ h / pred_var
        mean = mean + kalman_gain * residual
        cov = cov - np.outer(kalman_gain, h) @ cov
        cov = (cov + cov.T) / 2.0

        error = learning_error(row.feedback_type, row.rotation, row.clamp_error, row.y)
        g = gen[:, idx]
        mean_before_transition = mean.copy()
        mean = params["B"] * mean + params["A"] * g * error + params["C"]
        cov = params["B"] ** 2 * cov
        cov += _transition_covariance(family, params, mean_before_transition, g, error)
        cov = (cov + cov.T) / 2.0 + np.eye(n_targets) * 1e-8

    return float(nll)


def fit_model(
    data: pd.DataFrame,
    family: str,
    heldout_every: int = 5,
    maxiter: int = 160,
    fast: bool = True,
) -> FitResult:
    """Fit one model family by bounded numerical optimization."""

    start = default_params(family)
    keys = list(start)
    x0 = np.array([start[key] for key in keys], dtype=float)
    bounds = [BOUNDS[key] for key in keys]
    train_mask = (data.sort_values("trial").reset_index(drop=True).index.to_numpy() % heldout_every) != 0
    heldout_mask = ~train_mask

    if maxiter <= 0:
        train_nll = sequence_nll(data, family, start, mask=train_mask)
        heldout_nll = sequence_nll(data, family, start, mask=heldout_mask)
        total_nll = sequence_nll(data, family, start)
        k = len(keys)
        n = len(data)
        return FitResult(
            family=family,
            success=True,
            nll=total_nll,
            aic=2.0 * k + 2.0 * total_nll,
            bic=k * np.log(n) + 2.0 * total_nll,
            train_nll=train_nll,
            heldout_nll=heldout_nll,
            params=start,
        )

    def objective(values: np.ndarray) -> float:
        params = _params_from_vector(keys, values)
        return sequence_nll(data, family, params, mask=train_mask)

    if fast:
        clipped = x0
        result_success = True
    else:
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3},
        )
        clipped = np.array(
            [np.clip(value, low, high) for value, (low, high) in zip(result.x, bounds, strict=True)]
        )
        result_success = bool(result.success)
    result2 = minimize(
        objective,
        clipped,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    values = result2.x if result2.fun <= objective(clipped) else clipped
    params = _params_from_vector(keys, values)
    train_nll = sequence_nll(data, family, params, mask=train_mask)
    heldout_nll = sequence_nll(data, family, params, mask=heldout_mask)
    total_nll = sequence_nll(data, family, params)
    k = len(keys)
    n = len(data)
    return FitResult(
        family=family,
        success=bool(result2.success or result_success),
        nll=total_nll,
        aic=2.0 * k + 2.0 * total_nll,
        bic=k * np.log(n) + 2.0 * total_nll,
        train_nll=train_nll,
        heldout_nll=heldout_nll,
        params=params,
    )


def fit_all_models(data: pd.DataFrame, maxiter: int = 160, fast: bool = True) -> list[FitResult]:
    """Fit every candidate family to one dataset."""

    return [fit_model(data, family, maxiter=maxiter, fast=fast) for family in MODEL_FAMILIES]


def fit_kf_binary_constrained(
    data: pd.DataFrame,
    base_params: dict[str, float],
    maxiter: int = 120,
) -> tuple[FitResult, FitResult]:
    """Fit output-only vs additive-like process noise with shared mean dynamics.

    This constrains A, B, C, and width to `base_params` and fits only observation
    noise for output-only, and observation plus additive process noise for the
    process model.
    """

    def result_for(family: str, params: dict[str, float]) -> FitResult:
        total = sequence_nll(data, family, params)
        heldout_mask = (data.sort_values("trial").reset_index(drop=True).index.to_numpy() % 5) == 0
        train_mask = ~heldout_mask
        train = sequence_nll(data, family, params, mask=train_mask)
        heldout = sequence_nll(data, family, params, mask=heldout_mask)
        k = 1 if family == "output_only" else 2
        n = len(data)
        return FitResult(
            family=family,
            success=True,
            nll=total,
            aic=2.0 * k + 2.0 * total,
            bic=k * np.log(n) + 2.0 * total,
            train_nll=train,
            heldout_nll=heldout,
            params=params,
        )

    shared = {
        "A": base_params["A"],
        "B": base_params["B"],
        "C": base_params["C"],
        "width": base_params["width"],
    }

    def output_objective(values: np.ndarray) -> float:
        params = {**shared, "output_sd": float(values[0])}
        return sequence_nll(data, "output_only", params)

    output_fit = minimize(
        output_objective,
        np.array([base_params["output_sd"]]),
        method="L-BFGS-B",
        bounds=[BOUNDS["output_sd"]],
        options={"maxiter": maxiter},
    )
    output_params = {**shared, "output_sd": float(output_fit.x[0])}

    def process_objective(values: np.ndarray) -> float:
        params = {
            **shared,
            "output_sd": float(values[0]),
            "nC_sd": float(values[1]),
        }
        return sequence_nll(data, "bias_term", params)

    process_fit = minimize(
        process_objective,
        np.array([base_params["output_sd"], base_params.get("nC_sd", 1.0)]),
        method="L-BFGS-B",
        bounds=[BOUNDS["output_sd"], BOUNDS["nC_sd"]],
        options={"maxiter": maxiter},
    )
    process_params = {
        **shared,
        "output_sd": float(process_fit.x[0]),
        "nC_sd": float(process_fit.x[1]),
    }
    return result_for("output_only", output_params), result_for("bias_term", process_params)


def fit_mean_dynamics(
    data: pd.DataFrame,
    start_params: dict[str, float],
    maxiter: int = 180,
) -> dict[str, float]:
    """Estimate shared mean dynamics under an output-only state-space model."""

    keys = ["A", "B", "C", "output_sd", "width"]
    x0 = np.array([start_params[key] for key in keys], dtype=float)
    bounds = [BOUNDS[key] for key in keys]

    def objective(values: np.ndarray) -> float:
        params = _params_from_vector(keys, values)
        return sequence_nll(data, "output_only", params)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    values = result.x if result.success else x0
    return _params_from_vector(keys, values)


def fit_kf_binary_two_stage(
    data: pd.DataFrame,
    start_params: dict[str, float],
    maxiter_mean: int = 180,
    maxiter_noise: int = 120,
) -> tuple[FitResult, FitResult, dict[str, float]]:
    """Estimate mean dynamics, then compare output-only vs process noise."""

    mean_params = fit_mean_dynamics(data, start_params, maxiter=maxiter_mean)
    process_start = {**mean_params, "nC_sd": start_params.get("nC_sd", 1.0)}
    output_fit, process_fit = fit_kf_binary_constrained(
        data,
        process_start,
        maxiter=maxiter_noise,
    )
    return output_fit, process_fit, mean_params
