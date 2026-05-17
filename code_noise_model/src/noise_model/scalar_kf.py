"""Scalar single-target simulation and KF likelihood utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class ScalarFit:
    family: str
    nll: float
    params: dict[str, float]


@dataclass(frozen=True)
class ScalarData:
    y: np.ndarray
    feedback_code: np.ndarray
    rotation: np.ndarray
    clamp_error: np.ndarray


def prepare_scalar_data(data: pd.DataFrame) -> ScalarData:
    """Convert a trial table to arrays for repeated scalar likelihood calls."""

    feedback = data["feedback_type"].to_numpy()
    feedback_code = np.zeros(len(data), dtype=np.int8)
    feedback_code[feedback == "clamp"] = 1
    feedback_code[(feedback != "none") & (feedback != "clamp")] = 2
    return ScalarData(
        y=data["y"].to_numpy(dtype=float),
        feedback_code=feedback_code,
        rotation=data["rotation"].to_numpy(dtype=float),
        clamp_error=np.nan_to_num(data["clamp_error"].to_numpy(dtype=float), nan=0.0),
    )


def simulate_scalar(
    trials: pd.DataFrame,
    family: str,
    params: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Simulate a scalar single-target adaptation model."""

    rng = np.random.default_rng(seed)
    x = 0.0
    rows = []
    for _, trial in trials.reset_index(drop=True).iterrows():
        y = x + rng.normal(0.0, params["output_sd"])
        if trial.feedback_type == "none":
            error = 0.0
        elif trial.feedback_type == "clamp":
            error = float(trial.clamp_error)
        else:
            error = float(trial.rotation - y)
        if family == "output_only":
            x_next = params["B"] * x + params["A"] * error + params["C"]
        elif family in {"process", "bias_term"}:
            x_next = params["B"] * x + params["A"] * error + params["C"] + rng.normal(0.0, params["process_sd"])
        elif family == "error_term":
            nA = rng.normal(params["A"], params["nA_sd"])
            x_next = params["B"] * x + nA * error + params["C"]
        elif family == "retention_term":
            nB = rng.normal(params["B"], params["nB_sd"])
            x_next = nB * x + params["A"] * error + params["C"]
        elif family == "full_component":
            nA = rng.normal(params["A"], params["nA_sd"])
            nB = rng.normal(params["B"], params["nB_sd"])
            nC = rng.normal(params["C"], params["process_sd"])
            x_next = nB * x + nA * error + nC
        else:
            raise ValueError(family)
        rows.append(
            {
                **trial.to_dict(),
                "x_before": x,
                "y": y,
                "error": error,
                "x_after": x_next,
                "true_model": family,
            }
        )
        x = x_next
    return pd.DataFrame(rows)


def scalar_nll(data: pd.DataFrame, params: dict[str, float], process: bool) -> float:
    """Kalman filter negative log likelihood for scalar model."""

    return scalar_nll_prepared(prepare_scalar_data(data), params, process)


def scalar_nll_prepared(data: ScalarData, params: dict[str, float], process: bool) -> float:
    """Kalman filter negative log likelihood for prepared scalar arrays."""

    mean = 0.0
    var = 1e-4
    nll = 0.0
    q = params.get("process_sd", 0.0) ** 2 if process else 0.0
    r = params["output_sd"] ** 2
    a = params["A"]
    b = params["B"]
    c = params["C"]
    for i in range(len(data.y)):
        pred_var = var + r
        residual = data.y[i] - mean
        nll += 0.5 * (np.log(2 * np.pi * pred_var) + residual**2 / pred_var)
        gain = var / pred_var
        filt_mean = mean + gain * residual
        filt_var = (1.0 - gain) * var
        code = data.feedback_code[i]
        if code == 0:
            error = 0.0
        elif code == 1:
            error = data.clamp_error[i]
        else:
            error = data.rotation[i] - data.y[i]
        mean = b * filt_mean + a * error + c
        var = b**2 * filt_var + q + 1e-9
    return float(nll)


def fit_scalar_kf(data: pd.DataFrame, process: bool, maxiter: int = 300) -> ScalarFit:
    """Fit scalar output-only or additive-process model."""

    prepared = prepare_scalar_data(data)
    keys = ["A", "B", "C", "output_sd"] + (["process_sd"] if process else [])
    start = np.array([0.15, 0.98, 0.0, 3.0] + ([1.0] if process else []), dtype=float)
    bounds = [(-1.0, 1.0), (0.0, 1.2), (-5.0, 5.0), (0.2, 20.0)] + ([(0.001, 20.0)] if process else [])

    def objective(values: np.ndarray) -> float:
        params = dict(zip(keys, [float(value) for value in values], strict=True))
        return scalar_nll_prepared(prepared, params, process=process)

    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    values = result.x
    params = dict(zip(keys, [float(value) for value in values], strict=True))
    return ScalarFit("process" if process else "output_only", objective(values), params)
