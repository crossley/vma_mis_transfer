"""Statsmodels Kalman-filter likelihood fits for simplified noise models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.statespace.mlemodel import MLEModel

from .design import TARGETS_DEG
from .simulate import generalization_matrix, learning_error


@dataclass(frozen=True)
class StatsmodelsKFFit:
    """Fit summary for a statsmodels Kalman-filter likelihood model."""

    family: str
    success: bool
    nll: float
    aic: float
    bic: float
    params: dict[str, float]


class AdaptationSSM(MLEModel):
    """Linear Gaussian SSM for y = Hx + eps, x' = Bx + AGe + C + eta."""

    def __init__(self, data: pd.DataFrame, estimate_process_noise: bool):
        self.trial_data = data.sort_values("trial").reset_index(drop=True)
        self.estimate_process_noise = estimate_process_noise
        self.n_targets = len(TARGETS_DEG)
        super().__init__(
            endog=self.trial_data["y"].to_numpy(dtype=float),
            k_states=self.n_targets,
            k_posdef=self.n_targets,
            initialization="approximate_diffuse",
        )
        design = np.zeros((1, self.n_targets, len(self.trial_data)))
        for t, row in self.trial_data.iterrows():
            design[0, int(row.target_index), t] = 1.0
        self.ssm["design"] = design
        self.ssm["selection"] = np.eye(self.n_targets)

    @property
    def param_names(self) -> list[str]:
        names = ["A", "B", "C", "output_sd", "width"]
        if self.estimate_process_noise:
            names.append("process_sd")
        return names

    @property
    def start_params(self) -> np.ndarray:
        values = [0.15, 0.98, 0.0, 3.0, 20.0]
        if self.estimate_process_noise:
            values.append(1.0)
        return np.array(values, dtype=float)

    def update(self, params, transformed=True, includes_fixed=False, **kwargs) -> None:
        params = np.asarray(params, dtype=float)
        A, B, C, output_sd, width = params[:5]
        process_sd = params[5] if self.estimate_process_noise else 1e-9
        gen = generalization_matrix(width)
        intercept = np.zeros((self.n_targets, len(self.trial_data)))
        for t, row in self.trial_data.iterrows():
            error = learning_error(row.feedback_type, row.rotation, row.clamp_error, row.y)
            intercept[:, t] = A * gen[:, int(row.target_index)] * error + C
        self.ssm["transition"] = np.eye(self.n_targets) * B
        self.ssm["state_intercept"] = intercept
        self.ssm["obs_cov"] = np.array([[output_sd**2]])
        self.ssm["state_cov"] = np.eye(self.n_targets) * process_sd**2


def fit_statsmodels_kf(
    data: pd.DataFrame,
    family: str,
    maxiter: int = 300,
) -> StatsmodelsKFFit:
    """Fit output-only or additive-process SSM with statsmodels Kalman likelihood."""

    if family not in {"output_only", "process"}:
        raise ValueError("family must be 'output_only' or 'process'")
    model = AdaptationSSM(data, estimate_process_noise=(family == "process"))
    start = model.start_params
    bounds = [(-1.0, 1.0), (0.0, 1.2), (-5.0, 5.0), (0.2, 20.0), (1.0, 90.0)]
    if family == "process":
        bounds.append((0.001, 20.0))

    def objective(values: np.ndarray) -> float:
        return -float(model.loglike(values))

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    values = result.x
    nll = objective(values)
    params = dict(zip(model.param_names, [float(value) for value in values], strict=True))
    k = len(values)
    n = len(data)
    return StatsmodelsKFFit(
        family=family,
        success=bool(result.success),
        nll=nll,
        aic=2.0 * k + 2.0 * nll,
        bic=k * np.log(n) + 2.0 * nll,
        params=params,
    )


def fit_statsmodels_kf_binary(data: pd.DataFrame, maxiter: int = 300) -> tuple[StatsmodelsKFFit, StatsmodelsKFFit]:
    """Fit output-only and additive-process statsmodels SSMs."""

    return (
        fit_statsmodels_kf(data, "output_only", maxiter=maxiter),
        fit_statsmodels_kf(data, "process", maxiter=maxiter),
    )
