"""Model family definitions for implicit adaptation noise models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    """Metadata for one candidate stochastic update model."""

    name: str
    equation: str
    noise_params: tuple[str, ...]


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "output_only": ModelFamily(
        name="output_only",
        equation="x' = A G e + B x + C",
        noise_params=(),
    ),
    "additive": ModelFamily(
        name="additive",
        equation="x' = A G e + B x + C + n",
        noise_params=("process_sd",),
    ),
    "error_term": ModelFamily(
        name="error_term",
        equation="x' = nA G e + B x + C",
        noise_params=("nA_sd",),
    ),
    "retention_term": ModelFamily(
        name="retention_term",
        equation="x' = A G e + nB x + C",
        noise_params=("nB_sd",),
    ),
    "bias_term": ModelFamily(
        name="bias_term",
        equation="x' = A G e + B x + nC",
        noise_params=("nC_sd",),
    ),
    "full_component": ModelFamily(
        name="full_component",
        equation="x' = nA G e + nB x + nC",
        noise_params=("nA_sd", "nB_sd", "nC_sd"),
    ),
}


def default_params(family: str) -> dict[str, float]:
    """Return representative parameters for simulation and fitting starts."""

    params = {
        "A": 0.15,
        "B": 0.98,
        "C": 0.0,
        "output_sd": 3.0,
        "width": 20.0,
        "process_sd": 1.0,
        "nA_sd": 0.08,
        "nB_sd": 0.025,
        "nC_sd": 1.0,
    }
    keep = {"A", "B", "C", "output_sd", "width", *MODEL_FAMILIES[family].noise_params}
    return {key: value for key, value in params.items() if key in keep}
