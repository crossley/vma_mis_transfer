"""Plotting helpers for design-identifiability outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .recovery import confusion_matrix


def plot_design(design, path: str) -> None:
    """Plot rotations and feedback types for a trial design."""

    colors = {"rotated": "tab:blue", "clamp": "tab:orange", "none": "tab:gray"}
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for feedback_type, group in design.trials.groupby("feedback_type"):
        values = group["rotation"].copy()
        if feedback_type == "clamp":
            values = group["clamp_error"]
        ax.scatter(group["trial"], values, s=14, label=feedback_type, color=colors[feedback_type], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Rotation or clamp error (deg)")
    ax.set_title(design.name)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(results: pd.DataFrame, design: str, path: str, criterion: str = "heldout") -> None:
    """Plot one normalized model-confusion matrix."""

    matrix = confusion_matrix(results[results.design == design], criterion=criterion)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Recovered model")
    ax.set_ylabel("True model")
    ax.set_title(f"{design}: {criterion} recovery")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parameter_recovery(params: pd.DataFrame, design: str, path: str) -> None:
    """Plot true versus recovered parameters for correctly matched model fits."""

    subset = params[params.design == design]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = []
    values = []
    for parameter, group in subset.groupby("parameter"):
        labels.append(parameter)
        values.append(group["recovered_value"].to_numpy())
    ax.boxplot(values, labels=labels, showfliers=False)
    true_values = subset.groupby("parameter")["true_value"].first()
    for i, parameter in enumerate(labels, start=1):
        ax.scatter(i, true_values[parameter], color="tab:red", s=24, zorder=3)
    ax.set_ylabel("Recovered value")
    ax.set_title(f"{design}: parameter recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_classifier_confusion(matrix: pd.DataFrame, design: str, path: str) -> None:
    """Plot a classifier-based model-free confusion matrix."""

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Predicted model")
    ax.set_ylabel("True model")
    ax.set_title(f"{design}: model-free recovery")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
