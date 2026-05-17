"""Plot the full Hewitson replacement diagnostic design."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.design import (
    make_hewitson_replacement_diagnostic_design,
    make_hewitson_replacement_drift_block_design,
    make_hewitson_replacement_no_clamp_design,
    make_hewitson_replacement_no_clamp_shuffled_design,
    make_hewitson_replacement_process_focused_design,
)


OUTPUTS = ROOT / "outputs"


def _shuffle_no_more_than_two_same_target(targets: list[float], seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        candidate = list(targets)
        rng.shuffle(candidate)
        if all(candidate[i] != candidate[i - 1] or candidate[i] != candidate[i - 2] for i in range(2, len(candidate))):
            return candidate
    return candidate


def _shuffle_adaptation_feedback(seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        labels = ["none"] * 11 + ["rotated"] * 99
        rng.shuffle(labels)
        if labels[:5].count("none") > 0:
            continue
        if any(labels[i] == labels[i - 1] == "none" for i in range(1, len(labels))):
            continue
        bin_counts = [labels[start : start + 10].count("none") for start in range(0, 110, 10)]
        if max(bin_counts) <= 2:
            return labels
    return ["none" if i % 10 == 9 else "rotated" for i in range(110)]


def _shuffled_target_cycles(n_trials: int, seed: int) -> list[float]:
    """Return targets in shuffled five-target cycles."""

    rng = np.random.default_rng(seed)
    targets = [-30.0, -15.0, 0.0, 15.0, 30.0]
    out = []
    while len(out) < n_trials:
        cycle = targets.copy()
        rng.shuffle(cycle)
        out.extend(cycle)
    return out[:n_trials]


def shuffle_familiarization(pre_trials: pd.DataFrame, seed: int = 20260424) -> pd.DataFrame:
    """Shuffle the familiarization phase while keeping target counts fixed."""

    familiar = pre_trials[pre_trials["phase"] == "familiarization"].copy()
    rest = pre_trials[pre_trials["phase"] != "familiarization"].copy()
    targets = _shuffle_no_more_than_two_same_target(familiar["target_deg"].tolist(), seed)
    lookup = {-30.0: 0, -15.0: 1, 0.0: 2, 15.0: 3, 30.0: 4}
    familiar["target_deg"] = targets
    familiar["target_index"] = [lookup[target] for target in targets]
    out = pd.concat([familiar, rest], ignore_index=True)
    out["trial"] = np.arange(len(out))
    return out


def append_hewitson_adaptation_and_generalization(
    pre_trials: pd.DataFrame,
    *,
    shuffle_post_phases: bool = False,
    seed: int = 20260424,
    adaptation_rotation: float = 30.0,
) -> pd.DataFrame:
    """Append Hewitson-style adaptation and generalization phases to a replacement block."""

    rows = []
    next_trial = int(pre_trials["trial"].max()) + 1
    adaptation_feedback = (
        _shuffle_adaptation_feedback(seed + 1)
        if shuffle_post_phases
        else ["none" if i % 10 == 9 else "rotated" for i in range(110)]
    )
    targets = [-30.0, -15.0, 0.0, 15.0, 30.0]
    target_lookup = {-30.0: 0, -15.0: 1, 0.0: 2, 15.0: 3, 30.0: 4}
    adaptation_targets = _shuffled_target_cycles(110, seed + 2)
    for i in range(110):
        no_feedback = adaptation_feedback[i] == "none"
        target = adaptation_targets[i]
        rows.append(
            {
                "trial": next_trial + i,
                "target_index": target_lookup[target],
                "phase": "adaptation",
                "feedback_type": "none" if no_feedback else "rotated",
                "rotation": 0.0 if no_feedback else adaptation_rotation,
                "clamp_error": np.nan,
                "is_probe": no_feedback,
                "target_deg": target,
            }
        )
    next_trial += 110

    generalization_targets = _shuffled_target_cycles(66, seed + 3)
    for i, target in enumerate(generalization_targets):
        rows.append(
            {
                "trial": next_trial + i,
                "target_index": target_lookup[target],
                "phase": "generalization",
                "feedback_type": "none",
                "rotation": 0.0,
                "clamp_error": np.nan,
                "is_probe": True,
                "target_deg": target,
            }
        )
    return pd.concat([pre_trials, pd.DataFrame(rows)], ignore_index=True)


def plot_design(trials: pd.DataFrame, path: Path, title: str | None = None) -> None:
    type_order = [feedback_type for feedback_type in ["none", "rotated", "clamp"] if feedback_type in set(trials["feedback_type"])]
    type_labels = {"none": "no-fb", "rotated": "endpoint fb", "clamp": "clamp"}
    type_y = {name: i for i, name in enumerate(type_order)}
    colors = {"none": "#5f6368", "rotated": "#2563eb", "clamp": "#c2410c"}

    trial = trials["trial"].to_numpy()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 0.9]},
    )

    axes[0].axhline(0, color="#1f2937", linewidth=0.8)
    for feedback_type, group in trials.groupby("feedback_type", sort=False):
        values = group["rotation"].copy()
        if feedback_type == "clamp":
            values = group["clamp_error"]
        axes[0].scatter(
            group["trial"],
            values,
            s=18,
            color=colors[feedback_type],
            label=type_labels[feedback_type],
            alpha=0.9,
            linewidths=0,
        )
    axes[0].set_ylabel("Rotation\n(deg)")
    rotation_min = min(-10.5, float(np.nanmin(trials["rotation"].to_numpy(dtype=float))) - 3.0)
    rotation_max = max(10.5, float(np.nanmax(trials["rotation"].to_numpy(dtype=float))) + 3.0)
    axes[0].set_ylim(rotation_min, rotation_max)

    axes[1].plot(trial, trials["target_deg"], color="#111827", linewidth=0.8, alpha=0.35)
    axes[1].scatter(trial, trials["target_deg"], s=12, color="#111827", linewidths=0)
    axes[1].set_ylabel("Target\n(deg)")
    axes[1].set_yticks([-30, -15, 0, 15, 30])
    axes[1].set_ylim(-36, 36)

    y_values = trials["feedback_type"].map(type_y).to_numpy()
    for feedback_type in type_order:
        group = trials[trials["feedback_type"] == feedback_type]
        axes[2].scatter(
            group["trial"],
            group["feedback_type"].map(type_y),
            s=20,
            color=colors[feedback_type],
            label=type_labels[feedback_type],
            linewidths=0,
        )
    axes[2].set_ylabel("Trial\ntype")
    axes[2].set_yticks(list(type_y.values()), [type_labels[name] for name in type_order])
    axes[2].set_ylim(-0.5, len(type_order) - 0.5)
    axes[2].set_xlabel("Trial")

    phase_order = ["diagnostic", "adaptation", "generalization"]
    phase_labels = {
        "familiarization": "familiarization",
        "diagnostic": "diagnostic",
        "adaptation": "adaptation",
        "generalization": "generalization",
    }
    for ax in axes:
        for phase in phase_order:
            if phase in set(trials["phase"]):
                start = int(trials.loc[trials["phase"] == phase, "trial"].min())
                ax.axvline(start - 0.5, color="#111827", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for phase, label in phase_labels.items():
        if phase in set(trials["phase"]):
            subset = trials[trials["phase"] == phase]
            center = 0.5 * (subset["trial"].min() + subset["trial"].max())
            axes[0].text(
                center,
                rotation_max - 0.06 * (rotation_max - rotation_min),
                label,
                ha="center",
                va="top",
                fontsize=9,
                color="#111827",
            )

    if title:
        fig.suptitle(title, y=0.985)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = make_hewitson_replacement_diagnostic_design()
    design.trials.to_csv(OUTPUTS / "hewitson_replacement_diagnostic_trials.csv", index=False)
    plot_design(
        design.trials,
        OUTPUTS / "hewitson_replacement_diagnostic_design.png",
        "Hewitson Replacement Familiarization and Diagnostic Design",
    )
    no_clamp = make_hewitson_replacement_no_clamp_design(high_primary_density=True)
    no_clamp.trials.to_csv(OUTPUTS / "hewitson_replacement_no_clamp_dense_trials.csv", index=False)
    plot_design(
        no_clamp.trials,
        OUTPUTS / "hewitson_replacement_no_clamp_dense_design.png",
        "No-Clamp Hewitson Replacement Diagnostic Design",
    )
    full = append_hewitson_adaptation_and_generalization(no_clamp.trials)
    full.to_csv(OUTPUTS / "full_no_clamp_replacement_hewitson_experiment_trials.csv", index=False)
    plot_design(
        full,
        OUTPUTS / "full_no_clamp_replacement_hewitson_experiment_design.png",
        "Full Experiment: No-Clamp Diagnostic Replacement Plus Hewitson Adaptation and Generalization",
    )
    shuffled = make_hewitson_replacement_no_clamp_shuffled_design()
    shuffled.trials.to_csv(OUTPUTS / "hewitson_replacement_no_clamp_shuffled_trials.csv", index=False)
    shuffled_full = append_hewitson_adaptation_and_generalization(shuffled.trials)
    shuffled_full.to_csv(OUTPUTS / "full_no_clamp_shuffled_replacement_hewitson_experiment_trials.csv", index=False)
    plot_design(
        shuffled_full,
        OUTPUTS / "full_no_clamp_shuffled_replacement_hewitson_experiment_design.png",
        "Full Experiment: Shuffled No-Clamp Diagnostic Replacement Plus Hewitson Phases",
    )
    fully_shuffled_pre = shuffle_familiarization(shuffled.trials)
    fully_shuffled = append_hewitson_adaptation_and_generalization(
        fully_shuffled_pre,
        shuffle_post_phases=True,
        adaptation_rotation=10.0,
    )
    fully_shuffled.to_csv(OUTPUTS / "full_no_clamp_fully_shuffled_replacement_hewitson_experiment_trials.csv", index=False)
    plot_design(
        fully_shuffled,
        OUTPUTS / "full_no_clamp_fully_shuffled_replacement_hewitson_experiment_design.png",
    )
    process_focused_pre = make_hewitson_replacement_process_focused_design()
    process_focused = append_hewitson_adaptation_and_generalization(
        process_focused_pre.trials,
        shuffle_post_phases=True,
        adaptation_rotation=10.0,
    )
    process_focused.to_csv(OUTPUTS / "full_process_focused_replacement_hewitson_experiment_trials.csv", index=False)
    plot_design(
        process_focused,
        OUTPUTS / "full_process_focused_replacement_hewitson_experiment_design.png",
    )
    drift_block_pre = make_hewitson_replacement_drift_block_design()
    drift_block = append_hewitson_adaptation_and_generalization(
        drift_block_pre.trials,
        shuffle_post_phases=True,
        adaptation_rotation=10.0,
    )
    drift_block.to_csv(OUTPUTS / "full_drift_block_replacement_hewitson_experiment_trials.csv", index=False)
    plot_design(
        drift_block,
        OUTPUTS / "full_drift_block_replacement_hewitson_experiment_design.png",
    )
    print(OUTPUTS / "hewitson_replacement_diagnostic_design.png")
    print(OUTPUTS / "hewitson_replacement_diagnostic_design.pdf")
    print(OUTPUTS / "hewitson_replacement_no_clamp_dense_design.png")
    print(OUTPUTS / "hewitson_replacement_no_clamp_dense_design.pdf")
    print(OUTPUTS / "full_no_clamp_replacement_hewitson_experiment_design.png")
    print(OUTPUTS / "full_no_clamp_replacement_hewitson_experiment_design.pdf")
    print(OUTPUTS / "full_no_clamp_shuffled_replacement_hewitson_experiment_design.png")
    print(OUTPUTS / "full_no_clamp_shuffled_replacement_hewitson_experiment_design.pdf")
    print(OUTPUTS / "full_no_clamp_fully_shuffled_replacement_hewitson_experiment_design.png")
    print(OUTPUTS / "full_no_clamp_fully_shuffled_replacement_hewitson_experiment_design.pdf")
    print(OUTPUTS / "full_process_focused_replacement_hewitson_experiment_design.png")
    print(OUTPUTS / "full_process_focused_replacement_hewitson_experiment_design.pdf")
    print(OUTPUTS / "full_drift_block_replacement_hewitson_experiment_design.png")
    print(OUTPUTS / "full_drift_block_replacement_hewitson_experiment_design.pdf")


if __name__ == "__main__":
    main()
