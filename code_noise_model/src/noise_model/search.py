"""Motif-based trial-design search."""

from __future__ import annotations

import itertools

import pandas as pd

from .classifier_recovery import evaluate_designs_classifier
from .design import TrialDesign, _cyclic_probe_design, _finalize, make_aggressive_candidate_designs
from .diagnostics import conservative_threshold_table, output_vs_process_table, process_subtype_table


def generate_motif_search_designs(n_trials: int = 200, seed: int = 500) -> list[TrialDesign]:
    """Generate a constrained motif library for output-vs-process detection."""

    motifs = [
        ("clamp", "none", "none", "rotated"),
        ("clamp", "none", "none", "none", "rotated"),
        ("clamp", "none", "none", "none", "none", "rotated"),
        ("rotated", "none", "none", "clamp"),
        ("rotated", "none", "none", "none", "clamp"),
        ("rotated", "clamp", "none", "none", "none"),
        ("clamp", "rotated", "none", "none", "none"),
        ("clamp", "none", "clamp", "none", "rotated"),
        ("rotated", "none", "clamp", "none", "none"),
        ("clamp", "none", "none", "clamp", "none", "rotated"),
    ]
    center_schedules = [
        (0.0, 8.0, -8.0, 0.0),
        (10.0, -10.0, 10.0, -10.0),
        (0.0, 10.0, 10.0, 0.0, -10.0),
        (8.0, 8.0, -8.0, -8.0, 0.0),
        (0.0, 0.0, 10.0, -10.0),
        (0.0, 5.0, -5.0, 0.0),
        (5.0, -5.0, 5.0, -5.0),
        (0.0, 5.0, 10.0, 0.0, -5.0, -10.0),
        (0.0, 10.0, 5.0, 0.0, -10.0, -5.0),
        (5.0, 10.0, -5.0, -10.0, 0.0),
        (10.0, 5.0, -10.0, -5.0, 0.0),
        (0.0, 3.33, 6.66, 9.99, 0.0, -3.33, -6.66, -9.99),
        (3.33, -3.33, 6.66, -6.66, 9.99, -9.99, 0.0),
        (0.0, 9.99, 6.66, 3.33, 0.0, -9.99, -6.66, -3.33),
        (9.99, -9.99, 6.66, -6.66, 3.33, -3.33, 0.0),
    ]
    designs = []
    for i, (motif, centers) in enumerate(itertools.product(motifs, center_schedules)):
        motif_name = "".join(item[0] for item in motif)
        center_name = "z" if centers[0] == 0 else "s"
        designs.append(
            _cyclic_probe_design(
                name=f"search_{i:03d}_{motif_name}_{center_name}",
                n_trials=n_trials,
                seed=seed + i,
                motif=motif,
                nonzero_centers=centers,
                description=f"Search motif {motif} with centers {centers}.",
            )
        )
    return designs


def _smooth_center_sequence(level: float, hold: int) -> tuple[float, ...]:
    up = [0.0, 3.33, 6.66, level]
    down = [6.66, 3.33, 0.0]
    neg = [-3.33, -6.66, -level]
    back = [-6.66, -3.33, 0.0]
    return tuple(up + [level] * hold + down + neg + [-level] * hold + back)


def _smooth_design(
    name: str,
    n_trials: int,
    seed: int,
    motif: tuple[str, ...],
    centers: tuple[float, ...],
    clamp_mode: str,
) -> TrialDesign:
    import numpy as np
    import pandas as pd

    from .design import _balanced_targets

    rng = np.random.default_rng(seed)
    target_index = _balanced_targets(n_trials, rng)
    rows = []
    for trial in range(n_trials):
        item = motif[trial % len(motif)]
        cycle = trial // len(motif)
        center = centers[cycle % len(centers)]
        phase = "zero_mean" if np.isclose(center, 0.0) else "nonzero_mean"
        if item == "none":
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif item == "clamp":
            feedback_type = "clamp"
            rotation = 0.0
            if clamp_mode == "fixed_neg":
                clamp_error = -12.0
            elif clamp_mode == "fixed_signed":
                clamp_error = 12.0 if center >= 0 else -12.0
            else:
                clamp_error = float(rng.choice([-12.0, -8.0, -4.0, 4.0, 8.0, 12.0]))
            is_probe = True
        else:
            feedback_type = "rotated"
            base = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0][cycle % 7]
            rotation = float(np.clip(center + base, -12.0, 12.0))
            clamp_error = np.nan
            is_probe = False
        rows.append(
            {
                "target_index": int(target_index[trial]),
                "phase": phase,
                "feedback_type": feedback_type,
                "rotation": rotation,
                "clamp_error": clamp_error,
                "is_probe": bool(is_probe),
            }
        )
    return _finalize(name, pd.DataFrame(rows), f"Smooth constrained design {name}.")


def generate_smooth_search_designs(n_trials: int = 200, seed: int = 1500) -> list[TrialDesign]:
    """Generate smooth designs with bounded center steps and no abrupt sign flips."""

    motifs = [
        ("clamp", "none", "none", "none", "rotated"),
        ("clamp", "none", "none", "none", "none", "rotated"),
        ("clamp", "none", "none", "rotated", "none"),
        ("clamp", "none", "rotated", "none", "none"),
        ("rotated", "clamp", "none", "none", "none"),
    ]
    center_sequences = []
    for level in [9.99, 12.0]:
        for hold in [0, 2, 4, 6]:
            center_sequences.append(_smooth_center_sequence(level, hold))
    clamp_modes = ["fixed_neg", "fixed_signed", "varied"]
    designs = []
    index = 0
    for motif, centers, clamp_mode in itertools.product(motifs, center_sequences, clamp_modes):
        motif_name = "".join(item[0] for item in motif)
        designs.append(
            _smooth_design(
                name=f"smooth_{index:03d}_{motif_name}_{clamp_mode}",
                n_trials=n_trials,
                seed=seed + index,
                motif=motif,
                centers=centers,
                clamp_mode=clamp_mode,
            )
        )
        index += 1
    return designs


def _targeted_smooth_design(
    name: str,
    n_trials: int,
    seed: int,
    motif: tuple[str, ...],
    centers: tuple[float, ...],
    clamp_values: tuple[float, ...],
) -> TrialDesign:
    import numpy as np
    import pandas as pd

    from .design import _balanced_targets

    rng = np.random.default_rng(seed)
    target_index = _balanced_targets(n_trials, rng)
    rows = []
    clamp_counter = 0
    for trial in range(n_trials):
        item = motif[trial % len(motif)]
        cycle = trial // len(motif)
        center = centers[cycle % len(centers)]
        phase = "zero_mean" if np.isclose(center, 0.0) else "nonzero_mean"
        if item == "none":
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif item == "clamp":
            feedback_type = "clamp"
            rotation = 0.0
            clamp_error = clamp_values[clamp_counter % len(clamp_values)]
            clamp_counter += 1
            is_probe = True
        else:
            feedback_type = "rotated"
            # Keep rotations close to the center so state changes smoothly.
            offset = [-4.0, 0.0, 4.0, 0.0][cycle % 4]
            rotation = float(np.clip(center + offset, -12.0, 12.0))
            clamp_error = np.nan
            is_probe = False
        rows.append(
            {
                "target_index": int(target_index[trial]),
                "phase": phase,
                "feedback_type": feedback_type,
                "rotation": rotation,
                "clamp_error": clamp_error,
                "is_probe": bool(is_probe),
            }
        )
    return _finalize(name, pd.DataFrame(rows), f"Targeted smooth subtype design {name}.")


def generate_targeted_smooth_designs(n_trials: int = 240, seed: int = 2500) -> list[TrialDesign]:
    """Generate smooth designs aimed at process-subtype resolution."""

    center_sequences = [
        _smooth_center_sequence(9.99, hold=4),
        _smooth_center_sequence(12.0, hold=4),
        _smooth_center_sequence(9.99, hold=8),
        _smooth_center_sequence(12.0, hold=8),
    ]
    motifs = [
        ("clamp", "none", "none", "none", "rotated", "none"),
        ("clamp", "none", "none", "none", "none", "rotated"),
        ("clamp", "none", "clamp", "none", "none", "rotated"),
        ("rotated", "none", "clamp", "none", "none", "none"),
        ("clamp", "none", "none", "rotated", "none", "none"),
    ]
    clamp_sequences = [
        (-12.0, 12.0),
        (-8.0, 8.0),
        (-4.0, 4.0, -8.0, 8.0, -12.0, 12.0),
        (4.0, 8.0, 12.0, -4.0, -8.0, -12.0),
    ]
    designs = []
    index = 0
    for motif, centers, clamps in itertools.product(motifs, center_sequences, clamp_sequences):
        motif_name = "".join(item[0] for item in motif)
        designs.append(
            _targeted_smooth_design(
                name=f"targeted_{index:03d}_{motif_name}",
                n_trials=n_trials,
                seed=seed + index,
                motif=motif,
                centers=centers,
                clamp_values=clamps,
            )
        )
        index += 1
    return designs


def _hybrid_design(
    name: str,
    n_trials: int,
    seed: int,
    baseline_trials: int,
    motif: tuple[str, ...],
    centers: tuple[float, ...],
    clamp_values: tuple[float, ...],
    catch_every: int | None,
) -> TrialDesign:
    import numpy as np
    import pandas as pd

    from .design import _balanced_targets

    rng = np.random.default_rng(seed)
    target_index = _balanced_targets(n_trials, rng)
    rows = []
    clamp_counter = 0
    main_trial = 0
    for trial in range(n_trials):
        if trial < baseline_trials:
            item = "none" if trial % 2 else "rotated"
            center = 0.0
            phase = "baseline"
        else:
            cycle = main_trial // len(motif)
            center = centers[cycle % len(centers)]
            phase = "zero_mean" if np.isclose(center, 0.0) else "nonzero_mean"
            if catch_every is not None and cycle > 0 and cycle % catch_every == 0:
                item = "none"
            else:
                item = motif[main_trial % len(motif)]
            main_trial += 1

        if item == "none":
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif item == "clamp":
            feedback_type = "clamp"
            rotation = 0.0
            clamp_error = clamp_values[clamp_counter % len(clamp_values)]
            clamp_counter += 1
            is_probe = True
        else:
            feedback_type = "rotated"
            if phase == "baseline":
                rotation = 0.0
            else:
                offset = [-4.0, 0.0, 4.0, 0.0][(main_trial // len(motif)) % 4]
                rotation = float(np.clip(center + offset, -12.0, 12.0))
            clamp_error = np.nan
            is_probe = False

        rows.append(
            {
                "target_index": int(target_index[trial]),
                "phase": phase,
                "feedback_type": feedback_type,
                "rotation": rotation,
                "clamp_error": clamp_error,
                "is_probe": bool(is_probe),
            }
        )
    return _finalize(name, pd.DataFrame(rows), f"Hybrid smooth candidate {name}.")


def generate_hybrid_smooth_designs(n_trials: int = 240, seed: int = 3500) -> list[TrialDesign]:
    """Generate hybrid designs combining smooth_115-style motifs and subtype probes."""

    centers = [
        _smooth_center_sequence(9.99, hold=4),
        _smooth_center_sequence(12.0, hold=4),
        _smooth_center_sequence(9.99, hold=8),
        _smooth_center_sequence(12.0, hold=8),
    ]
    motifs = [
        ("rotated", "clamp", "none", "none", "none"),
        ("rotated", "clamp", "none", "none", "none", "none"),
        ("rotated", "clamp", "none", "clamp", "none"),
        ("rotated", "clamp", "none", "none", "clamp", "none"),
        ("rotated", "clamp", "none", "none", "none", "none", "none"),
    ]
    clamp_values = [
        (12.0, -12.0),
        (4.0, 8.0, 12.0, -4.0, -8.0, -12.0),
        (8.0, -8.0, 12.0, -12.0),
    ]
    baseline_options = [0, 20, 30]
    catch_options = [None, 5, 8]
    designs = []
    index = 0
    for baseline, motif, center_seq, clamps, catch_every in itertools.product(
        baseline_options,
        motifs,
        centers,
        clamp_values,
        catch_options,
    ):
        designs.append(
            _hybrid_design(
                name=f"hybrid_{index:03d}",
                n_trials=n_trials,
                seed=seed + index,
                baseline_trials=baseline,
                motif=motif,
                centers=center_seq,
                clamp_values=clamps,
                catch_every=catch_every,
            )
        )
        index += 1
    return designs


def run_motif_search(
    n_trials: int = 200,
    seed: int = 500,
    n_reps: int = 45,
    n_splits: int = 8,
    top_n: int = 10,
    include_hand_built: bool = True,
    parameter_sampler=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[TrialDesign]]:
    """Evaluate motif-search candidates and return ranked summaries."""

    designs = generate_motif_search_designs(n_trials=n_trials, seed=seed)
    if include_hand_built:
        designs = make_aggressive_candidate_designs(n_trials=n_trials, seed=1) + designs
    _, predictions, _, _ = evaluate_designs_classifier(
        designs,
        n_reps=n_reps,
        n_splits=n_splits,
        seed=seed,
        parameter_sampler=parameter_sampler,
    )
    conservative = conservative_threshold_table(predictions)
    binary = output_vs_process_table(predictions)
    subtype = process_subtype_table(predictions)
    top_design_names = (
        conservative[conservative.false_process_target == 0.05]
        .head(top_n)["design"]
        .tolist()
    )
    top_designs = [design for design in designs if design.name in top_design_names]
    return conservative, binary, subtype, predictions, top_designs
