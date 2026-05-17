"""Trial-sequence construction for design-identifiability analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS_DEG = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
ROTATION_SET = np.array([-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0])


@dataclass(frozen=True)
class TrialDesign:
    """A concrete trial sequence."""

    name: str
    trials: pd.DataFrame
    description: str


def _balanced_targets(n_trials: int, rng: np.random.Generator) -> np.ndarray:
    reps = int(np.ceil(n_trials / len(TARGETS_DEG)))
    targets = np.tile(np.arange(len(TARGETS_DEG)), reps)[:n_trials]
    rng.shuffle(targets)
    return targets


def _finalize(name: str, df: pd.DataFrame, description: str) -> TrialDesign:
    df = df.copy()
    df.insert(0, "trial", np.arange(len(df)))
    df["target_deg"] = TARGETS_DEG[df["target_index"].to_numpy()]
    return TrialDesign(name=name, trials=df, description=description)


def _structured_blocks(
    name: str,
    n_trials: int,
    seed: int,
    clamp_prob: float,
    none_prob: float,
    nonzero_start: int | None,
    description: str,
) -> TrialDesign:
    rng = np.random.default_rng(seed)
    target_index = _balanced_targets(n_trials, rng)
    rows = []
    for trial in range(n_trials):
        phase = "zero_mean"
        center = 0.0
        if nonzero_start is not None and trial >= nonzero_start:
            phase = "nonzero_mean"
            center = 8.0 if (trial // 20) % 2 == 0 else -8.0

        draw = rng.random()
        if draw < none_prob:
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif draw < none_prob + clamp_prob:
            feedback_type = "clamp"
            rotation = 0.0
            clamp_error = float(rng.choice([-12, -8, -4, 4, 8, 12]))
            is_probe = True
        else:
            feedback_type = "rotated"
            rotation = float(center + rng.choice(ROTATION_SET))
            rotation = float(np.clip(rotation, -12.0, 12.0))
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
    return _finalize(name, pd.DataFrame(rows), description)


def _cyclic_probe_design(
    name: str,
    n_trials: int,
    seed: int,
    motif: tuple[str, ...],
    nonzero_centers: tuple[float, ...],
    description: str,
) -> TrialDesign:
    rng = np.random.default_rng(seed)
    target_index = _balanced_targets(n_trials, rng)
    rows = []
    clamp_values = np.array([-12.0, -8.0, -4.0, 4.0, 8.0, 12.0])
    for trial in range(n_trials):
        feedback_type = motif[trial % len(motif)]
        center = nonzero_centers[(trial // len(motif)) % len(nonzero_centers)]
        phase = "zero_mean" if np.isclose(center, 0.0) else "nonzero_mean"
        if feedback_type == "none":
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif feedback_type == "clamp":
            rotation = 0.0
            clamp_error = float(clamp_values[(trial // len(motif) + trial) % len(clamp_values)])
            is_probe = True
        else:
            magnitude = ROTATION_SET[(trial // len(motif) + trial) % len(ROTATION_SET)]
            rotation = float(np.clip(center + magnitude, -12.0, 12.0))
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
    return _finalize(name, pd.DataFrame(rows), description)


def _phase_schedule_design(
    name: str,
    n_trials: int,
    seed: int,
    centers: tuple[float, ...],
    clamp_rate: float,
    none_rate: float,
    description: str,
) -> TrialDesign:
    rng = np.random.default_rng(seed)
    target_index = _balanced_targets(n_trials, rng)
    rows = []
    block_len = max(1, n_trials // len(centers))
    for trial in range(n_trials):
        center = centers[min(trial // block_len, len(centers) - 1)]
        phase = "zero_mean" if np.isclose(center, 0.0) else "nonzero_mean"
        draw = rng.random()
        if draw < none_rate:
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif draw < none_rate + clamp_rate:
            feedback_type = "clamp"
            rotation = 0.0
            clamp_error = float(rng.choice([-12, -8, -4, 4, 8, 12]))
            is_probe = True
        else:
            feedback_type = "rotated"
            base = rng.choice([-12, -8, -4, 0, 4, 8, 12])
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
    return _finalize(name, pd.DataFrame(rows), description)


def make_candidate_designs(n_trials: int = 200, seed: int = 1) -> list[TrialDesign]:
    """Return five hand-built candidate designs under a 200-trial budget."""

    return [
        _structured_blocks(
            "balanced_mixed",
            n_trials,
            seed,
            clamp_prob=0.30,
            none_prob=0.18,
            nonzero_start=100,
            description="Mixed rotated, clamp, and no-feedback trials with a mid-task nonzero-mean phase.",
        ),
        _structured_blocks(
            "clamp_heavy",
            n_trials,
            seed + 1,
            clamp_prob=0.45,
            none_prob=0.15,
            nonzero_start=100,
            description="Clamp-rich design to strengthen matched-error comparisons across state regimes.",
        ),
        _structured_blocks(
            "no_feedback_heavy",
            n_trials,
            seed + 2,
            clamp_prob=0.25,
            none_prob=0.30,
            nonzero_start=100,
            description="No-feedback-rich design to separate carryover state noise from current visual error.",
        ),
        _structured_blocks(
            "late_nonzero",
            n_trials,
            seed + 3,
            clamp_prob=0.30,
            none_prob=0.18,
            nonzero_start=130,
            description="Longer zero-mean phase followed by a shorter nonzero-mean phase.",
        ),
        _structured_blocks(
            "early_nonzero",
            n_trials,
            seed + 4,
            clamp_prob=0.30,
            none_prob=0.18,
            nonzero_start=70,
            description="Earlier transition into nonzero-mean perturbations for stronger state displacement.",
        ),
    ]


def make_aggressive_candidate_designs(n_trials: int = 200, seed: int = 101) -> list[TrialDesign]:
    """Return a larger candidate library designed to stress model separability."""

    designs = make_candidate_designs(n_trials=n_trials, seed=seed)
    designs.extend(
        [
            _cyclic_probe_design(
                "probe_burst_clamp",
                n_trials,
                seed + 10,
                motif=("rotated", "clamp", "none", "none"),
                nonzero_centers=(0.0, 8.0, -8.0, 0.0),
                description="Repeated rotated-clamp-no-feedback motifs to emphasize update carryover.",
            ),
            _cyclic_probe_design(
                "matched_error_bursts",
                n_trials,
                seed + 11,
                motif=("clamp", "none", "rotated", "none", "clamp"),
                nonzero_centers=(0.0, 10.0, 0.0, -10.0),
                description="Clamp-heavy repeated probes across zero and nonzero state regimes.",
            ),
            _cyclic_probe_design(
                "rotation_probe_bursts",
                n_trials,
                seed + 12,
                motif=("rotated", "rotated", "none", "clamp", "none"),
                nonzero_centers=(0.0, 8.0, -8.0, 8.0, -8.0),
                description="Alternating state-push rotations followed by no-feedback and clamp probes.",
            ),
            _phase_schedule_design(
                "alternating_state_push",
                n_trials,
                seed + 13,
                centers=(8.0, -8.0, 8.0, -8.0, 0.0),
                clamp_rate=0.32,
                none_rate=0.20,
                description="Alternating nonzero-mean blocks to create large state contrasts.",
            ),
            _phase_schedule_design(
                "zero_nonzero_zero",
                n_trials,
                seed + 14,
                centers=(0.0, 10.0, 10.0, 0.0, -10.0),
                clamp_rate=0.38,
                none_rate=0.18,
                description="Zero, positive, zero, and negative phases with frequent clamps.",
            ),
            _phase_schedule_design(
                "max_probe_density",
                n_trials,
                seed + 15,
                centers=(0.0, 8.0, -8.0, 0.0),
                clamp_rate=0.50,
                none_rate=0.28,
                description="High diagnostic-probe density with fewer ordinary rotation trials.",
            ),
            _phase_schedule_design(
                "rotation_dense_state",
                n_trials,
                seed + 16,
                centers=(10.0, 10.0, -10.0, -10.0, 0.0),
                clamp_rate=0.18,
                none_rate=0.16,
                description="Rotation-dense state displacement with sparse diagnostic probes.",
            ),
            _cyclic_probe_design(
                "alternating_clamp_sign",
                n_trials,
                seed + 17,
                motif=("clamp", "none", "clamp", "none", "rotated"),
                nonzero_centers=(10.0, -10.0, 10.0, -10.0, 0.0),
                description="Frequent signed clamps embedded in alternating state regimes.",
            ),
            _cyclic_probe_design(
                "nf_burst_after_rotation",
                n_trials,
                seed + 18,
                motif=("rotated", "none", "none", "none", "clamp"),
                nonzero_centers=(0.0, 8.0, -8.0, 0.0),
                description="Three-trial no-feedback bursts after state-updating rotation trials.",
            ),
            _cyclic_probe_design(
                "nf_burst_after_clamp",
                n_trials,
                seed + 19,
                motif=("clamp", "none", "none", "none", "rotated"),
                nonzero_centers=(0.0, 8.0, -8.0, 0.0),
                description="Three-trial no-feedback bursts after matched-error clamp trials.",
            ),
            _cyclic_probe_design(
                "long_nf_bursts",
                n_trials,
                seed + 20,
                motif=("rotated", "clamp", "none", "none", "none", "none"),
                nonzero_centers=(0.0, 10.0, -10.0, 0.0),
                description="Four-trial no-feedback bursts to expose latent-state drift and persistence.",
            ),
            _cyclic_probe_design(
                "nf_burst_state_alternation",
                n_trials,
                seed + 21,
                motif=("rotated", "none", "none", "clamp", "none", "none"),
                nonzero_centers=(10.0, -10.0, 10.0, -10.0),
                description="No-feedback bursts embedded in alternating nonzero state regimes.",
            ),
        ]
    )
    return designs


def make_three_phase_nfb_design(seed: int = 1200) -> TrialDesign:
    """Return the proposed 300-trial [base, base, NFB, probe, NFB, base, base] design."""

    rng = np.random.default_rng(seed)
    n_trials = 300
    target_index = _balanced_targets(n_trials, rng)
    phases = [
        ("phase_0", 0.0, np.array([3.0, 6.0, 9.0])),
        ("phase_3", 3.0, np.array([6.0, 9.0, 12.0])),
        ("phase_6", 6.0, np.array([9.0, 12.0, 15.0])),
    ]
    motif = ("base", "base", "none", "probe", "none", "base", "base")
    rows = []
    for trial in range(n_trials):
        phase_name, base_rotation, probe_magnitudes = phases[trial // 100]
        motif_item = motif[trial % len(motif)]
        if motif_item == "none":
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif motif_item == "probe":
            feedback_type = "rotated"
            sign = float(rng.choice([-1.0, 1.0]))
            rotation = float(sign * rng.choice(probe_magnitudes))
            clamp_error = np.nan
            is_probe = False
        else:
            feedback_type = "rotated"
            rotation = float(base_rotation)
            clamp_error = np.nan
            is_probe = False
        rows.append(
            {
                "target_index": int(target_index[trial]),
                "phase": phase_name,
                "feedback_type": feedback_type,
                "rotation": rotation,
                "clamp_error": clamp_error,
                "is_probe": bool(is_probe),
            }
        )
    return _finalize(
        "three_phase_nfb_probe",
        pd.DataFrame(rows),
        "Three 100-trial phases using base/base/NFB/random-rotation/NFB/base/base motifs.",
    )


def make_three_phase_rot_clamp_design(seed: int = 1300) -> TrialDesign:
    """Return the proposed three-phase design with random rotation and clamp probes."""

    rng = np.random.default_rng(seed)
    trials_per_phase = 96
    n_trials = trials_per_phase * 3
    target_index = _balanced_targets(n_trials, rng)
    phases = [
        ("phase_0", 0.0, np.array([3.0, 6.0, 9.0])),
        ("phase_3", 3.0, np.array([6.0, 9.0, 12.0])),
        ("phase_6", 6.0, np.array([9.0, 12.0, 15.0])),
    ]
    motif = (
        "base",
        "base",
        "none",
        "rotation_probe",
        "none",
        "base",
        "base",
        "none",
        "clamp_probe",
        "none",
        "base",
        "base",
    )
    rows = []
    for trial in range(n_trials):
        phase_name, base_rotation, probe_magnitudes = phases[trial // trials_per_phase]
        motif_item = motif[trial % len(motif)]
        if motif_item == "none":
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif motif_item == "rotation_probe":
            feedback_type = "rotated"
            sign = float(rng.choice([-1.0, 1.0]))
            rotation = float(sign * rng.choice(probe_magnitudes))
            clamp_error = np.nan
            is_probe = False
        elif motif_item == "clamp_probe":
            feedback_type = "clamp"
            sign = float(rng.choice([-1.0, 1.0]))
            rotation = 0.0
            clamp_error = float(sign * rng.choice(probe_magnitudes))
            is_probe = True
        else:
            feedback_type = "rotated"
            rotation = float(base_rotation)
            clamp_error = np.nan
            is_probe = False
        rows.append(
            {
                "target_index": int(target_index[trial]),
                "phase": phase_name,
                "feedback_type": feedback_type,
                "rotation": rotation,
                "clamp_error": clamp_error,
                "is_probe": bool(is_probe),
            }
        )
    return _finalize(
        "three_phase_rot_clamp_probe",
        pd.DataFrame(rows),
        "Three equal phases using base/base/NFB/rotation/NFB/base/base/NFB/clamp/NFB/base/base motifs.",
    )


def make_gradual_ramp_nfb_design(seed: int = 1400) -> TrialDesign:
    """Return clamp/NFB/NFB/NFB/rotated design with gradual state-center ramps."""

    rng = np.random.default_rng(seed)
    n_trials = 200
    target_index = _balanced_targets(n_trials, rng)
    motif = ("clamp", "none", "none", "none", "rotated")
    centers = np.array(
        [
            0.0,
            3.33,
            6.66,
            9.99,
            6.66,
            3.33,
            0.0,
            -3.33,
            -6.66,
            -9.99,
            -6.66,
            -3.33,
            0.0,
        ]
    )
    rows = []
    for trial in range(n_trials):
        motif_item = motif[trial % len(motif)]
        cycle = trial // len(motif)
        center = float(centers[cycle % len(centers)])
        phase = "zero_mean" if np.isclose(center, 0.0) else "nonzero_mean"
        if motif_item == "none":
            feedback_type = "none"
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif motif_item == "clamp":
            feedback_type = "clamp"
            rotation = 0.0
            clamp_error = -12.0
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
    return _finalize(
        "gradual_ramp_nfb_probe",
        pd.DataFrame(rows),
        "Clamp/NFB/NFB/NFB/rotated motif with gradual ramps through 0 and +/-9.99 deg.",
    )


def make_hold_clamp_ladder_design(seed: int = 1600) -> TrialDesign:
    """Return a smooth ramp/hold design with clamp ladders for subtype resolution."""

    rng = np.random.default_rng(seed)
    rows = []
    clamp_ladder = [3.0, -3.0, 6.0, -6.0, 9.0, -9.0, 12.0, -12.0]
    schedule: list[tuple[str, float, str, float | None]] = []

    for _ in range(20):
        schedule.append(("baseline", 0.0, "rotated", 0.0))
        schedule.append(("baseline", 0.0, "none", None))

    def add_ramp(name: str, centers: list[float]) -> None:
        for center in centers:
            schedule.append((name, center, "rotated", center))
            schedule.append((name, center, "none", None))
            schedule.append((name, center, "none", None))

    def add_hold(name: str, center: float, reps: int) -> None:
        for rep in range(reps):
            for clamp in clamp_ladder:
                schedule.append((name, center, "clamp", clamp))
                schedule.append((name, center, "none", None))
                schedule.append((name, center, "none", None))
                if rep % 2 == 0:
                    schedule.append((name, center, "rotated", center))

    add_ramp("ramp_pos", [3.0, 6.0, 9.0])
    add_hold("hold_pos", 9.0, reps=3)
    add_ramp("ramp_zero_from_pos", [6.0, 3.0, 0.0])
    add_hold("hold_zero", 0.0, reps=3)
    add_ramp("ramp_neg", [-3.0, -6.0, -9.0])
    add_hold("hold_neg", -9.0, reps=3)
    add_ramp("ramp_zero_from_neg", [-6.0, -3.0, 0.0])

    n_trials = len(schedule)
    target_index = _balanced_targets(n_trials, rng)
    for trial, (phase, center, feedback_type, value) in enumerate(schedule):
        if feedback_type == "none":
            rotation = 0.0
            clamp_error = np.nan
            is_probe = True
        elif feedback_type == "clamp":
            rotation = 0.0
            clamp_error = float(value)
            is_probe = True
        else:
            rotation = float(value)
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

    return _finalize(
        "hold_clamp_ladder",
        pd.DataFrame(rows),
        "Smooth ramp/hold design with repeated clamp ladders at positive, zero, and negative states.",
    )


def make_hewitson_replacement_diagnostic_design() -> TrialDesign:
    """Replacement familiarization/baseline block for noise and generalization diagnostics."""

    rows = []
    target_lookup = {-30.0: 0, -15.0: 1, 0.0: 2, 15.0: 3, 30.0: 4}
    target_sweep = [-30.0, -15.0, 0.0, 15.0, 30.0]
    for _ in range(6):
        for target in target_sweep:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "familiarization",
                    "feedback_type": "rotated",
                    "rotation": 0.0,
                    "clamp_error": np.nan,
                    "is_probe": False,
                }
            )

    perturbations = [
        3.0,
        -6.0,
        9.0,
        6.0,
        -3.0,
        -9.0,
        -6.0,
        3.0,
        9.0,
        -9.0,
        -3.0,
        6.0,
        9.0,
        -6.0,
        3.0,
        -3.0,
        6.0,
        -9.0,
        -6.0,
        9.0,
    ]
    probe_patterns = [
        [-15.0, 15.0, -30.0, 30.0],
        [15.0, -15.0, 30.0, -30.0],
        [-30.0, 30.0, -15.0, 15.0],
        [30.0, -30.0, 15.0, -15.0],
    ]
    for cycle, perturbation in enumerate(perturbations):
        probes = probe_patterns[cycle % len(probe_patterns)]
        motif = [
            (0.0, "rotated", 0.0, np.nan, False),
            (0.0, "none", 0.0, np.nan, True),
            (probes[0], "none", 0.0, np.nan, True),
            (0.0, "clamp", 0.0, perturbation, True),
            (0.0, "none", 0.0, np.nan, True),
            (probes[1], "none", 0.0, np.nan, True),
            (0.0, "rotated", perturbation, np.nan, False),
            (0.0, "none", 0.0, np.nan, True),
            (probes[2], "none", 0.0, np.nan, True),
            (probes[3], "none", 0.0, np.nan, True),
        ]
        for target, feedback_type, rotation, clamp_error, is_probe in motif:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "diagnostic",
                    "feedback_type": feedback_type,
                    "rotation": rotation,
                    "clamp_error": clamp_error,
                    "is_probe": is_probe,
                }
            )

    return _finalize(
        "hewitson_replacement_diagnostic",
        pd.DataFrame(rows),
        "Five-target familiarization plus zero-mean diagnostic block for process/output noise and local generalization.",
    )


def make_hewitson_replacement_no_clamp_design(high_primary_density: bool = True) -> TrialDesign:
    """No-clamp replacement block for process/output noise and local generalization."""

    rows = []
    target_lookup = {-30.0: 0, -15.0: 1, 0.0: 2, 15.0: 3, 30.0: 4}
    target_sweep = [-30.0, -15.0, 0.0, 15.0, 30.0]
    for _ in range(6):
        for target in target_sweep:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "familiarization",
                    "feedback_type": "rotated",
                    "rotation": 0.0,
                    "clamp_error": np.nan,
                    "is_probe": False,
                }
            )

    perturbations = [
        3.0,
        -6.0,
        9.0,
        6.0,
        -3.0,
        -9.0,
        -6.0,
        3.0,
        9.0,
        -9.0,
        -3.0,
        6.0,
        9.0,
        -6.0,
        3.0,
        -3.0,
        6.0,
        -9.0,
        -6.0,
        9.0,
    ]
    probe_patterns = [
        [-15.0, 15.0, -30.0, 30.0],
        [15.0, -15.0, 30.0, -30.0],
        [-30.0, 30.0, -15.0, 15.0],
        [30.0, -30.0, 15.0, -15.0],
    ]
    for cycle, perturbation in enumerate(perturbations):
        probes = probe_patterns[cycle % len(probe_patterns)]
        if high_primary_density:
            motif = [
                (0.0, "rotated", 0.0, np.nan, False),
                (0.0, "none", 0.0, np.nan, True),
                (probes[0], "none", 0.0, np.nan, True),
                (0.0, "rotated", perturbation, np.nan, False),
                (0.0, "none", 0.0, np.nan, True),
                (0.0, "rotated", perturbation, np.nan, False),
                (0.0, "none", 0.0, np.nan, True),
                (probes[1], "none", 0.0, np.nan, True),
                (0.0, "rotated", 0.0, np.nan, False),
                (0.0, "none", 0.0, np.nan, True),
            ]
        else:
            motif = [
                (0.0, "rotated", 0.0, np.nan, False),
                (0.0, "none", 0.0, np.nan, True),
                (probes[0], "none", 0.0, np.nan, True),
                (0.0, "rotated", perturbation, np.nan, False),
                (0.0, "none", 0.0, np.nan, True),
                (probes[1], "none", 0.0, np.nan, True),
                (0.0, "rotated", perturbation, np.nan, False),
                (0.0, "none", 0.0, np.nan, True),
                (probes[2], "none", 0.0, np.nan, True),
                (probes[3], "none", 0.0, np.nan, True),
            ]
        for target, feedback_type, rotation, clamp_error, is_probe in motif:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "diagnostic",
                    "feedback_type": feedback_type,
                    "rotation": rotation,
                    "clamp_error": clamp_error,
                    "is_probe": is_probe,
                }
            )

    name = "hewitson_replacement_no_clamp_dense" if high_primary_density else "hewitson_replacement_no_clamp_matched"
    return _finalize(
        name,
        pd.DataFrame(rows),
        "Five-target no-clamp diagnostic block with zero-mean rotated feedback and no-feedback readouts.",
    )


def make_hewitson_replacement_no_clamp_shuffled_design(seed: int = 20260423) -> TrialDesign:
    """Shuffled no-clamp diagnostic block with preserved trial counts and constraints."""

    rng = np.random.default_rng(seed)
    rows = []
    target_lookup = {-30.0: 0, -15.0: 1, 0.0: 2, 15.0: 3, 30.0: 4}
    target_sweep = [-30.0, -15.0, 0.0, 15.0, 30.0]
    for _ in range(6):
        for target in target_sweep:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "familiarization",
                    "feedback_type": "rotated",
                    "rotation": 0.0,
                    "clamp_error": np.nan,
                    "is_probe": False,
                }
            )

    perturbations = [
        3.0,
        -6.0,
        9.0,
        6.0,
        -3.0,
        -9.0,
        -6.0,
        3.0,
        9.0,
        -9.0,
        -3.0,
        6.0,
        9.0,
        -6.0,
        3.0,
        -3.0,
        6.0,
        -9.0,
        -6.0,
        9.0,
    ]
    probe_targets = [-30.0, -15.0, 15.0, 30.0] * 10
    rng.shuffle(probe_targets)

    previous_types: list[str] = []

    def acceptable(block: list[tuple[float, str, float, float, bool]]) -> bool:
        labels = ["probe" if target != 0.0 else feedback_type for target, feedback_type, _, _, _ in block]
        labels = previous_types[-2:] + labels
        run = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                run += 1
                if run > 3:
                    return False
            else:
                run = 1
        return True

    for cycle, perturbation in enumerate(perturbations):
        probes = [probe_targets.pop(), probe_targets.pop()]
        base_block = [
            (0.0, "rotated", 0.0, np.nan, False),
            (0.0, "rotated", 0.0, np.nan, False),
            (0.0, "rotated", perturbation, np.nan, False),
            (0.0, "rotated", perturbation, np.nan, False),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (probes[0], "none", 0.0, np.nan, True),
            (probes[1], "none", 0.0, np.nan, True),
        ]
        for _ in range(200):
            candidate = list(base_block)
            rng.shuffle(candidate)
            if acceptable(candidate):
                break
        else:
            candidate = base_block

        previous_types.extend("probe" if target != 0.0 else feedback_type for target, feedback_type, _, _, _ in candidate)
        for target, feedback_type, rotation, clamp_error, is_probe in candidate:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "diagnostic",
                    "feedback_type": feedback_type,
                    "rotation": rotation,
                    "clamp_error": clamp_error,
                    "is_probe": is_probe,
                }
            )

    return _finalize(
        "hewitson_replacement_no_clamp_shuffled",
        pd.DataFrame(rows),
        "Shuffled five-target no-clamp diagnostic block with zero-mean rotated feedback and no-feedback readouts.",
    )


def make_hewitson_replacement_process_focused_design(seed: int = 20260425) -> TrialDesign:
    """Longer no-clamp diagnostic block with more no-feedback bursts for process_sd recovery."""

    rng = np.random.default_rng(seed)
    rows = []
    target_lookup = {-30.0: 0, -15.0: 1, 0.0: 2, 15.0: 3, 30.0: 4}
    target_sweep = [-30.0, -15.0, 0.0, 15.0, 30.0]
    for _ in range(6):
        familiar_targets = list(target_sweep)
        rng.shuffle(familiar_targets)
        for target in familiar_targets:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "familiarization",
                    "feedback_type": "rotated",
                    "rotation": 0.0,
                    "clamp_error": np.nan,
                    "is_probe": False,
                }
            )

    perturbations = [
        3.0,
        -6.0,
        9.0,
        6.0,
        -3.0,
        -9.0,
        -6.0,
        3.0,
        9.0,
        -9.0,
        -3.0,
        6.0,
        9.0,
        -6.0,
        3.0,
        -3.0,
        6.0,
        -9.0,
        -6.0,
        9.0,
        3.0,
        -6.0,
        6.0,
        -3.0,
    ]
    probe_targets = [-30.0, -15.0, 15.0, 30.0] * 6
    rng.shuffle(probe_targets)

    previous_types: list[str] = []

    def acceptable(block: list[tuple[float, str, float, float, bool]]) -> bool:
        labels = ["probe" if target != 0.0 else feedback_type for target, feedback_type, _, _, _ in block]
        labels = previous_types[-3:] + labels
        run = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                run += 1
                if labels[i] == "none" and run > 4:
                    return False
                if labels[i] != "none" and run > 3:
                    return False
            else:
                run = 1
        return True

    for cycle, perturbation in enumerate(perturbations):
        probe = probe_targets.pop()
        base_block = [
            (0.0, "rotated", 0.0, np.nan, False),
            (0.0, "rotated", perturbation, np.nan, False),
            (0.0, "rotated", perturbation, np.nan, False),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (probe, "none", 0.0, np.nan, True),
        ]
        for _ in range(300):
            candidate = list(base_block)
            rng.shuffle(candidate)
            if acceptable(candidate):
                break
        else:
            candidate = base_block

        previous_types.extend("probe" if target != 0.0 else feedback_type for target, feedback_type, _, _, _ in candidate)
        for target, feedback_type, rotation, clamp_error, is_probe in candidate:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "diagnostic",
                    "feedback_type": feedback_type,
                    "rotation": rotation,
                    "clamp_error": clamp_error,
                    "is_probe": is_probe,
                }
            )

    return _finalize(
        "hewitson_replacement_process_focused",
        pd.DataFrame(rows),
        "Longer no-clamp diagnostic block with denser primary-target no-feedback bursts for process_sd recovery.",
    )


def make_hewitson_replacement_drift_block_design(seed: int = 20260426) -> TrialDesign:
    """No-clamp diagnostic with dedicated zero-rotation drift blocks for process_sd."""

    rng = np.random.default_rng(seed)
    rows = []
    target_lookup = {-30.0: 0, -15.0: 1, 0.0: 2, 15.0: 3, 30.0: 4}
    target_sweep = [-30.0, -15.0, 0.0, 15.0, 30.0]
    for _ in range(6):
        familiar_targets = list(target_sweep)
        rng.shuffle(familiar_targets)
        for target in familiar_targets:
            rows.append(
                {
                    "target_index": target_lookup[target],
                    "phase": "familiarization",
                    "feedback_type": "rotated",
                    "rotation": 0.0,
                    "clamp_error": np.nan,
                    "is_probe": False,
                }
            )

    perturbations = [
        3.0,
        -6.0,
        9.0,
        6.0,
        -3.0,
        -9.0,
        -6.0,
        3.0,
        9.0,
        -9.0,
        -3.0,
        6.0,
        9.0,
        -6.0,
        3.0,
        -3.0,
        6.0,
        -9.0,
        -6.0,
        9.0,
    ]
    probe_targets = [-30.0, -15.0, 15.0, 30.0] * 6
    rng.shuffle(probe_targets)

    schedule: list[tuple[float, str, float, float, bool]] = []
    for block_i, perturbation in enumerate(perturbations):
        probe = probe_targets.pop()
        block = [
            (0.0, "rotated", 0.0, np.nan, False),
            (0.0, "rotated", perturbation, np.nan, False),
            (0.0, "rotated", perturbation, np.nan, False),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (0.0, "none", 0.0, np.nan, True),
            (probe, "none", 0.0, np.nan, True),
        ]
        rng.shuffle(block)
        schedule.extend(block)
        if block_i in {1, 3, 6, 8, 11, 13, 16, 18}:
            schedule.extend(
                [
                    (0.0, "rotated", 0.0, np.nan, False),
                    (0.0, "none", 0.0, np.nan, True),
                    (0.0, "none", 0.0, np.nan, True),
                    (0.0, "none", 0.0, np.nan, True),
                    (0.0, "none", 0.0, np.nan, True),
                    (0.0, "rotated", 0.0, np.nan, False),
                ]
            )

    for target, feedback_type, rotation, clamp_error, is_probe in schedule:
        rows.append(
            {
                "target_index": target_lookup[target],
                "phase": "diagnostic",
                "feedback_type": feedback_type,
                "rotation": rotation,
                "clamp_error": clamp_error,
                "is_probe": is_probe,
            }
        )

    return _finalize(
        "hewitson_replacement_drift_block",
        pd.DataFrame(rows),
        "No-clamp diagnostic block with zero-rotation no-feedback drift blocks for process_sd recovery.",
    )


def save_designs(designs: list[TrialDesign], output_dir: str) -> None:
    """Write each design table as CSV."""

    for design in designs:
        design.trials.to_csv(f"{output_dir}/{design.name}_trials.csv", index=False)


def make_single_target(design: TrialDesign, target_deg: float = 0.0) -> TrialDesign:
    """Return a copy of a design with every trial assigned to one modeled target."""

    df = design.trials.copy()
    df["target_index"] = 2
    df["target_deg"] = target_deg
    return TrialDesign(
        name=f"single_{design.name}",
        trials=df,
        description=f"Single-target version of {design.name}.",
    )
