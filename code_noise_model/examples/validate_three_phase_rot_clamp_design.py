"""Validate the proposed three-phase rotation-plus-clamp probe design."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.design import make_three_phase_rot_clamp_design
from noise_model.validation import validate_design_with_parameter_variability


OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = make_three_phase_rot_clamp_design(seed=1300)
    design.trials.to_csv(OUTPUTS / "three_phase_rot_clamp_probe_trials.csv", index=False)
    features, predictions, matrix, conservative, binary, subtype = validate_design_with_parameter_variability(
        design,
        n_reps=160,
        n_splits=30,
        seed=1301,
    )
    features.to_csv(OUTPUTS / "three_phase_rot_clamp_probe_validation_features.csv", index=False)
    predictions.to_csv(OUTPUTS / "three_phase_rot_clamp_probe_validation_predictions.csv", index=False)
    matrix.to_csv(OUTPUTS / "three_phase_rot_clamp_probe_validation_sixway_confusion.csv")
    conservative.to_csv(OUTPUTS / "three_phase_rot_clamp_probe_validation_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / "three_phase_rot_clamp_probe_validation_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / "three_phase_rot_clamp_probe_validation_subtype.csv", index=False)

    print("Three-phase rotation-plus-clamp probe design")
    print(design.trials["feedback_type"].value_counts().to_string())
    print(design.trials["phase"].value_counts().sort_index().to_string())
    print("\nConservative output-vs-process:")
    print(conservative.to_string(index=False))
    print("\nRaw binary:")
    print(binary.to_string(index=False))
    print("\nSubtype:")
    print(subtype.to_string(index=False))


if __name__ == "__main__":
    main()
