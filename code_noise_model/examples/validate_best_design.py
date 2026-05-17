"""Validate the current best design under parameter variability."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.search import generate_motif_search_designs
from noise_model.validation import validate_design_with_parameter_variability


OUTPUTS = ROOT / "outputs"
BEST_DESIGN = "search_023_cnnnr_z"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    designs = generate_motif_search_designs(n_trials=200, seed=500)
    design = next(item for item in designs if item.name == BEST_DESIGN)
    features, predictions, matrix, conservative, binary, subtype = validate_design_with_parameter_variability(
        design,
        n_reps=160,
        n_splits=30,
        seed=900,
    )
    features.to_csv(OUTPUTS / f"{BEST_DESIGN}_validation_features.csv", index=False)
    predictions.to_csv(OUTPUTS / f"{BEST_DESIGN}_validation_predictions.csv", index=False)
    matrix.to_csv(OUTPUTS / f"{BEST_DESIGN}_validation_sixway_confusion.csv")
    conservative.to_csv(OUTPUTS / f"{BEST_DESIGN}_validation_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / f"{BEST_DESIGN}_validation_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / f"{BEST_DESIGN}_validation_subtype.csv", index=False)

    print(f"Validation for {BEST_DESIGN}")
    print("\nConservative output-vs-process:")
    print(conservative.groupby("false_process_target", group_keys=False).head(5).to_string(index=False))
    print("\nRaw binary:")
    print(binary.to_string(index=False))
    print("\nSubtype:")
    print(subtype.to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
