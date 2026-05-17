"""Run motif search with participant-level parameter variability."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.search import run_motif_search
from noise_model.validation import variable_parameter_sampler


OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    conservative, binary, subtype, predictions, top_designs = run_motif_search(
        n_trials=200,
        seed=700,
        n_reps=24,
        n_splits=5,
        top_n=10,
        include_hand_built=True,
        parameter_sampler=variable_parameter_sampler,
    )
    conservative.to_csv(OUTPUTS / "robust_motif_search_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / "robust_motif_search_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / "robust_motif_search_subtype.csv", index=False)
    predictions.to_csv(OUTPUTS / "robust_motif_search_predictions.csv", index=False)
    for design in top_designs:
        design.trials.to_csv(OUTPUTS / f"{design.name}_robust_trials.csv", index=False)

    top = conservative[conservative.false_process_target == 0.05].head(10)
    print("Top robust conservative designs at false process <= 0.05:")
    print(top.to_string(index=False))
    print("\nRaw binary for those designs:")
    print(binary[binary.design.isin(top.design)].to_string(index=False))
    print("\nSubtype for those designs:")
    print(subtype[subtype.design.isin(top.design)].to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
