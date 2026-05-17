"""Run motif-based search for conservative output-vs-process detection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.search import run_motif_search


OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    conservative, binary, subtype, predictions, top_designs = run_motif_search(
        n_trials=200,
        seed=500,
        n_reps=32,
        n_splits=6,
        top_n=10,
        include_hand_built=True,
    )
    conservative.to_csv(OUTPUTS / "motif_search_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / "motif_search_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / "motif_search_subtype.csv", index=False)
    predictions.to_csv(OUTPUTS / "motif_search_predictions.csv", index=False)
    for design in top_designs:
        design.trials.to_csv(OUTPUTS / f"{design.name}_trials.csv", index=False)

    print("Top conservative motif-search designs at false process <= 0.05:")
    print(
        conservative[conservative.false_process_target == 0.05]
        .head(10)
        .to_string(index=False)
    )
    print("\nCorresponding binary recovery:")
    names = conservative[conservative.false_process_target == 0.05].head(10)["design"]
    print(binary[binary.design.isin(names)].to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
