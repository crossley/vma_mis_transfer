"""Run robust search over smooth-constrained designs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.classifier_recovery import evaluate_designs_classifier
from noise_model.diagnostics import conservative_threshold_table, output_vs_process_table, process_subtype_table
from noise_model.search import generate_smooth_search_designs
from noise_model.validation import variable_parameter_sampler


OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    designs = generate_smooth_search_designs(n_trials=200, seed=1500)
    _, predictions, _, _ = evaluate_designs_classifier(
        designs,
        n_reps=28,
        n_splits=6,
        seed=1500,
        parameter_sampler=variable_parameter_sampler,
    )
    conservative = conservative_threshold_table(predictions)
    binary = output_vs_process_table(predictions)
    subtype = process_subtype_table(predictions)
    conservative.to_csv(OUTPUTS / "smooth_robust_search_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / "smooth_robust_search_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / "smooth_robust_search_subtype.csv", index=False)
    predictions.to_csv(OUTPUTS / "smooth_robust_search_predictions.csv", index=False)

    top = conservative[conservative.false_process_target == 0.05].head(10)
    for design in designs:
        if design.name in set(top.design):
            design.trials.to_csv(OUTPUTS / f"{design.name}_trials.csv", index=False)

    print("Top smooth robust designs at false process <= 0.05:")
    print(top.to_string(index=False))
    print("\nRaw binary for those designs:")
    print(binary[binary.design.isin(top.design)].to_string(index=False))
    print("\nSubtype for those designs:")
    print(subtype[subtype.design.isin(top.design)].to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
