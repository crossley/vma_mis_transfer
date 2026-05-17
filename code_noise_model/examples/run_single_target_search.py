"""Search single-target schedules for output-vs-process identifiability."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.classifier_recovery import evaluate_designs_classifier
from noise_model.design import make_single_target
from noise_model.diagnostics import conservative_threshold_table, output_vs_process_table, process_subtype_table
from noise_model.search import generate_hybrid_smooth_designs, generate_smooth_search_designs
from noise_model.validation import variable_parameter_sampler


OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    base_designs = []
    base_designs.extend(generate_smooth_search_designs(n_trials=200, seed=1500))
    hybrids = generate_hybrid_smooth_designs(n_trials=240, seed=3500)
    base_designs.extend(
        design
        for design in hybrids
        if int(design.name.split("_")[1]) < 144
        or int(design.name.split("_")[1]) in range(288, 360)
    )
    designs = [make_single_target(design) for design in base_designs]
    _, predictions, _, _ = evaluate_designs_classifier(
        designs,
        n_reps=28,
        n_splits=6,
        seed=4500,
        parameter_sampler=variable_parameter_sampler,
    )
    conservative = conservative_threshold_table(predictions)
    binary = output_vs_process_table(predictions)
    subtype = process_subtype_table(predictions)
    conservative.to_csv(OUTPUTS / "single_target_search_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / "single_target_search_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / "single_target_search_subtype.csv", index=False)
    predictions.to_csv(OUTPUTS / "single_target_search_predictions.csv", index=False)

    top = conservative[conservative.false_process_target == 0.05].head(12)
    keep = set(top.design)
    for design in designs:
        if design.name in keep:
            design.trials.to_csv(OUTPUTS / f"{design.name}_trials.csv", index=False)

    print("Top single-target designs at false process <= 0.05:")
    print(top.to_string(index=False))
    print("\nRaw binary for top designs:")
    print(binary[binary.design.isin(top.design)].to_string(index=False))
    print("\nSubtype for top designs:")
    print(subtype[subtype.design.isin(top.design)].to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
