"""Run robust search over hybrid smooth candidate designs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.classifier_recovery import evaluate_designs_classifier
from noise_model.diagnostics import conservative_threshold_table, output_vs_process_table, process_subtype_table
from noise_model.search import generate_hybrid_smooth_designs
from noise_model.validation import variable_parameter_sampler


OUTPUTS = ROOT / "outputs"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    designs = generate_hybrid_smooth_designs(n_trials=240, seed=3500)
    # Keep a focused, tractable subset: best-smooth-like motifs and early variants.
    designs = [
        design
        for design in designs
        if int(design.name.split("_")[1]) < 144
        or int(design.name.split("_")[1]) in range(288, 360)
    ]
    _, predictions, _, _ = evaluate_designs_classifier(
        designs,
        n_reps=20,
        n_splits=5,
        seed=3500,
        parameter_sampler=variable_parameter_sampler,
    )
    conservative = conservative_threshold_table(predictions)
    binary = output_vs_process_table(predictions)
    subtype = process_subtype_table(predictions)
    conservative.to_csv(OUTPUTS / "hybrid_smooth_search_conservative.csv", index=False)
    binary.to_csv(OUTPUTS / "hybrid_smooth_search_binary.csv", index=False)
    subtype.to_csv(OUTPUTS / "hybrid_smooth_search_subtype.csv", index=False)
    predictions.to_csv(OUTPUTS / "hybrid_smooth_search_predictions.csv", index=False)

    top_binary = conservative[conservative.false_process_target == 0.05].head(12)
    top_subtype = subtype.head(12)
    keep = set(top_binary.design) | set(top_subtype.design)
    for design in designs:
        if design.name in keep:
            design.trials.to_csv(OUTPUTS / f"{design.name}_trials.csv", index=False)

    print("Top hybrid smooth designs at false process <= 0.05:")
    print(top_binary.to_string(index=False))
    print("\nTop hybrid smooth designs by process subtype:")
    print(top_subtype.to_string(index=False))
    print("\nBinary for subtype-top designs:")
    print(binary[binary.design.isin(top_subtype.design)].to_string(index=False))
    print(f"\nOutputs: {OUTPUTS}")


if __name__ == "__main__":
    main()
