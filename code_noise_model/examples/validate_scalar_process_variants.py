"""Validate scalar single-target recovery of process-noise variants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.features import feature_frame
from noise_model.scalar_kf import simulate_scalar
from noise_model.search import generate_hybrid_smooth_designs


OUTPUTS = ROOT / "outputs"
FAMILIES = ["output_only", "bias_term", "error_term", "retention_term", "full_component"]


def sample_params(family: str, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    params = {
        "A": float(np.clip(rng.normal(0.15, 0.04), 0.05, 0.30)),
        "B": float(np.clip(rng.normal(0.98, 0.025), 0.90, 1.02)),
        "C": float(np.clip(rng.normal(0.0, 0.5), -1.5, 1.5)),
        "output_sd": float(np.clip(rng.normal(3.0, 0.75), 1.5, 5.5)),
        "process_sd": float(np.clip(rng.lognormal(np.log(1.0), 0.35), 0.25, 2.5)),
        "nA_sd": float(np.clip(rng.lognormal(np.log(0.08), 0.35), 0.02, 0.25)),
        "nB_sd": float(np.clip(rng.lognormal(np.log(0.025), 0.35), 0.005, 0.08)),
    }
    return params


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    design = next(d for d in generate_hybrid_smooth_designs(n_trials=240, seed=3500) if d.name == "hybrid_304")
    trials = design.trials.copy()
    trials["target_index"] = 0
    trials["target_deg"] = 0.0
    datasets = []
    n_reps = 160
    subject = 0
    for family in FAMILIES:
        for rep in range(n_reps):
            seed = 13100 + subject * 17
            params = sample_params(family, seed)
            data = simulate_scalar(trials, family, params, seed=seed)
            data["subject"] = subject
            data["design"] = "scalar_hybrid_304"
            datasets.append(data)
            subject += 1
    features = feature_frame(datasets)
    features.to_csv(OUTPUTS / "scalar_process_variants_features.csv", index=False)
    cols = [col for col in features.columns if col not in {"true_model", "design", "subject"}]
    x = features[cols].to_numpy(float)
    y = features["true_model"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=30, test_size=0.35, random_state=131)
    rows = []
    for split, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        proba = clf.predict_proba(x[test_idx])
        classes = list(clf.classes_)
        p_process = np.zeros(len(test_idx))
        for class_i, class_name in enumerate(classes):
            if class_name != "output_only":
                p_process += proba[:, class_i]
        for idx, pred_i, pp in zip(test_idx, pred, p_process, strict=True):
            rows.append(
                {
                    "split": split,
                    "true_model": y[idx],
                    "predicted_model": pred_i,
                    "process_probability": float(pp),
                }
            )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(OUTPUTS / "scalar_process_variants_predictions.csv", index=False)
    mat = confusion_matrix(predictions.true_model, predictions.predicted_model, labels=FAMILIES)
    mat = mat / mat.sum(axis=1, keepdims=True)
    matrix = pd.DataFrame(mat, index=FAMILIES, columns=FAMILIES)
    matrix.to_csv(OUTPUTS / "scalar_process_variants_confusion.csv")
    print("Five-way scalar confusion")
    print(matrix.round(3).to_string())
    print("accuracy", round((predictions.true_model == predictions.predicted_model).mean(), 3))

    proc = predictions[predictions.true_model != "output_only"].copy()
    proc_families = [family for family in FAMILIES if family != "output_only"]
    mat2 = confusion_matrix(proc.true_model, proc.predicted_model, labels=proc_families)
    mat2 = mat2 / mat2.sum(axis=1, keepdims=True)
    matrix2 = pd.DataFrame(mat2, index=proc_families, columns=proc_families)
    matrix2.to_csv(OUTPUTS / "scalar_process_variants_process_only_confusion.csv")
    print("\nProcess-only subtype confusion")
    print(matrix2.round(3).to_string())
    print("process subtype accuracy", round((proc.true_model == proc.predicted_model).mean(), 3))


if __name__ == "__main__":
    main()
