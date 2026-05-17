"""Model-free recovery using synthetic features and classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .design import TrialDesign
from .features import feature_frame
from .models import MODEL_FAMILIES, default_params
from .simulate import simulate_subject


def simulate_feature_table(
    design: TrialDesign,
    n_reps: int = 80,
    seed: int = 100,
    parameter_sampler=None,
) -> pd.DataFrame:
    """Simulate many datasets and extract model-free features."""

    datasets = []
    subject = 0
    for true_family in MODEL_FAMILIES:
        for rep in range(n_reps):
            params = (
                parameter_sampler(true_family, subject, seed + subject * 37)
                if parameter_sampler is not None
                else default_params(true_family)
            )
            datasets.append(
                simulate_subject(
                    design,
                    true_family,
                    params,
                    seed=seed + subject * 37,
                    subject=subject,
                )
            )
            subject += 1
    return feature_frame(datasets)


def classify_feature_table(
    features: pd.DataFrame,
    n_splits: int = 20,
    seed: int = 1,
    classifier: str = "logistic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test a classifier and return predictions plus confusion matrix."""

    labels = list(MODEL_FAMILIES)
    feature_cols = [col for col in features.columns if col not in {"true_model", "design", "subject"}]
    x = features[feature_cols].to_numpy(dtype=float)
    y = features["true_model"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.35, random_state=seed)
    rows = []

    for split, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        if classifier == "extra_trees":
            clf = ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed + split,
                n_jobs=-1,
            )
        else:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
        clf.fit(x[train_idx], y[train_idx])
        predicted = clf.predict(x[test_idx])
        probabilities = clf.predict_proba(x[test_idx])
        classes = list(clf.classes_)
        process_probability = np.zeros(len(test_idx))
        for class_index, class_name in enumerate(classes):
            if class_name != "output_only":
                process_probability += probabilities[:, class_index]
        for row_i, (idx, pred) in enumerate(zip(test_idx, predicted, strict=True)):
            rows.append(
                {
                    "split": split,
                    "design": features["design"].iloc[idx],
                    "subject": int(features["subject"].iloc[idx]),
                    "true_model": y[idx],
                    "predicted_model": pred,
                    "process_probability": float(process_probability[row_i]),
                    "correct": bool(pred == y[idx]),
                }
            )

    predictions = pd.DataFrame(rows)
    matrix = confusion_matrix(predictions["true_model"], predictions["predicted_model"], labels=labels)
    matrix = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    return predictions, matrix_df


def evaluate_design_classifier(
    design: TrialDesign,
    n_reps: int = 80,
    n_splits: int = 20,
    seed: int = 100,
    parameter_sampler=None,
    classifier: str = "logistic",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run model-free recovery for one design."""

    features = simulate_feature_table(
        design,
        n_reps=n_reps,
        seed=seed,
        parameter_sampler=parameter_sampler,
    )
    predictions, matrix = classify_feature_table(
        features,
        n_splits=n_splits,
        seed=seed,
        classifier=classifier,
    )
    return features, predictions, matrix


def evaluate_designs_classifier(
    designs: list[TrialDesign],
    n_reps: int = 80,
    n_splits: int = 20,
    seed: int = 100,
    parameter_sampler=None,
    classifier: str = "logistic",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate model-free recovery across candidate designs."""

    feature_frames = []
    prediction_frames = []
    matrices = {}
    ranking_rows = []

    for i, design in enumerate(designs):
        features, predictions, matrix = evaluate_design_classifier(
            design,
            n_reps=n_reps,
            n_splits=n_splits,
            seed=seed + i * 1000,
            parameter_sampler=parameter_sampler,
            classifier=classifier,
        )
        feature_frames.append(features)
        prediction_frames.append(predictions)
        matrices[design.name] = matrix
        ranking_rows.append(
            {
                "design": design.name,
                "classifier_accuracy": float(predictions["correct"].mean()),
                "worst_model_recall": float(np.diag(matrix.to_numpy()).min()),
                "mean_recall": float(np.diag(matrix.to_numpy()).mean()),
            }
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["classifier_accuracy", "worst_model_recall"],
        ascending=False,
    )
    return (
        pd.concat(feature_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        ranking,
        matrices,
    )
