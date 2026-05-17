"""Diagnostics for model-family confusability."""

from __future__ import annotations

import pandas as pd


def pairwise_confusion_table(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize symmetric pairwise confusion from normalized matrices."""

    rows = []
    for design, matrix in matrices.items():
        labels = list(matrix.index)
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                left_to_right = float(matrix.loc[left, right])
                right_to_left = float(matrix.loc[right, left])
                rows.append(
                    {
                        "design": design,
                        "model_a": left,
                        "model_b": right,
                        "a_to_b": left_to_right,
                        "b_to_a": right_to_left,
                        "symmetric_confusion": 0.5 * (left_to_right + right_to_left),
                    }
                )
    return pd.DataFrame(rows).sort_values("symmetric_confusion", ascending=False)


def cluster_recovery_table(
    predictions: pd.DataFrame,
    clusters: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute recovery accuracy after collapsing model families into clusters."""

    if clusters is None:
        clusters = {
            "output_only": "state_scaled_cluster",
            "error_term": "state_scaled_cluster",
            "retention_term": "state_scaled_cluster",
            "additive": "additive_bias_cluster",
            "bias_term": "additive_bias_cluster",
            "full_component": "additive_bias_cluster",
        }

    df = predictions.copy()
    df["true_cluster"] = df["true_model"].map(clusters)
    df["predicted_cluster"] = df["predicted_model"].map(clusters)
    grouped = (
        df.groupby("design")
        .apply(lambda group: float((group.true_cluster == group.predicted_cluster).mean()))
        .reset_index(name="cluster_accuracy")
    )
    return grouped.sort_values("cluster_accuracy", ascending=False)


def output_vs_process_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute primary recovery for output-only versus any process-noise model."""

    df = predictions.copy()
    df["true_binary"] = df["true_model"].where(df["true_model"] == "output_only", "process")
    df["predicted_binary"] = df["predicted_model"].where(df["predicted_model"] == "output_only", "process")
    rows = []
    for design, group in df.groupby("design"):
        true_output = group[group.true_binary == "output_only"]
        true_process = group[group.true_binary == "process"]
        rows.append(
            {
                "design": design,
                "binary_accuracy": float((group.true_binary == group.predicted_binary).mean()),
                "output_recall": float((true_output.predicted_binary == "output_only").mean()),
                "process_recall": float((true_process.predicted_binary == "process").mean()),
                "false_process_rate": float((true_output.predicted_binary == "process").mean()),
                "false_output_rate": float((true_process.predicted_binary == "output_only").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["binary_accuracy", "process_recall", "output_recall"],
        ascending=False,
    )


def process_subtype_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute subtype accuracy after excluding true output-only datasets."""

    df = predictions[predictions.true_model != "output_only"].copy()
    rows = []
    for design, group in df.groupby("design"):
        process_pred = group[group.predicted_model != "output_only"]
        rows.append(
            {
                "design": design,
                "process_subtype_accuracy": float((group.true_model == group.predicted_model).mean()),
                "process_subtype_accuracy_given_process_prediction": (
                    float((process_pred.true_model == process_pred.predicted_model).mean())
                    if len(process_pred)
                    else 0.0
                ),
                "process_predicted_as_output": float((group.predicted_model == "output_only").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("process_subtype_accuracy", ascending=False)


def output_pair_detectability(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise output-only versus each process model detectability."""

    rows = []
    process_models = sorted(set(predictions.true_model) - {"output_only"})
    for design, design_group in predictions.groupby("design"):
        for model in process_models:
            pair = design_group[design_group.true_model.isin(["output_only", model])].copy()
            pair["true_binary"] = pair["true_model"].where(pair.true_model == "output_only", "process")
            pair["predicted_binary"] = pair["predicted_model"].where(
                pair.predicted_model == "output_only",
                "process",
            )
            true_output = pair[pair.true_binary == "output_only"]
            true_process = pair[pair.true_binary == "process"]
            rows.append(
                {
                    "design": design,
                    "process_model": model,
                    "binary_accuracy": float((pair.true_binary == pair.predicted_binary).mean()),
                    "output_recall": float((true_output.predicted_binary == "output_only").mean()),
                    "process_recall": float((true_process.predicted_binary == "process").mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["binary_accuracy", "process_recall"],
        ascending=False,
    )


def conservative_threshold_table(
    predictions: pd.DataFrame,
    false_process_targets: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Rank designs by process recall under constrained false-process rates."""

    rows = []
    for design, group in predictions.groupby("design"):
        true_output = group[group.true_model == "output_only"]
        true_process = group[group.true_model != "output_only"]
        thresholds = sorted(set(group["process_probability"].round(6).tolist() + [0.0, 1.0]))
        for target in false_process_targets:
            best = None
            for threshold in thresholds:
                output_called_process = true_output.process_probability >= threshold
                process_called_process = true_process.process_probability >= threshold
                false_process_rate = float(output_called_process.mean())
                if false_process_rate <= target:
                    process_recall = float(process_called_process.mean())
                    output_recall = 1.0 - false_process_rate
                    binary_accuracy = float(
                        (
                            (group.true_model == "output_only")
                            == (group.process_probability < threshold)
                        ).mean()
                    )
                    candidate = {
                        "design": design,
                        "false_process_target": target,
                        "threshold": float(threshold),
                        "binary_accuracy": binary_accuracy,
                        "output_recall": output_recall,
                        "process_recall": process_recall,
                        "false_process_rate": false_process_rate,
                        "false_output_rate": 1.0 - process_recall,
                    }
                    if best is None or candidate["process_recall"] > best["process_recall"]:
                        best = candidate
            if best is not None:
                rows.append(best)
    return pd.DataFrame(rows).sort_values(
        ["false_process_target", "process_recall", "output_recall"],
        ascending=[True, False, False],
    )
