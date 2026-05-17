"""Feature extraction for model-free synthetic recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_stat(values: pd.Series, func: str) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return 0.0
    if func == "mean":
        return float(np.mean(arr))
    if func == "std":
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if func == "var":
        return float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
    raise ValueError(func)


def _lag_corr(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    x = values[:-1]
    y = values[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[valid]
    y = y_values[valid]
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, deg=1)[0])


def extract_features(data: pd.DataFrame) -> dict[str, float]:
    """Extract dataset-level features that target noise-model signatures."""

    df = data.sort_values("trial").reset_index(drop=True).copy()
    df["abs_error"] = df["error"].abs()
    df["dy"] = df["y"].diff()
    df["next_y"] = df["y"].shift(-1)
    df["next2_y"] = df["y"].shift(-2)
    df["next3_y"] = df["y"].shift(-3)
    df["carryover"] = df["next_y"] - df["y"]
    df["carryover2"] = df["next2_y"] - df["y"]
    df["carryover3"] = df["next3_y"] - df["y"]
    df["abs_y"] = df["y"].abs()
    df["none_run"] = (df["feedback_type"] != "none").cumsum()
    df["none_position"] = 0
    none_mask = df["feedback_type"] == "none"
    df.loc[none_mask, "none_position"] = df[none_mask].groupby("none_run").cumcount() + 1
    df["last_feedback_type"] = df["feedback_type"].where(df["feedback_type"] != "none").ffill()
    df["last_abs_error"] = df["abs_error"].where(df["feedback_type"] != "none").ffill().fillna(0.0)
    df["state_proxy"] = (
        df.groupby("target_index")["y"]
        .transform(lambda values: values.shift(1).ewm(alpha=0.35, adjust=False).mean())
        .fillna(0.0)
    )
    df["abs_state_proxy"] = df["state_proxy"].abs()

    features: dict[str, float] = {
        "y_mean": _safe_stat(df["y"], "mean"),
        "y_sd": _safe_stat(df["y"], "std"),
        "dy_sd": _safe_stat(df["dy"], "std"),
        "lag1_y": _lag_corr(df["y"].to_numpy(dtype=float)),
        "lag1_dy": _lag_corr(df["dy"].dropna().to_numpy(dtype=float)),
        "error_sd": _safe_stat(df["error"], "std"),
        "abs_error_mean": _safe_stat(df["abs_error"], "mean"),
        "carryover_sd": _safe_stat(df["carryover"], "std"),
        "carryover2_sd": _safe_stat(df["carryover2"], "std"),
        "carryover3_sd": _safe_stat(df["carryover3"], "std"),
        "abs_state_proxy_mean": _safe_stat(df["abs_state_proxy"], "mean"),
        "abs_state_proxy_sd": _safe_stat(df["abs_state_proxy"], "std"),
        "abs_y_abs_error_slope": _slope(df["abs_error"].to_numpy(float), df["abs_y"].to_numpy(float)),
        "carryover_abs_error_slope": _slope(
            df["abs_error"].to_numpy(float),
            df["carryover"].abs().to_numpy(float),
        ),
        "carryover_state_proxy_slope": _slope(
            df["abs_state_proxy"].to_numpy(float),
            df["carryover"].abs().to_numpy(float),
        ),
    }

    for feedback_type in ["rotated", "clamp", "none"]:
        subset = df[df.feedback_type == feedback_type]
        features[f"{feedback_type}_y_sd"] = _safe_stat(subset["y"], "std")
        features[f"{feedback_type}_dy_sd"] = _safe_stat(subset["dy"], "std")
        features[f"{feedback_type}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
        features[f"{feedback_type}_abs_error_mean"] = _safe_stat(subset["abs_error"], "mean")
        features[f"{feedback_type}_carryover2_sd"] = _safe_stat(subset["carryover2"], "std")
        features[f"{feedback_type}_carryover3_sd"] = _safe_stat(subset["carryover3"], "std")
        features[f"{feedback_type}_state_proxy_slope"] = _slope(
            subset["abs_state_proxy"].to_numpy(float),
            subset["carryover"].abs().to_numpy(float),
        )
        features[f"{feedback_type}_error_slope"] = _slope(
            subset["abs_error"].to_numpy(float),
            subset["carryover"].abs().to_numpy(float),
        )

    for phase in ["zero_mean", "nonzero_mean"]:
        subset = df[df.phase == phase]
        features[f"{phase}_y_sd"] = _safe_stat(subset["y"], "std")
        features[f"{phase}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
        features[f"{phase}_abs_error_mean"] = _safe_stat(subset["abs_error"], "mean")
        features[f"{phase}_state_proxy_mean"] = _safe_stat(subset["abs_state_proxy"], "mean")
        features[f"{phase}_state_proxy_slope"] = _slope(
            subset["abs_state_proxy"].to_numpy(float),
            subset["carryover"].abs().to_numpy(float),
        )

    low = df[df.abs_error <= 4.0]
    mid = df[(df.abs_error > 4.0) & (df.abs_error <= 8.0)]
    high = df[df.abs_error > 8.0]
    for label, subset in [("low_error", low), ("mid_error", mid), ("high_error", high)]:
        features[f"{label}_y_sd"] = _safe_stat(subset["y"], "std")
        features[f"{label}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
        features[f"{label}_carryover_abs_mean"] = _safe_stat(subset["carryover"].abs(), "mean")

    low_state = df[df.abs_state_proxy <= 3.0]
    mid_state = df[(df.abs_state_proxy > 3.0) & (df.abs_state_proxy <= 7.0)]
    high_state = df[df.abs_state_proxy > 7.0]
    for label, subset in [("low_state", low_state), ("mid_state", mid_state), ("high_state", high_state)]:
        features[f"{label}_y_sd"] = _safe_stat(subset["y"], "std")
        features[f"{label}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
        features[f"{label}_carryover_abs_mean"] = _safe_stat(subset["carryover"].abs(), "mean")

    for target_index, subset in df.groupby("target_index"):
        features[f"target_{int(target_index)}_y_sd"] = _safe_stat(subset["y"], "std")
        features[f"target_{int(target_index)}_lag1_y"] = _lag_corr(subset["y"].to_numpy(dtype=float))

    none = df[df.feedback_type == "none"].copy()
    features["none_lag1_y"] = _lag_corr(none["y"].to_numpy(dtype=float))
    features["none_lag1_dy"] = _lag_corr(none["dy"].dropna().to_numpy(dtype=float))
    features["none_state_proxy_slope"] = _slope(
        none["abs_state_proxy"].to_numpy(float),
        none["carryover"].abs().to_numpy(float),
    )
    features["none_last_error_slope"] = _slope(
        none["last_abs_error"].to_numpy(float),
        none["carryover"].abs().to_numpy(float),
    )
    for last_type in ["rotated", "clamp"]:
        subset = none[none.last_feedback_type == last_type]
        features[f"none_after_{last_type}_y_sd"] = _safe_stat(subset["y"], "std")
        features[f"none_after_{last_type}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
        features[f"none_after_{last_type}_last_error_slope"] = _slope(
            subset["last_abs_error"].to_numpy(float),
            subset["carryover"].abs().to_numpy(float),
        )
    for position in [1, 2, 3, 4]:
        subset = none[none.none_position == position]
        features[f"none_pos{position}_y_sd"] = _safe_stat(subset["y"], "std")
        features[f"none_pos{position}_abs_y_mean"] = _safe_stat(subset["y"].abs(), "mean")
        features[f"none_pos{position}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
    for min_len in [2, 3, 4]:
        endpoints = []
        slopes = []
        for _, run in none.groupby("none_run"):
            if len(run) >= min_len:
                y = run["y"].to_numpy(dtype=float)
                endpoints.append(y[min_len - 1] - y[0])
                x = np.arange(min_len, dtype=float)
                slopes.append(float(np.polyfit(x, y[:min_len], deg=1)[0]))
        features[f"none_run{min_len}_endpoint_sd"] = float(np.std(endpoints, ddof=1)) if len(endpoints) > 1 else 0.0
        features[f"none_run{min_len}_slope_sd"] = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0
        features[f"none_run{min_len}_abs_slope_mean"] = float(np.mean(np.abs(slopes))) if slopes else 0.0

    clamp = df[df.feedback_type == "clamp"]
    if len(clamp):
        for magnitude in [4.0, 8.0, 12.0]:
            subset = clamp[np.isclose(clamp["abs_error"], magnitude)]
            features[f"clamp_{int(magnitude)}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
            features[f"clamp_{int(magnitude)}_y_sd"] = _safe_stat(subset["y"], "std")
            features[f"clamp_{int(magnitude)}_carryover2_sd"] = _safe_stat(subset["carryover2"], "std")
            features[f"clamp_{int(magnitude)}_carryover3_sd"] = _safe_stat(subset["carryover3"], "std")
            features[f"clamp_{int(magnitude)}_state_proxy_mean"] = _safe_stat(
                subset["abs_state_proxy"],
                "mean",
            )
        features["clamp_error_to_carryover_abs_slope"] = _slope(
            clamp["abs_error"].to_numpy(float),
            clamp["carryover"].abs().to_numpy(float),
        )
        features["clamp_state_to_carryover_abs_slope"] = _slope(
            clamp["abs_state_proxy"].to_numpy(float),
            clamp["carryover"].abs().to_numpy(float),
        )
        for state_label, subset in [
            ("clamp_low_state", clamp[clamp.abs_state_proxy <= 3.0]),
            ("clamp_mid_state", clamp[(clamp.abs_state_proxy > 3.0) & (clamp.abs_state_proxy <= 7.0)]),
            ("clamp_high_state", clamp[clamp.abs_state_proxy > 7.0]),
        ]:
            features[f"{state_label}_carryover_sd"] = _safe_stat(subset["carryover"], "std")
            features[f"{state_label}_error_slope"] = _slope(
                subset["abs_error"].to_numpy(float),
                subset["carryover"].abs().to_numpy(float),
            )

    return features


def feature_frame(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Build a feature table from simulated datasets."""

    rows = []
    for data in datasets:
        features = extract_features(data)
        features["true_model"] = str(data["true_model"].iloc[0])
        features["design"] = str(data["design"].iloc[0])
        features["subject"] = int(data["subject"].iloc[0])
        rows.append(features)
    return pd.DataFrame(rows).fillna(0.0)
