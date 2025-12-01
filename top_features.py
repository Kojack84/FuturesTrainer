#!/usr/bin/env python
"""
top_features.py

Strict, multi-target feature selection for MNQ MFE/MAE dataset.

- Uses MFE/MAE from calculate_mfe_mae.py to build binary labels
  for multiple (profit_target, stop_size) combos per direction.
- Applies:
    - leakage-safe feature detection
    - variance + rarity filters
    - collinearity pruning
    - time-series cross-validation
    - permutation importance (delta AUC)
    - aggregation across configs
- Selects up to max_features features with:
    - positive median importance
    - reasonable stability
    - used in at least min_configs configs

Outputs:
    - importances_<config>.csv for each config that trains
    - feature_importances_aggregated.csv
    - selected_features.txt (≤ max_features)

Colab + Google Drive usage:

1) In Colab:

   from google.colab import drive
   drive.mount('/content/drive')

2) Run:

   !python top_features.py \
       --input "/content/drive/MyDrive/FeatureTrainer/MNQ_Advanced_MFE_MAE.csv" \
       --direction both \
       --profit-targets 25,50,75,100 \
       --stop-sizes 50,100,150,200 \
       --max-features 50 \
       --output-dir "FeatureTrainer/feature_selection_v1" \
       --gdrive-root "/content/drive/MyDrive"
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────

@dataclass
class TargetConfig:
    direction: str           # "long" or "short"
    profit_target: float     # TP in points
    stop_size: float         # SL in points
    label_column: str        # e.g. "y_long_25_10"


# ─────────────────────────────────────
# Argument parsing & IO
# ─────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict multi-target feature selection for MNQ MFE/MAE dataset"
    )
    parser.add_argument("--input", required=True, help="CSV with MFE/MAE and features")
    parser.add_argument(
        "--direction",
        choices=["long", "short", "both"],
        default="long",
        help="Which direction(s) to build configs for"
    )
    parser.add_argument(
        "--profit-targets",
        default="25,50,75,100",
        help="Comma-separated profit targets in points, e.g. '25,50,75,100'"
    )
    parser.add_argument(
        "--stop-sizes",
        default="5,10,15,20",
        help="Comma-separated stop sizes in points, e.g. '5,10,15,20'"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50,
        help="Maximum number of features to keep"
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=1e-6,
        help="Minimum variance threshold for features"
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for pruning collinear features"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of time-series CV splits"
    )
    parser.add_argument(
        "--output-dir",
        default="feature_selection",
        help="Directory (relative) to write outputs"
    )
    parser.add_argument(
        "--gdrive-root",
        default="",
        help=(
            "Optional: base directory in Google Drive, e.g. "
            "'/content/drive/MyDrive'. If provided, all outputs go under "
            "gdrive-root/output-dir."
        ),
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str, gdrive_root: str) -> str:
    """
    Resolve final output directory, supporting Google Drive in Colab.

    If gdrive_root is non-empty, final_output = gdrive_root/output_dir
    Otherwise final_output = output_dir
    """
    if gdrive_root:
        return os.path.join(gdrive_root, output_dir)
    return output_dir


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.sort_values("Time").reset_index(drop=True)
    return df


# ─────────────────────────────────────
# Feature identification & filters
# ─────────────────────────────────────

def get_candidate_features(df: pd.DataFrame) -> List[str]:
    """
    Get numeric feature columns, excluding outcomes, OHLCV, labels, etc.
    Safe against leakage.
    """
    exclude_exact = ["TF", "Time"]
    exclude_patterns = [
        "Open", "High", "Low", "Close", "Volume",
        "mfe", "mae", "quality", "bucket", "ideal", "disaster",
        "entry_price", "horizon", "subsession", "session_bucket",
        "long_hit", "short_hit", "max_up", "max_down",
    ]

    candidates: List[str] = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        col_lower = col.lower()
        if any(p.lower() in col_lower for p in exclude_patterns):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    return candidates


def apply_variance_and_rarity_filters(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_variance: float,
    min_positive_rate: float = 0.005,
    max_positive_rate: float = 0.995
) -> List[str]:
    """
    Remove low-variance features and ultra-rare binary-like features.
    """
    kept: List[str] = []
    for col in feature_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        var = series.var()
        if var < min_variance:
            continue

        # Rarity filter for binary / near-binary features
        unique_vals = series.unique()
        if len(unique_vals) <= 3:
            # Treat as indicator-like
            p1 = (series > 0).mean()
            if p1 < min_positive_rate or p1 > max_positive_rate:
                continue

        kept.append(col)

    return kept


def apply_collinearity_pruning(
    df: pd.DataFrame,
    feature_cols: List[str],
    corr_threshold: float
) -> List[str]:
    """
    Remove highly collinear features using absolute correlation matrix.

    Keeps the first occurrence and drops later ones when |rho| >= threshold.
    """
    if not feature_cols:
        return []

    sub = df[feature_cols].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()
    if sub.empty:
        # If everything becomes NaN, just return original list
        return feature_cols

    corr = sub.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set = set()
    for col in upper.columns:
        highly_corr = upper.index[upper[col] >= corr_threshold].tolist()
        for c in highly_corr:
            to_drop.add(c)

    pruned = [c for c in feature_cols if c not in to_drop]
    return pruned


# ─────────────────────────────────────
# Label construction for (T,S)
# ─────────────────────────────────────

def build_target_configs(
    direction: str,
    profit_targets: List[float],
    stop_sizes: List[float]
) -> List[TargetConfig]:
    """
    Build all (direction, T, S) combinations.
    """
    configs: List[TargetConfig] = []
    dirs = [direction] if direction != "both" else ["long", "short"]

    for d in dirs:
        for T in profit_targets:
            for S in stop_sizes:
                label = f"y_{d}_{int(T)}_{int(S)}"
                configs.append(
                    TargetConfig(
                        direction=d,
                        profit_target=float(T),
                        stop_size=float(S),
                        label_column=label,
                    )
                )
    return configs


def add_labels_for_config(df: pd.DataFrame, cfg: TargetConfig) -> pd.Series:
    """
    Construct binary label for a given (direction, T, S) using MFE/MAE.

    y = 1 if MFE >= T and MAE <= S else 0
    """
    if cfg.direction == "long":
        if "long_mfe" not in df.columns or "long_mae" not in df.columns:
            raise ValueError("Missing long_mfe/long_mae columns in input data.")
        mfe = df["long_mfe"]
        mae = df["long_mae"]
    else:
        if "short_mfe" not in df.columns or "short_mae" not in df.columns:
            raise ValueError("Missing short_mfe/short_mae columns in input data.")
        mfe = df["short_mfe"]
        mae = df["short_mae"]

    y = ((mfe >= cfg.profit_target) & (mae <= cfg.stop_size)).astype(int)
    df[cfg.label_column] = y
    return y


# ─────────────────────────────────────
# Time-series splits & permutation importance
# ─────────────────────────────────────

def time_series_splits(n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build time-ordered splits using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n_samples)))


def train_model_and_importance(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    For each split:
      - impute missing values (median)
      - scale features
      - train L1 logistic regression
      - compute permutation importance on validation fold (delta AUC)

    Always returns a pandas DataFrame (never None).
    """
    records: List[Dict] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        # Skip degenerate folds with only one class
        if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
            continue

        # Impute NaNs per fold using medians
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_valid_imputed = imputer.transform(X_valid)

        # Scale features after imputation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_valid_scaled = scaler.transform(X_valid_imputed)

        # Train logistic model
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=200
        )
        model.fit(X_train_scaled, y_train)

        # Baseline validation score
        y_pred_proba = model.predict_proba(X_valid_scaled)[:, 1]
        base_auc = roc_auc_score(y_valid, y_pred_proba)

        # Permutation importance on scaled validation data
        for j, name in enumerate(feature_names):
            X_perm = X_valid_scaled.copy()
            perm_col = X_perm[:, j].copy()
            np.random.shuffle(perm_col)
            X_perm[:, j] = perm_col

            y_perm_proba = model.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score(y_valid, y_perm_proba)
            delta_auc = base_auc - perm_auc

            records.append({
                "fold": fold_idx,
                "feature": name,
                "delta_auc": float(delta_auc),
            })

    # If no valid folds or no records, return an empty DataFrame
    if not records:
        return pd.DataFrame(
            columns=["feature", "median_delta_auc", "std_delta_auc", "mean_delta_auc", "n_folds"]
        )

    df_imp = pd.DataFrame(records)
    agg = df_imp.groupby("feature")["delta_auc"].agg(
        median_delta_auc="median",
        std_delta_auc="std",
        mean_delta_auc="mean",
        n_folds="count"
    ).reset_index()

    return agg


# ─────────────────────────────────────
# Aggregation across configs & selection
# ─────────────────────────────────────

def aggregate_across_configs(
    per_config_importances: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine importance stats across all (direction, T, S) configs.

    Returns a DataFrame with:
        feature, median_imp, mean_imp, std_imp, n_configs, n_records, stability_cv
    """
    frames: List[pd.DataFrame] = []

    for cfg_key, df_imp in per_config_importances.items():
        if df_imp is None or df_imp.empty:
            continue
        df_tmp = df_imp.copy()
        df_tmp["config"] = cfg_key
        frames.append(df_tmp)

    if not frames:
        return pd.DataFrame(
            columns=[
                "feature",
                "median_imp",
                "mean_imp",
                "std_imp",
                "n_configs",
                "n_records",
                "stability_cv",
            ]
        )

    all_imp = pd.concat(frames, ignore_index=True)

    agg = all_imp.groupby("feature").agg(
        median_imp=("median_delta_auc", "median"),
        mean_imp=("mean_delta_auc", "mean"),
        std_imp=("median_delta_auc", "std"),
        n_configs=("config", "nunique"),
        n_records=("config", "count"),
    ).reset_index()

    eps = 1e-6
    agg["stability_cv"] = agg["std_imp"] / (agg["median_imp"].abs() + eps)

    return agg


def select_features_from_agg(
    agg: pd.DataFrame,
    max_features: int,
    min_median_imp: float = 0.0,
    max_stability_cv: float = 2.0,
    min_configs: int = 2
) -> List[str]:
    """
    Apply strict selection rules on aggregated importances:
      - median importance > min_median_imp
      - stability_cv <= max_stability_cv
      - used in at least min_configs configs
      - then take top max_features by median_imp
    """
    df = agg.copy()
    df = df[df["median_imp"] > min_median_imp]
    df = df[df["stability_cv"] <= max_stability_cv]
    df = df[df["n_configs"] >= min_configs]

    df = df.sort_values("median_imp", ascending=False)
    selected = df["feature"].head(max_features).tolist()
    return selected


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────

def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.gdrive_root)

    # Safely create the output dir tree
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Failed to create output_dir={output_dir}: {e}")
        print("Falling back to local './feature_selection_fallback'")
        output_dir = "./feature_selection_fallback"
        os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(args.input)
    n_samples = len(df)
    if n_samples == 0:
        print("Input dataset is empty. Exiting.")
        return

    # 1) Feature identification
    feature_cols = get_candidate_features(df)
    feature_cols = apply_variance_and_rarity_filters(
        df, feature_cols, min_variance=args.min_variance
    )
    feature_cols = apply_collinearity_pruning(
        df, feature_cols, corr_threshold=args.corr_threshold
    )

    print(f"Initial candidate features after filters: {len(feature_cols)}")
    if not feature_cols:
        print("No candidate features after filters. Exiting.")
        return

    # 2) Build target configs
    profit_targets = [float(x) for x in args.profit_targets.split(",") if x.strip() != ""]
    stop_sizes = [float(x) for x in args.stop_sizes.split(",") if x.strip() != ""]
    configs = build_target_configs(args.direction, profit_targets, stop_sizes)
    print(f"Number of (direction,T,S) configs: {len(configs)}")

    splits = time_series_splits(n_samples, args.n_splits)

    # 3) Build X matrix and clean non-finite values
    X = df[feature_cols].to_numpy(dtype=float)
    mask_non_finite = ~np.isfinite(X)
    if mask_non_finite.any():
        X[mask_non_finite] = np.nan

    per_config_importances: Dict[str, pd.DataFrame] = {}

    for cfg in configs:
        y = add_labels_for_config(df, cfg)
        positives = int(y.sum())
        cfg_key = f"{cfg.direction}_T{int(cfg.profit_target)}_S{int(cfg.stop_size)}"

        if positives < 100:
            print(f"Config {cfg_key}: positives={positives} < 100, skipping.")
            per_config_importances[cfg_key] = pd.DataFrame(
                columns=["feature", "median_delta_auc", "std_delta_auc", "mean_delta_auc", "n_folds"]
            )
            continue

        print(f"Config {cfg_key}: positives={positives}, samples={len(y)}")

        imp_df = train_model_and_importance(X, y.to_numpy(), splits, feature_cols)

        if imp_df is None:
            imp_df = pd.DataFrame(
                columns=["feature", "median_delta_auc", "std_delta_auc", "mean_delta_auc", "n_folds"]
            )

        if not imp_df.empty:
            imp_path = os.path.join(output_dir, f"importances_{cfg_key}.csv")
            imp_df.to_csv(imp_path, index=False)

        per_config_importances[cfg_key] = imp_df

    # 4) Aggregate across configs
    agg = aggregate_across_configs(per_config_importances)
    if agg.empty:
        print("No importance results from any config. Exiting.")
        return

    agg_path = os.path.join(output_dir, "feature_importances_aggregated.csv")
    agg.to_csv(agg_path, index=False)

    # 5) Final selection
    selected_features = select_features_from_agg(
        agg,
        max_features=args.max_features,
        min_median_imp=0.0,
        max_stability_cv=2.0,
        min_configs=2
    )

    sel_path = os.path.join(output_dir, "selected_features.txt")
    with open(sel_path, "w") as f:
        for feat in selected_features:
            f.write(feat + "\n")

    print(f"Selected {len(selected_features)} features (<= {args.max_features}):")
    for name in selected_features:
        print("  -", name)
    print(f"\nSaved aggregated importances to: {agg_path}")
    print(f"Saved selected features to:      {sel_path}")


if __name__ == "__main__":
    main()
