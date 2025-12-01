#!/usr/bin/env python
"""
train_models.py

Train probability / EV models for MNQ MFE/MAE data using a selected feature set.

For each (direction, profit_target, stop_size):

  - Builds label:
        y = 1 if MFE >= T and MAE <= S else 0
  - Uses time-ordered cross-validation:
        * Pipeline: Median Imputer -> StandardScaler -> LogisticRegression(L1)
        * Computes AUC, win rate, EV per threshold
        * Searches probability threshold that maximizes mean PL/trade
  - Trains final model on all data with that Pipeline.
  - Writes artifacts:
        models/model_<direction>_T<T>_S<S>.joblib
        thresholds.json (all configs)
        features_order.txt (same as input feature list)
        metrics_<direction>_T<T>_S<S>.csv (fold-level metrics)

Colab + Google Drive usage:

1) In a Colab cell:

   from google.colab import drive
   drive.mount('/content/drive')

2) Run:

   !python train_models.py \
       --input "/content/drive/MyDrive/MNQ/MNQ_Advanced_MFE_MAE.csv" \
       --features "/content/drive/MyDrive/MNQ/selected_features.txt" \
       --direction both \
       --profit-targets 25,50,75,100 \
       --stop-sizes 5,10,15,20 \
       --output-dir "models_v1" \
       --gdrive-root "/content/drive/MyDrive/MNQ"

All outputs will be saved under:
    /content/drive/MyDrive/MNQ/models_v1
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump


# ─────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────

@dataclass
class ModelConfig:
    direction: str           # "long" or "short"
    profit_target: float     # TP in points
    stop_size: float         # SL in points
    label_column: str        # e.g. "y_long_25_10"


@dataclass
class FoldMetrics:
    config_key: str
    fold_index: int
    n_train: int
    n_valid: int
    auc: float
    best_threshold: float
    best_ev_per_trade: float
    best_win_rate: float
    best_n_trades: int


@dataclass
class ConfigThreshold:
    direction: str
    profit_target: float
    stop_size: float
    best_threshold: float
    cv_ev_per_trade: float
    cv_auc_mean: float
    cv_auc_std: float
    cv_trades_mean: float
    cv_trades_std: float


# ─────────────────────────────────────
# Argument parsing & IO
# ─────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EV models for MNQ MFE/MAE dataset using selected features"
    )
    parser.add_argument("--input", required=True, help="CSV with MFE/MAE and features")
    parser.add_argument("--features", required=True, help="Text file of selected feature names, one per line")
    parser.add_argument(
        "--direction",
        choices=["long", "short", "both"],
        default="both",
        help="Train models for long, short, or both"
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
        "--n-splits",
        type=int,
        default=5,
        help="Number of time-series CV splits"
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=100,
        help="Minimum number of positive samples required to train a model"
    )
    parser.add_argument(
        "--threshold-min-trades",
        type=int,
        default=200,
        help="Minimum number of trades for a threshold candidate during CV"
    )
    parser.add_argument(
        "--output-dir",
        default="models_output",
        help="Directory (relative) to write models, thresholds, and metrics"
    )
    parser.add_argument(
        "--gdrive-root",
        default="",
        help=(
            "Optional: base directory in Google Drive, e.g. "
            "'/content/drive/MyDrive/MNQ'. If provided, all outputs go under "
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


def load_feature_list(path: str) -> List[str]:
    with open(path, "r") as f:
        features = [line.strip() for line in f if line.strip() != ""]
    return features


# ─────────────────────────────────────
# Model config & labels
# ─────────────────────────────────────

def build_model_configs(
    direction: str,
    profit_targets: List[float],
    stop_sizes: List[float]
) -> List[ModelConfig]:
    configs: List[ModelConfig] = []
    dirs = [direction] if direction != "both" else ["long", "short"]

    for d in dirs:
        for T in profit_targets:
            for S in stop_sizes:
                label = f"y_{d}_{int(T)}_{int(S)}"
                configs.append(
                    ModelConfig(
                        direction=d,
                        profit_target=float(T),
                        stop_size=float(S),
                        label_column=label,
                    )
                )
    return configs


def add_label_column(df: pd.DataFrame, cfg: ModelConfig) -> pd.Series:
    """
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
# Time-series splits & threshold scan
# ─────────────────────────────────────

def build_time_splits(n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n_samples)))


def threshold_scan(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    profit_target: float,
    stop_size: float,
    min_trades: int
) -> Tuple[float, float, float, int]:
    """
    Scan probability thresholds and compute EV per trade using realized labels.

    Returns:
        best_threshold, best_ev_per_trade, best_win_rate, best_n_trades
    """
    thresholds = np.linspace(0.1, 0.9, 17)  # 0.10, 0.15, ..., 0.90
    best_thr = 0.5
    best_ev = -1e9
    best_wr = 0.0
    best_n = 0

    for thr in thresholds:
        mask = y_proba >= thr
        n_trades = int(mask.sum())
        if n_trades < min_trades:
            continue

        outcomes = np.where(y_true[mask] == 1, profit_target, -stop_size)
        ev = float(outcomes.mean())
        win_rate = float((outcomes > 0).mean())

        if ev > best_ev:
            best_ev = ev
            best_thr = float(thr)
            best_wr = win_rate
            best_n = n_trades

    return best_thr, best_ev, best_wr, best_n


# ─────────────────────────────────────
# CV training & metrics
# ─────────────────────────────────────

def create_pipeline() -> Pipeline:
    """
    Build the sklearn pipeline for training:
        Median Imputer -> StandardScaler -> LogisticRegression(L1)
    """
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=400,
    )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipeline


def run_cv_for_config(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ModelConfig,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    min_trades_per_threshold: int
) -> Tuple[List[FoldMetrics], float, float, float, float]:
    """
    Run time-series CV for one config, computing AUC and EV threshold per fold.

    Returns:
        fold_metrics: list of FoldMetrics
        avg_best_thr: weighted average best threshold (by n_trades)
        auc_mean: mean AUC across folds
        auc_std: std AUC across folds
        ev_mean: mean best EV per trade across folds
    """
    fold_metrics: List[FoldMetrics] = []
    aucs: List[float] = []
    evs: List[float] = []
    thresholds: List[float] = []
    trades_counts: List[int] = []

    config_key = f"{cfg.direction}_T{int(cfg.profit_target)}_S{int(cfg.stop_size)}"

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        # Skip degenerate folds
        if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
            continue

        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)

        y_proba_valid = pipeline.predict_proba(X_valid)[:, 1]

        # AUC
        auc = roc_auc_score(y_valid, y_proba_valid)
        aucs.append(float(auc))

        # Threshold scan for EV
        best_thr, best_ev, best_wr, best_n = threshold_scan(
            y_true=y_valid,
            y_proba=y_proba_valid,
            profit_target=cfg.profit_target,
            stop_size=cfg.stop_size,
            min_trades=min_trades_per_threshold
        )

        evs.append(float(best_ev))
        thresholds.append(float(best_thr))
        trades_counts.append(int(best_n))

        fm = FoldMetrics(
            config_key=config_key,
            fold_index=fold_idx,
            n_train=int(len(train_idx)),
            n_valid=int(len(valid_idx)),
            auc=float(auc),
            best_threshold=float(best_thr),
            best_ev_per_trade=float(best_ev),
            best_win_rate=float(best_wr),
            best_n_trades=int(best_n),
        )
        fold_metrics.append(fm)

    if not aucs:
        return fold_metrics, 0.5, 0.0, 0.0, 0.0

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    ev_mean = float(np.mean(evs))

    # Weighted average threshold by number of trades per fold
    total_trades = sum(trades_counts) if trades_counts else 0
    if total_trades > 0:
        avg_thr = float(
            np.average(
                thresholds,
                weights=[max(n, 1) for n in trades_counts]
            )
        )
    else:
        avg_thr = float(np.mean(thresholds))

    return fold_metrics, avg_thr, auc_mean, auc_std, ev_mean


# ─────────────────────────────────────
# Final model training & artifacts
# ─────────────────────────────────────

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ModelConfig
) -> Pipeline:
    """
    Train final pipeline on all data for a config.
    """
    pipeline = create_pipeline()
    pipeline.fit(X, y)
    return pipeline


def save_pipeline(
    pipeline: Pipeline,
    output_dir: str,
    cfg: ModelConfig
) -> str:
    """
    Save model pipeline as joblib file.
    """
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_name = f"model_{cfg.direction}_T{int(cfg.profit_target)}_S{int(cfg.stop_size)}.joblib"
    model_path = os.path.join(models_dir, model_name)
    dump(pipeline, model_path)
    return model_path


def save_features_order(
    feature_list: List[str],
    output_dir: str
) -> str:
    """
    Save the feature order used for training.
    """
    path = os.path.join(output_dir, "features_order.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        for feat in feature_list:
            f.write(feat + "\n")
    return path


def save_fold_metrics(
    fold_metrics: List[FoldMetrics],
    output_dir: str,
    cfg: ModelConfig
) -> str:
    """
    Save per-fold metrics to CSV for inspection.
    """
    if not fold_metrics:
        return ""

    df = pd.DataFrame([asdict(fm) for fm in fold_metrics])
    metrics_name = f"metrics_{cfg.direction}_T{int(cfg.profit_target)}_S{int(cfg.stop_size)}.csv"
    metrics_path = os.path.join(output_dir, metrics_name)
    df.to_csv(metrics_path, index=False)
    return metrics_path


def save_thresholds_json(
    thresholds: List[ConfigThreshold],
    output_dir: str
) -> str:
    """
    Save all config thresholds into a single JSON file.
    """
    path = os.path.join(output_dir, "thresholds.json")
    data = [asdict(thr) for thr in thresholds]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────

def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.gdrive_root)
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(args.input)
    feature_list = load_feature_list(args.features)

    if len(df) == 0:
        print("Input dataset is empty. Exiting.")
        return

    # Ensure all requested features exist
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print("The following requested features are missing from the dataset:")
        for m in missing:
            print("  -", m)
        print("Fix dataset or feature list and rerun.")
        return

    # Build X and clean non-finite
    X = df[feature_list].to_numpy(dtype=float)
    mask_non_finite = ~np.isfinite(X)
    if mask_non_finite.any():
        X[mask_non_finite] = np.nan

    n_samples = len(df)
    splits = build_time_splits(n_samples, args.n_splits)

    profit_targets = [float(x) for x in args.profit_targets.split(",") if x.strip() != ""]
    stop_sizes = [float(x) for x in args.stop_sizes.split(",") if x.strip() != ""]
    configs = build_model_configs(args.direction, profit_targets, stop_sizes)

    all_thresholds: List[ConfigThreshold] = []

    # Save feature order once
    features_order_path = save_features_order(feature_list, output_dir)
    print(f"Saved feature order to: {features_order_path}")

    for cfg in configs:
        y = add_label_column(df, cfg)
        positives = int(y.sum())
        config_key = f"{cfg.direction}_T{int(cfg.profit_target)}_S{int(cfg.stop_size)}"

        if positives < args.min_positives:
            print(f"[{config_key}] positives={positives} < {args.min_positives}, skipping.")
            continue

        print(f"[{config_key}] positives={positives}, samples={len(y)}")

        # CV
        fold_metrics, avg_thr, auc_mean, auc_std, ev_mean = run_cv_for_config(
            X=X,
            y=y.to_numpy(),
            cfg=cfg,
            splits=splits,
            min_trades_per_threshold=args.threshold_min_trades
        )

        metrics_path = save_fold_metrics(fold_metrics, output_dir, cfg)
        if metrics_path:
            print(f"[{config_key}] Saved fold metrics to: {metrics_path}")

        if not fold_metrics:
            print(f"[{config_key}] No valid folds (degenerate splits), skipping final model.")
            continue

        # Aggregate thresholds & metrics
        n_trades = [fm.best_n_trades for fm in fold_metrics]
        trades_mean = float(np.mean(n_trades))
        trades_std = float(np.std(n_trades))

        thr_info = ConfigThreshold(
            direction=cfg.direction,
            profit_target=cfg.profit_target,
            stop_size=cfg.stop_size,
            best_threshold=float(avg_thr),
            cv_ev_per_trade=float(ev_mean),
            cv_auc_mean=float(auc_mean),
            cv_auc_std=float(auc_std),
            cv_trades_mean=float(trades_mean),
            cv_trades_std=float(trades_std),
        )
        all_thresholds.append(thr_info)

        # Final model on all data
        final_pipeline = train_final_model(X, y.to_numpy(), cfg)
        model_path = save_pipeline(final_pipeline, output_dir, cfg)
        print(f"[{config_key}] Saved final model to: {model_path}")
        print(
            f"[{config_key}] CV AUC={auc_mean:.3f} (+/- {auc_std:.3f}), "
            f"EV/trade={ev_mean:.2f}, best_thr≈{avg_thr:.3f}, trades≈{trades_mean:.0f}"
        )

    if not all_thresholds:
        print("No models trained; no thresholds to save.")
        return

    thr_path = save_thresholds_json(all_thresholds, output_dir)
    print(f"Saved thresholds to: {thr_path}")


if __name__ == "__main__":
    main()
