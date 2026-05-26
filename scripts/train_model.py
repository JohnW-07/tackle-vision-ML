"""
Train a multi-output regression model to predict tackle safety scores.

Three approaches compared via Leave-One-Out CV:

  1. Baseline   — Ridge(α=10), all 164 features
  2. Reduced    — RidgeCV(tuned α), mean + at_poc + scalar features only (~35 cols)
  3. Ranker     — Pairwise linear ranker (210 pairs from 21 clips); evaluated by
                  Kendall's τ and converted to predicted scores via global ranking

Usage:
  python scripts/train_model.py --features out/features.csv --output out/model.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_COLS = [
    "label_head_neck",
    "label_upper_extremity",
    "label_com_spine",
    "label_lower_extremity",
    "label_overall_avg",
]
MIN_COVERAGE   = 0.40
RIDGE_ALPHAS   = [0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000]
DROP_STAT_SUFFIXES = ("_std", "_min", "_max", "_p10", "_p90")


# ---------------------------------------------------------------------------
# Data loading & feature selection
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, index_col="video_id")
    feature_cols = [c for c in df.columns if not c.startswith("label_")]
    target_cols  = [c for c in TARGET_COLS if c in df.columns]
    X = df[feature_cols].copy()
    Y = df[target_cols].copy()
    coverage = X.notna().mean()
    X = X[coverage[coverage >= MIN_COVERAGE].index]
    return X, Y


def reduce_features(X: pd.DataFrame) -> pd.DataFrame:
    """Keep only _mean, _at_poc, and scalar (no stat suffix) columns."""
    keep = [c for c in X.columns
            if not any(c.endswith(s) for s in DROP_STAT_SUFFIXES)]
    return X[keep]


# ---------------------------------------------------------------------------
# LOO evaluation (regression)
# ---------------------------------------------------------------------------

def loo_evaluate(X: pd.DataFrame, Y: pd.DataFrame, pipeline, name: str) -> dict:
    loo   = LeaveOneOut()
    preds = np.full((len(Y), len(Y.columns)), np.nan)
    for train_idx, test_idx in loo.split(X):
        pipeline.fit(X.iloc[train_idx], Y.iloc[train_idx])
        p = pipeline.predict(X.iloc[test_idx])
        preds[test_idx] = p if p.ndim == 2 else p.reshape(1, -1)

    results = {}
    rows = []
    for i, col in enumerate(Y.columns):
        y_true = Y.values[:, i]
        y_pred = preds[:, i]
        mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() < 2:
            continue
        mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
        ss_tot = np.sum((y_true[mask] - y_true[mask].mean()) ** 2)
        r2  = float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
        results[col] = {"mae": mae, "r2": r2}
        rows.append((col, mae, r2))

    overall_mae = float(np.mean([r["mae"] for r in results.values()])) if results else None
    overall_r2  = float(np.mean([r["r2"]  for r in results.values()])) if results else None
    results["_overall"] = {"mae": overall_mae, "r2": overall_r2}

    _print_results(name, rows, overall_mae, overall_r2, X.shape[1])
    return results


def _print_results(name, rows, overall_mae, overall_r2, n_features):
    print(f"\n{'─'*64}")
    print(f"  {name}  [{n_features} features]")
    print(f"{'─'*64}")
    print(f"  {'Target':<28} {'MAE':>6}  {'R²':>7}")
    print(f"  {'─'*28} {'─'*6}  {'─'*7}")
    for col, mae, r2 in rows:
        marker = " ✓" if r2 > 0 else ""
        print(f"  {col:<28} {mae:>6.3f}  {r2:>7.3f}{marker}")
    print(f"  {'─'*28} {'─'*6}  {'─'*7}")
    print(f"  {'OVERALL':<28} {overall_mae:>6.3f}  {overall_r2:>7.3f}")


# ---------------------------------------------------------------------------
# Pairwise ranker
# ---------------------------------------------------------------------------

def _fit_ranker(X_arr: np.ndarray, y_arr: np.ndarray):
    """
    Fit a linear scoring function from pairwise comparisons.
    Returns weight vector w such that score(x) = w @ x.
    """
    diffs, labels = [], []
    for i, j in combinations(range(len(y_arr)), 2):
        d = y_arr[i] - y_arr[j]
        if abs(d) < 1e-6:
            continue
        diff = X_arr[i] - X_arr[j]
        diffs.append(diff)
        labels.append(1 if d > 0 else 0)
        diffs.append(-diff)
        labels.append(0 if d > 0 else 1)

    if len(set(labels)) < 2 or len(diffs) < 4:
        return None
    lr = LogisticRegression(C=0.05, max_iter=2000, solver="lbfgs")
    lr.fit(np.array(diffs), labels)
    return lr.coef_[0]


def loo_ranker(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """
    LOO-CV for the pairwise ranker.
    For each held-out clip i:
      - Fit ranker on the remaining 20 clips
      - Score all 21 clips using the learned weights
      - Report Kendall's τ between predicted and true ranking
    Also converts scores → predicted values (linear rescale) for MAE.
    """
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp   = scaler.fit_transform(imputer.fit_transform(X))

    loo    = LeaveOneOut()
    results = {}
    rows    = []

    for i, col in enumerate(Y.columns):
        y_true     = Y.values[:, i]
        valid_mask = ~np.isnan(y_true)
        if valid_mask.sum() < 4:
            continue

        loo_scores = np.full(len(y_true), np.nan)

        for train_idx, test_idx in loo.split(X_imp):
            train_mask = valid_mask[train_idx]
            if train_mask.sum() < 4:
                continue
            w = _fit_ranker(X_imp[train_idx][train_mask], y_true[train_idx][train_mask])
            if w is None:
                continue
            loo_scores[test_idx] = float((X_imp[test_idx] @ w).item())

        # Only evaluate where we got scores and have true labels
        eval_mask = ~np.isnan(loo_scores) & valid_mask
        if eval_mask.sum() < 4:
            continue

        tau, _ = kendalltau(y_true[eval_mask], loo_scores[eval_mask])

        # Convert rank scores → predicted labels via linear rescale for MAE
        scores = loo_scores[eval_mask]
        y_sub  = y_true[eval_mask]
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            y_hat = y_sub.min() + (scores - lo) / (hi - lo) * (y_sub.max() - y_sub.min())
        else:
            y_hat = np.full_like(scores, y_sub.mean())
        mae = float(np.mean(np.abs(y_sub - y_hat)))

        results[col] = {"mae": mae, "tau": float(tau)}
        rows.append((col, mae, tau))

    overall_mae = float(np.mean([r["mae"] for r in results.values()])) if results else None
    overall_tau = float(np.mean([r["tau"] for r in results.values()])) if results else None
    results["_overall"] = {"mae": overall_mae, "tau": overall_tau}

    n_pairs = len(Y) * (len(Y) - 1) // 2
    print(f"\n{'─'*64}")
    print(f"  Pairwise Ranker  [{X.shape[1]} features, {n_pairs} pairs]")
    print(f"{'─'*64}")
    print(f"  {'Target':<28} {'MAE':>6}  {'τ (Kendall)':>11}")
    print(f"  {'─'*28} {'─'*6}  {'─'*11}")
    for col, mae, tau in rows:
        marker = " ✓" if tau > 0 else ""
        print(f"  {col:<28} {mae:>6.3f}  {tau:>11.3f}{marker}")
    print(f"  {'─'*28} {'─'*6}  {'─'*11}")
    print(f"  {'OVERALL':<28} {overall_mae:>6.3f}  {overall_tau:>11.3f}")
    print(f"  (τ = 1.0 → perfect ranking,  τ = 0 → random)")

    return results


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def build_baseline_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   MultiOutputRegressor(Ridge(alpha=10.0))),
    ])


def build_ridgecv_pipeline() -> Pipeline:
    """Per-target Ridge with alpha selected by inner LOO-CV."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   MultiOutputRegressor(
            RidgeCV(alphas=RIDGE_ALPHAS, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
        )),
    ])


# ---------------------------------------------------------------------------
# Feature importances (Ridge coefficients)
# ---------------------------------------------------------------------------

def print_ridge_weights(X: pd.DataFrame, Y: pd.DataFrame, top_n: int = 6) -> None:
    pipe = build_ridgecv_pipeline()
    pipe.fit(X, Y)
    print(f"\n{'─'*64}")
    print(f"  Top {top_n} features per target (RidgeCV |coef|)")
    print(f"{'─'*64}")
    feat = X.columns.tolist()
    for i, col in enumerate(Y.columns):
        est    = pipe.named_steps["model"].estimators_[i]
        coefs  = est.coef_
        top    = np.argsort(np.abs(coefs))[::-1][:top_n]
        alpha  = est.alpha_
        print(f"\n  {col}  [best α={alpha:.1f}]")
        for rank, idx in enumerate(top, 1):
            sign = "+" if coefs[idx] >= 0 else "-"
            print(f"    {rank}. {sign} {feat[idx]:<52} {coefs[idx]:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train multi-output tackle safety regressor.")
    p.add_argument("--features", type=Path, default=Path("out/features.csv"))
    p.add_argument("--output",   type=Path, default=Path("out/model.pkl"))
    args = p.parse_args(argv)

    print(f"Loading {args.features} …")
    X_full, Y = load_data(args.features)
    X_reduced  = reduce_features(X_full)

    print(f"\nFeatures — full: {X_full.shape[1]}  reduced: {X_reduced.shape[1]}")
    print(f"Samples: {len(Y)}")
    print(f"\nTarget stats:")
    for col in Y.columns:
        v = Y[col].dropna()
        print(f"  {col:<30} n={len(v)}  mean={v.mean():.2f}  "
              f"std={v.std():.2f}  range=[{v.min():.2f}, {v.max():.2f}]")

    print("\n\n══ Approach 1: Baseline (Ridge α=10, all features) ══")
    baseline_results = loo_evaluate(X_full, Y, build_baseline_pipeline(), "Baseline Ridge")

    print("\n\n══ Approach 2: Reduced features + RidgeCV (tuned α) ══")
    reduced_results = loo_evaluate(X_reduced, Y, build_ridgecv_pipeline(), "RidgeCV reduced")

    print("\n\n══ Approach 3: Pairwise ranker (reduced features) ══")
    ranker_results = loo_evaluate(X_reduced, Y, build_ridgecv_pipeline(), "sanity check")
    ranker_results = loo_ranker(X_reduced, Y)

    # Summary comparison
    print(f"\n\n{'═'*64}")
    print("  SUMMARY — LOO-CV overall MAE  (lower is better)")
    print(f"{'═'*64}")
    b_mae = baseline_results.get("_overall", {}).get("mae", 999)
    r_mae = reduced_results.get("_overall",  {}).get("mae", 999)
    rk_mae = ranker_results.get("_overall", {}).get("mae", 999)
    rk_tau = ranker_results.get("_overall", {}).get("tau", 0)
    best_mae = min(b_mae, r_mae, rk_mae)
    for label, mae, extra in [
        ("Baseline Ridge (164 feat)", b_mae,  ""),
        ("RidgeCV reduced (35 feat)", r_mae,  ""),
        (f"Pairwise ranker",          rk_mae, f"  τ={rk_tau:.3f}"),
    ]:
        marker = " ← best" if abs(mae - best_mae) < 1e-6 else ""
        print(f"  {label:<32} MAE={mae:.3f}{extra}{marker}")
    print()

    # Print weights for the best regression model
    best_X = X_reduced if r_mae <= b_mae else X_full
    print_ridge_weights(best_X, Y)

    # Save best regression model (Ridge or RidgeCV reduced)
    if r_mae <= b_mae:
        final_pipe = build_ridgecv_pipeline()
        final_pipe.fit(X_reduced, Y)
        saved_features = X_reduced.columns.tolist()
        model_name = "RidgeCV-reduced"
    else:
        final_pipe = build_baseline_pipeline()
        final_pipe.fit(X_full, Y)
        saved_features = X_full.columns.tolist()
        model_name = "Ridge-baseline"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump({
            "model":        final_pipe,
            "feature_cols": saved_features,
            "target_cols":  Y.columns.tolist(),
            "model_name":   model_name,
        }, f)

    results_path = args.output.with_suffix(".results.json")
    results_path.write_text(json.dumps(
        {"baseline": baseline_results, "reduced": reduced_results,
         "ranker": ranker_results, "best": model_name},
        indent=2,
    ))

    print(f"\nSaved model   → {args.output}  ({model_name})")
    print(f"Saved results → {results_path}")


if __name__ == "__main__":
    main()
