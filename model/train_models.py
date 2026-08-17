"""Train and evaluate all six classifiers, then persist them.

Run from the repository root:

    python -m model.train_models

Writes one .joblib per fitted pipeline into model/, refreshes test_data.csv, and
records every number quoted in the README into model/training_results.json so
the write-up can be checked against the run that produced it.
"""

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score

from .config import MODEL_DIR, MODEL_FILENAMES, RESULTS_FILE, TARGET
from .data import export_test_data, load_raw, make_split
from .metrics import (
    METRIC_ORDER,
    majority_baseline_accuracy,
    predict_with_proba,
    score_model,
)
from .pipelines import build_pipelines

# Pairs where a page-count column and its dwell-time column measure the same
# browsing behaviour. Their correlation is the concrete evidence for why Naive
# Bayes' conditional-independence assumption is violated on this dataset.
CORRELATED_PAIRS = [
    ("Administrative", "Administrative_Duration"),
    ("Informational", "Informational_Duration"),
    ("ProductRelated", "ProductRelated_Duration"),
    ("BounceRates", "ExitRates"),
]


def describe_dataset(df: pd.DataFrame, y_train, y_test) -> dict:
    """Shape, balance and split-integrity facts quoted in the README."""
    target_share = df[TARGET].astype(bool).mean()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_features": int(df.shape[1] - 1),
        "missing_values": int(df.isna().sum().sum()),
        "positive_rate_full": float(target_share),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "majority_baseline_accuracy": majority_baseline_accuracy(y_test),
    }


def feature_correlations(df: pd.DataFrame) -> dict:
    """Pearson r for each count/duration pair."""
    return {
        f"{left} vs {right}": float(df[left].corr(df[right]))
        for left, right in CORRELATED_PAIRS
    }


def top_feature_importances(pipeline, n: int = 10) -> dict:
    """Random Forest importances, mapped back to post-encoding feature names."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    ranked = (
        pd.Series(importances, index=names).sort_values(ascending=False).head(n)
    )
    return {name: float(value) for name, value in ranked.items()}


def train_all() -> dict:
    """Fit every pipeline, score it on the held-out split, save it to disk."""
    df = load_raw()
    X_train, X_test, y_train, y_test = make_split(df)
    export_test_data(X_test, y_test)

    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Train {len(y_train)} rows (positive rate {np.mean(y_train):.4f})")
    print(f"Test  {len(y_test)} rows (positive rate {np.mean(y_test):.4f})")
    print(f"Majority-class baseline accuracy on test: "
          f"{majority_baseline_accuracy(y_test):.4f}\n")

    results = {}
    fitted = {}
    for name, pipeline in build_pipelines().items():
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        scores = score_model(pipeline, X_test, y_test)
        train_labels, _ = predict_with_proba(pipeline, X_train)
        train_accuracy = accuracy_score(y_train, train_labels)

        results[name] = {
            **scores,
            "TrainAccuracy": float(train_accuracy),
            "OverfitGap": float(train_accuracy - scores["Accuracy"]),
            "FitSeconds": round(fit_seconds, 3),
        }
        fitted[name] = pipeline

        # Compressed: an uncompressed 300-tree forest is ~55 MB, which is slow to
        # clone and close to GitHub's per-file warning threshold.
        joblib.dump(pipeline, MODEL_DIR / MODEL_FILENAMES[name], compress=3)
        print(
            f"{name:<26} "
            + "  ".join(f"{key} {scores[key]:.4f}" for key in METRIC_ORDER)
            + f"  (fit {fit_seconds:.1f}s)"
        )

    encoded_features = fitted["Logistic Regression"].named_steps[
        "preprocess"
    ].get_feature_names_out()

    return {
        "dataset": describe_dataset(df, y_train, y_test),
        "n_features_after_encoding": int(len(encoded_features)),
        "metrics": results,
        "correlations": feature_correlations(df),
        "random_forest_top_features": top_feature_importances(
            fitted["Random Forest (Ensemble)"]
        ),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def comparison_table(results: dict) -> pd.DataFrame:
    """The README comparison table, rounded for presentation."""
    frame = pd.DataFrame(results["metrics"]).T[METRIC_ORDER]
    return frame.round(4)


def main() -> None:
    results = train_all()
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")

    print("\nComparison table")
    print(comparison_table(results).to_string())
    print("\nCount/duration correlations")
    for pair, value in results["correlations"].items():
        print(f"  {pair}: r = {value:.3f}")
    print("\nTop Random Forest features")
    for name, value in results["random_forest_top_features"].items():
        print(f"  {name}: {value:.4f}")
    print(f"\nSaved {len(results['metrics'])} pipelines to {MODEL_DIR}")
    print(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
