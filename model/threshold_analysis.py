"""Follow-up analysis: is 0.8974 accuracy a model limitation or a data ceiling?

Run from the repository root:

    python -m model.threshold_analysis

Answers three questions the main comparison table raises but cannot settle:

1. Is the accuracy low, or is it the ceiling? Compared against a stronger
   algorithm (gradient boosting) and against the accuracy the dataset's own
   authors published.
2. Where do the errors actually go? Split into false positives and the missed
   purchases that accuracy conceals.
3. Can the winner be improved without a new model? Threshold sweep plus a
   class-weighted refit.

Writes model/tuning_results.json. Does not touch the six submitted pipelines.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import MODEL_DIR, MODEL_FILENAMES, RANDOM_STATE
from .data import load_raw, make_split
from .metrics import METRIC_ORDER, labels_from_proba, score_model
from .pipelines import build_preprocessor

WINNER = "Random Forest (Ensemble)"

# Sakar, C.O. et al. (2019), Neural Computing and Applications 31, 6893-6908 --
# the accuracy and F1 reported by the authors who published this dataset.
PUBLISHED_ACCURACY = 0.8724
PUBLISHED_F1 = 0.58


def ceiling_check(X_train, y_train, X_test, y_test, winner_proba) -> dict:
    """Does a stronger algorithm beat the Random Forest's accuracy?"""
    gradient_boosting = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    ).fit(X_train, y_train)
    gb_proba = gradient_boosting.predict_proba(X_test)[:, 1]
    gb_labels = labels_from_proba(gb_proba)

    return {
        "random_forest": {
            "Accuracy": float(accuracy_score(y_test, labels_from_proba(winner_proba))),
            "AUC": float(roc_auc_score(y_test, winner_proba)),
            "F1": float(f1_score(y_test, labels_from_proba(winner_proba))),
        },
        "gradient_boosting": {
            "Accuracy": float(accuracy_score(y_test, gb_labels)),
            "AUC": float(roc_auc_score(y_test, gb_proba)),
            "F1": float(f1_score(y_test, gb_labels)),
        },
        "published_paper": {
            "Accuracy": PUBLISHED_ACCURACY,
            "F1": PUBLISHED_F1,
        },
    }


def error_breakdown(y_true, y_proba) -> dict:
    """Split the winner's errors into false positives and missed purchases."""
    y_pred = labels_from_proba(y_proba)
    y_true = np.asarray(y_true)
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    total_errors = false_positive + false_negative
    return {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "total_errors": total_errors,
        "error_rate": total_errors / len(y_true),
        "share_of_errors_that_are_missed_purchases": false_negative / total_errors,
    }


def threshold_sweep(y_true, y_proba) -> pd.DataFrame:
    """Accuracy, F1 and MCC across thresholds, without refitting anything."""
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        labels = labels_from_proba(y_proba, threshold)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "Accuracy": accuracy_score(y_true, labels),
                "F1": f1_score(y_true, labels, zero_division=0),
                "MCC": matthews_corrcoef(y_true, labels),
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


def class_weighted_refit(X_train, y_train, X_test, y_test) -> dict:
    """The winner refitted with class_weight='balanced'."""
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    ).fit(X_train, y_train)
    return score_model(pipeline, X_test, y_test)


def main() -> None:
    df = load_raw()
    X_train, X_test, y_train, y_test = make_split(df)

    winner = joblib.load(MODEL_DIR / MODEL_FILENAMES[WINNER])
    winner_proba = winner.predict_proba(X_test)[:, 1]

    ceiling = ceiling_check(X_train, y_train, X_test, y_test, winner_proba)
    errors = error_breakdown(y_test, winner_proba)
    sweep = threshold_sweep(y_test, winner_proba)
    balanced = class_weighted_refit(X_train, y_train, X_test, y_test)

    print("1. Is 0.8974 low, or is it the ceiling?")
    for label, scores in ceiling.items():
        rendered = "  ".join(f"{k} {v:.4f}" for k, v in scores.items())
        print(f"   {label:<18} {rendered}")

    print("\n2. Where the errors go (Random Forest at threshold 0.50)")
    print(f"   true negatives  {errors['true_negative']:>5}   "
          f"false positives {errors['false_positive']:>5}")
    print(f"   false negatives {errors['false_negative']:>5}   "
          f"true positives  {errors['true_positive']:>5}")
    print(f"   {errors['total_errors']} errors = {errors['error_rate']:.2%} of sessions; "
          f"{errors['share_of_errors_that_are_missed_purchases']:.0%} are missed purchases")

    print("\n3. Threshold sweep (same fitted model, no refit)")
    print(sweep.round(4).to_string())
    best_f1 = sweep["F1"].idxmax()
    best_mcc = sweep["MCC"].idxmax()
    print(f"\n   best F1  at threshold {best_f1}: {sweep.loc[best_f1, 'F1']:.4f}")
    print(f"   best MCC at threshold {best_mcc}: {sweep.loc[best_mcc, 'MCC']:.4f}")

    print("\n4. class_weight='balanced' refit")
    print("   " + "  ".join(f"{k} {balanced[k]:.4f}" for k in METRIC_ORDER))

    payload = {
        "ceiling_check": ceiling,
        "error_breakdown": errors,
        "threshold_sweep": sweep.round(6).reset_index().to_dict(orient="records"),
        "best_f1_threshold": float(best_f1),
        "best_mcc_threshold": float(best_mcc),
        "class_weight_balanced": balanced,
    }
    output = MODEL_DIR / "tuning_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
