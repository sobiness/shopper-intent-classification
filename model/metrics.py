"""The six evaluation metrics required by the assignment.

Kept in one place so the numbers the Streamlit app shows for an uploaded CSV are
produced by exactly the same code that produced the README comparison table.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Order matches the comparison table in the assignment brief.
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def score_predictions(y_true, y_pred, y_proba) -> dict:
    """Compute all six metrics.

    ``y_proba`` must be the positive-class probability, not a hard label: AUC
    ranks the ordering of scores, so feeding it 0/1 predictions collapses the
    curve to a single operating point and silently understates the model.
    The remaining five metrics are computed from the hard labels.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def predict_with_proba(pipeline, X) -> tuple[np.ndarray, np.ndarray]:
    """Return hard labels and positive-class probabilities for a fitted pipeline."""
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    return y_pred, y_proba


def score_model(pipeline, X, y_true) -> dict:
    """Fitted pipeline plus a feature frame and labels -> the six metrics."""
    y_pred, y_proba = predict_with_proba(pipeline, X)
    return score_predictions(y_true, y_pred, y_proba)


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix as a labelled frame, ready to display."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual: No Purchase", "Actual: Purchase"],
        columns=["Predicted: No Purchase", "Predicted: Purchase"],
    )


def majority_baseline_accuracy(y_true) -> float:
    """Accuracy of always predicting the majority class.

    The floor every model has to beat. On an 84.5/15.5 split this is high enough
    that accuracy alone cannot separate a useful model from a useless one.
    """
    values = pd.Series(np.asarray(y_true).ravel())
    return float(values.value_counts(normalize=True).max())
