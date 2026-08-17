"""Shared paths, column groupings and split settings.

Imported by the training script, the notebook and the Streamlit app so that all
three agree on how the 17 features are treated.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
RAW_DATA_FILE = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
TEST_DATA_FILE = PROJECT_ROOT / "test_data.csv"
RESULTS_FILE = MODEL_DIR / "training_results.json"

TARGET = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Genuinely continuous or ordinal. SpecialDay belongs here because its 0.0-1.0
# values encode closeness to a special day, so the ordering is meaningful.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Nominal. OperatingSystems, Browser, Region and TrafficType arrive as integers
# but the codes are labels, not quantities: Browser 4 is not twice Browser 2.
# One-hot encoding stops the linear and distance-based models from reading an
# ordering that does not exist.
CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
    "Support Vector Machine": "svm.joblib",
}
