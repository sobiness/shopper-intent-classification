"""Model definitions.

Every estimator is wrapped in a Pipeline whose first step is the shared
preprocessor, so scaling and one-hot encoding are fitted on the training fold
only and the same transformation is replayed automatically at predict time.

Only stock scikit-learn components are used. That keeps the pickled pipelines
loadable in the Streamlit app without importing this module.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    """Scale the numeric columns, one-hot the nominal ones.

    ``sparse_output=False`` is required because GaussianNB cannot consume a
    sparse matrix. ``handle_unknown="ignore"`` means an uploaded CSV containing
    a category level absent from the training split degrades to an all-zero
    dummy block instead of raising.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_estimators() -> dict:
    """The six classifiers, at deliberately near-default settings.

    No class weighting anywhere: the 84.5/15.5 imbalance is left untouched so
    that the gap between accuracy and MCC stays visible in the results table.
    The decision tree is left unconstrained for the same reason, to expose the
    train/test gap rather than tune it away.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        # SVC exposes no probabilities of its own, and AUC needs them. Platt
        # scaling via CalibratedClassifierCV is the supported way to get them;
        # SVC(probability=True) is deprecated from scikit-learn 1.9.
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False
        ),
    }


def build_pipelines() -> dict:
    """Map model name -> unfitted Pipeline(preprocess -> classifier)."""
    return {
        name: Pipeline(
            steps=[("preprocess", build_preprocessor()), ("classifier", estimator)]
        )
        for name, estimator in build_estimators().items()
    }
