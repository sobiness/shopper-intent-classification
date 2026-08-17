"""Streamlit front end for the shopper purchase-intent classifiers.

Loads the six pipelines fitted by ``python -m model.train_models`` and scores
whatever test CSV the user supplies. Nothing is trained here: training on every
page load would be far too slow for the Community Cloud free tier.
"""

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, roc_curve

from model.config import FEATURES, MODEL_DIR, MODEL_FILENAMES, TARGET
from model.data import coerce_target, load_test_data
from model.metrics import (
    METRIC_ORDER,
    confusion_frame,
    majority_baseline_accuracy,
    score_predictions,
)

PALETTE = {
    "ink": "#12103a",
    "accent": "#5b4bff",
    "accent_soft": "#eceafe",
    "positive": "#0f9d76",
    "negative": "#d1495b",
    "muted": "#6b7280",
}

METRIC_HELP = {
    "Accuracy": "Share of sessions classified correctly. Inflated here by the "
                "84.5% no-purchase majority.",
    "AUC": "Ranking quality across every threshold, computed from predicted "
           "probabilities. Unaffected by the threshold slider.",
    "Precision": "Of the sessions flagged as purchases, how many really were.",
    "Recall": "Of the real purchases, how many were caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between predictions and truth, using all four cells of "
           "the confusion matrix. The trustworthy summary on imbalanced data.",
}


st.set_page_config(
    page_title="Shopper Purchase Intent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .hero {{
            background: linear-gradient(115deg, {PALETTE['ink']} 0%, {PALETTE['accent']} 100%);
            padding: 1.6rem 1.9rem;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 1.4rem;
        }}
        .hero h1 {{ margin: 0; font-size: 1.85rem; letter-spacing: -0.02em; }}
        .hero p {{ margin: 0.45rem 0 0 0; opacity: 0.82; font-size: 0.96rem; }}
        .metric-card {{
            background: #ffffff;
            border: 1px solid #e6e6f0;
            border-left: 4px solid {PALETTE['accent']};
            border-radius: 11px;
            padding: 0.85rem 1rem;
            height: 100%;
        }}
        .metric-card .label {{
            font-size: 0.72rem; text-transform: uppercase;
            letter-spacing: 0.09em; color: {PALETTE['muted']};
        }}
        .metric-card .value {{
            font-size: 1.55rem; font-weight: 650; color: {PALETTE['ink']};
            line-height: 1.25;
        }}
        .note {{
            background: {PALETTE['accent_soft']};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 0.88rem;
            color: {PALETTE['ink']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading fitted pipelines…")
def load_models() -> dict:
    """Unpickle every saved pipeline once per server process."""
    models = {}
    for name, filename in MODEL_FILENAMES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_data(show_spinner=False)
def read_upload(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


@st.cache_data(show_spinner=False)
def sample_data() -> pd.DataFrame:
    return load_test_data()


@st.cache_data(show_spinner="Scoring models…")
def predict_probabilities(features: pd.DataFrame, model_names: tuple) -> dict:
    """Positive-class probabilities per model.

    Cached on the feature frame so moving the threshold slider only re-derives
    labels; it never re-runs the models.
    """
    models = load_models()
    return {
        name: models[name].predict_proba(features)[:, 1] for name in model_names
    }


def validate(df: pd.DataFrame) -> tuple[list, bool]:
    missing = [column for column in FEATURES if column not in df.columns]
    return missing, TARGET in df.columns


def metric_cards(scores: dict) -> None:
    for row_start in (0, 3):
        columns = st.columns(3)
        for column, key in zip(columns, METRIC_ORDER[row_start:row_start + 3]):
            with column:
                st.markdown(
                    f"<div class='metric-card'><div class='label'>{key}</div>"
                    f"<div class='value'>{scores[key]:.4f}</div></div>",
                    unsafe_allow_html=True,
                )
                st.caption(METRIC_HELP[key])


def plot_confusion(matrix: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(4.6, 3.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        annot_kws={"size": 13, "weight": "bold"},
        ax=axis,
    )
    axis.set_xticklabels(["No Purchase", "Purchase"])
    axis.set_yticklabels(["No Purchase", "Purchase"], rotation=0)
    axis.set_xlabel("Predicted", labelpad=8)
    axis.set_ylabel("Actual", labelpad=8)
    figure.tight_layout()
    return figure


def plot_roc(y_true, probabilities: dict, highlight: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5.2, 3.9))
    for name, proba in probabilities.items():
        false_positive, true_positive, _ = roc_curve(y_true, proba)
        is_focus = name == highlight
        axis.plot(
            false_positive,
            true_positive,
            linewidth=2.4 if is_focus else 1.1,
            alpha=1.0 if is_focus else 0.45,
            color=PALETTE["accent"] if is_focus else PALETTE["muted"],
            label=name if is_focus else None,
        )
    axis.plot([0, 1], [0, 1], "--", color="#c9c9d6", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"ROC — {highlight} highlighted", fontsize=10)
    axis.legend(loc="lower right", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def main() -> None:
    inject_styles()
    models = load_models()

    st.markdown(
        "<div class='hero'><h1>Online Shopper Purchase Intent</h1>"
        "<p>Six classifiers trained on 12,330 e-commerce sessions from the UCI "
        "Online Shoppers Purchasing Intention dataset. Upload a labelled test "
        "CSV to score them side by side.</p></div>",
        unsafe_allow_html=True,
    )

    if not models:
        st.error(
            "No fitted models found in `model/`. Run `python -m model.train_models` "
            "and commit the generated .joblib files."
        )
        st.stop()

    # ---------------- sidebar: data source, model, threshold ----------------
    st.sidebar.header("1 · Test data")
    upload = st.sidebar.file_uploader(
        "Upload a test CSV",
        type="csv",
        help="Needs the 17 feature columns. Include the Revenue column to get "
             "evaluation metrics.",
    )

    if upload is not None:
        df = read_upload(upload.getvalue())
        source = f"Uploaded · {upload.name}"
    else:
        df = sample_data()
        source = "Bundled held-out split · test_data.csv"
        st.sidebar.caption(
            "No file uploaded, so the repository's `test_data.csv` is being used."
        )

    missing, has_target = validate(df)
    if missing:
        st.error(f"That CSV is missing {len(missing)} required column(s): {missing}")
        st.stop()

    st.sidebar.header("2 · Model")
    model_name = st.sidebar.selectbox(
        "Classifier", list(models.keys()), index=len(models) - 2
    )

    st.sidebar.header("3 · Decision threshold")
    threshold = st.sidebar.slider(
        "Flag a session as a purchase when P(purchase) ≥",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="0.50 reproduces the README table. Raising or lowering it trades "
             "precision against recall without refitting anything.",
    )

    st.sidebar.divider()
    st.sidebar.caption(f"**Rows loaded:** {len(df):,}\n\n**Source:** {source}")

    features = df[FEATURES]
    probabilities = predict_probabilities(features, tuple(models.keys()))

    if not has_target:
        st.warning(
            f"No `{TARGET}` column found, so metrics cannot be computed. Showing "
            "predictions only — re-upload with the label column for a full report."
        )
        predictions = pd.DataFrame(
            {
                "P(purchase)": probabilities[model_name].round(4),
                "Prediction": np.where(
                    probabilities[model_name] >= threshold, "Purchase", "No Purchase"
                ),
            }
        )
        st.dataframe(predictions, width="stretch", height=420)
        st.stop()

    y_true = coerce_target(df[TARGET])
    labels = {
        name: (proba >= threshold).astype(int)
        for name, proba in probabilities.items()
    }
    scores = {
        name: score_predictions(y_true, labels[name], probabilities[name])
        for name in models
    }
    baseline = majority_baseline_accuracy(y_true)

    report_tab, compare_tab, data_tab = st.tabs(
        ["Model report", "Compare all models", "Test data"]
    )

    # ---------------------------- model report ----------------------------
    with report_tab:
        st.subheader(f"{model_name} — evaluation metrics")
        st.markdown(
            f"<div class='note'>Scored on <b>{len(df):,}</b> sessions at a "
            f"threshold of <b>{threshold:.2f}</b>. Always predicting "
            f"“no purchase” would already score <b>{baseline:.4f}</b> accuracy on "
            "this file, so accuracy alone proves very little — read MCC and AUC "
            "alongside it.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        metric_cards(scores[model_name])

        st.divider()
        left, right = st.columns([1, 1.15])
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(
                plot_confusion(confusion_frame(y_true, labels[model_name])),
                width="stretch",
            )
        with right:
            st.markdown("**ROC curve**")
            st.pyplot(
                plot_roc(y_true, probabilities, model_name),
                width="stretch",
            )

        st.markdown("**Classification report**")
        report = classification_report(
            y_true,
            labels[model_name],
            target_names=["No Purchase", "Purchase"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).T.round(4), width="stretch"
        )

    # -------------------------- compare all models --------------------------
    with compare_tab:
        st.subheader("All six classifiers on this file")
        table = pd.DataFrame(scores).T[METRIC_ORDER].round(4)
        st.dataframe(
            table.style.highlight_max(axis=0, color=PALETTE["accent_soft"]),
            width="stretch",
        )

        st.markdown(
            "**Accuracy vs MCC** — the gap between the two bars is the whole "
            "argument for not ranking models on accuracy when 84.5% of sessions "
            "end without a purchase."
        )
        st.bar_chart(table[["Accuracy", "MCC"]], height=320)

        winner = table["MCC"].idxmax()
        st.success(
            f"Best MCC on this file: **{winner}** "
            f"({table.loc[winner, 'MCC']:.4f} MCC, "
            f"{table.loc[winner, 'AUC']:.4f} AUC)."
        )

    # ------------------------------ test data ------------------------------
    with data_tab:
        st.subheader("What was scored")
        counts = y_true.value_counts().sort_index()
        left, middle, right = st.columns(3)
        left.metric("Sessions", f"{len(df):,}")
        middle.metric("Purchases", f"{int(counts.get(1, 0)):,}")
        right.metric("Positive rate", f"{y_true.mean():.2%}")
        st.dataframe(df.head(200), width="stretch", height=430)
        st.caption("First 200 rows shown.")


if __name__ == "__main__":
    main()
