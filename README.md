# Online Shopper Purchase Intent — Classification & Streamlit App

**M.Tech (AIML/DSE) · Machine Learning · Assignment 2**

Six classifiers predict whether an e-commerce browsing session ends in a purchase, evaluated on Accuracy, AUC, Precision, Recall, F1 and MCC, and served through an interactive Streamlit app.

| | |
|---|---|
| **Live Streamlit app** | _paste the Community Cloud URL here after deploying_ |
| **GitHub repository** | https://github.com/sobiness/shopper-intent-classification |
| **Dataset** | [UCI ML Repository — Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |

---

## a. Problem statement

An online retailer sees far more browsing than buying. Of the 12,330 sessions in this dataset, only 15.5% end in a transaction. Being able to tell, from clickstream and session metadata alone, which visitors are likely to convert is directly actionable: those sessions can be targeted with a discount, a live-chat prompt, or a retargeting ad, while the rest are left alone.

Framed as supervised learning, this is a **binary classification** problem. Given 17 features describing one session — how many pages of each category were viewed, how long was spent on them, bounce and exit rates, the month, the visitor's browser and region, and whether they are a returning visitor — predict the boolean `Revenue` label: did this session end in a purchase?

The interesting difficulty is not the modelling, it is the **class imbalance**. With 84.5% negatives, a model that predicts "no purchase" every single time is already 84.5% accurate while being completely useless. Accuracy alone therefore cannot rank models here, which is precisely why the assignment asks for six metrics rather than one. The analysis below leans on **MCC** and **AUC** to do the real ranking, and treats accuracy as a sanity check against that 0.8451 floor.

---

## b. Dataset description

**Source:** UCI Machine Learning Repository, dataset 468 — *Online Shoppers Purchasing Intention Dataset*.
**Reference:** Sakar, C.O., Polat, S.O., Katircioglu, M. & Kastro, Y. (2019). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31, 6893–6908.

| Property | Value |
|---|---|
| Instances | **12,330** sessions (assignment minimum: 500) |
| Features | **17** (assignment minimum: 12) |
| Target | `Revenue` — boolean, purchase vs no purchase |
| Class balance | 10,422 negative (84.53%) / 1,908 positive (**15.47%**) |
| Missing values | **0** — no imputation needed |
| Feature types | 10 numeric, 7 categorical |
| Features after one-hot encoding | 75 |

Each row is one user session, collected over a one-year period and deduplicated so that no single user contributes a disproportionate number of sessions.

### The 10 numeric features

| Feature | Meaning |
|---|---|
| `Administrative` | Number of account/admin pages viewed |
| `Administrative_Duration` | Seconds spent on those pages |
| `Informational` | Number of informational pages viewed |
| `Informational_Duration` | Seconds spent on those pages |
| `ProductRelated` | Number of product pages viewed |
| `ProductRelated_Duration` | Seconds spent on those pages |
| `BounceRates` | Average bounce rate of the pages visited |
| `ExitRates` | Average exit rate of the pages visited |
| `PageValues` | Average Google Analytics page value of the pages visited |
| `SpecialDay` | Closeness of the session date to a special day, 0.0–1.0 |

### The 7 categorical features

| Feature | Levels | Note |
|---|---|---|
| `Month` | 10 | January and April are absent from the data |
| `VisitorType` | 3 | `New_Visitor`, `Returning_Visitor`, `Other` |
| `Weekend` | 2 | Boolean |
| `OperatingSystems` | 8 | Integer-coded labels, **not** ordinal |
| `Browser` | 13 | Integer-coded labels, **not** ordinal |
| `Region` | 9 | Integer-coded labels, **not** ordinal |
| `TrafficType` | 20 | Integer-coded labels, **not** ordinal |

### Preprocessing decisions, and why

- **`SpecialDay` is kept numeric.** Its 0.0–1.0 values encode proximity to a special day, so the ordering is genuine, unlike the other integer-coded columns.
- **`OperatingSystems`, `Browser`, `Region` and `TrafficType` are one-hot encoded** even though they arrive as integers. Browser 4 is not "twice" Browser 2 — the integers are labels. Left raw, Logistic Regression would fit a single coefficient across a meaningless ordering, and kNN would treat unrelated categories as near neighbours. `TrafficType` alone contributes 20 dummy columns, which is most of the growth from 17 to 75 features.
- **The numeric columns are standardised.** `ProductRelated_Duration` reaches into the tens of thousands of seconds while `BounceRates` is bounded in [0, 0.2]. Unscaled, kNN's Euclidean distance is decided almost entirely by the duration columns, and Logistic Regression struggles to converge.
- **Scaling and encoding live inside each `Pipeline`**, so they are fitted on the training fold only. Fitting a scaler on the full dataset before splitting would leak test-set distribution into training — a subtle but real form of data leakage.
- **Gaussian, not Multinomial, Naive Bayes.** Multinomial NB requires non-negative count-like inputs; standardisation deliberately produces negative values, so Gaussian NB is the only valid choice of the two here.
- **Stratified 80/20 split** with `random_state=42`: 9,864 train / 2,466 test, positive rate 0.1547 vs 0.1549. At this level of imbalance an unstratified split can shift the positive rate between the two halves and quietly move every metric.
- **No class weighting, and the Decision Tree is left unconstrained.** Both are deliberate. Rebalancing would hide the accuracy-vs-MCC divergence that is the most instructive result in this dataset, and pruning the tree would hide its overfitting gap.

`test_data.csv` in this repository is exactly the held-out 20% (2,466 rows) **with the `Revenue` column included**, so the app can compute metrics on it.

---

## c. GitHub repository link

**https://github.com/sobiness/shopper-intent-classification**

```
shopper-intent-classification/
├── app.py                          # Streamlit application
├── requirements.txt                # pinned to the versions the pickles were fitted with
├── README.md
├── test_data.csv                   # held-out 20% (2,466 rows), target included
├── data/
│   └── online_shoppers_intention.csv   # full UCI dataset, as downloaded
└── model/
    ├── ML_Assignment_2.ipynb       # end-to-end analysis notebook (executed, with outputs)
    ├── config.py                   # column groupings and split settings
    ├── data.py                     # loading, target coercion, stratified split
    ├── pipelines.py                # preprocessing + the six estimators
    ├── metrics.py                  # the six evaluation metrics
    ├── train_models.py             # training entry point
    ├── threshold_analysis.py       # follow-up: threshold sweep and ceiling check
    ├── training_results.json       # every number quoted in this README
    ├── tuning_results.json         # output of threshold_analysis.py
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── svm.joblib
```

### Reproducing the results

```bash
pip install -r requirements.txt
python -m model.train_models   # refits all six models, rewrites the pickles and test_data.csv
streamlit run app.py
```

Training the whole set takes a few seconds; the calibrated SVM accounts for almost all of it.

---

## d. Models used

All six models were trained on the same stratified 80/20 split and evaluated on the same 2,466 held-out sessions. AUC is computed from `predict_proba`, never from hard labels; the other five metrics come from labels thresholded at 0.50.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8621 | 0.7410 | 0.5538 | 0.5654 | 0.5596 | 0.4779 |
| kNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes (Gaussian) | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| **Random Forest (Ensemble)** | **0.8974** | **0.9175** | 0.7590 | 0.4948 | **0.5990** | **0.5596** |
| Support Vector Machine _(extra)_ | 0.8877 | 0.8824 | 0.7234 | 0.4450 | 0.5511 | 0.5099 |

> **Majority-class baseline accuracy: 0.8451.** Predicting "no purchase" for all 2,466 test sessions scores this without learning anything. Four of the six models land between 1 and 5 points *above* it, so the accuracy column separates almost nothing — while MCC spreads from 0.13 to 0.56 across the same models. That contrast is the single most important thing in this table.

The Support Vector Machine is included as a sixth model because the brief refers to "6 ML models" while listing five; it is additional to the five required rows, not a substitute for any of them.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong ranker, timid classifier. Its AUC of 0.8828 is second only to the Random Forest and effectively ties the RBF-kernel SVM, so a linear boundary in the 75-dimensional encoded space captures most of the available signal. But at a 0.50 threshold it flags only 7.5% of sessions as purchases when 15.5% actually convert, giving high precision (0.7366) and the weakest recall of any usable model (0.3586) — it misses roughly two of every three real buyers. The gap between its good AUC and mediocre F1 is a threshold problem, not a learning problem. Train/test accuracy gap of 0.005 shows no overfitting whatsoever. |
| **Decision Tree** | The clearest overfitting case here: 100% training accuracy against 86.2% test accuracy, a gap of 0.138, because the tree was left unpruned and grew until every leaf was pure. Its AUC of 0.7410 is the second-worst in the table and deserves explanation — a fully-grown tree assigns every test session a probability of either 0.0 or 1.0, just **2 distinct values**, so its ROC curve has a single interior operating point and there is almost nothing for AUC to measure. Yet it posts the best recall of the non-degenerate models (0.5654) and a respectable MCC of 0.4779. A useful lesson: AUC punishes uncalibrated hard-splitting models even when their labels are decent. |
| **kNN** | Lowest MCC (0.4493) of the five usable models, and the reason is dimensionality. One-hot encoding expands 17 features into 75, and in that space Euclidean distances between sessions become much less discriminative — the classic curse-of-dimensionality problem for distance-based learners. Its AUC (0.7633) is also held down mechanically: with `k=5`, the predicted probability can only take **6 distinct values** (0, 0.2, … 1.0), which is far too coarse to rank 2,466 sessions well. Standardisation was essential; without it the duration columns, which span thousands of seconds, would have dominated every distance calculation. |
| **Naive Bayes (Gaussian)** | Fails dramatically on this dataset, and instructively so. Its 0.2729 accuracy is **57 points below the do-nothing baseline** — it labels 87.4% of sessions as purchases, harvesting 97.4% recall at 0.1726 precision. Two causes compound: first, conditional independence is violated by construction, since `BounceRates`/`ExitRates` correlate at r = 0.913 and `ProductRelated`/`ProductRelated_Duration` at r = 0.861, so the model multiplies what is effectively the same evidence several times over and drives its posteriors to saturation. Second, fitting a Gaussian to 65 sparse one-hot dummy columns is a category error: rare levels have near-zero within-class variance, and each contributes an extreme log-likelihood term. The saturation is severe enough that raising the threshold cannot rescue it — even at 0.99999, accuracy only reaches 0.3094. And yet its AUC is 0.7334, comfortably above chance, which is the real finding: the *ordering* NB produces still carries signal, only its probabilities are unusable. Accuracy and AUC are answering genuinely different questions here. |
| **Random Forest (Ensemble)** | Best model on five of the six metrics — accuracy 0.8974, AUC 0.9175, precision 0.7590, F1 0.5990 and MCC 0.5596 — and the winner by a clear margin rather than a rounding error. Its value shows up most sharply against the single Decision Tree: both reach 100% training accuracy, but bagging plus per-split feature subsampling cuts the generalisation gap from 0.138 to 0.103 and lifts AUC from 0.7410 to 0.9175. Averaging 300 trees also yields 253 distinct probability values instead of 2, which is what makes the ranking usable. The honest caveat is recall of 0.4948: even the best model here still misses about half of all real purchases at the default threshold. |
| **Support Vector Machine** _(extra)_ | Second-best MCC (0.5099) and a solid accuracy of 0.8877, but its AUC of 0.8824 is statistically indistinguishable from Logistic Regression's 0.8828. That is the interesting part: the RBF kernel's extra capacity buys a better decision boundary at the 0.50 threshold — recall improves from 0.3586 to 0.4450 — without improving the underlying ranking at all. It is also by far the most expensive model to fit, 4.1 seconds against 0.03 for Logistic Regression, a more than 100× cost for no gain in AUC. Probabilities come from Platt scaling via `CalibratedClassifierCV`, since `SVC` exposes no probabilities natively and AUC requires them. |
| **Overall winner for this dataset** | **Random Forest (Ensemble).** It leads on MCC (0.5596) *and* AUC (0.9175), which matters because those two metrics fail in different ways — MCC judges the labels at one threshold using all four confusion-matrix cells, AUC judges the ranking across every threshold. Winning both means the result is not an artefact of where the threshold happens to sit, and MCC in particular cannot be gamed by the 84.5% majority class the way accuracy can. It also handles this dataset's specific awkwardness best: mixed feature types, a heavily skewed dominant predictor, and correlated inputs that break Naive Bayes outright. The trade-off is interpretability and size — 300 trees against a single readable tree or a linear model with signed coefficients. If recall were the business priority, none of these models is ready at a 0.50 threshold, and the app's threshold slider exists to make that trade-off explicit. |

### What the Random Forest keys on

| Feature | Gini importance |
|---|---|
| `PageValues` | 0.328 |
| `ProductRelated_Duration` | 0.079 |
| `ExitRates` | 0.079 |
| `ProductRelated` | 0.071 |
| `Administrative_Duration` | 0.054 |

A single feature, `PageValues`, carries roughly a third of the total importance — more than four times the next strongest. That concentration explains much of the results table: one dominant, monotone predictor is exactly the structure tree ensembles exploit best, and it is also why even weak models clear the baseline on accuracy.

It is worth being sceptical about that feature, though. `PageValues` is a Google Analytics metric derived in part from pages that preceded conversions, so it is partially downstream of the very outcome being predicted. The 0.9175 AUC should be read as "this dataset is predictable" rather than "these features cause purchases."

### Is 0.8974 accuracy low? No — but the recall is

An obvious question about the table above is whether ~90% accuracy is a weak result. It is not: it is close to the ceiling for this dataset. Reproduce with `python -m model.threshold_analysis`.

| Reference point | Accuracy | AUC | F1 |
|---|---|---|---|
| Gradient boosting (stronger algorithm, for comparison only) | 0.8974 | 0.9258 | 0.6411 |
| **Our Random Forest** | **0.8974** | 0.9175 | 0.5990 |
| Published result — Sakar et al. (2019), MLP/LSTM | 0.8724 | — | 0.58 |

A `HistGradientBoostingClassifier` — a substantially more powerful algorithm than anything required here — reaches the *same* 0.8974 accuracy, improving only AUC and F1. When a much stronger model cannot move a number, that number is a property of the data rather than of the model. The result is also 2.5 points above the accuracy published by the authors who released the dataset.

What *is* weak is **recall of 0.4948**. The error breakdown shows why accuracy hides it:

| | Predicted: No Purchase | Predicted: Purchase |
|---|---|---|
| **Actual: No Purchase** | 2,024 | 60 |
| **Actual: Purchase** | **193** | 189 |

Only 253 of 2,466 sessions are misclassified (10.26%), but **193 of those 253 errors — 76% — are missed purchases**. The model catches barely half the buyers, and accuracy conceals this because 193 mistakes are a small fraction of a mostly-negative test set. This is the concrete reason the assignment asks for six metrics: accuracy and recall disagree sharply here, and only one of them reflects what the retailer cares about.

Crucially, this is an artefact of the **0.50 decision threshold**, not a limitation of what the model learned. Sweeping the threshold on the *already-fitted* Random Forest, with no refitting:

| Threshold | Accuracy | F1 | MCC |
|---|---|---|---|
| 0.30 | 0.8812 | 0.6549 | 0.5883 |
| **0.35** | 0.8929 | **0.6658** | **0.6026** |
| 0.45 | **0.9006** | 0.6402 | 0.5895 |
| 0.50 _(default, reported above)_ | 0.8974 | 0.5990 | 0.5596 |
| 0.70 | 0.8767 | 0.3532 | 0.4191 |

Moving the threshold from 0.50 to 0.35 lifts F1 from 0.5990 to 0.6658 and MCC from 0.5596 to 0.6026 while costing 0.45 points of accuracy. Refitting with `class_weight="balanced"` achieves something similar by a different route — recall 0.4948 → 0.7016, F1 0.6625, MCC 0.5979, at 0.8893 accuracy.

Both are improvements on the submitted configuration, and both were deliberately left out of the main comparison table. The six models are reported at near-default settings so that they remain comparable to one another and so that the imbalance problem stays visible; tuning only the winner would have made the table an unfair fight. The honest summary is that **0.50 is the wrong operating point for a problem with 15.5% positives**, and the app's threshold slider exists so that trade-off can be explored rather than buried in a footnote.

---

## Streamlit application

Deployed on Streamlit Community Cloud. The app loads the pre-fitted `.joblib` pipelines and never retrains on startup, which keeps it responsive on the free tier.

| Required feature | Where it is |
|---|---|
| **Dataset upload (CSV)** | Sidebar → *1 · Test data*. Falls back to the bundled `test_data.csv` when nothing is uploaded. |
| **Model selection dropdown** | Sidebar → *2 · Model*, all six classifiers. |
| **Display of evaluation metrics** | *Model report* tab — all six metrics as cards, each with a note on how to read it. |
| **Confusion matrix / classification report** | *Model report* tab — annotated confusion-matrix heatmap **and** a full per-class classification report. |

Beyond the required four:

- **Decision-threshold slider.** Probabilities are computed once and cached, so moving the threshold re-derives labels without re-running any model. This makes the precision/recall trade-off tangible — and demonstrates the Naive Bayes saturation described above, which no threshold can fix.
- **ROC curves for all six models** on one axis, with the selected model highlighted.
- **Compare-all-models tab** that recomputes the full six-metric table on *your* uploaded file, with best-in-column highlighting and an Accuracy-vs-MCC chart.
- **Robust CSV handling.** The target is accepted as `True/False`, `1/0` or `Yes/No`; a file without a target column falls back to prediction-only mode; missing feature columns produce a clear error naming them.

### Deploying it yourself

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **New app**.
3. Select this repository, branch `main`, main file `app.py`.
4. Under **Advanced settings**, set the Python version to **3.12** to match `requirements.txt`.
5. Deploy, then paste the resulting URL into the table at the top of this README.

---

## Environment

Trained and verified with the exact versions pinned in `requirements.txt`:

| Package | Version |
|---|---|
| Python | 3.12.11 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| streamlit | 1.61.1 |

The versions are pinned deliberately. A pickled scikit-learn pipeline carries no guarantee of loading under a different scikit-learn version, so an unpinned `requirements.txt` is a deployment failure waiting to happen. If you change a pinned version, re-run `python -m model.train_models` to regenerate the pickles.

---

## BITS Virtual Lab execution

`model/ML_Assignment_2.ipynb` was executed on BITS Virtual Lab; the screenshot is included in the submitted PDF.
