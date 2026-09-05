"""Shared data/model pipeline for the frailty classifier: loading, ODD validation, RYG.

RYG and the ODD gate are two independent signals, matching the thesis
architecture's layer separation (L4 Data Governance / Art. 10 vs. L1
Interface / Art. 14): the ODD gate is a data-validity check that runs
before inference; RYG is a clinical risk-severity band computed FROM the
model's p(Frail) output. An ODD-invalid input never reaches the model at
all (safe-state bypass) and therefore has no RYG value.
"""

import pickle

import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

DATA_PATH = "Felix_FrailtyData.csv"
MODEL_PATH = "frailty_model.pkl"

# RYG severity bands over the model's p(Frail) output (Art. 14 / L1).
# Independent of the ODD gate (Art. 10 / L4) - see module docstring.
# RED=0.5 ties directly to the model's own decision boundary (the point
# where CatBoost itself would classify the case as Frail) - not an
# arbitrary number. GREEN/YELLOW=0.2 is ~2x the population base rate
# (Frail prevalence 9.3%): "meaningfully elevated risk relative to a
# random patient" even if the model wouldn't yet call it Frail. See
# PROJECT_LOG.md for the full rationale and band-size comparison.
RYG_THRESHOLDS = (0.2, 0.5)  # p < lo -> GREEN, lo <= p < hi -> YELLOW, p >= hi -> RED

# Valid ranges - one entry per feature (17 total). Sourced from the
# authoritative Gateway to Global Aging "Harmonized ELSA" codebook
# (reference/5050_harmonized_elsa_g3_2002-2019.pdf, Version G.3, June 2023), which
# superseded the earlier project-specific ELSAfrailty_dictionary.csv after
# it was found to contain errors (raeducl's coding text was a copy-paste
# of raeduc_e's; r6uppermob was overstated). See PROJECT_LOG.md.
ODD_RANGES = {
    "r6agey":     (50, 90),   # study-cohort restriction (raw ELSA variable spans 28-90 incl. spouses)
    "raracem":    (1, 4),     # 1=white, 4=non-white
    "raeduc_e":   (1, 5),     # 1=lt high-school, 3=high-school grad, 4=some college, 5=college+
    "raeducl":    (1, 3),     # 1=less than upper secondary, 2=upper secondary & vocational, 3=tertiary
    "rarelig_e":  (1, 8),     # 1=Christian .. 6=Sikh, 7=Other non-Christian, 8=None
    "r6shlt":     (1, 5),     # 1=very good .. 5=very bad
    "r6hlthlm":   (0, 1),     # 0=no, 1=yes (health limits work)
    "r6eata":     (0, 1),     # 0=no difficulty, 1=some difficulty eating
    "r6adlwa":    (0, 3),     # ADL Wallace 3-item sum
    "r6adlwaa":   (0, 1),     # whether R has any diff, Wallace ADLs
    "r6adla":     (0, 5),     # ADL 5-item sum
    "r6iadla":    (0, 3),     # IADL 3-item sum
    "r6iadlza":   (0, 5),     # IADL 5-item sum
    "r6iadlzaa":  (0, 1),     # whether R has any diff, 5-item IADL
    "r6grossa":   (0, 5),     # gross motor index (walk/run/climb/bed/bath)
    "r6lowermob": (0, 4),     # lower body mobility index
    "r6uppermob": (0, 3),     # upper body mobility index
}

RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """Load Felix_FrailtyData.csv, drop ID/constant/undocumented-bound columns,
    and build the binary target as strict Frail (1) vs. Robust (0).

    Pre-frail rows are dropped rather than merged into either class: merging
    pre-frail into Robust collapses to a 96/4 majority-class trap (recall
    ~0.03 on Frail); merging into Frail dilutes the signal (balanced accuracy
    ~0.58). Dropping pre-frail and comparing strictly Frail vs. Robust gives
    balanced accuracy ~0.73 with recall ~0.47 on Frail - the framing with
    real signal. See PROJECT_LOG.md for the full comparison.
    """
    df = pd.read_csv(path)
    work = df.drop(columns=["idauniqc", "r6iadlzam"])
    work = work[work["frail_cat1"] != 2]  # drop pre-frail rows
    work["target"] = work["frail_cat1"].map({0: 1, 1: 0})  # 1=Frail, 0=Robust
    X = work.drop(columns=["frail_cat1", "target", "r6liv10a"])
    y = work["target"]
    return X, y


def validate_odd(row, ranges=ODD_RANGES):
    """Return (is_valid, violations). Each violation names the variable + value."""
    violations = []
    for col, (lo, hi) in ranges.items():
        val = row[col]
        if pd.isna(val) or val < lo or val > hi:
            violations.append(f"{col}={val} outside [{lo},{hi}]")
    return (len(violations) == 0, violations)


def make_model(random_state=RANDOM_STATE):
    """CatBoost: finalized model choice (PROJECT_LOG.md) - best recall on Frail
    (0.768 vs. RandomForest's 0.473) at equivalent ROC-AUC (~0.90) across all
    three benchmarked models, and confirmed compatible with shap.TreeExplainer.

    Hyperparameters tuned via GridSearchCV (depth, learning_rate, iterations,
    l2_leaf_reg; scoring=balanced_accuracy, same 5-fold CV used throughout).
    Modest but real improvement over the original untuned guess: balanced
    accuracy 0.843->0.855, recall on Frail 0.768->0.795, ROC-AUC 0.903->0.908.
    """
    return CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, l2_leaf_reg=5,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               random_state=random_state, verbose=False)


def train_model(X, y, random_state=RANDOM_STATE):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               stratify=y, random_state=random_state)
    model = make_model(random_state).fit(X_tr, y_tr)
    return model


def save_model(model, path=MODEL_PATH):
    """REQ-UC1-02: pickle serialization."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


# Confusion matrix row/column order: Frail (1) is the positive class, listed
# first per clinical-test convention (top-left cell = True Positive: actual
# Frail, predicted Frail).
CM_LABELS = ["Frail (1)", "Robust (0)"]


def cv_clinical_metrics(X, y, random_state=RANDOM_STATE):
    """Out-of-fold 5-fold CV: confusion matrix, per-class precision/recall, ROC-AUC.

    Out-of-fold predictions (not the single holdout) so the numbers reflect
    the same stability check as the CV verdict, not one lucky/unlucky split.

    Frail (1) is treated as the positive class throughout: the confusion
    matrix uses labels=[1, 0] (Frail row/col first, so cm[0,0]=TP, cm[0,1]=FN,
    cm[1,0]=FP, cm[1,1]=TN), matching CM_LABELS.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = make_model(random_state)
    y_pred = cross_val_predict(rf, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]

    cm = confusion_matrix(y, y_pred, labels=[1, 0])
    report = classification_report(
        y, y_pred, target_names=["Robust (0)", "Frail (1)"], output_dict=True
    )
    auc = roc_auc_score(y, y_proba)
    return {"confusion_matrix": cm, "report": report, "roc_auc": auc}


def risk_band(p_frail, thresholds=RYG_THRESHOLDS):
    """Map a p(Frail) probability to a RYG severity band (Art. 14 / L1)."""
    lo, hi = thresholds
    if p_frail < lo:
        return "GREEN"
    if p_frail < hi:
        return "YELLOW"
    return "RED"


def classify_all(X, model, ranges=ODD_RANGES):
    """Vectorized classification across every row in X (fast: avoids per-row
    model.predict calls). Returns a DataFrame indexed like X with columns:
    - odd_status: 'VALID' or 'INVALID' (L4 gate; INVALID = safe-state bypass)
    - ryg: GREEN/YELLOW/RED severity band (L1; NaN when odd_status is INVALID,
      since the model never runs on an ODD-invalid input)
    - p_frail: model probability (NaN when odd_status is INVALID)
    """
    valid = pd.Series(True, index=X.index)
    for col, (lo, hi) in ranges.items():
        valid &= X[col].between(lo, hi) & X[col].notna()

    odd_status = pd.Series("INVALID", index=X.index)
    odd_status.loc[valid] = "VALID"
    ryg = pd.Series(pd.NA, index=X.index, dtype="object")
    p_frail = pd.Series(float("nan"), index=X.index)

    if valid.any():
        p = model.predict_proba(X.loc[valid])[:, 1]
        p_frail.loc[valid] = p
        ryg.loc[valid] = [risk_band(pi) for pi in p]

    return pd.DataFrame({"odd_status": odd_status, "ryg": ryg, "p_frail": p_frail})


def explain_case(row, model):
    """Local SHAP feature attribution for a single case (Art. 13, L2).

    Returns a DataFrame (one row per feature, sorted by |shap_value|
    descending) with columns: feature, value, shap_value. Positive
    shap_value pushes the prediction toward Frail; negative pushes toward
    Robust. shap_value + base_value, passed through a sigmoid, reconstructs
    the model's p(Frail) - verified in testing.
    """
    explainer = shap.TreeExplainer(model)
    row_df = row.to_frame().T
    shap_values = explainer.shap_values(row_df)[0]
    return pd.DataFrame({
        "feature": row.index,
        "value": row.values,
        "shap_value": shap_values,
    }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)


def classify_case(row, model, ranges=ODD_RANGES):
    """Two independent signals, matching the L4 (Art. 10) / L1 (Art. 14) split:
    the ODD gate (data validity) runs first; only a valid input reaches the
    model and gets a RYG risk-severity band. An ODD violation is a safe-state
    bypass, not a RYG value."""
    is_valid, violations = validate_odd(row, ranges)
    if not is_valid:
        return {
            "odd_status": "INVALID",
            "safe_state": True,
            "violations": violations,
            "ryg": None,
            "p_frail": None,
        }
    p_frail = float(model.predict_proba(row.to_frame().T)[0][1])
    return {
        "odd_status": "VALID",
        "safe_state": False,
        "violations": [],
        "ryg": risk_band(p_frail),
        "p_frail": p_frail,
    }
