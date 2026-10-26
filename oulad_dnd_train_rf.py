# REVISION NOTE (2026-09-29): CGPA, total_credit_enrolled and total_credit_passed were
# removed from every feature list in this script (feature set 32 -> 29, attackable 29 -> 26,
# MLP input 41 -> 38 dims). The columns may still exist in oulad_early.csv but are not used.
# Older comments below that still mention them describe the previous version.
import os
import json
import random
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
# ============================================================
# DEFAULT CONFIG
# ============================================================
# Detect-and-default defence for the Random Forest on oulad_early.csv. A
# Mahalanobis validator (fit on the clean training split, in the RAW attackable
# feature space, with the empirical covariance) flags inputs whose
# feature COMBINATION is unlikely under the training data. A flagged row is
# treated as tampered and either overridden to 'dropout' (the safe / high-recall
# class) or rejected.
#
# The FGSM gradient source defaults to the BASELINE MLP (DEFAULT_EVAL_MLP_MODEL = None);
# pass --eval-mlp-model <defended checkpoint> to attack with the adversarially
# trained MLP instead (the surrogate used by the adversarial-training script). The RF detector itself is unchanged: RF has no
# penultimate layer, so the Mahalanobis distance stays in the raw feature space.
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "oulad_early.csv"
DEFAULT_RF_RESULTS_DIR = "rf_results_oulad_early"
# Source MLP: defines the authoritative feature list and provides the
# preprocessor (the standardized space the L-inf epsilon lives in). This is the
# BASELINE MLP, same role as in adv_train_rf.py.
DEFAULT_MLP_DIR = "mlp_results_oulad_early"
# EVAL surrogate: the BASELINE MLP by default (None). This is the gradient source for
# the evaluation FGSM. The MLP adversarial training does not save its own
# preprocessor, so the eval attack reuses the baseline MLP preprocessor
# (DEFAULT_EVAL_MLP_PREPROCESSOR = None -> reuse --mlp-dir preprocessor). Pass
# --eval-mlp-model <defended checkpoint> to use the MLP-defended surrogate instead.
DEFAULT_EVAL_MLP_MODEL = None
DEFAULT_EVAL_MLP_PREPROCESSOR = None
DEFAULT_OUTPUT_DIR = "rf_detect_and_default_oulad_early_results"
TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout
# epsilon is an L-inf bound in the MLP's PREPROCESSED space, same as fgsm_rf.py /
# adv_train_rf.py. The final FGSM attack used 1.0.
EVAL_EPSILON_LIST = [1.0]
# detector: false-positive rate targeted when calibrating the detection threshold
# on the clean test split. Xu et al. calibrate the threshold so a small fraction
# of clean inputs are flagged, then measure how many adversarial inputs exceed it.
DEFAULT_DETECTION_TARGET_FPR = 0.05
# fallback network shape only; checkpoint values override these when present
DEFAULT_HIDDEN_DIMS = (128, 64, 32)
# ============================================================
# FEATURE DEFINITIONS (identical set to adv_train_rf.py / fgsm_rf.py)
# ============================================================
# id_student and target are never predictors. code_module / code_presentation are
# one-hot categoricals. module_credits exists and stays frozen. num_of_prev_attempts
# and studied_credits do NOT exist in this dataset version.
NON_FEATURE_COLS = ["id_student", TARGET_COL]
DEFAULT_SELECTED_FEATURES = [
    # static / enrolment (all FROZEN except the credit_* record, see below)
    "module_credits",          # FROZEN (fixed module property, not student-controllable)
    "code_module",             # categorical, FROZEN
    "code_presentation",       # categorical, FROZEN
    # prior-year credit record (years 1-3) + current-year enrolment -> ATTACKABLE
    "credit_enrolled_1styear",
    "credit_passed_1styear",
    "credit_enrolled_2ndyear",
    "credit_passed_2ndyear",
    "credit_enrolled_3rdyear",
    "credit_passed_3rdyear",
    "credit_enrolled_4thyear",
    # prior-year academic record -> ATTACKABLE
    "GPA_1styear",
    "GPA_2ndyear",
    "GPA_3rdyear",
    # early window, cut at day 65
    "early_avg_score_65",
    "early_n_submitted_65",
    "early_first_submit_day_65",
    "early_total_clicks_65",
    "early_active_days_65",
    "early_unique_resources_65",
    "early_quiz_clicks_65",
    "early_forum_clicks_65",
    # early window, cut at day 153
    "early_avg_score_153",
    "early_n_submitted_153",
    "early_first_submit_day_153",
    "early_total_clicks_153",
    "early_active_days_153",
    "early_unique_resources_153",
    "early_quiz_clicks_153",
    "early_forum_clicks_153",
]
# ============================================================
# ATTACKABLE FEATURES (identical to adv_train_rf.py, 29 features)
# ============================================================
ATTACKABLE_FEATURES = [
    "early_avg_score_65",
    "early_n_submitted_65",
    "early_first_submit_day_65",
    "early_total_clicks_65",
    "early_active_days_65",
    "early_unique_resources_65",
    "early_quiz_clicks_65",
    "early_forum_clicks_65",
    "early_avg_score_153",
    "early_n_submitted_153",
    "early_first_submit_day_153",
    "early_total_clicks_153",
    "early_active_days_153",
    "early_unique_resources_153",
    "early_quiz_clicks_153",
    "early_forum_clicks_153",
    "GPA_1styear",
    "GPA_2ndyear",
    "GPA_3rdyear",
    "credit_enrolled_4thyear",
    "credit_enrolled_1styear",
    "credit_passed_1styear",
    "credit_enrolled_2ndyear",
    "credit_passed_2ndyear",
    "credit_enrolled_3rdyear",
    "credit_passed_3rdyear",
]
FROZEN_FEATURES_HINT = [
    "code_module",
    "code_presentation",
    "module_credits",
]
# ============================================================
# DERIVED FEATURES (recomputed after perturbation for internal consistency)
# ============================================================
# Order no longer strictly matters (credits_remaining, the only column that
# depended on the totals, has been removed). CGPA and the two totals are each
# recomputed independently from their components.
DERIVED_FEATURE_COMPONENTS = {}  # CGPA and the credit totals were removed from the feature set
# ============================================================
# DOMAIN CONSTRAINTS (identical to adv_train_rf.py)
# ============================================================
ROUND_TO_INT_FEATURES = [
    # early-window integer counts (both windows)
    "early_n_submitted_65",
    "early_first_submit_day_65",
    "early_total_clicks_65",
    "early_active_days_65",
    "early_unique_resources_65",
    "early_quiz_clicks_65",
    "early_forum_clicks_65",
    "early_n_submitted_153",
    "early_first_submit_day_153",
    "early_total_clicks_153",
    "early_active_days_153",
    "early_unique_resources_153",
    "early_quiz_clicks_153",
    "early_forum_clicks_153",
    # credit record: all integers
    "credit_enrolled_4thyear",
    "credit_enrolled_1styear",
    "credit_passed_1styear",
    "credit_enrolled_2ndyear",
    "credit_passed_2ndyear",
    "credit_enrolled_3rdyear",
    "credit_passed_3rdyear",
]
DECIMAL_2DP_FEATURES = [
    "early_avg_score_65",
    "early_avg_score_153",
    "GPA_1styear",
    "GPA_2ndyear",
    "GPA_3rdyear",
]
# Score features are averages on a 0-100 scale -> clip to [0, 100].
PERCENTAGE_FEATURES = [
    "early_avg_score_65",
    "early_avg_score_153",
]
# Non-negative features. first_submit_day is intentionally left OUT (the train
# min/max clip already bounds it). GPA_* / CGPA are 0-4 (non-negative, never
# clipped to 100). All credit_* columns are non-negative too.
NONNEGATIVE_FEATURES = sorted(set([
    # early window (all except first_submit_day)
    "early_avg_score_65",
    "early_n_submitted_65",
    "early_total_clicks_65",
    "early_active_days_65",
    "early_unique_resources_65",
    "early_quiz_clicks_65",
    "early_forum_clicks_65",
    "early_avg_score_153",
    "early_n_submitted_153",
    "early_total_clicks_153",
    "early_active_days_153",
    "early_unique_resources_153",
    "early_quiz_clicks_153",
    "early_forum_clicks_153",
    # academic record
    "GPA_1styear",
    "GPA_2ndyear",
    "GPA_3rdyear",
    # credit record
    "credit_enrolled_4thyear",
    "credit_enrolled_1styear",
    "credit_passed_1styear",
    "credit_enrolled_2ndyear",
    "credit_passed_2ndyear",
    "credit_enrolled_3rdyear",
    "credit_passed_3rdyear",
]))
# ============================================================
# SOURCE MLP MODEL (must match the training MLP exactly)
# ============================================================
class DropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=DEFAULT_HIDDEN_DIMS, dropout=0.30):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
# ============================================================
# HELPERS
# ============================================================
def set_global_seed(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_python(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
def one_hot(labels, n_classes=2):
    labels = np.asarray(labels).astype(int)
    return np.eye(n_classes, dtype=np.float32)[labels]
def normalize_target(series):
    target_normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "non dropout": "non-dropout",
            "non_dropout": "non-dropout",
            "drop out": "dropout",
        })
    )
    target_map = {"non-dropout": 0, "dropout": 1, "0": 0, "1": 1}
    y = target_normalized.map(target_map)
    if y.isna().any():
        bad_values = series.loc[y.isna()].unique().tolist()
        raise ValueError(f"Some target values could not be mapped: {bad_values}")
    return y.astype(int)
def prepare_features_and_label(df, selected_features):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    missing_features = [c for c in selected_features if c not in df.columns]
    if missing_features:
        raise ValueError(f"Selected features missing from the CSV: {missing_features}")
    X = df[selected_features].copy()
    y = normalize_target(df[TARGET_COL])
    if "id_student" in df.columns:
        student_ids = df["id_student"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    return X, y, student_ids
def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # SAME split logic and random_state as the RF / MLP training scripts.
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index, name="row_id")
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids, test_size=0.30, stratify=y, random_state=random_state,
    )
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp, y_temp, id_temp, test_size=1.0 / 3.0, stratify=y_temp, random_state=random_state,
    )
    split_report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "validation_rows": len(X_val),
        "train_label_distribution": y_train.value_counts().sort_index().to_dict(),
        "test_label_distribution": y_test.value_counts().sort_index().to_dict(),
        "validation_label_distribution": y_val.value_counts().sort_index().to_dict(),
    }
    return (X_train, X_test, X_val, y_train, y_test, y_val,
            id_train, id_test, id_val, split_report)
# ============================================================
# PATH RESOLUTION AND LOADING
# ============================================================
def resolve_first_existing(candidates, what):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{what} not found. Checked:\n" + "\n".join(candidates))
def resolve_rf_model_path(rf_results_dir):
    return resolve_first_existing([os.path.join(rf_results_dir, "oulad_rf_model.pkl")], "RF model (oulad_rf_model.pkl)")
def resolve_rf_summary_path(rf_results_dir):
    path = os.path.join(rf_results_dir, "oulad_rf_summary.json")
    return path if os.path.exists(path) else None
def load_rf_threshold(summary_path, fallback=0.5):
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        candidates = [
            summary.get("selection", {}).get("best_threshold"),
            summary.get("best_threshold"),
        ]
        for threshold in candidates:
            if threshold is not None:
                return float(threshold), summary_path
    return float(fallback), "fallback_0.5"
def resolve_mlp_model_path(mlp_dir):
    return resolve_first_existing([os.path.join(mlp_dir, "oulad_mlp_model.pt")], "MLP model (oulad_mlp_model.pt)")
def resolve_mlp_preprocessor_path(mlp_dir):
    return resolve_first_existing([os.path.join(mlp_dir, "oulad_mlp_preprocessor.pkl")], "MLP preprocessor (oulad_mlp_preprocessor.pkl)")
def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
def read_mlp_checkpoint(path, device):
    obj = load_torch_checkpoint(path, device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        info = {
            "hidden_dims": tuple(obj.get("hidden_dims", DEFAULT_HIDDEN_DIMS)),
            "dropout": float(obj.get("dropout", 0.30)),
            "state_dict": obj["model_state_dict"],
            "selected_features": obj.get("selected_features", None),
            "feature_set": obj.get("feature_set", None),
            "input_dim": obj.get("input_dim", None),
        }
    else:
        info = {
            "hidden_dims": DEFAULT_HIDDEN_DIMS,
            "dropout": 0.30,
            "state_dict": obj,
            "selected_features": None,
            "feature_set": None,
            "input_dim": None,
        }
    return info
def build_mlp_from_state(info, input_dim, device):
    model = DropoutMLP(input_dim=input_dim, hidden_dims=info["hidden_dims"], dropout=info["dropout"]).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    return model
def wrap_mlp_for_art(model, input_dim):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=2,
    )
# ============================================================
# PREPROCESSOR HELPERS AND MASKS
# ============================================================
def transform_to_float32(preprocessor, X):
    arr = preprocessor.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)
def get_cols_from_preprocessor(preprocessor, transformer_name):
    for name, transformer, cols in preprocessor.transformers_:
        if name == transformer_name:
            return list(cols)
    return []
def get_numeric_cols_from_preprocessor(preprocessor):
    cols = get_cols_from_preprocessor(preprocessor, "num")
    if not cols:
        raise ValueError("Transformer 'num' not found in preprocessor.")
    return cols
def get_scaler_from_preprocessor(preprocessor):
    num_pipeline = preprocessor.named_transformers_["num"]
    if "scaler" not in num_pipeline.named_steps:
        raise ValueError("Scaler not found in numeric pipeline.")
    return num_pipeline.named_steps["scaler"]
def make_preprocessed_attack_mask(preprocessor, selected_features, input_dim):
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(f"Attackable features not in selected_features: {missing_attackable}")
    mask = np.zeros(input_dim, dtype=np.float32)
    for idx, col in enumerate(numeric_cols):
        mask[idx] = 1.0 if col in attackable_raw else 0.0
    if int(mask.sum()) == 0:
        raise ValueError("No preprocessed feature is attackable. Check ATTACKABLE_FEATURES.")
    return mask, numeric_cols
# OULAD credit-record rules, taken from how oulad_cleaning.py builds the data:
#   * every credit value is a multiple of 30 (one module = 30 credits),
#   * each prior year (1st, 2nd, 3rd) is enrolled with 30 to 120 credits,
#   * credit_enrolled_4thyear = 360 - (year1 + year2 + year3) and lies in 30..120,
#   * credit_passed_Nthyear is a multiple of 30 and never exceeds the enrolled
#     credits of the same year,
#   * total_credit_enrolled / total_credit_passed are the sums of years 1 to 3,
#   * CGPA = round(mean(GPA_1styear, GPA_2ndyear, GPA_3rdyear), 2).
CREDIT_UNIT = 30
DEGREE_TOTAL_CREDITS = 360
YEAR_MIN_CREDITS = 30
YEAR_MAX_CREDITS = 120
ENROLLED_YEAR_COLS = ["credit_enrolled_1styear", "credit_enrolled_2ndyear", "credit_enrolled_3rdyear"]
PASSED_YEAR_COLS = ["credit_passed_1styear", "credit_passed_2ndyear", "credit_passed_3rdyear"]
YEAR4_ENROLLED_COL = "credit_enrolled_4thyear"
CREDIT_RULE_TEXT = (
    "Credit record projected onto the oulad_early.csv rules after clip/round: enrolled "
    "years 1-3 snapped to multiples of 30 in [30, 120]; if needed, 30 credits are added "
    "to the smallest year or removed from the largest year so that "
    "credit_enrolled_4thyear = 360 - (year1 + year2 + year3) lies in [30, 120]; passed "
    "years 1-3 snapped to multiples of 30 and capped at the enrolled credits of the same "
    "year. CGPA and the credit totals are not model features."
)
def snap_to_credit_grid(values, lower, upper):
    # round to the nearest multiple of CREDIT_UNIT, then keep it inside [lower, upper]
    values = np.asarray(values, dtype=float)
    return np.clip(np.round(values / CREDIT_UNIT) * CREDIT_UNIT, lower, upper)
# ============================================================
# FEASIBILITY RULES ADDED 2026-09-30: early-window activity and submissions
# ============================================================
# Each rule follows from how oulad_cleaning.py builds the early-window features
# from the raw click and assessment logs, and each holds for 100% of the rows of
# oulad_early.csv:
#   (1) the day-65 window is contained in the day-153 window, so every day-65
#       count is <= its day-153 counterpart (total, active days, unique
#       resources, quiz, forum, submissions);
#   (2) days 66 to 153 contain 88 days, so active_days_153 - active_days_65 <= 88;
#   (3) quiz and forum clicks are disjoint subsets of all clicks, so
#       quiz + forum <= total clicks in each window;
#   (4) every active day and every distinct resource needs at least one click,
#       so active_days <= total clicks and unique_resources <= total clicks;
#   (5) with no submission in a window there is no score and no first
#       submission day, so n_submitted == 0 implies avg_score == 0 and
#       first_submit_day == 0.
# Rules (1) to (4) are first repaired upward (larger quantity raised), then
# downward if a raised value would pass the training maximum. Rule (5) is
# applied last. Rows that still break a rule after the repair keep their clean
# activity values (fallback); no clean row is changed, because clean rows
# already satisfy every rule. Runs AFTER enforce_derived_consistency.
_OULAD_WINDOWS = ("65", "153")
_OULAD_WINDOW_GAP_DAYS = 88
_OULAD_MONOTONE = ("total_clicks", "active_days", "unique_resources",
                   "quiz_clicks", "forum_clicks", "n_submitted")

def _oc(name, w):
    return f"early_{name}_{w}"

def oulad_feasibility_violations(X):
    # Boolean per row: True if the row breaks any of rules (1) to (5).
    X = X.reset_index(drop=True)
    g = lambda c: X[c].to_numpy(dtype=float)
    bad = np.zeros(len(X), dtype=bool)
    for name in _OULAD_MONOTONE:
        bad |= g(_oc(name, "65")) > g(_oc(name, "153"))
    bad |= g(_oc("active_days", "153")) - g(_oc("active_days", "65")) > _OULAD_WINDOW_GAP_DAYS
    for w in _OULAD_WINDOWS:
        tot = g(_oc("total_clicks", w))
        bad |= g(_oc("quiz_clicks", w)) + g(_oc("forum_clicks", w)) > tot
        bad |= g(_oc("active_days", w)) > tot
        bad |= g(_oc("unique_resources", w)) > tot
        nsub0 = g(_oc("n_submitted", w)) == 0
        bad |= nsub0 & ((g(_oc("avg_score", w)) != 0) | (g(_oc("first_submit_day", w)) != 0))
    return bad

def enforce_activity_consistency(X_adv_raw, X_clean_reset, train_min, train_max):
    needed = [_oc(n, w) for n in _OULAD_MONOTONE + ("avg_score", "first_submit_day") for w in _OULAD_WINDOWS]
    if not all(c in X_adv_raw.columns for c in needed):
        return X_adv_raw
    V = {c: X_adv_raw[c].to_numpy(dtype=float).copy() for c in needed}
    # a value is only capped at the training maximum if this function raised it
    hi = {c: np.maximum(float(train_max[c]) if c in train_max.index else np.inf, V[c]) for c in needed}

    def cap_down():
        # downward repairs, used only when an upward repair passed the train maximum
        for w in _OULAD_WINDOWS:
            tot = V[_oc("total_clicks", w)]
            q, f = V[_oc("quiz_clicks", w)], V[_oc("forum_clicks", w)]
            s = q + f
            over = s > tot
            if over.any():
                scale = np.where(s[over] > 0, tot[over] / s[over], 0.0)
                q[over] = np.floor(q[over] * scale)
                f[over] = np.floor(f[over] * scale)
            V[_oc("active_days", w)] = np.minimum(V[_oc("active_days", w)], tot)
            V[_oc("unique_resources", w)] = np.minimum(V[_oc("unique_resources", w)], tot)
        for name in _OULAD_MONOTONE:
            V[_oc(name, "65")] = np.minimum(V[_oc(name, "65")], V[_oc(name, "153")])
        V[_oc("active_days", "153")] = np.minimum(
            V[_oc("active_days", "153")], V[_oc("active_days", "65")] + _OULAD_WINDOW_GAP_DAYS)

    # upward repairs
    for name in ("active_days", "unique_resources", "quiz_clicks", "forum_clicks", "n_submitted"):
        V[_oc(name, "153")] = np.maximum(V[_oc(name, "153")], V[_oc(name, "65")])
    V[_oc("active_days", "153")] = np.minimum(
        V[_oc("active_days", "153")], V[_oc("active_days", "65")] + _OULAD_WINDOW_GAP_DAYS)
    for w in _OULAD_WINDOWS:
        V[_oc("total_clicks", w)] = np.maximum.reduce([
            V[_oc("total_clicks", w)],
            V[_oc("active_days", w)],
            V[_oc("unique_resources", w)],
            V[_oc("quiz_clicks", w)] + V[_oc("forum_clicks", w)],
        ])
    V[_oc("total_clicks", "153")] = np.maximum(V[_oc("total_clicks", "153")], V[_oc("total_clicks", "65")])
    # respect the training maximum, then repair downward until stable
    for c in needed:
        V[c] = np.minimum(V[c], hi[c])
    for _ in range(10):
        cap_down()
    # rule (5): no submission -> no score and no first submission day
    for w in _OULAD_WINDOWS:
        nsub0 = V[_oc("n_submitted", w)] == 0
        V[_oc("avg_score", w)][nsub0] = 0.0
        V[_oc("first_submit_day", w)][nsub0] = 0.0
    for c in needed:
        X_adv_raw[c] = V[c]
    # fallback: any row still infeasible keeps its clean activity block
    still_bad = oulad_feasibility_violations(X_adv_raw)
    if still_bad.any():
        for c in needed:
            X_adv_raw.loc[still_bad, c] = X_clean_reset.loc[still_bad, c].to_numpy(dtype=float)
    return X_adv_raw

def assert_feasible(bad_rows, attacked_row_mask, where):
    # Hard check: every attacked row must satisfy the feasibility rules.
    bad_rows = np.asarray(bad_rows).astype(bool)
    if attacked_row_mask is not None:
        bad_rows = bad_rows & np.asarray(attacked_row_mask).astype(bool)
    if bad_rows.any():
        raise AssertionError(f"{int(bad_rows.sum())} adversarial row(s) break a feasibility rule ({where}).")

def enforce_derived_consistency(X_adv_raw):
    # Applied LAST, after clip/round, so every adversarial row obeys the same
    # credit rules as the clean oulad_early.csv rows (see the constants above).
    #   1) enrolled years 1..3 are snapped to the 30-credit grid inside [30, 120];
    #   2) feasibility repair: year 4 must be 360 - sum(years 1..3) and inside
    #      [30, 120], so sum(years 1..3) must lie in [240, 330]. While the sum is
    #      too small, 30 credits are added to the smallest year (< 120); while it
    #      is too large, 30 credits are removed from the largest year (> 30);
    #   3) credit_enrolled_4thyear is recomputed as 360 - sum(years 1..3);
    #   4) passed years 1..3 are snapped to the 30-credit grid and capped at the
    #      enrolled credits of the same year (floor 0);
    #   5) totals are the exact sums of years 1..3 and CGPA is the mean of the
    #      three (2dp) yearly GPAs, rounded to 2dp.
    if all(c in X_adv_raw.columns for c in ENROLLED_YEAR_COLS):
        E = np.column_stack([
            snap_to_credit_grid(X_adv_raw[c].to_numpy(dtype=float), YEAR_MIN_CREDITS, YEAR_MAX_CREDITS)
            for c in ENROLLED_YEAR_COLS
        ])
        min_sum = DEGREE_TOTAL_CREDITS - YEAR_MAX_CREDITS
        max_sum = DEGREE_TOTAL_CREDITS - YEAR_MIN_CREDITS
        max_iter = int((YEAR_MAX_CREDITS - YEAR_MIN_CREDITS) * len(ENROLLED_YEAR_COLS) / CREDIT_UNIT) + 1
        for _ in range(max_iter):
            year_sum = E.sum(axis=1)
            too_small = np.where(year_sum < min_sum)[0]
            too_large = np.where(year_sum > max_sum)[0]
            if len(too_small) == 0 and len(too_large) == 0:
                break
            if len(too_small) > 0:
                candidates = np.where(E[too_small] < YEAR_MAX_CREDITS, E[too_small], np.inf)
                E[too_small, np.argmin(candidates, axis=1)] += CREDIT_UNIT
            if len(too_large) > 0:
                candidates = np.where(E[too_large] > YEAR_MIN_CREDITS, E[too_large], -np.inf)
                E[too_large, np.argmax(candidates, axis=1)] -= CREDIT_UNIT
        for j, c in enumerate(ENROLLED_YEAR_COLS):
            X_adv_raw[c] = E[:, j].astype(float)
        if YEAR4_ENROLLED_COL in X_adv_raw.columns:
            X_adv_raw[YEAR4_ENROLLED_COL] = (DEGREE_TOTAL_CREDITS - E.sum(axis=1)).astype(float)
        for j, c in enumerate(PASSED_YEAR_COLS):
            if c in X_adv_raw.columns:
                passed = snap_to_credit_grid(X_adv_raw[c].to_numpy(dtype=float), 0.0, YEAR_MAX_CREDITS)
                X_adv_raw[c] = np.minimum(passed, E[:, j]).astype(float)
    return X_adv_raw
def preprocessed_adv_to_raw(
    X_adv_preprocessed, X_clean_raw, selected_features, preprocessor,
    train_min, train_max, attacked_row_mask=None, enforce_consistency=True,
):
    # Convert adversarial preprocessed data back to raw space. Only the numeric
    # block is inverse-scaled; frozen features and non-attacked rows are restored
    # from the clean data so the attack scope stays strict. The RF consumes the
    # raw dataframe (its pkl pipeline does its own preprocessing), so only the raw
    # frame is returned here. Logic mirrors adv_train_rf.py exactly.
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    scaler = get_scaler_from_preprocessor(preprocessor)
    n_num = len(numeric_cols)
    adv_numeric_scaled = X_adv_preprocessed[:, :n_num]
    adv_numeric_raw = scaler.inverse_transform(adv_numeric_scaled)
    X_adv_raw = X_clean_raw[selected_features].reset_index(drop=True).copy()
    clean_reset = X_clean_raw[selected_features].reset_index(drop=True)
    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    for col in numeric_cols:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=train_min[col], upper=train_max[col])
    for col in NONNEGATIVE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0)
    for col in PERCENTAGE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0, upper=100)
    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col]).astype(float)
    for col in DECIMAL_2DP_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col].astype(float), 2)
    # enforce domain identities LAST (enforce-last, matching zenodo). Hardcoded on.
    X_adv_raw = enforce_derived_consistency(X_adv_raw)
    X_adv_raw = enforce_activity_consistency(X_adv_raw, clean_reset, train_min, train_max)
    assert_feasible(oulad_feasibility_violations(X_adv_raw), attacked_row_mask, 'OULAD projection')
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        not_attacked = ~attacked_row_mask
        if not_attacked.any():
            for col in selected_features:
                values = clean_reset.loc[not_attacked, col]
                if pd.api.types.is_numeric_dtype(X_adv_raw[col]):
                    values = values.astype(X_adv_raw[col].dtype, copy=False)
                X_adv_raw.loc[not_attacked, col] = values.values
    return X_adv_raw[selected_features]
# ============================================================
# ATTACK LABELS AND SCOPE
# ============================================================
def make_attack_labels(attack_mode, y_true):
    if attack_mode == "untargeted":
        return one_hot(y_true), False
    if attack_mode == "target_non_dropout":
        target_labels = np.zeros(len(y_true), dtype=int)
        return one_hot(target_labels), True
    raise ValueError("attack_mode must be 'untargeted' or 'target_non_dropout'.")
def make_attack_scope_mask(attack_scope, y_true, clean_pred):
    y_true = np.asarray(y_true).astype(int)
    clean_pred = np.asarray(clean_pred).astype(int)
    if attack_scope == "all":
        return np.ones(len(y_true), dtype=bool)
    if attack_scope == "dropout_only":
        return y_true == POSITIVE_LABEL
    if attack_scope == "true_positive_only":
        return (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL)
    raise ValueError("attack_scope must be 'all', 'dropout_only', or 'true_positive_only'.")
def generate_fgsm_for_scope(
    art_mlp, X_clean_raw, X_clean_preprocessed, y_true, clean_pred, epsilon,
    attack_mode, attack_scope, perturbable_mask, preprocessor, selected_features,
    train_min, train_max, enforce_consistency=True, batch_size=128,
):
    # Run FGSM only on the selected rows; untouched rows stay clean, then map
    # everything back to raw space with the domain constraints applied. The
    # gradient comes from art_mlp (the eval surrogate: baseline MLP by default).
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()
    if int(attack_idx.sum()) > 0:
        attack_y, targeted = make_attack_labels(attack_mode, np.asarray(y_true)[attack_idx])
        fgsm = FastGradientMethod(estimator=art_mlp, eps=float(epsilon), targeted=targeted, batch_size=batch_size)
        X_adv_subset = fgsm.generate(x=X_clean_preprocessed[attack_idx], y=attack_y, mask=perturbable_mask)
        frozen = perturbable_mask == 0
        X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
        X_adv_preprocessed[attack_idx] = X_adv_subset
    X_adv_raw = preprocessed_adv_to_raw(
        X_adv_preprocessed=X_adv_preprocessed,
        X_clean_raw=X_clean_raw,
        selected_features=selected_features,
        preprocessor=preprocessor,
        train_min=train_min,
        train_max=train_max,
        attacked_row_mask=attack_idx,
        enforce_consistency=enforce_consistency,
    )
    return X_adv_raw, attack_idx
# ============================================================
# INPUT VALIDATION (test-time rejection defence) -- RAW feature space
# ============================================================
class InputValidator:
    # Mahalanobis distance over the RAW attackable numeric features, input-global
    # (one mean and one covariance for the whole clean training split, not
    # class-conditional). The same detector is used for RF, XGBoost and MLP.
    #
    # Covariance estimator: the empirical covariance of the raw attackable features
    # (np.cov, the maximum-likelihood style estimate used by Lee et al. and Zhang et
    # al.). A small ridge (1e-6 on the diagonal) is added and the matrix is inverted
    # with a pseudo-inverse, because some attackable features are exact or
    # near-exact linear combinations of others, so the covariance is singular or
    # nearly singular. A comparison run with a Ledoit-Wolf shrinkage covariance gave
    # almost identical flag decisions (over 99% agreement on the OULAD MLP run).
    def __init__(self, attackable_features, ridge=1e-6):
        self.attackable_features = list(attackable_features)
        self.ridge = float(ridge)
        self.cols = None
        self.mu = None
        self.cov_inv = None
        self.condition_number = None
    def fit(self, X_train, numeric_cols):
        # only validate the attackable numeric features (the ones the attack moves)
        self.cols = [c for c in numeric_cols if c in set(self.attackable_features)]
        M = X_train[self.cols].to_numpy(dtype=float)
        self.mu = np.nanmean(M, axis=0)
        cov = np.cov(M, rowvar=False)
        cov = cov + self.ridge * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.pinv(cov)
        eigenvalues = np.linalg.eigvalsh(cov)
        self.condition_number = float(eigenvalues.max() / max(eigenvalues.min(), np.finfo(float).tiny))
        return self
    def mahalanobis_scores(self, X):
        M = X[self.cols].to_numpy(dtype=float)
        d = M - self.mu
        left = d @ self.cov_inv
        sq = np.einsum("ij,ij->i", left, d)
        return np.sqrt(np.clip(sq, 0, None))
# ============================================================
# EVALUATION
# ============================================================
def evaluate_rf_plain(model, X_raw, y_true, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_proba = model.predict_proba(X_raw)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba
def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)
def evaluate_with_defence(model, X_raw, y_true, threshold, validator, maha_threshold,
                          action, condition, split, eval_epsilon):
    # action == "none"            -> plain model prediction (no defence)
    # action == "default_dropout" -> flagged rows forced to dropout, evaluate on ALL rows
    # action == "abstain"         -> flagged rows rejected, evaluate on ACCEPTED rows only
    y_true = np.asarray(y_true).astype(int)
    y_proba = model.predict_proba(X_raw)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    flags = validator.mahalanobis_scores(X_raw) > maha_threshold
    n_flagged = int(flags.sum())
    if action == "none":
        y_pred_final = y_pred
        eval_mask = np.ones(len(y_true), dtype=bool)
    elif action == "default_dropout":
        y_pred_final = y_pred.copy()
        y_pred_final[flags] = POSITIVE_LABEL
        eval_mask = np.ones(len(y_true), dtype=bool)
    elif action == "abstain":
        y_pred_final = y_pred.copy()
        eval_mask = ~flags  # keep only accepted (non-flagged) rows
    else:
        raise ValueError("action must be 'none', 'default_dropout', or 'abstain'.")
    yt = y_true[eval_mask]
    yp = y_pred_final[eval_mask]
    result = {
        "model": "Random Forest",
        "condition": condition,
        "action": action,
        "split": split,
        "eval_epsilon": eval_epsilon,
        "threshold": float(threshold),
        "n_rows_evaluated": int(eval_mask.sum()),
        "n_flagged": n_flagged,
        "flag_rate": float(np.mean(flags)),
        "accuracy": accuracy_score(yt, yp) if len(yt) else float("nan"),
        "precision_dropout": precision_score(yt, yp, pos_label=POSITIVE_LABEL, zero_division=0) if len(yt) else float("nan"),
        "recall_dropout": recall_score(yt, yp, pos_label=POSITIVE_LABEL, zero_division=0) if len(yt) else float("nan"),
        "f1_dropout": f1_score(yt, yp, pos_label=POSITIVE_LABEL, zero_division=0) if len(yt) else float("nan"),
    }
    cm = confusion_matrix(yt, yp, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]) if len(yt) else None
    return result, cm
# ============================================================
# PER-STUDENT TRACE (for the detect-and-default check-students script)
# ============================================================
def save_student_trace(model, X_clean_raw, X_adv_raw, y_true, student_ids, threshold,
                       validator, maha_threshold, actions, attack_idx, eval_epsilon,
                       split, selected_features, path):
    # One row per student for a single (split, eval_epsilon). This is the file the
    # DnD check-students script reads. Row order is positional: X_clean_raw,
    # X_adv_raw, y_true, student_ids and attack_idx all come from the same split in
    # the same order (X_adv_raw is reset_index'd inside preprocessed_adv_to_raw, and
    # predict_proba returns positionally), so index i is the same student in all of
    # them.
    #
    # The detect-and-default RF is the UNCHANGED baseline RF; the only thing the
    # defence adds is the Mahalanobis flag + an override. So a hidden dropout is
    # only recovered when the detector flags its adversarial row (flagged_adv).
    y_true = np.asarray(y_true).astype(int)
    ids = np.asarray(student_ids)
    # undefended baseline RF on clean and on adversarial inputs
    clean_proba = model.predict_proba(X_clean_raw[selected_features])[:, 1]
    clean_pred = (clean_proba >= threshold).astype(int)
    adv_proba = model.predict_proba(X_adv_raw[selected_features])[:, 1]
    adv_pred = (adv_proba >= threshold).astype(int)
    # Mahalanobis detector scores + flags (raw attackable-feature space)
    score_clean = validator.mahalanobis_scores(X_clean_raw[selected_features])
    score_adv = validator.mahalanobis_scores(X_adv_raw[selected_features])
    flagged_clean = score_clean > maha_threshold
    flagged_adv = score_adv > maha_threshold
    trace = pd.DataFrame({
        "id_student": ids,
        "split": split,
        "eval_epsilon": float(eval_epsilon),
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        "in_attack_scope": np.asarray(attack_idx).astype(bool),
        # undefended baseline RF: clean vs adversarial (defines attacked / hidden)
        "rf_clean_pred_label": clean_pred,
        "rf_clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        "rf_clean_prob_dropout": clean_proba,
        "rf_adv_pred_label": adv_pred,
        "rf_adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        "rf_adv_prob_dropout": adv_proba,
        # detector
        "maha_threshold": float(maha_threshold),
        "maha_score_clean": score_clean,
        "maha_score_adv": score_adv,
        "flagged_clean": flagged_clean,
        "flagged_adv": flagged_adv,
    })
    # per-action defended DECISION under attack
    for action in actions:
        if action == "default_dropout":
            final = adv_pred.copy()
            final[flagged_adv] = POSITIVE_LABEL
            trace["dnd_default_dropout_pred_label"] = final
            trace["dnd_default_dropout_pred_target"] = np.where(final == 1, "dropout", "non-dropout")
        elif action == "abstain":
            # flagged rows are rejected (sent to review), not given a dropout label
            trace["dnd_abstain_decision"] = np.where(
                flagged_adv, "abstained",
                np.where(adv_pred == 1, "dropout", "non-dropout"))
        elif action == "none":
            trace["dnd_none_pred_label"] = adv_pred
        else:
            raise ValueError(f"unknown action for trace: {action}")
    # convenience flags for the check-students script. successfully_attacked =
    # real dropout, caught by the clean baseline RF, hidden once attacked.
    hidden = (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL) & (adv_pred == NEGATIVE_LABEL)
    trace["successfully_attacked"] = hidden
    # recovery under default_dropout = the hidden dropout's adversarial row is
    # flagged and therefore forced back to dropout. Under abstain the same flag
    # sends it to review ("caught") instead of letting it pass silently.
    if "default_dropout" in actions:
        trace["recovered_default_dropout"] = hidden & flagged_adv
    if "abstain" in actions:
        trace["caught_abstain"] = hidden & flagged_adv
    trace.to_csv(path, index=False)
    return trace
# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_detect_and_default_experiment(
    df, csv_path, output_dir, rf_results_dir, mlp_dir, feature_set_hint,
    eval_epsilon_list, random_state, attack_mode, eval_attack_scope,
    detection_target_fpr, actions, eval_mlp_model_path_arg, eval_mlp_preprocessor_arg,
    enforce_consistency,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("RF DETECT-AND-DEFAULT (MAHALANOBIS RAW-SPACE FLAG -> OVERRIDE, oulad_early)")
    print("=" * 80)
    print(f"CSV path            : {csv_path}")
    print(f"Output dir          : {output_dir}")
    print(f"Attack mode         : {attack_mode}")
    print(f"Eval attack scope   : {eval_attack_scope}")
    print(f"Detection target FPR: {detection_target_fpr}")
    print(f"Actions             : {actions}")
    print(f"Device              : {device}")
    print()
    # ---- source (baseline) MLP: authoritative feature list + preprocessor ----
    mlp_model_path = resolve_mlp_model_path(mlp_dir)
    mlp_preprocessor_path = resolve_mlp_preprocessor_path(mlp_dir)
    mlp_info = read_mlp_checkpoint(mlp_model_path, device)
    if mlp_info["selected_features"] is not None:
        selected_features = list(mlp_info["selected_features"])
        feature_source = "mlp_checkpoint"
    else:
        selected_features = DEFAULT_SELECTED_FEATURES.copy()
        feature_source = "fallback_default"
    print(f"Feature list source : {feature_source} ({len(selected_features)} features)")
    missing_attackable = [f for f in ATTACKABLE_FEATURES if f not in selected_features]
    if missing_attackable:
        raise ValueError(f"Attackable features not in the model feature list: {missing_attackable}")
    frozen_but_attackable = [f for f in FROZEN_FEATURES_HINT if f in ATTACKABLE_FEATURES]
    if frozen_but_attackable:
        raise ValueError(f"These must stay frozen but are in ATTACKABLE_FEATURES: {frozen_but_attackable}")
    X, y, student_ids = prepare_features_and_label(df, selected_features)
    (X_train, X_test, X_val, y_train, y_test, y_val,
     id_train, id_test, id_val, split_report) = split_70_20_10_stratified(
        X, y, student_ids, random_state=random_state)
    print("Split report:")
    print(json.dumps(to_python(split_report), indent=4))
    print()
    # ---- load baseline RF + its tuned threshold ----
    rf_baseline = joblib.load(resolve_rf_model_path(rf_results_dir))
    rf_threshold, rf_threshold_source = load_rf_threshold(resolve_rf_summary_path(rf_results_dir))
    # ---- baseline MLP preprocessor + (optional) baseline surrogate ----
    preprocessor = joblib.load(mlp_preprocessor_path)
    X_train_pre = transform_to_float32(preprocessor, X_train)
    X_test_pre = transform_to_float32(preprocessor, X_test)
    X_val_pre = transform_to_float32(preprocessor, X_val)
    input_dim = X_train_pre.shape[1]
    if mlp_info["input_dim"] is not None and int(mlp_info["input_dim"]) != int(input_dim):
        raise ValueError("Preprocessed input_dim != source MLP checkpoint input_dim. Preprocessor/checkpoint mismatch.")
    base_mlp_model = build_mlp_from_state(mlp_info, input_dim=input_dim, device=device)
    base_art_mlp = wrap_mlp_for_art(base_mlp_model, input_dim=input_dim)
    base_mask, numeric_cols = make_preprocessed_attack_mask(preprocessor, selected_features, input_dim)
    # ---- EVAL surrogate: the BASELINE MLP (gradient source for the FGSM) ----
    # Detect-and-default does not retrain the model, so the model it protects is the
    # baseline RF; by default the attack weapon is FGSM transferred from the baseline
    # MLP (the same weapon as the attack chapter), i.e. the strongest valid attack on
    # the model this defence actually deploys.
    if eval_mlp_model_path_arg:
        if not os.path.exists(eval_mlp_model_path_arg):
            raise FileNotFoundError(
                f"Eval surrogate checkpoint not found: {eval_mlp_model_path_arg}. "
                "Run the MLP adversarial training first (it produces "
                "oulad_mlp_defended_model_train_eps_1p0.pt), or pass --eval-mlp-model \"\" "
                "to fall back to the baseline MLP as the weapon."
            )
        eval_attack_source = os.path.basename(eval_mlp_model_path_arg)
        eval_mlp_model_path = eval_mlp_model_path_arg
        eval_mlp_preprocessor_path = eval_mlp_preprocessor_arg or mlp_preprocessor_path
        eval_mlp_info = read_mlp_checkpoint(eval_mlp_model_path, device)
        eval_preprocessor = joblib.load(eval_mlp_preprocessor_path)
        X_test_pre_eval = transform_to_float32(eval_preprocessor, X_test)
        X_val_pre_eval = transform_to_float32(eval_preprocessor, X_val)
        eval_input_dim = X_test_pre_eval.shape[1]
        if eval_mlp_info["input_dim"] is not None and int(eval_mlp_info["input_dim"]) != int(eval_input_dim):
            raise ValueError("Eval preprocessed input_dim != eval MLP checkpoint input_dim. Preprocessor/checkpoint mismatch.")
        if eval_mlp_info["selected_features"] is not None \
                and list(eval_mlp_info["selected_features"]) != selected_features:
            raise ValueError("The eval (defended) MLP was trained on a different feature list than the source MLP. They must match for the transfer eval to be valid.")
        eval_mlp_model = build_mlp_from_state(eval_mlp_info, input_dim=eval_input_dim, device=device)
        eval_art_mlp = wrap_mlp_for_art(eval_mlp_model, input_dim=eval_input_dim)
        eval_perturbable_mask, _ = make_preprocessed_attack_mask(eval_preprocessor, selected_features, eval_input_dim)
        print("Eval FGSM surrogate (SEPARATE from the model being defended):")
        print(f"  checkpoint  : {eval_mlp_model_path}")
        print(f"  preprocessor: {eval_mlp_preprocessor_path}")
        print(f"  hidden dims : {eval_mlp_info['hidden_dims']} | dropout: {eval_mlp_info['dropout']}")
        print(f"  eval_attack_source: {eval_attack_source}")
    else:
        eval_attack_source = "baseline_mlp"
        eval_mlp_model_path = mlp_model_path
        eval_mlp_preprocessor_path = mlp_preprocessor_path
        eval_art_mlp = base_art_mlp
        eval_preprocessor = preprocessor
        eval_perturbable_mask = base_mask
        X_test_pre_eval = X_test_pre
        X_val_pre_eval = X_val_pre
        print("Eval FGSM surrogate: baseline MLP (same-weapon fallback).")
    print()
    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)
    # baseline RF clean predictions define the true_positive_only attack scope
    base_test_pred, _ = evaluate_rf_plain(rf_baseline, X_test, y_test.values, rf_threshold)
    base_val_pred, _ = evaluate_rf_plain(rf_baseline, X_val, y_val.values, rf_threshold)
    # ---- fit the Mahalanobis validator (RAW space) + calibrate on CLEAN TEST ----
    validator = InputValidator(ATTACKABLE_FEATURES).fit(X_train, numeric_cols)
    clean_test_scores = validator.mahalanobis_scores(X_test[selected_features])
    maha_threshold = float(np.quantile(clean_test_scores, 1.0 - detection_target_fpr))
    print(f"Baseline RF threshold : {rf_threshold} | source: {rf_threshold_source}")
    print(f"Detector covariance   : empirical covariance of raw attackable features + ridge 1e-6, pseudo-inverse "
          f"(condition number = {validator.condition_number:.3e})")
    print(f"Mahalanobis threshold : {maha_threshold} (calibrated to {detection_target_fpr} FPR on clean test)")
    print()
    # ---- generate adversarial examples once per eval epsilon, per split ----
    adv_by_eps = {}
    for eval_epsilon in eval_epsilon_list:
        adv_by_eps[eval_epsilon] = {}
        for split_name, X_split, X_split_pre_eval, y_split, clean_pred in [
            ("test", X_test, X_test_pre_eval, y_test, base_test_pred),
            ("validation", X_val, X_val_pre_eval, y_val, base_val_pred),
        ]:
            X_adv_raw, attack_idx = generate_fgsm_for_scope(
                art_mlp=eval_art_mlp, X_clean_raw=X_split, X_clean_preprocessed=X_split_pre_eval,
                y_true=y_split.values, clean_pred=clean_pred, epsilon=eval_epsilon,
                attack_mode=attack_mode, attack_scope=eval_attack_scope,
                perturbable_mask=eval_perturbable_mask, preprocessor=eval_preprocessor,
                selected_features=selected_features, train_min=train_min, train_max=train_max,
                enforce_consistency=enforce_consistency,
            )
            adv_by_eps[eval_epsilon][split_name] = (X_adv_raw, attack_idx)
    all_results = []
    raw_clean = {"test": X_test[selected_features], "validation": X_val[selected_features]}
    # ---- undefended reference (no defence) ----
    for split_name in ["test", "validation"]:
        y_split = y_test if split_name == "test" else y_val
        res, _ = evaluate_with_defence(rf_baseline, raw_clean[split_name], y_split.values, rf_threshold,
                                       validator, maha_threshold, "none", "Undefended Clean", split_name, 0.0)
        all_results.append(res)
    for eval_epsilon in eval_epsilon_list:
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
            res, _ = evaluate_with_defence(rf_baseline, X_adv_raw, y_split.values, rf_threshold,
                                           validator, maha_threshold, "none",
                                           f"Undefended FGSM {attack_mode}", split_name, eval_epsilon)
            res["attacked_rows_count"] = int(attack_idx.sum())
            all_results.append(res)
    # ---- defended: detect-and-default, per action ----
    for action in actions:
        for split_name in ["test", "validation"]:  # clean input (cost of the defence)
            y_split = y_test if split_name == "test" else y_val
            res, cm = evaluate_with_defence(rf_baseline, raw_clean[split_name], y_split.values, rf_threshold,
                                            validator, maha_threshold, action,
                                            f"Defended Clean ({action})", split_name, 0.0)
            all_results.append(res)
            if cm is not None:
                save_confusion_matrix(cm, os.path.join(output_dir, f"oulad_rf_defended_clean_{action}_{split_name}_cm.csv"))
        for eval_epsilon in eval_epsilon_list:  # adversarial input (the robust number)
            for split_name in ["test", "validation"]:
                y_split = y_test if split_name == "test" else y_val
                X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
                res, cm = evaluate_with_defence(rf_baseline, X_adv_raw, y_split.values, rf_threshold,
                                                validator, maha_threshold, action,
                                                f"Defended FGSM {attack_mode} ({action})", split_name, eval_epsilon)
                res["attacked_rows_count"] = int(attack_idx.sum())
                all_results.append(res)
                if cm is not None:
                    save_confusion_matrix(cm, os.path.join(
                        output_dir, f"oulad_rf_defended_fgsm_{action}_{split_name}_cm_eval_eps_{str(eval_epsilon).replace('.', 'p')}.csv"))
    # ---- per-student trace (one row per student, per split + eval epsilon) ----
    # This is what the DnD check-students script consumes. It records the
    # undefended clean/adversarial RF predictions (to define attacked / hidden),
    # the Mahalanobis flag, and the per-action defended decision under attack.
    student_trace_files = []
    for eval_epsilon in eval_epsilon_list:
        eps_tag = str(eval_epsilon).replace(".", "p")
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            id_split = id_test if split_name == "test" else id_val
            X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
            trace_path = os.path.join(
                output_dir, f"oulad_rf_dnd_{split_name}_student_trace_eval_eps_{eps_tag}.csv")
            save_student_trace(
                model=rf_baseline, X_clean_raw=raw_clean[split_name], X_adv_raw=X_adv_raw,
                y_true=y_split.values, student_ids=id_split.values, threshold=rf_threshold,
                validator=validator, maha_threshold=maha_threshold, actions=actions,
                attack_idx=attack_idx, eval_epsilon=eval_epsilon, split=split_name,
                selected_features=selected_features, path=trace_path,
            )
            student_trace_files.append(trace_path)
            print(f"Saved student trace: {trace_path}")
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "oulad_rf_detect_and_default_results.csv")
    results_df.to_csv(results_path, index=False)
    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "defence": "Random Forest detect-and-default (Mahalanobis flag -> override to dropout / abstain)",
        "detector": ("Mahalanobis distance over the RAW attackable numeric features (input-global, not "
                     "class-conditional), with the empirical covariance of the raw attackable features "
                     "(ridge 1e-6 on the diagonal, pseudo-inverse)."),
        "detector_space": "raw_feature_space",
        "detector_covariance_estimator": "np.cov on the raw attackable features + ridge 1e-6 on the diagonal, inverted with np.linalg.pinv (empirical covariance, no standardisation, no shrinkage)",
        "detector_ridge": validator.ridge,
        "detector_condition_number": validator.condition_number,
        "override_rule": "A flagged row's prediction is set to dropout (default_dropout), or the row is rejected (abstain).",
        "target_model": "Baseline Random Forest (undefended, unchanged)",
        "gradient_model": ("MLP-defended surrogate (adversarial-training checkpoint)"
                           if eval_attack_source != "baseline_mlp" else "Baseline MLP"),
        "attack_used_for_eval": "FGSM (transfer from the MLP surrogate)",
        "eval_attack_source": eval_attack_source,
        "eval_mlp_model_path": eval_mlp_model_path,
        "eval_mlp_preprocessor_path": eval_mlp_preprocessor_path,
        "source_mlp_model_path": mlp_model_path,
        "source_mlp_preprocessor_path": mlp_preprocessor_path,
        "attack_mode": attack_mode,
        "eval_attack_scope": eval_attack_scope,
        "eval_epsilon_list": eval_epsilon_list,
        "enforce_derived_consistency": bool(enforce_consistency),
        "credit_record_rule": CREDIT_RULE_TEXT,
        "actions": actions,
        "rf_threshold": rf_threshold,
        "rf_threshold_source": rf_threshold_source,
        "mahalanobis_threshold": maha_threshold,
        "detection_target_fpr": detection_target_fpr,
        "feature_list_source": feature_source,
        "n_raw_features": len(selected_features),
        "comparison_note": ("Both RF defences (adversarial training in adv_train_rf.py and "
                            "detect-and-default here) are evaluated with FGSM transferred from an MLP surrogate. "
                            "Adversarial training is evaluated with the adversarially trained MLP as the surrogate; "
                            f"the surrogate used in this detect-and-default run is: {eval_attack_source}."),
        "note_abstain": "For action=abstain, accuracy/recall are computed only on ACCEPTED (non-flagged) rows; read them together with flag_rate.",
        "caveat_adaptive": "Detection assumes a non-adaptive attacker. An attacker aware of the Mahalanobis filter could constrain perturbations to stay near the training distribution and reduce detection.",
        "split": split_report,
        "results_file": results_path,
        "student_trace_files": student_trace_files,
        "student_trace_note": ("One row per student per (split, eval_epsilon). Columns: undefended "
                               "baseline RF clean/adv predictions, Mahalanobis flag on clean and adv "
                               "inputs, and per-action defended decision under attack. "
                               "successfully_attacked = real dropout caught clean but hidden under "
                               "attack; recovered_default_dropout / caught_abstain = that hidden row "
                               "was flagged by the detector."),
    }
    with open(os.path.join(output_dir, "oulad_rf_detect_and_default_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)
    print("=" * 80)
    print("RF DETECT-AND-DEFAULT DONE")
    print("=" * 80)
    print(f"Results saved to: {results_path}")
    return summary
# ============================================================
# ARGUMENTS
# ============================================================
def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]
def format_list(values):
    return ",".join(str(v) for v in values)
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Detect-and-default defence for Random Forest (Mahalanobis flag -> override) on oulad_early, "
                    "attacked with FGSM transferred from an MLP surrogate (baseline MLP by default).")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--feature-set", default="early_both", choices=["early_65", "early_153", "early_both"])
    parser.add_argument("--rf-results-dir", default=DEFAULT_RF_RESULTS_DIR)
    parser.add_argument("--mlp-dir", default=DEFAULT_MLP_DIR,
                        help="Source MLP dir: feature list + preprocessor (baseline MLP).")
    parser.add_argument("--eval-mlp-model", default=DEFAULT_EVAL_MLP_MODEL,
                        help="Gradient source for the eval FGSM. Default: the baseline MLP. Pass a defended checkpoint path to use it instead. "
                             "Pass \"\" to fall back to the baseline MLP as the weapon.")
    parser.add_argument("--eval-mlp-preprocessor", default=DEFAULT_EVAL_MLP_PREPROCESSOR,
                        help="Preprocessor for the eval surrogate. Default: reuse the --mlp-dir preprocessor.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--attack-mode", default="target_non_dropout",
                        choices=["untargeted", "target_non_dropout"])
    parser.add_argument("--eval-attack-scope", default="true_positive_only",
                        choices=["all", "dropout_only", "true_positive_only"])
    parser.add_argument("--eval-epsilons", default=format_list(EVAL_EPSILON_LIST),
                        help=f"Comma-separated eval epsilon list. Default: {format_list(EVAL_EPSILON_LIST)}")
    parser.add_argument("--detection-target-fpr", type=float, default=DEFAULT_DETECTION_TARGET_FPR)
    parser.add_argument("--actions", default="default_dropout,abstain",
                        help="Comma-separated actions for flagged rows: default_dropout,abstain.")
    # Domain-identity enforcement is HARDCODED ON (no flag): after clip/round the
    # per-year passed<=enrolled cap, credit totals, and CGPA are always re-imposed.
    return parser
def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    eval_mlp_model_arg = args.eval_mlp_model if args.eval_mlp_model else None
    run_detect_and_default_experiment(
        df=df, csv_path=args.csv_path, output_dir=args.output_dir,
        rf_results_dir=args.rf_results_dir, mlp_dir=args.mlp_dir,
        feature_set_hint=args.feature_set,
        eval_epsilon_list=parse_float_list(args.eval_epsilons), random_state=args.random_state,
        attack_mode=args.attack_mode, eval_attack_scope=args.eval_attack_scope,
        detection_target_fpr=args.detection_target_fpr, actions=actions,
        eval_mlp_model_path_arg=eval_mlp_model_arg,
        eval_mlp_preprocessor_arg=args.eval_mlp_preprocessor,
        enforce_consistency=True,  # hardcoded ON
    )
if __name__ == "__main__":
    main()