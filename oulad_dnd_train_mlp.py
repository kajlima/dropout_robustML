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
from art.attacks.evasion import ProjectedGradientDescent
# ============================================================
# DEFAULT CONFIG
# ============================================================
# Detect-and-default defence for the MLP on oulad_early.csv, RAW INPUT-GLOBAL
# variant. The Mahalanobis validator is fit in the shared RAW attackable-feature
# space (distance to the training mean under the empirical covariance,
# input-global and NOT class-conditional). This is the identical detector used for
# RF and XGBoost, so the detect-and-default detector is held constant across all
# three classifiers and the cross-model robustness comparison is controlled (only
# the classifier differs). A flagged row's prediction is OVERRIDDEN to 'dropout'
# (the safe / high-recall class). White-box PGD setup matches the standalone attack
# in pgd_mlp.py so the defence is tested against the same threat reported in the
# attack chapter.
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "oulad_early.csv"
DEFAULT_MLP_RESULTS_DIR = "mlp_results_oulad_early"
DEFAULT_OUTPUT_DIR = "mlp_dnd_feature_classcond_oulad_early_results"
DEFAULT_DETECTOR_COVARIANCE = "tied"  # unused; kept only for CLI compatibility (detector is raw input-global)
TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout
# epsilon is an L-inf bound in the MLP PREPROCESSED (standardized) space, same as
# pgd_mlp.py / adv_train_mlp.py.
EVAL_EPSILON_LIST = [1.0]
PGD_STEP = 0.01
# Match the standalone PGD attack (pgd_mlp.py used 100 iterations). Evaluating the
# so the default here is the full-strength threat. Override with --pgd-iterations.
PGD_ITERATIONS = 100
# detector: false-positive rate targeted when calibrating on the clean test split.
DEFAULT_DETECTION_TARGET_FPR = 0.05
DEFAULT_HIDDEN_DIMS = (128, 64, 32)
# Documentation only: detect-and-default does NOT select its own decision cutoff.
# It loads the operating threshold straight from the baseline MLP checkpoint (see
# load_mlp_checkpoint) and wraps the unchanged baseline model at that point.
# MIN_PRECISION records the precision floor the baseline used when it originally
# picked that threshold (max recall subject to dropout precision >= MIN_PRECISION,
# else F1-optimal); 0.73 to match mlp_model.py / adv_train_mlp.py on both datasets.
MIN_PRECISION = 0.73
# ============================================================
# FEATURE DEFINITIONS (early_both feature set on oulad_early.csv)
# ============================================================
# id_student and target are never predictors. The authoritative feature list is
# read from the MLP checkpoint at runtime; this fallback matches the 'early_both'
# set used by mlp_model.py / rf_model.py / xgboost_model.py.
# num_of_prev_attempts and studied_credits do NOT exist in this dataset version.
NON_FEATURE_COLS = ["id_student", TARGET_COL]
DEFAULT_SELECTED_FEATURES = [
    # static / enrolment (frozen)
    "module_credits",
    "code_module",
    "code_presentation",
    # prior-year credit record (years 1-3) + current-year enrolment
    "credit_enrolled_1styear",
    "credit_passed_1styear",
    "credit_enrolled_2ndyear",
    "credit_passed_2ndyear",
    "credit_enrolled_3rdyear",
    "credit_passed_3rdyear",
    "credit_enrolled_4thyear",
    # prior-year academic record
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
# ATTACKABLE FEATURES (identical to pgd_mlp.py / adv_train_mlp.py, 29 features)
# ============================================================
# Everything else stays frozen, including the one-hot module/presentation columns
# and module_credits.
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
# Explicitly frozen (for reporting only). id_student is not a model feature.
FROZEN_FEATURES_HINT = [
    "code_module",
    "code_presentation",
    "module_credits",
]
# Derived columns and how to recompute them from their (perturbed) components.
# Enforcement is ALWAYS ON (hardcoded, no flag), mirroring the always-on credit
# identities in the zenodo scripts. Verified on oulad_early.csv:
#   total_credit_enrolled = sum of enrolled years 1-3   (exact)
#   total_credit_passed   = sum of passed   years 1-3   (exact)
#   CGPA                  = mean(GPA_1styear, GPA_2ndyear, GPA_3rdyear) (exact to 2 dp)
DERIVED_FEATURE_COMPONENTS = {}  # CGPA and the credit totals were removed from the feature set
# Per-year (enrolled, passed) pairs (kept for reference; the credit rules are enforced
# in enforce_derived_consistency). Originally used to enforce passed <= enrolled on the
# final adversarial rows. Year 4 has only an 'enrolled' column (passed_4thyear is
# the outcome being predicted), so it has no pair and is not capped.
CREDIT_YEAR_PAIRS = [
    ("credit_enrolled_1styear", "credit_passed_1styear"),
    ("credit_enrolled_2ndyear", "credit_passed_2ndyear"),
    ("credit_enrolled_3rdyear", "credit_passed_3rdyear"),
]
# ============================================================
# DOMAIN CONSTRAINTS (calibrated to oulad_early.csv)
# ============================================================
# Integer-valued features -> round back to whole numbers after the attack.
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
# Decimal features -> round to 2 decimal places after the attack.
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
# Non-negative features. first_submit_day is deliberately left OUT: it is bounded
# only by the train min/max clip below (which is the primary realism bound for
# every feature). GPA_* and CGPA are on a 0-4 scale; all credit_* columns are
# non-negative too.
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
# MLP MODEL (must match the training MLP exactly)
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
    # SAME split logic and random_state as the MLP / RF / XGBoost scripts, so the
    # test/validation rows here are exactly the rows those models were evaluated on.
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
def resolve_mlp_model_path(mlp_results_dir):
    return resolve_first_existing([os.path.join(mlp_results_dir, "oulad_mlp_model.pt")], "MLP model (oulad_mlp_model.pt)")
def resolve_mlp_preprocessor_path(mlp_results_dir):
    return resolve_first_existing([os.path.join(mlp_results_dir, "oulad_mlp_preprocessor.pkl")], "MLP preprocessor (oulad_mlp_preprocessor.pkl)")
def resolve_mlp_summary_path(mlp_results_dir):
    path = os.path.join(mlp_results_dir, "oulad_mlp_summary.json")
    return path if os.path.exists(path) else None
def find_threshold_in_dict(obj):
    if isinstance(obj, dict):
        for key in ["best_threshold", "threshold"]:
            if key in obj and obj[key] is not None:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            found = find_threshold_in_dict(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_threshold_in_dict(item)
            if found is not None:
                return found
    return None
def load_summary_threshold(summary_path):
    if summary_path is None or not os.path.exists(summary_path):
        return None, None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return find_threshold_in_dict(summary), summary_path
def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
def load_mlp_checkpoint(model_path, input_dim_from_data, device, summary_threshold=None):
    obj = load_torch_checkpoint(model_path, device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        input_dim = int(obj.get("input_dim", input_dim_from_data))
        hidden_dims = tuple(obj.get("hidden_dims", DEFAULT_HIDDEN_DIMS))
        dropout = float(obj.get("dropout", 0.30))
        checkpoint_threshold = obj.get("best_threshold", None)
        selected_features = obj.get("selected_features", None)
        feature_set = obj.get("feature_set", None)
        state_dict = obj["model_state_dict"]
    else:
        input_dim = input_dim_from_data
        hidden_dims = DEFAULT_HIDDEN_DIMS
        dropout = 0.30
        checkpoint_threshold = None
        selected_features = None
        feature_set = None
        state_dict = obj
    if input_dim != input_dim_from_data:
        raise ValueError(
            f"Model input dim ({input_dim}) differs from data ({input_dim_from_data}). "
            "Preprocessor/checkpoint mismatch.")
    if checkpoint_threshold is not None:
        threshold = float(checkpoint_threshold)
        threshold_source = "oulad_mlp_model.pt"
    elif summary_threshold is not None:
        threshold = float(summary_threshold)
        threshold_source = "oulad_mlp_summary.json"
    else:
        threshold = 0.5
        threshold_source = "fallback_0.5"
    model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "selected_features": selected_features,
        "feature_set": feature_set,
    }
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
def make_preprocessed_attack_mask(preprocessor, selected_features, input_dim, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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
def preprocessed_adv_to_raw_and_scaled(
    X_adv_preprocessed, X_clean_preprocessed, X_clean_raw, selected_features,
    preprocessor, train_min, train_max, perturbable_mask_preprocessed,
    attacked_row_mask=None,
):
    # Convert adversarial preprocessed data back to raw space. Only the numeric
    # block is inverse-scaled; frozen features and non-attacked rows are restored
    # from the clean data so the attack scope stays strict.
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    scaler = get_scaler_from_preprocessor(preprocessor)
    n_num = len(numeric_cols)
    adv_numeric_scaled = X_adv_preprocessed[:, :n_num]
    adv_numeric_raw = scaler.inverse_transform(adv_numeric_scaled)
    X_adv_raw = X_clean_raw[selected_features].reset_index(drop=True).copy()
    clean_reset = X_clean_raw[selected_features].reset_index(drop=True)
    # write adversarial numeric block back to raw columns
    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]
    # restore non-attackable raw features
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    # clip numeric values to train min/max (primary realism bound)
    for col in numeric_cols:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=train_min[col], upper=train_max[col])
    # domain constraints
    for col in NONNEGATIVE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0)
    for col in PERCENTAGE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0, upper=100)
    # rounding: integer counts -> whole numbers; decimals -> 2 dp. Done after
    # clipping so the rounded value still obeys the range bounds.
    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col]).astype(float)
    for col in DECIMAL_2DP_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col].astype(float), 2)
    # enforce derived-feature consistency LAST, on the finalized clipped+rounded
    # components, so the rows are domain-valid at their final values (per-year
    # passed<=enrolled, exact totals, CGPA=mean(GPA)). Always on, like zenodo.
    X_adv_raw = enforce_derived_consistency(X_adv_raw)
    X_adv_raw = enforce_activity_consistency(X_adv_raw, clean_reset, train_min, train_max)
    assert_feasible(oulad_feasibility_violations(X_adv_raw), attacked_row_mask, 'OULAD projection')
    # restore untouched rows completely to their clean values
    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        not_attacked = ~attacked_row_mask
        if not_attacked.any():
            for col in selected_features:
                values = clean_reset.loc[not_attacked, col]
                if pd.api.types.is_numeric_dtype(X_adv_raw[col]):
                    values = values.astype(X_adv_raw[col].dtype, copy=False)
                X_adv_raw.loc[not_attacked, col] = values.values
    # re-transform to preprocessed space and hard-restore frozen dims / rows
    X_projected_preprocessed = transform_to_float32(preprocessor, X_adv_raw[selected_features])
    frozen = perturbable_mask_preprocessed == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]
    if attacked_row_mask is not None:
        not_attacked = ~np.asarray(attacked_row_mask).astype(bool)
        if not_attacked.any():
            X_projected_preprocessed[not_attacked, :] = X_clean_preprocessed[not_attacked, :]
    return X_adv_raw[selected_features], X_projected_preprocessed
# ============================================================
# MLP PREDICTION
# ============================================================
def predict_mlp_proba(model, X_preprocessed, device, batch_size=4096):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_preprocessed), batch_size):
            batch = torch.tensor(X_preprocessed[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.append(batch_probs.cpu().numpy())
    return np.concatenate(probs)
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
def generate_pgd_for_scope(
    art_mlp, X_clean_preprocessed, y_true, clean_pred, epsilon, pgd_step, pgd_iterations,
    attack_mode, attack_scope, perturbable_mask, batch_size=128,
):
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()
    if int(attack_idx.sum()) == 0:
        return X_adv_preprocessed, attack_idx
    attack_y, targeted = make_attack_labels(attack_mode, y_true=np.asarray(y_true)[attack_idx])
    effective_step = min(float(pgd_step), float(epsilon))
    pgd = ProjectedGradientDescent(
        estimator=art_mlp, eps=float(epsilon), eps_step=effective_step,
        max_iter=int(pgd_iterations), targeted=targeted, batch_size=batch_size,
    )
    X_adv_subset = pgd.generate(x=X_clean_preprocessed[attack_idx], y=attack_y, mask=perturbable_mask)
    frozen = perturbable_mask == 0
    X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
    X_adv_preprocessed[attack_idx] = X_adv_subset
    return X_adv_preprocessed, attack_idx
def generate_pgd_raw_and_scaled_for_scope(
    art_mlp, X_clean_raw, X_clean_preprocessed, y_true, clean_pred, epsilon, pgd_step,
    pgd_iterations, attack_mode, attack_scope, perturbable_mask, preprocessor,
    selected_features, train_min, train_max,
):
    X_adv_pre, attack_idx = generate_pgd_for_scope(
        art_mlp=art_mlp, X_clean_preprocessed=X_clean_preprocessed, y_true=y_true,
        clean_pred=clean_pred, epsilon=epsilon, pgd_step=pgd_step, pgd_iterations=pgd_iterations,
        attack_mode=attack_mode, attack_scope=attack_scope, perturbable_mask=perturbable_mask,
    )
    X_adv_raw, _ = preprocessed_adv_to_raw_and_scaled(
        X_adv_preprocessed=X_adv_pre, X_clean_preprocessed=X_clean_preprocessed,
        X_clean_raw=X_clean_raw, selected_features=selected_features, preprocessor=preprocessor,
        train_min=train_min, train_max=train_max, perturbable_mask_preprocessed=perturbable_mask,
        attacked_row_mask=attack_idx,
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
# EVALUATION AND SAVING
# ============================================================
def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)
# ============================================================
# DETECT-AND-DEFAULT EVALUATION
# ============================================================
def evaluate_with_defence(model, X_raw, selected_features, preprocessor, y_true, threshold,
                          validator, maha_threshold, action, condition, split, eval_epsilon, device):
    # MLP predicts on the preprocessed vector, so X_raw is preprocessed here first.
    y_true = np.asarray(y_true).astype(int)
    X_pre = transform_to_float32(preprocessor, X_raw[selected_features])
    y_proba = predict_mlp_proba(model, X_pre, device)
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
        eval_mask = ~flags
    else:
        raise ValueError("action must be 'none', 'default_dropout', or 'abstain'.")
    yt = y_true[eval_mask]
    yp = y_pred_final[eval_mask]
    result = {
        "model": "MLP",
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
def save_student_trace(model, preprocessor, device, X_clean_raw, X_adv_raw, y_true,
                       student_ids, threshold, validator, maha_threshold, actions,
                       attack_idx, eval_epsilon, split, selected_features, path):
    # One row per student for a single (split, eval_epsilon). This is the file the
    # DnD check-students script reads. Row order is positional: X_clean_raw,
    # X_adv_raw, y_true, student_ids and attack_idx all come from the same split in
    # the same order (X_adv_raw is reset_index'd inside preprocessed_adv_to_raw_and_
    # scaled, and the MLP predicts positionally), so index i is the same student in
    # all of them.
    #
    # The detect-and-default MLP is the UNCHANGED baseline MLP; the only thing the
    # defence adds is the Mahalanobis flag + an override. So a hidden dropout is
    # only recovered when the detector flags its adversarial row (flagged_adv).
    y_true = np.asarray(y_true).astype(int)
    ids = np.asarray(student_ids)
    # undefended baseline MLP on clean and on adversarial inputs (preprocessed)
    clean_pre = transform_to_float32(preprocessor, X_clean_raw[selected_features])
    adv_pre = transform_to_float32(preprocessor, X_adv_raw[selected_features])
    clean_proba = predict_mlp_proba(model, clean_pre, device)
    adv_proba = predict_mlp_proba(model, adv_pre, device)
    clean_pred = (clean_proba >= threshold).astype(int)
    adv_pred = (adv_proba >= threshold).astype(int)
    # raw input-global Mahalanobis detector scores + flags (raw attackable-feature space)
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
        # undefended baseline MLP: clean vs adversarial (defines attacked / hidden)
        "mlp_clean_pred_label": clean_pred,
        "mlp_clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        "mlp_clean_prob_dropout": clean_proba,
        "mlp_adv_pred_label": adv_pred,
        "mlp_adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        "mlp_adv_prob_dropout": adv_proba,
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
    # real dropout, caught by the clean baseline MLP, hidden once attacked.
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
    df, csv_path, output_dir, mlp_results_dir, feature_set_hint, eval_epsilon_list,
    pgd_step, pgd_iterations, random_state, attack_mode, eval_attack_scope,
    detection_target_fpr, actions,
    detector_covariance=DEFAULT_DETECTOR_COVARIANCE,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("MLP DETECT-AND-DEFAULT (MAHALANOBIS FLAG -> OVERRIDE, oulad_early)")
    print("=" * 80)
    print(f"CSV path            : {csv_path}")
    print(f"Output dir          : {output_dir}")
    print(f"Attack mode         : {attack_mode}")
    print(f"Eval attack scope   : {eval_attack_scope}")
    print(f"PGD step / iters    : {pgd_step} / {pgd_iterations}")
    print(f"Derived consist.    : always ON (30-credit grid, 30-120 per year, year4 = 360 - sum(years 1-3), passed <= enrolled; CGPA and credit totals not used)")
    print(f"Detection target FPR: {detection_target_fpr}")
    print(f"Actions             : {actions}")
    print(f"Device              : {device}")
    print()
    mlp_model_path = resolve_mlp_model_path(mlp_results_dir)
    mlp_preprocessor_path = resolve_mlp_preprocessor_path(mlp_results_dir)
    summary_threshold, _ = load_summary_threshold(resolve_mlp_summary_path(mlp_results_dir))
    peek = load_torch_checkpoint(mlp_model_path, device)
    ckpt_selected = peek.get("selected_features") if isinstance(peek, dict) else None
    if ckpt_selected is not None:
        selected_features = list(ckpt_selected)
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
    preprocessor = joblib.load(mlp_preprocessor_path)
    X_train_pre = transform_to_float32(preprocessor, X_train)
    X_test_pre = transform_to_float32(preprocessor, X_test)
    X_val_pre = transform_to_float32(preprocessor, X_val)
    input_dim = X_train_pre.shape[1]
    mlp_ckpt = load_mlp_checkpoint(mlp_model_path, input_dim, device, summary_threshold=summary_threshold)
    mlp_model = mlp_ckpt["model"]
    mlp_threshold = mlp_ckpt["threshold"]
    art_mlp = wrap_mlp_for_art(mlp_model, input_dim=input_dim)
    perturbable_mask, numeric_cols = make_preprocessed_attack_mask(preprocessor, selected_features, input_dim, output_dir)
    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)
    base_test_pred = (predict_mlp_proba(mlp_model, X_test_pre, device) >= mlp_threshold).astype(int)
    base_val_pred = (predict_mlp_proba(mlp_model, X_val_pre, device) >= mlp_threshold).astype(int)
    validator = InputValidator(ATTACKABLE_FEATURES).fit(X_train, numeric_cols)
    detector_desc = ("Mahalanobis distance over the RAW attackable numeric features (input-global, not "
                     "class-conditional), with the empirical covariance of the raw attackable features "
                     "(ridge 1e-6 on the diagonal, pseudo-inverse). "
                     "Identical to the RF/XGBoost detect-and-default detector so the cross-model "
                     "comparison is controlled (only the classifier differs).")
    print(f"Detector            : raw_global ({detector_desc})")
    print(f"Detector covariance : empirical covariance of raw attackable features + ridge 1e-6, pseudo-inverse "
          f"(condition number = {validator.condition_number:.3e})")
    clean_test_scores = validator.mahalanobis_scores(X_test[selected_features])
    maha_threshold = float(np.quantile(clean_test_scores, 1.0 - detection_target_fpr))
    print(f"Baseline MLP threshold: {mlp_threshold} | source: {mlp_ckpt['threshold_source']}")
    print(f"Mahalanobis threshold : {maha_threshold} (calibrated to {detection_target_fpr} FPR on clean test)")
    print()
    adv_by_eps = {}
    for eval_epsilon in eval_epsilon_list:
        adv_by_eps[eval_epsilon] = {}
        for split_name, X_split, X_split_pre, y_split, clean_pred in [
            ("test", X_test, X_test_pre, y_test, base_test_pred),
            ("validation", X_val, X_val_pre, y_val, base_val_pred),
        ]:
            X_adv_raw, attack_idx = generate_pgd_raw_and_scaled_for_scope(
                art_mlp=art_mlp, X_clean_raw=X_split, X_clean_preprocessed=X_split_pre,
                y_true=y_split.values, clean_pred=clean_pred, epsilon=eval_epsilon,
                pgd_step=pgd_step, pgd_iterations=pgd_iterations, attack_mode=attack_mode,
                attack_scope=eval_attack_scope, perturbable_mask=perturbable_mask,
                preprocessor=preprocessor, selected_features=selected_features,
                train_min=train_min, train_max=train_max,
            )
            adv_by_eps[eval_epsilon][split_name] = (X_adv_raw, attack_idx)
    all_results = []
    raw_clean = {"test": X_test[selected_features], "validation": X_val[selected_features]}
    # undefended reference
    for split_name in ["test", "validation"]:
        y_split = y_test if split_name == "test" else y_val
        res, cm = evaluate_with_defence(mlp_model, raw_clean[split_name], selected_features, preprocessor,
                                        y_split.values, mlp_threshold, validator, maha_threshold, "none",
                                        "Undefended Clean", split_name, 0.0, device)
        all_results.append(res)
    for eval_epsilon in eval_epsilon_list:
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
            res, cm = evaluate_with_defence(mlp_model, X_adv_raw, selected_features, preprocessor,
                                            y_split.values, mlp_threshold, validator, maha_threshold, "none",
                                            f"Undefended PGD {attack_mode}", split_name, eval_epsilon, device)
            res["attacked_rows_count"] = int(attack_idx.sum())
            all_results.append(res)
    # defended: detect-and-default
    for action in actions:
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            res, cm = evaluate_with_defence(mlp_model, raw_clean[split_name], selected_features, preprocessor,
                                            y_split.values, mlp_threshold, validator, maha_threshold, action,
                                            f"Defended Clean ({action})", split_name, 0.0, device)
            all_results.append(res)
            if cm is not None:
                save_confusion_matrix(cm, os.path.join(output_dir, f"oulad_mlp_defended_clean_{action}_{split_name}_cm.csv"))
        for eval_epsilon in eval_epsilon_list:
            for split_name in ["test", "validation"]:
                y_split = y_test if split_name == "test" else y_val
                X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
                res, cm = evaluate_with_defence(mlp_model, X_adv_raw, selected_features, preprocessor,
                                                y_split.values, mlp_threshold, validator, maha_threshold, action,
                                                f"Defended PGD {attack_mode} ({action})", split_name, eval_epsilon, device)
                res["attacked_rows_count"] = int(attack_idx.sum())
                all_results.append(res)
                if cm is not None:
                    save_confusion_matrix(cm, os.path.join(
                        output_dir, f"oulad_mlp_defended_pgd_{action}_{split_name}_cm_eval_eps_{str(eval_epsilon).replace('.', 'p')}.csv"))
    # ---- per-student trace (one row per student, per split + eval epsilon) ----
    # This is what the DnD check-students script consumes. It records the
    # undefended clean/adversarial MLP predictions (to define attacked / hidden),
    # the Mahalanobis flag, and the per-action defended decision under attack.
    student_trace_files = []
    for eval_epsilon in eval_epsilon_list:
        eps_tag = str(eval_epsilon).replace(".", "p")
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            id_split = id_test if split_name == "test" else id_val
            X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
            trace_path = os.path.join(
                output_dir, f"oulad_mlp_dnd_{split_name}_student_trace_eval_eps_{eps_tag}.csv")
            save_student_trace(
                model=mlp_model, preprocessor=preprocessor, device=device,
                X_clean_raw=raw_clean[split_name], X_adv_raw=X_adv_raw,
                y_true=y_split.values, student_ids=id_split.values, threshold=mlp_threshold,
                validator=validator, maha_threshold=maha_threshold, actions=actions,
                attack_idx=attack_idx, eval_epsilon=eval_epsilon, split=split_name,
                selected_features=selected_features, path=trace_path,
            )
            student_trace_files.append(trace_path)
            print(f"Saved student trace: {trace_path}")
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "oulad_mlp_detect_and_default_results.csv")
    results_df.to_csv(results_path, index=False)
    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "defence": "MLP detect-and-default (Mahalanobis flag -> override to dropout)",
        "detector": detector_desc,
        "detector_type": "raw_global",
        "detector_space": "raw_feature_space",
        "detector_covariance": "Empirical covariance of the raw attackable features, ridge 1e-6, pseudo-inverse (input-global; not class-conditional)",
        "detector_covariance_estimator": "np.cov on the raw attackable features + ridge 1e-6 on the diagonal, inverted with np.linalg.pinv (empirical covariance, no standardisation, no shrinkage)",
        "detector_ridge": validator.ridge,
        "detector_condition_number": validator.condition_number,
        "override_rule": "A flagged row's prediction is set to dropout (default_dropout), or the row is rejected (abstain).",
        "target_model": "Baseline MLP (undefended, unchanged)",
        "attack_used_for_eval": "PGD (white-box)",
        "attack_mode": attack_mode,
        "eval_attack_scope": eval_attack_scope,
        "pgd_step": pgd_step,
        "pgd_iterations": pgd_iterations,
        "enforce_derived_consistency": True,
        "credit_record_rule": CREDIT_RULE_TEXT,
        "derived_consistency_rule": ("Always on. On the final clipped+rounded rows: "
                                     "per-year credit_passed<=credit_enrolled on the 30-credit grid; "
                                     "CGPA and the credit totals are not model features."),
        "eval_epsilon_list": eval_epsilon_list,
        "epsilon_space": "L-inf bound in the MLP preprocessed/standardized feature space",
        "actions": actions,
        "mlp_threshold": mlp_threshold,
        "mlp_threshold_source": mlp_ckpt["threshold_source"],
        "mahalanobis_threshold": maha_threshold,
        "detection_target_fpr": detection_target_fpr,
        "mlp_model_path": mlp_model_path,
        "mlp_preprocessor_path": mlp_preprocessor_path,
        "feature_list_source": feature_source,
        "n_raw_features": len(selected_features),
        "n_attackable_raw_features": len(ATTACKABLE_FEATURES),
        "attackable_raw_features": list(ATTACKABLE_FEATURES),
        "frozen_raw_features": list(FROZEN_FEATURES_HINT),
        "note_abstain": "For action=abstain, accuracy/recall are computed only on ACCEPTED (non-flagged) rows; read them together with flag_rate.",
        "caveat_adaptive": "Detection assumes a non-adaptive attacker. An attacker aware of the Mahalanobis filter could constrain perturbations to stay near the training distribution and reduce detection.",
        "split": split_report,
        "results_file": results_path,
        "student_trace_files": student_trace_files,
        "student_trace_note": ("One row per student per (split, eval_epsilon). Columns: undefended "
                               "baseline MLP clean/adv predictions, Mahalanobis flag on clean and adv "
                               "inputs, and per-action defended decision under attack. "
                               "successfully_attacked = real dropout caught clean but hidden under "
                               "attack; recovered_default_dropout / caught_abstain = that hidden row "
                               "was flagged by the detector."),
    }
    with open(os.path.join(output_dir, "oulad_mlp_detect_and_default_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)
    print("=" * 80)
    print("MLP DETECT-AND-DEFAULT DONE")
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
        description="Detect-and-default defence for the MLP using a RAW input-global "
                    "Mahalanobis detector (identical to the RF/XGBoost detector) and white-box PGD on oulad_early.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--feature-set", default="early_both", choices=["early_65", "early_153", "early_both"])
    parser.add_argument("--mlp-results-dir", default=DEFAULT_MLP_RESULTS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--attack-mode", default="target_non_dropout",
                        choices=["untargeted", "target_non_dropout"])
    parser.add_argument("--eval-attack-scope", default="true_positive_only",
                        choices=["all", "dropout_only", "true_positive_only"])
    parser.add_argument("--eval-epsilons", default=format_list(EVAL_EPSILON_LIST),
                        help=f"Comma-separated eval epsilon list. Default: {format_list(EVAL_EPSILON_LIST)}")
    parser.add_argument("--pgd-step", type=float, default=PGD_STEP)
    parser.add_argument("--pgd-iterations", type=int, default=PGD_ITERATIONS,
                        help=f"PGD iterations for the eval attack. Default: {PGD_ITERATIONS} "
                             "(matches the standalone pgd_mlp.py attack).")
    parser.add_argument("--detection-target-fpr", type=float, default=DEFAULT_DETECTION_TARGET_FPR)
    parser.add_argument("--detector-covariance", default=DEFAULT_DETECTOR_COVARIANCE,
                        choices=["tied", "per_class"],
                        help="Not used: the detector is always the raw input-global Mahalanobis detector. "
                             "Kept only so older command lines still run.")
    parser.add_argument("--actions", default="default_dropout,abstain",
                        help="Comma-separated actions for flagged rows: default_dropout,abstain. Default: default_dropout,abstain")
    # Derived-consistency enforcement is ALWAYS ON (hardcoded, no flag), matching
    # pgd_mlp.py / adv_train_mlp.py so the eval PGD examples are constructed
    # identically and stay domain-valid at their final values.
    return parser
def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    run_detect_and_default_experiment(
        df=df, csv_path=args.csv_path, output_dir=args.output_dir,
        mlp_results_dir=args.mlp_results_dir, feature_set_hint=args.feature_set,
        eval_epsilon_list=parse_float_list(args.eval_epsilons), pgd_step=args.pgd_step,
        pgd_iterations=args.pgd_iterations, random_state=args.random_state,
        attack_mode=args.attack_mode, eval_attack_scope=args.eval_attack_scope,
        detection_target_fpr=args.detection_target_fpr, actions=actions,
        detector_covariance=args.detector_covariance,
    )
if __name__ == "__main__":
    main()