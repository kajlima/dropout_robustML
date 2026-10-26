# REVISION NOTE (2026-09-29): CGPA, total_credit_enrolled and total_credit_passed were
# removed from every feature list in this script (feature set 32 -> 29, attackable 29 -> 26,
# MLP input 41 -> 38 dims). The columns may still exist in oulad_early.csv but are not used.
# Older comments below that still mention them describe the previous version.
import os
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
# ============================================================
# WHAT THIS SCRIPT DOES  (Option C for detect-and-default)
# ============================================================
# After-defence proximity and sensitivity for oulad_early.csv, DETECT-AND-DEFAULT
# defence. Unlike adversarial training, DnD does NOT retrain the model and does
# NOT change the adversarial input: it keeps the baseline model, runs the same
# attack, then flags inputs with a Mahalanobis validator and OVERRIDES flagged
# predictions to dropout (or abstains). So the input geometry equals the attack on
# the baseline model; the defence's value shows up in DETECTION and OVERRIDE, not
# in new distances. This script reports both views (Option C):
#
#   (A) Detection / override effect:
#       n_flagged, flag_rate, detection_recall_on_attacked, and the dropout hiding
#       rate BEFORE the defence vs AFTER default_dropout (hides recovered).
#   (B) Proximity / sensitivity split by detector outcome on the adversarial input:
#       row_scope in {all_rows, attacked_rows_only, attacked_flagged,
#       attacked_accepted}. attacked_accepted are the adversarial rows that slipped
#       past the filter -- the interesting ones (a good detector should leave only
#       the small-perturbation evasions).
#
# The three DnD cases reproduce the project's dnd_train_* scripts exactly:
#   RF  : baseline RF,  FGSM from the BASELINE MLP,  raw-space Mahalanobis over attackable features
#   XGB : baseline XGB, FGSM from the BASELINE MLP,  raw-space Mahalanobis over attackable features
#   MLP : baseline MLP, white-box PGD (grad=target), raw-space input-global Mahalanobis over
#         attackable features (identical detector to RF/XGB, for a controlled comparison)
# all at eps 1.0, target_non_dropout, true_positive_only, detector calibrated to a
# 5% false-positive rate on the clean test split. The detector covariance is the
# empirical covariance of the raw attackable features (ridge 1e-6, pseudo-inverse),
# the same estimator as in the dnd_train_* scripts.

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "oulad_early.csv"
DEFAULT_MLP_DIR = "mlp_results_oulad_early"
DEFAULT_RF_DIR = "rf_results_oulad_early"
DEFAULT_XGB_DIR = "xgb_results_oulad_early"
DEFAULT_OUTPUT_DIR = "proximity_sensitivity_detect_and_default_oulad_early_results"
# FGSM gradient source for RF/XGB DnD = the baseline MLP by default (None); DnD wraps the baseline model.
DEFAULT_EVAL_MLP_MODEL = None
DEFAULT_EPSILON = 1.0
DEFAULT_PGD_STEP = 0.01
DEFAULT_PGD_ITERATIONS = 100
DEFAULT_DETECTION_TARGET_FPR = 0.05
DEFAULT_DETECTOR_COVARIANCE = "tied"  # retained for CLI compatibility; NOT used (detector is raw input-global)
DEFAULT_ATTACK_MODE = "target_non_dropout"
DEFAULT_ATTACK_SCOPE = "true_positive_only"
TARGET_COL = "target"
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
EPS = 1e-8
DEFAULT_HIDDEN_DIMS = (128, 64, 32)
# ============================================================
# FEATURE SET (oulad_early, 29 features after removing CGPA and the credit totals; matches the attack/defence scripts)
# ============================================================
DEFAULT_SELECTED_FEATURES = [
    "module_credits", "code_module", "code_presentation",
    "credit_enrolled_1styear", "credit_passed_1styear", "credit_enrolled_2ndyear",
    "credit_passed_2ndyear", "credit_enrolled_3rdyear", "credit_passed_3rdyear",
    "credit_enrolled_4thyear", 
    "GPA_1styear", "GPA_2ndyear", "GPA_3rdyear", 
    "early_avg_score_65", "early_n_submitted_65", "early_first_submit_day_65", "early_total_clicks_65",
    "early_active_days_65", "early_unique_resources_65", "early_quiz_clicks_65", "early_forum_clicks_65",
    "early_avg_score_153", "early_n_submitted_153", "early_first_submit_day_153", "early_total_clicks_153",
    "early_active_days_153", "early_unique_resources_153", "early_quiz_clicks_153", "early_forum_clicks_153",
]
ATTACKABLE_FEATURES = [
    "early_avg_score_65", "early_n_submitted_65", "early_first_submit_day_65", "early_total_clicks_65",
    "early_active_days_65", "early_unique_resources_65", "early_quiz_clicks_65", "early_forum_clicks_65",
    "early_avg_score_153", "early_n_submitted_153", "early_first_submit_day_153", "early_total_clicks_153",
    "early_active_days_153", "early_unique_resources_153", "early_quiz_clicks_153", "early_forum_clicks_153",
    "GPA_1styear", "GPA_2ndyear", "GPA_3rdyear", 
    "credit_enrolled_4thyear", "credit_enrolled_1styear", "credit_passed_1styear", "credit_enrolled_2ndyear",
    "credit_passed_2ndyear", "credit_enrolled_3rdyear", "credit_passed_3rdyear",
    
]
FROZEN_FEATURES_HINT = ["code_module", "code_presentation", "module_credits"]
DERIVED_FEATURE_COMPONENTS = {}  # CGPA and the credit totals were removed from the feature set
# Per-year (passed, enrolled) pairs for the passed <= enrolled cap (year 4 has no
# passed column -> not capped), matching dnd_train_rf/xgb/mlp on oulad_early.
CREDIT_YEAR_PAIRS = [
    ("credit_passed_1styear", "credit_enrolled_1styear"),
    ("credit_passed_2ndyear", "credit_enrolled_2ndyear"),
    ("credit_passed_3rdyear", "credit_enrolled_3rdyear"),
]
ROUND_TO_INT_FEATURES = [
    "early_n_submitted_65", "early_first_submit_day_65", "early_total_clicks_65", "early_active_days_65",
    "early_unique_resources_65", "early_quiz_clicks_65", "early_forum_clicks_65",
    "early_n_submitted_153", "early_first_submit_day_153", "early_total_clicks_153", "early_active_days_153",
    "early_unique_resources_153", "early_quiz_clicks_153", "early_forum_clicks_153",
    "credit_enrolled_4thyear", "credit_enrolled_1styear", "credit_passed_1styear", "credit_enrolled_2ndyear",
    "credit_passed_2ndyear", "credit_enrolled_3rdyear", "credit_passed_3rdyear",
    
]
DECIMAL_2DP_FEATURES = ["early_avg_score_65", "early_avg_score_153", "GPA_1styear", "GPA_2ndyear", "GPA_3rdyear"]
PERCENTAGE_FEATURES = ["early_avg_score_65", "early_avg_score_153"]
NONNEGATIVE_FEATURES = sorted(set([
    "early_avg_score_65", "early_n_submitted_65", "early_total_clicks_65", "early_active_days_65",
    "early_unique_resources_65", "early_quiz_clicks_65", "early_forum_clicks_65",
    "early_avg_score_153", "early_n_submitted_153", "early_total_clicks_153", "early_active_days_153",
    "early_unique_resources_153", "early_quiz_clicks_153", "early_forum_clicks_153",
    "GPA_1styear", "GPA_2ndyear", "GPA_3rdyear", 
    "credit_enrolled_4thyear", "credit_enrolled_1styear", "credit_passed_1styear", "credit_enrolled_2ndyear",
    "credit_passed_2ndyear", "credit_enrolled_3rdyear", "credit_passed_3rdyear",
    
]))
# ============================================================
# SOURCE MLP MODEL
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
# GENERAL HELPERS
# ============================================================
def set_global_seed(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
def one_hot(labels, n_classes=2):
    labels = np.asarray(labels).astype(int)
    return np.eye(n_classes, dtype=np.float32)[labels]
def normalize_target(series: pd.Series) -> pd.Series:
    target_normalized = (
        series.astype(str).str.strip().str.lower().replace({
            "non dropout": "non-dropout", "non_dropout": "non-dropout", "drop out": "dropout",
        })
    )
    target_map = {"non-dropout": 0, "dropout": 1, "0": 0, "1": 1}
    y = target_normalized.map(target_map)
    if y.isna().any():
        raise ValueError(f"some target values could not be mapped: {series.loc[y.isna()].unique().tolist()}")
    return y.astype(int)
def prepare_features_and_label(df, selected_features):
    if TARGET_COL not in df.columns:
        raise ValueError(f"target column '{TARGET_COL}' not found.")
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"selected features missing from the CSV: {missing}")
    X = df[selected_features].copy()
    y = normalize_target(df[TARGET_COL])
    student_ids = df["id_student"].copy() if "id_student" in df.columns else pd.Series(np.arange(len(df)), index=df.index)
    return X, y, student_ids
def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index)
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids, test_size=0.30, stratify=y, random_state=random_state)
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp, y_temp, id_temp, test_size=1.0 / 3.0, stratify=y_temp, random_state=random_state)
    data = {}
    for split, (Xs, ys) in {"train": (X_train, y_train), "test": (X_test, y_test), "validation": (X_val, y_val)}.items():
        data[split] = {"X_raw": Xs.reset_index(drop=True).copy(), "y": ys.reset_index(drop=True).values.astype(int)}
    return data
# ============================================================
# LOADING
# ============================================================
def find_existing_path(patterns):
    for path in patterns:
        if os.path.exists(path):
            return path
    return None
def resolve_first_existing(candidates, what):
    path = find_existing_path(candidates)
    if path is None:
        raise FileNotFoundError(f"{what} not found. checked:\n" + "\n".join(candidates))
    return path
def resolve_mlp_model_path(mlp_dir):
    return resolve_first_existing([os.path.join(mlp_dir, "oulad_mlp_model.pt")], "MLP model (oulad_mlp_model.pt)")
def resolve_mlp_preprocessor_path(mlp_dir):
    return resolve_first_existing([os.path.join(mlp_dir, "oulad_mlp_preprocessor.pkl")], "MLP preprocessor (oulad_mlp_preprocessor.pkl)")
def resolve_pkl_model_path(results_dir, filename):
    return resolve_first_existing([os.path.join(results_dir, filename)], f"model ({filename})")
def load_summary_threshold(summary_path, fallback=0.5):
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        for candidate in [summary.get("selection", {}).get("best_threshold"), summary.get("best_threshold")]:
            if candidate is not None:
                return float(candidate), summary_path
    return float(fallback), "fallback_0.5"
def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
def read_mlp_checkpoint(path, device):
    obj = load_torch_checkpoint(path, device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return {
            "hidden_dims": tuple(obj.get("hidden_dims", DEFAULT_HIDDEN_DIMS)),
            "dropout": float(obj.get("dropout", 0.30)),
            "state_dict": obj["model_state_dict"],
            "selected_features": obj.get("selected_features", None),
            "input_dim": obj.get("input_dim", None),
            "best_threshold": obj.get("best_threshold", None),
        }
    return {"hidden_dims": DEFAULT_HIDDEN_DIMS, "dropout": 0.30, "state_dict": obj,
            "selected_features": None, "input_dim": None, "best_threshold": None}
def build_mlp_from_state(info, input_dim, device):
    model = DropoutMLP(input_dim=input_dim, hidden_dims=info["hidden_dims"], dropout=info["dropout"]).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    return model
def wrap_mlp_for_art(model, input_dim):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), optimizer=optimizer,
                             input_shape=(input_dim,), nb_classes=2)
# ============================================================
# PREPROCESSOR HELPERS AND MASKS
# ============================================================
def transform_to_float32(preprocessor, X):
    arr = preprocessor.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)
def get_transformer_columns(preprocessor, transformer_name):
    for name, transformer, cols in preprocessor.transformers_:
        if name == transformer_name:
            return list(cols)
    return []
def get_numeric_cols_from_preprocessor(preprocessor):
    cols = get_transformer_columns(preprocessor, "num")
    if not cols:
        raise ValueError("transformer 'num' not found in preprocessor.")
    return cols
def get_scaler_from_preprocessor(preprocessor):
    num_pipeline = preprocessor.named_transformers_["num"]
    if "scaler" not in num_pipeline.named_steps:
        raise ValueError("scaler not found in numeric pipeline.")
    return num_pipeline.named_steps["scaler"]
def get_transformer_slice(preprocessor, transformer_name, input_dim):
    if hasattr(preprocessor, "output_indices_") and transformer_name in preprocessor.output_indices_:
        return preprocessor.output_indices_[transformer_name]
    if transformer_name == "num":
        n_cols = len(get_transformer_columns(preprocessor, "num"))
        return slice(0, min(n_cols, input_dim))
    return slice(0, 0)
def make_masks(preprocessor, input_dim, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(input_dim)]
    perturbable_mask = np.zeros(input_dim, dtype=bool)
    numerical_mask = np.zeros(input_dim, dtype=bool)
    categorical_mask = np.zeros(input_dim, dtype=bool)
    numeric_cols = get_transformer_columns(preprocessor, "num")
    num_slice = get_transformer_slice(preprocessor, "num", input_dim)
    num_indices = list(range(num_slice.start or 0, num_slice.stop or 0))
    attackable = set(ATTACKABLE_FEATURES)
    for local_idx, raw_col in enumerate(numeric_cols):
        if local_idx >= len(num_indices):
            continue
        idx = num_indices[local_idx]
        numerical_mask[idx] = True
        if raw_col in attackable:
            perturbable_mask[idx] = True
    for name, _, _ in preprocessor.transformers_:
        if name in ["num", "remainder"]:
            continue
        if hasattr(preprocessor, "output_indices_") and name in preprocessor.output_indices_:
            categorical_mask[preprocessor.output_indices_[name]] = True
    if categorical_mask.sum() == 0:
        categorical_mask[~numerical_mask] = True
    if int(perturbable_mask.sum()) == 0:
        raise ValueError("no preprocessed feature is attackable.")
    mask_df = pd.DataFrame({
        "preprocessed_index": np.arange(input_dim), "feature_name": feature_names,
        "is_attackable_feature": perturbable_mask, "is_numerical_feature": numerical_mask,
        "is_categorical_or_other_feature": categorical_mask,
    })
    mask_df.to_csv(os.path.join(output_dir, "oulad_preprocessed_feature_masks.csv"), index=False)
    return perturbable_mask, numerical_mask, categorical_mask, feature_names, mask_df
# ============================================================
# ATTACK MACHINERY (verbatim from the oulad_early attack/defence scripts)
# ============================================================
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
def preprocessed_adv_to_raw_and_scaled(X_adv_preprocessed, X_clean_preprocessed, X_clean_raw, selected_features,
                                       preprocessor, train_min, train_max, perturbable_mask_preprocessed,
                                       attacked_row_mask=None, enforce_consistency=True):
    # inverse-scale the numeric block, restore frozen features, apply oulad domain
    # constraints (clip + round), THEN re-impose the derived identities LAST so they
    # hold on the finalized values (matches dnd_train_* on oulad_early).
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    scaler = get_scaler_from_preprocessor(preprocessor)
    n_num = len(numeric_cols)
    adv_numeric_raw = scaler.inverse_transform(X_adv_preprocessed[:, :n_num])
    X_adv_raw = X_clean_raw[selected_features].reset_index(drop=True).copy()
    clean_reset = X_clean_raw[selected_features].reset_index(drop=True)
    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    # clip to train min/max
    for col in numeric_cols:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=train_min[col], upper=train_max[col])
    # non-negative floor + percentage bound
    for col in NONNEGATIVE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0)
    for col in PERCENTAGE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0, upper=100)
    # rounding grids
    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col]).astype(float)
    for col in DECIMAL_2DP_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col].astype(float), 2)
    # enforce derived identities LAST, on the clipped+rounded components
    if enforce_consistency:
        X_adv_raw = enforce_derived_consistency(X_adv_raw)
        X_adv_raw = enforce_activity_consistency(X_adv_raw, clean_reset, train_min, train_max)
        assert_feasible(oulad_feasibility_violations(X_adv_raw), attacked_row_mask, 'OULAD projection')
    # restore frozen features again after clip/round/enforce
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    if attacked_row_mask is not None:
        not_attacked = ~np.asarray(attacked_row_mask).astype(bool)
        if not_attacked.any():
            for col in selected_features:
                values = clean_reset.loc[not_attacked, col]
                if pd.api.types.is_numeric_dtype(X_adv_raw[col]):
                    values = values.astype(X_adv_raw[col].dtype, copy=False)
                X_adv_raw.loc[not_attacked, col] = values.values
    X_projected_preprocessed = transform_to_float32(preprocessor, X_adv_raw[selected_features])
    frozen = np.asarray(perturbable_mask_preprocessed) == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]
    if attacked_row_mask is not None:
        not_attacked = ~np.asarray(attacked_row_mask).astype(bool)
        if not_attacked.any():
            X_projected_preprocessed[not_attacked, :] = X_clean_preprocessed[not_attacked, :]
    return X_adv_raw[selected_features], X_projected_preprocessed
def make_attack_labels(attack_mode, y_true):
    if attack_mode == "untargeted":
        return one_hot(y_true), False
    if attack_mode == "target_non_dropout":
        return one_hot(np.zeros(len(y_true), dtype=int)), True
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
def true_positive_attack_mask(y_true, clean_pred):
    return (np.asarray(y_true) == POSITIVE_LABEL) & (np.asarray(clean_pred) == POSITIVE_LABEL)
def generate_adv_for_scope(attack, X_clean_preprocessed, y_true, clean_pred, attack_mode, attack_scope, perturbable_float):
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()
    if int(attack_idx.sum()) == 0:
        return X_adv_preprocessed, attack_idx
    attack_y, _ = make_attack_labels(attack_mode, np.asarray(y_true)[attack_idx])
    X_adv_subset = attack.generate(x=X_clean_preprocessed[attack_idx], y=attack_y, mask=perturbable_float)
    frozen = perturbable_float == 0
    X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
    X_adv_preprocessed[attack_idx] = X_adv_subset
    return X_adv_preprocessed, attack_idx
def predict_mlp_proba(model, X_preprocessed, device, batch_size=4096):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_preprocessed), batch_size):
            batch = torch.tensor(X_preprocessed[start:start + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)
# ============================================================
# DETECTOR (raw input-global Mahalanobis; identical for RF, XGB and MLP)
# ============================================================
class RawMahalanobisValidator:
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
    def scores(self, X):
        M = X[self.cols].to_numpy(dtype=float)
        d = M - self.mu
        left = d @ self.cov_inv
        sq = np.einsum("ij,ij->i", left, d)
        return np.sqrt(np.clip(sq, 0, None))
# ============================================================
# METRIC ENGINE
# ============================================================
def safe_subset(X, row_mask):
    if row_mask is None:
        return X
    row_mask = np.asarray(row_mask).astype(bool)
    return X[:0] if row_mask.sum() == 0 else X[row_mask]
def compute_proximity_metrics(X_clean, X_adv, perturbable_mask, row_mask=None):
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return {"row_count_used_for_distance": 0}
    diff = Xa - Xc
    abs_diff = np.abs(diff)
    pm = perturbable_mask.astype(bool)
    frozen = ~pm
    diff_att = diff[:, pm]
    abs_frozen = np.abs(diff[:, frozen]) if frozen.any() else np.zeros((len(Xc), 0), dtype=np.float32)
    return {
        "row_count_used_for_distance": int(len(Xc)),
        "mean_l1_all": float(np.mean(np.linalg.norm(diff, ord=1, axis=1))),
        "mean_l2_all": float(np.mean(np.linalg.norm(diff, ord=2, axis=1))),
        "mean_linf_all": float(np.mean(np.linalg.norm(diff, ord=np.inf, axis=1))),
        "mean_l1_attackable": float(np.mean(np.linalg.norm(diff_att, ord=1, axis=1))),
        "median_l1_attackable": float(np.median(np.linalg.norm(diff_att, ord=1, axis=1))),
        "mean_l2_attackable": float(np.mean(np.linalg.norm(diff_att, ord=2, axis=1))),
        "median_l2_attackable": float(np.median(np.linalg.norm(diff_att, ord=2, axis=1))),
        "mean_linf_attackable": float(np.mean(np.linalg.norm(diff_att, ord=np.inf, axis=1))),
        "median_linf_attackable": float(np.median(np.linalg.norm(diff_att, ord=np.inf, axis=1))),
        "mean_abs_change_attackable": float(np.mean(np.abs(diff_att))),
        "max_abs_change_attackable": float(np.max(np.abs(diff_att))),
        "max_abs_change_frozen": float(np.max(abs_frozen)) if abs_frozen.size else 0.0,
        "frozen_changed_samples": int(np.sum(np.any(abs_frozen > EPS, axis=1))) if abs_frozen.size else 0,
    }
def compute_sensitivity_metrics(X_clean, X_adv, numerical_mask, categorical_mask, row_mask=None):
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return {"row_count_used_for_sensitivity": 0}
    abs_diff = np.abs(Xa - Xc)
    num_mask = numerical_mask.astype(bool)
    cat_mask = categorical_mask.astype(bool)
    per_sample = abs_diff[:, num_mask].sum(axis=1) if num_mask.sum() > 0 else np.zeros(len(Xc), dtype=np.float32)
    cat_rate = (abs_diff[:, cat_mask] > EPS).astype(np.float32).mean(axis=1) if cat_mask.sum() > 0 else np.zeros(len(Xc), dtype=np.float32)
    return {
        "row_count_used_for_sensitivity": int(len(Xc)),
        "mean_sensitivity": float(np.mean(per_sample)),
        "median_sensitivity": float(np.median(per_sample)),
        "max_sensitivity": float(np.max(per_sample)),
        "mean_categorical_change_rate": float(np.mean(cat_rate)),
    }
def feature_sensitivity_summary(X_clean, X_adv, feature_names, perturbable_mask, numerical_mask, categorical_mask,
                                model, attack, split, epsilon, row_scope, row_mask=None):
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return pd.DataFrame()
    diff = Xa - Xc
    abs_diff = np.abs(diff)
    num_mask = numerical_mask.astype(bool)
    cat_mask = categorical_mask.astype(bool)
    sens_matrix = np.zeros_like(abs_diff, dtype=np.float32)
    if num_mask.sum() > 0:
        sens_matrix[:, num_mask] = abs_diff[:, num_mask]
    df = pd.DataFrame({
        "model": model, "attack": attack, "split": split, "epsilon": epsilon, "row_scope": row_scope,
        "feature_name": feature_names,
        "is_attackable_feature": perturbable_mask.astype(bool),
        "is_numerical_feature": num_mask, "is_categorical_or_other_feature": cat_mask,
        "mean_abs_change": abs_diff.mean(axis=0), "max_abs_change": abs_diff.max(axis=0),
        "mean_signed_change": diff.mean(axis=0),
        "changed_sample_count": (abs_diff > EPS).sum(axis=0),
        "changed_sample_rate": (abs_diff > EPS).mean(axis=0),
        "mean_sensitivity": sens_matrix.mean(axis=0), "max_sensitivity": sens_matrix.max(axis=0),
    })
    df["sensitivity_rank"] = df["mean_sensitivity"].rank(ascending=False, method="dense")
    return df.sort_values("mean_sensitivity", ascending=False)
# ============================================================
# CASE PROCESSOR (per model, per split)
# ============================================================
def process_case(case, split_name, split_data, preprocessor, art_mlp_for_attack, mlp_target_model, device,
                 validator, maha_threshold, perturbable_mask, numerical_mask, categorical_mask, feature_names,
                 train_min, train_max, selected_features, epsilon, attack_mode, attack_scope,
                 pgd_step, pgd_iterations, enforce_consistency):
    model_key = case["model_key"]
    model_name = case["model_name"]
    attack_name = case["attack_name"]
    threshold = case["threshold"]
    target_model = case["target_model"]  # pipeline for RF/XGB, None for MLP
    X_clean_raw = split_data[split_name]["X_raw"]
    y_true = split_data[split_name]["y"]
    X_clean_pre = transform_to_float32(preprocessor, X_clean_raw)
    perturbable_float = perturbable_mask.astype(np.float32)
    # clean predictions on the (baseline) target -> scope + baseline hiding reference
    if model_key == "MLP":
        clean_pred = (predict_mlp_proba(mlp_target_model, X_clean_pre, device) >= threshold).astype(int)
    else:
        clean_pred = (target_model.predict_proba(X_clean_raw)[:, 1] >= threshold).astype(int)
    # build the attack object
    _, targeted = make_attack_labels(attack_mode, y_true[:1])
    if model_key == "MLP":
        attack = ProjectedGradientDescent(estimator=art_mlp_for_attack, eps=float(epsilon),
                                          eps_step=min(float(pgd_step), float(epsilon)),
                                          max_iter=int(pgd_iterations), targeted=targeted, batch_size=128)
    else:
        attack = FastGradientMethod(estimator=art_mlp_for_attack, eps=float(epsilon), targeted=targeted, batch_size=128)
    X_adv_pre_raw, attack_idx = generate_adv_for_scope(attack, X_clean_pre, y_true, clean_pred,
                                                       attack_mode, attack_scope, perturbable_float)
    X_adv_raw, X_adv_pre = preprocessed_adv_to_raw_and_scaled(
        X_adv_preprocessed=X_adv_pre_raw, X_clean_preprocessed=X_clean_pre, X_clean_raw=X_clean_raw,
        selected_features=selected_features, preprocessor=preprocessor, train_min=train_min, train_max=train_max,
        perturbable_mask_preprocessed=perturbable_float, attacked_row_mask=attack_idx, enforce_consistency=enforce_consistency)
    # adversarial predictions on the (baseline) target, no defence yet
    if model_key == "MLP":
        adv_pred_raw = (predict_mlp_proba(mlp_target_model, X_adv_pre, device) >= threshold).astype(int)
    else:
        adv_pred_raw = (target_model.predict_proba(X_adv_raw)[:, 1] >= threshold).astype(int)
    # detector flags on the ADVERSARIAL input (raw), then default_dropout override
    flags = validator.scores(X_adv_raw) > maha_threshold
    adv_pred_default = adv_pred_raw.copy()
    adv_pred_default[flags] = POSITIVE_LABEL
    # --- (A) detection / override effect ---
    attacked = true_positive_attack_mask(y_true, clean_pred)
    n_attacked = int(attacked.sum())
    attacked_flagged = attacked & flags
    attacked_accepted = attacked & (~flags)
    hidden_no_def = (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL) & (adv_pred_raw == NEGATIVE_LABEL)
    hidden_after = (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL) & (adv_pred_default == NEGATIVE_LABEL)
    detection = {
        "n_flagged": int(flags.sum()),
        "flag_rate": float(np.mean(flags)),
        "n_attacked": n_attacked,
        "n_attacked_flagged": int(attacked_flagged.sum()),
        "n_attacked_accepted": int(attacked_accepted.sum()),
        "detection_recall_on_attacked": float(attacked_flagged.sum() / n_attacked) if n_attacked else 0.0,
        "dropout_hiding_rate_no_defence": float(hidden_no_def.sum() / n_attacked) if n_attacked else 0.0,
        "dropout_hiding_rate_after_default_dropout": float(hidden_after.sum() / n_attacked) if n_attacked else 0.0,
        "hides_recovered_by_defence": int(hidden_no_def.sum() - hidden_after.sum()),
    }
    # --- (B) proximity/sensitivity by detector outcome ---
    scopes = [
        ("all_rows", None),
        ("attacked_rows_only", attacked),
        ("attacked_flagged", attacked_flagged),
        ("attacked_accepted", attacked_accepted),
    ]
    rows, feature_dfs = [], []
    for row_scope, row_mask in scopes:
        row = {"model": model_name, "attack": attack_name, "condition": "After Defence (Detect-and-Default)",
               "split": split_name, "epsilon": epsilon, "row_scope": row_scope,
               "detector_threshold": float(maha_threshold)}
        row.update(detection)
        row.update(compute_proximity_metrics(X_clean_pre, X_adv_pre, perturbable_mask, row_mask=row_mask))
        row.update(compute_sensitivity_metrics(X_clean_pre, X_adv_pre, numerical_mask, categorical_mask, row_mask=row_mask))
        rows.append(row)
        feature_dfs.append(feature_sensitivity_summary(
            X_clean_pre, X_adv_pre, feature_names, perturbable_mask, numerical_mask, categorical_mask,
            model=model_name, attack=attack_name, split=split_name, epsilon=epsilon, row_scope=row_scope, row_mask=row_mask))
    print(f"OK {model_name} {split_name}: attacked={n_attacked}, flagged_among_attacked={int(attacked_flagged.sum())} "
          f"(recall={detection['detection_recall_on_attacked']:.3f}), "
          f"hide {detection['dropout_hiding_rate_no_defence']:.3f} -> {detection['dropout_hiding_rate_after_default_dropout']:.3f}")
    return rows, feature_dfs
# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="After-defence (detect-and-default) proximity/sensitivity for oulad_early. "
                                                  "Option C: detection/override effect + flagged/accepted proximity split.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--mlp-dir", default=DEFAULT_MLP_DIR, help="Feature list, preprocessor, baseline MLP.")
    parser.add_argument("--eval-mlp-model", default=DEFAULT_EVAL_MLP_MODEL,
                        help="FGSM gradient source for RF/XGB DnD (baseline MLP by default). Pass \"\" to use the baseline MLP.")
    parser.add_argument("--rf-dir", default=DEFAULT_RF_DIR)
    parser.add_argument("--xgb-dir", default=DEFAULT_XGB_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--splits", default="test,validation")
    parser.add_argument("--models", default="rf,xgb,mlp")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--pgd-step", type=float, default=DEFAULT_PGD_STEP)
    parser.add_argument("--pgd-iterations", type=int, default=DEFAULT_PGD_ITERATIONS)
    parser.add_argument("--attack-mode", default=DEFAULT_ATTACK_MODE, choices=["untargeted", "target_non_dropout"])
    parser.add_argument("--attack-scope", default=DEFAULT_ATTACK_SCOPE, choices=["all", "dropout_only", "true_positive_only"])
    parser.add_argument("--detection-target-fpr", type=float, default=DEFAULT_DETECTION_TARGET_FPR)
    parser.add_argument("--detector-covariance", default=DEFAULT_DETECTOR_COVARIANCE, choices=["tied", "per_class"])
    parser.add_argument("--no-enforce-derived-consistency", dest="enforce_consistency", action="store_false")
    parser.set_defaults(enforce_consistency=True)
    args = parser.parse_args()
    set_global_seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_names = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    print("=" * 80)
    print("OULAD_EARLY AFTER-DEFENCE (DETECT-AND-DEFAULT) PROXIMITY AND SENSITIVITY")
    print("=" * 80)
    print(f"CSV: {args.csv_path} | models: {wanted} | splits: {split_names} | device: {device}")
    print(f"Detector target FPR: {args.detection_target_fpr} | detector: raw input-global Mahalanobis (all models)")
    print(f"PGD: step={args.pgd_step}, iters={args.pgd_iterations} (dnd_train_mlp default = 100)")
    print()
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"dataset not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    # baseline MLP: feature list + preprocessor (+ target for the MLP case + white-box gradient)
    mlp_model_path = resolve_mlp_model_path(args.mlp_dir)
    mlp_preprocessor_path = resolve_mlp_preprocessor_path(args.mlp_dir)
    mlp_summary_path = os.path.join(args.mlp_dir, "oulad_mlp_summary.json")
    mlp_info = read_mlp_checkpoint(mlp_model_path, device)
    selected_features = list(mlp_info["selected_features"]) if mlp_info["selected_features"] is not None else DEFAULT_SELECTED_FEATURES.copy()
    missing = [f for f in ATTACKABLE_FEATURES if f not in selected_features]
    if missing:
        raise ValueError(f"attackable features not in model feature list: {missing}")
    if [f for f in FROZEN_FEATURES_HINT if f in ATTACKABLE_FEATURES]:
        raise ValueError("a frozen-hint feature is marked attackable.")
    X, y, student_ids = prepare_features_and_label(df, selected_features)
    split_data = split_70_20_10_stratified(X, y, student_ids, random_state=args.random_state)
    preprocessor = joblib.load(mlp_preprocessor_path)
    X_test_pre = transform_to_float32(preprocessor, split_data["test"]["X_raw"])
    input_dim = X_test_pre.shape[1]
    if mlp_info["input_dim"] is not None and int(mlp_info["input_dim"]) != int(input_dim):
        raise ValueError("preprocessed input_dim != MLP checkpoint input_dim.")
    perturbable_mask, numerical_mask, categorical_mask, feature_names, mask_df = make_masks(preprocessor, input_dim, args.output_dir)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    baseline_mlp = build_mlp_from_state(mlp_info, input_dim=input_dim, device=device)
    baseline_art = wrap_mlp_for_art(baseline_mlp, input_dim=input_dim)
    # MLP operating threshold: checkpoint best_threshold -> summary -> 0.5, matching
    # dnd_train_mlp.py's load_mlp_checkpoint precedence (checkpoint is the primary
    # source because the baseline saves best_threshold in the .pt).
    if mlp_info["best_threshold"] is not None:
        mlp_threshold, mlp_threshold_source = float(mlp_info["best_threshold"]), "oulad_mlp_model.pt"
    else:
        mlp_threshold, mlp_threshold_source = load_summary_threshold(
            mlp_summary_path if os.path.exists(mlp_summary_path) else None)
    # optional FGSM gradient source for RF/XGB DnD (default: baseline MLP)
    if args.eval_mlp_model:
        if not os.path.exists(args.eval_mlp_model):
            raise FileNotFoundError(f"defended MLP not found: {args.eval_mlp_model} (run adv_train_mlp first, or pass --eval-mlp-model \"\").")
        eval_info = read_mlp_checkpoint(args.eval_mlp_model, device)
        if eval_info["selected_features"] is not None and list(eval_info["selected_features"]) != selected_features:
            raise ValueError("defended MLP feature list differs from the baseline MLP.")
        defended_mlp = build_mlp_from_state(eval_info, input_dim=input_dim, device=device)
        defended_art = wrap_mlp_for_art(defended_mlp, input_dim=input_dim)
        fgsm_grad_source = os.path.basename(args.eval_mlp_model)
    else:
        defended_art = baseline_art
        fgsm_grad_source = "baseline_mlp"
    train_min = split_data["train"]["X_raw"][selected_features].min(numeric_only=True)
    train_max = split_data["train"]["X_raw"][selected_features].max(numeric_only=True)
    print(f"Preprocessed dim: {input_dim} | attackable dims: {int(perturbable_mask.sum())} "
          f"| numeric dims: {int(numerical_mask.sum())} | categorical/other dims: {int(categorical_mask.sum())}")
    print(f"FGSM gradient (RF/XGB): {fgsm_grad_source} | MLP PGD: white-box baseline")
    print(f"Thresholds -> MLP: {mlp_threshold} (source: {mlp_threshold_source})")
    print()
    # fit + calibrate the SINGLE shared detector (raw input-global, used by RF/XGB and MLP alike).
    # Calibrate on clean test -> target FPR. The MLP now uses this identical detector so the
    # cross-model detect-and-default comparison is controlled (only the classifier differs).
    raw_validator = RawMahalanobisValidator(ATTACKABLE_FEATURES).fit(split_data["train"]["X_raw"], numeric_cols)
    raw_maha_thr = float(np.quantile(raw_validator.scores(split_data["test"]["X_raw"][selected_features]), 1.0 - args.detection_target_fpr))
    print(f"Detector covariance: empirical covariance of raw attackable features + ridge 1e-6, pseudo-inverse "
          f"(condition number = {raw_validator.condition_number:.3e})")
    print(f"Raw-space input-global Mahalanobis threshold (all models): {raw_maha_thr}")
    print()
    cases = []
    rf_threshold_source = xgb_threshold_source = None
    if "rf" in wanted:
        rf_model = joblib.load(resolve_pkl_model_path(args.rf_dir, "oulad_rf_model.pkl"))
        rf_thr, rf_threshold_source = load_summary_threshold(os.path.join(args.rf_dir, "oulad_rf_summary.json"))
        if rf_threshold_source == "fallback_0.5":
            print("WARNING: RF threshold not found in oulad_rf_summary.json; using fallback 0.5. "
                  "Hiding-rate / detection numbers will NOT match the reported baseline. Check --rf-dir.")
        cases.append({"model_key": "RF", "model_name": "Random Forest (DnD)", "attack_name": "FGSM via baseline MLP",
                      "threshold": rf_thr, "target_model": rf_model, "art": defended_art,
                      "validator": raw_validator, "maha_thr": raw_maha_thr})
    if "xgb" in wanted:
        xgb_model = joblib.load(resolve_pkl_model_path(args.xgb_dir, "oulad_xgb_model.pkl"))
        xgb_thr, xgb_threshold_source = load_summary_threshold(os.path.join(args.xgb_dir, "oulad_xgb_summary.json"))
        if xgb_threshold_source == "fallback_0.5":
            print("WARNING: XGB threshold not found in oulad_xgb_summary.json; using fallback 0.5. "
                  "Hiding-rate / detection numbers will NOT match the reported baseline. Check --xgb-dir.")
        cases.append({"model_key": "XGB", "model_name": "XGBoost (DnD)", "attack_name": "FGSM via baseline MLP",
                      "threshold": xgb_thr, "target_model": xgb_model, "art": defended_art,
                      "validator": raw_validator, "maha_thr": raw_maha_thr})
    if "mlp" in wanted:
        cases.append({"model_key": "MLP", "model_name": "MLP (DnD)", "attack_name": "PGD white-box",
                      "threshold": mlp_threshold, "target_model": None, "art": baseline_art,
                      "validator": raw_validator, "maha_thr": raw_maha_thr})
    if not cases:
        raise ValueError("no valid models selected.")
    all_rows, all_feature_dfs = [], []
    for case in cases:
        for split_name in split_names:
            rows, feature_dfs = process_case(
                case=case, split_name=split_name, split_data=split_data, preprocessor=preprocessor,
                art_mlp_for_attack=case["art"], mlp_target_model=baseline_mlp, device=device,
                validator=case["validator"], maha_threshold=case["maha_thr"],
                perturbable_mask=perturbable_mask, numerical_mask=numerical_mask, categorical_mask=categorical_mask,
                feature_names=feature_names, train_min=train_min, train_max=train_max,
                selected_features=selected_features, epsilon=args.epsilon, attack_mode=args.attack_mode,
                attack_scope=args.attack_scope, pgd_step=args.pgd_step, pgd_iterations=args.pgd_iterations,
                enforce_consistency=args.enforce_consistency)
            all_rows.extend(rows)
            all_feature_dfs.extend(feature_dfs)
    summary_df = pd.DataFrame(all_rows)
    feature_df = pd.concat([f for f in all_feature_dfs if not f.empty], ignore_index=True) if all_feature_dfs else pd.DataFrame()
    summary_path = os.path.join(args.output_dir, "oulad_proximity_sensitivity_dnd_summary.csv")
    feature_path = os.path.join(args.output_dir, "oulad_feature_sensitivity_dnd_summary.csv")
    top15_path = os.path.join(args.output_dir, "oulad_top15_sensitive_features_dnd.csv")
    json_path = os.path.join(args.output_dir, "oulad_proximity_sensitivity_dnd_summary.json")
    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    if not feature_df.empty:
        top15_df = (feature_df[feature_df["is_attackable_feature"] == True]
                    .sort_values(["model", "split", "row_scope", "mean_sensitivity"], ascending=[True, True, True, False])
                    .groupby(["model", "split", "row_scope"]).head(15).reset_index(drop=True))
    else:
        top15_df = pd.DataFrame()
    top15_df.to_csv(top15_path, index=False)
    excel_path = os.path.join(args.output_dir, "oulad_proximity_sensitivity_dnd_analysis.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            if not feature_df.empty:
                feature_df.to_excel(writer, sheet_name="Feature Sensitivity", index=False)
            if not top15_df.empty:
                top15_df.to_excel(writer, sheet_name="Top 15 Sensitive", index=False)
            mask_df.to_excel(writer, sheet_name="Preprocessed Masks", index=False)
    except Exception as e:
        excel_path = f"skipped ({e})"
    json_summary = {
        "credit_record_rule": CREDIT_RULE_TEXT,
        "dataset": args.csv_path, "feature_set": "oulad_early (29 features)", "defence": "Detect-and-default",
        "analysis": "Option C: (A) detection/override effect + (B) proximity/sensitivity split by detector outcome.",
        "key_caveat": "DnD does not change the adversarial input. Proximity/sensitivity of the input equal the "
                      "attack against the baseline model; the defence acts by flagging and overriding predictions. "
                      "row_scope=attacked_accepted are the adversarial rows that evaded the detector.",
        "detectors": {"RF": "raw-space input-global Mahalanobis over attackable features, empirical covariance (ridge 1e-6, pseudo-inverse)",
                      "XGB": "raw-space input-global Mahalanobis over attackable features, empirical covariance (ridge 1e-6, pseudo-inverse)",
                      "MLP": "raw-space input-global Mahalanobis over attackable features, empirical covariance (ridge 1e-6, pseudo-inverse) (identical to RF/XGB)"},
        "detector_covariance_estimator": "np.cov on the raw attackable features + ridge 1e-6 on the diagonal, inverted with np.linalg.pinv (empirical covariance, no standardisation, no shrinkage)",
        "detector_ridge": raw_validator.ridge,
        "detector_condition_number": raw_validator.condition_number,
        "fgsm_gradient_source_rf_xgb": fgsm_grad_source, "mlp_attack": "white-box PGD on the baseline MLP",
        "epsilon": args.epsilon, "pgd_step": args.pgd_step, "pgd_iterations": args.pgd_iterations,
        "pgd_iterations_note": "dnd_train_mlp.py default is 100 "
                               "and matches the standalone pgd_mlp.py attack.",
        "attack_mode": args.attack_mode, "attack_scope": args.attack_scope,
        "detection_target_fpr": args.detection_target_fpr,
        "detector_thresholds": {"raw_all_models": raw_maha_thr},
        "operating_thresholds": {"mlp": mlp_threshold, "mlp_source": mlp_threshold_source,
                                 "rf_source": rf_threshold_source, "xgb_source": xgb_threshold_source},
        "derived_identities": "per-year credit_passed <= credit_enrolled; year-4 enrolment = 360 - sum of "
                              "years 1-3 (CGPA and credit totals not used). Enforced LAST, after clip/round, "
                              "matching dnd_train_rf/xgb/mlp on oulad_early.",
        "override_rule": "flagged rows -> dropout (default_dropout); hiding rate reported before vs after.",
        "files_created": {"summary_csv": summary_path, "feature_sensitivity_csv": feature_path,
                          "top15_sensitive_csv": top15_path, "excel": excel_path},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)
    print("\nDetection / override effect (per model/split):")
    eff_cols = [c for c in ["model", "split", "n_attacked", "n_attacked_flagged", "detection_recall_on_attacked",
                            "dropout_hiding_rate_no_defence", "dropout_hiding_rate_after_default_dropout",
                            "hides_recovered_by_defence", "flag_rate"] if c in summary_df.columns]
    print(summary_df[summary_df["row_scope"] == "all_rows"][eff_cols].to_string(index=False))
    print("\nProximity by detector outcome (mean_l2_attackable):")
    prox_cols = [c for c in ["model", "split", "row_scope", "row_count_used_for_distance",
                             "mean_l2_attackable", "mean_linf_attackable", "mean_sensitivity"] if c in summary_df.columns]
    print(summary_df[prox_cols].to_string(index=False))
    print("\nSaved:")
    for p in [summary_path, feature_path, top15_path, excel_path, json_path]:
        print(p)
    print("\nDone.")
if __name__ == "__main__":
    main()