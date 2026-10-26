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
# After-defence proximity and sensitivity for zenodo.csv, DETECT-AND-DEFAULT (DnD)
# defence. This is the zenodo counterpart of the oulad_early DnD proximity script:
# same two-view structure and the same "regenerate the attack, then flag + override"
# flow. Only the dataset-specific pieces differ (feature list, attackable/frozen
# sets, the num/flag/one-hot preprocessing blocks, the rounding grids, and the
# zenodo credit identities instead of oulad's CGPA/credit-sum identities).
#
# Like adversarial training, DnD does NOT retrain the model and does NOT change the
# adversarial input: it keeps the baseline model, runs the same attack, then flags
# inputs with a Mahalanobis validator and OVERRIDES flagged predictions to dropout
# (default_dropout). So the input geometry equals the attack on the baseline model;
# the defence's value shows up in DETECTION and OVERRIDE, not in new distances.
# This script reports both views (Option C):
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
# The three DnD cases reproduce the project's zenodo dnd_train_* scripts exactly:
#   RF  : baseline RF,  FGSM from the BASELINE MLP, raw-space Mahalanobis over attackable features
#   XGB : baseline XGB, FGSM from the BASELINE MLP, raw-space Mahalanobis over attackable features
#   MLP : baseline MLP, white-box PGD (grad=target), raw-space input-global Mahalanobis over
#         attackable features (identical detector to RF/XGB, for a controlled comparison)
# all at eps 1.0, target_non_dropout, true_positive_only, detector calibrated to a
# 5% false-positive rate on the clean test split.

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_MLP_DIR = "mlp_results_zenodo"
DEFAULT_RF_DIR = "rf_results_zenodo"
DEFAULT_XGB_DIR = "xgb_results_zenodo"
DEFAULT_OUTPUT_DIR = "proximity_sensitivity_detect_and_default_zenodo_results"

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
ID_COL = "student_id"
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
EPS = 1e-8
DEFAULT_HIDDEN_DIMS = (128, 64, 32)

# ============================================================
# FEATURE SET (zenodo, 34 features; matches the attack/defence scripts)
# ============================================================
# The authoritative feature list is read from the MLP checkpoint at runtime; this
# is the fallback (every column except student_id and target, in CSV column order).
DEFAULT_SELECTED_FEATURES = [
    "academic_year", "n_courses", "courses_mean", "degree_size", "seniority",
    "highest_course_year_enrolled", "credits_enrolled_semester_a", "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year", "total_credits_passed_academic_year",
    "pass_ratio_sem_a", "pass_ratio_sem_b", "total_performance",
    "obtained_new_degree_flag", "adapted_studies_flag",
    "completion_rate_one_year_before", "completion_rate_two_years_before", "completion_rate_three_years_before",
    "pass_ratio_sem_a_and_zero_flag", "pass_ratio_sem_b_and_zero_flag", "total_performance_and_zero_flag",
    "completion_rate_one_year_before_missing_flag", "completion_rate_two_years_before_missing_flag",
    "completion_rate_three_years_before_missing_flag",
    "lms_events_sem_a", "lms_events_sem_b", "lms_assignment_submissions_sem_a", "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_a", "lms_test_submissions_sem_b", "lms_total_minutes_sem_a", "lms_total_minutes_sem_b",
    "attendance_days_sem_a", "attendance_days_sem_b",
]

# 21 attackable raw features, all numeric.
ATTACKABLE_FEATURES = [
    "courses_mean", "credits_enrolled_semester_a", "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year", "total_credits_passed_academic_year",
    "pass_ratio_sem_a", "pass_ratio_sem_b", "total_performance",
    "completion_rate_one_year_before", "completion_rate_two_years_before", "completion_rate_three_years_before",
    "lms_events_sem_a", "lms_events_sem_b", "lms_assignment_submissions_sem_a", "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_a", "lms_test_submissions_sem_b", "lms_total_minutes_sem_a", "lms_total_minutes_sem_b",
    "attendance_days_sem_a", "attendance_days_sem_b",
]

# Frozen (for the sanity check only): the one-hot categoricals, the frozen numeric
# descriptor n_courses, and all 8 binary flags.
FROZEN_FEATURES_HINT = [
    "academic_year", "n_courses", "degree_size", "seniority", "highest_course_year_enrolled",
    "obtained_new_degree_flag", "adapted_studies_flag",
    "pass_ratio_sem_a_and_zero_flag", "pass_ratio_sem_b_and_zero_flag", "total_performance_and_zero_flag",
    "completion_rate_one_year_before_missing_flag", "completion_rate_two_years_before_missing_flag",
    "completion_rate_three_years_before_missing_flag",
]

# ============================================================
# CREDIT IDENTITIES (zenodo) -- replaces oulad's CGPA/credit-sum identities
# ============================================================
# These three attackable columns are re-tied after the attack:
#   1) total_credits_passed_academic_year <= total_credits_enrolled_academic_year
#   2) total_performance = passed / enrolled * 100 (0 when enrolled == 0)
CREDITS_ENROLLED_COL = "total_credits_enrolled_academic_year"
CREDITS_PASSED_COL = "total_credits_passed_academic_year"
TOTAL_PERFORMANCE_COL = "total_performance"

# ============================================================
# ROUNDING / DOMAIN GRIDS (zenodo)
# ============================================================
# Integer-valued counts (lms_total_minutes_* included, per the zenodo scripts).
ROUND_TO_INT_FEATURES = [
    "lms_events_sem_a", "lms_events_sem_b", "lms_assignment_submissions_sem_a", "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_a", "lms_test_submissions_sem_b", "lms_total_minutes_sem_a", "lms_total_minutes_sem_b",
    "attendance_days_sem_a", "attendance_days_sem_b",
]
# Credit columns snap to the nearest .0 / .5.
ROUND_TO_HALF_FEATURES = [
    "credits_enrolled_semester_a", "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year", "total_credits_passed_academic_year",
]
# Decimal (2dp) features. courses_mean is a ~0-10 grade: decimal but NOT a percentage.
DECIMAL_2DP_FEATURES = [
    "courses_mean", "pass_ratio_sem_a", "pass_ratio_sem_b", "total_performance",
    "completion_rate_one_year_before", "completion_rate_two_years_before", "completion_rate_three_years_before",
]
# Percentage features bounded to [0, 100].
PERCENTAGE_FEATURES = [
    "pass_ratio_sem_a", "pass_ratio_sem_b", "total_performance",
    "completion_rate_one_year_before", "completion_rate_two_years_before", "completion_rate_three_years_before",
]
# Non-negative floor = every attackable feature.
NONNEGATIVE_FEATURES = sorted(set(ATTACKABLE_FEATURES))


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


def round_to_nearest_half(x):
    # snap each value to the nearest .0 or .5 (12.2 -> 12.0, 12.3 -> 12.5).
    return np.round(np.asarray(x, dtype=float) * 2.0) / 2.0


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
    student_ids = df[ID_COL].copy() if ID_COL in df.columns else pd.Series(np.arange(len(df)), index=df.index)
    return X, y, student_ids


def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # SAME split logic and random_state as the training/defence scripts, so the
    # test/validation rows line up with the models' evaluation rows.
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
    return resolve_first_existing([os.path.join(mlp_dir, "zenodo_mlp_model.pt")], "MLP model (zenodo_mlp_model.pt)")


def resolve_mlp_preprocessor_path(mlp_dir):
    return resolve_first_existing([os.path.join(mlp_dir, "zenodo_mlp_preprocessor.pkl")], "MLP preprocessor (zenodo_mlp_preprocessor.pkl)")


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
    # zenodo preprocessor blocks: "num" (scaled numeric) -> "flag" (0/1) -> "cat"
    # (one-hot). Attackable perturbations live only in the numeric block; the flag
    # and one-hot blocks are folded together into the "categorical_or_other" mask
    # (any block that is not "num"), which only feeds the reported change-rate.
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
    mask_df.to_csv(os.path.join(output_dir, "zenodo_preprocessed_feature_masks.csv"), index=False)
    return perturbable_mask, numerical_mask, categorical_mask, feature_names, mask_df


# ============================================================
# ATTACK MACHINERY (verbatim logic from the zenodo attack/defence scripts)
# ============================================================
# ============================================================
# FEASIBILITY RULE ADDED 2026-09-30: semester credits vs yearly total
# ============================================================
# Verified on zenodo.csv (12,309 rows): credits_enrolled_semester_a +
# credits_enrolled_semester_b <= total_credits_enrolled_academic_year holds for
# 100% of rows. The yearly total may be larger than the two semesters because it
# also holds credits outside semesters A and B (for example internship or
# mobility credits). The attack moves the three columns independently, so this
# rule is re-imposed on the final adversarial rows while each student's own
# out-of-semester credits (clean total - clean sem_a - clean sem_b) are kept:
#     total_adv = max(total_adv, sem_a_adv + sem_b_adv + gap_clean)
# If that value would exceed the training maximum of the total, the two
# semester values are lowered proportionally (on the 0.5-credit grid) so that
# sem_a + sem_b + gap_clean fits under the maximum. Must run BEFORE
# enforce_credit_identities, so passed <= total and total_performance are
# computed from the repaired total.
SEM_A_ENROLLED_COL = "credits_enrolled_semester_a"
SEM_B_ENROLLED_COL = "credits_enrolled_semester_b"
YEAR_TOTAL_ENROLLED_COL = "total_credits_enrolled_academic_year"

def _floor_to_half(values):
    return np.floor(np.asarray(values, dtype=float) * 2.0 + 1e-9) / 2.0

def enforce_semester_total(X_adv_raw, X_clean_reset, train_min, train_max):
    cols = (SEM_A_ENROLLED_COL, SEM_B_ENROLLED_COL, YEAR_TOTAL_ENROLLED_COL)
    if not all(c in X_adv_raw.columns for c in cols):
        return X_adv_raw
    a = X_adv_raw[SEM_A_ENROLLED_COL].to_numpy(dtype=float).copy()
    b = X_adv_raw[SEM_B_ENROLLED_COL].to_numpy(dtype=float).copy()
    total = X_adv_raw[YEAR_TOTAL_ENROLLED_COL].to_numpy(dtype=float).copy()
    gap = (X_clean_reset[YEAR_TOTAL_ENROLLED_COL].to_numpy(dtype=float)
           - X_clean_reset[SEM_A_ENROLLED_COL].to_numpy(dtype=float)
           - X_clean_reset[SEM_B_ENROLLED_COL].to_numpy(dtype=float))
    gap = np.clip(gap, 0.0, None)
    t_max = float(train_max[YEAR_TOTAL_ENROLLED_COL]) if YEAR_TOTAL_ENROLLED_COL in train_max.index else np.inf
    a_min = float(train_min[SEM_A_ENROLLED_COL]) if SEM_A_ENROLLED_COL in train_min.index else 0.0
    b_min = float(train_min[SEM_B_ENROLLED_COL]) if SEM_B_ENROLLED_COL in train_min.index else 0.0
    # semesters too large to fit under the total maximum: shrink them proportionally
    room = t_max - gap
    over = (a + b) > room
    if over.any():
        s = a[over] + b[over]
        scale = np.where(s > 0, np.clip(room[over], 0.0, None) / s, 0.0)
        a[over] = np.maximum(_floor_to_half(a[over] * scale), a_min)
        b[over] = np.maximum(_floor_to_half(b[over] * scale), b_min)
    total = np.maximum(total, a + b + gap)
    total = np.minimum(total, t_max)
    X_adv_raw[SEM_A_ENROLLED_COL] = a
    X_adv_raw[SEM_B_ENROLLED_COL] = b
    X_adv_raw[YEAR_TOTAL_ENROLLED_COL] = total
    return X_adv_raw

def keep_integer_counts_in_range(X_adv_raw, train_min, train_max):
    # Rounding to whole numbers happens after the train min/max clip, so a count
    # whose training maximum is not a whole number (lms_total_minutes_* is stored
    # with decimals) could be rounded one unit above that maximum. Clip the rounded
    # counts to [ceil(train_min), floor(train_max)] so they stay in range.
    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=np.ceil(train_min[col]), upper=np.floor(train_max[col]))
    return X_adv_raw

def assert_feasible(bad_rows, attacked_row_mask, where):
    # Hard check: every attacked row must satisfy the feasibility rules.
    bad_rows = np.asarray(bad_rows).astype(bool)
    if attacked_row_mask is not None:
        bad_rows = bad_rows & np.asarray(attacked_row_mask).astype(bool)
    if bad_rows.any():
        raise AssertionError(f"{int(bad_rows.sum())} adversarial row(s) break a feasibility rule ({where}).")

def upv_feasibility_violations(X):
    # Boolean per row: True if the row breaks any enforced UPV rule. Used as a
    # hard check on the final adversarial rows (clean rows give 0 violations).
    X = X.reset_index(drop=True)
    tol = 1e-6
    enr = X["total_credits_enrolled_academic_year"].to_numpy(dtype=float)
    pas = X["total_credits_passed_academic_year"].to_numpy(dtype=float)
    a = X[SEM_A_ENROLLED_COL].to_numpy(dtype=float)
    b = X[SEM_B_ENROLLED_COL].to_numpy(dtype=float)
    perf = X["total_performance"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        perf_expected = np.where(enr > 0, pas / enr * 100.0, 0.0)
    bad = np.zeros(len(X), dtype=bool)
    bad |= a + b > enr + tol
    bad |= pas > enr + tol
    bad |= np.abs(perf - np.round(perf_expected, 2)) > 0.011
    return bad

def enforce_credit_identities(X_adv_raw):
    # applied AFTER clipping/rounding so the enforced values track the final inputs:
    #   1) passed capped at enrolled (may equal, never exceed), kept on the .5 grid
    #   2) total_performance = passed / enrolled * 100, 0 when enrolled == 0
    needed = (CREDITS_ENROLLED_COL, CREDITS_PASSED_COL, TOTAL_PERFORMANCE_COL)
    if not all(c in X_adv_raw.columns for c in needed):
        return X_adv_raw
    enrolled = X_adv_raw[CREDITS_ENROLLED_COL].to_numpy(dtype=float)
    passed = X_adv_raw[CREDITS_PASSED_COL].to_numpy(dtype=float)
    passed = np.minimum(passed, enrolled)
    passed = np.clip(passed, 0.0, None)
    passed = round_to_nearest_half(passed)
    X_adv_raw[CREDITS_PASSED_COL] = passed
    with np.errstate(divide="ignore", invalid="ignore"):
        performance = np.where(enrolled > 0, passed / enrolled * 100.0, 0.0)
    performance = np.clip(performance, 0.0, 100.0)
    X_adv_raw[TOTAL_PERFORMANCE_COL] = np.round(performance, 2)
    return X_adv_raw


def apply_domain_constraints(X_adv_raw, numeric_cols, train_min, train_max):
    # clip to train min/max, floor at 0, clip percentages to [0, 100], then round to
    # the correct grid for each feature. Frozen columns are restored by the caller.
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
    for col in ROUND_TO_HALF_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = round_to_nearest_half(X_adv_raw[col])
    for col in DECIMAL_2DP_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col].astype(float), 2)
    return X_adv_raw


def preprocessed_adv_to_raw_and_scaled(X_adv_preprocessed, X_clean_preprocessed, X_clean_raw, selected_features,
                                       preprocessor, train_min, train_max, perturbable_mask_preprocessed,
                                       attacked_row_mask=None, enforce_consistency=True):
    # inverse-scale the numeric block, restore frozen features and non-attacked
    # rows, apply zenodo domain constraints, then re-impose the credit identities.
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
    X_adv_raw = apply_domain_constraints(X_adv_raw, numeric_cols, train_min, train_max)
    if enforce_consistency:
        X_adv_raw = keep_integer_counts_in_range(X_adv_raw, train_min, train_max)
        X_adv_raw = enforce_semester_total(X_adv_raw, clean_reset, train_min, train_max)
        X_adv_raw = enforce_credit_identities(X_adv_raw)
        assert_feasible(upv_feasibility_violations(X_adv_raw), attacked_row_mask, 'UPV projection')
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
    # raw-space input-global Mahalanobis over the attackable numeric features (RF, XGB AND MLP DnD).
    def __init__(self, attackable_features, ridge=1e-6):
        self.attackable_features = list(attackable_features)
        self.ridge = float(ridge)
        self.cols = None
        self.mu = None
        self.cov_inv = None

    def fit(self, X_train, numeric_cols):
        self.cols = [c for c in numeric_cols if c in set(self.attackable_features)]
        M = X_train[self.cols].to_numpy(dtype=float)
        self.mu = np.nanmean(M, axis=0)
        cov = np.cov(M, rowvar=False) + self.ridge * np.eye(M.shape[1])
        self.cov_inv = np.linalg.pinv(cov)
        return self

    def scores(self, X):
        M = X[self.cols].to_numpy(dtype=float)
        d = M - self.mu
        return np.sqrt(np.clip(np.einsum("ij,ij->i", d @ self.cov_inv, d), 0, None))


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
    parser = argparse.ArgumentParser(description="After-defence (detect-and-default) proximity/sensitivity for zenodo. "
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
    print("ZENODO AFTER-DEFENCE (DETECT-AND-DEFAULT) PROXIMITY AND SENSITIVITY")
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
    mlp_summary_path = os.path.join(args.mlp_dir, "zenodo_mlp_summary.json")
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
    # dnd_train_mlp.py's load_mlp_checkpoint precedence.
    if mlp_info["best_threshold"] is not None:
        mlp_threshold, mlp_threshold_source = float(mlp_info["best_threshold"]), "zenodo_mlp_model.pt"
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
    print(f"Raw-space input-global Mahalanobis threshold (all models): {raw_maha_thr}")
    print()

    cases = []
    rf_threshold_source = xgb_threshold_source = None
    if "rf" in wanted:
        rf_model = joblib.load(resolve_pkl_model_path(args.rf_dir, "zenodo_rf_model.pkl"))
        rf_thr, rf_threshold_source = load_summary_threshold(os.path.join(args.rf_dir, "zenodo_rf_summary.json"))
        if rf_threshold_source == "fallback_0.5":
            print("WARNING: RF threshold not found in zenodo_rf_summary.json; using fallback 0.5. "
                  "Hiding-rate / detection numbers will NOT match the reported baseline. "
                  "Check --rf-dir.")
        cases.append({"model_key": "RF", "model_name": "Random Forest (DnD)", "attack_name": ("FGSM via baseline MLP" if fgsm_grad_source == "baseline_mlp" else "FGSM via defended MLP"),
                      "threshold": rf_thr, "target_model": rf_model, "art": defended_art,
                      "validator": raw_validator, "maha_thr": raw_maha_thr})
    if "xgb" in wanted:
        xgb_model = joblib.load(resolve_pkl_model_path(args.xgb_dir, "zenodo_xgb_model.pkl"))
        xgb_thr, xgb_threshold_source = load_summary_threshold(os.path.join(args.xgb_dir, "zenodo_xgb_summary.json"))
        if xgb_threshold_source == "fallback_0.5":
            print("WARNING: XGB threshold not found in zenodo_xgb_summary.json; using fallback 0.5. "
                  "Hiding-rate / detection numbers will NOT match the reported baseline. "
                  "Check --xgb-dir.")
        cases.append({"model_key": "XGB", "model_name": "XGBoost (DnD)", "attack_name": ("FGSM via baseline MLP" if fgsm_grad_source == "baseline_mlp" else "FGSM via defended MLP"),
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

    summary_path = os.path.join(args.output_dir, "zenodo_proximity_sensitivity_dnd_summary.csv")
    feature_path = os.path.join(args.output_dir, "zenodo_feature_sensitivity_dnd_summary.csv")
    top15_path = os.path.join(args.output_dir, "zenodo_top15_sensitive_features_dnd.csv")
    json_path = os.path.join(args.output_dir, "zenodo_proximity_sensitivity_dnd_summary.json")

    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)

    if not feature_df.empty:
        top15_df = (feature_df[feature_df["is_attackable_feature"] == True]
                    .sort_values(["model", "split", "row_scope", "mean_sensitivity"], ascending=[True, True, True, False])
                    .groupby(["model", "split", "row_scope"]).head(15).reset_index(drop=True))
    else:
        top15_df = pd.DataFrame()
    top15_df.to_csv(top15_path, index=False)

    excel_path = os.path.join(args.output_dir, "zenodo_proximity_sensitivity_dnd_analysis.xlsx")
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
        "dataset": args.csv_path,
        "feature_set": "zenodo (34 features: 22 numeric, 8 binary flags, 4 one-hot categoricals)",
        "defence": "Detect-and-default",
        "analysis": "Option C: (A) detection/override effect + (B) proximity/sensitivity split by detector outcome.",
        "key_caveat": "DnD does not change the adversarial input. Proximity/sensitivity of the input equal the "
                      "attack against the baseline model; the defence acts by flagging and overriding predictions. "
                      "row_scope=attacked_accepted are the adversarial rows that evaded the detector.",
        "attackable_note": "21 attackable raw features, all numeric. Frozen: n_courses (numeric), the 8 *_flag columns, "
                           "and the 4 one-hot categoricals.",
        "detectors": {"RF": "raw-space input-global Mahalanobis over attackable features",
                      "XGB": "raw-space input-global Mahalanobis over attackable features",
                      "MLP": "raw-space input-global Mahalanobis over attackable features (identical to RF/XGB)"},
        "fgsm_gradient_source_rf_xgb": fgsm_grad_source, "mlp_attack": "white-box PGD on the baseline MLP",
        "epsilon": args.epsilon, "pgd_step": args.pgd_step, "pgd_iterations": args.pgd_iterations,
        "pgd_iterations_note": "dnd_train_mlp.py default is 100 "
                               "and matches the standalone pgd_mlp.py attack.",
        "attack_mode": args.attack_mode, "attack_scope": args.attack_scope,
        "detection_target_fpr": args.detection_target_fpr,
        "detector_thresholds": {"raw_all_models": raw_maha_thr},
        "operating_thresholds": {"mlp": mlp_threshold, "mlp_source": mlp_threshold_source,
                                 "rf_source": rf_threshold_source, "xgb_source": xgb_threshold_source},
        "credit_identities": "passed <= enrolled; total_performance = passed/enrolled*100 (0 when enrolled == 0).",
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