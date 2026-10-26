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
# TARGET MODEL CONFIG (this file: XGBoost)
# ============================================================
# The gradient/source model is always the MLP, because XGBoost has no gradients
# we can differentiate through. We craft the perturbation on the MLP and then
# feed the adversarial rows to the XGBoost target (a transfer attack).
#
# Attack goal: HIDE dropout students, i.e. push true-dropout rows so the target
# predicts them as non-dropout. That is why the defaults are
# attack_mode = target_non_dropout and attack_scope = true_positive_only
# (attack only the true-dropout rows the target already caught -> cleanest
# evasion). See make_attack_labels / make_attack_scope_mask below.
TARGET_MODEL_NAME = "XGBoost"
TARGET_MODEL_TAG = "xgb"
DEFAULT_TARGET_RESULTS_DIR = "xgb_results_zenodo"
TARGET_MODEL_FILENAME = "zenodo_xgb_model.pkl"
TARGET_SUMMARY_FILENAME = "zenodo_xgb_summary.json"
DEFAULT_OUTPUT_DIR = "fgsm_xgb_zenodo_results"

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_MLP_DIR = "mlp_results_zenodo"

TARGET_COL = "target"
ID_COL = "student_id"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# epsilon is an L-inf bound in the MLP's PREPROCESSED space (standardized numeric
# + one-hot). It is NOT in raw units: 1.0 means up to 1.0 standard deviations on
# the scaled numeric block, not 1.0 credits / clicks / percentage points.
EPSILON_LIST = [1.0]

# fallback network shape only; checkpoint values override these when present
DEFAULT_HIDDEN_DIMS = (128, 64, 32)

# ============================================================
# FEATURE DEFINITIONS (zenodo.csv)
# ============================================================
# student_id and target are never predictors. academic_year is a one-hot
# categorical. The 8 *_flag columns are binary flags. Everything else is numeric.
NON_FEATURE_COLS = [ID_COL, TARGET_COL]

# Authoritative feature list is read from the MLP checkpoint at runtime. This
# fallback matches the training scripts: every column except id and target.
DEFAULT_SELECTED_FEATURES = [
    "academic_year",
    "n_courses",
    "courses_mean",
    "degree_size",
    "seniority",
    "highest_course_year_enrolled",
    "credits_enrolled_semester_a",
    "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year",
    "total_credits_passed_academic_year",
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
    "obtained_new_degree_flag",
    "adapted_studies_flag",
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "pass_ratio_sem_a_and_zero_flag",
    "pass_ratio_sem_b_and_zero_flag",
    "total_performance_and_zero_flag",
    "completion_rate_one_year_before_missing_flag",
    "completion_rate_two_years_before_missing_flag",
    "completion_rate_three_years_before_missing_flag",
    "lms_events_sem_a",
    "lms_events_sem_b",
    "lms_assignment_submissions_sem_a",
    "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_a",
    "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_a",
    "lms_total_minutes_sem_b",
    "attendance_days_sem_a",
    "attendance_days_sem_b",
]

# ============================================================
# ATTACKABLE FEATURES
# ============================================================
# Only these raw features may move. Everything else stays frozen: student_id,
# target, academic_year (one-hot), the enrolment/degree descriptors
# (n_courses, degree_size, seniority, highest_course_year_enrolled) and all 8
# binary flags.
ATTACKABLE_FEATURES = [
    "courses_mean",
    "credits_enrolled_semester_a",
    "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year",
    "total_credits_passed_academic_year",
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "lms_events_sem_a",
    "lms_events_sem_b",
    "lms_assignment_submissions_sem_a",
    "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_a",
    "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_a",
    "lms_total_minutes_sem_b",
    "attendance_days_sem_a",
    "attendance_days_sem_b",
]

# Explicitly frozen (for reporting / sanity check only). student_id and target
# are not model features so they are not listed here.
FROZEN_FEATURES_HINT = [
    "academic_year",
    "n_courses",
    "degree_size",
    "seniority",
    "highest_course_year_enrolled",
    "obtained_new_degree_flag",
    "adapted_studies_flag",
    "pass_ratio_sem_a_and_zero_flag",
    "pass_ratio_sem_b_and_zero_flag",
    "total_performance_and_zero_flag",
    "completion_rate_one_year_before_missing_flag",
    "completion_rate_two_years_before_missing_flag",
    "completion_rate_three_years_before_missing_flag",
]

# ============================================================
# DERIVED IDENTITIES (always enforced after the attack)
# ============================================================
# These three columns are attackable, so FGSM moves them independently and can
# produce combinations that are impossible in the real data. After every attack
# we force the following identities back onto the adversarial rows, so there is
# no on/off switch for this - it always runs:
#   1) total_credits_passed_academic_year <= total_credits_enrolled_academic_year
#      (passed may EQUAL enrolled, but never exceed it)
#   2) total_performance = (total_credits_passed_academic_year /
#                           total_credits_enrolled_academic_year) * 100
#      (when enrolled == 0 the ratio is undefined, so performance is set to 0,
#      matching the raw data where the 2 enrolled==0 rows have passed=0,
#      performance=0)
# Both identities were verified to hold for 100% of the clean rows.
CREDITS_ENROLLED_COL = "total_credits_enrolled_academic_year"
CREDITS_PASSED_COL = "total_credits_passed_academic_year"
TOTAL_PERFORMANCE_COL = "total_performance"

# ============================================================
# DOMAIN CONSTRAINTS (calibrated to zenodo.csv)
# ============================================================
# Integer-valued counts -> round back to whole numbers after the attack.
# lms_total_minutes_* are float in the raw data but requested as integers here.
ROUND_TO_INT_FEATURES = [
    "lms_events_sem_a",
    "lms_events_sem_b",
    "lms_assignment_submissions_sem_a",
    "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_a",
    "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_a",
    "lms_total_minutes_sem_b",
    "attendance_days_sem_a",
    "attendance_days_sem_b",
]

# Credit features -> snap to the nearest .0 or .5, whichever the value is closer
# to (e.g. 12.2 -> 12.0, 12.3 -> 12.5, 12.8 -> 13.0). See round_to_nearest_half.
ROUND_TO_HALF_FEATURES = [
    "credits_enrolled_semester_a",
    "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year",
    "total_credits_passed_academic_year",
]

# Decimal features -> round to 2 decimal places after the attack.
DECIMAL_2DP_FEATURES = [
    "courses_mean",
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
]

# Percentage features on a 0-100 scale -> clip to [0, 100]. courses_mean is a
# grade on a ~0-10 scale, so it is decimal but NOT a percentage and is not here.
PERCENTAGE_FEATURES = [
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
]

# Non-negative features (every attackable feature). The train min/max clip is the
# primary bound; this is an extra floor at 0.
NONNEGATIVE_FEATURES = sorted(set(ATTACKABLE_FEATURES))

# ============================================================
# MODEL (must match the training MLP exactly)
# ============================================================
class DropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=DEFAULT_HIDDEN_DIMS, dropout=0.30):
        super().__init__()
        layers = []
        previous_dim = input_dim
        # each hidden block: Linear -> BatchNorm -> ReLU -> Dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous_dim = hidden_dim
        # output layer: 2 logits (non-dropout, dropout)
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
    # convert numpy types to plain python so json.dump works
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

def round_to_nearest_half(x):
    # snap each value to the nearest .0 or .5, chosen by which one it is closer
    # to: 12.2 -> 12.0, 12.3 -> 12.5, 12.7 -> 12.5, 12.8 -> 13.0.
    return np.round(np.asarray(x, dtype=float) * 2.0) / 2.0

def normalize_target(series):
    # same mapping as rf_model.py / mlp_model.py / xgboost_model.py
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
    target_map = {
        "non-dropout": 0,
        "dropout": 1,
        "0": 0,
        "1": 1,
    }
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
        raise ValueError(
            f"Selected features missing from the CSV: {missing_features}\n"
            "The MLP checkpoint expects columns that are not in this dataset. "
            "Make sure --csv-path points to the same data the models were trained on."
        )
    X = df[selected_features].copy()
    y = normalize_target(df[TARGET_COL])
    if ID_COL in df.columns:
        student_ids = df[ID_COL].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    return X, y, student_ids

def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # SAME split logic and random_state as the RF / MLP / XGBoost training
    # scripts. The split depends only on n_samples, y (stratify) and
    # random_state, so the test/validation rows here are exactly the rows those
    # models were evaluated on, regardless of which feature columns are in X.
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index, name="row_id")
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids,
        test_size=0.30, stratify=y, random_state=random_state,
    )
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp, y_temp, id_temp,
        test_size=1.0 / 3.0, stratify=y_temp, random_state=random_state,
    )
    split_report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "validation_rows": len(X_val),
        "train_pct": len(X_train) / len(X),
        "test_pct": len(X_test) / len(X),
        "validation_pct": len(X_val) / len(X),
        "train_label_distribution": y_train.value_counts().sort_index().to_dict(),
        "test_label_distribution": y_test.value_counts().sort_index().to_dict(),
        "validation_label_distribution": y_val.value_counts().sort_index().to_dict(),
    }
    return (
        X_train, X_test, X_val,
        y_train, y_test, y_val,
        id_train, id_test, id_val,
        split_report,
    )

# ============================================================
# PATH RESOLUTION
# ============================================================
def resolve_first_existing(candidates, what):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{what} not found. Checked:\n" + "\n".join(candidates))

def resolve_target_model_path(target_results_dir):
    return resolve_first_existing(
        [os.path.join(target_results_dir, TARGET_MODEL_FILENAME)],
        f"{TARGET_MODEL_NAME} model ({TARGET_MODEL_FILENAME})",
    )

def resolve_target_summary_path(target_results_dir):
    path = os.path.join(target_results_dir, TARGET_SUMMARY_FILENAME)
    return path if os.path.exists(path) else None

def load_target_threshold(summary_path, fallback=0.5):
    # reuse the target model's tuned decision threshold; fall back to 0.5 if the
    # summary json is missing or has no threshold stored.
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        threshold = summary.get("selection", {}).get("best_threshold", None)
        if threshold is not None:
            return float(threshold), summary_path
    return float(fallback), "fallback_0.5"

def resolve_mlp_model_path(mlp_dir):
    return resolve_first_existing(
        [os.path.join(mlp_dir, "zenodo_mlp_model.pt")],
        "MLP model (zenodo_mlp_model.pt)",
    )

def resolve_mlp_preprocessor_path(mlp_dir):
    return resolve_first_existing(
        [os.path.join(mlp_dir, "zenodo_mlp_preprocessor.pkl")],
        "MLP preprocessor (zenodo_mlp_preprocessor.pkl)",
    )

def load_torch_checkpoint(path, device):
    # pytorch versions differ on weights_only support/default
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def read_mlp_checkpoint(path, device):
    # read the checkpoint once and pull out everything we need. selected_features
    # from here is the authoritative feature list for building X.
    obj = load_torch_checkpoint(path, device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        info = {
            "hidden_dims": tuple(obj.get("hidden_dims", DEFAULT_HIDDEN_DIMS)),
            "dropout": float(obj.get("dropout", 0.30)),
            "state_dict": obj["model_state_dict"],
            "selected_features": obj.get("selected_features", None),
            "best_threshold": obj.get("best_threshold", None),
            "input_dim": obj.get("input_dim", None),
        }
    else:
        # bare state_dict
        info = {
            "hidden_dims": DEFAULT_HIDDEN_DIMS,
            "dropout": 0.30,
            "state_dict": obj,
            "selected_features": None,
            "best_threshold": None,
            "input_dim": None,
        }
    return info

def build_mlp_from_state(info, input_dim, device):
    model = DropoutMLP(
        input_dim=input_dim,
        hidden_dims=info["hidden_dims"],
        dropout=info["dropout"],
    ).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    return model

def wrap_mlp_for_art(model, input_dim):
    # ART needs an optimizer even for inference-only attacks
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
    # Build the ART mask in preprocessed space. The MLP ColumnTransformer emits
    # columns in the order: numeric block -> flag block -> categorical one-hot
    # block. Every attackable feature in zenodo is numeric, so only numeric
    # columns whose raw name is attackable may move; the flag and one-hot blocks
    # stay frozen (mask starts as all-zeros).
    os.makedirs(output_dir, exist_ok=True)
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)

    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(
            f"Attackable features not in selected_features: {missing_attackable}"
        )
    # every attackable feature must land in the numeric block (never flag/cat)
    attackable_not_numeric = sorted([f for f in attackable_raw if f not in numeric_cols])
    if attackable_not_numeric:
        raise ValueError(
            "These attackable features are not in the numeric preprocessing block "
            f"(they were routed to flags/one-hot): {attackable_not_numeric}"
        )

    mask = np.zeros(input_dim, dtype=np.float32)
    rows = []
    # numeric block starts at column 0 and spans len(numeric_cols) columns
    for idx, col in enumerate(numeric_cols):
        is_attackable = col in attackable_raw
        mask[idx] = 1.0 if is_attackable else 0.0
        rows.append({
            "preprocessed_index": idx,
            "raw_feature": col,
            "block": "num",
            "perturbable_mask": int(is_attackable),
            "attack_status": "attackable" if is_attackable else "frozen",
        })

    # also save a raw-level mask because it is easier to read
    raw_mask_df = pd.DataFrame({
        "feature": selected_features,
        "raw_attackable": [int(f in attackable_raw) for f in selected_features],
        "round_to_int": [int(f in ROUND_TO_INT_FEATURES) for f in selected_features],
        "round_to_half": [int(f in ROUND_TO_HALF_FEATURES) for f in selected_features],
        "round_to_2dp": [int(f in DECIMAL_2DP_FEATURES) for f in selected_features],
        "percentage_0_100": [int(f in PERCENTAGE_FEATURES) for f in selected_features],
        "attack_status": ["attackable" if f in attackable_raw else "frozen" for f in selected_features],
    })
    preprocessed_mask_df = pd.DataFrame(rows)

    raw_mask_df.to_csv(os.path.join(output_dir, "zenodo_attack_mask_raw_features.csv"), index=False)
    preprocessed_mask_df.to_csv(os.path.join(output_dir, "zenodo_attack_mask_preprocessed_numeric_block.csv"), index=False)
    np.save(os.path.join(output_dir, "zenodo_perturbable_mask_preprocessed.npy"), mask)

    if int(mask.sum()) == 0:
        raise ValueError("No preprocessed feature is attackable. Check ATTACKABLE_FEATURES.")
    return mask, raw_mask_df, preprocessed_mask_df

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
    # Always applied after the attack, AFTER the components have been clipped and
    # rounded, so the enforced values track the final perturbed inputs:
    #   1) passed is capped at enrolled (passed may equal enrolled, never exceed)
    #   2) total_performance is recomputed as (passed / enrolled) * 100, with
    #      performance = 0 wherever enrolled == 0 (ratio undefined).
    needed = (CREDITS_ENROLLED_COL, CREDITS_PASSED_COL, TOTAL_PERFORMANCE_COL)
    if not all(c in X_adv_raw.columns for c in needed):
        return X_adv_raw

    enrolled = X_adv_raw[CREDITS_ENROLLED_COL].to_numpy(dtype=float)
    passed = X_adv_raw[CREDITS_PASSED_COL].to_numpy(dtype=float)

    # 1) passed cannot exceed enrolled; keep it non-negative and on the .5 grid
    passed = np.minimum(passed, enrolled)
    passed = np.clip(passed, 0.0, None)
    passed = round_to_nearest_half(passed)
    X_adv_raw[CREDITS_PASSED_COL] = passed

    # 2) performance = passed / enrolled * 100, or 0 when enrolled == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        performance = np.where(enrolled > 0, passed / enrolled * 100.0, 0.0)
    performance = np.clip(performance, 0.0, 100.0)
    X_adv_raw[TOTAL_PERFORMANCE_COL] = np.round(performance, 2)
    return X_adv_raw

def apply_domain_constraints(X_adv_raw, numeric_cols, train_min, train_max):
    # clip to train min/max, floor at 0, clip percentages to [0, 100], then round
    # to the correct grid for each feature. Only attackable columns are touched
    # here; frozen columns are restored by the caller.
    # 1) clip to the range actually observed in the training split
    for col in numeric_cols:
        if col in X_adv_raw.columns and col in train_min.index and col in train_max.index:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=train_min[col], upper=train_max[col])
    # 2) non-negative floor
    for col in NONNEGATIVE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0)
    # 3) percentages bounded to [0, 100]
    for col in PERCENTAGE_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = X_adv_raw[col].clip(lower=0, upper=100)
    # 4) rounding grids (done after clipping so rounded values still obey bounds)
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

def preprocessed_adv_to_raw_and_scaled(
    X_adv_preprocessed,
    X_clean_preprocessed,
    X_clean_raw,
    selected_features,
    preprocessor,
    train_min,
    train_max,
    perturbable_mask_preprocessed,
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
    # write adversarial numeric block back to raw columns
    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]

    # restore non-attackable raw features to their clean values
    clean_reset = X_clean_raw[selected_features].reset_index(drop=True)
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values

    # apply per-feature domain constraints (clip + round) to the attacked columns
    X_adv_raw = apply_domain_constraints(X_adv_raw, numeric_cols, train_min, train_max)

    # always enforce the credit identities on the finalized components:
    #   passed <= enrolled, and total_performance = passed / enrolled * 100.
    # Done after clipping/rounding so the enforced values track the final inputs.
    X_adv_raw = keep_integer_counts_in_range(X_adv_raw, train_min, train_max)
    X_adv_raw = enforce_semester_total(X_adv_raw, clean_reset, train_min, train_max)
    X_adv_raw = enforce_credit_identities(X_adv_raw)
    assert_feasible(upv_feasibility_violations(X_adv_raw), attacked_row_mask, 'UPV projection')

    # restore frozen features again after clipping/rounding (belt and suspenders)
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values

    # restore non-attacked rows completely (column-by-column to avoid dtype warnings)
    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        not_attacked = ~attacked_row_mask
        if not_attacked.any():
            for col in selected_features:
                values = clean_reset.loc[not_attacked, col]
                if pd.api.types.is_numeric_dtype(X_adv_raw[col]):
                    values = values.astype(X_adv_raw[col].dtype, copy=False)
                X_adv_raw.loc[not_attacked, col] = values.values

    X_projected_preprocessed = transform_to_float32(preprocessor, X_adv_raw[selected_features])

    # hard-restore frozen preprocessed dimensions
    frozen = perturbable_mask_preprocessed == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]

    # restore non-attacked rows in preprocessed space too
    if attacked_row_mask is not None:
        not_attacked = ~attacked_row_mask
        if not_attacked.any():
            X_projected_preprocessed[not_attacked, :] = X_clean_preprocessed[not_attacked, :]

    return X_adv_raw[selected_features], X_projected_preprocessed

# ============================================================
# EVALUATION AND SAVING
# ============================================================
def evaluate_target_with_threshold(target_model, X_raw, y_true, threshold, condition, epsilon, split):
    # The target model pkl is a full imblearn pipeline (preprocessor + model), so
    # it takes the RAW dataframe and does its own preprocessing internally.
    y_true = np.asarray(y_true).astype(int)
    y_proba = target_model.predict_proba(X_raw)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "model": TARGET_MODEL_NAME,
        "condition": condition,
        "epsilon": float(epsilon),
        "split": split,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_dropout": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_dropout": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_dropout": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    return result, cm, y_pred, y_proba

def perturbation_stats(X_clean_preprocessed, X_adv_preprocessed, perturbable_mask, attacked_row_mask=None):
    # summarize how much moved, split by attackable vs frozen dimensions
    diff = X_adv_preprocessed - X_clean_preprocessed
    pm = perturbable_mask.astype(bool)
    frozen = ~pm
    stats = {
        "mean_abs_perturbation_all_preprocessed": float(np.abs(diff).mean()),
        "mean_abs_perturbation_attacked_preprocessed": float(np.abs(diff[:, pm]).mean()),
        "max_abs_perturbation_attacked_preprocessed": float(np.abs(diff[:, pm]).max()),
        "max_abs_perturbation_frozen_preprocessed": float(np.abs(diff[:, frozen]).max()) if frozen.any() else 0.0,
    }
    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        if attacked_row_mask.any():
            attacked_diff = diff[attacked_row_mask]
            stats["mean_abs_perturbation_attacked_rows_attackable_preprocessed"] = float(np.abs(attacked_diff[:, pm]).mean())
            stats["max_abs_perturbation_attacked_rows_attackable_preprocessed"] = float(np.abs(attacked_diff[:, pm]).max())
        else:
            stats["mean_abs_perturbation_attacked_rows_attackable_preprocessed"] = 0.0
            stats["max_abs_perturbation_attacked_rows_attackable_preprocessed"] = 0.0
    return stats

def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)

def save_attack_predictions(
    student_ids, y_true, clean_pred, adv_pred, clean_proba, adv_proba, epsilon, split, path,
):
    y_true = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        ID_COL: student_ids.values,
        "split": split,
        "epsilon": float(epsilon),
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        f"{TARGET_MODEL_TAG}_clean_pred_label": clean_pred,
        f"{TARGET_MODEL_TAG}_clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        f"{TARGET_MODEL_TAG}_adv_pred_label": adv_pred,
        f"{TARGET_MODEL_TAG}_adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        f"{TARGET_MODEL_TAG}_clean_prob_dropout": clean_proba,
        f"{TARGET_MODEL_TAG}_adv_prob_dropout": adv_proba,
        "prob_dropout_change": adv_proba - clean_proba,
        "prediction_changed": clean_pred != adv_pred,
        "dropout_hidden_tp_to_fn": ((y_true == 1) & (clean_pred == 1) & (adv_pred == 0)),
        "false_alarm_tn_to_fp": ((y_true == 0) & (clean_pred == 0) & (adv_pred == 1)),
    })
    pred_df.to_csv(path, index=False)

def save_clean_vs_adv_sample(
    X_clean_raw, X_adv_raw, y_true, clean_pred, adv_pred, clean_proba, adv_proba,
    selected_features, epsilon, split, path, max_samples=50,
):
    # per-feature clean vs adversarial values for a handful of rows, for inspection
    n = min(max_samples, len(X_clean_raw))
    rows = []
    X_clean_raw = X_clean_raw.reset_index(drop=True)
    X_adv_raw = X_adv_raw.reset_index(drop=True)
    y_true = np.asarray(y_true).astype(int)
    for i in range(n):
        row = {
            "sample_id": i,
            "split": split,
            "epsilon": float(epsilon),
            "true_label": int(y_true[i]),
            "true_target": "dropout" if y_true[i] == 1 else "non-dropout",
            f"{TARGET_MODEL_TAG}_clean_probability_dropout": float(clean_proba[i]),
            f"{TARGET_MODEL_TAG}_adversarial_probability_dropout": float(adv_proba[i]),
            f"{TARGET_MODEL_TAG}_clean_prediction": int(clean_pred[i]),
            f"{TARGET_MODEL_TAG}_adversarial_prediction": int(adv_pred[i]),
            "prediction_changed": bool(clean_pred[i] != adv_pred[i]),
            "hidden_dropout": bool(y_true[i] == 1 and clean_pred[i] == 1 and adv_pred[i] == 0),
        }
        for feature in selected_features:
            clean_val = X_clean_raw.loc[i, feature]
            adv_val = X_adv_raw.loc[i, feature]
            row[f"clean_{feature}"] = clean_val
            row[f"adv_{feature}"] = adv_val
            c = pd.to_numeric(clean_val, errors="coerce")
            a = pd.to_numeric(adv_val, errors="coerce")
            row[f"diff_{feature}"] = (a - c) if pd.notna(a) and pd.notna(c) else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

# ============================================================
# ATTACK LABELS AND SCOPE
# ============================================================
def make_attack_labels(attack_mode, y_true):
    if attack_mode == "untargeted":
        # source-model attack: maximize loss against the TRUE labels
        return one_hot(y_true), False
    if attack_mode == "target_non_dropout":
        # push selected rows toward class 0 = non-dropout (hide the dropout)
        target_labels = np.zeros(len(y_true), dtype=int)
        return one_hot(target_labels), True
    raise ValueError("attack_mode must be 'untargeted' or 'target_non_dropout'.")

def make_attack_scope_mask(attack_scope, y_true, clean_pred):
    # decide which rows are actually perturbed:
    # all = every row; dropout_only = true dropout rows;
    # true_positive_only = true dropout rows the target already detected (cleanest evasion)
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
    fgsm, X_clean_preprocessed, y_true, clean_pred, attack_mode, attack_scope, perturbable_mask,
):
    # run FGSM only on the selected rows; untouched rows stay clean
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()
    if int(attack_idx.sum()) == 0:
        return X_adv_preprocessed, attack_idx
    attack_y, _ = make_attack_labels(attack_mode, y_true=np.asarray(y_true)[attack_idx])
    X_adv_subset = fgsm.generate(
        x=X_clean_preprocessed[attack_idx],
        y=attack_y,
        mask=perturbable_mask,
    )
    # hard-restore frozen dimensions for attacked rows
    frozen = perturbable_mask == 0
    X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
    X_adv_preprocessed[attack_idx] = X_adv_subset
    return X_adv_preprocessed, attack_idx

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_single_fgsm_experiment(
    df, csv_path, output_dir, target_results_dir, mlp_dir,
    epsilon_list, random_state, attack_mode, attack_scope,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"FGSM ATTACK ON {TARGET_MODEL_NAME.upper()} VIA SOURCE MLP (transfer)")
    print("=" * 80)
    print(f"CSV path         : {csv_path}")
    print(f"Output dir       : {output_dir}")
    print(f"Target dir       : {target_results_dir}")
    print(f"MLP (source) dir : {mlp_dir}")
    print(f"Attack mode      : {attack_mode}")
    print(f"Attack scope     : {attack_scope}")
    print(f"Credit identities: always enforced (passed<=enrolled; performance=passed/enrolled*100)")
    print(f"Device           : {device}")
    print(f"Dataset shape    : {df.shape}")
    print()

    # source MLP checkpoint first: it defines the authoritative feature list
    mlp_model_path = resolve_mlp_model_path(mlp_dir)
    mlp_preprocessor_path = resolve_mlp_preprocessor_path(mlp_dir)
    mlp_info = read_mlp_checkpoint(mlp_model_path, device)

    if mlp_info["selected_features"] is not None:
        selected_features = list(mlp_info["selected_features"])
        feature_source = "mlp_checkpoint"
    else:
        selected_features = DEFAULT_SELECTED_FEATURES.copy()
        feature_source = "fallback_default"
    print(f"Feature list source: {feature_source} ({len(selected_features)} features)")

    # sanity: attackable set must be a subset of the model's features, and no
    # frozen-by-spec feature may be attackable
    missing_attackable = [f for f in ATTACKABLE_FEATURES if f not in selected_features]
    if missing_attackable:
        raise ValueError(f"Attackable features not in the model feature list: {missing_attackable}")
    frozen_but_attackable = [f for f in FROZEN_FEATURES_HINT if f in ATTACKABLE_FEATURES]
    if frozen_but_attackable:
        raise ValueError(f"These must stay frozen but are in ATTACKABLE_FEATURES: {frozen_but_attackable}")

    X, y, student_ids = prepare_features_and_label(df, selected_features)
    (
        X_train, X_test, X_val,
        y_train, y_test, y_val,
        id_train, id_test, id_val,
        split_report,
    ) = split_70_20_10_stratified(X, y, student_ids, random_state=random_state)
    print("Split report:")
    print(json.dumps(to_python(split_report), indent=4))
    print()

    # load target model (XGBoost) and its tuned threshold
    target_model_path = resolve_target_model_path(target_results_dir)
    target_summary_path = resolve_target_summary_path(target_results_dir)
    target_threshold, target_threshold_source = load_target_threshold(target_summary_path)
    target_model = joblib.load(target_model_path)

    # load source MLP preprocessor and build the torch model
    preprocessor = joblib.load(mlp_preprocessor_path)
    X_train_preprocessed = transform_to_float32(preprocessor, X_train)
    X_test_preprocessed = transform_to_float32(preprocessor, X_test)
    X_val_preprocessed = transform_to_float32(preprocessor, X_val)
    input_dim = X_test_preprocessed.shape[1]

    if mlp_info["input_dim"] is not None and int(mlp_info["input_dim"]) != int(input_dim):
        raise ValueError(
            f"Preprocessed input_dim ({input_dim}) != MLP checkpoint input_dim "
            f"({mlp_info['input_dim']}). The preprocessor and checkpoint do not match. "
            "Make sure --mlp-dir points to the run that produced this dataset's model."
        )

    mlp_model = build_mlp_from_state(mlp_info, input_dim=input_dim, device=device)
    art_mlp = wrap_mlp_for_art(mlp_model, input_dim=input_dim)

    perturbable_mask, raw_mask_df, preprocessed_mask_df = make_preprocessed_attack_mask(
        preprocessor=preprocessor,
        selected_features=selected_features,
        input_dim=input_dim,
        output_dir=output_dir,
    )

    print(f"Loaded target model ({TARGET_MODEL_NAME}):")
    print(target_model_path)
    print(f"Target threshold: {target_threshold} | source: {target_threshold_source}")
    print()
    print("Loaded source MLP model:")
    print(mlp_model_path)
    print("Loaded MLP preprocessor:")
    print(mlp_preprocessor_path)
    print(f"MLP hidden dims: {mlp_info['hidden_dims']} | dropout: {mlp_info['dropout']}")
    print()
    print("Attackable raw features:")
    print(raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist())
    print()
    print("Frozen raw features:")
    print(raw_mask_df.loc[raw_mask_df["raw_attackable"] == 0, "feature"].tolist())
    print()
    print("Preprocessed attackable dimensions:")
    print(int(perturbable_mask.sum()), "of", len(perturbable_mask))
    print()

    # clean target baseline on test and validation
    clean_test_result, clean_test_cm, clean_test_pred, clean_test_proba = evaluate_target_with_threshold(
        target_model, X_test, y_test.values, target_threshold, "Clean", 0.0, "test",
    )
    clean_val_result, clean_val_cm, clean_val_pred, clean_val_proba = evaluate_target_with_threshold(
        target_model, X_val, y_val.values, target_threshold, "Clean", 0.0, "validation",
    )
    save_confusion_matrix(clean_test_cm, os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_clean_test_confusion_matrix.csv"))
    save_confusion_matrix(clean_val_cm, os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_clean_validation_confusion_matrix.csv"))

    # train min/max used to clip adversarial raw values
    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)

    all_results = [clean_test_result, clean_val_result]
    all_stats = []

    print("Clean test result:")
    print(clean_test_result)
    print("Clean test confusion matrix:")
    print(clean_test_cm)
    print()
    print("Clean validation result:")
    print(clean_val_result)
    print("Clean validation confusion matrix:")
    print(clean_val_cm)
    print()

    for epsilon in epsilon_list:
        print("-" * 80)
        print(f"FGSM epsilon = {epsilon} (L-inf in preprocessed/standardized space)")
        print("-" * 80)

        _, targeted = make_attack_labels(attack_mode, y_true=y_test.values[:1])
        fgsm = FastGradientMethod(
            estimator=art_mlp,
            eps=float(epsilon),
            targeted=targeted,
            batch_size=128,
        )

        # generate FGSM in preprocessed space for the chosen scope
        X_test_adv_pre_raw, test_attack_idx = generate_fgsm_for_scope(
            fgsm, X_test_preprocessed, y_test.values, clean_test_pred,
            attack_mode, attack_scope, perturbable_mask,
        )
        X_val_adv_pre_raw, val_attack_idx = generate_fgsm_for_scope(
            fgsm, X_val_preprocessed, y_val.values, clean_val_pred,
            attack_mode, attack_scope, perturbable_mask,
        )

        # map adversarial examples back to raw space, apply constraints
        X_test_adv_raw, X_test_adv_preprocessed = preprocessed_adv_to_raw_and_scaled(
            X_adv_preprocessed=X_test_adv_pre_raw,
            X_clean_preprocessed=X_test_preprocessed,
            X_clean_raw=X_test,
            selected_features=selected_features,
            preprocessor=preprocessor,
            train_min=train_min,
            train_max=train_max,
            perturbable_mask_preprocessed=perturbable_mask,
            attacked_row_mask=test_attack_idx,
        )
        X_val_adv_raw, X_val_adv_preprocessed = preprocessed_adv_to_raw_and_scaled(
            X_adv_preprocessed=X_val_adv_pre_raw,
            X_clean_preprocessed=X_val_preprocessed,
            X_clean_raw=X_val,
            selected_features=selected_features,
            preprocessor=preprocessor,
            train_min=train_min,
            train_max=train_max,
            perturbable_mask_preprocessed=perturbable_mask,
            attacked_row_mask=val_attack_idx,
        )

        # evaluate the adversarial raw examples on the target model
        test_result, test_cm, test_adv_pred, test_adv_proba = evaluate_target_with_threshold(
            target_model, X_test_adv_raw, y_test.values, target_threshold,
            f"FGSM {attack_mode}", epsilon, "test",
        )
        val_result, val_cm, val_adv_pred, val_adv_proba = evaluate_target_with_threshold(
            target_model, X_val_adv_raw, y_val.values, target_threshold,
            f"FGSM {attack_mode}", epsilon, "validation",
        )

        # attack-effect counters for test
        hidden_test = int(np.sum((y_test.values == 1) & (clean_test_pred == 1) & (test_adv_pred == 0)))
        false_alarm_test = int(np.sum((y_test.values == 0) & (clean_test_pred == 0) & (test_adv_pred == 1)))
        changed_test = int(np.sum(clean_test_pred != test_adv_pred))
        test_result["attack_scope"] = attack_scope
        test_result["attacked_rows_count"] = int(test_attack_idx.sum())
        test_result["attacked_dropout_rows_count"] = int(np.sum(test_attack_idx & (y_test.values == 1)))
        test_result["prediction_changed_count"] = changed_test
        test_result["dropout_to_nondropout_flips"] = hidden_test
        test_result["nondropout_to_dropout_flips"] = false_alarm_test

        # attack-effect counters for validation
        hidden_val = int(np.sum((y_val.values == 1) & (clean_val_pred == 1) & (val_adv_pred == 0)))
        false_alarm_val = int(np.sum((y_val.values == 0) & (clean_val_pred == 0) & (val_adv_pred == 1)))
        changed_val = int(np.sum(clean_val_pred != val_adv_pred))
        val_result["attack_scope"] = attack_scope
        val_result["attacked_rows_count"] = int(val_attack_idx.sum())
        val_result["attacked_dropout_rows_count"] = int(np.sum(val_attack_idx & (y_val.values == 1)))
        val_result["prediction_changed_count"] = changed_val
        val_result["dropout_to_nondropout_flips"] = hidden_val
        val_result["nondropout_to_dropout_flips"] = false_alarm_val

        all_results.extend([test_result, val_result])

        test_stats = perturbation_stats(
            X_test_preprocessed, X_test_adv_preprocessed, perturbable_mask,
            attacked_row_mask=test_attack_idx,
        )
        test_stats.update({
            "epsilon": float(epsilon), "split": "test",
            "attack_scope": attack_scope, "attacked_rows_count": int(test_attack_idx.sum()),
        })
        val_stats = perturbation_stats(
            X_val_preprocessed, X_val_adv_preprocessed, perturbable_mask,
            attacked_row_mask=val_attack_idx,
        )
        val_stats.update({
            "epsilon": float(epsilon), "split": "validation",
            "attack_scope": attack_scope, "attacked_rows_count": int(val_attack_idx.sum()),
        })
        all_stats.extend([test_stats, val_stats])

        # sanity check: frozen dimensions must not move
        if test_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("Frozen transformed features changed in test set.")
        if val_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("Frozen transformed features changed in validation set.")

        print("Test result:")
        print(test_result)
        print("Test confusion matrix:")
        print(test_cm)
        print()
        print("Validation result:")
        print(val_result)
        print("Validation confusion matrix:")
        print(val_cm)
        print()

        # save arrays and raw CSVs
        eps_tag = str(epsilon).replace(".", "p")
        np.save(os.path.join(output_dir, f"zenodo_X_test_fgsm_preprocessed_eps_{eps_tag}.npy"), X_test_adv_preprocessed)
        np.save(os.path.join(output_dir, f"zenodo_X_validation_fgsm_preprocessed_eps_{eps_tag}.npy"), X_val_adv_preprocessed)
        X_test_adv_raw.to_csv(os.path.join(output_dir, f"zenodo_X_test_fgsm_raw_eps_{eps_tag}.csv"), index=False)
        X_val_adv_raw.to_csv(os.path.join(output_dir, f"zenodo_X_validation_fgsm_raw_eps_{eps_tag}.csv"), index=False)
        save_confusion_matrix(test_cm, os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_fgsm_test_confusion_matrix_eps_{eps_tag}.csv"))
        save_confusion_matrix(val_cm, os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_fgsm_validation_confusion_matrix_eps_{eps_tag}.csv"))
        save_attack_predictions(
            id_test, y_test.values, clean_test_pred, test_adv_pred,
            clean_test_proba, test_adv_proba, epsilon, "test",
            os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_fgsm_test_predictions_eps_{eps_tag}.csv"),
        )
        save_attack_predictions(
            id_val, y_val.values, clean_val_pred, val_adv_pred,
            clean_val_proba, val_adv_proba, epsilon, "validation",
            os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_fgsm_validation_predictions_eps_{eps_tag}.csv"),
        )
        save_clean_vs_adv_sample(
            X_test, X_test_adv_raw, y_test.values, clean_test_pred, test_adv_pred,
            clean_test_proba, test_adv_proba, selected_features, epsilon, "test",
            os.path.join(output_dir, f"zenodo_clean_vs_fgsm_test_sample_eps_{eps_tag}.csv"), max_samples=50,
        )
        save_clean_vs_adv_sample(
            X_val, X_val_adv_raw, y_val.values, clean_val_pred, val_adv_pred,
            clean_val_proba, val_adv_proba, selected_features, epsilon, "validation",
            os.path.join(output_dir, f"zenodo_clean_vs_fgsm_validation_sample_eps_{eps_tag}.csv"), max_samples=50,
        )

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)
    results_path = os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_fgsm_attack_results.csv")
    stats_path = os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_fgsm_perturbation_stats.csv")
    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "attack": f"FGSM generated using source MLP gradients, evaluated on {TARGET_MODEL_NAME}",
        "attack_goal": "Hide dropout students -> make the target predict non-dropout.",
        "attack_mode": attack_mode,
        "attack_scope": attack_scope,
        "epsilons": epsilon_list,
        "epsilon_space": "L-inf bound in the MLP preprocessed/standardized feature space",
        "target_model": TARGET_MODEL_NAME,
        "target_model_path": target_model_path,
        "target_threshold": target_threshold,
        "target_threshold_source": target_threshold_source,
        "gradient_model": "Source MLP trained on true labels",
        "mlp_model_path": mlp_model_path,
        "mlp_preprocessor_path": mlp_preprocessor_path,
        "mlp_hidden_dims": list(mlp_info["hidden_dims"]),
        "mlp_dropout": mlp_info["dropout"],
        "feature_list_source": feature_source,
        "n_raw_features": len(selected_features),
        "n_preprocessed_features": int(input_dim),
        "n_attackable_preprocessed_features": int(perturbable_mask.sum()),
        "n_frozen_preprocessed_features": int(len(perturbable_mask) - perturbable_mask.sum()),
        "attackable_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist(),
        "frozen_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 0, "feature"].tolist(),
        "derived_identities_note": (
            "Always enforced after the attack: (1) total_credits_passed_academic_year "
            "is capped at total_credits_enrolled_academic_year (may equal, never "
            "exceed); (2) total_performance is recomputed as passed / enrolled * 100, "
            "set to 0 when enrolled == 0. Both identities hold for 100% of the clean "
            "rows."
        ),
        "constraints": {
            "frozen_features": ("Restored to clean values after FGSM (academic_year, "
                                "n_courses, degree_size, seniority, "
                                "highest_course_year_enrolled, and all 8 *_flag columns)."),
            "non_attacked_rows": "Restored completely to clean values.",
            "raw_value_clip": "Numeric features clipped to train min/max when available.",
            "percentage_features": ("pass_ratio_sem_a/b, total_performance and the three "
                                    "completion_rate_*_before features clipped to [0, 100]."),
            "half_credit_features": ("credits_enrolled_semester_a/b, "
                                     "total_credits_enrolled_academic_year and "
                                     "total_credits_passed_academic_year snapped to the "
                                     "nearest .0 or .5 (whichever is closer)."),
            "credit_identities": ("passed <= enrolled and total_performance = "
                                  "passed / enrolled * 100 are always re-imposed."),
            "integer_count_features": ("lms_* and attendance_days_* rounded to whole "
                                       "numbers (lms_total_minutes_* included, even though "
                                       "they are float in the raw data)."),
            "decimal_features": ("courses_mean plus the percentage features rounded to 2 "
                                 "decimal places."),
            "nonnegative_features": "Every attackable feature clipped to a lower bound of 0.",
        },
        "split": split_report,
        "results_file": results_path,
        "perturbation_stats_file": stats_path,
    }
    summary_path = os.path.join(output_dir, f"zenodo_{TARGET_MODEL_TAG}_fgsm_attack_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Stats saved to  : {stats_path}")
    print(f"Summary saved to: {summary_path}")
    return summary

# ============================================================
# ARGUMENTS
# ============================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=f"FGSM transfer attack on {TARGET_MODEL_NAME} using the source MLP model (zenodo)."
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH,
                        help=f"Dataset CSV. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--target-results-dir", default=DEFAULT_TARGET_RESULTS_DIR,
                        help=f"{TARGET_MODEL_NAME} results folder. Default: {DEFAULT_TARGET_RESULTS_DIR}")
    parser.add_argument("--mlp-dir", default=DEFAULT_MLP_DIR,
                        help=f"MLP source-model folder. Default: {DEFAULT_MLP_DIR}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"FGSM output folder. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE,
                        help=f"Random state. Default: {DEFAULT_RANDOM_STATE}")
    parser.add_argument(
        "--attack-mode", default="target_non_dropout",
        choices=["untargeted", "target_non_dropout"],
        help=("untargeted = maximize loss against true labels. "
              "target_non_dropout = targeted attack pushing predictions toward non-dropout "
              "(this is the one that hides dropout students)."),
    )
    parser.add_argument(
        "--attack-scope", default="true_positive_only",
        choices=["all", "dropout_only", "true_positive_only"],
        help=("all = attack every row. "
              "dropout_only = attack true dropout rows only. "
              "true_positive_only = attack only dropout rows the target already detected "
              "(cleanest evasion; default)."),
    )
    parser.add_argument("--epsilons", default=",".join(str(e) for e in EPSILON_LIST),
                        help=f"Comma-separated epsilon list. Default: {','.join(str(e) for e in EPSILON_LIST)}")
    return parser

def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")
    epsilon_list = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    df = pd.read_csv(args.csv_path)
    run_single_fgsm_experiment(
        df=df,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        target_results_dir=args.target_results_dir,
        mlp_dir=args.mlp_dir,
        epsilon_list=epsilon_list,
        random_state=args.random_state,
        attack_mode=args.attack_mode,
        attack_scope=args.attack_scope,
    )

if __name__ == "__main__":
    main()