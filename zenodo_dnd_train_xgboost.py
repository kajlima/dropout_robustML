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
# Detect-and-default defence for the XGBoost model on zenodo.csv. A Mahalanobis
# validator (fit on the clean training split, in the RAW attackable feature space)
# flags inputs whose feature COMBINATION is unlikely under the training data. A
# flagged row is treated as tampered and either overridden to 'dropout' (the safe
# / high-recall class) or rejected.
#
# This is the XGBoost twin of dnd_train_rf.py. The FGSM gradient source defaults to the BASELINE MLP (DEFAULT_EVAL_MLP_MODEL = None);
# pass --eval-mlp-model <defended checkpoint> to attack with the adversarially
# trained MLP instead (the surrogate used by the adversarial-training script). The XGBoost pkl is a full
# imblearn Pipeline: it takes the RAW dataframe and does its own preprocessing
# (median impute + one-hot) internally, so the detector stays in the raw feature
# space (XGBoost has no penultimate layer to work in).
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_XGB_RESULTS_DIR = "xgb_results_zenodo"
# Source (baseline) MLP: defines the authoritative feature list and provides the
# preprocessor (the standardized space the L-inf epsilon lives in). Same role as
# in adv_train_xgboost.py.
DEFAULT_MLP_DIR = "mlp_results_zenodo"
# EVAL surrogate: the BASELINE MLP by default (None). This is the gradient source for the
# evaluation FGSM. The MLP adversarial training does not save its own preprocessor,
# so the eval attack reuses the baseline MLP preprocessor
# (DEFAULT_EVAL_MLP_PREPROCESSOR = None -> reuse --mlp-dir preprocessor). Pass
# --eval-mlp-model <defended checkpoint> to use the MLP-defended surrogate instead.
DEFAULT_EVAL_MLP_MODEL = None
DEFAULT_EVAL_MLP_PREPROCESSOR = None
DEFAULT_OUTPUT_DIR = "xgb_detect_and_default_zenodo_results"

TARGET_COL = "target"
ID_COL = "student_id"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# epsilon is an L-inf bound in the MLP's PREPROCESSED space, same as
# fgsm_xgboost.py / adv_train_xgboost.py. The final FGSM attack used 1.0.
EVAL_EPSILON_LIST = [1.0]

# detector: false-positive rate targeted when calibrating the detection threshold
# on the clean test split. Xu et al. calibrate the threshold so a small fraction of
# clean inputs are flagged, then measure how many adversarial inputs exceed it.
DEFAULT_DETECTION_TARGET_FPR = 0.05

# fallback network shape only; checkpoint values override these when present
DEFAULT_HIDDEN_DIMS = (128, 64, 32)

# ============================================================
# XGBOOST OPERATING THRESHOLD
# ============================================================
# The XGBoost decision cutoff is NOT re-tuned here. It is read straight from the
# baseline's zenodo_xgb_summary.json (the threshold xgboost_model.py already
# selected with its precision-floor rule). This mirrors oulad_dnd_train_xgb.py so
# the two datasets use IDENTICAL threshold logic in the detect-and-default stage.

# ============================================================
# FEATURE DEFINITIONS (identical set to adv_train_xgboost.py / fgsm_xgboost.py, zenodo.csv)
# ============================================================
# student_id and target are never predictors. academic_year is a one-hot
# categorical. The 8 *_flag columns are binary flags. Everything else is numeric.
NON_FEATURE_COLS = [ID_COL, TARGET_COL]
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
# ATTACKABLE FEATURES (identical to adv_train_xgboost.py, 21 features)
# ============================================================
# Everything else stays frozen: student_id, target, academic_year (one-hot), the
# enrolment/degree descriptors and all 8 binary flags.
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
# DERIVED IDENTITIES (recomputed after perturbation for internal consistency)
# ============================================================
# total_credits_passed_academic_year, total_credits_enrolled_academic_year and
# total_performance are all attackable, so FGSM can move them independently into
# combinations impossible in the real data. When enforce_consistency is on
# (default), the following identities are re-imposed on the adversarial rows
# (verified to hold for 100% of clean rows). This is TOGGLEABLE via
# --no-enforce-derived-consistency, matching adv_train_xgboost.py:
#   1) passed <= enrolled (passed may EQUAL enrolled, never exceed)
#   2) total_performance = (passed / enrolled) * 100, set to 0 when enrolled == 0
CREDITS_ENROLLED_COL = "total_credits_enrolled_academic_year"
CREDITS_PASSED_COL = "total_credits_passed_academic_year"
TOTAL_PERFORMANCE_COL = "total_performance"

# ============================================================
# DOMAIN CONSTRAINTS (identical to adv_train_xgboost.py / fgsm_xgboost.py)
# ============================================================
# Integer-valued counts -> whole numbers after the attack. lms_total_minutes_* are
# float in the raw data but treated as integer minutes here (same as the XGBoost
# attack scripts).
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

# Credit features -> snap to the nearest .0 or .5, whichever is closer.
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

# Percentage features on a 0-100 scale -> clip to [0, 100]. courses_mean is a grade
# on a ~0-10 scale, so it is decimal but NOT a percentage and is not here.
PERCENTAGE_FEATURES = [
    "pass_ratio_sem_a",
    "pass_ratio_sem_b",
    "total_performance",
    "completion_rate_one_year_before",
    "completion_rate_two_years_before",
    "completion_rate_three_years_before",
]

# Non-negative features (every attackable feature). The train min/max clip is the
# primary realism bound; this is an extra floor at 0.
NONNEGATIVE_FEATURES = sorted(set(ATTACKABLE_FEATURES))

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

def round_to_nearest_half(x):
    return np.round(np.asarray(x, dtype=float) * 2.0) / 2.0

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
    if ID_COL in df.columns:
        student_ids = df[ID_COL].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    return X, y, student_ids

def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # SAME split logic and random_state as the XGBoost / MLP training scripts.
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

def resolve_xgb_model_path(xgb_results_dir):
    return resolve_first_existing([os.path.join(xgb_results_dir, "zenodo_xgb_model.pkl")], "XGBoost model (zenodo_xgb_model.pkl)")

def resolve_xgb_summary_path(xgb_results_dir):
    path = os.path.join(xgb_results_dir, "zenodo_xgb_summary.json")
    return path if os.path.exists(path) else None

def load_xgb_threshold(summary_path, fallback=0.5):
    # Read the baseline XGBoost's tuned decision threshold straight from the
    # summary. This is the operating threshold used by the defence; it is NOT
    # re-tuned here. Same logic as oulad_dnd_train_xgb.py.
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
    return resolve_first_existing([os.path.join(mlp_dir, "zenodo_mlp_model.pt")], "MLP model (zenodo_mlp_model.pt)")

def resolve_mlp_preprocessor_path(mlp_dir):
    return resolve_first_existing([os.path.join(mlp_dir, "zenodo_mlp_preprocessor.pkl")], "MLP preprocessor (zenodo_mlp_preprocessor.pkl)")

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
            "input_dim": obj.get("input_dim", None),
        }
    else:
        info = {
            "hidden_dims": DEFAULT_HIDDEN_DIMS,
            "dropout": 0.30,
            "state_dict": obj,
            "selected_features": None,
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
    # The MLP ColumnTransformer emits columns in the order: numeric block -> flag
    # block -> categorical one-hot block. Every attackable zenodo feature is
    # numeric, so only numeric columns whose raw name is attackable may move; the
    # flag and one-hot blocks stay frozen (mask starts all-zeros).
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(f"Attackable features not in selected_features: {missing_attackable}")
    attackable_not_numeric = sorted([f for f in attackable_raw if f not in numeric_cols])
    if attackable_not_numeric:
        raise ValueError(
            "These attackable features are not in the numeric preprocessing block "
            f"(routed to flags/one-hot): {attackable_not_numeric}")
    mask = np.zeros(input_dim, dtype=np.float32)
    for idx, col in enumerate(numeric_cols):
        mask[idx] = 1.0 if col in attackable_raw else 0.0
    if int(mask.sum()) == 0:
        raise ValueError("No preprocessed feature is attackable. Check ATTACKABLE_FEATURES.")
    return mask, numeric_cols

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
    # Applied after the components have been clipped and rounded, so the enforced
    # values track the final perturbed inputs:
    #   1) passed capped at enrolled (may equal, never exceed), kept on the .5 grid
    #   2) total_performance = passed / enrolled * 100, set to 0 when enrolled == 0
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

def preprocessed_adv_to_raw(
    X_adv_preprocessed, X_clean_raw, selected_features, preprocessor,
    train_min, train_max, attacked_row_mask=None, enforce_consistency=True,
):
    # Convert adversarial preprocessed data back to raw space. Only the numeric
    # block is inverse-scaled (it is the first block); frozen features and
    # non-attacked rows are restored from the clean data so the attack scope stays
    # strict. The XGBoost pipeline consumes the raw dataframe (its pkl pipeline does
    # its own preprocessing), so only the raw frame is returned. Logic mirrors
    # adv_train_xgboost.py exactly.
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
    # restore non-attackable raw features
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    # per-feature domain constraints (clip + round) on the attacked columns
    X_adv_raw = apply_domain_constraints(X_adv_raw, numeric_cols, train_min, train_max)
    # optionally re-impose the credit identities on the finalized components
    if enforce_consistency:
        X_adv_raw = keep_integer_counts_in_range(X_adv_raw, train_min, train_max)
        X_adv_raw = enforce_semester_total(X_adv_raw, clean_reset, train_min, train_max)
        X_adv_raw = enforce_credit_identities(X_adv_raw)
        assert_feasible(upv_feasibility_violations(X_adv_raw), attacked_row_mask, 'UPV projection')
    # restore frozen features again after clipping/rounding
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
    # restore non-attacked rows completely
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
    # everything back to raw space with the domain constraints applied. The gradient
    # comes from art_mlp (the eval surrogate: baseline MLP by default).
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
    # Mahalanobis distance to the training mean under the training covariance,
    # computed over the RAW attackable numeric features. XGBoost has no penultimate
    # layer, so like the RF detector this stays in raw feature space. Catches
    # unusual COMBINATIONS of features even when each feature alone looks plausible.
    def __init__(self, attackable_features, ridge=1e-6):
        self.attackable_features = list(attackable_features)
        self.ridge = float(ridge)
        self.cols = None
        self.mu = None
        self.cov_inv = None

    def fit(self, X_train, numeric_cols):
        # only validate the attackable numeric features (the ones the attack moves)
        self.cols = [c for c in numeric_cols if c in set(self.attackable_features)]
        M = X_train[self.cols].to_numpy(dtype=float)
        self.mu = np.nanmean(M, axis=0)
        cov = np.cov(M, rowvar=False)
        cov = cov + self.ridge * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.pinv(cov)
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
def evaluate_xgb_plain(model, X_raw, y_true, threshold):
    # the XGBoost pkl is a full pipeline: it takes the RAW dataframe and does its
    # own preprocessing internally.
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
        "model": "XGBoost",
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
    # One row per student for a single (split, eval_epsilon). Row order is
    # positional: X_clean_raw, X_adv_raw, y_true, student_ids and attack_idx all
    # come from the same split in the same order, so index i is the same student in
    # all of them.
    #
    # The detect-and-default XGBoost is the UNCHANGED baseline XGBoost; the defence
    # only adds the Mahalanobis flag + an override, so a hidden dropout is recovered
    # only when the detector flags its adversarial row (flagged_adv).
    y_true = np.asarray(y_true).astype(int)
    ids = np.asarray(student_ids)
    clean_proba = model.predict_proba(X_clean_raw[selected_features])[:, 1]
    clean_pred = (clean_proba >= threshold).astype(int)
    adv_proba = model.predict_proba(X_adv_raw[selected_features])[:, 1]
    adv_pred = (adv_proba >= threshold).astype(int)
    score_clean = validator.mahalanobis_scores(X_clean_raw[selected_features])
    score_adv = validator.mahalanobis_scores(X_adv_raw[selected_features])
    flagged_clean = score_clean > maha_threshold
    flagged_adv = score_adv > maha_threshold
    trace = pd.DataFrame({
        ID_COL: ids,
        "split": split,
        "eval_epsilon": float(eval_epsilon),
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        "in_attack_scope": np.asarray(attack_idx).astype(bool),
        "xgb_clean_pred_label": clean_pred,
        "xgb_clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        "xgb_clean_prob_dropout": clean_proba,
        "xgb_adv_pred_label": adv_pred,
        "xgb_adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        "xgb_adv_prob_dropout": adv_proba,
        "maha_threshold": float(maha_threshold),
        "maha_score_clean": score_clean,
        "maha_score_adv": score_adv,
        "flagged_clean": flagged_clean,
        "flagged_adv": flagged_adv,
    })
    for action in actions:
        if action == "default_dropout":
            final = adv_pred.copy()
            final[flagged_adv] = POSITIVE_LABEL
            trace["dnd_default_dropout_pred_label"] = final
            trace["dnd_default_dropout_pred_target"] = np.where(final == 1, "dropout", "non-dropout")
        elif action == "abstain":
            trace["dnd_abstain_decision"] = np.where(
                flagged_adv, "abstained",
                np.where(adv_pred == 1, "dropout", "non-dropout"))
        elif action == "none":
            trace["dnd_none_pred_label"] = adv_pred
        else:
            raise ValueError(f"unknown action for trace: {action}")
    hidden = (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL) & (adv_pred == NEGATIVE_LABEL)
    trace["successfully_attacked"] = hidden
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
    df, csv_path, output_dir, xgb_results_dir, mlp_dir,
    eval_epsilon_list, random_state, attack_mode, eval_attack_scope,
    detection_target_fpr, actions, eval_mlp_model_path_arg, eval_mlp_preprocessor_arg,
    enforce_consistency,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("XGB DETECT-AND-DEFAULT (MAHALANOBIS RAW-SPACE FLAG -> OVERRIDE, zenodo)")
    print("=" * 80)
    print(f"CSV path            : {csv_path}")
    print(f"Output dir          : {output_dir}")
    print(f"Attack mode         : {attack_mode}")
    print(f"Eval attack scope   : {eval_attack_scope}")
    print(f"Enforce derived     : {enforce_consistency}")
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

    # ---- load baseline XGBoost + its tuned threshold from zenodo_xgb_summary.json ----
    xgb_baseline = joblib.load(resolve_xgb_model_path(xgb_results_dir))
    xgb_threshold, xgb_threshold_source = load_xgb_threshold(resolve_xgb_summary_path(xgb_results_dir))
    if xgb_threshold_source == "fallback_0.5":
        print("WARNING: baseline XGBoost threshold not found in summary; using fallback 0.5. "
              "Results will NOT be comparable to the tuned baseline. "
              "Check --xgb-results-dir and the zenodo_xgb_summary.json file.")

    # ---- baseline MLP preprocessor + baseline surrogate ----
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
    if eval_mlp_model_path_arg:
        if not os.path.exists(eval_mlp_model_path_arg):
            raise FileNotFoundError(
                f"Eval surrogate checkpoint not found: {eval_mlp_model_path_arg}. "
                "Run the MLP adversarial training first (it produces "
                "zenodo_mlp_defended_model_train_eps_1p0.pt), or pass --eval-mlp-model \"\" "
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

    # ---- operating threshold: read straight from zenodo_xgb_summary.json ----
    # NOT re-tuned. Both detect-and-default scripts (zenodo and oulad) read the
    # baseline's saved threshold from the summary, so the two datasets use identical
    # threshold logic in this stage.
    print(f"Baseline XGBoost threshold (from summary): {xgb_threshold} | source: {xgb_threshold_source}")
    print()

    # baseline XGBoost clean predictions define the true_positive_only attack scope
    base_test_pred, _ = evaluate_xgb_plain(xgb_baseline, X_test, y_test.values, xgb_threshold)
    base_val_pred, _ = evaluate_xgb_plain(xgb_baseline, X_val, y_val.values, xgb_threshold)

    # ---- fit the Mahalanobis validator (RAW space) + calibrate on CLEAN TEST ----
    validator = InputValidator(ATTACKABLE_FEATURES).fit(X_train, numeric_cols)
    clean_test_scores = validator.mahalanobis_scores(X_test[selected_features])
    maha_threshold = float(np.quantile(clean_test_scores, 1.0 - detection_target_fpr))
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
        res, _ = evaluate_with_defence(xgb_baseline, raw_clean[split_name], y_split.values, xgb_threshold,
                                       validator, maha_threshold, "none", "Undefended Clean", split_name, 0.0)
        all_results.append(res)
    for eval_epsilon in eval_epsilon_list:
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
            res, _ = evaluate_with_defence(xgb_baseline, X_adv_raw, y_split.values, xgb_threshold,
                                           validator, maha_threshold, "none",
                                           f"Undefended FGSM {attack_mode}", split_name, eval_epsilon)
            res["attacked_rows_count"] = int(attack_idx.sum())
            all_results.append(res)

    # ---- defended: detect-and-default, per action ----
    for action in actions:
        for split_name in ["test", "validation"]:  # clean input (cost of the defence)
            y_split = y_test if split_name == "test" else y_val
            res, cm = evaluate_with_defence(xgb_baseline, raw_clean[split_name], y_split.values, xgb_threshold,
                                            validator, maha_threshold, action,
                                            f"Defended Clean ({action})", split_name, 0.0)
            all_results.append(res)
            if cm is not None:
                save_confusion_matrix(cm, os.path.join(output_dir, f"zenodo_xgb_defended_clean_{action}_{split_name}_cm.csv"))
        for eval_epsilon in eval_epsilon_list:  # adversarial input (the robust number)
            for split_name in ["test", "validation"]:
                y_split = y_test if split_name == "test" else y_val
                X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
                res, cm = evaluate_with_defence(xgb_baseline, X_adv_raw, y_split.values, xgb_threshold,
                                                validator, maha_threshold, action,
                                                f"Defended FGSM {attack_mode} ({action})", split_name, eval_epsilon)
                res["attacked_rows_count"] = int(attack_idx.sum())
                all_results.append(res)
                if cm is not None:
                    save_confusion_matrix(cm, os.path.join(
                        output_dir, f"zenodo_xgb_defended_fgsm_{action}_{split_name}_cm_eval_eps_{str(eval_epsilon).replace('.', 'p')}.csv"))

    # ---- per-student trace (one row per student, per split + eval epsilon) ----
    student_trace_files = []
    for eval_epsilon in eval_epsilon_list:
        eps_tag = str(eval_epsilon).replace(".", "p")
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            id_split = id_test if split_name == "test" else id_val
            X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
            trace_path = os.path.join(
                output_dir, f"zenodo_xgb_dnd_{split_name}_student_trace_eval_eps_{eps_tag}.csv")
            save_student_trace(
                model=xgb_baseline, X_clean_raw=raw_clean[split_name], X_adv_raw=X_adv_raw,
                y_true=y_split.values, student_ids=id_split.values, threshold=xgb_threshold,
                validator=validator, maha_threshold=maha_threshold, actions=actions,
                attack_idx=attack_idx, eval_epsilon=eval_epsilon, split=split_name,
                selected_features=selected_features, path=trace_path,
            )
            student_trace_files.append(trace_path)
            print(f"Saved student trace: {trace_path}")

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "zenodo_xgb_detect_and_default_results.csv")
    results_df.to_csv(results_path, index=False)

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "defence": "XGBoost detect-and-default (Mahalanobis flag -> override to dropout / abstain)",
        "detector": "Mahalanobis distance to the training mean under the training covariance, over the RAW attackable numeric features (XGBoost has no penultimate layer).",
        "detector_space": "raw_feature_space",
        "override_rule": "A flagged row's prediction is set to dropout (default_dropout), or the row is rejected (abstain).",
        "target_model": "Baseline XGBoost (undefended, unchanged)",
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
        "actions": actions,
        "xgb_threshold": xgb_threshold,
        "xgb_threshold_source": xgb_threshold_source,
        "xgb_threshold_rule": ("Loaded directly from zenodo_xgb_summary.json (the baseline "
                               "XGBoost's saved threshold), not re-tuned. Same threshold logic "
                               "as oulad_dnd_train_xgb.py so the two datasets are comparable."),
        "mahalanobis_threshold": maha_threshold,
        "detection_target_fpr": detection_target_fpr,
        "feature_list_source": feature_source,
        "n_raw_features": len(selected_features),
        "n_attackable_raw_features": len(ATTACKABLE_FEATURES),
        "attackable_raw_features": list(ATTACKABLE_FEATURES),
        "frozen_raw_features": list(FROZEN_FEATURES_HINT),
        "derived_identities_note": (
            "When enforce_derived_consistency is on (default, matching adv_train_xgboost.py): "
            "total_credits_passed_academic_year capped at total_credits_enrolled_academic_year, "
            "and total_performance = passed / enrolled * 100 (0 when enrolled == 0). "
            "Both hold for 100% of the clean rows."),
        "constraints": {
            "half_credit_features": ("credits_enrolled_semester_a/b, "
                                     "total_credits_enrolled/passed_academic_year snapped to nearest .0/.5."),
            "integer_count_features": "lms_* and attendance_days_* rounded to whole numbers.",
            "decimal_features": "courses_mean plus the percentage features rounded to 2 dp.",
            "percentage_features": ("pass_ratio_sem_a/b, total_performance and the three "
                                    "completion_rate_*_before features clipped to [0, 100]."),
            "nonnegative_features": "Every attackable feature floored at 0.",
            "raw_value_clip": "Numeric features clipped to train min/max when available.",
        },
        "comparison_note": ("Both XGBoost defences (adversarial training in adv_train_xgboost.py and "
                            "detect-and-default here) are evaluated with FGSM transferred from an MLP surrogate. "
                            "Adversarial training is evaluated with the adversarially trained MLP as the surrogate; "
                            f"the surrogate used in this detect-and-default run is: {eval_attack_source}."),
        "note_abstain": "For action=abstain, accuracy/recall are computed only on ACCEPTED (non-flagged) rows; read them together with flag_rate.",
        "caveat_adaptive": "Detection assumes a non-adaptive attacker. An attacker aware of the Mahalanobis filter could constrain perturbations to stay near the training distribution and reduce detection.",
        "split": split_report,
        "results_file": results_path,
        "student_trace_files": student_trace_files,
        "student_trace_note": ("One row per student per (split, eval_epsilon). Columns: undefended "
                               "baseline XGBoost clean/adv predictions, Mahalanobis flag on clean and "
                               "adv inputs, and per-action defended decision under attack. "
                               "successfully_attacked = real dropout caught clean but hidden under "
                               "attack; recovered_default_dropout / caught_abstain = that hidden row "
                               "was flagged by the detector."),
    }
    with open(os.path.join(output_dir, "zenodo_xgb_detect_and_default_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("XGB DETECT-AND-DEFAULT DONE")
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
        description="Detect-and-default defence for XGBoost (Mahalanobis flag -> override) on zenodo, "
                    "attacked with FGSM transferred from an MLP surrogate (baseline MLP by default).")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--xgb-results-dir", default=DEFAULT_XGB_RESULTS_DIR)
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
    parser.add_argument("--no-enforce-derived-consistency", dest="enforce_consistency",
                        action="store_false",
                        help="Disable recomputing the credit identities after the attack.")
    parser.set_defaults(enforce_consistency=True)
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
        xgb_results_dir=args.xgb_results_dir, mlp_dir=args.mlp_dir,
        eval_epsilon_list=parse_float_list(args.eval_epsilons), random_state=args.random_state,
        attack_mode=args.attack_mode, eval_attack_scope=args.eval_attack_scope,
        detection_target_fpr=args.detection_target_fpr, actions=actions,
        eval_mlp_model_path_arg=eval_mlp_model_arg,
        eval_mlp_preprocessor_arg=args.eval_mlp_preprocessor,
        enforce_consistency=args.enforce_consistency,
    )

if __name__ == "__main__":
    main()