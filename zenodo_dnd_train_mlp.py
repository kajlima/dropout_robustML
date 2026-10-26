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
# Detect-and-default defence for the MLP on zenodo.csv, RAW INPUT-GLOBAL variant.
# The Mahalanobis validator is fit in the shared RAW attackable-feature space
# (distance to the training mean under the training covariance, input-global and
# NOT class-conditional). This is the identical detector used for RF and XGBoost,
# so the detect-and-default detector is held constant across all three classifiers
# and the cross-model robustness comparison is controlled (only the classifier
# differs). A flagged row's prediction is OVERRIDDEN to 'dropout' (the safe /
# high-recall class) or the row is rejected. White-box PGD setup matches the
# standalone attack in pgd_mlp.py (zenodo) so the defence is tested against the
# same threat reported in the attack chapter: same attackable/frozen feature sets,
# the always-on credit identities, and the .0/.5 credit snapping.
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_MLP_RESULTS_DIR = "mlp_results_zenodo"
DEFAULT_OUTPUT_DIR = "mlp_dnd_feature_classcond_zenodo_results"
DEFAULT_DETECTOR_COVARIANCE = "tied"  # unused; kept only for CLI compatibility (detector is raw input-global)

TARGET_COL = "target"
ID_COL = "student_id"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# epsilon is an L-inf bound in the MLP PREPROCESSED (standardized) space, same as
# pgd_mlp.py / adv_train_mlp.py (zenodo).
EVAL_EPSILON_LIST = [1.0]
PGD_STEP = 0.01
# Match the standalone PGD attack: pgd_mlp.py (zenodo) uses 100 iterations, so the
# DnD defence is evaluated against the full-strength threat reported in the attack
# chapter. Override with --pgd-iterations.
PGD_ITERATIONS = 100

# detector: false-positive rate targeted when calibrating on the clean test split.
DEFAULT_DETECTION_TARGET_FPR = 0.05
DEFAULT_HIDDEN_DIMS = (128, 64, 32)

# ============================================================
# THRESHOLD (inherited from the baseline MLP)
# ============================================================
# Detect-and-default does NOT retrain the classifier, so it does not select its
# own decision cutoff: it loads the operating threshold straight from the baseline
# MLP checkpoint (see load_mlp_checkpoint) and wraps the unchanged baseline model
# at that point. MIN_PRECISION is kept here for documentation only: it records the
# precision floor the baseline used when it originally picked that threshold
# (max recall subject to dropout precision >= MIN_PRECISION, else F1-optimal),
# and it is 0.73 to match mlp_model.py / adv_train_mlp.py across both datasets.
MIN_PRECISION = 0.73

# ============================================================
# FEATURE DEFINITIONS (zenodo.csv)
# ============================================================
# student_id and target are never predictors. The authoritative feature list is
# read from the MLP checkpoint at runtime; this fallback matches the training
# scripts (every column except id and target). academic_year is one-hot, the 8
# *_flag columns are binary flags, everything else is numeric.
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
# ATTACKABLE FEATURES (identical to pgd_mlp.py / adv_train_mlp.py, 21 features)
# ============================================================
# Only these raw features may move. Everything else stays frozen: student_id,
# target, academic_year (one-hot), the enrolment/degree descriptors (n_courses,
# degree_size, seniority, highest_course_year_enrolled) and all 8 binary flags.
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

# Explicitly frozen (for reporting / sanity check only). student_id and target are
# not model features so they are not listed here.
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
# total_credits_passed_academic_year, total_credits_enrolled_academic_year and
# total_performance are all attackable, so PGD can move them independently into
# combinations impossible in the real data. After every attack we force these two
# identities back onto the adversarial rows (verified to hold for 100% of clean
# rows). There is no on/off switch - this always runs, matching pgd_mlp.py, so the
# eval PGD examples are constructed identically to the reported attack:
#   1) passed <= enrolled (passed may EQUAL enrolled, never exceed)
#   2) total_performance = (passed / enrolled) * 100, set to 0 when enrolled == 0
#      (the 2 enrolled==0 rows in the raw data have passed=0, performance=0)
CREDITS_ENROLLED_COL = "total_credits_enrolled_academic_year"
CREDITS_PASSED_COL = "total_credits_passed_academic_year"
TOTAL_PERFORMANCE_COL = "total_performance"

# ============================================================
# DOMAIN CONSTRAINTS (calibrated to zenodo.csv)
# ============================================================
# Integer-valued counts -> round back to whole numbers after the attack.
# lms_total_minutes_* are float in the raw data but treated as integer minutes here
# (same choice as pgd_mlp.py).
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

# Credit features -> snap to the nearest .0 or .5, whichever is closer (12.2 ->
# 12.0, 12.3 -> 12.5, 12.8 -> 13.0). See round_to_nearest_half.
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

def round_to_nearest_half(x):
    # snap each value to the nearest .0 or .5.
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
    return resolve_first_existing([os.path.join(mlp_results_dir, "zenodo_mlp_model.pt")], "MLP model (zenodo_mlp_model.pt)")

def resolve_mlp_preprocessor_path(mlp_results_dir):
    return resolve_first_existing([os.path.join(mlp_results_dir, "zenodo_mlp_preprocessor.pkl")], "MLP preprocessor (zenodo_mlp_preprocessor.pkl)")

def resolve_mlp_summary_path(mlp_results_dir):
    path = os.path.join(mlp_results_dir, "zenodo_mlp_summary.json")
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
        state_dict = obj["model_state_dict"]
    else:
        input_dim = input_dim_from_data
        hidden_dims = DEFAULT_HIDDEN_DIMS
        dropout = 0.30
        checkpoint_threshold = None
        selected_features = None
        state_dict = obj
    if input_dim != input_dim_from_data:
        raise ValueError(
            f"Model input dim ({input_dim}) differs from data ({input_dim_from_data}). "
            "Preprocessor/checkpoint mismatch.")
    # operating threshold, resolved once from the baseline artefacts and used as-is
    # (checkpoint -> summary -> 0.5 fallback). DnD does not retrain the model, so it
    # keeps the baseline's saved cutoff rather than re-selecting one here.
    if checkpoint_threshold is not None:
        ref_threshold = float(checkpoint_threshold)
        ref_source = "zenodo_mlp_model.pt"
    elif summary_threshold is not None:
        ref_threshold = float(summary_threshold)
        ref_source = "zenodo_mlp_summary.json"
    else:
        ref_threshold = 0.5
        ref_source = "fallback_0.5"
    model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "ref_threshold": ref_threshold,
        "ref_threshold_source": ref_source,
        "selected_features": selected_features,
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
    # The MLP ColumnTransformer emits columns in the order: numeric block -> flag
    # block -> categorical one-hot block. Every attackable zenodo feature is
    # numeric, so only numeric columns whose raw name is attackable may move; the
    # flag and one-hot blocks stay frozen (mask starts all-zeros).
    os.makedirs(output_dir, exist_ok=True)
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
    # clip to train min/max, floor at 0, clip percentages to [0, 100], then round to
    # the correct grid for each feature. Only attackable columns are touched here;
    # frozen columns are restored by the caller.
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

def preprocessed_adv_to_raw_and_scaled(
    X_adv_preprocessed, X_clean_preprocessed, X_clean_raw, selected_features,
    preprocessor, train_min, train_max, perturbable_mask_preprocessed,
    attacked_row_mask=None,
):
    # Convert adversarial preprocessed data back to raw space. Only the numeric
    # block is inverse-scaled (it is the first block); frozen features and
    # non-attacked rows are restored from the clean data so the attack scope stays
    # strict. The credit identities are always re-imposed afterwards.
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
    # per-feature domain constraints (clip + round) on the attacked columns
    X_adv_raw = apply_domain_constraints(X_adv_raw, numeric_cols, train_min, train_max)
    # always enforce the credit identities on the finalized components
    X_adv_raw = keep_integer_counts_in_range(X_adv_raw, train_min, train_max)
    X_adv_raw = enforce_semester_total(X_adv_raw, clean_reset, train_min, train_max)
    X_adv_raw = enforce_credit_identities(X_adv_raw)
    assert_feasible(upv_feasibility_violations(X_adv_raw), attacked_row_mask, 'UPV projection')
    # restore frozen features again after clipping/rounding (belt and suspenders)
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values
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
    # Mahalanobis distance to the training mean under the training covariance,
    # computed over the RAW attackable numeric features. This is the SAME detector
    # used for RF and XGBoost (see dnd_train_rf / dnd_train_xgboost): input-global,
    # not class-conditional, and operating in the shared raw feature space rather
    # than any model-specific representation. It is used here so the detect-and-
    # default detector is identical across MLP, RF and XGBoost, making the
    # cross-model robustness comparison a controlled one (only the classifier
    # differs). Catches unusual COMBINATIONS of features even when each feature
    # alone looks plausible. mahalanobis_scores(X) takes a RAW dataframe.
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
    # One row per student for a single (split, eval_epsilon). Row order is
    # positional: X_clean_raw, X_adv_raw, y_true, student_ids and attack_idx all
    # come from the same split in the same order, so index i is the same student in
    # all of them.
    #
    # The detect-and-default MLP is the UNCHANGED baseline MLP; the defence only
    # adds the Mahalanobis flag + an override, so a hidden dropout is recovered only
    # when the detector flags its adversarial row (flagged_adv).
    y_true = np.asarray(y_true).astype(int)
    ids = np.asarray(student_ids)
    clean_pre = transform_to_float32(preprocessor, X_clean_raw[selected_features])
    adv_pre = transform_to_float32(preprocessor, X_adv_raw[selected_features])
    clean_proba = predict_mlp_proba(model, clean_pre, device)
    adv_proba = predict_mlp_proba(model, adv_pre, device)
    clean_pred = (clean_proba >= threshold).astype(int)
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
        "mlp_clean_pred_label": clean_pred,
        "mlp_clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        "mlp_clean_prob_dropout": clean_proba,
        "mlp_adv_pred_label": adv_pred,
        "mlp_adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        "mlp_adv_prob_dropout": adv_proba,
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
    # successfully_attacked = real dropout, caught by the clean baseline MLP, hidden
    # once attacked.
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
    df, csv_path, output_dir, mlp_results_dir, eval_epsilon_list,
    pgd_step, pgd_iterations, random_state, attack_mode, eval_attack_scope,
    detection_target_fpr, actions,
    detector_covariance=DEFAULT_DETECTOR_COVARIANCE,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("MLP DETECT-AND-DEFAULT (MAHALANOBIS FLAG -> OVERRIDE, zenodo)")
    print("=" * 80)
    print(f"CSV path            : {csv_path}")
    print(f"Output dir          : {output_dir}")
    print(f"Attack mode         : {attack_mode}")
    print(f"Eval attack scope   : {eval_attack_scope}")
    print(f"PGD step / iters    : {pgd_step} / {pgd_iterations}")
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
    art_mlp = wrap_mlp_for_art(mlp_model, input_dim=input_dim)
    perturbable_mask, numeric_cols = make_preprocessed_attack_mask(preprocessor, selected_features, input_dim, output_dir)

    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)

    # ---- operating threshold: loaded directly from the baseline MLP checkpoint ----
    # Detect-and-default does NOT retrain the classifier, so it inherits the exact
    # operating threshold the baseline saved (precision-floor rule with
    # MIN_PRECISION on the same clean test split). This is the same mechanism as
    # oulad_dnd_train_mlp.py; the DnD wraps the unchanged baseline model at its own
    # operating point instead of re-deriving the cutoff here.
    mlp_threshold = mlp_ckpt["ref_threshold"]
    print(f"Baseline MLP threshold: {mlp_threshold} | source: {mlp_ckpt['ref_threshold_source']}")

    base_test_pred = (predict_mlp_proba(mlp_model, X_test_pre, device) >= mlp_threshold).astype(int)
    base_val_pred = (predict_mlp_proba(mlp_model, X_val_pre, device) >= mlp_threshold).astype(int)

    validator = InputValidator(ATTACKABLE_FEATURES).fit(X_train, numeric_cols)
    detector_desc = ("Mahalanobis distance to the training mean under the training covariance, over "
                     "the RAW attackable numeric features (input-global, not class-conditional). "
                     "Identical to the RF/XGBoost detect-and-default detector so the cross-model "
                     "comparison is controlled (only the classifier differs).")
    print(f"Detector            : raw_global ({detector_desc})")

    clean_test_scores = validator.mahalanobis_scores(X_test[selected_features])
    maha_threshold = float(np.quantile(clean_test_scores, 1.0 - detection_target_fpr))
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
                save_confusion_matrix(cm, os.path.join(output_dir, f"zenodo_mlp_defended_clean_{action}_{split_name}_cm.csv"))
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
                        output_dir, f"zenodo_mlp_defended_pgd_{action}_{split_name}_cm_eval_eps_{str(eval_epsilon).replace('.', 'p')}.csv"))

    # ---- per-student trace (one row per student, per split + eval epsilon) ----
    student_trace_files = []
    for eval_epsilon in eval_epsilon_list:
        eps_tag = str(eval_epsilon).replace(".", "p")
        for split_name in ["test", "validation"]:
            y_split = y_test if split_name == "test" else y_val
            id_split = id_test if split_name == "test" else id_val
            X_adv_raw, attack_idx = adv_by_eps[eval_epsilon][split_name]
            trace_path = os.path.join(
                output_dir, f"zenodo_mlp_dnd_{split_name}_student_trace_eval_eps_{eps_tag}.csv")
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
    results_path = os.path.join(output_dir, "zenodo_mlp_detect_and_default_results.csv")
    results_df.to_csv(results_path, index=False)

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "defence": "MLP detect-and-default (Mahalanobis flag -> override to dropout)",
        "detector": detector_desc,
        "detector_type": "raw_global",
        "detector_space": "raw_feature_space",
        "detector_covariance": "Empirical covariance of the raw attackable features, ridge 1e-6, pseudo-inverse (input-global; not class-conditional)",
        "override_rule": "A flagged row's prediction is set to dropout (default_dropout), or the row is rejected (abstain).",
        "target_model": "Baseline MLP (undefended, unchanged)",
        "attack_used_for_eval": "PGD (white-box)",
        "attack_mode": attack_mode,
        "eval_attack_scope": eval_attack_scope,
        "pgd_step": pgd_step,
        "pgd_iterations": pgd_iterations,
        "eval_epsilon_list": eval_epsilon_list,
        "epsilon_space": "L-inf bound in the MLP preprocessed/standardized feature space",
        "actions": actions,
        "mlp_threshold": mlp_threshold,
        "mlp_threshold_source": mlp_ckpt["ref_threshold_source"],
        "mlp_threshold_rule": (f"Inherited from the baseline MLP checkpoint (DnD does not "
                               f"retrain). The baseline picked it with the precision-floor rule "
                               f"(max dropout recall subject to dropout precision >= {MIN_PRECISION}, "
                               f"else F1-optimal). Same mechanism as oulad_dnd_train_mlp.py."),
        "min_precision": MIN_PRECISION,
        "mahalanobis_threshold": maha_threshold,
        "detection_target_fpr": detection_target_fpr,
        "mlp_model_path": mlp_model_path,
        "mlp_preprocessor_path": mlp_preprocessor_path,
        "feature_list_source": feature_source,
        "n_raw_features": len(selected_features),
        "n_attackable_raw_features": len(ATTACKABLE_FEATURES),
        "attackable_raw_features": list(ATTACKABLE_FEATURES),
        "frozen_raw_features": list(FROZEN_FEATURES_HINT),
        "derived_identities_note": (
            "Always enforced after the attack (no toggle, matching pgd_mlp.py): "
            "total_credits_passed_academic_year capped at total_credits_enrolled_"
            "academic_year, and total_performance = passed / enrolled * 100 (0 when "
            "enrolled == 0). Both hold for 100% of the clean rows."),
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
        "note_abstain": "For action=abstain, accuracy/recall are computed only on ACCEPTED (non-flagged) rows; read them together with flag_rate.",
        "caveat_adaptive": "Detection assumes a non-adaptive attacker. An attacker aware of the Mahalanobis filter could constrain perturbations to stay near the training distribution and reduce detection.",
        "split": split_report,
        "results_file": results_path,
        "threshold_tuning_file": None,  # no tuning file: DnD inherits the baseline threshold
        "student_trace_files": student_trace_files,
        "student_trace_note": ("One row per student per (split, eval_epsilon). Columns: undefended "
                               "baseline MLP clean/adv predictions, Mahalanobis flag on clean and adv "
                               "inputs, and per-action defended decision under attack. "
                               "successfully_attacked = real dropout caught clean but hidden under "
                               "attack; recovered_default_dropout / caught_abstain = that hidden row "
                               "was flagged by the detector."),
    }
    with open(os.path.join(output_dir, "zenodo_mlp_detect_and_default_summary.json"), "w", encoding="utf-8") as f:
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
                    "Mahalanobis detector (identical to the RF/XGBoost detector) and white-box PGD on zenodo.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--mlp-results-dir", default=DEFAULT_MLP_RESULTS_DIR)
    parser.add_argument("--mlp-dir", dest="mlp_results_dir", help="Alias for --mlp-results-dir.")
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
                             "(matches the standalone pgd_mlp.py attack on zenodo).")
    parser.add_argument("--detection-target-fpr", type=float, default=DEFAULT_DETECTION_TARGET_FPR)
    parser.add_argument("--detector-covariance", default=DEFAULT_DETECTOR_COVARIANCE,
                        choices=["tied", "per_class"],
                        help="Not used: the detector is always the raw input-global Mahalanobis detector. "
                             "Kept only so older command lines still run.")
    parser.add_argument("--actions", default="default_dropout,abstain",
                        help="Comma-separated actions for flagged rows: default_dropout,abstain. Default: default_dropout,abstain")
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
        mlp_results_dir=args.mlp_results_dir,
        eval_epsilon_list=parse_float_list(args.eval_epsilons), pgd_step=args.pgd_step,
        pgd_iterations=args.pgd_iterations, random_state=args.random_state,
        attack_mode=args.attack_mode, eval_attack_scope=args.eval_attack_scope,
        detection_target_fpr=args.detection_target_fpr, actions=actions,
        detector_covariance=args.detector_covariance,
    )

if __name__ == "__main__":
    main()