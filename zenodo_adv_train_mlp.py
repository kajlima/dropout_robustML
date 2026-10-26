import os
import json
import ast
import copy
import random
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
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
# Adversarial training for the zenodo MLP. This is the zenodo counterpart of the
# oulad_early MLP adversarial-training script: same training methodology (train a
# fresh model from scratch on clean + PGD samples, fixed epoch budget chosen by
# CV inside the mixed train set (no early stopping, final weights, test never
# consulted during training), threshold tuned on test, validation held out for
# the final measurement),
# same attack scope/mode options and defaults. Only the dataset-specific pieces
# differ: the feature list, the attackable/frozen sets, the num/flag/one-hot
# preprocessing blocks, the rounding grids, and the derived credit identities.
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_MLP_RESULTS_DIR = "mlp_results_zenodo"
DEFAULT_OUTPUT_DIR = "mlp_adversarial_training_zenodo_results"

TARGET_COL = "target"
ID_COL = "student_id"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

TRAIN_EPSILON_LIST = [1.0]
EVAL_EPSILON_LIST = [1.0]
PGD_STEP = 0.01
PGD_ITERATIONS = 100
THRESHOLDS = np.arange(0.05, 0.96, 0.01)
DEFAULT_HIDDEN_DIMS = (128, 64, 32)

# Minimum dropout precision accepted when picking the defended threshold. Among
# all thresholds that keep dropout precision at or above this floor, take the one
# with the highest recall. Same rule and value as the baseline mlp_model.py.
MIN_PRECISION = 0.73

# NO early stopping for the defended MLP: fixed epoch budget with final weights,
# analogous to the baseline mlp_model.py / zenodo_mlp_model.py. The budget is
# estimated ONCE by a short stratified k-fold CV INSIDE the mixed (clean + PGD
# adversarial) training matrix, so the test/validation splits are never consulted
# during training. This removes the test-peeking that early stopping introduced.
EPOCHS_CAP = 100            # max epochs explored by the CV that picks the budget
CV_N_SPLITS_FOR_EPOCHS = 3  # folds in the single short epoch-budget CV
MIN_FIXED_EPOCHS = 5        # floor, guards against a noisy tiny argmin

# ============================================================
# FEATURE DEFINITIONS (zenodo.csv)
# ============================================================
# student_id and target are never predictors. The authoritative feature list is
# read from the MLP checkpoint at runtime; this fallback matches the training
# scripts (every column except id and target, in CSV column order).
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
# ATTACKABLE FEATURES
# ============================================================
# Only these raw features may move. Everything else stays frozen: academic_year
# (one-hot), the enrolment/degree descriptors (n_courses, degree_size, seniority,
# highest_course_year_enrolled) and all 8 binary flags. Every attackable feature
# is numeric, so only numeric preprocessing columns can be perturbed.
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

# Explicitly frozen (for reporting / sanity checks only). student_id and target
# are not model features, so they are not listed here.
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
# DERIVED IDENTITIES (zenodo)
# ============================================================
# These three columns are attackable, so PGD moves them independently and can
# produce combinations that are impossible in the real data. When consistency is
# enforced (default), the following identities are re-imposed on the adversarial
# rows after clipping/rounding:
#   1) total_credits_passed_academic_year <= total_credits_enrolled_academic_year
#      (passed may EQUAL enrolled, but never exceed it)
#   2) total_performance = (passed / enrolled) * 100, set to 0 when enrolled == 0
# Both identities hold for 100% of the clean rows. This is the zenodo equivalent
# of the CGPA / credit-total identities in the oulad_early script.
CREDITS_ENROLLED_COL = "total_credits_enrolled_academic_year"
CREDITS_PASSED_COL = "total_credits_passed_academic_year"
TOTAL_PERFORMANCE_COL = "total_performance"

# ============================================================
# DOMAIN CONSTRAINTS (calibrated to zenodo.csv)
# ============================================================
# Integer-valued counts -> round back to whole numbers after the attack.
# lms_total_minutes_* are float in the raw data but treated as integers here.
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

# Credit features -> snap to the nearest .0 or .5 (whichever is closer).
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
# grade on a ~0-10 scale, so it is decimal but NOT a percentage.
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

def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def one_hot(labels, n_classes=2):
    labels = np.asarray(labels).astype(int)
    return np.eye(n_classes, dtype=np.float32)[labels]

def round_to_nearest_half(x):
    # snap each value to the nearest .0 or .5
    return np.round(np.asarray(x, dtype=float) * 2.0) / 2.0

def parse_class_weight(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    if isinstance(value, str):
        value_strip = value.strip()
        if value_strip in ["None", "none", "null"]:
            return None
        if value_strip == "balanced":
            return "balanced"
        try:
            parsed = ast.literal_eval(value_strip)
            if isinstance(parsed, dict):
                return {int(k): float(v) for k, v in parsed.items()}
            if parsed is None:
                return None
        except Exception:
            pass
    return value

def make_criterion(class_weight, y_train_used, device):
    class_weight = parse_class_weight(class_weight)
    if class_weight is None:
        return nn.CrossEntropyLoss()
    if class_weight == "balanced":
        counts = np.bincount(y_train_used, minlength=2).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError(f"Invalid class count for balanced weights: {counts}")
        weights = len(y_train_used) / (2.0 * counts)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    if isinstance(class_weight, dict):
        weights = np.array(
            [float(class_weight.get(0, 1.0)), float(class_weight.get(1, 1.0))],
            dtype=np.float32,
        )
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"Unknown class_weight: {class_weight}")

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
            "Make sure --csv-path points to the same data the model was trained on."
        )
    X = df[selected_features].copy()
    y = normalize_target(df[TARGET_COL])
    if ID_COL in df.columns:
        student_ids = df[ID_COL].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    return X, y, student_ids

def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # SAME split logic and random_state as the RF / MLP / XGBoost zenodo scripts,
    # so the test/validation rows here are exactly the rows those models saw.
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
# PATHS AND CHECKPOINTS
# ============================================================
def resolve_first_existing(candidates, what):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{what} not found. Checked:\n" + "\n".join(candidates))

def resolve_mlp_model_path(mlp_results_dir):
    return resolve_first_existing(
        [os.path.join(mlp_results_dir, "zenodo_mlp_model.pt")],
        "MLP model (zenodo_mlp_model.pt)",
    )

def resolve_mlp_preprocessor_path(mlp_results_dir):
    return resolve_first_existing(
        [os.path.join(mlp_results_dir, "zenodo_mlp_preprocessor.pkl")],
        "MLP preprocessor (zenodo_mlp_preprocessor.pkl)",
    )

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
        best_params = obj.get("best_params", {}) or {}
        selected_features = obj.get("selected_features", None)
        winning_strategy = obj.get("winning_strategy", None)
        state_dict = obj["model_state_dict"]
    else:
        input_dim = input_dim_from_data
        hidden_dims = DEFAULT_HIDDEN_DIMS
        dropout = 0.30
        checkpoint_threshold = None
        best_params = {}
        selected_features = None
        winning_strategy = None
        state_dict = obj

    if input_dim != input_dim_from_data:
        raise ValueError(
            f"Model input dim ({input_dim}) differs from data ({input_dim_from_data}). "
            "The preprocessor and checkpoint do not match. Make sure --mlp-results-dir "
            "points to the run that produced this dataset's model."
        )

    if checkpoint_threshold is not None:
        threshold = float(checkpoint_threshold)
        threshold_source = "zenodo_mlp_model.pt"
    elif summary_threshold is not None:
        threshold = float(summary_threshold)
        threshold_source = "zenodo_mlp_summary.json"
    else:
        threshold = 0.5
        threshold_source = "fallback_0.5"

    model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "state_dict": state_dict,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "best_params": best_params,
        "selected_features": selected_features,
        "winning_strategy": winning_strategy,
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
        raise ValueError("'num' transformer not found in preprocessor.")
    return cols

def get_scaler_from_preprocessor(preprocessor):
    num_pipeline = preprocessor.named_transformers_["num"]
    if "scaler" not in num_pipeline.named_steps:
        raise ValueError("Scaler not found in numeric pipeline.")
    return num_pipeline.named_steps["scaler"]

def make_preprocessed_attack_mask(preprocessor, selected_features, input_dim, output_dir):
    # The zenodo ColumnTransformer emits columns in the order:
    #   numeric block -> flag block -> categorical one-hot block.
    # Every attackable feature is numeric, so only numeric columns whose raw name
    # is attackable may move; the flag and one-hot blocks stay frozen (mask
    # starts as all-zeros).
    os.makedirs(output_dir, exist_ok=True)
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)

    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(f"Attackable features not in selected_features:\n{missing_attackable}")
    attackable_not_numeric = sorted([f for f in attackable_raw if f not in numeric_cols])
    if attackable_not_numeric:
        raise ValueError(
            "These attackable features are not in the numeric preprocessing block "
            f"(they were routed to flags/one-hot): {attackable_not_numeric}"
        )
    if len(numeric_cols) > input_dim:
        raise ValueError("Number of numeric columns exceeds preprocessed input_dim.")

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
        raise ValueError("No preprocessed feature is attackable.")
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
    # Applied after clipping/rounding, so the enforced values track the final
    # perturbed inputs:
    #   1) passed is capped at enrolled (passed may equal enrolled, never exceed)
    #   2) total_performance = (passed / enrolled) * 100, 0 when enrolled == 0
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
    # clip to train min/max, floor at 0, clip percentages to [0, 100], then round
    # to the correct grid for each feature. Only attackable columns are touched
    # here; frozen columns are restored by the caller.
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
    clean_reset = X_clean_raw[selected_features].reset_index(drop=True)
    for j, col in enumerate(numeric_cols):
        X_adv_raw[col] = adv_numeric_raw[:, j]

    # restore non-attackable raw features to their clean values
    for col in selected_features:
        if col not in attackable_raw:
            X_adv_raw[col] = clean_reset[col].values

    # per-feature domain constraints (clip + round) on the attacked columns
    X_adv_raw = apply_domain_constraints(X_adv_raw, numeric_cols, train_min, train_max)

    # enforce the derived credit identities on the finalized components (always on,
    # hardcoded, matching zenodo_pgd_mlp.py / zenodo_dnd_train_mlp.py and the OULAD scripts)
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

    # hard-restore frozen preprocessed dimensions (flag + one-hot blocks etc.)
    frozen = perturbable_mask_preprocessed == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]

    # restore non-attacked rows in preprocessed space too
    if attacked_row_mask is not None:
        not_attacked = ~np.asarray(attacked_row_mask).astype(bool)
        if not_attacked.any():
            X_projected_preprocessed[not_attacked, :] = X_clean_preprocessed[not_attacked, :]

    return X_adv_raw[selected_features], X_projected_preprocessed

# ============================================================
# PREDICTION, EVALUATION, SAVING
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

def predict_mlp_label(model, X_preprocessed, threshold, device):
    proba = predict_mlp_proba(model, X_preprocessed, device)
    pred = (proba >= threshold).astype(int)
    return pred, proba

def tune_threshold_from_proba(y_true, y_proba, min_precision=MIN_PRECISION):
    # Defended-model threshold selection on the TEST split. Matches the baseline
    # mlp_model.py rule: sweep thresholds, then among the ones that keep dropout
    # precision at or above min_precision, pick the highest recall (f1 then
    # precision break ties). If no threshold reaches the precision floor, fall
    # back to best f1 so the function still returns a usable threshold.
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        })
    threshold_df = pd.DataFrame(rows)

    eligible = threshold_df[threshold_df["precision"] >= min_precision]
    if len(eligible) > 0:
        # precision floor is reachable: maximize recall inside the eligible set
        best_row = eligible.sort_values(
            ["recall", "f1_score", "precision"], ascending=[False, False, False]
        ).iloc[0]
    else:
        # no threshold reaches the precision floor: fall back to best f1
        best_row = threshold_df.sort_values(
            ["f1_score", "recall", "precision"], ascending=[False, False, False]
        ).iloc[0]

    return float(best_row["threshold"]), threshold_df, best_row.to_dict()

def evaluate_mlp_with_threshold(model, X_preprocessed, y_true, threshold, condition, train_epsilon, eval_epsilon, split, device):
    y_true = np.asarray(y_true).astype(int)
    y_proba = predict_mlp_proba(model, X_preprocessed, device)
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "model": "MLP Defended",
        "condition": condition,
        "train_epsilon": train_epsilon,
        "eval_epsilon": eval_epsilon,
        "epsilon": eval_epsilon,
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
    diff = X_adv_preprocessed - X_clean_preprocessed
    pm = perturbable_mask.astype(bool)
    frozen = ~pm
    stats = {
        "mean_abs_perturbation_all_preprocessed": float(np.abs(diff).mean()),
        "mean_abs_perturbation_attackable_preprocessed": float(np.abs(diff[:, pm]).mean()),
        "max_abs_perturbation_attackable_preprocessed": float(np.abs(diff[:, pm]).max()),
        "max_abs_perturbation_frozen_preprocessed": float(np.abs(diff[:, frozen]).max()) if frozen.any() else 0.0,
    }
    if attacked_row_mask is not None:
        attacked_row_mask = np.asarray(attacked_row_mask).astype(bool)
        stats["attacked_rows_count"] = int(attacked_row_mask.sum())
        if attacked_row_mask.any():
            attacked_diff = diff[attacked_row_mask]
            stats["mean_abs_perturbation_attacked_rows_attackable_preprocessed"] = float(np.abs(attacked_diff[:, pm]).mean())
            stats["max_abs_perturbation_attacked_rows_attackable_preprocessed"] = float(np.abs(attacked_diff[:, pm]).max())
        else:
            stats["mean_abs_perturbation_attacked_rows_attackable_preprocessed"] = 0.0
            stats["max_abs_perturbation_attacked_rows_attackable_preprocessed"] = 0.0
    return stats

def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(cm, index=["Actual_non_dropout", "Actual_dropout"], columns=["Pred_non_dropout", "Pred_dropout"])
    cm_df.to_csv(path, index=True)

def save_predictions(student_ids, y_true, clean_pred, adv_pred, clean_proba, adv_proba, threshold, condition, train_epsilon, eval_epsilon, path):
    y_true = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        ID_COL: student_ids.values,
        "condition": condition,
        "train_epsilon": train_epsilon,
        "eval_epsilon": eval_epsilon,
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        "clean_pred_label": clean_pred,
        "clean_pred_target": np.where(clean_pred == 1, "dropout", "non-dropout"),
        "adv_pred_label": adv_pred,
        "adv_pred_target": np.where(adv_pred == 1, "dropout", "non-dropout"),
        "clean_prob_dropout": clean_proba,
        "adv_prob_dropout": adv_proba,
        "prob_dropout_change": adv_proba - clean_proba,
        "prediction_changed": clean_pred != adv_pred,
        "dropout_hidden_tp_to_fn": ((y_true == 1) & (clean_pred == 1) & (adv_pred == 0)),
        "threshold": threshold,
    })
    pred_df.to_csv(path, index=False)

def make_training_sample_csv(X_clean_raw, X_adv_raw, y, selected_features, selected_rows, train_epsilon, path, max_samples=50):
    selected_rows = np.asarray(selected_rows).astype(bool)
    clean = X_clean_raw.reset_index(drop=True).loc[selected_rows].reset_index(drop=True)
    adv = X_adv_raw.reset_index(drop=True).loc[selected_rows].reset_index(drop=True)
    labels = np.asarray(y).astype(int)[selected_rows]
    n = min(max_samples, len(clean))
    rows = []
    for i in range(n):
        row = {
            "sample_id": i,
            "label": int(labels[i]),
            "label_text": "dropout" if labels[i] == 1 else "non-dropout",
            "train_epsilon": train_epsilon,
        }
        for feature in selected_features:
            clean_val = clean.loc[i, feature]
            adv_val = adv.loc[i, feature]
            row[f"clean_{feature}"] = clean_val
            row[f"adv_{feature}"] = adv_val
            c = pd.to_numeric(clean_val, errors="coerce")
            a = pd.to_numeric(adv_val, errors="coerce")
            row[f"diff_{feature}"] = (a - c) if pd.notna(a) and pd.notna(c) else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

# ============================================================
# ATTACK LABELS, SCOPE, AND GENERATION
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
    art_mlp,
    X_clean_preprocessed,
    y_true,
    clean_pred,
    epsilon,
    pgd_step,
    pgd_iterations,
    attack_mode,
    attack_scope,
    perturbable_mask,
    batch_size=128,
):
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()
    if int(attack_idx.sum()) == 0:
        return X_adv_preprocessed, attack_idx
    attack_y, targeted = make_attack_labels(attack_mode, y_true=np.asarray(y_true)[attack_idx])
    effective_step = min(float(pgd_step), float(epsilon))
    pgd = ProjectedGradientDescent(
        estimator=art_mlp,
        eps=float(epsilon),
        eps_step=effective_step,
        max_iter=int(pgd_iterations),
        targeted=targeted,
        batch_size=batch_size,
    )
    X_adv_subset = pgd.generate(
        x=X_clean_preprocessed[attack_idx],
        y=attack_y,
        mask=perturbable_mask,
    )
    frozen = perturbable_mask == 0
    X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
    X_adv_preprocessed[attack_idx] = X_adv_subset
    return X_adv_preprocessed, attack_idx

def generate_pgd_raw_and_scaled_for_scope(
    art_mlp,
    X_clean_raw,
    X_clean_preprocessed,
    y_true,
    clean_pred,
    epsilon,
    pgd_step,
    pgd_iterations,
    attack_mode,
    attack_scope,
    perturbable_mask,
    preprocessor,
    selected_features,
    train_min,
    train_max,
):
    X_adv_pre_raw, attack_idx = generate_pgd_for_scope(
        art_mlp=art_mlp,
        X_clean_preprocessed=X_clean_preprocessed,
        y_true=y_true,
        clean_pred=clean_pred,
        epsilon=epsilon,
        pgd_step=pgd_step,
        pgd_iterations=pgd_iterations,
        attack_mode=attack_mode,
        attack_scope=attack_scope,
        perturbable_mask=perturbable_mask,
    )
    X_adv_raw, X_adv_preprocessed = preprocessed_adv_to_raw_and_scaled(
        X_adv_preprocessed=X_adv_pre_raw,
        X_clean_preprocessed=X_clean_preprocessed,
        X_clean_raw=X_clean_raw,
        selected_features=selected_features,
        preprocessor=preprocessor,
        train_min=train_min,
        train_max=train_max,
        perturbable_mask_preprocessed=perturbable_mask,
        attacked_row_mask=attack_idx,
    )
    return X_adv_raw, X_adv_preprocessed, attack_idx

# ============================================================
# TRAINING
# ============================================================
def _val_loss_curve_defended(X_tr_np, y_tr_np, X_va_np, y_va_np, input_dim, hidden_dims,
                             dropout, learning_rate, weight_decay, class_weight,
                             device, random_state, n_epochs, batch_size=128):
    # Train the defended config on a CV fold and return the per-epoch validation
    # loss list. Used ONLY to estimate the fixed epoch budget. The matrices are
    # already in preprocessed model space (clean + adversarial), so no preprocessing
    # is refit here.
    set_global_seed(random_state)
    model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = make_criterion(class_weight, y_tr_np, device)
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    generator = torch.Generator()
    generator.manual_seed(random_state)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_np, dtype=torch.float32),
                      torch.tensor(y_tr_np, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, generator=generator,
        drop_last=len(y_tr_np) > batch_size,
    )
    X_va_tensor = torch.tensor(X_va_np, dtype=torch.float32, device=device)
    y_va_tensor = torch.tensor(y_va_np, dtype=torch.long, device=device)
    val_losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_losses.append(float(criterion(model(X_va_tensor), y_va_tensor).item()))
    return val_losses

def estimate_fixed_epochs_via_cv_defended(X_mixed, y_mixed, input_dim, hidden_dims, dropout,
                                          learning_rate, weight_decay, class_weight,
                                          device, random_state,
                                          n_splits=CV_N_SPLITS_FOR_EPOCHS,
                                          epochs_cap=EPOCHS_CAP, batch_size=128):
    # ONE short stratified k-fold CV INSIDE the mixed (clean + PGD adversarial)
    # training matrix to choose a single fixed epoch budget for the defended MLP.
    # The test/validation splits are never touched. The chosen budget is the epoch
    # that minimizes the mean validation loss averaged across folds.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    curve_sum = np.zeros(epochs_cap)
    curve_cnt = np.zeros(epochs_cap)
    print(f"Estimating fixed epoch budget for defended MLP via {n_splits}-fold CV "
          f"inside the mixed (clean+adv) train set:")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_mixed, y_mixed), 1):
        losses = _val_loss_curve_defended(
            X_mixed[tr_idx], y_mixed[tr_idx], X_mixed[va_idx], y_mixed[va_idx],
            input_dim, hidden_dims, dropout, learning_rate, weight_decay,
            class_weight, device, random_state, epochs_cap, batch_size,
        )
        for e, l in enumerate(losses):
            curve_sum[e] += l
            curve_cnt[e] += 1
        print(f"  fold {fold}/{n_splits}: min val loss at epoch {int(np.argmin(losses)) + 1}")
    mean_curve = curve_sum / np.maximum(curve_cnt, 1)
    best_epoch = max(int(np.argmin(mean_curve)) + 1, MIN_FIXED_EPOCHS)
    print(f"Chosen fixed epoch budget (defended): {best_epoch} "
          f"(argmin of mean CV val loss, floored at {MIN_FIXED_EPOCHS})")
    print()
    return best_epoch, mean_curve.tolist()

def train_defended_mlp(
    model,
    X_train_mixed,
    y_train_mixed,
    train_epsilon,
    output_dir,
    device,
    random_state,
    n_epochs,
    learning_rate=1e-3,
    weight_decay=1e-4,
    class_weight=None,
    batch_size=128,
):
    # Fixed epoch budget with FINAL weights and NO early stopping. The test and
    # validation splits are never consulted during training (the budget is chosen
    # by CV inside the mixed train set). Same regime as the baseline
    # mlp_model.py / zenodo_mlp_model.py, so the clean baseline and the defended
    # model are trained the same way and their difference reflects the defence
    # rather than a change of training procedure.
    set_global_seed(random_state)
    X_train_tensor = torch.tensor(X_train_mixed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_mixed, dtype=torch.long)

    generator = torch.Generator()
    generator.manual_seed(random_state)
    loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=len(y_train_mixed) > batch_size,
    )

    criterion = make_criterion(class_weight, y_train_mixed, device)
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    history = []
    print(f"\nTraining defended MLP with clean + PGD adversarial samples "
          f"(fixed budget of {n_epochs} epochs, no early stopping)...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        history.append({
            "epoch": epoch,
            "train_epsilon": train_epsilon,
            "train_loss": train_loss,
        })
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")

    eps_tag = str(train_epsilon).replace(".", "p")
    history_path = os.path.join(output_dir, f"zenodo_mlp_defended_training_history_train_eps_{eps_tag}.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    return model, history, history_path

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(
    df,
    csv_path,
    output_dir,
    mlp_results_dir,
    train_epsilon_list,
    eval_epsilon_list,
    pgd_step,
    pgd_iterations,
    random_state,
    attack_mode,
    train_attack_scope,
    eval_attack_scope,
    eval_attack_source,
    eval_splits,
    class_weight_override,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MLP ADVERSARIAL TRAINING WITH PGD (zenodo)")
    print("=" * 80)
    print(f"CSV path           : {csv_path}")
    print(f"Output dir         : {output_dir}")
    print(f"MLP results dir    : {mlp_results_dir}")
    print(f"Attack mode        : {attack_mode}")
    print(f"Train attack scope : {train_attack_scope}")
    print(f"Eval attack scope  : {eval_attack_scope}")
    print(f"Eval attack source : {eval_attack_source}")
    print(f"Credit identities  : always enforced (passed<=enrolled; performance=passed/enrolled*100)")
    print(f"PGD step           : {pgd_step}")
    print(f"PGD iterations     : {pgd_iterations}")
    print(f"Eval splits        : {eval_splits}")
    print(f"Device             : {device}")
    print(f"Dataset shape      : {df.shape}")
    print()

    model_path = resolve_mlp_model_path(mlp_results_dir)
    preprocessor_path = resolve_mlp_preprocessor_path(mlp_results_dir)
    summary_path = resolve_mlp_summary_path(mlp_results_dir)
    summary_threshold, _ = load_summary_threshold(summary_path)

    checkpoint_peek = load_torch_checkpoint(model_path, device)
    if isinstance(checkpoint_peek, dict) and checkpoint_peek.get("selected_features") is not None:
        selected_features = list(checkpoint_peek["selected_features"])
        feature_source = "mlp_checkpoint"
    else:
        selected_features = DEFAULT_SELECTED_FEATURES.copy()
        feature_source = "fallback_default"
    print(f"Feature list source: {feature_source} ({len(selected_features)} features)")
    print()

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

    preprocessor = joblib.load(preprocessor_path)
    X_train_pre = transform_to_float32(preprocessor, X_train)
    X_test_pre = transform_to_float32(preprocessor, X_test)
    X_val_pre = transform_to_float32(preprocessor, X_val)
    input_dim = X_train_pre.shape[1]

    checkpoint = load_mlp_checkpoint(model_path, input_dim, device, summary_threshold=summary_threshold)
    base_model = checkpoint["model"]
    hidden_dims = checkpoint["hidden_dims"]
    dropout = checkpoint["dropout"]
    baseline_threshold = checkpoint["threshold"]
    best_params = checkpoint["best_params"]
    winning_strategy = checkpoint["winning_strategy"]

    if checkpoint["selected_features"] is not None and list(checkpoint["selected_features"]) != list(selected_features):
        raise ValueError("selected_features in the checkpoint changed between the peek and full load.")

    learning_rate = float(best_params.get("learning_rate", best_params.get("lr", 1e-3)))
    weight_decay = float(best_params.get("weight_decay", 1e-4))
    checkpoint_class_weight = best_params.get("class_weight", None)

    if class_weight_override == "auto":
        class_weight = checkpoint_class_weight
        class_weight_source = "checkpoint_best_params"
    else:
        class_weight = parse_class_weight(class_weight_override)
        class_weight_source = "cli_override"

    # If the baseline's winning strategy was a resampler (SMOTE / random_over /
    # etc.) rather than class weights, this script does NOT resample. Warn so the
    # imbalance handling is not silently dropped; pass --class-weight to add it.
    resampler_strategies = {"smote", "random_over", "random_under", "smotetomek"}
    if class_weight_source == "checkpoint_best_params" and winning_strategy in resampler_strategies \
            and (class_weight is None):
        print(f"WARNING: baseline winning_strategy is '{winning_strategy}' (a resampler), but this "
              f"adversarial-training script does not resample and no class_weight was found. "
              f"The defended model will train without imbalance handling unless you pass "
              f"--class-weight (e.g. 'balanced' or \"{{0:1,1:8}}\").")
        print()

    perturbable_mask, raw_mask_df, preprocessed_mask_df = make_preprocessed_attack_mask(
        preprocessor, selected_features, input_dim, output_dir
    )

    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)

    art_base_mlp = wrap_mlp_for_art(base_model, input_dim=input_dim)

    print("Loaded baseline MLP:")
    print(model_path)
    print("Loaded preprocessor:")
    print(preprocessor_path)
    if summary_path:
        print("Loaded summary:")
        print(summary_path)
    print(f"Baseline threshold : {baseline_threshold} | source: {checkpoint['threshold_source']}")
    print(f"Winning strategy   : {winning_strategy}")
    print(f"Hidden dims        : {hidden_dims}")
    print(f"Dropout            : {dropout}")
    print(f"Learning rate      : {learning_rate}")
    print(f"Weight decay       : {weight_decay}")
    print(f"Class weight       : {class_weight} | source: {class_weight_source}")
    print()
    print("Attackable raw features:")
    print(raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist())
    print()
    print("Preprocessed attackable dimensions:")
    print(int(perturbable_mask.sum()), "of", len(perturbable_mask))
    print()

    baseline_clean_test_result, baseline_clean_test_cm, baseline_test_pred, baseline_test_proba = evaluate_mlp_with_threshold(
        base_model, X_test_pre, y_test.values, baseline_threshold, "Baseline Clean", np.nan, 0.0, "test", device
    )
    baseline_train_pred, baseline_train_proba = predict_mlp_label(base_model, X_train_pre, baseline_threshold, device)
    baseline_clean_val_result, baseline_clean_val_cm, baseline_val_pred, baseline_val_proba = evaluate_mlp_with_threshold(
        base_model, X_val_pre, y_val.values, baseline_threshold, "Baseline Clean", np.nan, 0.0, "validation", device
    )
    print("Baseline clean test result:")
    print(baseline_clean_test_result)
    print(baseline_clean_test_cm)
    print()

    all_results = [baseline_clean_test_result, baseline_clean_val_result]
    all_stats = []
    all_summary = []

    save_confusion_matrix(baseline_clean_test_cm, os.path.join(output_dir, "zenodo_mlp_baseline_clean_test_cm.csv"))
    save_confusion_matrix(baseline_clean_val_cm, os.path.join(output_dir, "zenodo_mlp_baseline_clean_validation_cm.csv"))

    for train_epsilon in train_epsilon_list:
        print("=" * 80)
        print(f"TRAIN DEFENDED MLP | train epsilon = {train_epsilon}")
        print("=" * 80)

        X_train_adv_raw, X_train_adv_pre, train_attack_idx = generate_pgd_raw_and_scaled_for_scope(
            art_mlp=art_base_mlp,
            X_clean_raw=X_train,
            X_clean_preprocessed=X_train_pre,
            y_true=y_train.values,
            clean_pred=baseline_train_pred,
            epsilon=train_epsilon,
            pgd_step=pgd_step,
            pgd_iterations=pgd_iterations,
            attack_mode=attack_mode,
            attack_scope=train_attack_scope,
            perturbable_mask=perturbable_mask,
            preprocessor=preprocessor,
            selected_features=selected_features,
            train_min=train_min,
            train_max=train_max,
        )

        train_stats = perturbation_stats(X_train_pre, X_train_adv_pre, perturbable_mask, attacked_row_mask=train_attack_idx)
        train_stats.update({
            "train_epsilon": train_epsilon,
            "eval_epsilon": train_epsilon,
            "split": "train_adversarial_generation",
            "pgd_step": pgd_step,
            "pgd_iterations": pgd_iterations,
            "attack_scope": train_attack_scope,
            "train_attack_scope": train_attack_scope,
        })
        all_stats.append(train_stats)

        if train_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("Frozen transformed features changed in adversarial training data.")

        eps_tag = str(train_epsilon).replace(".", "p")
        X_train_adv_raw.to_csv(os.path.join(output_dir, f"zenodo_X_train_pgd_raw_train_eps_{eps_tag}.csv"), index=False)
        np.save(os.path.join(output_dir, f"zenodo_X_train_pgd_preprocessed_train_eps_{eps_tag}.npy"), X_train_adv_pre)

        make_training_sample_csv(
            X_clean_raw=X_train,
            X_adv_raw=X_train_adv_raw,
            y=y_train.values,
            selected_features=selected_features,
            selected_rows=train_attack_idx,
            train_epsilon=train_epsilon,
            path=os.path.join(output_dir, f"zenodo_clean_vs_pgd_train_sample_train_eps_{eps_tag}.csv"),
        )

        X_train_adv_subset = X_train_adv_pre[train_attack_idx]
        y_train_adv_subset = y_train.values.astype(int)[train_attack_idx]
        X_train_mixed = np.vstack([X_train_pre, X_train_adv_subset]).astype(np.float32)
        y_train_mixed = np.concatenate([y_train.values.astype(int), y_train_adv_subset])
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X_train_mixed))
        X_train_mixed = X_train_mixed[idx]
        y_train_mixed = y_train_mixed[idx]

        print(f"Clean train rows             : {len(X_train_pre)}")
        print(f"Adversarial train rows added : {len(X_train_adv_subset)}")
        print(f"Mixed train rows             : {len(X_train_mixed)}")
        print()

        # Estimate the fixed epoch budget ONCE via CV inside the mixed (clean+adv)
        # train set, using the defended config. Test/validation are never touched.
        fixed_epochs, _ = estimate_fixed_epochs_via_cv_defended(
            X_train_mixed, y_train_mixed, input_dim, hidden_dims, dropout,
            learning_rate, weight_decay, class_weight, device, random_state,
        )

        # Train from scratch: fresh random init, NOT warm-started from the
        # baseline weights, so any change vs the baseline comes from the training
        # data rather than from inheriting the baseline's learned weights. The
        # seed is reset here so the random initialization is reproducible.
        set_global_seed(random_state)
        defended_model = DropoutMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
        defended_model, history, history_path = train_defended_mlp(
            model=defended_model,
            X_train_mixed=X_train_mixed,
            y_train_mixed=y_train_mixed,
            train_epsilon=train_epsilon,
            output_dir=output_dir,
            device=device,
            random_state=random_state,
            n_epochs=fixed_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weight=class_weight,
            batch_size=128,
        )

        test_clean_proba_for_threshold = predict_mlp_proba(defended_model, X_test_pre, device)
        defended_threshold, threshold_df, best_threshold_row = tune_threshold_from_proba(
            y_test.values, test_clean_proba_for_threshold
        )
        threshold_df.to_csv(os.path.join(output_dir, f"zenodo_mlp_defended_threshold_tuning_train_eps_{eps_tag}.csv"), index=False)

        defended_model_path = os.path.join(output_dir, f"zenodo_mlp_defended_model_train_eps_{eps_tag}.pt")
        checkpoint_out = {
            "model_state_dict": defended_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "selected_features": selected_features,
            "train_epsilon": train_epsilon,
            "fixed_epochs": fixed_epochs,
            "early_stopping": "none (fixed epoch budget, final weights)",
            "epoch_budget_source": "CV inside mixed (clean+adv) train set; test/val untouched",
            "best_threshold": defended_threshold,
            "best_threshold_row": best_threshold_row,
            "threshold_rule": f"max dropout recall subject to dropout precision >= {MIN_PRECISION}, else best F1 (same as baseline mlp_model.py)",
            "min_precision": MIN_PRECISION,
            "training_source_model": model_path,
            "baseline_winning_strategy": winning_strategy,
            "initialization": "random_from_scratch (baseline weights NOT loaded)",
            "training_method": "Trained from scratch (fresh random init, no warm-start) on clean training data + PGD adversarial samples generated from the training attack scope",
            "attack_mode": attack_mode,
            "train_attack_scope": train_attack_scope,
            "eval_attack_scope": eval_attack_scope,
            "pgd_step": pgd_step,
            "pgd_iterations": pgd_iterations,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "class_weight": str(class_weight),
            "enforce_derived_consistency": True,
        }
        torch.save(checkpoint_out, defended_model_path)
        print(f"Saved defended MLP model: {defended_model_path}")
        print(f"Defended MLP threshold : {defended_threshold}")
        print()

        defended_clean_preds = {}
        defended_clean_probas = {}
        for split_name, X_pre_split, y_split, id_split in [
            ("train", X_train_pre, y_train, id_train),
            ("test", X_test_pre, y_test, id_test),
            ("validation", X_val_pre, y_val, id_val),
        ]:
            result, cm, pred, proba = evaluate_mlp_with_threshold(
                defended_model, X_pre_split, y_split.values, defended_threshold,
                "Defended Clean", train_epsilon, 0.0, split_name, device
            )
            defended_clean_preds[split_name] = pred
            defended_clean_probas[split_name] = proba
            all_results.append(result)
            save_confusion_matrix(cm, os.path.join(output_dir, f"zenodo_mlp_defended_clean_{split_name}_cm_train_eps_{eps_tag}.csv"))
            save_predictions(
                student_ids=id_split,
                y_true=y_split.values,
                clean_pred=pred,
                adv_pred=pred,
                clean_proba=proba,
                adv_proba=proba,
                threshold=defended_threshold,
                condition="Defended Clean",
                train_epsilon=train_epsilon,
                eval_epsilon=0.0,
                path=os.path.join(output_dir, f"zenodo_mlp_defended_clean_{split_name}_predictions_train_eps_{eps_tag}.csv"),
            )
            print(f"Defended clean {split_name}:")
            print(result)
            print(cm)
            print()

        if eval_attack_source == "defended":
            art_eval_mlp = wrap_mlp_for_art(defended_model, input_dim=input_dim)
        elif eval_attack_source == "baseline":
            art_eval_mlp = art_base_mlp
        else:
            raise ValueError("eval_attack_source must be 'defended' or 'baseline'.")

        available_splits = {
            "train": (X_train, X_train_pre, y_train, id_train),
            "test": (X_test, X_test_pre, y_test, id_test),
            "validation": (X_val, X_val_pre, y_val, id_val),
        }

        for eval_epsilon in eval_epsilon_list:
            eval_tag = str(eval_epsilon).replace(".", "p")
            print("-" * 80)
            print(f"EVALUATE DEFENDED MLP | train eps = {train_epsilon} | eval eps = {eval_epsilon}")
            print("-" * 80)
            for split_name in eval_splits:
                if split_name not in available_splits:
                    raise ValueError(f"Unknown eval split: {split_name}")
                X_raw_split, X_pre_split, y_split, id_split = available_splits[split_name]
                clean_pred_for_scope = defended_clean_preds[split_name]
                clean_proba_for_scope = defended_clean_probas[split_name]

                X_adv_raw, X_adv_pre, eval_attack_idx = generate_pgd_raw_and_scaled_for_scope(
                    art_mlp=art_eval_mlp,
                    X_clean_raw=X_raw_split,
                    X_clean_preprocessed=X_pre_split,
                    y_true=y_split.values,
                    clean_pred=clean_pred_for_scope,
                    epsilon=eval_epsilon,
                    pgd_step=pgd_step,
                    pgd_iterations=pgd_iterations,
                    attack_mode=attack_mode,
                    attack_scope=eval_attack_scope,
                    perturbable_mask=perturbable_mask,
                    preprocessor=preprocessor,
                    selected_features=selected_features,
                    train_min=train_min,
                    train_max=train_max,
                )

                stats = perturbation_stats(X_pre_split, X_adv_pre, perturbable_mask, attacked_row_mask=eval_attack_idx)
                stats.update({
                    "train_epsilon": train_epsilon,
                    "eval_epsilon": eval_epsilon,
                    "split": split_name,
                    "pgd_step": pgd_step,
                    "pgd_iterations": pgd_iterations,
                    "attack_scope": eval_attack_scope,
                    "train_attack_scope": train_attack_scope,
                    "eval_attack_scope": eval_attack_scope,
                    "eval_attack_source": eval_attack_source,
                })
                all_stats.append(stats)

                if stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
                    raise AssertionError(f"Frozen transformed features changed in {split_name}.")

                X_adv_raw.to_csv(os.path.join(output_dir, f"zenodo_X_{split_name}_pgd_raw_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"), index=False)
                np.save(os.path.join(output_dir, f"zenodo_X_{split_name}_pgd_preprocessed_train_eps_{eps_tag}_eval_eps_{eval_tag}.npy"), X_adv_pre)

                result, cm, adv_pred, adv_proba = evaluate_mlp_with_threshold(
                    defended_model, X_adv_pre, y_split.values, defended_threshold,
                    f"Defended PGD {attack_mode}", train_epsilon, eval_epsilon, split_name, device
                )

                hidden = int(np.sum((y_split.values == 1) & (clean_pred_for_scope == 1) & (adv_pred == 0)))
                false_alarm = int(np.sum((y_split.values == 0) & (clean_pred_for_scope == 0) & (adv_pred == 1)))
                changed = int(np.sum(clean_pred_for_scope != adv_pred))
                result["attack_scope"] = eval_attack_scope
                result["train_attack_scope"] = train_attack_scope
                result["eval_attack_scope"] = eval_attack_scope
                result["eval_attack_source"] = eval_attack_source
                result["attacked_rows_count"] = int(eval_attack_idx.sum())
                result["attacked_dropout_rows_count"] = int(np.sum(eval_attack_idx & (y_split.values == 1)))
                result["dropout_to_nondropout_flips"] = hidden
                result["nondropout_to_dropout_flips"] = false_alarm
                result["prediction_changed_count"] = changed
                all_results.append(result)

                save_confusion_matrix(cm, os.path.join(output_dir, f"zenodo_mlp_defended_pgd_{split_name}_cm_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"))
                save_predictions(
                    student_ids=id_split,
                    y_true=y_split.values,
                    clean_pred=clean_pred_for_scope,
                    adv_pred=adv_pred,
                    clean_proba=clean_proba_for_scope,
                    adv_proba=adv_proba,
                    threshold=defended_threshold,
                    condition=f"Defended PGD {attack_mode}",
                    train_epsilon=train_epsilon,
                    eval_epsilon=eval_epsilon,
                    path=os.path.join(output_dir, f"zenodo_mlp_defended_pgd_{split_name}_predictions_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                )
                print(f"Defended PGD {split_name}:")
                print(result)
                print(cm)
                print()

        all_summary.append({
            "model": "MLP Defended",
            "defence": "Adversarial training with PGD samples",
            "attack_mode": attack_mode,
            "train_attack_scope": train_attack_scope,
            "eval_attack_scope": eval_attack_scope,
            "eval_attack_source": eval_attack_source,
            "train_epsilon": train_epsilon,
            "defended_threshold": defended_threshold,
            "best_threshold_row": best_threshold_row,
            "clean_train_rows": int(len(X_train_pre)),
            "pgd_train_rows_added": int(len(X_train_adv_subset)),
            "mixed_train_rows": int(len(X_train_mixed)),
            "model_path": defended_model_path,
            "history_path": history_path,
            "fixed_epochs": fixed_epochs,
            "early_stopping": "none (fixed epoch budget, final weights)",
            "epoch_budget_source": "CV inside mixed (clean+adv) train set; test/val untouched",
        })

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)
    summary_df = pd.DataFrame(all_summary)
    results_path = os.path.join(output_dir, "zenodo_mlp_adversarial_training_results.csv")
    stats_path = os.path.join(output_dir, "zenodo_mlp_adversarial_training_perturbation_stats.csv")
    summary_csv_path = os.path.join(output_dir, "zenodo_mlp_adversarial_training_summary_by_train_epsilon.csv")
    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "defence": "MLP adversarial training",
        "training_method": f"Trained from scratch (fresh random init, no warm-start from baseline weights) on clean training data + PGD adversarial samples generated from train_attack_scope={train_attack_scope} (attacked rows only).",
        "defended_initialization": "random_from_scratch",
        "target_model": "MLP Defended",
        "gradient_model": "MLP itself (white-box)",
        "attack_used_for_training": "PGD",
        "early_stopping": "none (fixed epoch budget, final weights; budget via CV inside mixed train set)",
        "defended_threshold_rule": (f"On the test split, pick the threshold with the highest dropout "
                                    f"recall subject to dropout precision >= {MIN_PRECISION}; if no "
                                    f"threshold meets the floor, fall back to the F1-optimal threshold. "
                                    f"Same rule as the baseline mlp_model.py."),
        "min_precision": MIN_PRECISION,
        "split_usage": {
            "train": "defended model fitting (clean + PGD adversarial samples)",
            "test": "defended threshold selection only (not used for stopping; the epoch budget is chosen by CV inside the mixed train set)",
            "validation": "final held-out evaluation only",
        },
        "attack_mode": attack_mode,
        "train_attack_scope": train_attack_scope,
        "eval_attack_scope": eval_attack_scope,
        "eval_attack_source": eval_attack_source,
        "enforce_derived_consistency": True,
        "derived_features_note": ("Always on (hardcoded). total_credits_passed_academic_year is capped "
                                  "at total_credits_enrolled_academic_year and total_performance is "
                                  "recomputed as passed/enrolled*100 on the finalized clipped+rounded "
                                  "rows, so the adversarial rows stay domain-valid at their final values."),
        "epsilon_space": "L-inf bound in the MLP preprocessed/standardized feature space",
        "train_epsilon_list": train_epsilon_list,
        "eval_epsilon_list": eval_epsilon_list,
        "pgd_step": pgd_step,
        "pgd_iterations": pgd_iterations,
        "baseline_mlp_model_path": model_path,
        "baseline_threshold": baseline_threshold,
        "baseline_threshold_source": checkpoint["threshold_source"],
        "baseline_winning_strategy": winning_strategy,
        "preprocessor_path": preprocessor_path,
        "feature_list_source": feature_source,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "class_weight": str(class_weight),
        "class_weight_source": class_weight_source,
        "n_raw_features": len(selected_features),
        "n_preprocessed_features": int(input_dim),
        "n_attackable_preprocessed_features": int(perturbable_mask.sum()),
        "n_frozen_preprocessed_features": int(len(perturbable_mask) - perturbable_mask.sum()),
        "attackable_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist(),
        "frozen_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 0, "feature"].tolist(),
        "constraints": {
            "frozen_features": ("Restored to clean values after PGD (academic_year, n_courses, "
                                "degree_size, seniority, highest_course_year_enrolled and all 8 "
                                "*_flag columns)."),
            "non_attacked_rows": "Restored completely to clean values.",
            "raw_value_clip": "Numeric features clipped to train min/max when available.",
            "percentage_features": ("pass_ratio_sem_a/b, total_performance and the three "
                                    "completion_rate_*_before features clipped to [0, 100]."),
            "half_credit_features": ("credits_enrolled_semester_a/b, "
                                     "total_credits_enrolled_academic_year and "
                                     "total_credits_passed_academic_year snapped to the nearest .0/.5."),
            "credit_identities": ("passed <= enrolled and total_performance = passed/enrolled*100 "
                                  "re-imposed when enforce_derived_consistency is on (default)."),
            "integer_count_features": ("lms_* and attendance_days_* rounded to whole numbers "
                                       "(lms_total_minutes_* included)."),
            "decimal_features": "courses_mean plus the percentage features rounded to 2 decimal places.",
            "nonnegative_features": "Every attackable feature clipped to a lower bound of 0.",
            "resampling_note": ("This script does NOT resample. If the baseline MLP's winning strategy "
                                "used a sampler, pass --class-weight to add imbalance handling."),
        },
        "split": split_report,
        "results_file": results_path,
        "stats_file": stats_path,
        "summary_by_train_epsilon_file": summary_csv_path,
    }
    summary_path_out = os.path.join(output_dir, "zenodo_mlp_adversarial_training_summary.json")
    with open(summary_path_out, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("MLP ADVERSARIAL TRAINING DONE")
    print("=" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Stats saved to  : {stats_path}")
    print(f"Summary saved to: {summary_path_out}")
    return summary

# ============================================================
# ARGUMENTS
# ============================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Adversarial training for the MLP using PGD (zenodo).")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help=f"Dataset CSV. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--mlp-results-dir", default=DEFAULT_MLP_RESULTS_DIR, help=f"Baseline MLP results folder. Default: {DEFAULT_MLP_RESULTS_DIR}")
    parser.add_argument("--mlp-dir", dest="mlp_results_dir", help="Alias for --mlp-results-dir.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Defended MLP output folder. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help=f"Random state. Default: {DEFAULT_RANDOM_STATE}")
    parser.add_argument("--attack-mode", default="target_non_dropout", choices=["untargeted", "target_non_dropout"])
    parser.add_argument("--train-attack-scope", default="dropout_only", choices=["all", "dropout_only", "true_positive_only"],
                        help="Scope used to generate PGD samples for adversarial training. Default: dropout_only.")
    parser.add_argument("--eval-attack-scope", default="true_positive_only", choices=["all", "dropout_only", "true_positive_only"],
                        help="Scope used for the final PGD evaluation. Default: true_positive_only.")
    parser.add_argument("--attack-scope", default=None, choices=["all", "dropout_only", "true_positive_only"],
                        help="Backward-compatible alias. If set, applies to both train and eval scope.")
    parser.add_argument("--eval-attack-source", default="defended", choices=["defended", "baseline"],
                        help="defended = adaptive white-box PGD against the defended MLP. baseline = transfer from the baseline MLP.")
    parser.add_argument("--class-weight", default="auto",
                        help="'auto' keeps the checkpoint best_params class_weight; otherwise pass 'balanced', "
                             "'None', or a dict like \"{0:1,1:8}\". Use this if the baseline used a resampler.")
    parser.add_argument("--train-epsilons", default=",".join(str(v) for v in TRAIN_EPSILON_LIST),
                        help=f"Comma-separated train epsilon list. Default: {','.join(str(v) for v in TRAIN_EPSILON_LIST)}")
    parser.add_argument("--eval-epsilons", default=",".join(str(v) for v in EVAL_EPSILON_LIST),
                        help=f"Comma-separated eval epsilon list. Default: {','.join(str(v) for v in EVAL_EPSILON_LIST)}")
    # Credit-identity enforcement is ALWAYS ON (hardcoded, no flag), matching
    # zenodo_pgd_mlp.py / zenodo_dnd_train_mlp.py and the OULAD scripts: after every
    # PGD attack passed is capped at enrolled and total_performance is recomputed as
    # passed/enrolled*100 on the finalized rows.
    parser.add_argument("--pgd-step", type=float, default=PGD_STEP, help=f"PGD eps_step. Default: {PGD_STEP}")
    parser.add_argument("--pgd-iterations", type=int, default=PGD_ITERATIONS, help=f"PGD max_iter. Default: {PGD_ITERATIONS}")
    parser.add_argument("--eval-splits", default="test,validation", help="Comma-separated eval splits. Default: test,validation")
    return parser

def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    train_epsilon_list = parse_float_list(args.train_epsilons)
    eval_epsilon_list = parse_float_list(args.eval_epsilons)
    eval_splits = [x.strip().lower() for x in args.eval_splits.split(",") if x.strip()]
    if args.attack_scope is not None:
        args.train_attack_scope = args.attack_scope
        args.eval_attack_scope = args.attack_scope
    run_experiment(
        df=df,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        mlp_results_dir=args.mlp_results_dir,
        train_epsilon_list=train_epsilon_list,
        eval_epsilon_list=eval_epsilon_list,
        pgd_step=args.pgd_step,
        pgd_iterations=args.pgd_iterations,
        random_state=args.random_state,
        attack_mode=args.attack_mode,
        train_attack_scope=args.train_attack_scope,
        eval_attack_scope=args.eval_attack_scope,
        eval_attack_source=args.eval_attack_source,
        eval_splits=eval_splits,
        class_weight_override=args.class_weight,
    )

if __name__ == "__main__":
    main()