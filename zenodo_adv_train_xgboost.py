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
from sklearn.base import clone
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
# Adversarial training for XGBoost on zenodo.csv. The FGSM samples are
# generated with the source MLP's gradients (same transfer setup as fgsm_xgboost.py),
# mapped back to raw space, then mixed into the XGBoost's training data. The XGBoost model
# file (zenodo_xgb_model.pkl) is a full imblearn Pipeline: it takes the raw dataframe and
# does its own preprocessing (impute + scale + flags + one-hot) internally.
#
# This is the zenodo counterpart of the oulad_early XGBoost adversarial-training
# script. Same training methodology, scope/mode options, defaults, and the
# separate eval-surrogate ("unseen weapon") logic. Only the dataset-specific
# pieces differ: the feature list, the attackable/frozen sets, the num/flag/one-
# hot preprocessing blocks, the rounding grids, and the derived credit identities.
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_XGB_RESULTS_DIR = "xgb_results_zenodo"
DEFAULT_MLP_DIR = "mlp_results_zenodo"
# Default EVAL surrogate: the MLP-defended checkpoint produced by the zenodo MLP
# adversarial training. Baking this in means a plain run (no flags) evaluates the
# defended tree with FGSM transferred from a surrogate UNSEEN during training
# (training uses the baseline MLP). Override with --eval-mlp-model, or pass
# --eval-mlp-model "" to fall back to the baseline same-weapon eval.
DEFAULT_EVAL_MLP_MODEL = "mlp_adversarial_training_zenodo_results/zenodo_mlp_defended_model_train_eps_1p0.pt"
# None -> reuse the baseline --mlp-dir preprocessor (the MLP adversarial training
# reuses the baseline preprocessor, so this is correct).
DEFAULT_EVAL_MLP_PREPROCESSOR = None
DEFAULT_OUTPUT_DIR = "xgb_adversarial_training_zenodo_results"

TARGET_COL = "target"
ID_COL = "student_id"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# epsilon is an L-inf bound in the MLP's PREPROCESSED space (standardized numeric
# + one-hot). It is NOT in raw units. The final FGSM attack on this dataset used
# 1.0, so both training and evaluation epsilons default to 1.0 here.
TRAIN_EPSILON_LIST = [1.0]
EVAL_EPSILON_LIST = [1.0]

# thresholds swept when re-tuning the defended XGBoost's decision cutoff
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# Minimum dropout precision accepted when picking the defended threshold. Among
# all thresholds that keep dropout precision at or above this floor, take the one
# with the highest recall. Same rule and value as the baseline xgboost_model.py.
MIN_PRECISION = 0.73

# fallback network shape only; checkpoint values override these when present
DEFAULT_HIDDEN_DIMS = (128, 64, 32)

# ============================================================
# FEATURE DEFINITIONS (zenodo.csv)
# ============================================================
# student_id and target are never predictors. academic_year is a one-hot
# categorical (with seniority, degree_size, highest_course_year_enrolled). The 8
# *_flag columns are binary flags. Everything else is numeric. The authoritative
# feature list is read from the MLP checkpoint at runtime.
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
#
# NOTE on internal consistency: total_credits_passed_academic_year and
# total_performance are DERIVED from the credit columns:
#   passed <= enrolled
#   total_performance = passed / enrolled * 100  (0 when enrolled == 0)
# FGSM perturbs every attackable column independently, so after the attack these
# identities are generally broken. With --enforce-derived-consistency ON
# (default) the derived columns are re-imposed after perturbation.
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

# Explicitly frozen (for reporting only). student_id and target are not model
# features so they are not listed here.
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
# Re-imposed after the attack when --enforce-derived-consistency is on (default):
#   1) total_credits_passed_academic_year <= total_credits_enrolled_academic_year
#   2) total_performance = (passed / enrolled) * 100, set to 0 when enrolled == 0
# Both identities hold for 100% of the clean rows.
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
    # snap each value to the nearest .0 or .5
    return np.round(np.asarray(x, dtype=float) * 2.0) / 2.0

def normalize_target(series):
    # same mapping as xgb_model.py / mlp_model.py / fgsm_xgboost.py
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
    # SAME split logic and random_state as the XGBoost / MLP training scripts, so the
    # train/test/validation rows here are exactly the rows those models used.
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
# PATH RESOLUTION AND LOADING
# ============================================================
def resolve_first_existing(candidates, what):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{what} not found. Checked:\n" + "\n".join(candidates))

def resolve_xgb_model_path(xgb_results_dir):
    return resolve_first_existing(
        [os.path.join(xgb_results_dir, "zenodo_xgb_model.pkl")],
        "XGBoost model (zenodo_xgb_model.pkl)",
    )

def resolve_xgb_summary_path(xgb_results_dir):
    path = os.path.join(xgb_results_dir, "zenodo_xgb_summary.json")
    return path if os.path.exists(path) else None

def load_xgb_threshold(summary_path, fallback=0.5):
    # reuse the RF's tuned decision threshold; fall back to 0.5 if the summary is
    # missing or has no threshold. Matches fgsm_xgboost.py's threshold source.
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

def load_xgb_best_params_for_report(summary_path):
    # read the tuned XGB params only for the summary JSON. The defended XGBoost itself
    # is a clone of the baseline pipeline, so it already carries these params.
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        params = summary.get("selection", {}).get("best_xgb_params")
        strategy = summary.get("selection", {}).get("winning_strategy") or summary.get("winning_strategy")
        if isinstance(params, dict):
            return params, strategy, summary_path
    return None, None, "not_found"

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
    # Build the ART mask in the MLP's preprocessed space. The zenodo
    # ColumnTransformer emits columns in the order:
    #   numeric block -> flag block -> categorical one-hot block.
    # Every attackable feature is numeric, so only numeric columns whose raw name
    # is attackable may move; the flag and one-hot blocks stay frozen.
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
            f"(they were routed to flags/one-hot): {attackable_not_numeric}"
        )

    mask = np.zeros(input_dim, dtype=np.float32)
    rows = []
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
    enforce_consistency=True,
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
    art_mlp,
    X_clean_raw,
    X_clean_preprocessed,
    y_true,
    clean_pred,
    epsilon,
    attack_mode,
    attack_scope,
    perturbable_mask,
    preprocessor,
    selected_features,
    train_min,
    train_max,
    enforce_consistency=True,
    batch_size=128,
):
    # run FGSM only on the selected rows; untouched rows stay clean, then map
    # everything back to raw space with the domain constraints applied
    attack_idx = make_attack_scope_mask(attack_scope, y_true, clean_pred)
    X_adv_preprocessed = X_clean_preprocessed.copy()
    if int(attack_idx.sum()) > 0:
        attack_y, targeted = make_attack_labels(attack_mode, np.asarray(y_true)[attack_idx])
        fgsm = FastGradientMethod(
            estimator=art_mlp,
            eps=float(epsilon),
            targeted=targeted,
            batch_size=batch_size,
        )
        X_adv_subset = fgsm.generate(
            x=X_clean_preprocessed[attack_idx],
            y=attack_y,
            mask=perturbable_mask,
        )
        frozen = perturbable_mask == 0
        X_adv_subset[:, frozen] = X_clean_preprocessed[attack_idx][:, frozen]
        X_adv_preprocessed[attack_idx] = X_adv_subset

    X_adv_raw, X_adv_projected = preprocessed_adv_to_raw_and_scaled(
        X_adv_preprocessed=X_adv_preprocessed,
        X_clean_preprocessed=X_clean_preprocessed,
        X_clean_raw=X_clean_raw,
        selected_features=selected_features,
        preprocessor=preprocessor,
        train_min=train_min,
        train_max=train_max,
        perturbable_mask_preprocessed=perturbable_mask,
        attacked_row_mask=attack_idx,
        enforce_consistency=enforce_consistency,
    )
    return X_adv_raw, X_adv_projected, attack_idx

# ============================================================
# EVALUATION AND SAVING
# ============================================================
def tune_threshold_from_proba(y_true, y_proba, min_precision=MIN_PRECISION):
    # Defended-model threshold selection on the clean TEST set. Matches the
    # baseline xgboost_model.py rule: sweep thresholds, then among the ones that keep
    # dropout precision at or above min_precision, pick the highest recall (f1
    # then precision break ties). If no threshold reaches the precision floor,
    # fall back to best f1.
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

def evaluate_xgb_with_threshold(model, X_raw, y, condition, train_epsilon, eval_epsilon, split, threshold):
    # the XGBoost pkl is a full imblearn pipeline (preprocessor + model), so it takes
    # the RAW dataframe and preprocesses internally
    y_true = np.asarray(y).astype(int)
    y_proba = model.predict_proba(X_raw)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "model": "XGBoost",
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
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)

def save_predictions(student_ids, y_true, y_pred, y_proba, threshold, condition, train_epsilon, eval_epsilon, path):
    y_true = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        ID_COL: student_ids.values,
        "condition": condition,
        "train_epsilon": train_epsilon,
        "eval_epsilon": eval_epsilon,
        "actual_label": y_true,
        "actual_target": np.where(y_true == 1, "dropout", "non-dropout"),
        "predicted_label": y_pred,
        "predicted_target": np.where(y_pred == 1, "dropout", "non-dropout"),
        "prob_dropout": y_proba,
        "threshold": threshold,
    })
    pred_df.to_csv(path, index=False)

def save_training_sample_csv(X_clean_raw, X_adv_raw, y, selected_features, attack_idx, train_epsilon, path, max_samples=50):
    # per-feature clean vs adversarial values for a handful of attacked rows.
    # academic_year (categorical, frozen) has its diff coerced to numeric and
    # left blank when it cannot be computed.
    attack_idx = np.asarray(attack_idx).astype(bool)
    chosen = np.where(attack_idx)[0][:max_samples]
    rows = []
    X_clean_raw = X_clean_raw.reset_index(drop=True)
    X_adv_raw = X_adv_raw.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    for i in chosen:
        row = {
            "sample_id": int(i),
            "label": int(y[i]),
            "label_text": "dropout" if y[i] == 1 else "non-dropout",
            "train_epsilon": float(train_epsilon),
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
# XGBOOST TRAINING
# ============================================================
def train_defended_xgb(xgb_baseline, X_train_mixed, y_train_mixed):
    # clone the baseline XGBoost pipeline so the defended model keeps the exact same
    # preprocessing, sampler, and tuned hyperparameters, then refit it on the
    # clean + adversarial mixed data. NOTE: if the baseline pipeline contains a
    # resampler (SMOTE / over / under), that resampler runs on the mixed set, so
    # the balancing is applied on top of the adversarial augmentation.
    xgb_defended = clone(xgb_baseline)
    xgb_defended.fit(X_train_mixed, y_train_mixed)
    return xgb_defended

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_single_adversarial_training_experiment(
    df,
    csv_path,
    output_dir,
    xgb_results_dir,
    mlp_dir,
    train_epsilon_list,
    eval_epsilon_list,
    random_state,
    attack_mode,
    train_attack_scope,
    eval_attack_scope,
    enforce_consistency=True,
    eval_mlp_model_path_arg=None,
    eval_mlp_preprocessor_arg=None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("XGBOOST ADVERSARIAL TRAINING WITH FGSM VIA SOURCE MLP (zenodo)")
    print("=" * 80)
    print(f"CSV path           : {csv_path}")
    print(f"Output dir         : {output_dir}")
    print(f"XGB results dir     : {xgb_results_dir}")
    print(f"MLP dir            : {mlp_dir}")
    print(f"Attack mode        : {attack_mode}")
    print(f"Train attack scope : {train_attack_scope}")
    print(f"Eval attack scope  : {eval_attack_scope}")
    print(f"Enforce identities : {enforce_consistency}")
    print(f"Device             : {device}")
    print(f"Dataset shape      : {df.shape}")
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

    # load baseline XGBoost pipeline and its tuned threshold
    xgb_baseline_model_path = resolve_xgb_model_path(xgb_results_dir)
    xgb_summary_path = resolve_xgb_summary_path(xgb_results_dir)
    xgb_baseline = joblib.load(xgb_baseline_model_path)
    xgb_baseline_threshold, xgb_threshold_source = load_xgb_threshold(xgb_summary_path)
    xgb_params_report, xgb_winning_strategy, xgb_params_source = load_xgb_best_params_for_report(xgb_summary_path)

    # load source MLP preprocessor and build the torch model
    preprocessor = joblib.load(mlp_preprocessor_path)
    X_train_preprocessed = transform_to_float32(preprocessor, X_train)
    X_test_preprocessed = transform_to_float32(preprocessor, X_test)
    X_val_preprocessed = transform_to_float32(preprocessor, X_val)
    input_dim = X_train_preprocessed.shape[1]
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
    train_min = X_train[selected_features].min(numeric_only=True)
    train_max = X_train[selected_features].max(numeric_only=True)

    # ------------------------------------------------------------------
    # EVAL SURROGATE (optional, separate from the training surrogate)
    # ------------------------------------------------------------------
    # By default the evaluation FGSM reuses the SAME baseline MLP that produced
    # the training adversarial samples, so the defended model is tested against
    # the very weapon it was hardened on. If --eval-mlp-model is given (point it
    # at the MLP-defended checkpoint), the evaluation FGSM is generated from that
    # separate surrogate instead: an "unseen weapon" test. This stays a TRANSFER
    # attack - trees have no gradients, so the gradient is always borrowed from
    # some MLP; only which MLP changes. The MLP adversarial training reuses the
    # baseline preprocessor, so --eval-mlp-preprocessor defaults to the training
    # surrogate's preprocessor and only the network weights differ.
    if eval_mlp_model_path_arg:
        if not os.path.exists(eval_mlp_model_path_arg):
            raise FileNotFoundError(
                f"Eval surrogate checkpoint not found: {eval_mlp_model_path_arg}. "
                "Run the MLP adversarial training first (it produces "
                "zenodo_mlp_defended_model_train_eps_1p0.pt), or pass --eval-mlp-model with the "
                "correct path. To use the baseline same-weapon eval instead, pass "
                "--eval-mlp-model \"\" (empty)."
            )
        eval_attack_source = os.path.basename(eval_mlp_model_path_arg)
        eval_mlp_model_path = eval_mlp_model_path_arg
        eval_mlp_preprocessor_path = eval_mlp_preprocessor_arg or mlp_preprocessor_path
        eval_mlp_info = read_mlp_checkpoint(eval_mlp_model_path, device)
        eval_preprocessor = joblib.load(eval_mlp_preprocessor_path)
        X_test_preprocessed_eval = transform_to_float32(eval_preprocessor, X_test)
        X_val_preprocessed_eval = transform_to_float32(eval_preprocessor, X_val)
        eval_input_dim = X_test_preprocessed_eval.shape[1]
        if eval_mlp_info["input_dim"] is not None and int(eval_mlp_info["input_dim"]) != int(eval_input_dim):
            raise ValueError(
                f"Eval preprocessed input_dim ({eval_input_dim}) != eval MLP checkpoint "
                f"input_dim ({eval_mlp_info['input_dim']}). The eval preprocessor and "
                "checkpoint do not match. Check --eval-mlp-model / --eval-mlp-preprocessor."
            )
        if eval_mlp_info["selected_features"] is not None \
                and list(eval_mlp_info["selected_features"]) != selected_features:
            raise ValueError(
                "The eval MLP surrogate was trained on a different feature list than "
                "the training surrogate. They must match for the transfer eval to be valid."
            )
        eval_mlp_model = build_mlp_from_state(eval_mlp_info, input_dim=eval_input_dim, device=device)
        eval_art_mlp = wrap_mlp_for_art(eval_mlp_model, input_dim=eval_input_dim)
        eval_perturbable_mask, _, _ = make_preprocessed_attack_mask(
            preprocessor=eval_preprocessor,
            selected_features=selected_features,
            input_dim=eval_input_dim,
            output_dir=os.path.join(output_dir, "eval_surrogate_mask"),
        )
        print("Eval FGSM surrogate (SEPARATE from training):")
        print(f"  checkpoint  : {eval_mlp_model_path}")
        print(f"  preprocessor: {eval_mlp_preprocessor_path}")
        print(f"  hidden dims : {eval_mlp_info['hidden_dims']} | dropout: {eval_mlp_info['dropout']}")
        print(f"  eval_attack_source label: {eval_attack_source}")
        print()
    else:
        eval_attack_source = "baseline_mlp"
        eval_mlp_model_path = mlp_model_path
        eval_mlp_preprocessor_path = mlp_preprocessor_path
        eval_mlp_info = mlp_info
        eval_art_mlp = art_mlp
        eval_preprocessor = preprocessor
        eval_perturbable_mask = perturbable_mask
        X_test_preprocessed_eval = X_test_preprocessed
        X_val_preprocessed_eval = X_val_preprocessed
        print("Eval FGSM surrogate: same as training (baseline MLP).")
        print()

    print("Loaded baseline XGBoost:")
    print(xgb_baseline_model_path)
    print(f"Baseline XGBoost threshold: {xgb_baseline_threshold} | source: {xgb_threshold_source}")
    print(f"XGB params source     : {xgb_params_source}")
    print(f"XGB winning strategy  : {xgb_winning_strategy}")
    print(f"XGB params            : {xgb_params_report}")
    print()
    print("Loaded source MLP:")
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

    # baseline clean predictions (used to generate the training attack samples)
    baseline_train_clean, baseline_train_cm, baseline_train_pred, baseline_train_proba = evaluate_xgb_with_threshold(
        xgb_baseline, X_train, y_train.values, "Baseline Clean", np.nan, 0.0, "train", xgb_baseline_threshold
    )
    baseline_test_clean, baseline_test_cm, baseline_test_pred, baseline_test_proba = evaluate_xgb_with_threshold(
        xgb_baseline, X_test, y_test.values, "Baseline Clean", np.nan, 0.0, "test", xgb_baseline_threshold
    )
    baseline_val_clean, baseline_val_cm, baseline_val_pred, baseline_val_proba = evaluate_xgb_with_threshold(
        xgb_baseline, X_val, y_val.values, "Baseline Clean", np.nan, 0.0, "validation", xgb_baseline_threshold
    )
    all_results = [baseline_train_clean, baseline_test_clean, baseline_val_clean]
    all_stats = []
    all_summary = []
    print("Baseline clean test result:")
    print(baseline_test_clean)
    print(baseline_test_cm)
    print()
    save_confusion_matrix(baseline_test_cm, os.path.join(output_dir, "zenodo_xgb_baseline_clean_test_cm.csv"))
    save_confusion_matrix(baseline_val_cm, os.path.join(output_dir, "zenodo_xgb_baseline_clean_validation_cm.csv"))

    for train_epsilon in train_epsilon_list:
        print("=" * 80)
        print(f"TRAIN DEFENDED XGBOOST | train epsilon = {train_epsilon}")
        print("=" * 80)
        X_train_adv_raw, X_train_adv_preprocessed, train_attack_idx = generate_fgsm_for_scope(
            art_mlp=art_mlp,
            X_clean_raw=X_train,
            X_clean_preprocessed=X_train_preprocessed,
            y_true=y_train.values,
            clean_pred=baseline_train_pred,
            epsilon=train_epsilon,
            attack_mode=attack_mode,
            attack_scope=train_attack_scope,
            perturbable_mask=perturbable_mask,
            preprocessor=preprocessor,
            selected_features=selected_features,
            train_min=train_min,
            train_max=train_max,
            enforce_consistency=enforce_consistency,
        )
        train_stats = perturbation_stats(
            X_clean_preprocessed=X_train_preprocessed,
            X_adv_preprocessed=X_train_adv_preprocessed,
            perturbable_mask=perturbable_mask,
            attacked_row_mask=train_attack_idx,
        )
        train_stats.update({
            "train_epsilon": float(train_epsilon),
            "eval_epsilon": float(train_epsilon),
            "split": "train_adversarial_generation",
            "attack_scope": train_attack_scope,
            "train_attack_scope": train_attack_scope,
        })
        all_stats.append(train_stats)
        if train_stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
            raise AssertionError("Frozen features changed in adversarial training data.")

        eps_tag = str(train_epsilon).replace(".", "p")
        X_train_adv_raw.to_csv(os.path.join(output_dir, f"zenodo_X_train_fgsm_raw_train_eps_{eps_tag}.csv"), index=False)
        np.save(os.path.join(output_dir, f"zenodo_X_train_fgsm_preprocessed_train_eps_{eps_tag}.npy"), X_train_adv_preprocessed)
        save_training_sample_csv(
            X_clean_raw=X_train,
            X_adv_raw=X_train_adv_raw,
            y=y_train.values,
            selected_features=selected_features,
            attack_idx=train_attack_idx,
            train_epsilon=train_epsilon,
            path=os.path.join(output_dir, f"zenodo_clean_vs_fgsm_train_sample_train_eps_{eps_tag}.csv"),
            max_samples=50,
        )

        # clean training data + adversarial twins for the attacked rows only.
        # With train_attack_scope=dropout_only, the added rows are the FGSM'd
        # dropout rows: this is the "adversarial data used" in the thesis table.
        X_adv_train_only = X_train_adv_raw.loc[train_attack_idx, selected_features].reset_index(drop=True)
        y_adv_train_only = y_train.values.astype(int)[train_attack_idx]
        X_train_mixed = pd.concat([
            X_train[selected_features].reset_index(drop=True),
            X_adv_train_only,
        ], axis=0, ignore_index=True)
        y_train_mixed = np.concatenate([y_train.values.astype(int), y_adv_train_only])
        # shuffle the mixed set
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X_train_mixed))
        X_train_mixed = X_train_mixed.iloc[idx].reset_index(drop=True)
        y_train_mixed = y_train_mixed[idx]

        print(f"Clean train rows              : {len(X_train)}")
        print(f"Adversarial train rows added  : {len(X_adv_train_only)}")
        print(f"Mixed train rows              : {len(X_train_mixed)}")
        print()

        xgb_defended = train_defended_xgb(
            xgb_baseline=xgb_baseline,
            X_train_mixed=X_train_mixed,
            y_train_mixed=y_train_mixed,
        )

        # pick the defended threshold on the clean test set
        test_clean_proba_for_threshold = xgb_defended.predict_proba(X_test)[:, 1]
        defended_threshold, threshold_df, best_threshold_row = tune_threshold_from_proba(
            y_true=y_test.values,
            y_proba=test_clean_proba_for_threshold,
        )
        threshold_df.to_csv(os.path.join(output_dir, f"zenodo_xgb_defended_threshold_tuning_train_eps_{eps_tag}.csv"), index=False)

        model_path = os.path.join(output_dir, f"zenodo_xgb_defended_model_train_eps_{eps_tag}.pkl")
        joblib.dump(xgb_defended, model_path)
        print(f"Saved defended XGBoost model: {model_path}")
        print(f"Defended XGBoost threshold  : {defended_threshold}")
        print()

        defended_clean_preds = {}
        defended_clean_probas = {}
        for split_name, X_split, y_split, id_split in [
            ("train", X_train, y_train, id_train),
            ("test", X_test, y_test, id_test),
            ("validation", X_val, y_val, id_val),
        ]:
            result, cm, pred, proba = evaluate_xgb_with_threshold(
                model=xgb_defended,
                X_raw=X_split,
                y=y_split.values,
                condition="Defended Clean",
                train_epsilon=float(train_epsilon),
                eval_epsilon=0.0,
                split=split_name,
                threshold=defended_threshold,
            )
            all_results.append(result)
            defended_clean_preds[split_name] = pred
            defended_clean_probas[split_name] = proba
            save_confusion_matrix(cm, os.path.join(output_dir, f"zenodo_xgb_defended_clean_{split_name}_cm_train_eps_{eps_tag}.csv"))
            save_predictions(
                student_ids=id_split,
                y_true=y_split.values,
                y_pred=pred,
                y_proba=proba,
                threshold=defended_threshold,
                condition="Defended Clean",
                train_epsilon=float(train_epsilon),
                eval_epsilon=0.0,
                path=os.path.join(output_dir, f"zenodo_xgb_defended_clean_{split_name}_predictions_train_eps_{eps_tag}.csv"),
            )
            print(f"Defended clean {split_name}:")
            print(result)
            print(cm)
            print()

        for eval_epsilon in eval_epsilon_list:
            eval_tag = str(eval_epsilon).replace(".", "p")
            print("-" * 80)
            print(f"EVALUATE DEFENDED XGBOOST | train eps = {train_epsilon} | eval eps = {eval_epsilon}")
            print("-" * 80)
            for split_name, X_split, X_split_pre, y_split, id_split in [
                ("test", X_test, X_test_preprocessed_eval, y_test, id_test),
                ("validation", X_val, X_val_preprocessed_eval, y_val, id_val),
            ]:
                clean_pred_for_scope = defended_clean_preds[split_name]
                X_adv_raw, X_adv_preprocessed, eval_attack_idx = generate_fgsm_for_scope(
                    art_mlp=eval_art_mlp,
                    X_clean_raw=X_split,
                    X_clean_preprocessed=X_split_pre,
                    y_true=y_split.values,
                    clean_pred=clean_pred_for_scope,
                    epsilon=eval_epsilon,
                    attack_mode=attack_mode,
                    attack_scope=eval_attack_scope,
                    perturbable_mask=eval_perturbable_mask,
                    preprocessor=eval_preprocessor,
                    selected_features=selected_features,
                    train_min=train_min,
                    train_max=train_max,
                    enforce_consistency=enforce_consistency,
                )
                stats = perturbation_stats(
                    X_clean_preprocessed=X_split_pre,
                    X_adv_preprocessed=X_adv_preprocessed,
                    perturbable_mask=eval_perturbable_mask,
                    attacked_row_mask=eval_attack_idx,
                )
                stats.update({
                    "train_epsilon": float(train_epsilon),
                    "eval_epsilon": float(eval_epsilon),
                    "split": split_name,
                    "attack_scope": eval_attack_scope,
                    "train_attack_scope": train_attack_scope,
                    "eval_attack_scope": eval_attack_scope,
                    "eval_attack_source": eval_attack_source,
                })
                all_stats.append(stats)
                if stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
                    raise AssertionError(f"Frozen features changed in {split_name} eval data.")

                X_adv_raw.to_csv(
                    os.path.join(output_dir, f"zenodo_X_{split_name}_fgsm_raw_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                    index=False,
                )
                result, cm, pred, proba = evaluate_xgb_with_threshold(
                    model=xgb_defended,
                    X_raw=X_adv_raw,
                    y=y_split.values,
                    condition=f"Defended FGSM {attack_mode}",
                    train_epsilon=float(train_epsilon),
                    eval_epsilon=float(eval_epsilon),
                    split=split_name,
                    threshold=defended_threshold,
                )
                result["attack_scope"] = eval_attack_scope
                result["train_attack_scope"] = train_attack_scope
                result["eval_attack_scope"] = eval_attack_scope
                result["eval_attack_source"] = eval_attack_source
                result["attacked_rows_count"] = int(eval_attack_idx.sum())
                result["attacked_dropout_rows_count"] = int(np.sum(eval_attack_idx & (y_split.values == POSITIVE_LABEL)))
                result["dropout_to_nondropout_flips"] = int(np.sum((y_split.values == 1) & (clean_pred_for_scope == 1) & (pred == 0)))
                result["nondropout_to_dropout_flips"] = int(np.sum((y_split.values == 0) & (clean_pred_for_scope == 0) & (pred == 1)))
                result["prediction_changed_count"] = int(np.sum(clean_pred_for_scope != pred))
                all_results.append(result)
                save_confusion_matrix(
                    cm,
                    os.path.join(output_dir, f"zenodo_xgb_defended_fgsm_{split_name}_cm_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                )
                save_predictions(
                    student_ids=id_split,
                    y_true=y_split.values,
                    y_pred=pred,
                    y_proba=proba,
                    threshold=defended_threshold,
                    condition=f"Defended FGSM {attack_mode}",
                    train_epsilon=float(train_epsilon),
                    eval_epsilon=float(eval_epsilon),
                    path=os.path.join(output_dir, f"zenodo_xgb_defended_fgsm_{split_name}_predictions_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
                )
                print(f"Defended FGSM {split_name}:")
                print(result)
                print(cm)
                print()

        all_summary.append({
            "model": "XGBoost Defended",
            "defence": "Adversarial training with FGSM samples generated via source MLP",
            "attack_mode": attack_mode,
            "train_attack_scope": train_attack_scope,
            "eval_attack_scope": eval_attack_scope,
            "eval_attack_source": eval_attack_source,
            "enforce_derived_consistency": enforce_consistency,
            "train_epsilon": float(train_epsilon),
            "defended_threshold": float(defended_threshold),
            "best_threshold_row": best_threshold_row,
            "clean_train_rows": int(len(X_train)),
            "fgsm_train_rows_added": int(len(X_adv_train_only)),
            "mixed_train_rows": int(len(X_train_mixed)),
            "model_path": model_path,
        })

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)
    summary_df = pd.DataFrame(all_summary)
    results_path = os.path.join(output_dir, "zenodo_xgb_adversarial_training_results.csv")
    stats_path = os.path.join(output_dir, "zenodo_xgb_adversarial_training_perturbation_stats.csv")
    summary_csv_path = os.path.join(output_dir, "zenodo_xgb_adversarial_training_summary_by_train_epsilon.csv")
    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "defence": "XGBoost adversarial training",
        "training_method": (
            f"Clean training data plus FGSM adversarial samples generated from "
            f"train_attack_scope={train_attack_scope} (attacked rows only)."
        ),
        "target_model": "XGBoost Defended",
        "gradient_model": "Source MLP trained on true labels",
        "attack_used_for_training": "FGSM",
        "defended_threshold_rule": (f"On the clean test split, pick the threshold with the highest dropout "
                                    f"recall subject to dropout precision >= {MIN_PRECISION}; if no threshold "
                                    f"meets the floor, fall back to the F1-optimal threshold. Same rule as "
                                    f"the baseline xgboost_model.py."),
        "min_precision": MIN_PRECISION,
        "attack_mode": attack_mode,
        "train_attack_scope": train_attack_scope,
        "eval_attack_scope": eval_attack_scope,
        "enforce_derived_consistency": enforce_consistency,
        "epsilon_space": "L-inf bound in the MLP preprocessed/standardized feature space",
        "train_epsilon_list": train_epsilon_list,
        "eval_epsilon_list": eval_epsilon_list,
        "xgb_baseline_model_path": xgb_baseline_model_path,
        "xgb_baseline_threshold": xgb_baseline_threshold,
        "xgb_baseline_threshold_source": xgb_threshold_source,
        "xgb_params": xgb_params_report,
        "xgb_params_source": xgb_params_source,
        "xgb_baseline_winning_strategy": xgb_winning_strategy,
        "mlp_model_path": mlp_model_path,
        "mlp_preprocessor_path": mlp_preprocessor_path,
        "mlp_hidden_dims": list(mlp_info["hidden_dims"]),
        "mlp_dropout": mlp_info["dropout"],
        "eval_attack_source": eval_attack_source,
        "eval_mlp_model_path": eval_mlp_model_path,
        "eval_mlp_preprocessor_path": eval_mlp_preprocessor_path,
        "eval_mlp_hidden_dims": list(eval_mlp_info["hidden_dims"]),
        "eval_mlp_dropout": eval_mlp_info["dropout"],
        "feature_list_source": feature_source,
        "n_raw_features": len(selected_features),
        "n_preprocessed_features": int(input_dim),
        "n_attackable_preprocessed_features": int(perturbable_mask.sum()),
        "n_frozen_preprocessed_features": int(len(perturbable_mask) - perturbable_mask.sum()),
        "attackable_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 1, "feature"].tolist(),
        "frozen_raw_features": raw_mask_df.loc[raw_mask_df["raw_attackable"] == 0, "feature"].tolist(),
        "derived_features_note": (
            "total_credits_passed_academic_year and total_performance are derived "
            "from the credit columns. With enforce_derived_consistency off FGSM "
            "breaks those identities; with it on (default) passed is capped at "
            "enrolled and total_performance is recomputed as passed/enrolled*100."
        ),
        "constraints": {
            "frozen_features": ("Restored to clean values after FGSM (academic_year, n_courses, "
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
            "resampler_note": ("If the baseline XGBoost pipeline contains a resampler, it runs on the "
                               "mixed (clean + adversarial) training set."),
        },
        "split": split_report,
        "results_file": results_path,
        "stats_file": stats_path,
        "summary_by_train_epsilon_file": summary_csv_path,
    }
    summary_path = os.path.join(output_dir, "zenodo_xgb_adversarial_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("XGBOOST ADVERSARIAL TRAINING DONE")
    print("=" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Stats saved to  : {stats_path}")
    print(f"Summary saved to: {summary_path}")
    return summary

# ============================================================
# ARGUMENTS
# ============================================================
def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def format_float_list(values):
    return ",".join(str(float(v)) for v in values)

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Adversarial training for XGBoost using FGSM via source MLP (zenodo)."
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH,
                        help=f"Dataset CSV. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--xgb-results-dir", default=DEFAULT_XGB_RESULTS_DIR,
                        help=f"Baseline XGBoost results folder. Default: {DEFAULT_XGB_RESULTS_DIR}")
    parser.add_argument("--mlp-dir", default=DEFAULT_MLP_DIR,
                        help=f"Source MLP results folder. Default: {DEFAULT_MLP_DIR}")
    parser.add_argument(
        "--eval-mlp-model", default=DEFAULT_EVAL_MLP_MODEL,
        help=("Path to a SEPARATE MLP checkpoint (.pt) used ONLY to generate the "
              "evaluation FGSM examples (an unseen surrogate relative to training, which "
              "uses --mlp-dir). Defaults to the MLP-defended checkpoint, so a plain run "
              "already does the unseen-surrogate eval. Pass --eval-mlp-model \"\" to "
              f"fall back to the baseline same-weapon eval. Default: {DEFAULT_EVAL_MLP_MODEL}"),
    )
    parser.add_argument(
        "--eval-mlp-preprocessor", default=DEFAULT_EVAL_MLP_PREPROCESSOR,
        help=("Preprocessor (.pkl) matching --eval-mlp-model. The MLP adversarial training "
              "reuses the baseline preprocessor, so this can usually be omitted; it then "
              "defaults to the --mlp-dir preprocessor."),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Defended XGBoost output folder. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE,
                        help=f"Random state. Default: {DEFAULT_RANDOM_STATE}")
    parser.add_argument(
        "--attack-mode", default="target_non_dropout",
        choices=["untargeted", "target_non_dropout"],
        help=("untargeted = general degradation. "
              "target_non_dropout = targeted push toward non-dropout."),
    )
    parser.add_argument(
        "--train-attack-scope", default="dropout_only",
        choices=["all", "dropout_only", "true_positive_only"],
        help="Scope used to generate FGSM samples for adversarial training. Default: dropout_only.",
    )
    parser.add_argument(
        "--eval-attack-scope", default="true_positive_only",
        choices=["all", "dropout_only", "true_positive_only"],
        help="Scope used for the final FGSM evaluation. Default: true_positive_only.",
    )
    parser.add_argument(
        "--attack-scope", default=None,
        choices=["all", "dropout_only", "true_positive_only"],
        help="Backward-compatible alias. If set, applies to both train and eval scope.",
    )
    # Consistency enforcement is ON by default: passed<=enrolled and
    # total_performance=passed/enrolled*100 are re-imposed after each FGSM attack
    # (both the training-sample generation and the evaluation). Pass
    # --no-enforce-derived-consistency to turn it off and let FGSM move those
    # derived columns freely.
    parser.add_argument(
        "--no-enforce-derived-consistency", dest="enforce_derived_consistency",
        action="store_false", default=True,
        help=("Disable consistency enforcement. By default the credit identities are "
              "re-imposed after the attack; this flag lets FGSM perturb those derived "
              "columns freely instead."),
    )
    parser.add_argument(
        "--train-epsilons", default=format_float_list(TRAIN_EPSILON_LIST),
        help=f"Comma-separated train epsilon list. Default: {format_float_list(TRAIN_EPSILON_LIST)}",
    )
    parser.add_argument(
        "--eval-epsilons", default=format_float_list(EVAL_EPSILON_LIST),
        help=f"Comma-separated eval epsilon list. Default: {format_float_list(EVAL_EPSILON_LIST)}",
    )
    return parser

def main():
    args = build_arg_parser().parse_args()
    set_global_seed(args.random_state)
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    train_epsilon_list = parse_float_list(args.train_epsilons)
    eval_epsilon_list = parse_float_list(args.eval_epsilons)
    # --attack-scope overrides both train and eval scope when given
    if args.attack_scope is not None:
        args.train_attack_scope = args.attack_scope
        args.eval_attack_scope = args.attack_scope
    run_single_adversarial_training_experiment(
        df=df,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        xgb_results_dir=args.xgb_results_dir,
        mlp_dir=args.mlp_dir,
        train_epsilon_list=train_epsilon_list,
        eval_epsilon_list=eval_epsilon_list,
        random_state=args.random_state,
        attack_mode=args.attack_mode,
        train_attack_scope=args.train_attack_scope,
        eval_attack_scope=args.eval_attack_scope,
        enforce_consistency=args.enforce_derived_consistency,
        eval_mlp_model_path_arg=args.eval_mlp_model,
        eval_mlp_preprocessor_arg=args.eval_mlp_preprocessor,
    )

if __name__ == "__main__":
    main()