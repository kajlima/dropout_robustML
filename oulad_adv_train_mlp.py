# REVISION NOTE (2026-09-29): CGPA, total_credit_enrolled and total_credit_passed were
# removed from every feature list in this script (feature set 32 -> 29, attackable 29 -> 26,
# MLP input 41 -> 38 dims). The columns may still exist in oulad_early.csv but are not used.
# Older comments below that still mention them describe the previous version.
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
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "oulad_early.csv"
DEFAULT_MLP_RESULTS_DIR = "mlp_results_oulad_early"
DEFAULT_OUTPUT_DIR = "mlp_adversarial_training_oulad_early_results"
TARGET_COL = "target"
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
# with the highest recall. Same rule and value as zenodo_mlp_model.py.
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
# FEATURE DEFINITIONS
# ============================================================
NON_FEATURE_COLS = ["id_student", TARGET_COL]
DEFAULT_SELECTED_FEATURES = [
    "module_credits",
    "code_module",
    "code_presentation",
    "credit_enrolled_1styear",
    "credit_passed_1styear",
    "credit_enrolled_2ndyear",
    "credit_passed_2ndyear",
    "credit_enrolled_3rdyear",
    "credit_passed_3rdyear",
    "credit_enrolled_4thyear",
    "GPA_1styear",
    "GPA_2ndyear",
    "GPA_3rdyear",
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
]
# ============================================================
# ATTACKABLE FEATURES
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
DERIVED_FEATURE_COMPONENTS = {}  # CGPA and the credit totals were removed from the feature set
# Per-year (enrolled, passed) pairs (kept for reference; the credit rules are enforced
# in enforce_derived_consistency). Originally used to enforce passed <= enrolled on the
# final adversarial rows. Year 4 has only an 'enrolled' column (passed_4thyear is
# the outcome being predicted), so it has no pair and is not capped. Enforcement
# of these identities is ALWAYS ON (hardcoded, no flag), mirroring the zenodo
# scripts' always-on credit identities.
CREDIT_YEAR_PAIRS = [
    ("credit_enrolled_1styear", "credit_passed_1styear"),
    ("credit_enrolled_2ndyear", "credit_passed_2ndyear"),
    ("credit_enrolled_3rdyear", "credit_passed_3rdyear"),
]
# ============================================================
# DOMAIN CONSTRAINTS (calibrated to oulad_early.csv)
# ============================================================
ROUND_TO_INT_FEATURES = [
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
PERCENTAGE_FEATURES = [
    "early_avg_score_65",
    "early_avg_score_153",
]
NONNEGATIVE_FEATURES = sorted(set([
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
]))
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
    if "id_student" in df.columns:
        student_ids = df["id_student"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    return X, y, student_ids
def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
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
        [os.path.join(mlp_results_dir, "oulad_mlp_model.pt")],
        "MLP model (oulad_mlp_model.pt)",
    )
def resolve_mlp_preprocessor_path(mlp_results_dir):
    return resolve_first_existing(
        [os.path.join(mlp_results_dir, "oulad_mlp_preprocessor.pkl")],
        "MLP preprocessor (oulad_mlp_preprocessor.pkl)",
    )
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
        best_params = obj.get("best_params", {}) or {}
        selected_features = obj.get("selected_features", None)
        feature_set = obj.get("feature_set", None)
        winning_strategy = obj.get("winning_strategy", None)
        state_dict = obj["model_state_dict"]
    else:
        input_dim = input_dim_from_data
        hidden_dims = DEFAULT_HIDDEN_DIMS
        dropout = 0.30
        checkpoint_threshold = None
        best_params = {}
        selected_features = None
        feature_set = None
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
        "state_dict": state_dict,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "best_params": best_params,
        "selected_features": selected_features,
        "feature_set": feature_set,
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
    os.makedirs(output_dir, exist_ok=True)
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(f"Attackable features not in selected_features:\n{missing_attackable}")
    if len(numeric_cols) > input_dim:
        raise ValueError("Number of numeric columns exceeds preprocessed input_dim.")
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
        "round_to_2dp": [int(f in DECIMAL_2DP_FEATURES) for f in selected_features],
        "attack_status": ["attackable" if f in attackable_raw else "frozen" for f in selected_features],
    })
    preprocessed_mask_df = pd.DataFrame(rows)
    raw_mask_df.to_csv(os.path.join(output_dir, "oulad_attack_mask_raw_features.csv"), index=False)
    preprocessed_mask_df.to_csv(os.path.join(output_dir, "oulad_attack_mask_preprocessed_numeric_block.csv"), index=False)
    np.save(os.path.join(output_dir, "oulad_perturbable_mask_preprocessed.npy"), mask)
    if int(mask.sum()) == 0:
        raise ValueError("No preprocessed feature is attackable.")
    return mask, raw_mask_df, preprocessed_mask_df
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
    # enforce derived-feature consistency LAST, on the finalized clipped+rounded
    # components (per-year passed<=enrolled, exact totals, CGPA=mean(GPA)); always on.
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
    X_projected_preprocessed = transform_to_float32(preprocessor, X_adv_raw[selected_features])
    frozen = perturbable_mask_preprocessed == 0
    X_projected_preprocessed[:, frozen] = X_clean_preprocessed[:, frozen]
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
    # mlp_model.py / zenodo rule: among thresholds that keep dropout precision at
    # or above min_precision, pick the highest recall (f1 then precision break
    # ties). If no threshold reaches the precision floor, fall back to best f1.
    rows = []
    for threshold in THRESHOLDS:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_dropout": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "recall_dropout": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
            "f1_dropout": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        })
    threshold_df = pd.DataFrame(rows)

    eligible = threshold_df[threshold_df["precision_dropout"] >= min_precision]
    if len(eligible) > 0:
        # precision floor is reachable: maximize recall inside the eligible set
        best_row = eligible.sort_values(
            ["recall_dropout", "f1_dropout", "precision_dropout"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        # no threshold reaches the precision floor: fall back to best f1
        best_row = threshold_df.sort_values(
            ["f1_dropout", "recall_dropout", "precision_dropout"],
            ascending=[False, False, False],
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
        "id_student": student_ids.values,
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
    history_path = os.path.join(output_dir, f"oulad_mlp_defended_training_history_train_eps_{eps_tag}.csv")
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
    feature_set_hint,
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
    print("MLP ADVERSARIAL TRAINING WITH PGD (oulad_early)")
    print("=" * 80)
    print(f"CSV path           : {csv_path}")
    print(f"Output dir         : {output_dir}")
    print(f"MLP results dir    : {mlp_results_dir}")
    print(f"Attack mode        : {attack_mode}")
    print(f"Train attack scope : {train_attack_scope}")
    print(f"Eval attack scope  : {eval_attack_scope}")
    print(f"Eval attack source : {eval_attack_source}")
    print(f"Derived consist.   : always ON (30-credit grid, 30-120 per year, year4 = 360 - sum(years 1-3), passed <= enrolled; CGPA and credit totals not used)")
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
        checkpoint_feature_set = checkpoint_peek.get("feature_set", None)
    else:
        selected_features = DEFAULT_SELECTED_FEATURES.copy()
        feature_source = "fallback_default"
        checkpoint_feature_set = None
    print(f"Feature list source: {feature_source} ({len(selected_features)} features)")
    if checkpoint_feature_set is not None and feature_set_hint is not None \
            and checkpoint_feature_set != feature_set_hint:
        print(f"WARNING: MLP checkpoint feature_set is '{checkpoint_feature_set}' but "
              f"--feature-set was '{feature_set_hint}'. Using the checkpoint features.")
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
    base_state_dict = checkpoint["state_dict"]
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
    save_confusion_matrix(baseline_clean_test_cm, os.path.join(output_dir, "oulad_mlp_baseline_clean_test_cm.csv"))
    save_confusion_matrix(baseline_clean_val_cm, os.path.join(output_dir, "oulad_mlp_baseline_clean_validation_cm.csv"))
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
        X_train_adv_raw.to_csv(os.path.join(output_dir, f"oulad_X_train_pgd_raw_train_eps_{eps_tag}.csv"), index=False)
        np.save(os.path.join(output_dir, f"oulad_X_train_pgd_preprocessed_train_eps_{eps_tag}.npy"), X_train_adv_pre)
        make_training_sample_csv(
            X_clean_raw=X_train,
            X_adv_raw=X_train_adv_raw,
            y=y_train.values,
            selected_features=selected_features,
            selected_rows=train_attack_idx,
            train_epsilon=train_epsilon,
            path=os.path.join(output_dir, f"oulad_clean_vs_pgd_train_sample_train_eps_{eps_tag}.csv"),
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
        # Train from scratch: fresh random init, NOT warm-started from the baseline
        # weights. Matches the paper's Algorithm 2 ("train a fresh model with
        # hyperparameters H"), so any change vs the baseline comes from the training
        # data rather than from inheriting the baseline's learned weights. The seed is
        # reset here so the random initialization is reproducible run to run.
        # Estimate the fixed epoch budget ONCE via CV inside the mixed (clean+adv)
        # train set, using the defended config. Test/validation are never touched.
        fixed_epochs, _ = estimate_fixed_epochs_via_cv_defended(
            X_train_mixed, y_train_mixed, input_dim, hidden_dims, dropout,
            learning_rate, weight_decay, class_weight, device, random_state,
        )

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
        threshold_df.to_csv(os.path.join(output_dir, f"oulad_mlp_defended_threshold_tuning_train_eps_{eps_tag}.csv"), index=False)
        defended_model_path = os.path.join(output_dir, f"oulad_mlp_defended_model_train_eps_{eps_tag}.pt")
        checkpoint_out = {
            "model_state_dict": defended_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "selected_features": selected_features,
            "feature_set": checkpoint_feature_set,
            "train_epsilon": train_epsilon,
            "fixed_epochs": fixed_epochs,
            "early_stopping": "none (fixed epoch budget, final weights)",
            "epoch_budget_source": "CV inside mixed (clean+adv) train set; test/val untouched",
            "best_threshold": defended_threshold,
            "best_threshold_row": best_threshold_row,
            "training_source_model": model_path,
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
            save_confusion_matrix(cm, os.path.join(output_dir, f"oulad_mlp_defended_clean_{split_name}_cm_train_eps_{eps_tag}.csv"))
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
                path=os.path.join(output_dir, f"oulad_mlp_defended_clean_{split_name}_predictions_train_eps_{eps_tag}.csv"),
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
                X_adv_raw.to_csv(os.path.join(output_dir, f"oulad_X_{split_name}_pgd_raw_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"), index=False)
                np.save(os.path.join(output_dir, f"oulad_X_{split_name}_pgd_preprocessed_train_eps_{eps_tag}_eval_eps_{eval_tag}.npy"), X_adv_pre)
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
                save_confusion_matrix(cm, os.path.join(output_dir, f"oulad_mlp_defended_pgd_{split_name}_cm_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"))
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
                    path=os.path.join(output_dir, f"oulad_mlp_defended_pgd_{split_name}_predictions_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
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
    results_path = os.path.join(output_dir, "oulad_mlp_adversarial_training_results.csv")
    stats_path = os.path.join(output_dir, "oulad_mlp_adversarial_training_perturbation_stats.csv")
    summary_csv_path = os.path.join(output_dir, "oulad_mlp_adversarial_training_summary_by_train_epsilon.csv")
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
        "credit_record_rule": CREDIT_RULE_TEXT,
        "derived_features_note": ("Always on. On the final clipped+rounded rows: per-year "
                                  "credit_passed<=credit_enrolled on the 30-credit grid. CGPA and the credit totals "
                                  "are not model features. "
                                  "Rows stay domain-valid at their final adversarial values."),
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
            "frozen_features": "Restored to clean values after PGD.",
            "non_attacked_rows": "Restored completely to clean values.",
            "raw_value_clip": "Numeric features clipped to train min/max when available.",
            "percentage_features": "avg_score features clipped to [0, 100].",
            "integer_count_features": "Rounded after inverse transform.",
            "nonnegative_features": "Clipped to lower bound 0 (first_submit_day excluded, bounded only by train min/max).",
            "resampling_note": "This script does NOT resample. If the baseline MLP's winning "
                               "strategy used a sampler, pass --class-weight to add imbalance handling.",
        },
        "split": split_report,
        "results_file": results_path,
        "stats_file": stats_path,
        "summary_by_train_epsilon_file": summary_csv_path,
    }
    summary_path_out = os.path.join(output_dir, "oulad_mlp_adversarial_training_summary.json")
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
    parser = argparse.ArgumentParser(description="Adversarial training for the MLP using PGD (oulad_early).")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help=f"Dataset CSV. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--feature-set", default="early_both",
                        choices=["early_65", "early_153", "early_both"],
                        help="Informational only; the real feature list comes from the MLP checkpoint. Default: early_both.")
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
    # Derived-consistency enforcement is ALWAYS ON (hardcoded, no flag), mirroring
    # the always-on credit identities in the zenodo scripts: after every PGD attack
    # the per-year passed<=enrolled cap, the exact credit totals and CGPA=mean(GPA)
    # are re-imposed so the adversarial rows stay domain-valid at their final values.
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
        feature_set_hint=args.feature_set,
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