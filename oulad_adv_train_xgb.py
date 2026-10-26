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
# Adversarial training for XGBoost on oulad_early.csv. The FGSM samples
# are generated with the source MLP's gradients (same transfer setup as
# fgsm_xgboost.py), mapped back to raw space, then mixed into the XGBoost's training data.
# The XGBoost model file (oulad_xgb_model.pkl) is a full imblearn Pipeline: it takes the
# raw dataframe and does its own preprocessing (median impute + one-hot) internally.
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "oulad_early.csv"
DEFAULT_XGB_RESULTS_DIR = "xgb_results_oulad_early"
DEFAULT_MLP_DIR = "mlp_results_oulad_early"
# Default EVAL surrogate: the MLP-defended checkpoint. Baking this in as the
# default means a plain run (no flags) evaluates the defended tree model with
# FGSM transferred from the MLP-defended surrogate - a weapon UNSEEN during
# training (training used the baseline MLP). Override with --eval-mlp-model, or
# pass --eval-mlp-model "" to fall back to the baseline same-weapon eval.
DEFAULT_EVAL_MLP_MODEL = "mlp_adversarial_training_oulad_early_results/oulad_mlp_defended_model_train_eps_1p0.pt"
# None -> reuse the baseline --mlp-dir preprocessor (the MLP adversarial
# training does not save a separate preprocessor, so this is correct).
DEFAULT_EVAL_MLP_PREPROCESSOR = None
DEFAULT_OUTPUT_DIR = "xgb_adversarial_training_oulad_early_results"
TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout
# epsilon is an L-inf bound in the MLP's PREPROCESSED space (standardized numeric
# + one-hot). It is NOT in raw units. The final FGSM attack on this dataset used
# 1.0, so both the training and evaluation epsilons default to 1.0 here.
TRAIN_EPSILON_LIST = [1.0]
EVAL_EPSILON_LIST = [1.0]
# thresholds swept when re-tuning the defended XGBoost's decision cutoff
THRESHOLDS = np.arange(0.05, 0.96, 0.01)
# fallback network shape only; checkpoint values override these when present
DEFAULT_HIDDEN_DIMS = (128, 64, 32)
# ============================================================
# FEATURE DEFINITIONS (identical set to fgsm_xgboost.py)
# ============================================================
# id_student and target are never predictors. code_module / code_presentation are
# one-hot categoricals. There are no binary flag columns in this dataset version.
NON_FEATURE_COLS = ["id_student", TARGET_COL]
# Fallback feature list, used ONLY if the MLP checkpoint does not store
# selected_features. This is the 'early_both' feature set as defined in
# xgboost_model.py / mlp_model.py for oulad_early.csv (static/enrolment + day-65 window
# + day-153 window). num_of_prev_attempts and studied_credits do NOT exist in this
# dataset, so they are not listed. module_credits DOES exist and stays frozen.
# The authoritative list is read from the MLP checkpoint at runtime.
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
# ATTACKABLE FEATURES
# ============================================================
# Only these raw features may move. Everything else stays frozen, including the
# one-hot module/presentation columns and module_credits.
#
# NOTE on internal consistency: CGPA, total_credit_enrolled and
# total_credit_passed are DERIVED from other attackable columns in the raw data:
#   CGPA                  = mean(GPA_1styear, GPA_2ndyear, GPA_3rdyear)
#   total_credit_enrolled = credit_enrolled_1styear + 2ndyear + 3rdyear
#   total_credit_passed   = credit_passed_1styear   + 2ndyear + 3rdyear
# FGSM perturbs every attackable column independently, so after the attack these
# identities are generally broken. Domain consistency is enforced ALWAYS
# (hardcoded ON, to mirror the zenodo adversarial-training script): the per-year
# credit cap (passed <= enrolled), the two credit totals and CGPA are recomputed
# from their (clipped, rounded) components after perturbation.
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
# Derived columns and how to recompute them from their components. Enforced
# ALWAYS (hardcoded ON) by enforce_derived_consistency, after clip/round.
DERIVED_FEATURE_COMPONENTS = {}  # CGPA and the credit totals were removed from the feature set
# Per-year (passed, enrolled) pairs. The domain cap passed <= enrolled is enforced
# on these pairs (year 4 has only an enrolled column, no passed, so it is absent).
CREDIT_YEAR_PAIRS = [
    ("credit_passed_1styear", "credit_enrolled_1styear"),
    ("credit_passed_2ndyear", "credit_enrolled_2ndyear"),
    ("credit_passed_3rdyear", "credit_enrolled_3rdyear"),
]
# ============================================================
# DOMAIN CONSTRAINTS (calibrated to oulad_early.csv)
# ============================================================
# Integer-valued features -> round back to whole numbers after the attack.
# All click/submission/day/resource counts, both first_submit_day columns, and
# every credit_* column are integers.
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
# Decimal features -> round to 2 decimal places after the attack. These are the
# only non-integer attackable columns (avg_score on 0-100, GPA/CGPA on 0-4).
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
# Non-negative features. first_submit_day is left OUT on purpose: the train
# min/max clip already bounds it, and keeping it out means the code stays correct
# if a future dataset reintroduces negative submit-day values. GPA_* and CGPA are
# on a 0-4 scale (non-negative but never clipped to 100). All credit_* columns are
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
# SOURCE MLP MODEL (must match the training MLP exactly)
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
def normalize_target(series):
    # same mapping as xgboost_model.py / mlp_model.py / fgsm_xgboost.py
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
    if "id_student" in df.columns:
        student_ids = df["id_student"].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    return X, y, student_ids
def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # SAME split logic and random_state as the XGBoost / MLP training scripts, so the
    # train/test/validation rows here are exactly the rows those models used.
    if student_ids is None:
        student_ids = pd.Series(np.arange(len(X)), index=X.index, name="row_id")
    # 70% train / 30% temp
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids,
        test_size=0.30, stratify=y, random_state=random_state,
    )
    # split the 30% temp -> 20% test, 10% validation
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
        [os.path.join(xgb_results_dir, "oulad_xgb_model.pkl")],
        "XGBoost model (oulad_xgb_model.pkl)",
    )
def resolve_xgb_summary_path(xgb_results_dir):
    path = os.path.join(xgb_results_dir, "oulad_xgb_summary.json")
    return path if os.path.exists(path) else None
def load_xgb_threshold(summary_path, fallback=0.5):
    # reuse the XGBoost's tuned decision threshold; fall back to 0.5 if the summary
    # is missing or has no threshold. Matches fgsm_xgboost.py's threshold source.
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
    # read the tuned XGB params only for the summary JSON. The defended XGBoost
    # itself is a clone of the baseline pipeline, so it already carries these
    # params. Returns (params, winning_strategy, source) to match the zenodo
    # adversarial-training script.
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
        [os.path.join(mlp_dir, "oulad_mlp_model.pt")],
        "MLP model (oulad_mlp_model.pt)",
    )
def resolve_mlp_preprocessor_path(mlp_dir):
    return resolve_first_existing(
        [os.path.join(mlp_dir, "oulad_mlp_preprocessor.pkl")],
        "MLP preprocessor (oulad_mlp_preprocessor.pkl)",
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
            "feature_set": obj.get("feature_set", None),
            "best_threshold": obj.get("best_threshold", None),
            "input_dim": obj.get("input_dim", None),
        }
    else:
        info = {
            "hidden_dims": DEFAULT_HIDDEN_DIMS,
            "dropout": 0.30,
            "state_dict": obj,
            "selected_features": None,
            "feature_set": None,
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
    # Build the ART mask in the MLP's preprocessed space. Column order produced by
    # the MLP ColumnTransformer is numeric block -> (flag block) -> one-hot block.
    # Only numeric columns whose raw name is attackable may move; the one-hot
    # block stays frozen (mask starts as all-zeros).
    os.makedirs(output_dir, exist_ok=True)
    attackable_raw = set(ATTACKABLE_FEATURES)
    numeric_cols = get_numeric_cols_from_preprocessor(preprocessor)
    missing_attackable = sorted([f for f in attackable_raw if f not in selected_features])
    if missing_attackable:
        raise ValueError(
            f"Attackable features not in selected_features: {missing_attackable}"
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
    # also save a raw-level mask because it is easier to read. round_to_half is
    # always 0 for oulad_early (credits are integers, no half-credit grid); the
    # column is kept so the CSV schema matches the zenodo attack scripts.
    raw_mask_df = pd.DataFrame({
        "feature": selected_features,
        "raw_attackable": [int(f in attackable_raw) for f in selected_features],
        "round_to_int": [int(f in ROUND_TO_INT_FEATURES) for f in selected_features],
        "round_to_half": [0 for f in selected_features],
        "round_to_2dp": [int(f in DECIMAL_2DP_FEATURES) for f in selected_features],
        "percentage_0_100": [int(f in PERCENTAGE_FEATURES) for f in selected_features],
        "attack_status": ["attackable" if f in attackable_raw else "frozen" for f in selected_features],
    })
    preprocessed_mask_df = pd.DataFrame(rows)
    raw_mask_df.to_csv(os.path.join(output_dir, "oulad_attack_mask_raw_features.csv"), index=False)
    preprocessed_mask_df.to_csv(os.path.join(output_dir, "oulad_attack_mask_preprocessed_numeric_block.csv"), index=False)
    np.save(os.path.join(output_dir, "oulad_perturbable_mask_preprocessed.npy"), mask)
    if int(mask.sum()) == 0:
        raise ValueError("No preprocessed feature is attackable. Check ATTACKABLE_FEATURES.")
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
    # clip numeric values to train min/max when bounds exist. This is the primary
    # realism bound: it keeps every attacked feature inside the range observed in
    # the training split.
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
    # rounding: integer counts -> whole numbers; decimals (avg_score, GPA, CGPA)
    # -> 2 decimal places. Done after clipping so the rounded value still obeys
    # the range bounds.
    for col in ROUND_TO_INT_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col]).astype(float)
    for col in DECIMAL_2DP_FEATURES:
        if col in X_adv_raw.columns:
            X_adv_raw[col] = np.round(X_adv_raw[col].astype(float), 2)
    # ALWAYS enforce domain consistency on the finalized (clipped, rounded)
    # components: per-year passed <= enrolled, totals = sum of components, and
    # CGPA = mean of GPAs. Hardcoded ON to mirror the zenodo adversarial-training
    # script and to guarantee domain-valid adversarial rows on the final values.
    X_adv_raw = enforce_derived_consistency(X_adv_raw)
    X_adv_raw = enforce_activity_consistency(X_adv_raw, clean_reset, train_min, train_max)
    assert_feasible(oulad_feasibility_violations(X_adv_raw), attacked_row_mask, 'OULAD projection')
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
    # hard-restore frozen preprocessed dimensions
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
        # keep frozen dimensions untouched inside the attacked subset
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
    )
    return X_adv_raw, X_adv_projected, attack_idx
# ============================================================
# EVALUATION AND SAVING
# ============================================================
# Minimum dropout precision accepted when picking the threshold. Among all
# thresholds that keep dropout precision at or above this floor, take the one
# with the highest recall (F1 then precision break ties); if none reach the
# floor, fall back to best F1. Same rule and value as the zenodo tree models.
MIN_PRECISION = 0.73

def tune_threshold_from_proba(y_true, y_proba, min_precision=MIN_PRECISION):
    # sweep thresholds, then among the ones that keep dropout precision at or
    # above min_precision, pick the highest recall (f1 then precision break ties).
    # goal: catch as many dropouts as possible while precision stays >= floor.
    # if no threshold reaches the precision floor, fall back to best f1 so the
    # function still returns a usable threshold instead of failing.
    # Same rule and column names as the zenodo tree models.
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
    # the XGBoost pkl is a full imblearn pipeline (preprocessor + model), so it
    # takes the RAW dataframe and preprocesses internally
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
    # summarize how much moved, split by attackable vs frozen dimensions
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
        "id_student": student_ids.values,
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
    # categorical columns (code_module, code_presentation) are strings, so their
    # diff is coerced to numeric and left blank when it cannot be computed.
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
    # clone the baseline XGBoost pipeline so the defended model keeps the exact
    # same preprocessing, sampler, and tuned hyperparameters, then refit it on the
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
    feature_set_hint,
    train_epsilon_list,
    eval_epsilon_list,
    random_state,
    attack_mode,
    train_attack_scope,
    eval_attack_scope,
    eval_mlp_model_path_arg=None,
    eval_mlp_preprocessor_arg=None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("XGBOOST ADVERSARIAL TRAINING WITH FGSM VIA SOURCE MLP (oulad_early)")
    print("=" * 80)
    print(f"CSV path           : {csv_path}")
    print(f"Output dir         : {output_dir}")
    print(f"XGB results dir     : {xgb_results_dir}")
    print(f"MLP dir            : {mlp_dir}")
    print(f"Attack mode        : {attack_mode}")
    print(f"Train attack scope : {train_attack_scope}")
    print(f"Eval attack scope  : {eval_attack_scope}")
    print(f"Enforce derived    : always ON (30-credit grid, 30-120 per year, year4 = 360 - sum(years 1-3), passed <= enrolled; CGPA and credit totals not used)")
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
    if mlp_info["feature_set"] is not None and feature_set_hint is not None \
            and mlp_info["feature_set"] != feature_set_hint:
        print(f"WARNING: MLP checkpoint feature_set is '{mlp_info['feature_set']}' but "
              f"--feature-set was '{feature_set_hint}'. Using the checkpoint features.")
    print()
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
    # load baseline XGBoost pipeline and its tuned threshold
    xgb_baseline_model_path = resolve_xgb_model_path(xgb_results_dir)
    xgb_summary_path = resolve_xgb_summary_path(xgb_results_dir)
    xgb_baseline = joblib.load(xgb_baseline_model_path)
    xgb_baseline_threshold, xgb_threshold_source = load_xgb_threshold(xgb_summary_path)
    if xgb_threshold_source == "fallback_0.5":
        print("WARNING: baseline XGBoost threshold not found in summary; using fallback 0.5 "
              "for the baseline-clean evaluation. Check --xgb-results-dir and the "
              "oulad_xgb_summary.json file.")
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
                "oulad_mlp_defended_model_train_eps_1p0.pt), or pass --eval-mlp-model with the "
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
    save_confusion_matrix(baseline_test_cm, os.path.join(output_dir, "oulad_xgb_baseline_clean_test_cm.csv"))
    save_confusion_matrix(baseline_val_cm, os.path.join(output_dir, "oulad_xgb_baseline_clean_validation_cm.csv"))
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
        X_train_adv_raw.to_csv(os.path.join(output_dir, f"oulad_X_train_fgsm_raw_train_eps_{eps_tag}.csv"), index=False)
        np.save(os.path.join(output_dir, f"oulad_X_train_fgsm_preprocessed_train_eps_{eps_tag}.npy"), X_train_adv_preprocessed)
        save_training_sample_csv(
            X_clean_raw=X_train,
            X_adv_raw=X_train_adv_raw,
            y=y_train.values,
            selected_features=selected_features,
            attack_idx=train_attack_idx,
            train_epsilon=train_epsilon,
            path=os.path.join(output_dir, f"oulad_clean_vs_fgsm_train_sample_train_eps_{eps_tag}.csv"),
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
        threshold_df.to_csv(os.path.join(output_dir, f"oulad_xgb_defended_threshold_tuning_train_eps_{eps_tag}.csv"), index=False)
        model_path = os.path.join(output_dir, f"oulad_xgb_defended_model_train_eps_{eps_tag}.pkl")
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
            save_confusion_matrix(cm, os.path.join(output_dir, f"oulad_xgb_defended_clean_{split_name}_cm_train_eps_{eps_tag}.csv"))
            save_predictions(
                student_ids=id_split,
                y_true=y_split.values,
                y_pred=pred,
                y_proba=proba,
                threshold=defended_threshold,
                condition="Defended Clean",
                train_epsilon=float(train_epsilon),
                eval_epsilon=0.0,
                path=os.path.join(output_dir, f"oulad_xgb_defended_clean_{split_name}_predictions_train_eps_{eps_tag}.csv"),
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
                })
                all_stats.append(stats)
                if stats["max_abs_perturbation_frozen_preprocessed"] > 1e-6:
                    raise AssertionError(f"Frozen features changed in {split_name} eval data.")
                X_adv_raw.to_csv(
                    os.path.join(output_dir, f"oulad_X_{split_name}_fgsm_raw_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
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
                    os.path.join(output_dir, f"oulad_xgb_defended_fgsm_{split_name}_cm_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
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
                    path=os.path.join(output_dir, f"oulad_xgb_defended_fgsm_{split_name}_predictions_train_eps_{eps_tag}_eval_eps_{eval_tag}.csv"),
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
            "enforce_derived_consistency": True,
            "credit_record_rule": CREDIT_RULE_TEXT,
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
    results_path = os.path.join(output_dir, "oulad_xgb_adversarial_training_results.csv")
    stats_path = os.path.join(output_dir, "oulad_xgb_adversarial_training_perturbation_stats.csv")
    summary_csv_path = os.path.join(output_dir, "oulad_xgb_adversarial_training_summary_by_train_epsilon.csv")
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
        "enforce_derived_consistency": True,
        "credit_record_rule": CREDIT_RULE_TEXT,
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
            "CGPA and the credit totals are not model features. Domain consistency is "
            "ALWAYS enforced (hardcoded ON) after clip/round: per-year credit_passed <= "
            "credit_enrolled and credit_enrolled_4thyear = 360 - sum(years 1-3)."
        ),
        "constraints": {
            "frozen_features": "Restored to clean values after FGSM (code_module, code_presentation, module_credits).",
            "non_attacked_rows": "Restored completely to clean values.",
            "raw_value_clip": "Numeric features clipped to train min/max when available.",
            "percentage_features": "avg_score features clipped to [0, 100].",
            "integer_count_features": "Rounded to whole numbers after inverse transform.",
            "decimal_features": "avg_score and GPA rounded to 2 decimal places.",
            "nonnegative_features": "Clipped to lower bound 0 (first_submit_day excluded, bounded only by train min/max).",
            "credit_identities": ("Per-year passed <= enrolled and year-4 enrolment = 360 - sum(years 1-3), "
                                  "always re-imposed after clip/round."),
            "resampler_note": "If the baseline XGBoost pipeline contains a resampler, it "
                              "runs on the mixed (clean + adversarial) training set.",
        },
        "split": split_report,
        "results_file": results_path,
        "stats_file": stats_path,
        "summary_by_train_epsilon_file": summary_csv_path,
    }
    summary_path = os.path.join(output_dir, "oulad_xgb_adversarial_training_summary.json")
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
        description="Adversarial training for XGBoost using FGSM via source MLP (oulad_early)."
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH,
                        help=f"Dataset CSV. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--feature-set", default="early_both",
                        choices=["early_65", "early_153", "early_both"],
                        help=("Informational only; the real feature list comes from the MLP "
                              "checkpoint. Default: early_both."))
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
    # Domain-consistency enforcement is now ALWAYS ON (hardcoded), matching the
    # zenodo adversarial-training script: per-year passed <= enrolled, totals =
    # sum of components, CGPA = mean of GPAs, re-imposed after clip/round. The old
    # --no-enforce-derived-consistency flag has been removed so the default run is
    # always domain-valid.
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
        feature_set_hint=args.feature_set,
        train_epsilon_list=train_epsilon_list,
        eval_epsilon_list=eval_epsilon_list,
        random_state=args.random_state,
        attack_mode=args.attack_mode,
        train_attack_scope=args.train_attack_scope,
        eval_attack_scope=args.eval_attack_scope,
        eval_mlp_model_path_arg=args.eval_mlp_model,
        eval_mlp_preprocessor_arg=args.eval_mlp_preprocessor,
    )
if __name__ == "__main__":
    main()