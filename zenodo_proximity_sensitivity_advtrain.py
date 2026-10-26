import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# WHAT THIS SCRIPT DOES
# ============================================================
# After-defence proximity and sensitivity analysis for zenodo.csv, for the
# ADVERSARIAL TRAINING defence. It READS the adversarial artifacts the defended
# training scripts already saved and computes the metrics on them (it does NOT
# regenerate the attack). That is the faithful "After Defence" view, because the
# saved arrays are exactly the adversarial examples the defended models faced.
#
# The adversarial-training scripts persist, per split:
#   RF  (rf_adversarial_training_zenodo_results):
#         zenodo_X_{split}_fgsm_raw_train_eps_{te}_eval_eps_{ee}.csv                 (adv raw)
#         zenodo_rf_defended_fgsm_{split}_predictions_train_eps_{te}_eval_eps_{ee}.csv
#         zenodo_rf_defended_clean_{split}_predictions_train_eps_{te}.csv
#   XGB (xgb_adversarial_training_zenodo_results): same names, xgb_ prefix
#   MLP (mlp_adversarial_training_zenodo_results):
#         zenodo_X_{split}_pgd_preprocessed_train_eps_{te}_eval_eps_{ee}.npy         (adv preprocessed)
#         zenodo_X_{split}_pgd_raw_train_eps_{te}_eval_eps_{ee}.csv                  (adv raw, backup)
#         zenodo_mlp_defended_pgd_{split}_predictions_train_eps_{te}_eval_eps_{ee}.csv
#         zenodo_mlp_defended_clean_{split}_predictions_train_eps_{te}.csv
#
# RF/XGB save the adversarial eval data in RAW form only; it is re-transformed
# here with the MLP preprocessor to reach the standardized space the metrics live
# in. In the CURRENT defence scripts the RF/XGB eval attack is FGSM transferred
# from the MLP-DEFENDED surrogate ("unseen weapon"), which reuses the SAME
# baseline MLP preprocessor, so re-transforming with that preprocessor is the
# correct standardized space. MLP saves the preprocessed adversarial array
# directly, so it is loaded as-is.
#
# UPDATED vs the previous version of this analysis script:
#   1. train/eval epsilons and the eval attack source are now READ from each
#      defence's *_adversarial_training_summary.json instead of being hardcoded,
#      so this script tracks whatever the defence scripts actually ran.
#   2. the RF/XGB attack is labelled by its real source (MLP-defended unseen
#      weapon vs baseline MLP), not the stale "via source MLP" text.
#   3. eval_attack_source is written into the summary outputs.
#
# NOTE on the prediction files:
#   - RF/XGB adv prediction file has 'predicted_label' (the adversarial pred);
#     the clean pred is read from the separate '*_clean_*_predictions' file.
#   - MLP adv prediction file already carries both 'clean_pred_label' and
#     'adv_pred_label' in the same CSV, so the clean file is only a fallback.

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "zenodo.csv"
DEFAULT_MLP_RESULTS_DIR = "mlp_results_zenodo"                       # source of the preprocessor
DEFAULT_RF_DEFENCE_DIR = "rf_adversarial_training_zenodo_results"
DEFAULT_XGB_DEFENCE_DIR = "xgb_adversarial_training_zenodo_results"
DEFAULT_MLP_DEFENCE_DIR = "mlp_adversarial_training_zenodo_results"
DEFAULT_OUTPUT_DIR = "proximity_sensitivity_adv_training_zenodo_results"

# Fallback epsilon if a defence summary is missing. The real values are read from
# the *_adversarial_training_summary.json files at runtime (train/eval both 1.0
# in the current zenodo run).
FALLBACK_EPSILON = 1.0

# Summary JSON each defence script writes; used to read the real epsilons and the
# eval attack source instead of hardcoding them.
SUMMARY_JSON = {
    "RF": "zenodo_rf_adversarial_training_summary.json",
    "XGB": "zenodo_xgb_adversarial_training_summary.json",
    "MLP": "zenodo_mlp_adversarial_training_summary.json",
}

TARGET_COL = "target"
ID_COL = "student_id"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout
EPS = 1e-8

# ============================================================
# FEATURE SET (zenodo, 34 features; matches the training/attack scripts)
# ============================================================
FULL_FEATURES = [
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

# Only these 21 raw features may move; everything else is frozen. All attackable
# features are numeric, so only columns of the scaled numeric ("num") block can be
# perturbed. Frozen items include: the one-hot categoricals (academic_year,
# seniority, degree_size, highest_course_year_enrolled), the frozen numeric
# descriptor n_courses, and all 8 binary flags.
ATTACKABLE_FEATURES_FULL = [
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

# ============================================================
# DATA HELPERS
# ============================================================
def eps_tag(value: float) -> str:
    return str(value).replace(".", "p")


def candidate_eps_strings(value: float) -> List[str]:
    raw = str(value)
    return list(dict.fromkeys([eps_tag(value), raw, raw.rstrip("0").rstrip(".")]))


def find_existing_path(patterns: List[str]) -> Optional[str]:
    for path in patterns:
        if os.path.exists(path):
            return path
    return None


def read_defence_summary(defence_dir: str, model_key: str) -> Tuple[dict, Optional[str]]:
    # read the defence's own summary JSON to recover the real epsilons and the
    # eval attack source. returns ({}, None) if it cannot be read.
    path = os.path.join(defence_dir, SUMMARY_JSON[model_key])
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path
    except Exception:
        return {}, None


def resolve_case_settings(defence_dir, model_key, cli_train_eps, cli_eval_eps):
    # precedence for epsilons: explicit CLI value -> value in the defence summary
    # -> FALLBACK_EPSILON. eval_attack_source is taken from the summary when
    # available (RF/XGB store the surrogate checkpoint basename or "baseline_mlp";
    # MLP stores "defended"/"baseline").
    summary, summary_path = read_defence_summary(defence_dir, model_key)

    def first_float(lst, fallback):
        if isinstance(lst, list) and lst:
            try:
                return float(lst[0])
            except (TypeError, ValueError):
                return fallback
        return fallback

    train_eps = cli_train_eps if cli_train_eps is not None else first_float(
        summary.get("train_epsilon_list"), FALLBACK_EPSILON)
    eval_eps = cli_eval_eps if cli_eval_eps is not None else first_float(
        summary.get("eval_epsilon_list"), FALLBACK_EPSILON)
    eval_attack_source = summary.get("eval_attack_source", "unknown")
    return train_eps, eval_eps, eval_attack_source, summary_path


def attack_label(model_key: str, eval_attack_source: str) -> str:
    # human-readable attack name that reflects the REAL eval weapon.
    if model_key == "MLP":
        return "PGD white-box"
    src = str(eval_attack_source).lower()
    if src in ("baseline_mlp", "baseline"):
        return "FGSM transfer (baseline MLP)"
    if "defended" in src:
        return "FGSM transfer (unseen weapon: MLP-defended surrogate)"
    return "FGSM transfer (MLP surrogate)"


def normalize_target(series: pd.Series) -> pd.Series:
    target_normalized = (
        series.astype(str).str.strip().str.lower().replace({
            "non dropout": "non-dropout",
            "non_dropout": "non-dropout",
            "drop out": "dropout",
        })
    )
    target_map = {"non-dropout": 0, "dropout": 1, "0": 0, "1": 1}
    y = target_normalized.map(target_map)
    if y.isna().any():
        bad_values = series.loc[y.isna()].unique().tolist()
        raise ValueError(f"some target values could not be mapped: {bad_values}")
    return y.astype(int)


def load_and_split_dataset(csv_path: str, random_state: int):
    df = pd.read_csv(csv_path)
    missing = [c for c in FULL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing FULL_FEATURES columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"missing target column: {TARGET_COL}")

    X = df[FULL_FEATURES].copy()
    y = normalize_target(df[TARGET_COL])

    if ID_COL in df.columns:
        student_ids = df[ID_COL].copy()
    else:
        student_ids = pd.Series(np.arange(len(df)), index=df.index, name="row_id")

    # SAME stratified 70/20/10 split and random_state as the training/defence
    # scripts, so the clean rows line up row-by-row with the saved adversarial
    # arrays.
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, student_ids, test_size=0.30, stratify=y, random_state=random_state,
    )
    X_test, X_val, y_test, y_val, id_test, id_val = train_test_split(
        X_temp, y_temp, id_temp, test_size=1.0 / 3.0, stratify=y_temp, random_state=random_state,
    )

    split_data = {}
    for split, (Xs, ys, ids) in {
        "train": (X_train, y_train, id_train),
        "test": (X_test, y_test, id_test),
        "validation": (X_val, y_val, id_val),
    }.items():
        split_data[split] = {
            "X_raw": Xs.reset_index(drop=True).copy(),
            "y": ys.reset_index(drop=True).values.astype(int),
            "student_ids": ids.reset_index(drop=True),
        }
    return split_data


def load_preprocessor(mlp_results_dir: str):
    candidates = [
        os.path.join(mlp_results_dir, "zenodo_mlp_preprocessor.pkl"),
        os.path.join(mlp_results_dir, "full", "zenodo_mlp_preprocessor.pkl"),
    ]
    path = find_existing_path(candidates)
    if path is None:
        raise FileNotFoundError("MLP preprocessor not found. checked:\n" + "\n".join(candidates))
    return joblib.load(path), path


def transform_to_float32(preprocessor, X: pd.DataFrame) -> np.ndarray:
    arr = preprocessor.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)


# ============================================================
# PREPROCESSOR MASK HELPERS
# ============================================================
# The zenodo preprocessor is a ColumnTransformer with three blocks:
#   "num"  -> SimpleImputer(median) + StandardScaler   (22 numeric columns)
#   "flag" -> SimpleImputer(most_frequent), kept 0/1   (8 binary flags)
#   "cat"  -> SimpleImputer + OneHotEncoder            (4 categoricals expanded)
# The scaled numeric block is where all attackable perturbations live. The flag
# and one-hot blocks are all frozen, so they are grouped together into the
# "categorical_or_other" mask (any block that is not "num").
def get_transformer_columns(preprocessor, transformer_name: str) -> List[str]:
    for name, transformer, cols in preprocessor.transformers_:
        if name == transformer_name:
            return list(cols)
    return []


def get_transformer_slice(preprocessor, transformer_name: str, input_dim: int):
    if hasattr(preprocessor, "output_indices_") and transformer_name in preprocessor.output_indices_:
        return preprocessor.output_indices_[transformer_name]
    if transformer_name == "num":
        n_cols = len(get_transformer_columns(preprocessor, "num"))
        return slice(0, min(n_cols, input_dim))
    return slice(0, 0)


def make_preprocessed_masks(preprocessor, input_dim: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(input_dim)]

    perturbable_mask = np.zeros(input_dim, dtype=bool)      # attackable numeric columns
    numerical_mask = np.zeros(input_dim, dtype=bool)        # the full scaled "num" block
    categorical_ohe_mask = np.zeros(input_dim, dtype=bool)  # flag + one-hot blocks

    numeric_cols = get_transformer_columns(preprocessor, "num")
    num_slice = get_transformer_slice(preprocessor, "num", input_dim)
    num_indices = list(range(num_slice.start or 0, num_slice.stop or 0))

    attackable = set(ATTACKABLE_FEATURES_FULL)
    for local_idx, raw_col in enumerate(numeric_cols):
        if local_idx >= len(num_indices):
            continue
        idx = num_indices[local_idx]
        numerical_mask[idx] = True
        if raw_col in attackable:
            perturbable_mask[idx] = True

    # every block that is not "num" (i.e. "flag" and "cat") is treated as
    # categorical/other; these are all frozen in the zenodo attack.
    for name, _, _ in preprocessor.transformers_:
        if name in ["num", "remainder"]:
            continue
        if hasattr(preprocessor, "output_indices_") and name in preprocessor.output_indices_:
            categorical_ohe_mask[preprocessor.output_indices_[name]] = True
    if categorical_ohe_mask.sum() == 0:
        categorical_ohe_mask[~numerical_mask] = True

    mask_df = pd.DataFrame({
        "preprocessed_index": np.arange(input_dim),
        "feature_name": feature_names,
        "is_attackable_feature": perturbable_mask,
        "is_numerical_feature": numerical_mask,
        "is_categorical_or_other_feature": categorical_ohe_mask,
    })
    mask_df.to_csv(os.path.join(output_dir, "zenodo_preprocessed_feature_masks.csv"), index=False)
    return perturbable_mask, numerical_mask, categorical_ohe_mask, feature_names, mask_df


# ============================================================
# PREDICTION CSV HELPERS
# ============================================================
def extract_predictions_from_csv(pred_path: str, clean_pred_path: Optional[str] = None):
    pred_df = pd.read_csv(pred_path)
    if "actual_label" not in pred_df.columns:
        raise ValueError(f"actual_label not found in {pred_path}")
    y_true = pred_df["actual_label"].values.astype(int)

    if "adv_pred_label" in pred_df.columns:
        adv_pred = pred_df["adv_pred_label"].values.astype(int)
    elif "predicted_label" in pred_df.columns:
        adv_pred = pred_df["predicted_label"].values.astype(int)
    else:
        raise ValueError(f"no adv/predicted label column found in {pred_path}")

    if "clean_pred_label" in pred_df.columns:
        clean_pred = pred_df["clean_pred_label"].values.astype(int)
    elif clean_pred_path is not None and os.path.exists(clean_pred_path):
        clean_df = pd.read_csv(clean_pred_path)
        if "predicted_label" in clean_df.columns:
            clean_pred = clean_df["predicted_label"].values.astype(int)
        elif "clean_pred_label" in clean_df.columns:
            clean_pred = clean_df["clean_pred_label"].values.astype(int)
        elif "adv_pred_label" in clean_df.columns:
            clean_pred = clean_df["adv_pred_label"].values.astype(int)
        else:
            raise ValueError(f"no predicted_label in clean prediction file: {clean_pred_path}")
    else:
        raise ValueError(f"clean_pred_label not in {pred_path} and clean_pred_path unavailable.")

    return y_true, clean_pred, adv_pred, pred_df


def true_positive_attack_mask(y_true: np.ndarray, clean_pred: np.ndarray) -> np.ndarray:
    # matches eval_attack_scope = true_positive_only: only dropout rows the
    # defended model already caught on clean data are attacked.
    return (np.asarray(y_true) == POSITIVE_LABEL) & (np.asarray(clean_pred) == POSITIVE_LABEL)


# ============================================================
# METRIC FUNCTIONS
# ============================================================
def safe_subset(X: np.ndarray, row_mask: Optional[np.ndarray]) -> np.ndarray:
    if row_mask is None:
        return X
    row_mask = np.asarray(row_mask).astype(bool)
    if row_mask.sum() == 0:
        return X[:0]
    return X[row_mask]


def compute_proximity_metrics(X_clean, X_adv, perturbable_mask, row_mask=None) -> Dict[str, float]:
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return {"row_count_used_for_distance": 0}
    diff = Xa - Xc
    abs_diff = np.abs(diff)
    pm = perturbable_mask.astype(bool)
    frozen = ~pm
    diff_attackable = diff[:, pm]
    abs_frozen = np.abs(diff[:, frozen]) if frozen.any() else np.zeros((len(Xc), 0), dtype=np.float32)
    return {
        "row_count_used_for_distance": int(len(Xc)),
        "mean_l1_all": float(np.mean(np.linalg.norm(diff, ord=1, axis=1))),
        "median_l1_all": float(np.median(np.linalg.norm(diff, ord=1, axis=1))),
        "mean_l2_all": float(np.mean(np.linalg.norm(diff, ord=2, axis=1))),
        "median_l2_all": float(np.median(np.linalg.norm(diff, ord=2, axis=1))),
        "mean_linf_all": float(np.mean(np.linalg.norm(diff, ord=np.inf, axis=1))),
        "median_linf_all": float(np.median(np.linalg.norm(diff, ord=np.inf, axis=1))),
        "mean_l1_attackable": float(np.mean(np.linalg.norm(diff_attackable, ord=1, axis=1))),
        "median_l1_attackable": float(np.median(np.linalg.norm(diff_attackable, ord=1, axis=1))),
        "mean_l2_attackable": float(np.mean(np.linalg.norm(diff_attackable, ord=2, axis=1))),
        "median_l2_attackable": float(np.median(np.linalg.norm(diff_attackable, ord=2, axis=1))),
        "mean_linf_attackable": float(np.mean(np.linalg.norm(diff_attackable, ord=np.inf, axis=1))),
        "median_linf_attackable": float(np.median(np.linalg.norm(diff_attackable, ord=np.inf, axis=1))),
        "mean_abs_change_all": float(np.mean(abs_diff)),
        "mean_abs_change_attackable": float(np.mean(np.abs(diff_attackable))),
        "max_abs_change_all": float(np.max(abs_diff)),
        "max_abs_change_attackable": float(np.max(np.abs(diff_attackable))),
        "mean_abs_change_frozen": float(np.mean(abs_frozen)) if abs_frozen.size else 0.0,
        "max_abs_change_frozen": float(np.max(abs_frozen)) if abs_frozen.size else 0.0,
        "frozen_changed_total": int(np.sum(abs_frozen > EPS)) if abs_frozen.size else 0,
        "frozen_changed_samples": int(np.sum(np.any(abs_frozen > EPS, axis=1))) if abs_frozen.size else 0,
    }


def compute_sensitivity_metrics(X_clean, X_adv, numerical_mask, categorical_ohe_mask, row_mask=None) -> Dict[str, float]:
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return {"row_count_used_for_sensitivity": 0}
    abs_diff = np.abs(Xa - Xc)
    num_mask = numerical_mask.astype(bool)
    cat_mask = categorical_ohe_mask.astype(bool)
    # numeric block is StandardScaler output, so |adv - clean| there already
    # equals |raw_adv - raw_clean| / sigma_j. sensitivity = sum of those over the
    # numeric columns, per sample.
    per_sample_sen = abs_diff[:, num_mask].sum(axis=1) if num_mask.sum() > 0 else np.zeros(len(Xc), dtype=np.float32)
    cat_change_rate = (abs_diff[:, cat_mask] > EPS).astype(np.float32).mean(axis=1) if cat_mask.sum() > 0 else np.zeros(len(Xc), dtype=np.float32)
    return {
        "row_count_used_for_sensitivity": int(len(Xc)),
        "mean_sensitivity": float(np.mean(per_sample_sen)),
        "median_sensitivity": float(np.median(per_sample_sen)),
        "max_sensitivity": float(np.max(per_sample_sen)),
        "mean_categorical_change_rate": float(np.mean(cat_change_rate)),
    }


def compute_attack_effect(y_true, clean_pred, adv_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    clean_pred = np.asarray(clean_pred).astype(int)
    adv_pred = np.asarray(adv_pred).astype(int)
    attack_idx = true_positive_attack_mask(y_true, clean_pred)
    hidden = (y_true == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL) & (adv_pred == NEGATIVE_LABEL)
    non_to_drop = (y_true == NEGATIVE_LABEL) & (clean_pred == NEGATIVE_LABEL) & (adv_pred == POSITIVE_LABEL)
    attacked_count = int(attack_idx.sum())
    hidden_count = int(hidden.sum())
    return {
        "attacked_rows_count": attacked_count,
        "attacked_dropout_rows_count": int(np.sum(attack_idx & (y_true == POSITIVE_LABEL))),
        "dropout_to_nondropout_flips": hidden_count,
        "dropout_hiding_rate": float(hidden_count / attacked_count) if attacked_count else 0.0,
        "nondropout_to_dropout_flips": int(non_to_drop.sum()),
        "prediction_changed_count": int(np.sum(clean_pred != adv_pred)),
    }


def feature_sensitivity_summary(X_clean, X_adv, feature_names, perturbable_mask, numerical_mask,
                                categorical_ohe_mask, model, attack, split, train_epsilon, eval_epsilon,
                                row_scope, row_mask=None) -> pd.DataFrame:
    Xc = safe_subset(X_clean, row_mask)
    Xa = safe_subset(X_adv, row_mask)
    if len(Xc) == 0:
        return pd.DataFrame()
    diff = Xa - Xc
    abs_diff = np.abs(diff)
    num_mask = numerical_mask.astype(bool)
    cat_mask = categorical_ohe_mask.astype(bool)
    sens_matrix = np.zeros_like(abs_diff, dtype=np.float32)
    if num_mask.sum() > 0:
        sens_matrix[:, num_mask] = abs_diff[:, num_mask]
    df = pd.DataFrame({
        "model": model, "attack": attack, "split": split,
        "train_epsilon": train_epsilon, "eval_epsilon": eval_epsilon, "row_scope": row_scope,
        "feature_name": feature_names,
        "is_attackable_feature": perturbable_mask.astype(bool),
        "is_numerical_feature": num_mask,
        "is_categorical_or_other_feature": cat_mask,
        "mean_abs_change": abs_diff.mean(axis=0),
        "max_abs_change": abs_diff.max(axis=0),
        "mean_signed_change": diff.mean(axis=0),
        "changed_sample_count": (abs_diff > EPS).sum(axis=0),
        "changed_sample_rate": (abs_diff > EPS).mean(axis=0),
        "mean_sensitivity": sens_matrix.mean(axis=0),
        "max_sensitivity": sens_matrix.max(axis=0),
    })
    df["sensitivity_rank"] = df["mean_sensitivity"].rank(ascending=False, method="dense")
    return df.sort_values("mean_sensitivity", ascending=False)


# ============================================================
# FILE PATH BUILDERS (match the zenodo adversarial-training output names)
# ============================================================
def defended_raw_adv_candidates(defence_dir, model_key, split, train_eps, eval_eps) -> List[str]:
    split = split.lower()
    te_tags = candidate_eps_strings(train_eps)
    ee_tags = candidate_eps_strings(eval_eps)
    out = []
    attack = "pgd" if model_key == "MLP" else "fgsm"
    for te in te_tags:
        for ee in ee_tags:
            out.append(os.path.join(defence_dir, f"zenodo_X_{split}_{attack}_raw_train_eps_{te}_eval_eps_{ee}.csv"))
    return out


def defended_pre_adv_candidates(defence_dir, model_key, split, train_eps, eval_eps) -> List[str]:
    split = split.lower()
    te_tags = candidate_eps_strings(train_eps)
    ee_tags = candidate_eps_strings(eval_eps)
    out = []
    attack = "pgd" if model_key == "MLP" else "fgsm"
    for te in te_tags:
        for ee in ee_tags:
            out.append(os.path.join(defence_dir, f"zenodo_X_{split}_{attack}_preprocessed_train_eps_{te}_eval_eps_{ee}.npy"))
            out.append(os.path.join(defence_dir, f"zenodo_X_{split}_{attack}_scaled_train_eps_{te}_eval_eps_{ee}.npy"))
    return out


def defended_pred_candidates(defence_dir, model_key, split, train_eps, eval_eps) -> List[str]:
    split = split.lower()
    prefix = {"RF": "zenodo_rf_defended", "XGB": "zenodo_xgb_defended", "MLP": "zenodo_mlp_defended"}[model_key]
    attack = "pgd" if model_key == "MLP" else "fgsm"
    te_tags = candidate_eps_strings(train_eps)
    ee_tags = candidate_eps_strings(eval_eps)
    out = []
    for te in te_tags:
        for ee in ee_tags:
            out.append(os.path.join(defence_dir, f"{prefix}_{attack}_{split}_predictions_train_eps_{te}_eval_eps_{ee}.csv"))
    return out


def defended_clean_pred_candidates(defence_dir, model_key, split, train_eps) -> List[str]:
    split = split.lower()
    prefix = {"RF": "zenodo_rf_defended", "XGB": "zenodo_xgb_defended", "MLP": "zenodo_mlp_defended"}[model_key]
    te_tags = candidate_eps_strings(train_eps)
    return [os.path.join(defence_dir, f"{prefix}_clean_{split}_predictions_train_eps_{te}.csv") for te in te_tags]


# ============================================================
# CASE PROCESSOR
# ============================================================
def process_case(model_key, model_name, attack_name, eval_attack_source, defence_dir, train_eps, eval_eps,
                 split_name, split_data, preprocessor, perturbable_mask, numerical_mask,
                 categorical_ohe_mask, feature_names) -> Tuple[List[Dict], List[pd.DataFrame]]:
    split_lower = split_name.lower()
    X_clean_raw = split_data[split_lower]["X_raw"]
    X_clean_pre = transform_to_float32(preprocessor, X_clean_raw)

    raw_path = find_existing_path(defended_raw_adv_candidates(defence_dir, model_key, split_lower, train_eps, eval_eps))
    pre_path = find_existing_path(defended_pre_adv_candidates(defence_dir, model_key, split_lower, train_eps, eval_eps))
    pred_path = find_existing_path(defended_pred_candidates(defence_dir, model_key, split_lower, train_eps, eval_eps))
    clean_pred_path = find_existing_path(defended_clean_pred_candidates(defence_dir, model_key, split_lower, train_eps))

    if raw_path is None and pre_path is None:
        print(f"SKIP {model_name} {split_name}: no adversarial sample file in {defence_dir}")
        return [], []
    if pred_path is None:
        print(f"SKIP {model_name} {split_name}: no defended prediction CSV in {defence_dir}")
        return [], []

    # MLP stores the preprocessed adversarial array directly; RF/XGB store raw
    # (crafted in MLP space via an MLP surrogate) and it is re-transformed here.
    if pre_path is not None:
        X_adv_pre = np.load(pre_path).astype(np.float32)
        adv_source = pre_path
    else:
        X_adv_raw = pd.read_csv(raw_path)
        X_adv_pre = transform_to_float32(preprocessor, X_adv_raw[FULL_FEATURES])
        adv_source = raw_path

    y_true, clean_pred, adv_pred, _ = extract_predictions_from_csv(pred_path, clean_pred_path)
    if len(y_true) != len(X_clean_pre):
        raise ValueError(f"length mismatch {model_name} {split_name}: predictions {len(y_true)} vs clean {len(X_clean_pre)}")
    if X_adv_pre.shape != X_clean_pre.shape:
        raise ValueError(f"shape mismatch {model_name} {split_name}: clean {X_clean_pre.shape} vs adv {X_adv_pre.shape}")

    attack_idx = true_positive_attack_mask(y_true, clean_pred)
    effect = compute_attack_effect(y_true, clean_pred, adv_pred)

    rows, feature_dfs = [], []
    for row_scope, row_mask in [("all_rows", None), ("attacked_rows_only", attack_idx)]:
        row = {
            "model": model_name, "attack": attack_name, "condition": "After Defence (Adversarial Training)",
            "eval_attack_source": eval_attack_source,
            "split": split_name, "train_epsilon": train_eps, "eval_epsilon": eval_eps, "epsilon": eval_eps,
            "row_scope": row_scope, "adversarial_sample_source": adv_source,
            "prediction_source": pred_path,
            "clean_prediction_source": clean_pred_path or "included_in_prediction_source",
        }
        row.update(effect)
        row.update(compute_proximity_metrics(X_clean_pre, X_adv_pre, perturbable_mask, row_mask=row_mask))
        row.update(compute_sensitivity_metrics(X_clean_pre, X_adv_pre, numerical_mask, categorical_ohe_mask, row_mask=row_mask))
        rows.append(row)
        feature_dfs.append(feature_sensitivity_summary(
            X_clean_pre, X_adv_pre, feature_names, perturbable_mask, numerical_mask, categorical_ohe_mask,
            model=model_name, attack=attack_name, split=split_name, train_epsilon=train_eps,
            eval_epsilon=eval_eps, row_scope=row_scope, row_mask=row_mask,
        ))

    print(f"OK {model_name} {split_name}: train_eps={train_eps}, eval_eps={eval_eps}, "
          f"source={eval_attack_source}, attacked={effect['attacked_rows_count']}, "
          f"hidden={effect['dropout_to_nondropout_flips']}, hide_rate={effect['dropout_hiding_rate']:.3f}")
    return rows, feature_dfs


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="After-defence (adversarial training) proximity and sensitivity "
                                                  "for zenodo, reading the saved defended artifacts.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--mlp-results-dir", default=DEFAULT_MLP_RESULTS_DIR, help="Source of the MLP preprocessor.")
    parser.add_argument("--rf-defence-dir", default=DEFAULT_RF_DEFENCE_DIR)
    parser.add_argument("--xgb-defence-dir", default=DEFAULT_XGB_DEFENCE_DIR)
    parser.add_argument("--mlp-defence-dir", default=DEFAULT_MLP_DEFENCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--splits", default="test,validation", help="comma-separated: test,validation,train")
    parser.add_argument("--models", default="rf,xgb,mlp", help="comma-separated subset of: rf,xgb,mlp")
    # Epsilons default to None -> read from each defence's summary JSON; pass a
    # value only to override what the defence actually ran with.
    parser.add_argument("--rf-train-epsilon", type=float, default=None)
    parser.add_argument("--rf-eval-epsilon", type=float, default=None)
    parser.add_argument("--xgb-train-epsilon", type=float, default=None)
    parser.add_argument("--xgb-eval-epsilon", type=float, default=None)
    parser.add_argument("--mlp-train-epsilon", type=float, default=None)
    parser.add_argument("--mlp-eval-epsilon", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    split_names = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    print("=" * 80)
    print("ZENODO AFTER-DEFENCE (ADVERSARIAL TRAINING) PROXIMITY AND SENSITIVITY")
    print("=" * 80)
    print(f"CSV path       : {args.csv_path}")
    print(f"MLP preprocessor from: {args.mlp_results_dir}")
    print(f"RF  defence dir: {args.rf_defence_dir}")
    print(f"XGB defence dir: {args.xgb_defence_dir}")
    print(f"MLP defence dir: {args.mlp_defence_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Splits         : {split_names} | Models: {wanted}")
    print()

    split_data = load_and_split_dataset(args.csv_path, args.random_state)
    preprocessor, preprocessor_path = load_preprocessor(args.mlp_results_dir)
    X_test_pre = transform_to_float32(preprocessor, split_data["test"]["X_raw"])
    input_dim = X_test_pre.shape[1]
    perturbable_mask, numerical_mask, categorical_ohe_mask, feature_names, mask_df = make_preprocessed_masks(
        preprocessor, input_dim, args.output_dir)
    print(f"Loaded preprocessor: {preprocessor_path}")
    print(f"Preprocessed dim: {input_dim} | attackable dims: {int(perturbable_mask.sum())} "
          f"| numeric dims: {int(numerical_mask.sum())} | categorical/other dims: {int(categorical_ohe_mask.sum())}")
    print()

    # Resolve each defence's real epsilons + eval attack source from its summary
    # JSON (CLI overrides win when provided).
    base_cases = [
        {"model_key": "RF", "model_name": "Random Forest Defended", "defence_dir": args.rf_defence_dir,
         "cli_train": args.rf_train_epsilon, "cli_eval": args.rf_eval_epsilon, "flag": "rf"},
        {"model_key": "XGB", "model_name": "XGBoost Defended", "defence_dir": args.xgb_defence_dir,
         "cli_train": args.xgb_train_epsilon, "cli_eval": args.xgb_eval_epsilon, "flag": "xgb"},
        {"model_key": "MLP", "model_name": "MLP Defended", "defence_dir": args.mlp_defence_dir,
         "cli_train": args.mlp_train_epsilon, "cli_eval": args.mlp_eval_epsilon, "flag": "mlp"},
    ]
    cases = []
    for c in base_cases:
        if c["flag"] not in wanted:
            continue
        train_eps, eval_eps, eval_attack_source, summary_path = resolve_case_settings(
            c["defence_dir"], c["model_key"], c["cli_train"], c["cli_eval"])
        c.update({
            "train_eps": train_eps, "eval_eps": eval_eps,
            "eval_attack_source": eval_attack_source,
            "attack_name": attack_label(c["model_key"], eval_attack_source),
            "summary_path": summary_path,
        })
        cases.append(c)
        print(f"{c['model_name']:22s} train_eps={train_eps} eval_eps={eval_eps} "
              f"source={eval_attack_source} attack='{c['attack_name']}' "
              f"(summary: {summary_path or 'not found -> fallback'})")
    print()

    all_rows, all_feature_dfs = [], []
    for case in cases:
        for split_name in split_names:
            rows, feature_dfs = process_case(
                model_key=case["model_key"], model_name=case["model_name"], attack_name=case["attack_name"],
                eval_attack_source=case["eval_attack_source"], defence_dir=case["defence_dir"],
                train_eps=case["train_eps"], eval_eps=case["eval_eps"],
                split_name=split_name, split_data=split_data, preprocessor=preprocessor,
                perturbable_mask=perturbable_mask, numerical_mask=numerical_mask,
                categorical_ohe_mask=categorical_ohe_mask, feature_names=feature_names,
            )
            all_rows.extend(rows)
            all_feature_dfs.extend(feature_dfs)

    if not all_rows:
        raise FileNotFoundError(
            "no rows produced. check that the adversarial-training output folders exist and contain the "
            "adversarial sample files plus defended prediction CSVs (run adv_train_rf/xgboost/mlp first).")

    summary_df = pd.DataFrame(all_rows)
    feature_df = pd.concat([f for f in all_feature_dfs if not f.empty], ignore_index=True) if all_feature_dfs else pd.DataFrame()

    summary_path = os.path.join(args.output_dir, "zenodo_proximity_sensitivity_adv_training_summary.csv")
    feature_path = os.path.join(args.output_dir, "zenodo_feature_sensitivity_adv_training_summary.csv")
    top15_path = os.path.join(args.output_dir, "zenodo_top15_sensitive_features_adv_training.csv")
    json_path = os.path.join(args.output_dir, "zenodo_proximity_sensitivity_adv_training_summary.json")

    summary_df.to_csv(summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)

    if not feature_df.empty:
        top15_df = (feature_df[feature_df["is_attackable_feature"] == True]
                    .sort_values(["model", "split", "row_scope", "mean_sensitivity"], ascending=[True, True, True, False])
                    .groupby(["model", "split", "row_scope"]).head(15).reset_index(drop=True))
    else:
        top15_df = pd.DataFrame()
    top15_df.to_csv(top15_path, index=False)

    excel_path = os.path.join(args.output_dir, "zenodo_proximity_sensitivity_adv_training_analysis.xlsx")
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

    case_settings = {
        c["flag"]: {
            "train_epsilon": c["train_eps"], "eval_epsilon": c["eval_eps"],
            "eval_attack_source": c["eval_attack_source"], "attack": c["attack_name"],
            "summary_json": c["summary_path"],
        } for c in cases
    }

    json_summary = {
        "dataset": args.csv_path,
        "feature_set": "zenodo (34 features: 22 numeric, 8 binary flags, 4 one-hot categoricals)",
        "defence": "Adversarial training",
        "analysis": "After-defence proximity and sensitivity for defended RF/XGB (FGSM transfer) and defended MLP (white-box PGD).",
        "source": "Reads the adversarial arrays and defended predictions saved by adv_train_rf/xgboost/mlp (not regenerated).",
        "epsilon_and_source_note": "train/eval epsilons and eval_attack_source are read from each defence's "
                                   "*_adversarial_training_summary.json; CLI epsilon flags override when given.",
        "eval_weapon_note": "In the current defence scripts the RF/XGB eval attack is FGSM transferred from the "
                            "MLP-DEFENDED surrogate (an 'unseen weapon' vs the baseline MLP used to build the "
                            "training adversarials). That defended MLP reuses the baseline MLP preprocessor, so "
                            "re-transforming the saved raw adversarials with the baseline preprocessor lands them "
                            "in the correct standardized space.",
        "important_note": "Read attacked_rows_only for true_positive_only attacks: non-attacked rows are restored to "
                          "clean values, so their distances are zero by construction.",
        "attackable_note": "21 attackable raw features, all numeric. Frozen: n_courses (numeric), the 8 *_flag columns, "
                           "and the 4 one-hot categoricals (academic_year, seniority, degree_size, "
                           "highest_course_year_enrolled).",
        "case_settings": case_settings,
        "proximity_definition": "L1/L2/Linf distances between clean and adversarial preprocessed samples.",
        "sensitivity_definition": "Sum over numeric features of |adv - clean| / sigma_j, averaged over samples "
                                  "(numeric block is StandardScaler output). Flags and one-hot columns are grouped "
                                  "into the categorical/other change-rate, not the numeric sensitivity sum.",
        "files_created": {"summary_csv": summary_path, "feature_sensitivity_csv": feature_path,
                          "top15_sensitive_csv": top15_path, "excel": excel_path},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)

    print("\nSummary:")
    keep = [c for c in ["model", "split", "row_scope", "eval_attack_source", "eval_epsilon",
                        "attacked_rows_count", "dropout_to_nondropout_flips", "dropout_hiding_rate",
                        "mean_l2_attackable", "mean_linf_attackable", "mean_sensitivity",
                        "max_abs_change_frozen"] if c in summary_df.columns]
    print(summary_df[keep].to_string(index=False))

    print("\nSaved:")
    for p in [summary_path, feature_path, top15_path, excel_path, json_path]:
        print(p)
    print("\nDone.")


if __name__ == "__main__":
    main()