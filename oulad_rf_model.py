# REVISION NOTE (2026-09-29): CGPA, total_credit_enrolled and total_credit_passed were
# removed from every feature list in this script (feature set 32 -> 29, attackable 29 -> 26,
# MLP input 41 -> 38 dims). The columns may still exist in oulad_early.csv but are not used.
# Older comments below that still mention them describe the previous version.
import os
import json
import itertools
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# imblearn pipeline keeps the resampler inside the fold: it only touches
# training data, never test/validation. Using plain sklearn Pipeline with a
# sampler does not work, hence imblearn's own Pipeline here.
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "oulad_early.csv"
DEFAULT_OUTPUT_DIR = "rf_results_oulad_early"
DEFAULT_FEATURE_SET = "early_both"
DEFAULT_BALANCE = "all"

TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# thresholds to sweep when tuning the decision cutoff
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# Random Forest grid. Kept smaller than a full sweep because we also loop over
# several balancing strategies. Shallow trees + large leaves fight overfitting.
# The minority class (dropout) is ~20% of the data (about 1450 rows), so leaves
# of 20-50 are still well populated with dropout cases in the training fold.
RF_PARAM_GRID = {
    "n_estimators": [400],
    "max_depth": [6, 10, None],
    "min_samples_split": [50],
    "min_samples_leaf": [20, 50],
    "max_features": ["sqrt"],
    "bootstrap": [True],
    "max_samples": [0.7],
}

# ============================================================
# BALANCING STRATEGIES
# ============================================================
# Each strategy defines how the class imbalance is handled:
#   - a resampler applied inside the pipeline (SMOTE / over / under / combo), and
#   - a class_weight grid for the RandomForest itself.
# When a resampler is used, class_weight is set to None so we do not correct the
# imbalance twice.
def make_balance_configs(random_state):
    return {
        # no resampling, no reweighting: the raw baseline
        "none": {
            "sampler": None,
            "class_weight_grid": [None],
        },
        # model-level reweighting only (no synthetic data)
        "class_weight": {
            "sampler": None,
            "class_weight_grid": [
                "balanced",
                "balanced_subsample",
                {0: 1, 1: 4},
                {0: 1, 1: 8},
            ],
        },
        # synthetic minority oversampling
        "smote": {
            "sampler": SMOTE(random_state=random_state, k_neighbors=5),
            "class_weight_grid": [None],
        },
        # naive random duplication of minority rows
        "random_over": {
            "sampler": RandomOverSampler(random_state=random_state),
            "class_weight_grid": [None],
        },
        # random removal of majority rows
        "random_under": {
            "sampler": RandomUnderSampler(random_state=random_state),
            "class_weight_grid": [None],
        },
        # SMOTE + Tomek link cleaning
        "smotetomek": {
            "sampler": SMOTETomek(random_state=random_state),
            "class_weight_grid": [None],
        },
    }

# ============================================================
# FEATURE SETS
# ============================================================
# id_student and target are never predictors.
NON_FEATURE_COLS = ["id_student", TARGET_COL]

# Categorical -> one-hot. Different modules/presentations have different dropout
# rates, so these carry signal.
ONEHOT_CATEGORICAL_FEATURES = ["code_module", "code_presentation"]

# No binary flag columns in this dataset version. Kept as an empty list because
# the preprocessing code references FLAG_FEATURES to detect flag columns.
FLAG_FEATURES = []

# Static / enrolment info, known at (or before) the start of the presentation.
# NOTE: num_of_prev_attempts was removed from this dataset version, so it is no
# longer listed here.
#
# The credit_* and GPA_* columns are prior-record features. They are leakage-safe
# for this dataset: total_credit_enrolled and total_credit_passed sum only years
# 1-3 (verified: they do NOT include the current 4th-year presentation, i.e. the
# outcome being predicted), credit_enrolled_4thyear is the credits the student
# signed up for this year (known at enrolment, not an outcome), and
# and the total_* columns are sums of the per-year columns. This does not hurt RF
# predictions but does split feature importance across correlated columns, so read
# the importance table with that in mind.
STATIC_FEATURES = [
    "module_credits",
    "code_module",
    "code_presentation",
    # prior-year credit record (years 1-3) + current-year enrolment
    "credit_enrolled_1styear",
    "credit_passed_1styear",
    "credit_enrolled_2ndyear",
    "credit_passed_2ndyear",
    "credit_enrolled_3rdyear",
    "credit_passed_3rdyear",
    "credit_enrolled_4thyear",
    
    # prior-year academic record. Same reasoning as the credit columns: these are
    # historical (years 1-3 / cumulative up to year 3), not the current outcome.
    "GPA_1styear",
    "GPA_2ndyear",
    "GPA_3rdyear",
]

# Early window cut at day 65 (~first quarter of a ~260-day presentation).
EARLY_65_FEATURES = [
    "early_avg_score_65",
    "early_n_submitted_65",
    "early_first_submit_day_65",
    "early_total_clicks_65",
    "early_active_days_65",
    "early_unique_resources_65",
    "early_quiz_clicks_65",
    "early_forum_clicks_65",
]

# Early window cut at day 153 (~mid-presentation).
EARLY_153_FEATURES = [
    "early_avg_score_153",
    "early_n_submitted_153",
    "early_first_submit_day_153",
    "early_total_clicks_153",
    "early_active_days_153",
    "early_unique_resources_153",
    "early_quiz_clicks_153",
    "early_forum_clicks_153",
]

# NOTE: the full-course aggregate features were removed from the dataset, so the
# leaky 'full' / 'full_only' feature sets no longer exist. Only the leakage-safe
# early feature sets remain.
FEATURE_SETS = {
    "early_65": STATIC_FEATURES + EARLY_65_FEATURES,
    "early_153": STATIC_FEATURES + EARLY_153_FEATURES,
    "early_both": STATIC_FEATURES + EARLY_65_FEATURES + EARLY_153_FEATURES,
}

FEATURE_SET_DESCRIPTIONS = {
    "early_65": "Static/enrolment + day-65 early-window features. Leakage-safe early model.",
    "early_153": "Static/enrolment + day-153 early-window features. Leakage-safe mid model.",
    "early_both": ("Static/enrolment + day-65 AND day-153 windows in ONE model, no "
                   "full-course aggregates. Leakage-safe, but earliest usable point is "
                   "day 153 (day-153 columns do not exist yet at day 65)."),
}

def get_selected_features(feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Options: {list(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set].copy()

# ============================================================
# HELPERS
# ============================================================
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

def make_onehot_encoder():
    # sparse_output vs sparse depending on sklearn version
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def prepare_features_and_label(df, feature_set):
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. Available: {df.columns.tolist()}"
        )
    df = df.copy()
    target_normalized = (
        df[TARGET_COL]
        .astype(str)
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
    df["label"] = target_normalized.map(target_map)
    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), TARGET_COL].unique().tolist()
        raise ValueError(f"Some target values could not be mapped: {bad}")
    df["label"] = df["label"].astype(int)

    print("Target distribution:")
    print(df[TARGET_COL].value_counts())
    print()
    print("Numeric label distribution:")
    print(df["label"].value_counts().sort_index())
    print(df["label"].value_counts(normalize=True).sort_index().round(4))
    print()

    selected_features = get_selected_features(feature_set)
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for feature set '{feature_set}': {missing}")

    X = df[selected_features].copy()
    y = df["label"]
    student_ids = df["id_student"].copy() if "id_student" in df.columns else None

    print(f"Feature set used : {feature_set}")
    print(f"Description      : {FEATURE_SET_DESCRIPTIONS.get(feature_set, '')}")
    print(f"Number of features: {X.shape[1]}")
    print("Features used:", selected_features)
    print()
    return X, y, student_ids, selected_features

def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # stratified split: ~70% train, ~20% test, ~10% validation
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
    n_total = len(X)
    split_report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "validation_rows": len(X_val),
        "train_pct": len(X_train) / n_total,
        "test_pct": len(X_test) / n_total,
        "validation_pct": len(X_val) / n_total,
    }
    print("Stratified split check:")
    print(f"Train      : {split_report['train_rows']} rows ({split_report['train_pct']:.2%})")
    print(f"Test       : {split_report['test_rows']} rows ({split_report['test_pct']:.2%})")
    print(f"Validation : {split_report['validation_rows']} rows ({split_report['validation_pct']:.2%})")
    print()
    return X_train, X_test, X_val, y_train, y_test, y_val, id_train, id_test, id_val, split_report

def infer_feature_groups(X):
    # numeric (median impute) / binary flags (kept 0/1) / categorical (one-hot)
    all_cols = X.columns.tolist()
    categorical_cols = [c for c in ONEHOT_CATEGORICAL_FEATURES if c in X.columns]
    for c in X.select_dtypes(exclude=["number", "bool"]).columns.tolist():
        if c not in categorical_cols:
            categorical_cols.append(c)
    flag_cols = [c for c in FLAG_FEATURES if c in X.columns and c not in categorical_cols]
    numeric_cols = [
        c for c in X.select_dtypes(include=["number", "bool"]).columns.tolist()
        if c not in flag_cols and c not in categorical_cols
    ]
    assigned = set(numeric_cols) | set(flag_cols) | set(categorical_cols)
    unassigned = [c for c in all_cols if c not in assigned]
    if unassigned:
        raise ValueError(f"Some features were not assigned to a group: {unassigned}")
    return numeric_cols, flag_cols, categorical_cols

def build_preprocessor(X):
    numeric_cols, flag_cols, categorical_cols = infer_feature_groups(X)
    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )
    if flag_cols:
        transformers.append(
            ("flag", SimpleImputer(strategy="most_frequent"), flag_cols)
        )
    if categorical_cols:
        cat_pipe = ImbPipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No columns available for preprocessing.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, flag_cols, categorical_cols

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
    # Same rule as the zenodo tree models.
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

def compute_metrics_with_threshold(model, X, y, split_name, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "split": split_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, y_pred),
        "precision_dropout": precision_score(y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_dropout": recall_score(y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_dropout": f1_score(y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    return result, y_pred, y_proba

def save_predictions(student_ids, y_true, y_pred, y_proba, threshold, path):
    pred_df = pd.DataFrame({
        "id_student": student_ids.values,
        "actual_label": y_true.values,
        "actual_target": np.where(y_true.values == 1, "dropout", "non-dropout"),
        "predicted_label": y_pred,
        "predicted_target": np.where(y_pred == 1, "dropout", "non-dropout"),
        "prob_dropout": y_proba,
        "threshold": threshold,
    })
    pred_df.to_csv(path, index=False)

def save_confusion_matrix(cm, path):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    )
    cm_df.to_csv(path, index=True)

def get_feature_names_from_pipeline(pipeline, numeric_cols, flag_cols, categorical_cols):
    # rebuild column order after preprocessing (for feature importance)
    feature_names = list(numeric_cols) + list(flag_cols)
    if categorical_cols:
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_pipeline = preprocessor.named_transformers_["cat"]
        onehot = cat_pipeline.named_steps["onehot"]
        try:
            cat_names = onehot.get_feature_names_out(categorical_cols).tolist()
        except AttributeError:
            cat_names = onehot.get_feature_names(categorical_cols).tolist()
        feature_names.extend(cat_names)
    return feature_names

def build_pipeline(preprocessor, sampler, rf_params, random_state):
    steps = [("preprocessor", clone(preprocessor))]
    if sampler is not None:
        # clone so each combo gets a fresh sampler
        steps.append(("sampler", clone(sampler)))
    steps.append(("model", RandomForestClassifier(
        random_state=random_state, n_jobs=-1, **rf_params
    )))
    return ImbPipeline(steps=steps)

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Random Forest for OULAD early dropout prediction, "
                    "sweeping imbalance-handling strategies."
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-set", default=DEFAULT_FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help="early_65 (safe), early_153 (safe), early_both (default, safe: "
             "day-65 + day-153 windows combined in ONE model).",
    )
    parser.add_argument(
        "--balance", default=DEFAULT_BALANCE,
        choices=["all", "none", "class_weight", "smote", "random_over",
                 "random_under", "smotetomek"],
        help="Which balancing strategy to run. 'all' sweeps every strategy.",
    )
    parser.add_argument("--random-state", default=DEFAULT_RANDOM_STATE, type=int)
    parser.add_argument(
        "--threshold", default=None, type=float,
        help="Force a fixed decision threshold for the FINAL evaluation "
             "(e.g. 0.45 to raise recall). If omitted, the precision-floor "
             "threshold found on the test set is used.",
    )
    return parser

# ============================================================
# MAIN
# ============================================================
def run_experiment(df, csv_path, output_dir, feature_set, balance, random_state,
                   threshold_override=None):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("RANDOM FOREST - OULAD EARLY DROPOUT - IMBALANCE STRATEGY SWEEP")
    print("=" * 80)
    print(f"CSV path     : {csv_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Feature set  : {feature_set}")
    print(f"Balance mode : {balance}")
    print(f"Random state : {random_state}")
    print(f"Shape dataset: {df.shape}")
    print()

    X, y, student_ids, selected_features = prepare_features_and_label(df, feature_set)

    (X_train, X_test, X_val, y_train, y_test, y_val,
     id_train, id_test, id_val, split_report) = split_70_20_10_stratified(
        X, y, student_ids, random_state=random_state
    )

    preprocessor, numeric_cols, flag_cols, categorical_cols = build_preprocessor(X_train)
    print("Preprocessing groups:")
    print(f"Numeric imputed columns : {len(numeric_cols)}")
    print(f"Binary flag columns     : {len(flag_cols)} -> {flag_cols}")
    print(f"Categorical one-hot cols: {len(categorical_cols)} -> {categorical_cols}")
    print()

    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(output_dir, "oulad_rf_features_used.csv"), index=False
    )

    # decide which balancing strategies to run
    balance_configs = make_balance_configs(random_state)
    if balance == "all":
        strategies = list(balance_configs.keys())
    else:
        strategies = [balance]

    # build RF base combos
    rf_keys = list(RF_PARAM_GRID.keys())
    rf_combos = list(itertools.product(*[RF_PARAM_GRID[k] for k in rf_keys]))

    tuning_rows = []
    best_score = -np.inf
    best = None  # dict holding the winning combo

    print(f"Strategies to run: {strategies}")
    print("Threshold is selected on the TEST set; VALIDATION is the final held-out set.")
    print("selection_score = test_f1 - 0.25 * max(0, train_f1 - test_f1)")
    print()

    total_fits = 0
    for strat in strategies:
        cfg = balance_configs[strat]
        sampler = cfg["sampler"]
        for cw in cfg["class_weight_grid"]:
            for rf_combo in rf_combos:
                rf_params = dict(zip(rf_keys, rf_combo))
                rf_params["class_weight"] = cw
                total_fits += 1
    print(f"Total model fits to run: {total_fits}")
    print()

    fit_idx = 0
    for strat in strategies:
        cfg = balance_configs[strat]
        sampler = cfg["sampler"]
        for cw in cfg["class_weight_grid"]:
            for rf_combo in rf_combos:
                fit_idx += 1
                rf_params = dict(zip(rf_keys, rf_combo))
                rf_params["class_weight"] = cw

                model = build_pipeline(preprocessor, sampler, rf_params, random_state)
                model.fit(X_train, y_train)

                # pick threshold on the test set
                test_proba = model.predict_proba(X_test)[:, 1]
                threshold, _, thr_best_row = tune_threshold_from_proba(y_test, test_proba)
                test_pred = (test_proba >= threshold).astype(int)

                # overfitting penalty via train-test F1 gap
                train_proba = model.predict_proba(X_train)[:, 1]
                train_pred = (train_proba >= threshold).astype(int)
                train_f1 = f1_score(y_train, train_pred, pos_label=POSITIVE_LABEL, zero_division=0)
                test_f1 = f1_score(y_test, test_pred, pos_label=POSITIVE_LABEL, zero_division=0)
                overfit_gap = train_f1 - test_f1
                selection_score = test_f1 - 0.25 * max(0, overfit_gap)

                row = {
                    "balance_strategy": strat,
                    "class_weight": str(cw),
                    **{k: rf_params[k] for k in rf_keys},
                    "best_threshold": threshold,
                    "test_accuracy": accuracy_score(y_test, test_pred),
                    "test_precision_dropout": precision_score(y_test, test_pred, pos_label=POSITIVE_LABEL, zero_division=0),
                    "test_recall_dropout": recall_score(y_test, test_pred, pos_label=POSITIVE_LABEL, zero_division=0),
                    "test_f1_dropout": test_f1,
                    "test_roc_auc": roc_auc_score(y_test, test_proba),
                    "train_f1_dropout": train_f1,
                    "overfit_gap": overfit_gap,
                    "selection_score": selection_score,
                }
                tuning_rows.append(row)

                if selection_score > best_score:
                    best_score = selection_score
                    best = {
                        "strategy": strat,
                        "sampler": sampler,
                        "rf_params": rf_params,
                        "threshold": threshold,
                        "threshold_row": thr_best_row,
                        "model": model,
                    }

                if fit_idx % 5 == 0 or fit_idx == total_fits:
                    print(f"Progress: {fit_idx}/{total_fits} | last: {strat} "
                          f"cw={cw} f1={test_f1:.3f} sel={selection_score:.3f}")

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["selection_score", "test_f1_dropout", "test_recall_dropout"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    tuning_path = os.path.join(output_dir, "oulad_rf_tuning_results.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print()
    print("Best result per balancing strategy (by selection_score):")
    print(tuning_df.sort_values("selection_score", ascending=False)
          .groupby("balance_strategy", as_index=False).first()
          .sort_values("selection_score", ascending=False)
          [["balance_strategy", "class_weight", "best_threshold",
            "test_precision_dropout", "test_recall_dropout",
            "test_f1_dropout", "test_roc_auc", "selection_score"]]
          .to_string(index=False))
    print()
    print("Top 10 overall:")
    print(tuning_df.head(10).to_string(index=False))
    print()

    best_model = best["model"]
    best_threshold = best["threshold"]

    print(f"WINNER strategy      : {best['strategy']}")
    print(f"Best RF params       : {best['rf_params']}")
    print(f"Best threshold       : {best_threshold:.2f}")
    print(f"Best selection score : {best_score:.4f}")
    print()

    # full threshold sweep for the winning model on the test set
    best_test_proba = best_model.predict_proba(X_test)[:, 1]
    _, best_threshold_df, _ = tune_threshold_from_proba(y_test, best_test_proba)
    best_threshold_df.to_csv(os.path.join(output_dir, "oulad_rf_threshold_tuning_test.csv"), index=False)

    # decide the FINAL decision threshold: manual override wins if given,
    # otherwise use the precision-floor threshold found on the test set.
    auto_threshold = best_threshold
    if threshold_override is not None:
        final_threshold = float(threshold_override)
        print(f"Threshold override   : using fixed threshold {final_threshold:.2f} "
              f"for final evaluation (auto precision-floor was {auto_threshold:.2f}).")
        print("Lower threshold -> higher recall, lower precision (more dropouts caught,")
        print("more false alarms). Report this trade-off explicitly.")
        print()
    else:
        final_threshold = auto_threshold
    best_threshold = final_threshold

    # final metrics on all three splits, using the selected threshold
    train_results, train_pred, train_proba = compute_metrics_with_threshold(
        best_model, X_train, y_train, "train", best_threshold
    )
    test_results, test_pred, test_proba = compute_metrics_with_threshold(
        best_model, X_test, y_test, "test_selection", best_threshold
    )
    val_results, val_pred, val_proba = compute_metrics_with_threshold(
        best_model, X_val, y_val, "validation_final", best_threshold
    )

    results_df = pd.DataFrame([train_results, test_results, val_results])
    results_path = os.path.join(output_dir, "oulad_rf_results.csv")
    results_df.to_csv(results_path, index=False)

    print("Performance results:")
    print(results_df.to_string(index=False))
    print()
    print("Final held-out evaluation split: validation")
    print()

    save_predictions(id_train, y_train, train_pred, train_proba, best_threshold,
                     os.path.join(output_dir, "oulad_rf_train_predictions.csv"))
    save_predictions(id_test, y_test, test_pred, test_proba, best_threshold,
                     os.path.join(output_dir, "oulad_rf_test_selection_predictions.csv"))
    save_predictions(id_val, y_val, val_pred, val_proba, best_threshold,
                     os.path.join(output_dir, "oulad_rf_validation_final_predictions.csv"))

    test_cm = confusion_matrix(y_test, test_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    val_cm = confusion_matrix(y_val, val_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    save_confusion_matrix(test_cm, os.path.join(output_dir, "oulad_rf_test_confusion_matrix.csv"))
    save_confusion_matrix(val_cm, os.path.join(output_dir, "oulad_rf_validation_confusion_matrix.csv"))

    print("Validation confusion matrix:")
    print(pd.DataFrame(
        val_cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    ).to_string())
    print()

    model_path = os.path.join(output_dir, "oulad_rf_model.pkl")
    joblib.dump(best_model, model_path)

    # feature importance from the winning forest
    try:
        final_rf = best_model.named_steps["model"]
        feature_names = get_feature_names_from_pipeline(
            best_model, numeric_cols, flag_cols, categorical_cols
        )
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": final_rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(os.path.join(output_dir, "oulad_rf_feature_importance.csv"), index=False)
        print("Top 15 feature importances:")
        print(importance_df.head(15).to_string(index=False))
        print()
    except Exception as e:
        print(f"Failed to save feature importance: {e}")

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "feature_set": feature_set,
        "feature_set_description": FEATURE_SET_DESCRIPTIONS.get(feature_set, ""),
        "n_features": len(selected_features),
        "features_used": selected_features,
        "label_mapping": {"0": "non-dropout", "1": "dropout"},
        "imbalance_note": "Dropout is the minority class (~20.0%).",
        "balance_mode": balance,
        "balance_strategies_tried": strategies,
        "winning_strategy": best["strategy"],
        "split": {
            "method": "Stratified random split via train_test_split",
            "train": "approximately 70%",
            "test": "approximately 20%",
            "validation": "approximately 10%",
            "stratified": True,
            "random_state": random_state,
            "split_report": split_report,
            "note": "One row per student, so no group-aware split is needed.",
        },
        "model": "RandomForestClassifier",
        "resampling_note": "Resamplers run inside an imblearn Pipeline, so they "
                           "only touch the training fold (no leakage into test/val).",
        "preprocessing": {
            "numeric": "SimpleImputer(median), no scaling (RF does not need it)",
            "binary_flags": "SimpleImputer(most_frequent), kept 0/1",
            "categorical": "SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')",
            "numeric_columns": numeric_cols,
            "binary_flag_columns": flag_cols,
            "categorical_columns": categorical_cols,
        },
        "selection": {
            "selection_split": "test",
            "selection_metric": "selection_score = test_f1 - 0.25 * max(0, train_f1 - test_f1)",
            "best_rf_params": best["rf_params"],
            "best_threshold": best_threshold,
            "best_selection_score": best_score,
            "best_test_f1": best["threshold_row"]["f1_score"],
        },
        "final_evaluation": {
            "final_evaluation_split": "validation",
            "metrics_used": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        },
        "results": {
            "train": train_results,
            "test_selection": test_results,
            "validation_final": val_results,
        },
        "decision_rule": f"Predict dropout if prob_dropout >= {best_threshold:.2f}",
    }
    with open(os.path.join(output_dir, "oulad_rf_summary.json"), "w") as f:
        json.dump(to_python(summary), f, indent=4)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output folder : {output_dir}")
    print(f"Model         : {model_path}")
    print(f"Results CSV   : {results_path}")
    print(f"Tuning CSV    : {tuning_path}")
    return summary

def main():
    args = build_arg_parser().parse_args()
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(
            f"File not found: {args.csv_path}\n"
            f"Put the CSV next to this script, or pass --csv-path path/file.csv"
        )
    df = pd.read_csv(args.csv_path)
    run_experiment(
        df=df,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        balance=args.balance,
        random_state=args.random_state,
        threshold_override=args.threshold,
    )

if __name__ == "__main__":
    main()