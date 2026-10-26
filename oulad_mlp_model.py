# REVISION NOTE (2026-09-29): CGPA, total_credit_enrolled and total_credit_passed were
# removed from every feature list in this script (feature set 32 -> 29, attackable 29 -> 26,
# MLP input 41 -> 38 dims). The columns may still exist in oulad_early.csv but are not used.
# Older comments below that still mention them describe the previous version.
import os
import json
import copy
import random
import argparse
import itertools
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
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# imblearn samplers. Unlike the RF script, we do NOT use imblearn's Pipeline
# here, because the MLP is trained manually in PyTorch. Instead we resample the
# training matrix by hand AFTER preprocessing and BEFORE training. Test and
# validation sets are never resampled, so there is no leakage.
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEFAULT_RANDOM_STATE = 42
DEFAULT_CSV_PATH = "oulad_early.csv"
DEFAULT_OUTPUT_DIR = "mlp_results_oulad_early"
DEFAULT_FEATURE_SET = "early_both"
DEFAULT_BALANCE = "all"

TARGET_COL = "target"
NEGATIVE_LABEL = 0  # non-dropout
POSITIVE_LABEL = 1  # dropout

# thresholds to sweep when tuning the decision cutoff
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# minimum dropout precision we are willing to accept when picking the threshold.
# among all thresholds that keep dropout precision at or above this floor, we take
# the one with the highest recall. this trades some precision for catching more
# dropouts, but never lets precision drop below MIN_PRECISION on the split the
# threshold is tuned on (the test set). Same rule and value as zenodo_mlp_model.py.
MIN_PRECISION = 0.73

# network shape (funnel) + training hyperparameters
HIDDEN_DIMS = (128, 64, 32)

# Non-balancing hyperparameters that are always swept. The class_weight is NOT
# here: it is controlled per balancing strategy below (see make_balance_configs),
# so we never combine a resampler with class weights at the same time.
MLP_PARAM_GRID = {
    "learning_rate": [1e-3, 5e-4],
    "dropout": [0.20, 0.30],
    "weight_decay": [1e-4, 1e-3],
}
BATCH_SIZE = 128

# NO early stopping (Option 1: fixed training budget, analogous to a fixed
# n_estimators in rf_model.py / xgboost_model.py). The budget is NOT hardcoded:
# it is estimated ONCE by a short stratified k-fold CV inside the 70% train set
# (see estimate_fixed_epochs_via_cv). Every config is then trained on the FULL
# 70% train for exactly that many epochs, using the final weights. The test set
# is never used to decide when to stop, so it is treated exactly like in the
# tree scripts (threshold + strategy + hyperparameter selection only). The
# 70/20/10 split is preserved untouched. Same regime as zenodo_mlp_model.py.
EPOCHS_CAP = 100            # max epochs explored by the CV that picks the budget
CV_N_SPLITS_FOR_EPOCHS = 3  # folds in the single short epoch-budget CV
MIN_FIXED_EPOCHS = 5        # floor, guards against a noisy tiny argmin
# Reference config for the epoch-budget CV: the plainest baseline (no resampling,
# no class weights, mid-grid lr/dropout/weight_decay). One budget is estimated
# with this config and shared by every strategy/config in the main sweep.
EPOCH_CV_REFERENCE = {
    "learning_rate": 1e-3,
    "dropout": 0.30,
    "weight_decay": 1e-4,
    "class_weight": None,
}

# ============================================================
# BALANCING STRATEGIES
# ============================================================
# Mirrors the strategy sweep in rf_model.py. Each strategy defines:
#   - a sampler applied to the (already preprocessed) TRAINING matrix only, and
#   - a class_weight grid for the loss function.
# When a sampler is used, class_weight is forced to None so the imbalance is not
# corrected twice.
def make_balance_configs(random_state):
    return {
        # no resampling, no reweighting: the raw baseline
        "none": {
            "sampler": None,
            "class_weight_grid": [None],
        },
        # loss-level reweighting only (no synthetic data)
        "class_weight": {
            "sampler": None,
            "class_weight_grid": [
                "balanced",
                {0: 1, 1: 4},
                {0: 1, 1: 6},
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
# FEATURE SETS (identical definitions to rf_model.py)
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
# The credit_* and GPA_* columns are prior-record features and are leakage-safe
# for this dataset (verified against the data):
#   - total_credit_enrolled and total_credit_passed sum only years 1-3 (checked:
#     they do NOT include the current 4th-year presentation, i.e. the outcome).
#   - credit_enrolled_4thyear is the credits the student signed up for this year
#     (known at enrolment, not an outcome).
#   - CGPA reconstructs exactly as (GPA_1styear + GPA_2ndyear + GPA_3rdyear) / 3
#     across 100% of rows, so it is a simple mean of years 1-3 only, no year 4.
# the total_* columns are sums of the per-year columns, and CGPA is fully derived
# from the three GPA columns. This is fine for an MLP (it just learns around the
# redundancy) but keep it in mind when reasoning about which features matter.
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
    # prior-year academic record (years 1-3 / cumulative up to year 3)
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
    "early_both": ("Static/enrolment + day-65 AND day-153 windows, no full-course "
                   "aggregates. Leakage-safe, but earliest usable point is day 153."),
}

def get_selected_features(feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Options: {list(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set].copy()

# ============================================================
# MODEL
# ============================================================
class DropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=0.30):
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
def set_global_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    # note: benchmark=True favors speed over bit-exact reproducibility on GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

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
    # normalize target text before mapping to 0/1
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
    y = df["label"].copy()
    student_ids = df["id_student"].copy() if "id_student" in df.columns else None

    print(f"Feature set used : {feature_set}")
    print(f"Description      : {FEATURE_SET_DESCRIPTIONS.get(feature_set, '')}")
    print(f"Number of features: {X.shape[1]}")
    print("Features used:", selected_features)
    print()
    return X, y, student_ids, selected_features

def split_70_20_10_stratified(X, y, student_ids=None, random_state=42):
    # stratified split ~70/20/10; SAME logic and random_state as rf_model.py so
    # the RF and MLP models are trained/evaluated on identical partitions.
    # The split depends only on n_samples, y (stratify) and random_state, not on
    # which feature columns are in X, so all feature sets share the same split.
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
    # numeric (scaled) / binary flags (kept 0/1) / categorical (one-hot)
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
        # MLP needs scaling (unlike RF)
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_transformer, numeric_cols))
    if flag_cols:
        flag_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
        transformers.append(("flag", flag_transformer, flag_cols))
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ])
        transformers.append(("cat", categorical_transformer, categorical_cols))
    if not transformers:
        raise ValueError("No columns available for preprocessing.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, flag_cols, categorical_cols

def transform_to_float32(preprocessor, X):
    arr = preprocessor.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return arr.astype(np.float32)

def resample_training_matrix(sampler, X_train_np, y_train_np, random_state):
    # Apply the sampler to the preprocessed TRAINING matrix only.
    # Note: SMOTE/SMOTETomek interpolate across ALL columns, including the
    # one-hot module/presentation columns and the 0/1 flag columns, so synthetic
    # rows can contain fractional values there. This is the same behaviour as the
    # RF imblearn pipeline (preprocessor -> sampler -> model) and is standard
    # practice; just be aware of it when interpreting results.
    if sampler is None:
        return X_train_np, y_train_np
    sampler = copy.deepcopy(sampler)
    X_res, y_res = sampler.fit_resample(X_train_np, y_train_np)
    return X_res.astype(np.float32), np.asarray(y_res).astype(int)

def make_criterion(class_weight, y_train_used, device):
    # build the loss, applying class weights to handle imbalance.
    # y_train_used is the label vector actually trained on (already resampled if
    # a sampler was used), so 'balanced' reflects the real training balance.
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

def _build_loader(X_train_np, y_train_np, random_state):
    X_tr_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train_np, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(random_state)
    # drop last partial batch (BatchNorm breaks on a batch of size 1)
    drop_last = len(y_train_np) > BATCH_SIZE
    return DataLoader(
        TensorDataset(X_tr_tensor, y_tr_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=drop_last,
    )

def train_mlp_fixed(params, X_train_np, y_train_np, input_dim, device,
                    random_state, n_epochs, record_history=False):
    # Train for a FIXED number of epochs on the full training matrix. There is
    # NO early stopping and NO best-weight restoration: the returned model is the
    # one after n_epochs, and the test set is never consulted during training.
    # same seed per combo so comparisons differ only by hyperparameters
    set_global_seed(random_state)
    model = DropoutMLP(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=params["dropout"],
    ).to(device)
    criterion = make_criterion(params["class_weight"], y_train_np, device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    loader = _build_loader(X_train_np, y_train_np, random_state)
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        # standard training step: zero_grad -> forward -> loss -> backward -> step
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        if record_history:
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
            })
    return model, history

def _val_loss_curve(params, X_tr_np, y_tr_np, X_va_np, y_va_np, input_dim,
                    device, random_state, n_epochs):
    # Train the reference config and return the per-epoch validation loss list.
    # Used ONLY to estimate the fixed epoch budget: runs the full n_epochs with
    # no early stopping so the whole loss curve is available for argmin.
    set_global_seed(random_state)
    model = DropoutMLP(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=params["dropout"],
    ).to(device)
    criterion = make_criterion(params["class_weight"], y_tr_np, device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    loader = _build_loader(X_tr_np, y_tr_np, random_state)
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
            vloss = float(criterion(model(X_va_tensor), y_va_tensor).item())
        val_losses.append(vloss)
    return val_losses

def estimate_fixed_epochs_via_cv(X_train, y_train, random_state,
                                 n_splits=CV_N_SPLITS_FOR_EPOCHS,
                                 epochs_cap=EPOCHS_CAP):
    # ONE short stratified k-fold CV INSIDE the train set to choose a single
    # fixed epoch budget shared by every config in the main sweep. Preprocessing
    # is refit inside each fold (on the fold-train only), so nothing leaks and no
    # test/val data is touched. The chosen budget is the epoch that minimizes the
    # mean validation loss averaged across folds.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    curve_sum = np.zeros(epochs_cap)
    curve_cnt = np.zeros(epochs_cap)
    print(f"Estimating fixed epoch budget via {n_splits}-fold CV inside train "
          f"(reference config {EPOCH_CV_REFERENCE}):")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr_fold = X_train.iloc[tr_idx]
        y_tr_fold = y_train.iloc[tr_idx]
        X_va_fold = X_train.iloc[va_idx]
        y_va_fold = y_train.iloc[va_idx]
        pre, _, _, _ = build_preprocessor(X_tr_fold)
        X_tr_np = pre.fit_transform(X_tr_fold)
        if hasattr(X_tr_np, "toarray"):
            X_tr_np = X_tr_np.toarray()
        X_tr_np = X_tr_np.astype(np.float32)
        X_va_np = transform_to_float32(pre, X_va_fold)
        y_tr_np = y_tr_fold.values.astype(int)
        y_va_np = y_va_fold.values.astype(int)
        losses = _val_loss_curve(
            EPOCH_CV_REFERENCE, X_tr_np, y_tr_np, X_va_np, y_va_np,
            X_tr_np.shape[1], device, random_state, epochs_cap,
        )
        for e, l in enumerate(losses):
            curve_sum[e] += l
            curve_cnt[e] += 1
        print(f"  fold {fold}/{n_splits}: min val loss at epoch {int(np.argmin(losses)) + 1}")
    mean_curve = curve_sum / np.maximum(curve_cnt, 1)
    best_epoch = max(int(np.argmin(mean_curve)) + 1, MIN_FIXED_EPOCHS)
    print(f"Chosen fixed epoch budget: {best_epoch} "
          f"(argmin of mean CV val loss, floored at {MIN_FIXED_EPOCHS})")
    print()
    return best_epoch, mean_curve.tolist()

def predict_proba_mlp(model, X_np, device, batch_size=4096):
    model.eval()
    probs = []
    dataset = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.append(batch_probs.cpu().numpy())
    return np.concatenate(probs)

def tune_threshold_from_proba(y_true, y_proba, min_precision=MIN_PRECISION):
    # sweep thresholds, then among the ones that keep dropout precision at or
    # above min_precision, pick the highest recall (f1 then precision break ties).
    # goal: catch as many dropouts as possible while precision stays >= floor.
    # if no threshold reaches the precision floor, fall back to best f1 so the
    # function still returns a usable threshold instead of failing.
    # Same rule as zenodo_mlp_model.py.
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

def compute_metrics_with_threshold(model, X_np, y, split_name, threshold, device):
    y_np = np.asarray(y).astype(int)
    y_proba = predict_proba_mlp(model, X_np, device)
    y_pred = (y_proba >= threshold).astype(int)
    result = {
        "split": split_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_np, y_pred),
        "precision_dropout": precision_score(y_np, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_dropout": recall_score(y_np, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_dropout": f1_score(y_np, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y_np, y_proba),
    }
    return result, y_pred, y_proba

def save_predictions(student_ids, y_true, y_pred, y_proba, threshold, path):
    y_true_np = np.asarray(y_true).astype(int)
    pred_df = pd.DataFrame({
        "id_student": student_ids.values,
        "actual_label": y_true_np,
        "actual_target": np.where(y_true_np == 1, "dropout", "non-dropout"),
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

def get_onehot_feature_names(preprocessor, categorical_cols):
    if not categorical_cols:
        return []
    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]
    try:
        return onehot.get_feature_names_out(categorical_cols).tolist()
    except AttributeError:
        return onehot.get_feature_names(categorical_cols).tolist()

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="MLP for OULAD early dropout prediction, sweeping "
                    "imbalance-handling strategies (SMOTE / class_weight / "
                    "over / under / smotetomek)."
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
             "(e.g. 0.35 to raise recall). If omitted, the precision-floor "
             "threshold (max recall s.t. precision >= MIN_PRECISION) found on "
             "the test set is used.",
    )
    return parser

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(df, csv_path, output_dir, feature_set, balance, random_state,
                   threshold_override=None):
    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MLP - OULAD EARLY DROPOUT - IMBALANCE STRATEGY SWEEP")
    print("=" * 80)
    print(f"CSV path     : {csv_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Feature set  : {feature_set}")
    print(f"Balance mode : {balance}")
    print(f"Random state : {random_state}")
    print(f"Device       : {device}")
    print(f"Shape dataset: {df.shape}")
    print()

    X, y, student_ids, selected_features = prepare_features_and_label(df, feature_set)

    (X_train, X_test, X_val, y_train, y_test, y_val,
     id_train, id_test, id_val, split_report) = split_70_20_10_stratified(
        X, y, student_ids, random_state=random_state
    )

    preprocessor, numeric_cols, flag_cols, categorical_cols = build_preprocessor(X_train)
    print("Preprocessing groups:")
    print(f"Numeric scaled columns  : {len(numeric_cols)}")
    print(f"Binary flag columns     : {len(flag_cols)} -> {flag_cols}")
    print(f"Categorical one-hot cols: {len(categorical_cols)} -> {categorical_cols}")
    print()

    # fit preprocessing on train only, then apply to test/val
    X_train_np = preprocessor.fit_transform(X_train)
    if hasattr(X_train_np, "toarray"):
        X_train_np = X_train_np.toarray()
    X_train_np = X_train_np.astype(np.float32)
    X_test_np = transform_to_float32(preprocessor, X_test)
    X_val_np = transform_to_float32(preprocessor, X_val)

    y_train_np = y_train.values.astype(int)
    y_test_np = y_test.values.astype(int)
    y_val_np = y_val.values.astype(int)

    input_dim_base = X_train_np.shape[1]

    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(output_dir, "oulad_mlp_features_used.csv"), index=False
    )
    preprocessor_path = os.path.join(output_dir, "oulad_mlp_preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)

    # decide which balancing strategies to run
    balance_configs = make_balance_configs(random_state)
    if balance == "all":
        strategies = list(balance_configs.keys())
    else:
        strategies = [balance]

    # non-balancing hyperparameter combos (lr / dropout / weight_decay)
    mlp_keys = list(MLP_PARAM_GRID.keys())
    mlp_combos = list(itertools.product(*[MLP_PARAM_GRID[k] for k in mlp_keys]))

    # count total fits for the progress bar
    total_fits = 0
    for strat in strategies:
        cfg = balance_configs[strat]
        total_fits += len(cfg["class_weight_grid"]) * len(mlp_combos)

    print(f"Strategies to run: {strategies}")
    print("Resampling touches TRAINING data only. Threshold is selected on the")
    print("TEST set; VALIDATION is the final held-out set.")
    print("selection_score = test_f1 - 0.25 * max(0, train_f1 - test_f1)")
    print(f"Total model fits to run: {total_fits}")
    print()

    # Estimate the fixed epoch budget ONCE via CV inside the 70% train set.
    # Shared by every config below; test/validation are never touched here.
    fixed_epochs, cv_mean_curve = estimate_fixed_epochs_via_cv(
        X_train, y_train, random_state
    )

    tuning_rows = []
    best_score = -np.inf
    best = None
    fit_idx = 0

    for strat in strategies:
        cfg = balance_configs[strat]
        sampler = cfg["sampler"]
        # resample ONCE per strategy (deterministic given random_state), then
        # reuse for all lr/dropout/weight_decay combos in this strategy.
        X_tr_used, y_tr_used = resample_training_matrix(
            sampler, X_train_np, y_train_np, random_state
        )
        for cw in cfg["class_weight_grid"]:
            for mlp_combo in mlp_combos:
                fit_idx += 1
                params = dict(zip(mlp_keys, mlp_combo))
                params["class_weight"] = cw

                model, history = train_mlp_fixed(
                    params=params,
                    X_train_np=X_tr_used,
                    y_train_np=y_tr_used,
                    input_dim=input_dim_base,
                    device=device,
                    random_state=random_state,
                    n_epochs=fixed_epochs,
                    record_history=True,
                )

                # pick threshold on the test set
                test_proba = predict_proba_mlp(model, X_test_np, device)
                threshold, _, thr_best_row = tune_threshold_from_proba(y_test_np, test_proba)
                test_pred = (test_proba >= threshold).astype(int)

                # train-test F1 gap as an overfitting penalty.
                # NOTE: train_f1 is measured on the (possibly resampled) training
                # matrix the model actually saw, so on balanced data.
                train_proba = predict_proba_mlp(model, X_tr_used, device)
                train_pred = (train_proba >= threshold).astype(int)
                train_f1 = f1_score(y_tr_used, train_pred, pos_label=POSITIVE_LABEL, zero_division=0)
                test_f1 = f1_score(y_test_np, test_pred, pos_label=POSITIVE_LABEL, zero_division=0)
                overfit_gap = train_f1 - test_f1
                selection_score = test_f1 - 0.25 * max(0, overfit_gap)

                row = {
                    "balance_strategy": strat,
                    "class_weight": str(cw),
                    "learning_rate": params["learning_rate"],
                    "dropout": params["dropout"],
                    "weight_decay": params["weight_decay"],
                    "fixed_epochs": fixed_epochs,
                    "best_threshold": threshold,
                    "test_accuracy": accuracy_score(y_test_np, test_pred),
                    "test_precision_dropout": precision_score(y_test_np, test_pred, pos_label=POSITIVE_LABEL, zero_division=0),
                    "test_recall_dropout": recall_score(y_test_np, test_pred, pos_label=POSITIVE_LABEL, zero_division=0),
                    "test_f1_dropout": test_f1,
                    "test_roc_auc": roc_auc_score(y_test_np, test_proba),
                    "train_f1_dropout": train_f1,
                    "overfit_gap": overfit_gap,
                    "selection_score": selection_score,
                }
                tuning_rows.append(row)

                if selection_score > best_score:
                    best_score = selection_score
                    best = {
                        "strategy": strat,
                        "params": copy.deepcopy(params),
                        "model": copy.deepcopy(model),
                        "threshold": threshold,
                        "threshold_row": thr_best_row,
                        "history": copy.deepcopy(history),
                        "fixed_epochs": fixed_epochs,
                    }

                if fit_idx % 5 == 0 or fit_idx == total_fits:
                    print(f"Progress: {fit_idx}/{total_fits} | last: {strat} "
                          f"cw={cw} f1={test_f1:.3f} sel={selection_score:.3f}")

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["selection_score", "test_f1_dropout", "test_recall_dropout"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    tuning_path = os.path.join(output_dir, "oulad_mlp_tuning_results.csv")
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
    auto_threshold = best["threshold"]

    print(f"WINNER strategy      : {best['strategy']}")
    print(f"Best MLP params      : {best['params']}")
    print(f"Best threshold       : {auto_threshold:.2f}")
    print(f"Best selection score : {best_score:.4f}")
    print()

    pd.DataFrame(best["history"]).to_csv(
        os.path.join(output_dir, "oulad_mlp_training_history.csv"), index=False
    )

    # full threshold sweep for the winning model on the test set
    best_test_proba = predict_proba_mlp(best_model, X_test_np, device)
    _, best_threshold_df, _ = tune_threshold_from_proba(y_test_np, best_test_proba)
    best_threshold_df.to_csv(os.path.join(output_dir, "oulad_mlp_threshold_tuning_test.csv"), index=False)

    # decide the FINAL decision threshold: manual override wins if given,
    # otherwise use the precision-floor threshold found on the test set.
    if threshold_override is not None:
        final_threshold = float(threshold_override)
        print(f"Threshold override   : using fixed threshold {final_threshold:.2f} "
              f"for final evaluation (auto precision-floor was {auto_threshold:.2f}).")
        print("Lower threshold -> higher recall, lower precision (more dropouts caught,")
        print("more false alarms). Report this trade-off explicitly.")
        print()
    else:
        final_threshold = auto_threshold

    # final metrics on all three splits, using the selected threshold
    train_results, train_pred, train_proba = compute_metrics_with_threshold(
        best_model, X_train_np, y_train_np, "train", final_threshold, device
    )
    test_results, test_pred, test_proba = compute_metrics_with_threshold(
        best_model, X_test_np, y_test_np, "test_selection", final_threshold, device
    )
    val_results, val_pred, val_proba = compute_metrics_with_threshold(
        best_model, X_val_np, y_val_np, "validation_final", final_threshold, device
    )

    results_df = pd.DataFrame([train_results, test_results, val_results])
    results_path = os.path.join(output_dir, "oulad_mlp_results.csv")
    results_df.to_csv(results_path, index=False)

    print("Performance results:")
    print(results_df.to_string(index=False))
    print()
    print("Final held-out evaluation split: validation")
    print()

    save_predictions(id_train, y_train_np, train_pred, train_proba, final_threshold,
                     os.path.join(output_dir, "oulad_mlp_train_predictions.csv"))
    save_predictions(id_test, y_test_np, test_pred, test_proba, final_threshold,
                     os.path.join(output_dir, "oulad_mlp_test_selection_predictions.csv"))
    save_predictions(id_val, y_val_np, val_pred, val_proba, final_threshold,
                     os.path.join(output_dir, "oulad_mlp_validation_final_predictions.csv"))

    test_cm = confusion_matrix(y_test_np, test_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    val_cm = confusion_matrix(y_val_np, val_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    save_confusion_matrix(test_cm, os.path.join(output_dir, "oulad_mlp_test_confusion_matrix.csv"))
    save_confusion_matrix(val_cm, os.path.join(output_dir, "oulad_mlp_validation_confusion_matrix.csv"))

    print("Validation confusion matrix:")
    print(pd.DataFrame(
        val_cm,
        index=["Actual_non_dropout", "Actual_dropout"],
        columns=["Pred_non_dropout", "Pred_dropout"],
    ).to_string())
    print()

    # save the winning model checkpoint
    model_path = os.path.join(output_dir, "oulad_mlp_model.pt")
    onehot_names = get_onehot_feature_names(preprocessor, categorical_cols)
    checkpoint = {
        "model_state_dict": best_model.state_dict(),
        "input_dim": input_dim_base,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": best["params"]["dropout"],
        "selected_features": selected_features,
        "feature_set": feature_set,
        "winning_strategy": best["strategy"],
        "best_params": best["params"],
        "best_threshold": final_threshold,
        "preprocessing_order": {
            "numeric_scaled": numeric_cols,
            "binary_flags": flag_cols,
            "onehot": onehot_names,
        },
        "label_mapping": {"0": "non-dropout", "1": "dropout"},
    }
    torch.save(checkpoint, model_path)

    summary = {
        "dataset": csv_path,
        "output_dir": output_dir,
        "feature_set": feature_set,
        "feature_set_description": FEATURE_SET_DESCRIPTIONS.get(feature_set, ""),
        "n_features": len(selected_features),
        "n_features_after_preprocessing": input_dim_base,
        "features_used": selected_features,
        "label_mapping": {"0": "non-dropout", "1": "dropout"},
        "imbalance_note": "Dropout is the minority class (~20.0%).",
        "balance_mode": balance,
        "balance_strategies_tried": strategies,
        "winning_strategy": best["strategy"],
        "resampling_note": ("Resamplers are applied to the preprocessed TRAINING "
                            "matrix only (never test/val), so there is no leakage. "
                            "SMOTE/SMOTETomek interpolate across one-hot and flag "
                            "columns, producing fractional synthetic values there."),
        "split": {
            "method": "Stratified random split via train_test_split",
            "train": "approximately 70%",
            "test": "approximately 20%",
            "validation": "approximately 10%",
            "stratified": True,
            "group_aware": False,
            "random_state": random_state,
            "split_report": split_report,
            "note": ("One row per student, so no group-aware split is needed. "
                     "Same split logic and random_state as rf_model.py."),
            "split_usage": {
                "train": "model fitting",
                "test": "hyperparameter + strategy selection, early stopping, threshold selection",
                "validation": "final held-out evaluation",
            },
        },
        "model": "DropoutMLP",
        "architecture": {
            "input_dim": input_dim_base,
            "hidden_dims": list(HIDDEN_DIMS),
            "output_dim": 2,
            "activation": "ReLU",
            "normalization": "BatchNorm1d",
            "dropout": best["params"]["dropout"],
        },
        "preprocessing": {
            "numeric": "SimpleImputer(median) + StandardScaler fitted on train only",
            "binary_flags": "SimpleImputer(most_frequent), kept 0/1",
            "categorical": "SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')",
            "numeric_columns": numeric_cols,
            "binary_flag_columns": flag_cols,
            "categorical_columns": categorical_cols,
            "preprocessor_file": preprocessor_path,
        },
        "training": {
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "batch_size": BATCH_SIZE,
            "early_stopping": "none (fixed epoch budget, final weights)",
            "epochs_cap_explored": EPOCHS_CAP,
            "cv_splits_for_epoch_budget": CV_N_SPLITS_FOR_EPOCHS,
            "fixed_epochs": best["fixed_epochs"],
            "epoch_budget_rule": (
                f"Chosen once by a {CV_N_SPLITS_FOR_EPOCHS}-fold stratified CV "
                f"inside the 70% train set (argmin of mean CV val loss, floored "
                f"at {MIN_FIXED_EPOCHS}). Test/validation never used for training. "
                f"Same regime as zenodo_mlp_model.py."
            ),
            "device": str(device),
        },
        "selection": {
            "selection_split": "test",
            "selection_metric": "selection_score = test_f1 - 0.25 * max(0, train_f1 - test_f1)",
            "best_strategy": best["strategy"],
            "best_params": best["params"],
            "best_threshold": final_threshold,
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
        "decision_rule": f"Predict dropout if prob_dropout >= {final_threshold:.2f}",
    }
    with open(os.path.join(output_dir, "oulad_mlp_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=4, ensure_ascii=False)

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