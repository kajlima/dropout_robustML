# REVISION NOTE (2026-09-29): CGPA, total_credit_enrolled and total_credit_passed were
# removed from every feature list in this script (feature set 32 -> 29, attackable 29 -> 26,
# MLP input 41 -> 38 dims). The columns may still exist in oulad_early.csv but are not used.
# Older comments below that still mention them describe the previous version.
from __future__ import annotations

import json
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# WHAT THIS SCRIPT DOES
# ============================================================
# For each model it collects the students that tell the recovery story, using
# four prediction "phases" for the same student:
#
#   phase 1  original model, clean input        -> should catch the dropout
#   phase 2  original model, adversarial input  -> dropout gets HIDDEN (attacked)
#   phase 3  defended model, clean input        -> should still catch the dropout
#                                                   (sanity: the defence did not
#                                                    just turn the model paranoid)
#   phase 4  defended model, adversarial input  -> dropout stays caught (recovered)
#
# Two flags summarise it:
#   "successfully attacked"  = phase 1 caught it AND phase 2 hid it
#                              (a real dropout, seen by the clean model, hidden
#                               once the undefended attack is applied).
#   "successfully recovered" = a successfully-attacked student whom the DEFENDED
#                              model still predicts as dropout under attack
#                              (phase 4 == dropout).
#
# Phase 3 is supporting evidence only; it does NOT change the definition of
# "successfully recovered". It exists so a recovery can be shown to be genuine
# (the defended model catches this student even on clean data) rather than an
# artefact of a model that predicts dropout for almost everyone.
#
# Recovery here is the ADVERSARIAL-TRAINING defence only. Detect-and-default has
# its own per-student trace files with a different recovery mechanism
# (abstain / default-to-dropout) and is handled separately.
#
# Attack per model (all epsilons = 1.0 in this project):
#   Random Forest -> FGSM   (transfer from the source MLP gradients)
#   XGBoost       -> FGSM   (transfer from the source MLP gradients)
#   MLP           -> PGD    (white-box on the MLP itself)
#
# NOTE on diff_* columns: the adversarial raw values come from the UNDEFENDED
# attack files (the perturbation that produced the phase-2 flip). That is
# deliberate and matches "successfully_attacked". The defended model is evaluated
# in phase 4 against a separately-crafted attack; those perturbations are not the
# ones shown in diff_*.

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = Path("oulad_early.csv")
OUTPUT_DIR = Path("oulad_attack_recovery_students_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EXCEL = OUTPUT_DIR / "oulad_successfully_attacked_and_recovered_students.xlsx"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "oulad_attack_recovery_summary.csv"
OUTPUT_JSON = OUTPUT_DIR / "oulad_attack_recovery_file_map.json"

TARGET_COL = "target"
STUDENT_COL = "id_student"
POSITIVE_LABEL = 1  # dropout
SPLITS = ("test", "validation")

# ============================================================
# FEATURE SETS (oulad_early.csv, "early_both" feature set)
# ============================================================
# Full predictor set used by the rf / xgb / mlp models. id_student and target are
# never predictors. code_module and code_presentation are one-hot categoricals.
FULL_FEATURES = [
    # static / enrolment
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
    # prior-year academic record
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

# Only these raw features move under attack. Everything else (code_module,
# code_presentation, module_credits) stays frozen. Matches ATTACKABLE_FEATURES in
# fgsm_rf.py / fgsm_xgboost.py / pgd_mlp.py.
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

# ============================================================
# OUTPUT FOLDERS PRODUCED BY THE ATTACK / DEFENCE SCRIPTS
# ============================================================
RF_ATTACK_DIR = "fgsm_rf_oulad_early_results"
RF_DEFENCE_DIR = "rf_adversarial_training_oulad_early_results"

XGB_ATTACK_DIR = "fgsm_xgb_oulad_early_results"
XGB_DEFENCE_DIR = "xgb_adversarial_training_oulad_early_results"

MLP_ATTACK_DIR = "pgd_mlp_oulad_early_results"
MLP_DEFENCE_DIR = "mlp_adversarial_training_oulad_early_results"

# ============================================================
# EPSILONS USED IN THE FINAL RUNS
# ============================================================
# Confirmed from the *_attack_summary.json and *_adversarial_training_summary.json
# files: every run in this project uses epsilon 1.0.
RF_ATTACK_EPS = 1.0
RF_DEFENCE_TRAIN_EPS = 1.0
RF_DEFENCE_EVAL_EPS = 1.0

XGB_ATTACK_EPS = 1.0
XGB_DEFENCE_TRAIN_EPS = 1.0
XGB_DEFENCE_EVAL_EPS = 1.0

MLP_ATTACK_EPS = 1.0
MLP_DEFENCE_TRAIN_EPS = 1.0
MLP_DEFENCE_EVAL_EPS = 1.0


def eps_tag(value: float) -> str:
    # 1.0 -> '1p0', same tag the attack and defence scripts write
    return str(float(value)).replace(".", "p")


# per-model file paths. {split} is filled in; epsilons come from the constants
# above, so each entry points at exactly one run.
#   defence_pred       -> defended model under attack   (phase 4)
#   defence_clean_pred -> defended model on clean input (phase 3)
MODEL_FILES = {
    "Random Forest": {
        "attack": "FGSM",
        "attack_pred": [f"{RF_ATTACK_DIR}/oulad_rf_fgsm_{{split}}_predictions_eps_{eps_tag(RF_ATTACK_EPS)}.csv"],
        "attack_raw": [f"{RF_ATTACK_DIR}/oulad_X_{{split}}_fgsm_raw_eps_{eps_tag(RF_ATTACK_EPS)}.csv"],
        "defence_pred": [
            f"{RF_DEFENCE_DIR}/oulad_rf_defended_fgsm_{{split}}_predictions"
            f"_train_eps_{eps_tag(RF_DEFENCE_TRAIN_EPS)}_eval_eps_{eps_tag(RF_DEFENCE_EVAL_EPS)}.csv"
        ],
        "defence_clean_pred": [
            f"{RF_DEFENCE_DIR}/oulad_rf_defended_clean_{{split}}_predictions_train_eps_{eps_tag(RF_DEFENCE_TRAIN_EPS)}.csv"
        ],
        "clean_prefix": "rf",
    },
    "XGBoost": {
        "attack": "FGSM",
        "attack_pred": [f"{XGB_ATTACK_DIR}/oulad_xgb_fgsm_{{split}}_predictions_eps_{eps_tag(XGB_ATTACK_EPS)}.csv"],
        "attack_raw": [f"{XGB_ATTACK_DIR}/oulad_X_{{split}}_fgsm_raw_eps_{eps_tag(XGB_ATTACK_EPS)}.csv"],
        "defence_pred": [
            f"{XGB_DEFENCE_DIR}/oulad_xgb_defended_fgsm_{{split}}_predictions"
            f"_train_eps_{eps_tag(XGB_DEFENCE_TRAIN_EPS)}_eval_eps_{eps_tag(XGB_DEFENCE_EVAL_EPS)}.csv"
        ],
        "defence_clean_pred": [
            f"{XGB_DEFENCE_DIR}/oulad_xgb_defended_clean_{{split}}_predictions_train_eps_{eps_tag(XGB_DEFENCE_TRAIN_EPS)}.csv"
        ],
        "clean_prefix": "xgb",
    },
    "MLP": {
        "attack": "PGD",
        "attack_pred": [f"{MLP_ATTACK_DIR}/oulad_mlp_pgd_{{split}}_predictions_eps_{eps_tag(MLP_ATTACK_EPS)}.csv"],
        "attack_raw": [f"{MLP_ATTACK_DIR}/oulad_X_{{split}}_pgd_raw_eps_{eps_tag(MLP_ATTACK_EPS)}.csv"],
        "defence_pred": [
            f"{MLP_DEFENCE_DIR}/oulad_mlp_defended_pgd_{{split}}_predictions"
            f"_train_eps_{eps_tag(MLP_DEFENCE_TRAIN_EPS)}_eval_eps_{eps_tag(MLP_DEFENCE_EVAL_EPS)}.csv"
        ],
        "defence_clean_pred": [
            f"{MLP_DEFENCE_DIR}/oulad_mlp_defended_clean_{{split}}_predictions_train_eps_{eps_tag(MLP_DEFENCE_TRAIN_EPS)}.csv"
        ],
        "clean_prefix": "mlp",
    },
}

# short sheet-name prefix per model
SHEET_PREFIX = {
    "Random Forest": "RF",
    "XGBoost": "XGB",
    "MLP": "MLP",
}


# ============================================================
# HELPERS
# ============================================================
def label_text(value) -> str:
    # NA -> "Missing" so a student absent from a defended file is visible, not
    # silently mislabelled.
    if pd.isna(value):
        return "Missing"
    try:
        return "Dropout" if int(value) == POSITIVE_LABEL else "Non-dropout"
    except Exception:
        return "Unknown"


def eps_from_name(path: str) -> Dict[str, Optional[float]]:
    # pull eval / train / plain epsilon out of a filename tag like eps_1p0 or
    # train_eps_1p0_eval_eps_1p0
    out = {"epsilon": None, "train_epsilon": None, "eval_epsilon": None}
    for key, pat in [("train_epsilon", r"train_eps_([0-9]+p[0-9]+)"),
                     ("eval_epsilon", r"eval_eps_([0-9]+p[0-9]+)"),
                     ("epsilon", r"(?<!train_)(?<!eval_)eps_([0-9]+p[0-9]+)")]:
        m = re.search(pat, path)
        if m:
            out[key] = float(m.group(1).replace("p", "."))
    return out


def resolve_one(patterns: List[str], split: str) -> Optional[Path]:
    # return the single file matching the patterns, or None. raise on more than one
    hits: List[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat.format(split=split)))
    hits = sorted(set(hits))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            "more than one file matches a pinned epsilon:\n  "
            + "\n  ".join(hits)
            + "\nremove or archive the duplicates before running."
        )
    return Path(hits[0])


def normalize_target(series: pd.Series) -> pd.Series:
    # normalize target text before mapping to 0/1
    norm = (series.astype(str).str.strip().str.lower()
            .replace({"non dropout": "non-dropout", "non_dropout": "non-dropout", "drop out": "dropout"}))
    mapping = {"non-dropout": 0, "dropout": 1, "0": 0, "1": 1}
    y = norm.map(mapping)
    if y.isna().any():
        bad = series.loc[y.isna()].unique().tolist()
        raise ValueError(f"target values could not be mapped: {bad}")
    return y.astype(int)


def _first_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None


# ============================================================
# READ AND STANDARDIZE PREDICTION FILES (keyed by id_student)
# ============================================================
def read_attack_predictions(path: Path, clean_prefix: str) -> pd.DataFrame:
    # undefended attack file -> tidy per-student frame. columns look like
    # rf_clean_pred_label / rf_adv_pred_label (or xgb_ / mlp_ prefix).
    df = pd.read_csv(path)
    clean_pred = _first_col(df, [f"{clean_prefix}_clean_pred_label", "clean_pred_label"])
    adv_pred = _first_col(df, [f"{clean_prefix}_adv_pred_label", "adv_pred_label"])
    clean_prob = _first_col(df, [f"{clean_prefix}_clean_prob_dropout", "clean_prob_dropout"])
    adv_prob = _first_col(df, [f"{clean_prefix}_adv_prob_dropout", "adv_prob_dropout"])
    if STUDENT_COL not in df.columns or clean_pred is None or adv_pred is None:
        raise KeyError(f"{path} is missing id_student / clean / adv prediction columns.")
    out = pd.DataFrame({
        STUDENT_COL: df[STUDENT_COL].values,
        "actual_label": df["actual_label"].astype(int).values,
        "clean_pred_label": df[clean_pred].astype(int).values,
        "attack_adv_pred_label": df[adv_pred].astype(int).values,
    })
    if clean_prob:
        out["clean_prob_dropout"] = pd.to_numeric(df[clean_prob], errors="coerce").values
    if adv_prob:
        out["attack_adv_prob_dropout"] = pd.to_numeric(df[adv_prob], errors="coerce").values
    return out


def read_defence_predictions(path: Path) -> pd.DataFrame:
    # defended-under-attack file (phase 4) -> per-student frame with the defended
    # adversarial prediction only. RF/XGB write a single defended prediction
    # (predicted_label / prob_dropout); the MLP script writes adv_pred_label /
    # adv_prob_dropout. Phase 3 (clean) is read separately, so it is not taken
    # from here.
    df = pd.read_csv(path)
    if STUDENT_COL not in df.columns:
        raise KeyError(f"{path} has no '{STUDENT_COL}' column.")
    adv_pred_col = _first_col(df, ["adv_pred_label", "predicted_label", "y_pred", "prediction"])
    if adv_pred_col is None:
        raise KeyError(f"{path} has no defended prediction column "
                       f"(looked for adv_pred_label / predicted_label). Columns: {list(df.columns)}")
    adv_prob_col = _first_col(df, ["adv_prob_dropout", "prob_dropout", "predicted_prob_dropout"])
    out = pd.DataFrame({
        STUDENT_COL: df[STUDENT_COL].values,
        "defended_adv_pred_label": df[adv_pred_col].astype(int).values,
    })
    if adv_prob_col:
        out["defended_adv_prob_dropout"] = pd.to_numeric(df[adv_prob_col], errors="coerce").values
    return out


def read_defence_clean_predictions(path: Path) -> pd.DataFrame:
    # defended model on CLEAN input (phase 3) -> per-student frame. RF/XGB write
    # predicted_label / prob_dropout; the MLP clean file writes clean_pred_label /
    # clean_prob_dropout (its adv_* columns are a copy of the clean ones). read
    # whichever is present.
    df = pd.read_csv(path)
    if STUDENT_COL not in df.columns:
        raise KeyError(f"{path} has no '{STUDENT_COL}' column.")
    pred_col = _first_col(df, ["clean_pred_label", "predicted_label", "y_pred", "prediction"])
    if pred_col is None:
        raise KeyError(f"{path} has no defended-clean prediction column "
                       f"(looked for clean_pred_label / predicted_label). Columns: {list(df.columns)}")
    prob_col = _first_col(df, ["clean_prob_dropout", "prob_dropout", "predicted_prob_dropout"])
    out = pd.DataFrame({
        STUDENT_COL: df[STUDENT_COL].values,
        "defended_clean_pred_label": df[pred_col].astype(int).values,
    })
    if prob_col:
        out["defended_clean_prob_dropout"] = pd.to_numeric(df[prob_col], errors="coerce").values
    return out


# ============================================================
# ADVERSARIAL RAW FEATURE DIFFERENCES
# ============================================================
def attach_adv_diffs(table: pd.DataFrame, attack_pred_path: Optional[Path],
                     raw_adv_path: Optional[Path], clean_by_id: Optional[pd.DataFrame]) -> pd.DataFrame:
    # the adversarial raw files hold feature columns only, no id_student, and their
    # rows are in the same order as the attack prediction file. tag the raw rows
    # with the prediction file's id_student (positional), join by id_student, then
    # compute adv - clean diffs on the attackable features.
    if attack_pred_path is None or raw_adv_path is None or clean_by_id is None or table.empty:
        return table
    pred_ids = pd.read_csv(attack_pred_path)[STUDENT_COL].values
    raw = pd.read_csv(raw_adv_path)
    if len(pred_ids) != len(raw):
        print(f"  Skipped adv diffs: row count mismatch "
              f"({len(pred_ids)} ids vs {len(raw)} raw rows) for {raw_adv_path.name}.")
        return table
    raw = raw.copy()
    raw[STUDENT_COL] = pred_ids
    attackable = [c for c in ATTACKABLE_FEATURES if c in raw.columns and c in clean_by_id.columns]
    adv = raw[[STUDENT_COL] + attackable].rename(columns={c: f"adv_{c}" for c in attackable})
    merged = table.merge(adv, on=STUDENT_COL, how="left")
    for c in attackable:
        clean_map = clean_by_id.set_index(STUDENT_COL)[c]
        merged[f"diff_{c}"] = merged[f"adv_{c}"] - merged[STUDENT_COL].map(clean_map)
    return merged


# ============================================================
# BUILD ONE MODEL x SPLIT TABLE
# ============================================================
def build_model_split(model: str, split: str, clean_by_id: pd.DataFrame,
                      file_map: dict) -> Tuple[Optional[pd.DataFrame], dict]:
    spec = MODEL_FILES[model]
    stats = {"model": model, "split": split, "attack": spec["attack"]}

    attack_pred_path = resolve_one(spec["attack_pred"], split)
    defence_pred_path = resolve_one(spec["defence_pred"], split)
    defence_clean_pred_path = resolve_one(spec["defence_clean_pred"], split)
    raw_adv_path = resolve_one(spec["attack_raw"], split)

    file_map[f"{model}|{split}"] = {
        "attack_predictions": str(attack_pred_path) if attack_pred_path else None,
        "defence_predictions_adv": str(defence_pred_path) if defence_pred_path else None,
        "defence_predictions_clean": str(defence_clean_pred_path) if defence_clean_pred_path else None,
        "attack_raw_features": str(raw_adv_path) if raw_adv_path else None,
    }

    if attack_pred_path is None or defence_pred_path is None:
        print(f"  {model} / {split}: missing attack or defence(adv) prediction file, skipped.")
        return None, stats

    atk = read_attack_predictions(attack_pred_path, spec["clean_prefix"])
    dfn = read_defence_predictions(defence_pred_path)

    stats.update({k: v for k, v in eps_from_name(attack_pred_path.name).items() if v is not None})
    stats.update({f"defence_{k}": v for k, v in eps_from_name(defence_pred_path.name).items() if v is not None})

    # phase 1 + phase 2: successfully attacked = real dropout, caught when clean,
    # hidden under the undefended attack.
    attacked = atk[(atk["actual_label"] == 1) &
                   (atk["clean_pred_label"] == 1) &
                   (atk["attack_adv_pred_label"] == 0)].copy()

    # phase 4: defended model under attack
    table = attacked.merge(dfn, on=STUDENT_COL, how="left")
    table["successfully_attacked"] = True

    # honest handling of a missing defended(adv) row: keep it distinguishable from
    # a genuine non-dropout prediction instead of silently counting it as
    # "not recovered".
    table["defended_adv_pred_label"] = table["defended_adv_pred_label"].astype("Int64")
    adv_found = table["defended_adv_pred_label"].notna()
    table["defended_adv_row_found"] = adv_found
    recovered = pd.Series(pd.NA, index=table.index, dtype="boolean")
    recovered[adv_found] = table.loc[adv_found, "defended_adv_pred_label"].eq(1)
    table["successfully_recovered"] = recovered  # definition unchanged: phase 4 == dropout

    # phase 3: defended model on clean input (supporting evidence only)
    if defence_clean_pred_path is not None:
        dfn_clean = read_defence_clean_predictions(defence_clean_pred_path)
        table = table.merge(dfn_clean, on=STUDENT_COL, how="left")
    if "defended_clean_pred_label" not in table.columns:
        table["defended_clean_pred_label"] = pd.array([pd.NA] * len(table), dtype="Int64")
    table["defended_clean_pred_label"] = table["defended_clean_pred_label"].astype("Int64")
    clean_found = table["defended_clean_pred_label"].notna()
    table["defended_clean_row_found"] = clean_found
    catches_clean = pd.Series(pd.NA, index=table.index, dtype="boolean")
    catches_clean[clean_found] = table.loc[clean_found, "defended_clean_pred_label"].eq(1)
    table["defended_catches_on_clean"] = catches_clean

    # a compact, human-readable verdict combining phase 3 and phase 4. This does
    # NOT redefine successfully_recovered; it just helps read the row at a glance.
    def verdict(row) -> str:
        if not row["defended_adv_row_found"]:
            return "missing_defended_adv_row"
        if row["successfully_recovered"] is True:
            if row["defended_catches_on_clean"] is True:
                return "recovered_and_sane"          # ideal: caught clean AND under attack
            if row["defended_catches_on_clean"] is False:
                return "recovered_but_missed_on_clean"  # suspicious: check it
            return "recovered_clean_unknown"
        return "not_recovered"
    table["recovery_verdict"] = table.apply(verdict, axis=1)

    # readable label columns (phase order)
    table["actual_text"] = table["actual_label"].map(label_text)
    table["clean_pred_text"] = table["clean_pred_label"].map(label_text)             # phase 1
    table["attack_adv_pred_text"] = table["attack_adv_pred_label"].map(label_text)   # phase 2
    table["defended_clean_pred_text"] = table["defended_clean_pred_label"].map(label_text)  # phase 3
    table["defended_adv_pred_text"] = table["defended_adv_pred_label"].map(label_text)      # phase 4

    # clean raw full features by id_student
    clean_cols = [STUDENT_COL] + [c for c in FULL_FEATURES if c in clean_by_id.columns]
    clean_small = clean_by_id[clean_cols].rename(
        columns={c: f"clean_{c}" for c in FULL_FEATURES if c in clean_by_id.columns})
    table = table.merge(clean_small, on=STUDENT_COL, how="left")

    # adversarial raw diffs (attackable features only, from the undefended attack)
    table = attach_adv_diffs(table, attack_pred_path, raw_adv_path, clean_by_id)

    # counts
    n_attacked = len(table)
    n_adv_found = int(adv_found.sum())
    n_adv_missing = int((~adv_found).sum())
    n_recovered = int((table["successfully_recovered"] == True).sum())
    n_sane_clean = int((table["defended_catches_on_clean"] == True).sum())
    n_recovered_and_sane = int((table["recovery_verdict"] == "recovered_and_sane").sum())

    stats["successfully_attacked_count"] = n_attacked
    stats["defended_adv_row_found_count"] = n_adv_found
    stats["defended_adv_row_missing_count"] = n_adv_missing
    stats["successfully_recovered_count"] = n_recovered
    # recovery rate over rows where the defended prediction actually exists
    stats["recovery_rate"] = (n_recovered / n_adv_found) if n_adv_found else np.nan
    stats["defended_catches_on_clean_count"] = n_sane_clean
    stats["recovered_and_sane_count"] = n_recovered_and_sane

    if n_attacked:
        rate = stats["recovery_rate"]
        rate_txt = f"{rate:.1%}" if pd.notna(rate) else "n/a"
        extra = f", missing defended rows={n_adv_missing}" if n_adv_missing else ""
        print(f"  {model} / {split}: attacked={n_attacked}, recovered={n_recovered} "
              f"({rate_txt}), recovered&sane={n_recovered_and_sane}{extra}")
    else:
        print(f"  {model} / {split}: no successfully-attacked students.")

    # tidy column order (phases 1->4, then flags, then probs, then diffs, then raw)
    lead = [STUDENT_COL, "actual_text",
            "clean_pred_text",            # phase 1: original model, clean
            "attack_adv_pred_text",       # phase 2: original model, adversarial
            "defended_clean_pred_text",   # phase 3: defended model, clean
            "defended_adv_pred_text",     # phase 4: defended model, adversarial
            "successfully_attacked", "successfully_recovered",
            "defended_catches_on_clean", "recovery_verdict",
            "defended_adv_row_found", "defended_clean_row_found"]
    prob_cols = [c for c in ["clean_prob_dropout", "attack_adv_prob_dropout",
                             "defended_clean_prob_dropout", "defended_adv_prob_dropout"]
                 if c in table.columns]
    diff_cols = [c for c in table.columns if c.startswith("diff_")]
    other = [c for c in table.columns if c not in lead + prob_cols + diff_cols]
    table = table[[c for c in lead if c in table.columns] + prob_cols + diff_cols +
                  [c for c in other if c not in lead + prob_cols + diff_cols]]
    return table, stats


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("OULAD EARLY - successfully attacked & recovered students")
    print("=" * 80)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {CSV_PATH}")
    raw = pd.read_csv(CSV_PATH)
    if STUDENT_COL not in raw.columns:
        raise KeyError(f"dataset has no '{STUDENT_COL}' column; cannot join predictions.")
    raw["actual_label"] = normalize_target(raw[TARGET_COL])
    clean_by_id = raw.drop_duplicates(subset=STUDENT_COL).reset_index(drop=True)

    file_map: dict = {}
    all_stats: List[dict] = []
    sheets: Dict[str, pd.DataFrame] = {}

    for model in MODEL_FILES:
        for split in SPLITS:
            table, stats = build_model_split(model, split, clean_by_id, file_map)
            all_stats.append(stats)
            if table is not None and not table.empty:
                sheet = f"{SHEET_PREFIX[model]}_{split}"[:31]
                sheets[sheet] = table

    if not sheets:
        raise RuntimeError("no attacked/recovered tables were built. check the result folders.")

    summary = pd.DataFrame(all_stats)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"\nSaved summary: {OUTPUT_SUMMARY_CSV}")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
    print(f"Saved workbook: {OUTPUT_EXCEL}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(file_map, f, indent=2)
    print(f"Saved file map: {OUTPUT_JSON}")

    print("\nSummary:")
    show = [c for c in ["model", "split", "successfully_attacked_count",
                        "successfully_recovered_count", "recovery_rate",
                        "defended_catches_on_clean_count", "recovered_and_sane_count",
                        "defended_adv_row_missing_count"] if c in summary.columns]
    print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()