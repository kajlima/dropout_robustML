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
# The detect-and-default (DnD) twin of the adversarial-training check-students
# script, for zenodo.csv. For each model it collects:
#   "successfully attacked" = a real dropout, caught by the clean baseline model,
#                             but hidden (predicted non-dropout) under attack.
#   "recovered"             = a successfully-attacked student the DnD defence
#                             saves. DnD does NOT retrain the model and does NOT
#                             change the input; it flags suspicious rows with a
#                             Mahalanobis detector and overrides the prediction.
#                             So recovery == the ADVERSARIAL row was flagged:
#                               default_dropout -> flagged adv row forced to dropout
#                               abstain         -> flagged adv row sent to review
#                             Both reduce to "flagged_adv" (verified against the
#                             trace files' own recovered_default_dropout /
#                             caught_abstain columns).
#
# FOUR-PHASE VIEW (parallels the adversarial-training script, but adapted to DnD):
#   phase 1  base model, clean input   -> catches the dropout   (clean_pred)
#   phase 2  base model, adv input     -> dropout HIDDEN         (adv_pred)   [attacked]
#   phase 3  DnD on the CLEAN row       -> detector behaviour on clean (flagged_clean)
#   phase 4  DnD on the ADV row         -> detector flags -> override (recovered)
#
# IMPORTANT difference from the adversarial-training version: DnD has no retrained
# model, so phase 3 is NOT "a defended model on clean data". The base model is
# unchanged, so on the clean row it already predicts what it always predicted.
# What DnD adds on the clean row is the DETECTOR decision. Phase 3 here therefore
# measures the SPECIFICITY of the flag:
#   - flagged_adv AND NOT flagged_clean -> the flag is attack-specific (ideal):
#     the detector reacted to the perturbation, not to the student.
#   - flagged_adv AND flagged_clean     -> the detector fires on this student
#     regardless of the attack; the "recovery" is not attack-specific -> examine.
# This is the DnD analogue of the adversarial-training sanity check "is the
# recovery genuine or did the model just turn paranoid".
#
# Everything is precomputed by the dnd_train_*.py trace files (one row per student
# per split, at eval epsilon 1.0), so this script only reads those traces. It does
# NOT reconstruct predictions or feature diffs (DnD saves no raw adversarial
# feature file; the input geometry equals the undefended attack and is covered by
# the proximity script).
#
# One trace per model, per split:
#   Random Forest -> raw-space Mahalanobis, FGSM transferred from the baseline MLP
#   XGBoost       -> raw-space Mahalanobis, FGSM transferred from the baseline MLP
#   MLP           -> penultimate class-conditional Mahalanobis, white-box PGD

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = Path("zenodo.csv")
OUTPUT_DIR = Path("zenodo_dnd_attack_recovery_students_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EXCEL = OUTPUT_DIR / "zenodo_dnd_attacked_and_recovered_students.xlsx"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "zenodo_dnd_attack_recovery_summary.csv"
OUTPUT_JSON = OUTPUT_DIR / "zenodo_dnd_attack_recovery_file_map.json"

TARGET_COL = "target"
STUDENT_COL = "student_id"
POSITIVE_LABEL = 1  # dropout
NEGATIVE_LABEL = 0  # non-dropout
SPLITS = ("test", "validation")
EVAL_EPS_TAG = "1p0"  # every DnD run in this project uses eval epsilon 1.0

# Full predictor set for zenodo, used only to attach the student's clean feature
# values to the per-student sheets. student_id and target are never predictors.
FULL_FEATURES = [
    "academic_year", "n_courses", "courses_mean", "degree_size", "seniority",
    "highest_course_year_enrolled",
    "credits_enrolled_semester_a", "credits_enrolled_semester_b",
    "total_credits_enrolled_academic_year", "total_credits_passed_academic_year",
    "pass_ratio_sem_a", "pass_ratio_sem_b", "total_performance",
    "obtained_new_degree_flag", "adapted_studies_flag",
    "completion_rate_one_year_before", "completion_rate_two_years_before",
    "completion_rate_three_years_before",
    "pass_ratio_sem_a_and_zero_flag", "pass_ratio_sem_b_and_zero_flag",
    "total_performance_and_zero_flag",
    "completion_rate_one_year_before_missing_flag",
    "completion_rate_two_years_before_missing_flag",
    "completion_rate_three_years_before_missing_flag",
    "lms_events_sem_a", "lms_events_sem_b",
    "lms_assignment_submissions_sem_a", "lms_assignment_submissions_sem_b",
    "lms_test_submissions_sem_a", "lms_test_submissions_sem_b",
    "lms_total_minutes_sem_a", "lms_total_minutes_sem_b",
    "attendance_days_sem_a", "attendance_days_sem_b",
]

# ============================================================
# OUTPUT FOLDERS PRODUCED BY THE dnd_train_*.py SCRIPTS (zenodo)
# ============================================================
RF_DND_DIR = "rf_detect_and_default_zenodo_results"
XGB_DND_DIR = "xgb_detect_and_default_zenodo_results"
MLP_DND_DIR = "mlp_dnd_feature_classcond_zenodo_results"

# per-model trace file. {split} is filled in; each pattern also lists a bare
# filename fallback so the script still finds the trace if it sits in the working
# directory instead of the results folder.
MODEL_FILES = {
    "Random Forest": {
        "prefix": "rf",
        "attack": "FGSM (transfer from baseline MLP)",
        "detector": "raw-space Mahalanobis over attackable features",
        "sheet": "RF",
        "trace": [
            f"{RF_DND_DIR}/zenodo_rf_dnd_{{split}}_student_trace_eval_eps_{EVAL_EPS_TAG}.csv",
            f"zenodo_rf_dnd_{{split}}_student_trace_eval_eps_{EVAL_EPS_TAG}.csv",
        ],
    },
    "XGBoost": {
        "prefix": "xgb",
        "attack": "FGSM (transfer from baseline MLP)",
        "detector": "raw-space Mahalanobis over attackable features",
        "sheet": "XGB",
        "trace": [
            f"{XGB_DND_DIR}/zenodo_xgb_dnd_{{split}}_student_trace_eval_eps_{EVAL_EPS_TAG}.csv",
            f"zenodo_xgb_dnd_{{split}}_student_trace_eval_eps_{EVAL_EPS_TAG}.csv",
        ],
    },
    "MLP": {
        "prefix": "mlp",
        "attack": "PGD white-box",
        "detector": "raw-space Mahalanobis over attackable features",
        "sheet": "MLP",
        "trace": [
            f"{MLP_DND_DIR}/zenodo_mlp_dnd_{{split}}_student_trace_eval_eps_{EVAL_EPS_TAG}.csv",
            f"zenodo_mlp_dnd_{{split}}_student_trace_eval_eps_{EVAL_EPS_TAG}.csv",
        ],
    },
}


# ============================================================
# HELPERS
# ============================================================
def label_text(value) -> str:
    if pd.isna(value):
        return "Missing"
    try:
        return "Dropout" if int(value) == POSITIVE_LABEL else "Non-dropout"
    except Exception:
        return "Unknown"


def resolve_one(patterns: List[str], split: str) -> Optional[Path]:
    hits: List[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat.format(split=split)))
    hits = sorted(set(hits))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            "more than one trace file matches:\n  " + "\n  ".join(hits)
            + "\nremove or archive the duplicates before running."
        )
    return Path(hits[0])


def normalize_target(series: pd.Series) -> pd.Series:
    norm = (series.astype(str).str.strip().str.lower()
            .replace({"non dropout": "non-dropout", "non_dropout": "non-dropout", "drop out": "dropout"}))
    mapping = {"non-dropout": 0, "dropout": 1, "0": 0, "1": 1}
    y = norm.map(mapping)
    if y.isna().any():
        bad = series.loc[y.isna()].unique().tolist()
        raise ValueError(f"target values could not be mapped: {bad}")
    return y.astype(int)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (series.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False).astype(bool))


def eps_from_name(path: str) -> Optional[float]:
    m = re.search(r"eval_eps_([0-9]+p[0-9]+)", path)
    return float(m.group(1).replace("p", ".")) if m else None


# ============================================================
# BUILD ONE MODEL x SPLIT TABLE FROM ITS TRACE
# ============================================================
def build_model_split(model: str, split: str, clean_by_id: pd.DataFrame,
                       file_map: dict) -> Tuple[Optional[pd.DataFrame], dict]:
    spec = MODEL_FILES[model]
    prefix = spec["prefix"]
    stats = {"model": model, "split": split, "attack": spec["attack"], "detector": spec["detector"]}

    trace_path = resolve_one(spec["trace"], split)
    file_map[f"{model}|{split}"] = {"student_trace": str(trace_path) if trace_path else None}
    if trace_path is None:
        print(f"  {model} / {split}: trace file not found, skipped.")
        return None, stats

    df = pd.read_csv(trace_path)
    stats["eval_epsilon"] = eps_from_name(trace_path.name)

    # base columns (recomputed here rather than trusting precomputed flags)
    clean_pred = df[f"{prefix}_clean_pred_label"].astype(int).values
    adv_pred = df[f"{prefix}_adv_pred_label"].astype(int).values
    actual = df["actual_label"].astype(int).values
    flagged_clean = _coerce_bool(df["flagged_clean"]).values
    flagged_adv = _coerce_bool(df["flagged_adv"]).values

    # phase 1 + phase 2: successfully attacked = real dropout, caught clean, hidden
    attacked = (actual == POSITIVE_LABEL) & (clean_pred == POSITIVE_LABEL) & (adv_pred == NEGATIVE_LABEL)

    # phase 4: recovery == the hidden dropout's ADVERSARIAL row was flagged. Prefer
    # the defence's own final-decision columns when present; fall back to flagged_adv.
    if "dnd_default_dropout_pred_label" in df.columns:
        default_final_adv = df["dnd_default_dropout_pred_label"].astype(int).values  # decision on the ADV row
        recovered_default = attacked & (default_final_adv == POSITIVE_LABEL)
    else:
        default_final_adv = np.where(flagged_adv, POSITIVE_LABEL, adv_pred)
        recovered_default = attacked & flagged_adv
    if "dnd_abstain_decision" in df.columns:
        caught_abstain = attacked & (df["dnd_abstain_decision"].astype(str).values == "abstained")
    else:
        caught_abstain = attacked & flagged_adv

    # phase 3: DnD decision on the CLEAN row under default_dropout, and the
    # SPECIFICITY of the flag. clean_final forces dropout where the clean row is
    # flagged; for attacked students clean_pred is already dropout, so what matters
    # for phase 3 is whether the clean row was flagged at all.
    default_final_clean = np.where(flagged_clean, POSITIVE_LABEL, clean_pred)
    # attack-specific flag = flagged on adv but NOT on clean (reacted to the attack)
    flag_attack_specific = flagged_adv & (~flagged_clean)

    # ---- detector-level stats over ALL rows in the split (the cost + the catch) ----
    n_rows = len(df)
    n_attacked = int(attacked.sum())
    n_recovered_default = int(recovered_default.sum())
    n_caught_abstain = int(caught_abstain.sum())
    n_specific = int((recovered_default & flag_attack_specific).sum())
    n_flags_clean_too = int((recovered_default & flagged_clean).sum())
    stats.update({
        "n_rows": n_rows,
        "successfully_attacked_count": n_attacked,
        "recovered_default_dropout_count": n_recovered_default,
        "caught_abstain_count": n_caught_abstain,
        "recovery_rate_default_dropout": (n_recovered_default / n_attacked) if n_attacked else np.nan,
        "caught_rate_abstain": (n_caught_abstain / n_attacked) if n_attacked else np.nan,
        "detection_recall_on_attacked": (int((attacked & flagged_adv).sum()) / n_attacked) if n_attacked else np.nan,
        # phase-3 specificity of the recoveries
        "recovered_attack_specific_count": n_specific,
        "recovered_flags_clean_too_count": n_flags_clean_too,
        # the honest cost: how often clean rows are flagged (false-positive pressure)
        "flag_rate_clean": float(np.mean(flagged_clean)),
        "flag_rate_adv": float(np.mean(flagged_adv)),
    })

    if n_attacked:
        print(f"  {model} / {split}: attacked={n_attacked}, "
              f"recovered={n_recovered_default} ({stats['recovery_rate_default_dropout']:.1%}) "
              f"[attack-specific={n_specific}, flags-clean-too={n_flags_clean_too}], "
              f"flag_rate clean={stats['flag_rate_clean']:.3f} adv={stats['flag_rate_adv']:.3f}")
    else:
        print(f"  {model} / {split}: no successfully-attacked students.")

    # ---- per-student table: attacked students only ----
    mask = attacked
    rec_d = recovered_default[mask]
    fc = flagged_clean[mask]
    fa = flagged_adv[mask]

    # human-readable verdict combining phase 3 (clean flag) and phase 4 (recovery)
    verdict = np.where(
        ~rec_d, "not_recovered",
        np.where(~fc, "recovered_attack_specific", "recovered_but_flags_clean_too"),
    )

    table = pd.DataFrame({
        STUDENT_COL: df[STUDENT_COL].values[mask],
        "actual_text": [label_text(v) for v in actual[mask]],
        "clean_pred_text": [label_text(v) for v in clean_pred[mask]],            # phase 1
        "attack_adv_pred_text": [label_text(v) for v in adv_pred[mask]],         # phase 2
        "dnd_clean_decision_text": [label_text(v) for v in default_final_clean[mask]],  # phase 3
        "dnd_adv_decision_text": [label_text(v) for v in default_final_adv[mask]],      # phase 4
        "flagged_clean": fc,
        "flagged_adv": fa,
        "successfully_attacked": True,
        "recovered_default_dropout": rec_d,
        "caught_abstain": caught_abstain[mask],
        "flag_attack_specific": flag_attack_specific[mask],
        "recovery_verdict": verdict,
    })

    # detector scores / probabilities / abstain text, where present
    passthrough = [
        "maha_score_clean", "maha_score_adv", "maha_threshold",
        f"{prefix}_clean_prob_dropout", f"{prefix}_adv_prob_dropout",
        "dnd_abstain_decision", "dnd_default_dropout_pred_target",
    ]
    for col in passthrough:
        if col in df.columns:
            table[col] = df[col].values[mask]

    if not table.empty:
        # attach the student's clean feature values (no adv diffs: DnD stores none)
        clean_cols = [STUDENT_COL] + [c for c in FULL_FEATURES if c in clean_by_id.columns]
        clean_small = clean_by_id[clean_cols].rename(
            columns={c: f"clean_{c}" for c in FULL_FEATURES if c in clean_by_id.columns})
        table = table.merge(clean_small, on=STUDENT_COL, how="left")

    return table, stats


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("ZENODO - detect-and-default: attacked & recovered students")
    print("=" * 80)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"dataset not found: {CSV_PATH}")
    raw = pd.read_csv(CSV_PATH)
    if STUDENT_COL not in raw.columns:
        raise KeyError(f"dataset has no '{STUDENT_COL}' column; cannot attach clean features.")
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
                sheet = f"{MODEL_FILES[model]['sheet']}_{split}"[:31]
                sheets[sheet] = table

    if not any(s.get("successfully_attacked_count") for s in all_stats):
        raise RuntimeError("no traces produced any attacked students. check the DnD result folders.")

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
    show = [c for c in ["model", "split", "eval_epsilon", "successfully_attacked_count",
                        "recovered_default_dropout_count", "recovery_rate_default_dropout",
                        "recovered_attack_specific_count", "recovered_flags_clean_too_count",
                        "flag_rate_clean", "flag_rate_adv"] if c in summary.columns]
    print(summary[show].to_string(index=False))
    print("\nNote: for DnD, 'recovered' means the detector flagged the adversarial row, "
          "not that the model corrected itself. Phase 3 (flagged_clean) splits recoveries "
          "into attack-specific (flag adv only) vs flags-clean-too (fires regardless). "
          "Read recovery together with flag_rate_clean, the false-positive cost on clean rows.")


if __name__ == "__main__":
    main()