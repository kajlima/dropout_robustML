"""
One combined OULAD dataset for early-vs-end-of-module dropout modelling.

Pipeline order (matches yours): cleaning -> filter -> sample -> synth, but the
sample step is a PLAIN STRATIFIED RANDOM sample. It deliberately does NOT filter
by avg_score (the old ouladsample.py band-filtering separated the classes on the
predictor and inflated accuracy). Class balance is set to a dropout minority to
match your RF design; note the NATURAL dropout rate in OULAD is ~53%
(Withdrawn+Fail).

New in this version (vs the file you sent):
  1. filter_implausible_credits(): drops rows whose studied_credits exceeds
     MAX_STUDIED_CREDITS (default 120 = one year of full-time UK study). Runs
     BEFORE sampling so only students at/under a normal full-time yearly load
     enter the sample. Raise to 360 (a full degree) for a looser cut.
  2. add_synth_credit_counts(): synthetic credit_enrolled_ / credit_passed_ for
     1st/2nd/3rd "year" + totals, in real credits (multiples of 30). passed is a
     per-module Binomial with only a WEAK link to that year's synthetic GPA (a
     low-GPA student can still pass every module, just with poor marks), so it is
     realistic and not a rescaled copy of GPA. Never uses final_result.

Output: one row per student. Columns:
  - static / Pre-Entry (demographics, registration, target)
  - end-of-module real features (full-presentation aggregates) -> for the end model
  - early windowed features, one block per cutoff in CUTOFFS (suffix _<cut>) -> for the early model(s)
  - synthetic GPA_1styear/2ndyear/3rdyear/CGPA + module_credits (from synth logic)
  - synthetic credit_enrolled_/credit_passed_ 1st/2nd/3rdyear + totals

Definitions kept identical to ouladcleaning.py:
  quiz='quiz' exact, forum='forumng', avg_score Opsi-2, imd ordinal + imd_missing.

Trimmed vs 67-col version: dropped date_registration (+ its two derived flags),
and the four per-cutoff flags (no_assess_due, early_any_submission_flag,
has_early_vle_flag, active_at_cutoff). active_at_cutoff was a leak from
date_unregistration, so removing it also cleans the last unregistration-derived
column out of the output.

IMPORTANT (thesis disclosure): GPA_* and credit_enrolled_/credit_passed_* do NOT
exist in the real OULAD dataset and are SIMULATED. OULAD has no concept of "year
of study". State this explicitly in your methodology, together with how they are
generated.

early_first_submit_day sign: date_submitted in OULAD is relative to the module
start day (day 0) and can be negative (submissions logged before the official
start; min observed = -11 in the raw data). This script takes the ABSOLUTE VALUE
of early_first_submit_day so the column is always >= 0. NOTE for interpretation:
abs() maps an early submitter (e.g. -11, submitted 11 days before start) onto the
same value as a late submitter (+11, submitted 11 days after start), so the two
opposite behaviours collapse to one number. This is an intentional simplification
here, not a data-cleaning fix (the negative values are valid).
"""

import os
import argparse
import numpy as np
import pandas as pd

TRIP = ["code_module", "code_presentation", "id_student"]
PRE = ["gender", "region", "highest_education", "imd_band", "age_band",
       "num_of_prev_attempts", "studied_credits", "disability"]
IMD_ORDER = {"0-10%": 1, "10-20%": 2, "20-30%": 3, "30-40%": 4, "40-50%": 5,
             "50-60%": 6, "60-70%": 7, "70-80%": 8, "80-90%": 9, "90-100%": 10}
MODULE_CREDITS = {"AAA": 60, "BBB": 60, "CCC": 30, "DDD": 60, "EEE": 30, "FFF": 60, "GGG": 30}
EDU_MAP = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
           "HE Qualification": 3, "Post Graduate Qualification": 4}
# Early windows: one feature block per cutoff (days since module start), suffix _<cut>.
# Configurable — edit this list to add/remove time points.
CUTOFFS = [65, 153]

# --- sample config (change freely; stays < 5000) ---
SAMPLE_TOTAL = 7259
DROPOUT_N = 1454          # 20% dropout minority; natural rate is ~53%
SAMPLE_SEED = 42
SYNTH_SEED = 7

# --- studied_credits sampling cap ---
# 120 credits = one year of full-time UK study, used here as the upper bound for
# who can enter the sample. Rows above 120 (students carrying more than a normal
# full-time yearly load at once) are dropped BEFORE sampling. ~93% of OULAD
# students sit at <=120, so the pool stays large. Raise back to 360 (a full
# degree) or 240 if you want a looser cut.
MAX_STUDIED_CREDITS = 120


# ============================ CLEANING ============================
def clean_end_of_module(raw, vle_path):
    si = pd.read_csv(f"{raw}/studentInfo.csv", na_values=["?"])
    si["imd_band"] = si["imd_band"].replace({"10-20": "10-20%"})

    reg = pd.read_csv(f"{raw}/studentRegistration.csv", na_values=["?"])
    for c in ["date_registration", "date_unregistration"]:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")

    sa = pd.read_csv(f"{raw}/studentAssessment.csv", na_values=["?"])
    ass = pd.read_csv(f"{raw}/assessments.csv", na_values=["?"])
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    ass["date"] = pd.to_numeric(ass["date"], errors="coerce")
    saj = sa.merge(ass[["id_assessment", "code_module", "code_presentation",
                        "assessment_type", "weight", "date"]], on="id_assessment", how="left")

    # Opsi-2 avg_score (full presentation)
    earned = (saj[saj["assessment_type"] != "Exam"].groupby(TRIP)["score"].sum().rename("earned"))
    navail = (ass[ass["assessment_type"] != "Exam"]
              .groupby(["code_module", "code_presentation"]).size().rename("navail"))
    other = (saj[saj["assessment_type"] != "Exam"].groupby(TRIP)
             .agg(num_assess_submitted=("id_assessment", "size"),
                  total_weight=("weight", "sum")))

    base = si.copy()
    base["target"] = np.where(base["final_result"].isin(["Withdrawn", "Fail"]), "dropout", "non-dropout")
    base = base.merge(reg[TRIP + ["date_registration", "date_unregistration"]], on=TRIP, how="left")

    base["imd_missing"] = base["imd_band"].isna().astype(int)
    base["imd_band"] = base["imd_band"].map(IMD_ORDER).fillna(0).astype(int)

    base = base.merge(other, on=TRIP, how="left").merge(earned.reset_index(), on=TRIP, how="left")
    base = base.merge(navail.reset_index(), on=["code_module", "code_presentation"], how="left")
    base["earned"] = base["earned"].fillna(0.0)
    base["num_assess_submitted"] = base["num_assess_submitted"].fillna(0).astype(int)
    base["total_weight"] = base["total_weight"].fillna(0.0)
    base["avg_score"] = (base["earned"] / base["navail"]).round(2)
    base = base.drop(columns=["earned", "navail"])
    base["module_credits"] = base["code_module"].map(MODULE_CREDITS).astype(int)
    return base, saj  # saj reused for early assessment windows


# ============================ FILTER (implausible studied_credits) ============================
def filter_implausible_credits(base, max_credits=MAX_STUDIED_CREDITS):
    """Drop rows whose studied_credits is not credible (> a full degree).
    studied_credits is the TOTAL credits across all modules the student is
    studying at once, not a single module, so large values are expected — but
    values above a whole degree (360) are implausible as a simultaneous load."""
    before = len(base)
    kept = base[base["studied_credits"] <= max_credits].copy()
    removed = before - len(kept)
    print(f"[filter] studied_credits > {max_credits}: removed {removed} rows "
          f"({removed / before * 100:.2f}%), kept {len(kept)}")
    return kept


# ============================ SAMPLE ============================
def stratified_sample(base):
    u = base.sample(frac=1, random_state=SAMPLE_SEED).drop_duplicates(subset="id_student", keep="first")
    nd = u[u["target"] == "non-dropout"]
    dr = u[u["target"] == "dropout"]
    n_non = SAMPLE_TOTAL - DROPOUT_N
    assert len(nd) >= n_non and len(dr) >= DROPOUT_N, "pool too small"
    samp = pd.concat([nd.sample(n_non, random_state=SAMPLE_SEED),
                      dr.sample(DROPOUT_N, random_state=SAMPLE_SEED)])
    return samp.sample(frac=1, random_state=SAMPLE_SEED).reset_index(drop=True)


# ============================ CLICKSTREAM (sampled students only) ============================
def clickstream_for_sample(vle_path, raw, sample):
    keyset = set(map(tuple, sample[TRIP].itertuples(index=False, name=None)))
    meta = pd.read_csv(f"{raw}/vle.csv", na_values=["?"])
    site_to_act = dict(zip(meta["id_site"], meta["activity_type"]))
    dt = {"code_module": "category", "code_presentation": "category", "id_student": "int32",
          "id_site": "int32", "date": "int32", "sum_click": "int32"}
    keep = []
    for ch in pd.read_csv(vle_path, dtype=dt, usecols=list(dt), chunksize=2_000_000):
        k = list(zip(ch["code_module"].astype(str), ch["code_presentation"].astype(str), ch["id_student"]))
        m = np.fromiter((t in keyset for t in k), bool, len(k))
        keep.append(ch[m])
    sv = pd.concat(keep, ignore_index=True)
    sv["code_module"] = sv["code_module"].astype(str)
    sv["code_presentation"] = sv["code_presentation"].astype(str)
    sv["activity_type"] = sv["id_site"].map(site_to_act)
    return sv


def end_clicks(sv):
    g = sv.groupby(TRIP).agg(total_clicks=("sum_click", "sum"),
                             active_days=("date", "nunique"),
                             unique_resources=("id_site", "nunique")).reset_index()
    q = sv[sv["activity_type"].eq("quiz")].groupby(TRIP)["sum_click"].sum().rename("quiz_clicks").reset_index()
    f = sv[sv["activity_type"].eq("forumng")].groupby(TRIP)["sum_click"].sum().rename("forum_clicks").reset_index()
    return g.merge(q, on=TRIP, how="left").merge(f, on=TRIP, how="left")


def early_features(sv, saj, ass_nonexam, cutoff):
    rows = sv[sv["date"] <= cutoff]
    g = rows.groupby(TRIP).agg(early_total_clicks=("sum_click", "sum"),
                               early_active_days=("date", "nunique"),
                               early_unique_resources=("id_site", "nunique")).reset_index()
    q = rows[rows["activity_type"].eq("quiz")].groupby(TRIP)["sum_click"].sum().rename("early_quiz_clicks").reset_index()
    f = rows[rows["activity_type"].eq("forumng")].groupby(TRIP)["sum_click"].sum().rename("early_forum_clicks").reset_index()
    clk = g.merge(q, on=TRIP, how="left").merge(f, on=TRIP, how="left")

    # Opsi-2 windowed avg_score (window = first `cutoff` days)
    navail = (ass_nonexam[ass_nonexam["date"] <= cutoff]
              .groupby(["code_module", "code_presentation"]).size().rename("navail").reset_index())
    due = set(ass_nonexam.loc[ass_nonexam["date"] <= cutoff, "id_assessment"])
    sub = saj[(saj["assessment_type"] != "Exam") & (saj["date_submitted"] <= cutoff)
              & (saj["id_assessment"].isin(due))]
    earned = sub.groupby(TRIP)["score"].sum().rename("earned").reset_index()
    nsub = sub.groupby(TRIP).size().rename("early_n_submitted").reset_index()
    fday = sub.groupby(TRIP)["date_submitted"].min().rename("early_first_submit_day").reset_index()
    return clk, navail, earned, nsub, fday


# ============================ SYNTH (GPA) ============================
def zscore(x):
    x = np.asarray(x, float); sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def add_synth_gpa(df):
    rng = np.random.default_rng(SYNTH_SEED)
    W_S, W_E, W_Q, W_A, W_N = 0.45, 0.20, 0.15, 0.15, 0.75
    z_avg = zscore(df["avg_score"])
    z_eng = zscore(df[["num_assess_submitted", "active_days", "total_clicks"]].rank(pct=True).mean(axis=1))
    z_qual = zscore(df["highest_education"].map(EDU_MAP).fillna(1))
    z_att = zscore(df["num_of_prev_attempts"])
    theta = zscore(W_S * z_avg + W_E * z_eng + W_Q * z_qual - W_A * z_att + W_N * rng.normal(0, 1, len(df)))
    center = np.clip(2.85 + 0.42 * theta, 0, 4)
    n = len(df)
    a = np.abs(rng.normal(0.45, 0.25, n))
    arch = rng.choice(["steady", "decline", "improve", "crash", "surge", "volatile"],
                      n, p=[0.24, 0.20, 0.18, 0.14, 0.10, 0.14])
    s = rng.choice([-1, 1], n)
    off = np.zeros((n, 3))
    off[arch == "decline"] = np.stack([a, 0 * a, -a], 1)[arch == "decline"]
    off[arch == "improve"] = np.stack([-a, 0 * a, a], 1)[arch == "improve"]
    off[arch == "crash"] = np.stack([0.6 * a, 0.6 * a, -1.2 * a], 1)[arch == "crash"]
    off[arch == "surge"] = np.stack([-0.6 * a, -0.6 * a, 1.2 * a], 1)[arch == "surge"]
    off[arch == "volatile"] = np.stack([s * a, -s * 1.3 * a, s * 0.3 * a], 1)[arch == "volatile"]
    g1 = np.round(np.clip(center + off[:, 0] + rng.normal(0, 0.12, n), 0, 4), 2)
    g2 = np.round(np.clip(center + off[:, 1] + rng.normal(0, 0.12, n), 0, 4), 2)
    g3 = np.round(np.clip(center + off[:, 2] + rng.normal(0, 0.12, n), 0, 4), 2)
    df["GPA_1styear"], df["GPA_2ndyear"], df["GPA_3rdyear"] = g1, g2, g3
    df["CGPA"] = np.round((g1 + g2 + g3) / 3.0, 2)
    return df


# ============================ SYNTH (credits enrolled / passed) ============================
# Anchored on the real OULAD studied_credits, treated as the YEAR-4 (current) load.
# A degree is TOTAL_DEGREE_CREDITS over 4 years; the remaining
# (TOTAL - year4) credits are spread across years 1-3 (not necessarily equally),
# in 30-credit modules. credit_passed for years 1-3 is derived from PER-MODULE
# grades drawn around that year's synthetic GPA: each module gets a grade ~
# Normal(GPA, GRADE_SIGMA) on the 0-4 scale, and the module passes if its grade
# clears the pass threshold (~1.0 = a D). Because both the GPA and the pass/fail
# decisions come from the same grade distribution, they can't contradict: a high
# GPA can't pass zero credits and a GPA well below the threshold can't pass
# everything. A GPA around 2.0 (C) still clears almost every module, just with low
# marks. Year 4 gets NO passed column: its outcome is exactly what the model
# predicts (the dropout target). studied_credits itself is left untouched; for the
# ~1% of rows not a multiple of 30 it is rounded to the nearest 30 only for
# building the transcript, so the four years sum to TOTAL_DEGREE_CREDITS.
TOTAL_DEGREE_CREDITS = 360
YEAR_MIN_UNITS, YEAR_MAX_UNITS = 1, 4   # 30..120 credits per prior year
PASS_THRESHOLD = 1.0    # GPA/grade (0-4) at the pass line (~D)
GRADE_SIGMA = 0.5       # spread of per-module grades around the year GPA


def _compositions(units, lo=YEAR_MIN_UNITS, hi=YEAR_MAX_UNITS):
    """All (a,b,c) with a+b+c=units and each in [lo,hi]."""
    out = []
    for a in range(lo, hi + 1):
        for b in range(lo, hi + 1):
            c = units - a - b
            if lo <= c <= hi:
                out.append((a, b, c))
    return out


def add_synth_credit_counts(df):
    """Synthetic credit_enrolled_/credit_passed_ for years 1-3 + year-4 enrolled
    (= studied_credits) + totals + credits_remaining. All in real credits
    (multiples of 30). Never uses final_result. Must run AFTER add_synth_gpa."""
    rng = np.random.default_rng(SYNTH_SEED + 2)
    n = len(df)

    # year 4 = current load = studied_credits (rounded to nearest 30 for transcript)
    yr4 = (np.round(df["studied_credits"].values / 30.0) * 30).astype(int)
    yr4 = np.clip(yr4, 30, TOTAL_DEGREE_CREDITS - 3 * YEAR_MIN_UNITS * 30)  # keep yrs1-3 feasible
    df["credit_enrolled_4thyear"] = yr4

    R_units = ((TOTAL_DEGREE_CREDITS - yr4) // 30).astype(int)

    # pre-compute valid splits per possible units total, then pick one per row
    comp_cache = {u: _compositions(u) for u in np.unique(R_units)}
    enr = np.zeros((n, 3), dtype=int)
    for i, u in enumerate(R_units):
        opts = comp_cache[u]
        enr[i] = opts[rng.integers(len(opts))]
    enr *= 30   # units -> credits

    tot_enr = np.zeros(n, dtype=int)
    tot_pas = np.zeros(n, dtype=int)
    maxm = YEAR_MAX_UNITS
    col_idx = np.arange(maxm)[None, :]
    for j, (yr, gcol) in enumerate([("1st", "GPA_1styear"), ("2nd", "GPA_2ndyear"),
                                    ("3rd", "GPA_3rdyear")]):
        enrolled = enr[:, j]
        n_modules = (enrolled // 30).astype(int)
        valid = col_idx < n_modules[:, None]
        # per-module grade noise, then RECENTER so the mean grade over the modules
        # a student actually took equals that year's GPA. This makes GPA and the
        # pass/fail decisions consistent: a year with GPA >= PASS_THRESHOLD must
        # have at least one module >= threshold (so credit_passed > 0), and a 1-
        # module year gets exactly its GPA. Failed modules still count toward the
        # (inclusive) GPA, so a low-GPA year can pass 0 credits.
        raw = rng.normal(0.0, GRADE_SIGMA, size=(n, maxm))
        row_mean = np.where(valid, raw, 0.0).sum(axis=1) / np.maximum(n_modules, 1)
        grades = np.clip(df[gcol].values[:, None] + (raw - row_mean[:, None]), 0, 4)
        passed_mask = (grades >= PASS_THRESHOLD) & valid
        passed = passed_mask.sum(axis=1) * 30
        df[f"credit_enrolled_{yr}year"] = enrolled
        df[f"credit_passed_{yr}year"] = passed.astype(int)
        tot_enr += enrolled
        tot_pas += passed

    df["total_credit_enrolled"] = tot_enr            # years 1-3 only
    df["total_credit_passed"] = tot_pas              # years 1-3 only
    df["credits_remaining"] = tot_enr - tot_pas      # unpassed credits carried into yr4
    return df


# ============================ MAIN ============================
def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=BASE)
    ap.add_argument("--vle-path", default=os.path.join(BASE, "studentVle.csv"))
    ap.add_argument("--out", default=os.path.join(BASE, "oulad_early.csv"))
    ap.add_argument("--max-credits", type=int, default=MAX_STUDIED_CREDITS,
                    help="drop rows with studied_credits above this (default 120)")
    args = ap.parse_args()

    base, saj = clean_end_of_module(args.raw_dir, args.vle_path)
    base = filter_implausible_credits(base, args.max_credits)   # <-- before sampling
    ass = pd.read_csv(f"{args.raw_dir}/assessments.csv", na_values=["?"])
    ass["date"] = pd.to_numeric(ass["date"], errors="coerce")
    ass_ne = ass[ass["assessment_type"] != "Exam"]

    sample = stratified_sample(base)                      # < 5000, no avg_score filter
    sv = clickstream_for_sample(args.vle_path, args.raw_dir, sample)

    # end-of-module clicks
    df = sample.merge(end_clicks(sv), on=TRIP, how="left")
    df["has_vle"] = df["total_clicks"].notna().astype(int)
    df["has_assessment"] = (df["num_assess_submitted"] > 0).astype(int)
    for c in ["total_clicks", "active_days", "unique_resources", "quiz_clicks", "forum_clicks"]:
        df[c] = df[c].fillna(0).astype(int)

    # early windowed (one feature block per cutoff in CUTOFFS -> suffix _<cut>)
    for cut in CUTOFFS:
        clk, navail, earned, nsub, fday = early_features(sv, saj, ass_ne, cut)
        d = df[TRIP].merge(navail, on=["code_module", "code_presentation"], how="left") \
                    .merge(earned, on=TRIP, how="left").merge(nsub, on=TRIP, how="left") \
                    .merge(fday, on=TRIP, how="left").merge(clk, on=TRIP, how="left")
        d["earned"] = d["earned"].fillna(0.0)
        d[f"early_avg_score_{cut}"] = np.where(d["navail"].fillna(0) > 0, (d["earned"] / d["navail"]).round(2), 0.0)
        d[f"early_n_submitted_{cut}"] = d["early_n_submitted"].fillna(0).astype(int)
        # ABS applied: negative submit days (submissions before module start) are
        # turned positive so early_first_submit_day is always >= 0. See docstring
        # for the interpretation caveat (early vs late submitters collapse).
        d[f"early_first_submit_day_{cut}"] = d["early_first_submit_day"].fillna(0).abs().astype(int)
        for col in ["early_total_clicks", "early_active_days", "early_unique_resources",
                    "early_quiz_clicks", "early_forum_clicks"]:
            d[f"{col}_{cut}"] = d[col].fillna(0).astype(int)
        wanted = ([f"early_avg_score_{cut}", f"early_n_submitted_{cut}", f"early_first_submit_day_{cut}"]
                  + [f"{col}_{cut}" for col in ["early_total_clicks", "early_active_days",
                     "early_unique_resources", "early_quiz_clicks", "early_forum_clicks"]])
        df = df.merge(d[TRIP + wanted], on=TRIP, how="left")

    df = add_synth_gpa(df)                                # GPA/CGPA on the sample
    df = add_synth_credit_counts(df)                      # credit_enrolled/passed 1st/2nd/3rd + totals
    df = df.drop(columns=["final_result", "date_unregistration", "date_registration"], errors="ignore")
    df.to_csv(args.out, index=False)
    print(f"[done] {args.out} shape={df.shape} NaN={int(df.isna().sum().sum())} "
          f"dropout%={(df['target']=='dropout').mean():.3f}")


if __name__ == "__main__":
    main()