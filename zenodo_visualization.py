from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# CONFIG
# ============================================================
# Zenodo (zenodo.csv) pipeline. Same three-model / two-defence story as the
# OULAD script, wired to the zenodo result files instead.
#   - Random Forest : attacked with FGSM transferred from a source MLP
#   - MLP           : attacked with white-box PGD
#   - XGBoost       : attacked with FGSM transferred from a source MLP
# Per split, every model runs the same journey:
#   baseline (clean) -> under attack -> after adversarial training (under attack)
#   -> after detect-and-default (under attack).
# Detect-and-default has two actions:
#   - default_dropout : a flagged row is forced to 'dropout'. Gives a clean,
#                       comparable F1, so this is the action used in the main
#                       pipeline figures.
#   - abstain         : a flagged row is rejected. Its dropout-F1 is computed on
#                       the ACCEPTED rows only, so abstain is shown separately as
#                       a coverage trade-off (flag rate + accuracy on accepted),
#                       never as a headline F1.
# All epsilon values are read from the result files, never hardcoded.
#
# One naming quirk of the zenodo files worth knowing: the MLP is called
# "MLP Main" in the attack file, "MLP Defended" in the adversarial-training file,
# and "MLP" in the DND / recovery / detector files. build_stage_table reads each
# per-model file directly so the in-file name does not matter there, and the
# recovery/detector lookups match on "MLP", which is what those files use.

OUTPUT_DIR = Path("zenodo_adversarial_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_TO_PLOT: tuple[str, ...] = ("test", "validation")

# zenodo-style result subfolders first, plain filenames as a fallback, so this
# runs both from the pipeline tree and from a flat folder. existing_paths()
# additionally rglobs by filename, so even an unexpected subfolder name is found.
ATTACK_FILES = {
    "rf": ["fgsm_rf_zenodo_results/zenodo_rf_fgsm_attack_results.csv", "zenodo_rf_fgsm_attack_results.csv"],
    "mlp": ["pgd_mlp_zenodo_results/zenodo_mlp_pgd_attack_results.csv", "zenodo_mlp_pgd_attack_results.csv"],
    "xgb": ["fgsm_xgb_zenodo_results/zenodo_xgb_fgsm_attack_results.csv", "zenodo_xgb_fgsm_attack_results.csv"],
}
ADVTRAIN_FILES = {
    "rf": ["rf_adversarial_training_zenodo_results/zenodo_rf_adversarial_training_results.csv",
           "zenodo_rf_adversarial_training_results.csv"],
    "mlp": ["mlp_adversarial_training_zenodo_results/zenodo_mlp_adversarial_training_results.csv",
            "zenodo_mlp_adversarial_training_results.csv"],
    "xgb": ["xgb_adversarial_training_zenodo_results/zenodo_xgb_adversarial_training_results.csv",
            "zenodo_xgb_adversarial_training_results.csv"],
}
DND_FILES = {
    "rf": ["rf_detect_and_default_zenodo_results/zenodo_rf_detect_and_default_results.csv",
           "zenodo_rf_detect_and_default_results.csv"],
    "mlp": ["mlp_dnd_feature_classcond_zenodo_results/zenodo_mlp_detect_and_default_results.csv",
            "zenodo_mlp_detect_and_default_results.csv"],
    "xgb": ["xgb_detect_and_default_zenodo_results/zenodo_xgb_detect_and_default_results.csv",
            "zenodo_xgb_detect_and_default_results.csv"],
}
ADVTRAIN_RECOVERY_CANDIDATES = ["zenodo_attack_recovery_summary.csv"]
DND_RECOVERY_CANDIDATES = ["zenodo_dnd_attack_recovery_summary.csv"]
PROX_DND_CANDIDATES = [
    "proximity_sensitivity_detect_and_default_zenodo_results/zenodo_proximity_sensitivity_dnd_summary.csv",
    "zenodo_proximity_sensitivity_dnd_summary.csv",
]
PROX_ADVTRAIN_CANDIDATES = [
    "proximity_sensitivity_adv_training_zenodo_results/zenodo_proximity_sensitivity_adv_training_summary.csv",
    "zenodo_proximity_sensitivity_adv_training_summary.csv",
]
FEATURE_DND_CANDIDATES = [
    "proximity_sensitivity_detect_and_default_zenodo_results/zenodo_top15_sensitive_features_dnd.csv",
    "proximity_sensitivity_detect_and_default_zenodo_results/zenodo_feature_sensitivity_dnd_summary.csv",
    "zenodo_top15_sensitive_features_dnd.csv",
    "zenodo_feature_sensitivity_dnd_summary.csv",
]

# fixed presentation order and per-model metadata
MODELS = [
    {"key": "rf", "label": "Random Forest", "attack": "FGSM", "attack_kw": "fgsm", "color_key": "rf"},
    {"key": "mlp", "label": "MLP", "attack": "PGD", "attack_kw": "pgd", "color_key": "mlp"},
    {"key": "xgb", "label": "XGBoost", "attack": "FGSM", "attack_kw": "fgsm", "color_key": "xgb"},
]
MODEL_ORDER = [m["label"] for m in MODELS]

# ============================================================
# THEME (colourblind-friendly, consistent across every figure)
# ============================================================
COL = {
    "rf": "#3B6EA5",          # Random Forest
    "mlp": "#E1812C",         # MLP
    "xgb": "#6A4C93",         # XGBoost
    "clean": "#2A9D8F",       # clean / safe / baseline
    "attack": "#D1495B",      # under attack
    "advtrain": "#2E5E8C",    # adversarial-training defence
    "dnd": "#4C9F70",         # detect-and-default defence
    "abstain": "#C58A3A",     # abstain action
    "def_clean": "#8AB6D6",   # defended clean
    "text": "#222222",
    "muted": "#6B6B6B",
    "grid": "#E3E3E3",
}
MODEL_COLOR = {"Random Forest": COL["rf"], "MLP": COL["mlp"], "XGBoost": COL["xgb"]}


def apply_theme():
    # IJACSA wants Times New Roman at 300 DPI. Times New Roman is used when it is
    # installed (e.g. on Windows); on machines without it, Liberation Serif is a
    # metric-identical substitute and STIX keeps any Greek/math glyphs in the
    # same serif style. savefig.dpi=300 (plus dpi=300 in save_plot) gives 300 DPI.
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COL["grid"],
        "grid.linewidth": 0.9,
        "xtick.color": COL["text"],
        "ytick.color": COL["text"],
        "text.color": COL["text"],
        "legend.frameon": False,
    })


def style_axis(ax, ygrid=True, xgrid=False, ymax=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def bar_value_labels(ax, bars, values, fmt="{:.2f}", dy=0.012, fontsize=9, color=None):
    valid = [v for v in values if pd.notna(v)]
    top = max(valid + [0])
    pad = dy if top <= 1.2 else top * dy
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                fmt.format(value), ha="center", va="bottom",
                fontsize=fontsize, color=color or COL["text"])


import textwrap


def figure_caption(fig, text):
    # wrapped so long captions stay inside the figure width
    fig.text(0.5, -0.02, textwrap.fill(text, 170), ha="center", va="top",
             fontsize=8.5, color=COL["muted"], style="italic")


def save_plot(fig, filename):
    # every figure filename is prefixed with zenodo_ on save, so all .png outputs
    # carry the dataset tag (matches the zenodo_ prefix used across the pipeline).
    path = OUTPUT_DIR / f"zenodo_{filename}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# DATA LOADING
# ============================================================

# files are searched under this root when a literal candidate path is missing;
# set to the folder that holds all the zenodo result files/subfolders.
SEARCH_ROOT = Path(".")



# ------------------------------------------------------------------
# Result-file picking (fixed 2026-09-25).
# The IJACSA2 tree holds several copies of the same result file (for example
# "taro di claude", "taro di claude - Copy", "zenodo - Copy"). Taking the copy
# with the shortest path silently picked stale files. Rule now:
#   1) ignore copies inside folders whose name contains one of STALE_DIR_TERMS,
#      unless no other copy exists;
#   2) among the remaining copies take the NEWEST (last modified);
#   3) print a warning whenever the copies differ in content.
# ------------------------------------------------------------------
import hashlib

STALE_DIR_TERMS = ("copy", "backup", "taro di claude")
_NOTED = set()


def _is_stale_location(path: Path) -> bool:
    folders = [p.lower() for p in Path(path).resolve().parts[:-1]]
    return any(term in folder for folder in folders for term in STALE_DIR_TERMS)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_best_copy(matches: List[Path], label: str = "") -> Optional[Path]:
    matches = [Path(m) for m in matches if Path(m).exists()]
    if not matches:
        return None
    unique, seen = [], set()
    for m in matches:
        r = m.resolve()
        if r not in seen:
            unique.append(m)
            seen.add(r)
    preferred = [m for m in unique if not _is_stale_location(m)]
    pool = preferred if preferred else unique
    best = max(pool, key=lambda m: m.stat().st_mtime)
    if len(unique) > 1:
        hashes = {_md5(m) for m in unique}
        if len(hashes) > 1 and best.resolve() not in _NOTED:
            _NOTED.add(best.resolve())
            print(f"  NOTE {label or best.name}: {len(unique)} copies with DIFFERENT content; "
                  f"using the newest non-backup copy -> {best}")
        if not preferred:
            print(f"  WARNING {best.name}: only backup/copy folders hold this file; using {best}")
    return best


def existing_paths(candidates: Iterable[str]) -> List[Path]:
    # for each candidate, use the literal path if it exists; otherwise search the
    # whole tree under SEARCH_ROOT for a file with the same name (handles result
    # files that live in per-experiment subfolders). candidate order is kept, so
    # the preferred path still wins.
    # one best copy per candidate name, candidate order kept (fixed 2026-09-25)
    paths, seen = [], set()
    for c in candidates:
        p = Path(c)
        matches: List[Path] = [p] if p.exists() else []
        try:
            matches += list(SEARCH_ROOT.rglob(p.name))
        except OSError:
            pass
        best = pick_best_copy(matches, p.name)
        if best is not None and best.resolve() not in seen:
            paths.append(best)
            seen.add(best.resolve())
    return paths


def read_first_csv(candidates: Iterable[str], label: str, required: bool = False) -> Optional[pd.DataFrame]:
    # take the first candidate that exists (candidates are already priority-ordered)
    for path in existing_paths(candidates):
        df = pd.read_csv(path)
        print(f"Loaded {label}: {path}")
        return df
    if required:
        checked = "\n".join(str(Path(c)) for c in candidates)
        raise FileNotFoundError(f"required {label} not found. checked:\n{checked}")
    print(f"Skipped optional {label}: no matching file found.")
    return None


def normalize_metrics(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    # unify the dropout metric names and make sure split/condition columns exist
    if df is None:
        return None
    df = df.copy()
    rename = {}
    if "f1_dropout" in df.columns and "f1_score" not in df.columns:
        rename["f1_dropout"] = "f1_score"
    if "precision_dropout" in df.columns and "precision" not in df.columns:
        rename["precision_dropout"] = "precision"
    if "recall_dropout" in df.columns and "recall" not in df.columns:
        rename["recall_dropout"] = "recall"
    df = df.rename(columns=rename)
    if "eval_epsilon" not in df.columns:
        df["eval_epsilon"] = df["epsilon"] if "epsilon" in df.columns else np.nan
    if "condition" not in df.columns:
        df["condition"] = ""
    df["split_norm"] = df["split"].astype(str).str.lower() if "split" in df.columns else "unknown"
    df["condition_norm"] = df["condition"].astype(str).str.lower()
    if "action" in df.columns:
        df["action_norm"] = df["action"].astype(str).str.lower()
    return df


def metric(row: Optional[pd.Series], col: str) -> float:
    if row is None or col not in row.index or pd.isna(row[col]):
        return np.nan
    return float(row[col])


# ============================================================
# ROW SELECTION helpers (epsilon read from data, never hardcoded)
# ============================================================

def _split_slice(df, split):
    return df[df["split_norm"] == split.lower()].copy()


def _pick(df, split, *, clean, attack_kw=None, action=None, condition_terms=None):
    # generic single-row selector; when several epsilons exist take the largest
    if df is None or df.empty:
        return None
    sub = _split_slice(df, split)
    if "action_norm" in sub.columns and action is not None:
        sub = sub[sub["action_norm"] == action.lower()]
    if condition_terms is not None:
        for term in condition_terms:
            sub = sub[sub["condition_norm"].str.contains(term.lower(), na=False)]
    eval_eps = pd.to_numeric(sub["eval_epsilon"], errors="coerce").fillna(0.0)
    if clean:
        sub = sub[eval_eps == 0.0]
    else:
        sub = sub[eval_eps > 0.0]
        if attack_kw is not None:
            sub = sub[sub["condition_norm"].str.contains(attack_kw.lower(), na=False)]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["_e"] = pd.to_numeric(sub["eval_epsilon"], errors="coerce")
    return sub.sort_values("_e", ascending=False).iloc[0]


# ============================================================
# STAGE TABLE (one tidy frame that every figure reads from)
# ============================================================

def build_stage_table(split, attack_data, advtrain_data, dnd_data) -> pd.DataFrame:
    rows = []

    def add(model_label, stage, row, hiding_from=None):
        prec = metric(row, "precision")
        rec = metric(row, "recall")
        f1 = metric(row, "f1_score")
        acc = metric(row, "accuracy")
        # dropout hiding rate = flips / attacked dropout rows, within this defence
        hide = np.nan
        src = hiding_from if hiding_from is not None else row
        flips = metric(src, "dropout_to_nondropout_flips")
        attacked = metric(src, "attacked_rows_count")
        if pd.notna(flips) and pd.notna(attacked) and attacked > 0:
            hide = flips / attacked
        rows.append({
            "split": split, "model": model_label, "stage": stage,
            "eval_epsilon": metric(row, "eval_epsilon"),
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
            "flag_rate": metric(row, "flag_rate"),
            "n_flagged": metric(row, "n_flagged"),
            "attacked_rows_count": metric(row, "attacked_rows_count"),
            "dropout_hiding_rate": hide,
        })

    for m in MODELS:
        key, label, kw = m["key"], m["label"], m["attack_kw"]
        atk = attack_data.get(key)
        adv = advtrain_data.get(key)
        dnd = dnd_data.get(key)

        add(label, "Baseline clean", _pick(atk, split, clean=True, condition_terms=["clean"]))
        add(label, "Under attack", _pick(atk, split, clean=False, attack_kw=kw))
        add(label, "Adv-train clean", _pick(adv, split, clean=True, condition_terms=["defended", "clean"]))
        add(label, "Adv-train attack", _pick(adv, split, clean=False, attack_kw=kw,
                                             condition_terms=["defend"]))
        # detect-and-default: default_dropout is the headline action
        add(label, "DND clean",
            _pick(dnd, split, clean=True, action="default_dropout", condition_terms=["defended"]))
        add(label, "DND attack",
            _pick(dnd, split, clean=False, action="default_dropout", condition_terms=["defended"]))
        # abstain kept separate for the coverage trade-off figure
        add(label, "DND clean (abstain)",
            _pick(dnd, split, clean=True, action="abstain", condition_terms=["defended"]))
        add(label, "DND attack (abstain)",
            _pick(dnd, split, clean=False, action="abstain", condition_terms=["defended"]))

    return pd.DataFrame(rows)


def fill_dnd_hiding_rate(perf, dnd_rec, split):
    # hidden after default-to-dropout = students hidden by the attack whose adversarial
    # record the detector did NOT flag, over all attacked rows (same denominator as the
    # no-defence hiding rate). Fixed 2026-09-25.
    if perf is None or dnd_rec is None or perf.empty:
        return perf
    perf = perf.copy()
    for model in MODEL_ORDER:
        s = dnd_rec[(dnd_rec["model"] == model) & (dnd_rec["split"].astype(str).str.lower() == split.lower())]
        mask = (perf["model"] == model) & (perf["stage"] == "DND attack")
        if s.empty or not mask.any():
            continue
        r = s.iloc[0]
        attacked = perf.loc[mask, "attacked_rows_count"].iloc[0]
        hidden = r.get("successfully_attacked_count", np.nan)
        rec = r.get("recovered_default_dropout_count", np.nan)
        if pd.notna(attacked) and attacked > 0 and pd.notna(hidden) and pd.notna(rec):
            perf.loc[mask, "dropout_hiding_rate"] = (float(hidden) - float(rec)) / float(attacked)
    return perf


def get_val(df, model, stage, col):
    sub = df[(df["model"] == model) & (df["stage"] == stage)]
    if sub.empty or col not in sub.columns or pd.isna(sub.iloc[0][col]):
        return np.nan
    return float(sub.iloc[0][col])


def attack_eps(df, model):
    v = get_val(df, model, "Under attack", "eval_epsilon")
    return v if pd.notna(v) else None


def eps_str(value):
    return f"{value:g}" if value is not None and pd.notna(value) else "?"


def grouped_positions(n_models, n_groups):
    x = np.arange(n_models)
    total = 0.8
    w = total / n_groups
    offsets = [(-total / 2) + w / 2 + i * w for i in range(n_groups)]
    return x, w, offsets


# ============================================================
# FIGURE 01 / 11: METRIC JOURNEY across the four pipeline stages
# ============================================================
# One grouped-bar figure showing a single metric at four stages
# (baseline clean -> under attack -> adv-training -> detect-and-default),
# for all three models. Used for F1 (headline) and for recall on its own.

def _plot_metric_journey(df, split, metric_col, metric_label, title, filename, extra_note=""):
    stages = ["Baseline clean", "Under attack", "Adv-train attack", "DND attack"]
    legend = ["Baseline (clean)", "Under attack", "After adv-training", "After adversarial detection"]
    colours = [COL["clean"], COL["attack"], COL["advtrain"], COL["dnd"]]
    data = {s: [get_val(df, m, s, metric_col) for m in MODEL_ORDER] for s in stages}
    if all(all(pd.isna(v) for v in vals) for vals in data.values()):
        print(f"Skipped {metric_label} journey ({split}).")
        return
    x, w, offsets = grouped_positions(len(MODEL_ORDER), len(stages))
    fig, ax = plt.subplots(figsize=(12, 6))
    for stage, lab, col, off in zip(stages, legend, colours, offsets):
        vals = data[stage]
        bars = ax.bar(x + off, np.nan_to_num(vals), w, label=lab, color=col)
        bar_value_labels(ax, bars, vals, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{title} ({split} set)", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    style_axis(ax, ymax=1.05)
    eps_note = " / ".join(f"{m}: {get_stage_attack(df, m)}" for m in MODEL_ORDER)
    tail = f" {extra_note}" if extra_note else ""
    figure_caption(fig, f"Attack budget read from data (\u03b5). {eps_note}. "
                        f"Adversarial detection uses default-to-dropout. Higher is better. "
                        "Adversarial training is evaluated with FGSM transferred from the adversarially trained MLP (Random Forest, XGBoost) and with adaptive PGD on the defended MLP; adversarial detection keeps the baseline model and is evaluated with FGSM from the baseline MLP (Random Forest, XGBoost) and PGD on the baseline MLP."
                        f"{tail}")
    save_plot(fig, filename)


def plot_f1_journey(df, split):
    _plot_metric_journey(
        df, split, "f1_score", "Dropout F1-score",
        "Dropout-detection F1 across the robustness pipeline",
        f"01_{split}_f1_journey.png")


def plot_recall_journey(df, split):
    _plot_metric_journey(
        df, split, "recall", "Dropout recall",
        "Dropout recall across the robustness pipeline",
        f"11_{split}_recall_journey.png",
        extra_note="Recall = share of true dropout students the model still catches; "
                   "this is the metric the hide-the-dropout attack targets.")


def get_stage_attack(df, model):
    label = {"Random Forest": "FGSM", "MLP": "PGD", "XGBoost": "FGSM"}[model]
    e = attack_eps(df, model)
    return f"{label} \u03b5={eps_str(e)}"


# ============================================================
# FIGURE 02: DROP AND RECOVERY (per-model panels, delta arrows)
# ============================================================

def _round_disp(v, d=2):
    return v if pd.isna(v) else round(float(v), d)


def _drop_recovery_panel(ax, df, model):
    attack = {"Random Forest": "FGSM", "MLP": "PGD", "XGBoost": "FGSM"}[model]
    base = _round_disp(get_val(df, model, "Baseline clean", "f1_score"))
    under = _round_disp(get_val(df, model, "Under attack", "f1_score"))
    adv = _round_disp(get_val(df, model, "Adv-train attack", "f1_score"))
    dnd = _round_disp(get_val(df, model, "DND attack", "f1_score"))
    vals = [base, under, adv, dnd]
    labels = ["Baseline\n(clean)", f"{attack}\nattack", "Adv-train\n(attack)", "Adv. detection\n(attack)"]
    colours = [COL["clean"], COL["attack"], COL["advtrain"], COL["dnd"]]
    bars = ax.bar(range(4), np.nan_to_num(vals), color=colours, width=0.66)
    bar_value_labels(ax, bars, vals, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Dropout F1-score")
    ax.set_title(model, color=MODEL_COLOR[model])
    style_axis(ax, ymax=1.12)
    # red arrow: F1 lost to the attack
    if pd.notna(base) and pd.notna(under):
        ax.annotate("", xy=(1, under + 0.03), xytext=(0, base + 0.03),
                    arrowprops=dict(arrowstyle="->", color=COL["attack"], lw=1.6))
        ax.text(0.5, max(base, under) + 0.08, f"{under - base:+.2f}", ha="center",
                color=COL["attack"], fontsize=8.5, fontweight="bold")
    # blue arrow: recovery from adversarial training
    if pd.notna(under) and pd.notna(adv):
        ax.annotate("", xy=(2, adv + 0.03), xytext=(1, under + 0.03),
                    arrowprops=dict(arrowstyle="->", color=COL["advtrain"], lw=1.5))
        ax.text(1.5, max(under, adv) + 0.08, f"{adv - under:+.2f}", ha="center",
                color=COL["advtrain"], fontsize=8.5, fontweight="bold")
    # green arrow: recovery from detect-and-default, measured from the same
    # attacked floor as the blue arrow so both deltas are "vs under attack"
    if pd.notna(under) and pd.notna(dnd):
        ax.annotate("", xy=(3, dnd + 0.03), xytext=(1, under + 0.03),
                    arrowprops=dict(arrowstyle="->", color=COL["dnd"], lw=1.5,
                                    connectionstyle="arc3,rad=-0.18"))
        ax.text(2.15, max(under, dnd) + 0.12, f"{dnd - under:+.2f}",
                ha="center", color=COL["dnd"], fontsize=8.5, fontweight="bold")


def plot_drop_recovery(df, split):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        _drop_recovery_panel(ax, df, model)
    fig.suptitle(f"F1 collapse under attack and recovery by each defence ({split} set)",
                 fontsize=14, fontweight="bold")
    figure_caption(fig, "Red arrow: F1 lost to the attack. Blue/green: recovery vs the attacked level "
                        "(adversarial training / adversarial detection). Deltas use displayed 2-dp values.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_plot(fig, f"02_{split}_drop_and_recovery.png")


# ============================================================
# FIGURE 03: PRECISION / RECALL / F1 UNDER ATTACK, per defence
# ============================================================

def _prf_panel(ax, df, model):
    metrics = ["precision", "recall", "f1_score"]
    names = ["Precision", "Recall", "F1"]
    series = {
        "Under attack (no defence)": ([get_val(df, model, "Under attack", mm) for mm in metrics], COL["attack"]),
        "Adv-training": ([get_val(df, model, "Adv-train attack", mm) for mm in metrics], COL["advtrain"]),
        "Adversarial detection": ([get_val(df, model, "DND attack", mm) for mm in metrics], COL["dnd"]),
    }
    x, w, offsets = grouped_positions(3, len(series))
    for (lab, (vals, col)), off in zip(series.items(), offsets):
        bars = ax.bar(x + off, np.nan_to_num(vals), w, label=lab, color=col)
        bar_value_labels(ax, bars, vals, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title(model, color=MODEL_COLOR[model])
    style_axis(ax, ymax=1.08)


def plot_prf_under_attack(df, split):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        _prf_panel(ax, df, model)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"Precision, recall and F1 under attack, by defence ({split} set)",
                 y=0.95, fontsize=14, fontweight="bold")
    figure_caption(fig, "Recall is what the hide-the-dropout attack destroys most; each defence is judged on "
                        "how much recall it buys back. " "Adversarial training is evaluated with FGSM transferred from the adversarially trained MLP (Random Forest, XGBoost) and with adaptive PGD on the defended MLP; adversarial detection keeps the baseline model and is evaluated with FGSM from the baseline MLP (Random Forest, XGBoost) and PGD on the baseline MLP.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save_plot(fig, f"03_{split}_precision_recall_f1_under_attack.png")


# ============================================================
# FIGURE 04: DROPOUT HIDING RATE (safety framing)
# ============================================================

def plot_hiding_rate(df, split):
    stages = ["Under attack", "Adv-train attack", "DND attack"]
    legend = ["No defence", "Adv-training", "Adversarial detection"]
    colours = [COL["attack"], COL["advtrain"], COL["dnd"]]
    data = {s: [get_val(df, m, s, "dropout_hiding_rate") for m in MODEL_ORDER] for s in stages}
    if all(all(pd.isna(v) for v in vals) for vals in data.values()):
        print(f"Skipped hiding rate ({split}).")
        return
    x, w, offsets = grouped_positions(len(MODEL_ORDER), len(stages))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for stage, lab, col, off in zip(stages, legend, colours, offsets):
        vals = data[stage]
        bars = ax.bar(x + off, np.nan_to_num(vals), w, label=lab, color=col)
        bar_value_labels(ax, bars, vals, fmt="{:.0%}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("True dropouts hidden (flip rate)")
    ax.set_title(f"Share of attacked dropout students the model still misses ({split} set)", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    allv = [v for vals in data.values() for v in vals if pd.notna(v)]
    top = max(allv + [0.1])
    style_axis(ax, ymax=min(1.0, top * 1.4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    figure_caption(fig, "Share of attacked dropout rows that end up predicted non-dropout, under each defence's own "
                        "evaluation. For adversarial detection (default-to-dropout) a hidden student counts as caught "
                        "when the detector flags the adversarial record. Lower is safer. " "Adversarial training is evaluated with FGSM transferred from the adversarially trained MLP (Random Forest, XGBoost) and with adaptive PGD on the defended MLP; adversarial detection keeps the baseline model and is evaluated with FGSM from the baseline MLP (Random Forest, XGBoost) and PGD on the baseline MLP.")
    save_plot(fig, f"04_{split}_dropout_hiding_rate.png")


# ============================================================
# FIGURE 05: CLEAN-PERFORMANCE COST OF EACH DEFENCE
# ============================================================

def plot_clean_cost(df, split):
    stages = ["Baseline clean", "Adv-train clean", "DND clean"]
    legend = ["Baseline (undefended)", "Adv-training", "Adversarial detection"]
    colours = [COL["clean"], COL["advtrain"], COL["dnd"]]
    data = {s: [get_val(df, m, s, "f1_score") for m in MODEL_ORDER] for s in stages}
    if all(all(pd.isna(v) for v in vals) for vals in data.values()):
        print(f"Skipped clean cost ({split}).")
        return
    x, w, offsets = grouped_positions(len(MODEL_ORDER), len(stages))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    baseline_vals = data["Baseline clean"]
    for stage, lab, col, off in zip(stages, legend, colours, offsets):
        vals = data[stage]
        bars = ax.bar(x + off, np.nan_to_num(vals), w, label=lab, color=col)
        bar_value_labels(ax, bars, vals, fontsize=8)
        # signed gap vs baseline for the two defences
        if stage != "Baseline clean":
            for xi, (b, v) in enumerate(zip(baseline_vals, vals)):
                if pd.notna(b) and pd.notna(v):
                    ax.text(x[xi] + off, v + 0.05, f"{v - b:+.2f}", ha="center",
                            color=COL["muted"], fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Clean dropout F1-score")
    ax.set_title(f"Cost of each defence on clean (un-attacked) data ({split} set)", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    style_axis(ax, ymax=1.05)
    figure_caption(fig, "Clean F1 before vs after each defence. A small drop means the defence keeps normal "
                        "accuracy when no attack is present. Adversarial detection uses default-to-dropout.")
    save_plot(fig, f"05_{split}_clean_cost_of_defence.png")


# ============================================================
# FIGURE 06: RECOVERY RATE (adv-training vs detect-and-default)
# ============================================================

def _recovery_lookup(adv_rec, dnd_rec, model, split):
    adv = dnd = np.nan
    if adv_rec is not None:
        s = adv_rec[(adv_rec["model"] == model) & (adv_rec["split"].str.lower() == split.lower())]
        if not s.empty:
            adv = float(s.iloc[0]["recovery_rate"])
    if dnd_rec is not None:
        s = dnd_rec[(dnd_rec["model"] == model) & (dnd_rec["split"].str.lower() == split.lower())]
        if not s.empty:
            adv_col = "recovery_rate_default_dropout" if "recovery_rate_default_dropout" in s.columns else "recovery_rate"
            dnd = float(s.iloc[0][adv_col])
    return adv, dnd


def plot_recovery_rate(adv_rec, dnd_rec, split):
    if adv_rec is None and dnd_rec is None:
        print(f"Skipped recovery rate ({split}): no summary files.")
        return
    adv_vals, dnd_vals = [], []
    for m in MODEL_ORDER:
        a, d = _recovery_lookup(adv_rec, dnd_rec, m, split)
        adv_vals.append(a)
        dnd_vals.append(d)
    if all(pd.isna(v) for v in adv_vals + dnd_vals):
        print(f"Skipped recovery rate ({split}): model names did not match.")
        return
    x, w, offsets = grouped_positions(len(MODEL_ORDER), 2)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    b1 = ax.bar(x + offsets[0], np.nan_to_num(adv_vals), w, label="Adversarial training", color=COL["advtrain"])
    b2 = ax.bar(x + offsets[1], np.nan_to_num(dnd_vals), w, label="Adversarial detection", color=COL["dnd"])
    bar_value_labels(ax, b1, adv_vals, fmt="{:.0%}", fontsize=9)
    bar_value_labels(ax, b2, dnd_vals, fmt="{:.0%}", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Recovery rate on successfully-attacked students")
    ax.set_title(f"How many hidden dropouts each defence recovers ({split} set)", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    style_axis(ax, ymax=1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    figure_caption(fig, "Recovery rate = share of students hidden by the attack that the defence puts back on the "
                        "dropout list. Taken directly from the recovery-analysis files. Adversarial detection uses default-to-dropout. "
                        "Adversarial training is evaluated with FGSM transferred from the adversarially trained MLP (Random Forest, XGBoost) and with adaptive PGD on the defended MLP; adversarial detection keeps the baseline model and is evaluated with FGSM from the baseline MLP (Random Forest, XGBoost) and PGD on the baseline MLP.")
    save_plot(fig, f"06_{split}_recovery_rate.png")


# ============================================================
# FIGURE 07: DETECTOR DIAGNOSTICS (why DND works or fails)
# ============================================================

def plot_detector(dnd_rec, split):
    if dnd_rec is None:
        print(f"Skipped detector diagnostics ({split}): no DND recovery file.")
        return
    recall, flag_clean, flag_adv = [], [], []
    for m in MODEL_ORDER:
        s = dnd_rec[(dnd_rec["model"] == m) & (dnd_rec["split"].str.lower() == split.lower())]
        if s.empty:
            recall.append(np.nan); flag_clean.append(np.nan); flag_adv.append(np.nan)
            continue
        r = s.iloc[0]
        recall.append(float(r.get("detection_recall_on_attacked", np.nan)))
        flag_clean.append(float(r.get("flag_rate_clean", np.nan)))
        flag_adv.append(float(r.get("flag_rate_adv", np.nan)))
    if all(pd.isna(v) for v in recall + flag_clean + flag_adv):
        print(f"Skipped detector diagnostics ({split}).")
        return
    x, w, offsets = grouped_positions(len(MODEL_ORDER), 3)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    b1 = ax.bar(x + offsets[0], np.nan_to_num(recall), w, label="Detection recall on hidden dropouts", color=COL["dnd"])
    b2 = ax.bar(x + offsets[1], np.nan_to_num(flag_clean), w, label="Flag rate (clean)", color=COL["def_clean"])
    b3 = ax.bar(x + offsets[2], np.nan_to_num(flag_adv), w, label="Flag rate (under attack)", color=COL["attack"])
    bar_value_labels(ax, b1, recall, fmt="{:.0%}", fontsize=8)
    bar_value_labels(ax, b2, flag_clean, fmt="{:.0%}", fontsize=8)
    bar_value_labels(ax, b3, flag_adv, fmt="{:.0%}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Rate")
    ax.set_title(f"Mahalanobis detector: catch rate vs false alarms ({split} set)", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    style_axis(ax, ymax=1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    figure_caption(fig, "Detection recall = share of successfully attacked (hidden) dropout students whose adversarial record the detector flags. All three models use the same raw-feature-space Mahalanobis detector (empirical covariance), calibrated to a 5% false-alarm rate on the clean test split; the flag rate on clean rows is the false-positive cost.")
    save_plot(fig, f"07_{split}_dnd_detector.png")


# ============================================================
# FIGURE 08: ABSTAIN vs DEFAULT-DROPOUT TRADE-OFF
# ============================================================

def plot_action_tradeoff(df, split):
    # abstain: accuracy is on ACCEPTED rows only, so pair it with flag rate (coverage lost)
    acc_default = [get_val(df, m, "DND attack", "accuracy") for m in MODEL_ORDER]
    acc_abstain = [get_val(df, m, "DND attack (abstain)", "accuracy") for m in MODEL_ORDER]
    flag_abstain = [get_val(df, m, "DND attack (abstain)", "flag_rate") for m in MODEL_ORDER]
    if all(pd.isna(v) for v in acc_default + acc_abstain):
        print(f"Skipped action trade-off ({split}).")
        return
    x, w, offsets = grouped_positions(len(MODEL_ORDER), 3)
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x + offsets[0], np.nan_to_num(acc_default), w,
                label="default-to-dropout accuracy", color=COL["dnd"])
    b2 = ax.bar(x + offsets[1], np.nan_to_num(acc_abstain), w,
                label="abstain accuracy (accepted)", color=COL["abstain"])
    b3 = ax.bar(x + offsets[2], np.nan_to_num(flag_abstain), w,
                label="abstain flag rate (coverage lost)", color=COL["muted"])
    bar_value_labels(ax, b1, acc_default, fmt="{:.2f}", fontsize=8)
    bar_value_labels(ax, b2, acc_abstain, fmt="{:.2f}", fontsize=8)
    bar_value_labels(ax, b3, flag_abstain, fmt="{:.0%}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Accuracy / flag rate (under attack)")
    ax.set_title(f"Adversarial detection: default-to-dropout vs abstain ({split} set)", pad=42)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5)
    style_axis(ax, ymax=1.12)
    figure_caption(fig, "default-to-dropout forces flagged rows to 'dropout' and keeps full coverage; abstain rejects "
                        "them, so its accuracy is measured only on accepted rows and its flag rate is the fraction "
                        "handed off for human review. Dropout-F1 is not shown for abstain because it is computed on a "
                        "residual set.")
    save_plot(fig, f"08_{split}_dnd_action_tradeoff.png")


# ============================================================
# FIGURE 09: PROXIMITY / SENSITIVITY (perturbation size on attacked rows)
# ============================================================

def _norm_model_name(name):
    n = str(name).lower()
    if "random" in n or n.strip() == "rf":
        return "Random Forest"
    if "xgb" in n or "xgboost" in n:
        return "XGBoost"
    if "mlp" in n:
        return "MLP"
    return name


def _prox_attacked(prox_df, split):
    if prox_df is None or prox_df.empty:
        return None
    df = prox_df.copy()
    df["split_norm"] = df["split"].astype(str).str.lower()
    df = df[df["split_norm"] == split.lower()]
    if "row_scope" in df.columns:
        att = df[df["row_scope"].astype(str).str.lower() == "attacked_rows_only"]
        if not att.empty:
            df = att
    if df.empty:
        return None
    df = df.copy()
    df["model_clean"] = df["model"].apply(_norm_model_name)
    return df


def _prox_bar(ax, df, col, title, ylabel):
    if df is None or col not in df.columns:
        ax.set_visible(False)
        return False
    present = [m for m in MODEL_ORDER if m in set(df["model_clean"])]
    if not present:
        ax.set_visible(False)
        return False
    d = df.set_index("model_clean").reindex(present)
    values = pd.to_numeric(d[col], errors="coerce")
    colours = [MODEL_COLOR[m] for m in present]
    bars = ax.bar(range(len(present)), np.nan_to_num(values.values), color=colours, width=0.6)
    bar_value_labels(ax, bars, values.values, fmt="{:.2f}", fontsize=9)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    top = float(np.nanmax(values.values)) if len(values) else 1.0
    style_axis(ax, ymax=top * 1.3 if top > 0 else 1.0)
    return True


def plot_proximity(prox_df, split):
    df = _prox_attacked(prox_df, split)
    if df is None:
        print(f"Skipped proximity ({split}): no data.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    ok1 = _prox_bar(axes[0], df, "mean_l2_attackable", "Mean L2 perturbation",
                    "Mean L2 (attackable features)")
    ok2 = _prox_bar(axes[1], df, "mean_sensitivity", "Mean sensitivity", "Mean sensitivity")
    if not (ok1 or ok2):
        plt.close(fig)
        print(f"Skipped proximity ({split}): required columns missing.")
        return
    fig.suptitle(f"How hard the attack had to push on each model ({split} set, attacked rows)",
                 fontsize=14, fontweight="bold")
    figure_caption(fig, "Mean L2 distance and mean sensitivity between clean and adversarial inputs in the "
                        "standardized feature space, computed on attacked rows only, for the attack on the baseline models "
                        "(same epsilon for all). Random Forest and XGBoost: FGSM transferred from the baseline MLP. "
                        "MLP: white-box PGD.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save_plot(fig, f"09_{split}_proximity_sensitivity.png")


# ============================================================
# FIGURE 10: TOP SENSITIVE FEATURES per model
# ============================================================

def _top_features_panel(ax, feat_df, split, model_keyword, model_label, colour, top_n=10):
    df = feat_df.copy()
    df["split_norm"] = df["split"].astype(str).str.lower()
    df = df[df["split_norm"] == split.lower()]
    if "row_scope" in df.columns:
        att = df[df["row_scope"].astype(str).str.lower() == "attacked_rows_only"]
        if not att.empty:
            df = att
    if "is_attackable_feature" in df.columns:
        df = df[df["is_attackable_feature"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    df = df[df["model"].astype(str).str.lower().str.contains(model_keyword.lower(), na=False)]
    feat_col = "feature_name" if "feature_name" in df.columns else None
    sens_col = "mean_sensitivity" if "mean_sensitivity" in df.columns else (
        "mean_abs_change" if "mean_abs_change" in df.columns else None)
    if df.empty or feat_col is None or sens_col is None:
        ax.set_visible(False)
        return False
    top = (df.groupby(feat_col, as_index=False)
             .agg(v=(sens_col, "mean"))
             .sort_values("v", ascending=False).head(top_n)
             .sort_values("v", ascending=True))
    # trim the sklearn preprocessing prefixes for readability
    names = top[feat_col].astype(str).str.replace(r"^(num__|cat__|remainder__)", "", regex=True)
    bars = ax.barh(names, top["v"], color=colour)
    offset = max(top["v"].max() * 0.01, 0.001)
    for bar, value in zip(bars, top["v"]):
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                va="center", fontsize=8, color=COL["text"])
    ax.set_xlabel("Mean sensitivity")
    ax.set_title(model_label, color=colour)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    return True


def plot_top_features(feat_df, split):
    if feat_df is None or feat_df.empty:
        print(f"Skipped top features ({split}): no feature file.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    specs = [("random", "Random Forest (FGSM)", COL["rf"]),
             ("mlp", "MLP (PGD)", COL["mlp"]),
             ("xgb", "XGBoost (FGSM)", COL["xgb"])]
    any_ok = False
    for ax, (kw, lab, col) in zip(axes, specs):
        any_ok |= _top_features_panel(ax, feat_df, split, kw, lab, col)
    if not any_ok:
        plt.close(fig)
        print(f"Skipped top features ({split}): required columns not found.")
        return
    fig.suptitle(f"Most-perturbed attackable features ({split} set)", fontsize=14, fontweight="bold")
    figure_caption(fig, "Features the attack leaned on most to disguise a dropout student (attacked rows).")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_plot(fig, f"10_{split}_top_sensitive_features.png")


# ============================================================
# TABLES
# ============================================================

def save_tables(combined, adv_rec, dnd_rec):
    csv_path = OUTPUT_DIR / "zenodo_performance_summary.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    excel_path = OUTPUT_DIR / "zenodo_visualization_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Performance by stage", index=False)
        if adv_rec is not None:
            adv_rec.to_excel(writer, sheet_name="Recovery adv-training", index=False)
        if dnd_rec is not None:
            dnd_rec.to_excel(writer, sheet_name="Recovery detect-default", index=False)
    print(f"Saved: {excel_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    apply_theme()
    print("=" * 80)
    print("UPV (ZENODO) ADVERSARIAL ROBUSTNESS VISUALIZATION (thesis edition)")
    print("=" * 80)
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Splits           : {list(SPLITS_TO_PLOT)}")
    print(f"Models           : {[m['label'] for m in MODELS]}\n")

    attack_data = {k: normalize_metrics(read_first_csv(v, f"{k.upper()} attack", required=(k == 'rf')))
                   for k, v in ATTACK_FILES.items()}
    advtrain_data = {k: normalize_metrics(read_first_csv(v, f"{k.upper()} adversarial training",
                                                         required=(k == 'rf')))
                     for k, v in ADVTRAIN_FILES.items()}
    dnd_data = {k: normalize_metrics(read_first_csv(v, f"{k.upper()} detect-and-default"))
                for k, v in DND_FILES.items()}

    adv_rec = read_first_csv(ADVTRAIN_RECOVERY_CANDIDATES, "adv-training recovery summary")
    dnd_rec = read_first_csv(DND_RECOVERY_CANDIDATES, "detect-and-default recovery summary")
    prox_dnd = read_first_csv(PROX_DND_CANDIDATES, "proximity/sensitivity (DND)")
    prox_adv = read_first_csv(PROX_ADVTRAIN_CANDIDATES, "proximity/sensitivity (adv-training)")
    feature_dnd = read_first_csv(FEATURE_DND_CANDIDATES, "feature sensitivity (DND)")

    all_perf = []
    for split in SPLITS_TO_PLOT:
        print(f"\n---- {split} ----")
        perf = build_stage_table(split, attack_data, advtrain_data, dnd_data)
        perf = fill_dnd_hiding_rate(perf, dnd_rec, split)
        if perf.empty:
            print(f"No rows for split '{split}'.")
            continue
        cols = ["model", "stage", "eval_epsilon", "accuracy", "precision", "recall",
                "f1_score", "flag_rate", "dropout_hiding_rate"]
        print(perf[[c for c in cols if c in perf.columns]].to_string(index=False))

        plot_f1_journey(perf, split)
        plot_recall_journey(perf, split)
        plot_drop_recovery(perf, split)
        plot_prf_under_attack(perf, split)
        plot_hiding_rate(perf, split)
        plot_clean_cost(perf, split)
        plot_recovery_rate(adv_rec, dnd_rec, split)
        plot_detector(dnd_rec, split)
        plot_action_tradeoff(perf, split)
        # the DND proximity file covers all three models against the detector;
        # fall back to the adv-training proximity file if it is missing
        plot_proximity(prox_dnd if prox_dnd is not None else prox_adv, split)
        plot_top_features(feature_dnd, split)
        all_perf.append(perf)

    if not all_perf:
        raise RuntimeError("no performance rows for any split. check the result files.")
    save_tables(pd.concat(all_perf, ignore_index=True), adv_rec, dnd_rec)
    print("\nDone. Figures and tables saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()