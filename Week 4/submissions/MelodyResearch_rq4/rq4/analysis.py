"""Analysis from saved results only. No model, no inference.

Order of aggregation (protocol): per-pair effects -> mean over the (up to) four conditions of each
base item -> ten-item mean with a base-item clustered bootstrap. Everything is stratified by layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

FMT_SHORT = {"multiple_choice": "mc", "open_ended": "oe"}


# ------------------------------------------------------------------ loading ----
def load_results(path: Path = C.RAW_JSONL) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([flatten_row(r) for r in rows])
    if df["key"].duplicated().any():
        raise ValueError("duplicate evaluation keys in results")
    return df


def flatten_row(r: dict) -> dict:
    keep = ["key", "stage", "base_item_id", "pair_id", "prompt_id", "context_condition", "question_polarity",
            "prompt_format", "prompt_hash", "arm", "layer", "seed", "status", "status_detail", "correct_role",
            "category", "parse_rule", "generation_raw", "follows_evidence", "evidence_applicable",
            "tf_margin_stereo_minus_counter", "activation_norm", "intended_displacement", "intended_ratio",
            "measured_displacement", "measured_ratio", "calibration_ratio", "target_ratio", "cos_with_target"]
    out = {k: r.get(k) for k in keep}
    for role, c in (r.get("teacher_forced") or {}).items():
        out[f"lp_{role}"] = c.get("logprob_sum")
    for rl, m in (r.get("readouts") or {}).items():
        out[f"rank_gap_{rl}"] = m["rank_gap"]
        out[f"set_margin_{rl}"] = m["set_logit_margin"]
    return out


# ------------------------------------------------------------ bootstrap -------
def bootstrap_mean_ci(values: np.ndarray, n_boot: int = C.N_BOOT, seed: int = C.BOOT_SEED, level: float = 0.95):
    """Clustered (by base item) bootstrap of the mean: one value per cluster in, resample clusters."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return {"mean": float(v.mean()), "lo": float(np.quantile(means, a)), "hi": float(np.quantile(means, 1 - a)), "n": int(len(v))}


# ------------------------------------------------------- per-pair effects -----
def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (prompt_id, layer) with clean + each arm's margin/category side by side."""
    clean = df[df.arm == C.CLEAN].set_index("prompt_id")
    parts = []
    for layer in C.LAYERS:
        sub = df[(df.layer == layer)]
        base = clean[["base_item_id", "pair_id", "context_condition", "question_polarity", "prompt_format",
                      "tf_margin_stereo_minus_counter", "category", "follows_evidence", "evidence_applicable"]].rename(
            columns={"tf_margin_stereo_minus_counter": "margin_clean", "category": "cat_clean",
                     "follows_evidence": "ev_clean"})
        base = base.copy()
        base["layer"] = layer
        for arm in C.ARMS_INTERVENTION:
            a = sub[sub.arm == arm].set_index("prompt_id")
            base[f"status_{arm}"] = a["status"]
            base[f"margin_{arm}"] = a["tf_margin_stereo_minus_counter"]
            base[f"cat_{arm}"] = a["category"]
            base[f"ev_{arm}"] = a["follows_evidence"]
            base[f"disp_{arm}"] = a["measured_displacement"]
        parts.append(base.reset_index())
    return pd.concat(parts, ignore_index=True)


def per_pair_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Matched-pair table. Rows: (pair_id, layer). Columns per format (mc_/oe_) and MC-minus-open diffs.

    Continuous family needs finite margins for clean, target and all three random seeds; behavioural family
    needs only ok status + a category for them. Each family has its own validity flag, so a missing
    teacher-forced score never silently drops behavioural evidence (and vice versa). A pair/layer is kept only
    if BOTH formats are valid (matched pairs only)."""
    w = _wide(df)
    rnd = list(C.ARM_RANDOM)
    ok = lambda r, arms: all(r[f"status_{a}"] == "ok" for a in arms)
    recs = []
    for (pair, layer), g in w.groupby(["pair_id", "layer"]):
        rec = {"pair_id": pair, "layer": layer, "base_item_id": g.base_item_id.iloc[0],
               "context_condition": g.context_condition.iloc[0], "question_polarity": g.question_polarity.iloc[0]}
        for _, r in g.iterrows():
            f = FMT_SHORT[r.prompt_format]
            arms_ok = ok(r, [C.ARM_TARGET] + rnd)
            margins = [r.margin_clean] + [r[f"margin_{a}"] for a in [C.ARM_TARGET] + rnd]
            cont_ok = arms_ok and all(m is not None and np.isfinite(m) for m in margins)
            rec[f"{f}_cont_valid"] = bool(cont_ok)
            if cont_ok:
                ab = lambda a: abs(r[f"margin_{a}"] - r.margin_clean)
                rand_abs = [ab(a) for a in rnd]
                rec[f"{f}_abs_target"] = ab(C.ARM_TARGET)
                rec[f"{f}_abs_random_mean"] = float(np.mean(rand_abs))
                rec[f"{f}_abs_random_seeds"] = rand_abs
                rec[f"{f}_excess_abs"] = ab(C.ARM_TARGET) - float(np.mean(rand_abs))
                rec[f"{f}_signed_target"] = r[f"margin_{C.ARM_TARGET}"] - r.margin_clean
                rec[f"{f}_signed_random_mean"] = float(np.mean([r[f"margin_{a}"] - r.margin_clean for a in rnd]))
                if r[f"status_{C.ARM_SHAM}"] == "ok" and r[f"margin_{C.ARM_SHAM}"] is not None:
                    rec[f"{f}_excess_abs_sham"] = ab(C.ARM_TARGET) - ab(C.ARM_SHAM)
            beh_ok = arms_ok and isinstance(r.cat_clean, str) and all(isinstance(r[f"cat_{a}"], str) for a in [C.ARM_TARGET] + rnd)
            rec[f"{f}_beh_valid"] = bool(beh_ok)
            if beh_ok:
                chg = lambda a: float(r[f"cat_{a}"] != r.cat_clean)
                rec[f"{f}_changed_target"] = chg(C.ARM_TARGET)
                rec[f"{f}_changed_random_mean"] = float(np.mean([chg(a) for a in rnd]))
                rec[f"{f}_excess_changed"] = chg(C.ARM_TARGET) - rec[f"{f}_changed_random_mean"]
                if r[f"status_{C.ARM_SHAM}"] == "ok" and isinstance(r[f"cat_{C.ARM_SHAM}"], str):
                    rec[f"{f}_excess_changed_sham"] = chg(C.ARM_TARGET) - chg(C.ARM_SHAM)
                if r.evidence_applicable:
                    rec[f"{f}_ev_acc_clean"] = float(bool(r.ev_clean))
                    rec[f"{f}_ev_acc_target"] = float(bool(r[f"ev_{C.ARM_TARGET}"]))
                    rec[f"{f}_ev_acc_random_mean"] = float(np.mean([bool(r[f"ev_{a}"]) for a in rnd]))
        for fam, col in (("cont", "excess_abs"), ("beh", "excess_changed"), ("cont_sham", "excess_abs_sham"), ("beh_sham", "excess_changed_sham")):
            if f"mc_{col}" in rec and f"oe_{col}" in rec and rec.get(f"mc_{fam.split('_')[0]}_valid") and rec.get(f"oe_{fam.split('_')[0]}_valid"):
                rec[f"diff_{col}"] = rec[f"mc_{col}"] - rec[f"oe_{col}"]
        recs.append(rec)
    return pd.DataFrame(recs)


# ------------------------------------------------------ item aggregation ------
VALUE_COLS = ["mc_excess_abs", "oe_excess_abs", "diff_excess_abs", "mc_excess_abs_sham", "oe_excess_abs_sham",
              "diff_excess_abs_sham", "mc_signed_target", "oe_signed_target", "mc_excess_changed", "oe_excess_changed",
              "diff_excess_changed", "mc_excess_changed_sham", "oe_excess_changed_sham", "diff_excess_changed_sham"]


def per_item(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean over the available conditions within each base item (aggregate first), per layer."""
    cols = [c for c in VALUE_COLS if c in pairs.columns]
    g = pairs.groupby(["base_item_id", "layer"])
    out = g[cols].mean().reset_index()
    out["n_conditions"] = g.pair_id.nunique().values
    return out


def summarize(items: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or [c for c in VALUE_COLS if c in items.columns]
    recs = []
    for layer, g in items.groupby("layer"):
        for c in cols:
            vals = g[c].dropna().values
            recs.append({"layer": layer, "metric": c, **bootstrap_mean_ci(vals)})
    return pd.DataFrame(recs)


def seed_spread(pairs: pd.DataFrame) -> pd.DataFrame:
    """SD across the three random seeds of the mean absolute margin change (pairs averaged first)."""
    recs = []
    for (layer), g in pairs.groupby("layer"):
        for f in ("mc", "oe"):
            col = f"{f}_abs_random_seeds"
            if col not in g.columns:
                continue
            arr = np.array([s for s in g[col].dropna()], dtype=float)
            if arr.size:
                m = arr.mean(axis=0)
                recs.append({"layer": layer, "format": f, "seed_means": m.tolist(), "seed_sd": float(m.std(ddof=1)) if len(m) > 1 else float("nan")})
    return pd.DataFrame(recs)


def category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each parsed category by format/arm/layer (includes hedged and unparseable)."""
    d = df[df.status == "ok"].copy()
    d["layer"] = d["layer"].fillna(-1)
    t = d.groupby(["prompt_format", "layer", "arm", "category"]).size().rename("n").reset_index()
    t["rate"] = t.n / t.groupby(["prompt_format", "layer", "arm"]).n.transform("sum")
    return t


def displacement_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.arm != C.CLEAN)]
    s = d.groupby(["layer", "arm", "status"]).agg(n=("key", "size"), mean_ratio=("measured_ratio", "mean"),
                                                  max_ratio=("measured_ratio", "max"),
                                                  mean_calibration=("calibration_ratio", "mean")).reset_index()
    return s


# ------------------------------------------------------------- gates ----------
def completion_status(df: pd.DataFrame, *, model_available: bool = True, machine_safe: bool = True) -> dict:
    """PILOT_ONLY < 10 base items; BLOCKED if model/machine unavailable; INCONCLUSIVE if gate items fail."""
    if not model_available or not machine_safe:
        return {"status": "BLOCKED", "reasons": ["model unavailable or machine unsafe"]}
    if df.empty:
        return {"status": "NOT_RUN", "reasons": ["no results"]}
    reasons = []
    if df.base_item_id.nunique() < len(C.BASE_ITEM_IDS):
        return {"status": "PILOT_ONLY", "reasons": [f"{df.base_item_id.nunique()} of {len(C.BASE_ITEM_IDS)} base items"]}
    expected = len(C.BASE_ITEM_IDS) * 4 * 2 * (1 + len(C.LAYERS) * len(C.ARMS_INTERVENTION))
    if len(df) != expected:
        reasons.append(f"{len(df)} rows != expected {expected}")
    if df.key.duplicated().any():
        reasons.append("duplicate keys")
    bad = df[df.status != "ok"]
    if len(bad):
        reasons.append(f"{len(bad)} non-ok rows ({bad.status.value_counts().to_dict()})")
    if df[df.arm != C.CLEAN].query("status == 'ok'").generation_raw.isna().any():
        reasons.append("missing generations")
    pairs = per_pair_effects(df)
    for l in C.LAYERS:
        n_valid = int(((pairs.layer == l) & pairs["diff_excess_abs"].notna()).sum()) if "diff_excess_abs" in pairs else 0
        if n_valid < 10:
            reasons.append(f"only {n_valid} matched valid pairs at L{l} (fewer than 10)")
    return {"status": "INCONCLUSIVE" if reasons else "COMPLETE_GATE_MET", "reasons": reasons}
