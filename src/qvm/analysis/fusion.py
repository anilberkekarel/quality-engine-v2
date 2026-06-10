"""STEP 4 — joint multi-channel HMM + ablation + the pre-defined separation test.

Step-3 finding being tested: the patent-only HMM could NOT separate MU
(control, cyclical) from NVDA ex ante — MU even switched up EARLIER (2017Q3
vs 2019Q1). Fig3 showed the separating information lives in the SECOND
channel: MU's margins collapse while its patents stay high; NVDA's channels
rise together.

ABLATION (structural, per company):
  A) patent-only      — Step 3 results REUSED verbatim, never refit
                        (the comparison bar must not move);
  B) financial-only   — gross margin + revenue YoY, no NB channel;
  C) joint            — all three channels, one latent state.
logL/BIC are NOT comparable across A/B/C (different observation sets) — no
likelihood race; the comparison is REGIME DATES. The predictive scoreboard
(out-of-sample fundamentals) is Step 5.

SEPARATION TEST (pre-defined; the result is reported however it comes out —
no tuning toward the hypothesis):
  - MU: does joint filtered P(high) exit with the margin collapse (~2019)
    earlier than patent-only's 2023Q1 exit?
  - NVDA: when does the joint model enter the high regime; does persistence
    survive?
  - AMD/MRVL: described honestly (AMD carries a suspected Xilinx-M&A
    composition effect in its patent channel — noted, not modeled).

Scope caveat (stamped in every output): Step 4 is DESCRIPTIVE — filing-dated
patents, as-filed financials, FULL-SAMPLE parameters, matching the Step-3
baseline. The real-time version (publication-dated channel, expanding-window
re-estimation, knowable_at discipline) is Step 5.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config
from ..models.fusion_hmm import Channel, FusionFit, fit_fusion_hmm
from .regimes import POINT_IN_TIME_CAVEAT, persistent_switches

logger = logging.getLogger(__name__)

FUSION_CAVEAT = ("descriptive Step-4 analysis (filing-dated patents, as_filed "
                 "financials, full-sample parameters); " + POINT_IN_TIME_CAVEAT)

_FIN_CHANNELS = ("gross_margin", "revenue_yoy")


# --------------------------------------------------------------------------- #
# data assembly: financial channels mapped onto the Step-3 quarterly grid
# --------------------------------------------------------------------------- #
def build_channel_set(res: dict, channels_df: pd.DataFrame) -> list[Channel]:
    """Step-3 grid + counts, financials joined by calendar quarter (NaN where
    absent — pre-2009 etc.; the emission simply skips those terms)."""
    quarters, y = res["quarters"], res["y"]
    g = channels_df[channels_df["ticker"] == res["ticker"]].set_index("quarter")
    gm = np.full(len(quarters), np.nan)
    yoy = np.full(len(quarters), np.nan)
    for i, q in enumerate(quarters):
        if str(q) in g.index:
            row = g.loc[str(q)]
            gm[i] = row["gross_margin"] if pd.notna(row["gross_margin"]) else np.nan
            yoy[i] = (row["revenue_yoy_growth"]
                      if pd.notna(row["revenue_yoy_growth"]) else np.nan)
    return [
        Channel("patent_count", "nb", y.astype(float)),
        Channel("gross_margin", "normal", gm),
        Channel("revenue_yoy", "normal", yoy),
    ]


# --------------------------------------------------------------------------- #
# model suite per company (A reused from Step 3 — NOT refit)
# --------------------------------------------------------------------------- #
def analyze_company_fusion(res: dict, channels_df: pd.DataFrame,
                           seed: int = 0) -> dict:
    chs = build_channel_set(res, channels_df)
    fin_chs = [c for c in chs if c.name in _FIN_CHANNELS]
    kw = dict(n_restarts=config.HMM_N_RESTARTS, max_iter=config.HMM_MAX_ITER,
              tol=config.HMM_TOL, seed=seed)
    # B: financial-only — no patent channel exists, so "high" is defined by
    # the (structural) gross-margin mean, documented in the outputs
    fit_b = fit_fusion_hmm(fin_chs, n_states=2, order_channel="gross_margin", **kw)
    fit_b3 = fit_fusion_hmm(fin_chs, n_states=3, order_channel="gross_margin", **kw)
    # C: joint — ordered by the patent mean for comparability with Step 3
    fit_c = fit_fusion_hmm(chs, n_states=2, order_channel="patent_count", **kw)
    fit_c3 = fit_fusion_hmm(chs, n_states=3, order_channel="patent_count", **kw)
    out = {
        "ticker": res["ticker"], "label": res["label"],
        "quarters": res["quarters"], "y": res["y"], "note": res["note"],
        "channels": chs,
        "A": None,  # Step-3 fit lives in res["nb2"]; kept there, never refit
        "B": fit_b, "B3": fit_b3, "C": fit_c, "C3": fit_c3,
        "switches": {
            "A": res["switches"],   # reused verbatim from Step 3
            "B": persistent_switches(res["quarters"], fit_b.filtered[:, -1]),
            "C": persistent_switches(res["quarters"], fit_c.filtered[:, -1]),
        },
        "step3": res,
    }
    logger.info("[%s] fusion fit done (B conv=%s, C conv=%s)",
                res["ticker"], fit_b.converged, fit_c.converged)
    return out


def analyze_all_fusion(step3_results: list[dict],
                       channels_df: pd.DataFrame) -> list[dict]:
    return [analyze_company_fusion(r, channels_df) for r in step3_results]


# --------------------------------------------------------------------------- #
# output tables
# --------------------------------------------------------------------------- #
def fusion_probabilities_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        for model in ("B", "C"):
            fit: FusionFit = res[model]
            for i, q in enumerate(res["quarters"]):
                rows.append({
                    "ticker": res["ticker"], "model": model, "quarter": str(q),
                    "p_high_filtered": float(fit.filtered[i, 1]),
                    "p_high_smoothed": float(fit.smoothed[i, 1]),
                    "viterbi_state": int(fit.viterbi_path[i]),
                    "caveat": FUSION_CAVEAT if i == 0 else "",
                })
    return pd.DataFrame(rows)


def _fmt_state_params(fit: FusionFit, k: int) -> dict:
    out = {}
    for name in fit.channel_names:
        p = fit.channel_params[name]
        if fit.families[name] == "nb":
            out[f"{name}_mean_s{k}"] = round(float(p["mus"][k]), 2)
        else:
            out[f"{name}_mean_s{k}"] = round(float(p["means"][k]), 4)
            out[f"{name}_sigma_s{k}"] = round(float(p["sigmas"][k]), 4)
    return out


def fusion_parameters_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        for model in ("B", "C"):
            fit: FusionFit = res[model]
            fit3: FusionFit = res[model + "3"]
            spread = max(fit.restart_logliks) - min(fit.restart_logliks)
            at_best = sum(1 for l in fit.restart_logliks
                          if abs(l - fit.log_likelihood) < 1e-3)
            row = {"ticker": res["ticker"], "model": model,
                   "channels": "+".join(fit.channel_names),
                   "order_channel": fit.order_channel,
                   "dispersion_r": None if fit.r is None else round(fit.r, 3)}
            for k in range(fit.n_states):
                row.update(_fmt_state_params(fit, k))
            row.update({
                "transmat": " | ".join(f"{p:.3f}" for p in fit.transmat.ravel()),
                "expected_durations_q": " | ".join(
                    f"{d:.1f}" for d in fit.expected_durations),
                "log_likelihood": round(fit.log_likelihood, 2),
                "bic_2state": round(fit.bic, 1),
                "bic_3state": round(fit3.bic, 1),
                "em_converged": fit.converged,
                "restarts_at_best_logl": f"{at_best}/{len(fit.restart_logliks)}",
                "restart_logl_spread": round(spread, 3),
                "caveat": FUSION_CAVEAT,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def ablation_switch_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        for model in ("A", "B", "C"):
            sw = res["switches"][model]
            if not sw:
                rows.append({"ticker": res["ticker"], "model": model,
                             "quarter": None, "direction": "none",
                             "p_high_at_switch": None})
            for s in sw:
                rows.append({"ticker": res["ticker"], "model": model, **s})
    df = pd.DataFrame(rows)
    df["note"] = ""
    df.loc[df.index[:1], "note"] = (
        "A=patent-only (Step 3, reused), B=financial-only, C=joint; "
        "logL/BIC NOT comparable across models (different observation sets); "
        + FUSION_CAVEAT)
    return df


def separation_test_table(results: list[dict]) -> pd.DataFrame:
    """The pre-defined headline test: high-regime entry/exit dates per model."""
    rows = []
    for res in results:
        for model in ("A", "B", "C"):
            sw = res["switches"][model]
            entries = [s["quarter"] for s in sw if s["direction"] == "up"]
            exits = [s["quarter"] for s in sw if s["direction"] == "down"]
            rows.append({
                "ticker": res["ticker"], "model": model,
                "high_regime_entries": "; ".join(entries) or "none",
                "high_regime_exits": "; ".join(exits) or "none",
                "still_high_at_sample_end": bool(
                    (res["step3"]["nb2"] if model == "A"
                     else res[model]).filtered[-1, 1] >= 0.5),
            })
    df = pd.DataFrame(rows)
    df["pre_defined_questions"] = ""
    df.loc[df.index[:1], "pre_defined_questions"] = (
        "MU: does C exit with the ~2019 margin collapse, earlier than A's "
        "2023Q1? NVDA: when does C enter, does persistence survive? "
        "AMD/MRVL: describe honestly (AMD patent channel carries a suspected "
        "Xilinx-M&A composition effect). " + FUSION_CAVEAT)
    return df


# --------------------------------------------------------------------------- #
# data-derived event markers for Fig5 (rule documented; no hand-tuning)
# --------------------------------------------------------------------------- #
def event_markers(channels_df: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """Descriptive markers: MU = starts of margin-collapse episodes (as-filed
    gross margin down >=15pp vs 4 quarters earlier); NVDA = first quarter
    with revenue YoY >= +100%. Derived from channels.csv by rule."""
    out: dict[str, list[tuple[str, str]]] = {}
    mu = channels_df[channels_df["ticker"] == "MU"].sort_values("quarter")
    drop = mu["gross_margin"] - mu["gross_margin"].shift(4)
    in_episode = False
    for q, d in zip(mu["quarter"], drop):
        if pd.notna(d) and d <= -0.15 and not in_episode:
            out.setdefault("MU", []).append((q, "margin collapse"))
            in_episode = True
        elif pd.notna(d) and d > -0.15:
            in_episode = False
    nv = channels_df[channels_df["ticker"] == "NVDA"].sort_values("quarter")
    boom = nv[nv["revenue_yoy_growth"] >= 1.0]
    if not boom.empty:
        out["NVDA"] = [(boom["quarter"].iloc[0], "revenue +100% YoY")]
    return out
