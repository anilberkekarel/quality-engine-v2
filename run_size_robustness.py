"""QVM-V2 STEP 5b — POST-HOC size-confound robustness (NOT pre-registered).

PURPOSE (a brake, not a positive hunt): Step 5b's only statistically
meaningful negative cell is the Tier-1 oracle patent channel,
  A_oracle  IC(t1_h4) = -0.072,  NW t = -2.58.
Before the Discussion reads that as "more patenting -> lower future growth",
we must rule out the mechanical alternative: a giant patent portfolio is a
proxy for a giant company, and big companies mechanically sit low in the
forward revenue-growth cross-section. This appendix conditions the A channel
on firm SIZE and asks whether the negative relation survives within-size.

NO NEW MODEL FIT. We reuse the pre-registered feature/target cache
(expanding_regime_probs.csv) and only (1) add a size proxy and (2) recompute
ICs. This file CANNOT change any pre-registered result; it exists solely to
calibrate the interpretation in Discussion.

Size proxy: log(revenue, AS-FILED), built with the SAME point-in-time rule as
sb.baseline_features — at cutoff = last day of quarter t, the newest revenue
observation with filed_date <= cutoff and fiscal quarter in [t-STALENESS_Q, t].
So size is knowable_at the prediction date, exactly like D1/F0.

  Tier-1 deconfound: per prediction quarter, rank the feature and size in the
    cross-section, regress feature-rank on size-rank (OLS), and take the IC of
    the RESIDUAL against the target (a semi-partial / part correlation — only
    the feature is residualized, per the task spec). Before vs after, on the
    identical paired names per quarter.
  Tier-2 deconfound: F0s = F0 + log-revenue. Compare IC(F0), IC(F0s), and the
    paired marginals IC(F0s+A) - IC(F0s) for the feasible and oracle A channels
    separately (the OLS learns A's sign, so a non-zero marginal = A carries
    growth information beyond fundamentals+size). Expanding-window OLS on
    rank-transformed inputs, identical to the pre-registered Tier-2 combiner.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import logging

import numpy as np
import pandas as pd

from qvm import config
from qvm.universe_registry import universe_specs
from qvm.analysis import scoreboard as sb
from run_scoreboard import load_financial_panel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("qvm.size_robustness")

PROBS_CSV = os.path.join(config.OUTPUT_DIR, "expanding_regime_probs.csv")
OUT_CSV = os.path.join(config.OUTPUT_DIR, "scoreboard_size_robustness.csv")

POSTHOC = ("POST-HOC robustness appendix (NOT pre-registered): size-confound "
           "check for the A patent channel. No new model fit — reuses the "
           "pre-registered expanding_regime_probs.csv cache. Size = "
           "log(revenue, as-filed), same knowable_at/point-in-time rule as the "
           "baseline features. Calibrates the Discussion's reading of the "
           "A_oracle negative IC ONLY; does not alter any pre-registered result.")

# F0s adds log-revenue to the pre-registered F0 = [rev YoY, margin, dmargin].
TIER2_SIZE = {
    "F0":         ["f_yoy", "f_gm", "f_dgm"],
    "F0s":        ["f_yoy", "f_gm", "f_dgm", "size"],
    "F0s+A_feas": ["f_yoy", "f_gm", "f_dgm", "size", "A_feas"],
    "F0s+A_orac": ["f_yoy", "f_gm", "f_dgm", "size", "A_orac"],
}
# rows must support every combiner above AND be paired across feasible/oracle.
INTERSECTION_COLS = ["f_yoy", "f_gm", "f_dgm", "size", "A_feas", "A_orac"]
LAG = 4  # NW lag = horizon, matching the pre-registered T1-h4 metric


# --------------------------------------------------------------------------- #
def size_proxy(panel: pd.DataFrame, dates: list[pd.Period]) -> pd.DataFrame:
    """log(revenue, as-filed) per (ticker, t) — mirrors sb.baseline_features'
    point-in-time 'newest' rule applied to the revenue level."""
    rows = []
    for t in dates:
        cutoff = str(t.end_time.date())
        sub = panel[(panel["quarter"] <= t) & (panel["quarter"] >= t - sb.STALENESS_Q)]
        for ticker, g in sub.groupby("ticker", sort=False):
            k = g[g["rev"].notna() & g["rev_filed"].notna()
                  & (g["rev_filed"].fillna("9999") <= cutoff)]
            if len(k):
                rev = float(k.sort_values("quarter")["rev"].iloc[-1])
                if rev > 0:
                    rows.append({"ticker": ticker, "quarter": t,
                                 "size": float(np.log(rev))})
    return pd.DataFrame(rows)


def ic_series(df: pd.DataFrame, col: str, target: str,
              dates: list[pd.Period]) -> pd.Series:
    out = {}
    for t in dates:
        g = df[df["quarter"] == t]
        out[t] = sb.spearman_ic(g[col].to_numpy(float), g[target].to_numpy(float))
    return pd.Series(out)


def pairwise_ic(df: pd.DataFrame, a: str, b: str,
                dates: list[pd.Period]) -> pd.Series:
    """Per-quarter Spearman between two feature columns (confound diagnostic)."""
    out = {}
    for t in dates:
        g = df[df["quarter"] == t]
        out[t] = sb.spearman_ic(g[a].to_numpy(float), g[b].to_numpy(float))
    return pd.Series(out)


def residualize_ic(df: pd.DataFrame, feature: str, target: str,
                   dates: list[pd.Period]) -> tuple[pd.Series, pd.Series]:
    """Per prediction quarter, on names with feature+size+target all present:
    rank feature & size, regress feature-rank ~ 1 + size-rank, return
    (raw IC, residual IC) of feature vs target on the IDENTICAL paired names."""
    raw, res = {}, {}
    for t in dates:
        g = df[df["quarter"] == t]
        m = (g[feature].notna() & g["size"].notna() & g[target].notna()).to_numpy()
        if m.sum() < sb.MIN_CROSS_SECTION:
            raw[t] = res[t] = np.nan
            continue
        f = g[feature].to_numpy(float)[m]
        s = g["size"].to_numpy(float)[m]
        y = g[target].to_numpy(float)[m]
        rf = pd.Series(f).rank().to_numpy()
        rs = pd.Series(s).rank().to_numpy().reshape(-1, 1)
        beta = sb.fit_ols(rs, rf)
        resid = rf - sb.predict_ols(beta, rs)
        raw[t] = sb.spearman_ic(f, y)
        res[t] = sb.spearman_ic(resid, y)
    return pd.Series(raw), pd.Series(res)


def tier2_size(cm: pd.DataFrame, target: str, horizon: int,
               scored: list[pd.Period]) -> pd.DataFrame:
    """Expanding-window OLS combiners on rank-transformed intersection rows —
    identical mechanics to run_scoreboard.tier2_predictions, F0s combiner set."""
    cm = cm.copy()
    rk = lambda c: c + "_rk"
    for name in TIER2_SIZE:
        cm[f"{name}|{target}"] = np.nan
    for t in scored:
        cutoff = str(t.end_time.date())
        train = cm[(cm["quarter"] <= t - horizon) & cm[target].notna()
                   & cm[target + "_filed"].notna()
                   & (cm[target + "_filed"].fillna("9999") <= cutoff)]
        if len(train) < sb.MIN_TRAIN_ROWS:
            continue
        cur = cm["quarter"] == t
        if not cur.any():
            continue
        for name, cols in TIER2_SIZE.items():
            X = train[[rk(c) for c in cols]].to_numpy(float)
            y = train[rk(target)].to_numpy(float)
            beta = sb.fit_ols(X, y)
            cm.loc[cur, f"{name}|{target}"] = sb.predict_ols(
                beta, cm.loc[cur, [rk(c) for c in cols]].to_numpy(float))
    return cm


# --------------------------------------------------------------------------- #
def main():
    rows = []  # output records

    def emit(section, competitor, target, metric, series, n_override=None):
        if metric == "AUC":
            m, t, n = sb.nw_tstat(series.to_numpy() - 0.5, LAG)
            m += 0.5
        else:
            m, t, n = sb.nw_tstat(series.to_numpy(), LAG)
        rows.append({"section": section, "competitor": competitor,
                     "target": target, "metric": metric,
                     "mean": m, "nw_t": t,
                     "n_quarters": n_override if n_override is not None else n,
                     "posthoc": POSTHOC if not rows else ""})
        return m, t, (n_override if n_override is not None else n)

    logger.info("=== assembling size proxy (no model fit) ===")
    specs = universe_specs()
    panel = load_financial_panel(specs)
    all_dates = list(pd.period_range(sb.FEATURE_START, sb.PRED_END, freq="Q"))
    scored = list(pd.period_range(sb.PRED_START, sb.PRED_END, freq="Q"))
    size_df = size_proxy(panel, all_dates)

    data = pd.read_csv(PROBS_CSV)
    data["quarter"] = pd.PeriodIndex(data["quarter"], freq="Q")
    data = data.merge(size_df, on=["ticker", "quarter"], how="left")

    n_orac = int(data["A_orac"].notna().sum())
    n_orac_nosize = int((data["A_orac"].notna() & data["size"].isna()).sum())
    n_feas_no_orac = int((data["A_feas"].notna() & data["A_orac"].isna()).sum())
    logger.info("A_orac rows: %d (%d without a size proxy); "
                "A_feas-without-A_orac rows: %d",
                n_orac, n_orac_nosize, n_feas_no_orac)

    # ---- diagnostic: how confounded are A and size? ----
    logger.info("=== diagnostics: size <-> growth, A <-> size ===")
    emit("diag", "size", "t1_h4", "IC", ic_series(data, "size", "t1_h4", scored))
    emit("diag", "size", "t1_h8", "IC", ic_series(data, "size", "t1_h8", scored))
    emit("diag", "corr(A_orac,size)", "—", "rho",
         pairwise_ic(data, "A_orac", "size", scored))
    emit("diag", "corr(A_feas,size)", "—", "rho",
         pairwise_ic(data, "A_feas", "size", scored))

    # ---- Tier-1: raw vs size-residualized IC, paired names ----
    logger.info("=== Tier-1 size-deconfounding (A_orac, A_feas) ===")
    t1 = {}
    for feat in ("A_orac", "A_feas"):
        for tgt in ("t1_h4", "t1_h8"):
            raw, res = residualize_ic(data, feat, tgt, scored)
            mr, tr, nr = emit("tier1_raw", feat, tgt, "IC", raw)
            ms, ts, ns = emit("tier1_resid_size", feat, tgt, "IC", res)
            t1[(feat, tgt)] = (mr, tr, ms, ts, nr)

    # ---- Tier-2: F0 / F0s / F0s+A combiners on the paired intersection ----
    logger.info("=== Tier-2 F0s combiners (paired intersection) ===")
    cm = data.copy()
    cm["common_size"] = cm[INTERSECTION_COLS].notna().all(axis=1)
    cmr = cm[cm["common_size"]].copy()
    cmr = sb.quarter_ranks(cmr, INTERSECTION_COLS + ["t1_h4", "t1_h8"])
    n_inter = int(cmr["quarter"].isin(scored).sum())
    logger.info("intersection rows (scored): %d", n_inter)

    t2 = {}
    for tgt, h in (("t1_h4", 4), ("t1_h8", 8)):
        cmt = tier2_size(cmr, tgt, h, scored)
        ic = {name: ic_series(cmt, f"{name}|{tgt}", tgt, scored)
              for name in TIER2_SIZE}
        for name in TIER2_SIZE:
            emit("tier2", name, tgt, "IC", ic[name])
        # paired marginals over F0s
        for a_chan in ("F0s+A_feas", "F0s+A_orac"):
            diff = ic[a_chan] - ic["F0s"]
            md, td, nd = emit("tier2_marginal",
                              f"{a_chan} - F0s", tgt, "dIC", diff)
            t2[(a_chan, tgt)] = (md, td, nd)
        # size's own marginal over plain F0, for context
        emit("tier2_marginal", "F0s - F0", tgt, "dIC", ic["F0s"] - ic["F0"])
        if tgt == "t1_h4":
            t2["IC_F0"] = sb.nw_tstat(ic["F0"].to_numpy(), LAG)
            t2["IC_F0s"] = sb.nw_tstat(ic["F0s"].to_numpy(), LAG)

    board = pd.DataFrame(rows)
    board.to_csv(OUT_CSV, index=False)
    logger.info("wrote -> %s (%d rows)", OUT_CSV, len(board))

    report(t1, t2, n_orac, n_inter)
    return 0


def report(t1, t2, n_orac, n_inter):
    f4 = lambda v: f"{v:+.4f}" if pd.notna(v) else "—"
    f2 = lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
    print("\n" + "=" * 78)
    print("STEP 5b — POST-HOC SIZE-CONFOUND ROBUSTNESS  (NOT pre-registered)")
    print("=" * 78)
    print(f"NOTE: {POSTHOC}\n")

    mr, tr, ms, ts, nr = t1[("A_orac", "t1_h4")]
    md, td, nd = t2[("F0s+A_orac", "t1_h4")]
    mdf, tdf, ndf = t2[("F0s+A_feas", "t1_h4")]
    f0m, f0t, _ = t2["IC_F0"]
    f0sm, f0st, _ = t2["IC_F0s"]

    print("--- THE THREE NUMBERS (A_oracle, target = 4Q-fwd rev-YoY rank) ---")
    print(f"  (1) A_oracle IC  BEFORE size-deconfound : {f4(mr)}  (NW t={f2(tr)}, n={nr})")
    print(f"  (2) A_oracle IC  AFTER  size-deconfound : {f4(ms)}  (NW t={f2(ts)}, n={nr})")
    print(f"  (3) marginal IC(F0s+A_oracle) - IC(F0s) : {f4(md)}  (NW t={f2(td)}, n={nd})")
    print(f"      [context] IC(F0)={f4(f0m)} (t={f2(f0t)})  "
          f"IC(F0s)={f4(f0sm)} (t={f2(f0st)})  "
          f"feasible marginal={f4(mdf)} (t={f2(tdf)})")

    # honest one-line verdict, derived from the numbers
    shrink = (1 - abs(ms) / abs(mr)) * 100 if mr else float("nan")
    resid_sig = pd.notna(ts) and abs(ts) >= 2.0
    if resid_sig and ms < 0:
        verdict = ("the negative A_oracle relation SURVIVES within size "
                   f"(IC {f4(mr)}->{f4(ms)}, still NW t={f2(ts)}): not a pure "
                   "size artifact.")
    elif pd.notna(ms) and abs(ms) < abs(mr) and not resid_sig:
        verdict = (f"the negative A_oracle relation is LARGELY a size effect: "
                   f"residualizing on size shrinks |IC| by ~{shrink:.0f}% "
                   f"({f4(mr)}->{f4(ms)}) and the residual is no longer "
                   f"significant (NW t={f2(ts)}); the Tier-2 marginal over "
                   f"F0s is {f4(md)} (t={f2(td)}).")
    else:
        verdict = (f"mixed: residual IC {f4(ms)} (t={f2(ts)}) vs raw {f4(mr)}; "
                   f"Tier-2 marginal over F0s {f4(md)} (t={f2(td)}).")
    print("\n--- VERDICT (size effect, or within-size?) ---")
    print(f"  {verdict}")
    print(f"\n(intersection rows scored: {n_inter}; A_orac rows: {n_orac})")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
