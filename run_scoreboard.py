"""QVM-V2 STEP 5b entry point — the pre-registered out-of-sample scoreboard.

All design constants live in qvm.analysis.scoreboard (the pre-registration);
this driver only orchestrates: data assembly -> expanding-window HMM features
(multiprocess, per-ticker resume cache) -> Cw nested w-selection -> Tier 1/2
metrics -> robustness + appendices -> CSVs, figures, console report.

Usage:
  python run_scoreboard.py             # full run (uses _cache/scoreboard/)
  python run_scoreboard.py --fresh     # ignore per-ticker fit caches
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import argparse
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from qvm import config
from qvm.universe_registry import apply_reattributions, universe_specs
from qvm.analysis import scoreboard as sb

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("qvm.scoreboard")

UNIVERSE_RAW_CSV = os.path.join(config.CACHE_DIR, "universe_raw_patents.csv")
FIT_CACHE_DIR = os.path.join(config.CACHE_DIR, "scoreboard")
SCOREBOARD_CSV = os.path.join(config.OUTPUT_DIR, "scoreboard.csv")
IC_TS_CSV = os.path.join(config.OUTPUT_DIR, "ic_timeseries.csv")
PROBS_CSV = os.path.join(config.OUTPUT_DIR, "expanding_regime_probs.csv")
WSEL_CSV = os.path.join(config.OUTPUT_DIR, "w_selection.csv")
ATTRITION_CSV = os.path.join(config.OUTPUT_DIR, "attrition.csv")
MA_CSV = os.path.join(config.OUTPUT_DIR, "ma_events.csv")

TIER1 = ["D1", "D2", "A_feas", "A_orac", "B", "C", "Cw"]
TIER2_SPECS = {                      # combiner name -> feature columns
    "F0": ["f_yoy", "f_gm", "f_dgm"],
    "F0+A": ["f_yoy", "f_gm", "f_dgm", "A_feas"],
    "F0+B": ["f_yoy", "f_gm", "f_dgm", "B"],
    "F0+A+B": ["f_yoy", "f_gm", "f_dgm", "A_feas", "B"],
    "F0+Cw": ["f_yoy", "f_gm", "f_dgm", "Cw"],
}
COMMON_COLS = ["f_yoy", "f_gm", "f_dgm", "A_feas", "B", "Cw"]  # intersection


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def load_patent_counts(specs) -> dict[str, pd.Series]:
    """Post-attribution quarterly application counts per ticker."""
    from run_universe import assign_rows
    df = pd.read_csv(UNIVERSE_RAW_CSV)
    df = assign_rows(df, specs)
    df, _ = apply_reattributions(df)
    d = df[df["_ticker"].notna()].copy()
    filing = pd.to_numeric(d["filing_date"], errors="coerce")
    d["_filing"] = pd.to_datetime(
        filing.where(filing > 0).astype("Int64").astype(str),
        format="%Y%m%d", errors="coerce")
    apps = (d.groupby(["_ticker", "application_number"])["_filing"].min()
            .dropna().dt.to_period("Q"))
    counts = apps.groupby(["_ticker", apps]).size()
    counts.index.names = ["ticker", "quarter"]
    return {t: counts.loc[t].sort_index()
            for t in counts.index.get_level_values(0).unique()}


def load_financial_panel(specs) -> pd.DataFrame:
    from qvm.data.sec_edgar_provider import SECEdgarProvider
    provider = SECEdgarProvider()
    fins = {}
    for spec in specs:
        try:
            fins[spec["ticker"]] = provider.get_financials_for_ciks(
                spec["ticker"], spec["ciks"])
        except Exception as e:
            logger.warning("[%s] financials unavailable: %s", spec["ticker"], e)
    panel = sb.build_financial_panel(fins)
    # membership windows (QRVO starts at the merger closing — see registry)
    for spec in specs:
        ms = spec.get("member_start")
        if ms:
            panel = panel[~((panel["ticker"] == spec["ticker"])
                            & (panel["quarter"] < pd.Period(ms, "Q")))]
    return panel.reset_index(drop=True)


def run_workers(specs, panel, counts, alive, fresh: bool) -> pd.DataFrame:
    os.makedirs(FIT_CACHE_DIR, exist_ok=True)
    jobs, results = [], []
    for spec in specs:
        ticker = spec["ticker"]
        path = os.path.join(FIT_CACHE_DIR, f"{ticker}.pkl")
        dates = sorted(alive[alive["ticker"] == ticker]["quarter"])
        if not dates:
            continue
        if not fresh and os.path.exists(path):
            with open(path, "rb") as fh:
                results.append(pickle.load(fh))
            continue
        fin = panel[panel["ticker"] == ticker][
            ["quarter", "gm", "gm_filed", "yoy", "yoy_filed"]]
        jobs.append({"ticker": ticker,
                     "counts": counts.get(ticker, pd.Series(dtype=float)),
                     "fin": fin, "dates": dates})
    if jobs:
        logger.info("expanding-window fits: %d tickers to run "
                    "(%d cached)", len(jobs), len(results))
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() - 2)) as ex:
            for out in ex.map(sb.company_worker, jobs):
                with open(os.path.join(FIT_CACHE_DIR,
                                       f"{out['ticker']}.pkl"), "wb") as fh:
                    pickle.dump(out, fh)
                results.append(out)
                logger.info("  [%s] done (%.0fs elapsed)",
                            out["ticker"], time.time() - t0)
    rows = [r for res in results for r in res["probs"]]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cw nested w-selection (fully causal mini-backtest, driver-side)
# --------------------------------------------------------------------------- #
def select_w(feat: pd.DataFrame, tf: pd.DataFrame,
             all_dates: list[pd.Period]) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = feat.merge(tf[["ticker", "quarter", "t1_h4", "t1_h4_filed"]],
                   on=["ticker", "quarter"], how="left")
    wcols = [f"Cw_{w}" for w in sb.W_GRID]
    feat = feat.copy()
    feat["Cw"] = np.nan
    sel_rows = []
    for t in all_dates:
        cutoff = str(t.end_time.date())
        pool = m[(m["quarter"] <= t - 4) & m["t1_h4"].notna()
                 & m["t1_h4_filed"].notna() & (m["t1_h4_filed"].fillna("9999") <= cutoff)]
        best_w, best_ic, n_q = 1.0, np.nan, 0
        if not pool.empty:
            for w in sb.W_GRID:           # ascending: ties keep the smaller w
                ics = [sb.spearman_ic(g[f"Cw_{w}"].to_numpy(),
                                      g["t1_h4"].to_numpy())
                       for _, g in pool.groupby("quarter")]
                ics = [x for x in ics if not np.isnan(x)]
                if len(ics) >= 4:
                    mean_ic = float(np.mean(ics))
                    if np.isnan(best_ic) or mean_ic > best_ic:
                        best_w, best_ic, n_q = w, mean_ic, len(ics)
        fallback = bool(np.isnan(best_ic))
        sel_rows.append({"quarter": t, "w_selected": best_w,
                         "inner_ic": best_ic, "n_inner_quarters": n_q,
                         "fallback_naive": fallback})
        sel = feat["quarter"] == t
        feat.loc[sel, "Cw"] = feat.loc[sel, f"Cw_{best_w}"]
    return feat, pd.DataFrame(sel_rows)


# --------------------------------------------------------------------------- #
# metrics over the scored window
# --------------------------------------------------------------------------- #
def ic_series(df: pd.DataFrame, feature: str, target: str,
              dates: list[pd.Period], mask: pd.Series | None = None,
              metric=sb.spearman_ic) -> pd.Series:
    d = df if mask is None else df[mask]
    out = {}
    for t in dates:
        g = d[d["quarter"] == t]
        out[t] = metric(g[feature].to_numpy(dtype=float),
                        g[target].to_numpy(dtype=float))
    return pd.Series(out)

def summarize(ics: pd.Series, lag: int) -> dict:
    mean, t, n = sb.nw_tstat(ics.to_numpy(), lag)
    return {"mean": mean, "nw_t": t, "n_quarters": n}


def tier2_predictions(cm: pd.DataFrame, scored: list[pd.Period],
                      horizon: int, target: str) -> pd.DataFrame:
    """Expanding-window OLS combiners on rank-transformed common rows.
    Returns cm + one score column per combiner ('{name}|{target}')."""
    cm = cm.copy()
    rk = lambda c: c + "_rk"
    for name in TIER2_SPECS:
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
        for name, cols in TIER2_SPECS.items():
            X = train[[rk(c) for c in cols]].to_numpy(float)
            y = train[rk(target)].to_numpy(float)
            beta = sb.fit_ols(X, y)
            cm.loc[cur, f"{name}|{target}"] = sb.predict_ols(
                beta, cm.loc[cur, [rk(c) for c in cols]].to_numpy(float))
    return cm


def decision_fusion(cm: pd.DataFrame, scored: list[pd.Period]) -> pd.DataFrame:
    """Track fusion: logistic on [P_A_feas, P_B], top-quintile target."""
    cm = cm.copy()
    cm["DF"] = np.nan
    for t in scored:
        cutoff = str(t.end_time.date())
        train = cm[(cm["quarter"] <= t - 4) & cm["t2"].notna()
                   & cm["t1_h4_filed"].notna() & (cm["t1_h4_filed"].fillna("9999") <= cutoff)]
        if len(train) < sb.MIN_TRAIN_ROWS:
            continue
        X = train[["A_feas", "B"]].to_numpy(float)
        b = sb.fit_logistic(X, train["t2"].to_numpy(float))
        cur = cm["quarter"] == t
        cm.loc[cur, "DF"] = sb.predict_logistic(
            b, cm.loc[cur, ["A_feas", "B"]].to_numpy(float))
    return cm


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="QVM-V2 Step 5b scoreboard.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore per-ticker fit caches")
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    specs = universe_specs()
    spec_by = {s["ticker"]: s for s in specs}
    all_dates = list(pd.period_range(sb.FEATURE_START, sb.PRED_END, freq="Q"))
    scored = list(pd.period_range(sb.PRED_START, sb.PRED_END, freq="Q"))

    logger.info("=== data assembly ===")
    counts = load_patent_counts(specs)
    panel = load_financial_panel(specs)
    tf = sb.target_frame(panel, all_dates)

    base = pd.concat([sb.baseline_features(panel, t) for t in all_dates],
                     ignore_index=True)
    alive = base[base["f_yoy"].notna()][["ticker", "quarter"]]
    logger.info("panel: %d tickers, %d alive company-quarters "
                "(%s..%s feature dates)", base["ticker"].nunique(),
                len(alive), sb.FEATURE_START, sb.PRED_END)

    logger.info("=== expanding-window HMM features ===")
    probs = run_workers(specs, panel, counts, alive, args.fresh)

    feat = base.merge(probs, on=["ticker", "quarter"], how="left")
    feat = feat.rename(columns={"f_yoy_quintile": "D2"})
    feat["D1"] = feat["f_yoy"]
    feat, wsel = select_w(feat, tf, all_dates)
    wsel.to_csv(WSEL_CSV, index=False)

    data = feat.merge(tf, on=["ticker", "quarter"], how="left")
    data["caveat"] = ""
    data.loc[data.index[:1], "caveat"] = sb.CAVEAT
    data.to_csv(PROBS_CSV, index=False)
    logger.info("wrote expanding regime-prob/feature cache -> %s (%d rows)",
                PROBS_CSV, len(data))

    is_scored = data["quarter"].isin(scored)

    # ---- attrition accounting (scored window, h=4) ----
    att_rows = []
    for t in scored:
        g = data[(data["quarter"] == t) & data["f_yoy"].notna()]
        miss = g[g["t1_h4"].isna()]
        n_delist = 0
        for ticker in miss["ticker"]:
            d = spec_by[ticker].get("delisted", "")
            if d:
                dq = pd.Period(d.split()[0], "Q")
                if t < dq <= t + 4 or dq <= t:
                    n_delist += 1
        att_rows.append({"quarter": t, "n_alive": len(g),
                         "n_with_target_h4": int(g["t1_h4"].notna().sum()),
                         "n_dropped_delisted": n_delist,
                         "n_dropped_data_gap": len(miss) - n_delist})
    att = pd.DataFrame(att_rows)
    att.to_csv(ATTRITION_CSV, index=False)

    # ---- Tier 2 prep: intersection rows, rank transforms ----
    cm = data.copy()
    cm["common"] = cm[COMMON_COLS].notna().all(axis=1)
    cmr = cm[cm["common"]].copy()
    cmr = sb.quarter_ranks(cmr, COMMON_COLS + ["t1_h4", "t1_h8", "t3"])
    cmr["t3_filed"] = cmr["t1_h4_filed"]   # gm files with the same 10-Q/10-K
    for target, h in (("t1_h4", 4), ("t1_h8", 8), ("t3", 4)):
        cmr = tier2_predictions(cmr, scored, h, target)
    cmr = decision_fusion(cmr, scored)

    # ---- M&A ±2Q robustness mask ----
    ma = pd.read_csv(MA_CSV)
    ma["q"] = pd.PeriodIndex(pd.to_datetime(ma["date"]), freq="Q")
    excl = set()
    for r in ma.itertuples():
        for k in range(-2, 3):
            excl.add((r.ticker, r.q + k))
    ma_mask_full = ~data.apply(
        lambda r: (r["ticker"], r["quarter"]) in excl, axis=1)
    ma_mask_cmr = ~cmr.apply(
        lambda r: (r["ticker"], r["quarter"]) in excl, axis=1)

    # ---- scoreboard ----
    rows, ts_rows = [], []
    def add(competitor, tier, df, col, target, lag, metric=sb.spearman_ic,
            label="", mask=None, keep_ts=True):
        ics = ic_series(df, col, target, scored, mask=mask, metric=metric)
        if metric is sb.auc_score:
            s = summarize(ics - 0.5, lag)   # AUC's null is 0.5, not 0
            s["mean"] += 0.5
        else:
            s = summarize(ics, lag)
        rows.append({"competitor": competitor, "tier": tier,
                     "target": label or target, "metric":
                     "AUC" if metric is sb.auc_score else "IC", **s,
                     "caveat": sb.CAVEAT if not rows else ""})
        if keep_ts:
            for q, v in ics.items():
                ts_rows.append({"competitor": competitor,
                                "target": label or target,
                                "quarter": str(q), "value": v})

    for c in TIER1:
        add(c, 1, data, c, "t1_h4", 4)
        add(c, 1, data, c, "t1_h8", 8)
        add(c, 1, data, c, "t2", 4, metric=sb.auc_score)
        add(c, 1, data, c, "t3", 4)
        add(c, 1, data, c, "t1_h4", 4, label="t1_h4_ma_excl",
            mask=ma_mask_full, keep_ts=False)
        add(c, 1, data, c, "t1_h4_latest", 4, label="t1_h4_latest(app)",
            keep_ts=False)
        add(c, 1, data, c, "ni_dir", 4, metric=sb.auc_score,
            label="ni_dir(app)", keep_ts=False)
    for name in TIER2_SPECS:
        add(name, 2, cmr, f"{name}|t1_h4", "t1_h4", 4)
        add(name, 2, cmr, f"{name}|t1_h8", "t1_h8", 8)
        add(name, 2, cmr, f"{name}|t1_h4", "t2", 4, metric=sb.auc_score)
        add(name, 2, cmr, f"{name}|t3", "t3", 4)
        add(name, 2, cmr, f"{name}|t1_h4", "t1_h4", 4,
            label="t1_h4_ma_excl", mask=ma_mask_cmr, keep_ts=False)
    add("DF", 2, cmr, "DF", "t1_h4", 4)
    add("DF", 2, cmr, "DF", "t2", 4, metric=sb.auc_score)

    board = pd.DataFrame(rows)
    board.to_csv(SCOREBOARD_CSV, index=False)
    ts = pd.DataFrame(ts_rows)
    ts.to_csv(IC_TS_CSV, index=False)
    logger.info("wrote scoreboard -> %s | IC time series -> %s",
                SCOREBOARD_CSV, IC_TS_CSV)

    # ---- headline: marginal patent contribution (paired per-quarter diffs) --
    headline = {}
    for a, b, name in (("F0+A", "F0", "F0+A - F0"),
                       ("F0+A+B", "F0", "F0+A+B - F0"),
                       ("F0+Cw", "F0", "F0+Cw - F0")):
        ia = ic_series(cmr, f"{a}|t1_h4", "t1_h4", scored)
        ib = ic_series(cmr, f"{b}|t1_h4", "t1_h4", scored)
        m, t, n = sb.nw_tstat((ia - ib).to_numpy(), 4)
        headline[name] = (m, t, n)
    io = ic_series(data, "A_orac", "t1_h4", scored)
    if_ = ic_series(data, "A_feas", "t1_h4", scored)
    m, t, n = sb.nw_tstat((io - if_).to_numpy(), 4)
    headline["A_oracle - A_feasible"] = (m, t, n)
    ic_c = ic_series(data, "C", "t1_h4", scored)
    ic_cw = ic_series(data, "Cw", "t1_h4", scored)
    m, t, n = sb.nw_tstat((ic_cw - ic_c).to_numpy(), 4)
    headline["Cw - C (naive)"] = (m, t, n)

    make_figures(ts, board, headline, wsel)
    report(board, headline, att, wsel, data, time.time() - t_start)
    return 0


def make_figures(ts, board, headline, wsel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    ext = os.environ.get("QVM_FIG_EXT", "png")  # "pdf" -> vector for the paper

    key = ["D1", "A_feas", "B", "C", "Cw", "F0", "F0+A+B"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c in key:
        sub = ts[(ts["competitor"] == c) & (ts["target"] == "t1_h4")]
        if sub.empty:
            continue
        x = pd.PeriodIndex(sub["quarter"], freq="Q").to_timestamp()
        ax.plot(x, pd.Series(sub["value"].to_numpy()).rolling(4, 2).mean(),
                label=c, lw=1.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Quarterly Spearman IC (4Q rolling mean), "
                 "target: 4Q-fwd revenue YoY rank")
    ax.set_ylabel("IC")
    ax.legend(ncol=4, fontsize=11)
    fig.tight_layout()
    p7 = os.path.join(config.OUTPUT_DIR, f"fig7_ic_timeseries.{ext}")
    fig.savefig(p7, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(headline)
    vals = [headline[n][0] for n in names]
    tstats = [headline[n][1] for n in names]
    bars = ax.bar(range(len(names)), vals, color=["#4c72b0"] * 3
                  + ["#dd8452", "#55a868"])
    for i, (v, tt) in enumerate(zip(vals, tstats)):
        ax.text(i, v + (0.002 if v >= 0 else -0.004),
                f"{v:+.3f}\n(t={tt:.2f})", ha="center", fontsize=11)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, fontsize=11)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Headline: marginal IC contributions "
                 "(paired per-quarter diffs, NW t)")
    ax.set_ylabel("Δ mean IC (T1, h=4)")
    fig.tight_layout()
    p8 = os.path.join(config.OUTPUT_DIR, f"fig8_headline.{ext}")
    fig.savefig(p8, dpi=220)
    plt.close(fig)
    logger.info("wrote figures -> %s | %s", p7, p8)


def report(board, headline, att, wsel, data, elapsed):
    pd.set_option("display.width", 160)
    print("\n" + "=" * 78)
    print("QVM-V2 STEP 5b — OUT-OF-SAMPLE PREDICTION SCOREBOARD (pre-registered)")
    print("=" * 78)
    print(f"\nNOTE: {sb.CAVEAT}")
    fmt = board.copy()
    for c in ("mean", "nw_t"):
        fmt[c] = fmt[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—")
    print("\n--- Tier 1 (single features) ---")
    print(fmt[(fmt["tier"] == 1)
              & ~fmt["target"].str.contains("app|ma_excl")]
          .pivot_table(index="competitor", columns=["target", "metric"],
                       values=["mean", "nw_t"], aggfunc="first")
          .to_string())
    print("\n--- Tier 2 (combiners, common rows) ---")
    print(fmt[(fmt["tier"] == 2)
              & ~fmt["target"].str.contains("app|ma_excl")]
          .pivot_table(index="competitor", columns=["target", "metric"],
                       values=["mean", "nw_t"], aggfunc="first")
          .to_string())
    print("\n--- HEADLINE: marginal contributions (paired diffs, NW t, n) ---")
    for k, (m, t, n) in headline.items():
        print(f"  {k:24s} ΔIC = {m:+.4f}   t = {t:+.2f}   n = {n}")
    print("\n--- M&A ±2Q robustness (T1 h4) ---")
    print(fmt[fmt["target"] == "t1_h4_ma_excl"][
        ["competitor", "tier", "mean", "nw_t", "n_quarters"]]
        .to_string(index=False))
    print("\n--- appendices: latest-series targets + NI direction ---")
    print(fmt[fmt["target"].str.contains("app")][
        ["competitor", "target", "metric", "mean", "nw_t", "n_quarters"]]
        .to_string(index=False))
    print("\n--- attrition (scored window) ---")
    print(f"  alive company-quarters: {att['n_alive'].sum()}, "
          f"with h4 target: {att['n_with_target_h4'].sum()}, "
          f"dropped (delisted in horizon): {att['n_dropped_delisted'].sum()}, "
          f"dropped (data gap): {att['n_dropped_data_gap'].sum()}")
    print("\n--- Cw selected-w distribution (scored windows) ---")
    sc = wsel[wsel["quarter"].isin(att["quarter"])]
    print(sc["w_selected"].value_counts().sort_index().to_string())
    print(f"  fallback (no inner pool): {int(sc['fallback_naive'].sum())}")
    n_af = data[data['quarter'].isin(att['quarter'])]['A_feas'].notna().sum()
    print(f"\nfeature coverage (scored): A_feas {n_af} rows; "
          f"elapsed {elapsed/60:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
