"""STEP 5b — the out-of-sample prediction scoreboard (PRE-REGISTERED).

Everything tunable is locked HERE, before any run; changing a constant after
results exist requires a documented deviation note. The scoreboard answers
"which fusion?" by out-of-sample predictive value, with the information-set
discipline Steps 3-4 deferred:

  * FEASIBLE patent channel: filing-dated counts are CUT at t-6 quarters
    (18-month publication secrecy: at time t, quarters <= t-6 are ~fully
    visible). ORACLE keeps counts through t (knowable only in hindsight);
    the oracle-feasible IC gap is question (c)'s number.
  * Financial channel: an observation enters only when its filed date
    (knowable_at) <= the prediction cutoff (last day of quarter t).
  * EXPANDING-WINDOW REFIT: at every feature date, HMM parameters are re-fit
    on data <= t only (resolves the Step 3-4 parameter-leak caveat).
    Cost control: warm start from the previous window's solution + 5 restarts
    (2 random), per the pre-registration; the first window in a chain uses
    FIRST_WINDOW_RESTARTS random restarts.

PANEL: prediction dates 2015Q1..2022Q4 (scored). Features are ALSO produced
for 2011Q1..2014Q4 — never scored, only used as training rows for the Tier-2
combiners and the Cw inner validation (a combiner trained at t may only see
feature values computed with windows <= s and targets realized by t).

TARGETS (as-filed series = a consistent information regime; latest-series
robustness is an appendix):
  T1  revenue YoY growth at t+4 (= rev[t+4]/rev[t] - 1), in-universe rank;
      metric = per-quarter cross-sectional Spearman IC, mean + Newey-West
      t-stat (lag = horizon). Secondary horizon h=8 (rev[t+8]/rev[t+4] - 1).
  T2  top-quintile (>=80th pct) binary of the T1 cross-section; metric AUC.
      Appendix: earnings-change direction (NetIncomeLoss at t+4 vs t).
  T3  exploratory: gross-margin change gm[t+4]-gm[t], universe-median-
      adjusted per quarter.
Company-quarters whose target never realizes (delisted within the horizon)
are DROPPED AND COUNTED — attrition is reported, not hidden (acquisitions
follow good performance; the censoring is not random).

COMPETITORS — Tier 1 (single features):
  D1 current revenue-YoY rank (momentum baseline)   D2 its quintile
  A  patent-only HMM filtered P(high)  [feasible and oracle rows]
  B  financial-only HMM filtered P(high)
  C  joint naive HMM filtered P(high)  [feasible patent channel]
  Cw tempered joint, NB weight w in {0,.05,.1,.25,.5,1}; w picked PER WINDOW
     by NESTED inner validation: a fully causal mini-backtest — for every
     candidate w, the cross-sectional IC of the window-s feature (p_high at s
     from the model fit on data <= s) against targets REALIZED (filed) by t,
     over s in [2011Q1, t-4]. No test-period data enters the choice. Ties ->
     smaller w; empty pool -> w=1 = naive, counted.
Tier 2 (combiners, expanding-window fit on pooled training rows; features
and target rank-transformed to [0,1] within each quarter; OLS):
  F0 = [rev YoY, margin level, margin change];  F0+A, F0+B, F0+A+B, F0+Cw.
  Headline: IC(F0+A)-IC(F0) and IC(F0+A+B)-IC(F0) = the patent-regime
  features' MARGINAL contribution. Tier-2 rows are restricted to the
  INTERSECTION where F0, A, B and Cw all exist, so the comparison is
  apples-to-apples. DF = decision fusion: logistic on [P_A, P_B].

STATE ORDERING (locked): A and C order by the patent-count mean (Step 3-4
convention); B and Cw order by the gross-margin mean — Cw's patent channel
is down-weighted, so a fully-weighted channel must define "high".

ROBUSTNESS (pre-registered): main analysis keeps M&A windows; the robustness
row recomputes T1-h4 ICs excluding company-quarters within +-2 quarters of
that company's ma_events entries.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..models.fusion_hmm import Channel, fit_fusion_hmm

logger = logging.getLogger(__name__)

# ----------------------------- locked constants ----------------------------- #
PRED_START, PRED_END = "2015Q1", "2022Q4"   # scored prediction quarters
FEATURE_START = "2011Q1"                    # training-row feature dates begin
W_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
FEASIBLE_LAG_Q = 6                          # 18-month publication secrecy
PATENT_GRID_START = "2005Q1"                # Step-3 convention
FIN_GRID_START = "2009Q1"                   # XBRL start
MIN_PATENT_OBS = 20                         # gate: observed patent quarters
MIN_PATENT_NONZERO = 8                      # gate: nonzero patent quarters
MIN_FIN_OBS = 16                            # gate: observed financial quarters
HORIZONS = (4, 8)
TOP_QUINTILE_PCT = 0.80
MIN_CROSS_SECTION = 8                       # min names for a quarterly IC/AUC
MIN_TRAIN_ROWS = 100                        # min pooled rows for a combiner
FIRST_WINDOW_RESTARTS = 10
WARM_RESTARTS = 5                           # warm exact + 2 jitter + 2 random
STALENESS_Q = 3                             # newest financial obs may lag <= 3Q

CAVEAT = ("Step 5b pre-registered scoreboard: feasible patent channel cut at "
          "t-6Q; financials gated on filed<=t; expanding-window HMM refits "
          "(warm start + 5 restarts); as-filed targets; attrition counted")


def _seed(*parts) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


# --------------------------------------------------------------------------- #
# financial panel (point-in-time columns) — pure, testable
# --------------------------------------------------------------------------- #
def _per_quarter_obs(fin, concept: str) -> dict:
    """Calendar-quarter map, first-wins on fiscal collisions (channels.py rule)."""
    from .channels import calendar_quarter
    out = {}
    for obs in fin.quarterly(concept):
        q = calendar_quarter(obs.end)
        if q not in out:
            out[q] = obs
    return out


def build_financial_panel(fins: dict) -> pd.DataFrame:
    """One row per (ticker, quarter): as-filed values + knowable_at dates +
    latest values (target robustness only). Quarter range = each company's
    own filed range; downstream masking handles point-in-time."""
    rows = []
    for ticker, fin in fins.items():
        rev = _per_quarter_obs(fin, "revenue")
        cor = _per_quarter_obs(fin, "cost_of_revenue")
        gp = _per_quarter_obs(fin, "gross_profit")
        ni = _per_quarter_obs(fin, "net_income")
        if not rev:
            continue
        for q in pd.period_range(min(rev), max(rev), freq="Q"):
            r = rev.get(q)
            row = {"ticker": ticker, "quarter": q,
                   "rev": np.nan, "rev_filed": None, "rev_latest": np.nan,
                   "gm": np.nan, "gm_filed": None, "gm_latest": np.nan,
                   "ni": np.nan, "ni_filed": None}
            if r is not None and r.value:
                row.update(rev=r.value, rev_filed=r.filed,
                           rev_latest=r.value_latest)
                g, co = gp.get(q), cor.get(q)
                if g is not None:
                    row.update(gm=g.value / r.value,
                               gm_filed=max(r.filed, g.filed))
                    if r.value_latest:
                        row["gm_latest"] = g.value_latest / r.value_latest
                elif co is not None:
                    row.update(gm=(r.value - co.value) / r.value,
                               gm_filed=max(r.filed, co.filed))
                    if r.value_latest:
                        row["gm_latest"] = ((r.value_latest - co.value_latest)
                                            / r.value_latest)
            n = ni.get(q)
            if n is not None:
                row.update(ni=n.value, ni_filed=n.filed)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["ticker", "quarter"])
    # YoY growth + margin change, with knowable_at = max of the two filings
    g = df.groupby("ticker", sort=False)
    prev = g[["rev", "rev_filed", "gm", "gm_filed", "rev_latest"]].shift(4)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["yoy"] = np.where(prev["rev"] > 0, df["rev"] / prev["rev"] - 1, np.nan)
        df["yoy_latest"] = np.where(prev["rev_latest"] > 0,
                                    df["rev_latest"] / prev["rev_latest"] - 1,
                                    np.nan)
    def _max_filed(a: pd.Series, b: pd.Series, valid: pd.Series) -> pd.Series:
        m = a.where(a.fillna("") >= b.fillna(""), b)  # ISO strings: lexicographic
        return m.where(valid & a.notna() & b.notna(), None)
    df["yoy_filed"] = _max_filed(df["rev_filed"], prev["rev_filed"],
                                 df["yoy"].notna())
    df["dgm"] = df["gm"] - prev["gm"]
    df["dgm_filed"] = _max_filed(df["gm_filed"], prev["gm_filed"],
                                 df["dgm"].notna())
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# targets — pure
# --------------------------------------------------------------------------- #
def target_frame(panel: pd.DataFrame, quarters: list[pd.Period]) -> pd.DataFrame:
    """Realized targets per (ticker, prediction quarter): T1 h4/h8 (as-filed +
    latest), T3 margin change (median-adjusted later), NI direction."""
    p = panel.set_index(["ticker", "quarter"])
    qset = set(quarters)
    rows = []
    for (ticker, q), r in p.iterrows():
        if q not in qset:
            continue
        get = lambda col, qq: p[col].get((ticker, qq), np.nan)
        ni_fwd = get("ni", q + 4)
        rows.append({
            "ticker": ticker, "quarter": q,
            "t1_h4": get("yoy", q + 4),
            "t1_h4_filed": p["yoy_filed"].get((ticker, q + 4), None),
            "t1_h8": get("yoy", q + 8),
            "t1_h8_filed": p["yoy_filed"].get((ticker, q + 8), None),
            "t1_h4_latest": get("yoy_latest", q + 4),
            "t3_raw": get("gm", q + 4) - (r["gm"] if pd.notna(r["gm"])
                                          else np.nan),
            "ni_dir": (np.nan if pd.isna(r["ni"]) or pd.isna(ni_fwd)
                       else float(ni_fwd > r["ni"])),
        })
    tf = pd.DataFrame(rows)
    # T3: universe(=sector)-median-adjusted per quarter
    med = tf.groupby("quarter")["t3_raw"].transform("median")
    tf["t3"] = tf["t3_raw"] - med
    # T2: top-quintile binary of the T1-h4 cross-section per quarter
    def quintile_flag(s):
        if s.notna().sum() < MIN_CROSS_SECTION:
            return pd.Series(np.nan, index=s.index)
        thr = s.quantile(TOP_QUINTILE_PCT)
        return (s >= thr).where(s.notna()).astype(float)
    tf["t2"] = tf.groupby("quarter")["t1_h4"].transform(quintile_flag)
    return tf


# --------------------------------------------------------------------------- #
# point-in-time baseline features — pure
# --------------------------------------------------------------------------- #
def baseline_features(panel: pd.DataFrame, t: pd.Period) -> pd.DataFrame:
    """D1/F0 at cutoff = last day of quarter t: newest observation per concept
    with filed <= cutoff and quarter within [t-STALENESS_Q, t]."""
    cutoff = str(t.end_time.date())
    sub = panel[(panel["quarter"] <= t) & (panel["quarter"] >= t - STALENESS_Q)]
    rows = []
    for ticker, g in sub.groupby("ticker", sort=False):
        def newest(col, filed_col):
            k = g[(g[col].notna()) & (g[filed_col].notna())
                  & (g[filed_col].fillna("9999") <= cutoff)]
            return float(k.sort_values("quarter")[col].iloc[-1]) if len(k) else np.nan
        rows.append({"ticker": ticker, "quarter": t,
                     "f_yoy": newest("yoy", "yoy_filed"),
                     "f_gm": newest("gm", "gm_filed"),
                     "f_dgm": newest("dgm", "dgm_filed")})
    df = pd.DataFrame(rows)
    yoy = df["f_yoy"]
    if yoy.notna().sum() >= MIN_CROSS_SECTION:
        df["f_yoy_quintile"] = pd.qcut(yoy.rank(method="first"), 5,
                                       labels=False, duplicates="drop") + 1.0
    else:
        df["f_yoy_quintile"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# expanding-window HMM worker (one ticker, all feature dates) — subprocess
# --------------------------------------------------------------------------- #
def _grid_series(counts: pd.Series, grid: pd.PeriodIndex,
                 first: pd.Period, last_obs: pd.Period) -> np.ndarray:
    """Counts on the grid: observed (zeros real) in [first, last_obs], NaN
    outside (before entity existence / after the feasibility cut)."""
    y = np.full(len(grid), np.nan)
    obs = (grid >= first) & (grid <= last_obs)
    y[obs] = counts.reindex(grid[obs]).fillna(0.0).to_numpy()
    return y


def company_worker(args: dict) -> dict:
    """All expanding-window fits for one ticker: per feature date and model,
    the filtered P(high) at t (for Cw: one column per candidate w; the
    per-window w choice is a cross-sectional decision made by the driver)."""
    ticker = args["ticker"]
    counts = args["counts"]            # pd.Series indexed by Period
    fin = args["fin"]                  # DataFrame quarter/gm/gm_filed/yoy/yoy_filed
    dates = args["dates"]              # list[pd.Period] feature dates (alive only)
    kw = dict(n_states=2, max_iter=500, tol=1e-6)

    pat_first = (max(counts.index.min(), pd.Period(PATENT_GRID_START, "Q"))
                 if len(counts) else None)
    fin_first = (max(fin["quarter"].min(), pd.Period(FIN_GRID_START, "Q"))
                 if len(fin) else None)

    warm: dict[str, dict] = {}
    probs = []
    for t in dates:
        cutoff = str(t.end_time.date())
        f = fin[(fin["quarter"] >= (fin_first or t)) & (fin["quarter"] <= t)]

        def fin_channels(grid):
            gm = np.full(len(grid), np.nan)
            yoy = np.full(len(grid), np.nan)
            pos = {q: i for i, q in enumerate(grid)}
            for r in f.itertuples():
                i = pos.get(r.quarter)
                if i is None:
                    continue
                if (pd.notna(r.gm) and isinstance(r.gm_filed, str)
                        and r.gm_filed <= cutoff):
                    gm[i] = r.gm
                if (pd.notna(r.yoy) and isinstance(r.yoy_filed, str)
                        and r.yoy_filed <= cutoff):
                    yoy[i] = r.yoy
            return gm, yoy

        def fit(key, channels, order):
            seed = _seed(ticker, key, t)
            if key in warm:
                fitres = fit_fusion_hmm(channels, order_channel=order,
                                        n_restarts=WARM_RESTARTS, seed=seed,
                                        warm_start=warm[key], **kw)
            else:
                fitres = fit_fusion_hmm(channels, order_channel=order,
                                        n_restarts=FIRST_WINDOW_RESTARTS,
                                        seed=seed, **kw)
            warm[key] = fitres.warm_start_dict()
            return fitres

        # ---- patent gates & grids ----
        res_t = {"ticker": ticker, "quarter": t}
        pat_ok_feas = pat_ok_orac = False
        if pat_first is not None and pat_first <= t:
            pgrid = pd.period_range(pat_first, t, freq="Q")
            y_orac = _grid_series(counts, pgrid, pat_first, t)
            y_feas = _grid_series(counts, pgrid, pat_first, t - FEASIBLE_LAG_Q)

            def gate(y):
                v = y[~np.isnan(y)]
                return (len(v) >= MIN_PATENT_OBS
                        and (v > 0).sum() >= MIN_PATENT_NONZERO
                        and v.var() > 0)
            pat_ok_feas, pat_ok_orac = gate(y_feas), gate(y_orac)
            if pat_ok_feas:
                ft = fit("A_feas", [Channel("patent_count", "nb", y_feas)],
                         "patent_count")
                res_t["A_feas"] = float(ft.filtered[-1, 1])
            if pat_ok_orac:
                ft = fit("A_orac", [Channel("patent_count", "nb", y_orac)],
                         "patent_count")
                res_t["A_orac"] = float(ft.filtered[-1, 1])

        # ---- financial-only ----
        fin_ok = False
        if fin_first is not None and fin_first <= t:
            fgrid = pd.period_range(fin_first, t, freq="Q")
            gm, yoy = fin_channels(fgrid)
            n_fin = int((~np.isnan(gm) | ~np.isnan(yoy)).sum())
            fin_ok = (n_fin >= MIN_FIN_OBS
                      and np.nanvar(gm) > 0 and np.nanvar(yoy) > 0)
            if fin_ok:
                ft = fit("B", [Channel("gross_margin", "normal", gm),
                               Channel("revenue_yoy", "normal", yoy)],
                         "gross_margin")
                res_t["B"] = float(ft.filtered[-1, 1])

        # ---- joint (feasible patent channel) ----
        if pat_ok_feas and fin_ok:
            jgrid = pd.period_range(min(pat_first, fin_first), t, freq="Q")
            yj = _grid_series(counts, jgrid, pat_first, t - FEASIBLE_LAG_Q)
            gmj, yoyj = fin_channels(jgrid)
            ft = fit("C", [Channel("patent_count", "nb", yj),
                           Channel("gross_margin", "normal", gmj),
                           Channel("revenue_yoy", "normal", yoyj)],
                     "patent_count")
            res_t["C"] = float(ft.filtered[-1, 1])
            for w in W_GRID:
                ft = fit(f"Cw_{w}",
                         [Channel("patent_count", "nb", yj, weight=w),
                          Channel("gross_margin", "normal", gmj),
                          Channel("revenue_yoy", "normal", yoyj)],
                         "gross_margin")
                res_t[f"Cw_{w}"] = float(ft.filtered[-1, 1])
        probs.append(res_t)
    return {"ticker": ticker, "probs": probs}


# --------------------------------------------------------------------------- #
# metrics — pure
# --------------------------------------------------------------------------- #
def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < MIN_CROSS_SECTION:
        return np.nan
    rho = spearmanr(x[m], y[m]).statistic
    return float(rho) if np.isfinite(rho) else np.nan

def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    m = ~(np.isnan(scores) | np.isnan(labels))
    s, l = scores[m], labels[m]
    n1, n0 = int((l == 1).sum()), int((l == 0).sum())
    if n1 < 2 or n0 < 2:
        return np.nan
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def nw_tstat(series: np.ndarray, lag: int) -> tuple[float, float, int]:
    """(mean, Newey-West t-stat with Bartlett kernel, n) of an IC series."""
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return (float(np.mean(x)) if n else np.nan, np.nan, n)
    m = float(x.mean())
    e = x - m
    g0 = float(e @ e) / n
    v = g0
    for l in range(1, min(lag, n - 1) + 1):
        gl = float(e[l:] @ e[:-l]) / n
        v += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    se = np.sqrt(max(v, 1e-12) / n)
    return m, m / se, n


def quarter_ranks(df: pd.DataFrame, cols: list[str],
                  by: str = "quarter") -> pd.DataFrame:
    """Rank-transform columns to [0,1] within each quarter (combiner inputs)."""
    out = df.copy()
    for c in cols:
        r = df.groupby(by)[c].rank(method="average")
        nn = df.groupby(by)[c].transform(lambda s: s.notna().sum())
        out[c + "_rk"] = (r - 1) / np.maximum(nn - 1, 1)
    return out


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta

def predict_ols(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta

def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-4,
                 n_iter: int = 50) -> np.ndarray:
    """Newton-Raphson logistic with tiny ridge (separation guard)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X1.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X1 @ b, -30, 30)))
        wdiag = np.maximum(p * (1 - p), 1e-8)
        grad = X1.T @ (y - p) - l2 * b
        hess = (X1 * wdiag[:, None]).T @ X1 + l2 * np.eye(len(b))
        step = np.linalg.solve(hess, grad)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b

def predict_logistic(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(
        np.column_stack([np.ones(len(X)), X]) @ b, -30, 30)))
