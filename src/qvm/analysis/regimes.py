"""STEP 3 — patent-only NB-HMM baseline: single-channel regime detection.

Role in the ladder (naive count -> THIS -> joint fusion): an honest baseline.
The expectation is that a single-channel regime model will likely find a high
regime in Micron TOO — that is not failure, it is the thesis' evidence that
one channel cannot separate structural (NVDA) from cyclical (MU) and fusion
is required. Our job here is honest measurement: regime dates, persistence,
uncertainty.

Data discipline:
  * quarterly filing counts per company from config.HMM_START_QUARTER;
  * quarters flagged incomplete (trailing ~2 years, 18-month publication
    secrecy) are CUT, never modeled — the artificial decline would fabricate
    a fake low regime; the cut date is logged and stamped into outputs;
  * each company is fit separately (4 independent HMMs) — documented in
    config; pooling is Step 5 work.

Three implementations are compared (if they agree on regime dates, the result
is solid): our NB-HMM, our Poisson-HMM (was NB necessary?), and hmmlearn's
GaussianHMM on log(1+count) (independent cross-check).

HONESTY CAVEAT carried on every probability output: filtered P(S_t|y_1:t) is
causal in the observations but the PARAMETERS were estimated on the full
sample — future leaks through the parameters. True out-of-sample = expanding
window re-estimation (Step 5). Documented, not solved, in this baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config
from ..data.base import CompanyPatents
from ..models.nb_hmm import HMMFit, fit_hmm

logger = logging.getLogger(__name__)

POINT_IN_TIME_CAVEAT = (
    "filtered probs are causal in observations but parameters are full-sample "
    "(future leaks via parameters); true OOS = expanding-window, Step 5")


# --------------------------------------------------------------------------- #
# data prep
# --------------------------------------------------------------------------- #
def prepare_counts(company: CompanyPatents) -> tuple[pd.PeriodIndex, np.ndarray, str]:
    """Quarterly filing counts from HMM_START_QUARTER to the last COMPLETE
    quarter. Returns (quarters, counts, cutoff_note). Incomplete tail is cut.
    """
    dates = pd.to_datetime([r.filing_date for r in company.records
                            if r.filing_date], errors="coerce")
    dates = dates[dates.notna()]
    counts = dates.to_period("Q").value_counts().sort_index()
    cutoff_year = counts.index.max().year - (config.INCOMPLETE_TRAILING_YEARS - 1)
    last_complete = pd.Period(f"{cutoff_year - 1}Q4", freq="Q")
    start = pd.Period(config.HMM_START_QUARTER, freq="Q")
    grid = pd.period_range(start, last_complete, freq="Q")
    y = counts.reindex(grid).fillna(0).to_numpy(dtype=float)
    note = (f"modeled {grid[0]}..{grid[-1]} ({len(grid)} quarters); "
            f"incomplete tail from {cutoff_year}Q1 cut (18mo secrecy artifact)")
    return grid, y, note


# --------------------------------------------------------------------------- #
# switch detection
# --------------------------------------------------------------------------- #
def persistent_switches(quarters: pd.PeriodIndex, p_high: np.ndarray,
                        min_persist: int = None) -> list[dict]:
    """Dates where filtered P(high) crosses 0.5 PERSISTENTLY (>= min_persist
    consecutive quarters on the new side). Single-quarter blips don't count.
    """
    mp = min_persist or config.HMM_SWITCH_MIN_PERSIST
    side = p_high >= 0.5
    switches = []
    cur = bool(side[0])
    t = 1
    while t < len(side):
        run = side[t:t + mp]
        # a switch requires the NEW side to hold for the next mp quarters;
        # too-short runs (blips, or an unconfirmable tail) are not switches
        if bool(side[t]) != cur and len(run) == mp and (run == side[t]).all():
            switches.append({
                "quarter": str(quarters[t]),
                "direction": "up" if side[t] else "down",
                "p_high_at_switch": float(p_high[t]),
            })
            cur = bool(side[t])
            t += mp
        else:
            t += 1
    return switches


# --------------------------------------------------------------------------- #
# per-company model suite
# --------------------------------------------------------------------------- #
def _gaussian_crosscheck(y: np.ndarray, n_restarts: int, seed: int = 0):
    """hmmlearn GaussianHMM on log1p(count) — independent implementation.
    Returns (p_high_smoothed, viterbi_high) or None if hmmlearn unavailable.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None
    x = np.log1p(y).reshape(-1, 1)
    best, best_ll = None, -np.inf
    for s in range(n_restarts):
        try:
            m = GaussianHMM(n_components=2, covariance_type="diag",
                            n_iter=500, tol=1e-6, random_state=seed + s)
            m.fit(x)
            ll = m.score(x)
            if ll > best_ll:
                best, best_ll = m, ll
        except Exception:
            continue
    if best is None:
        return None
    high = int(np.argmax(best.means_.ravel()))  # label guard for hmmlearn too
    post = best.predict_proba(x)[:, high]
    vit = (best.predict(x) == high).astype(int)
    return post, vit


def analyze_company(company: CompanyPatents, seed: int = 0) -> dict:
    """Fit the model suite for one company; return everything the CSVs need."""
    quarters, y, note = prepare_counts(company)
    logger.info("[%s] %s", company.ticker, note)
    kw = dict(n_restarts=config.HMM_N_RESTARTS, max_iter=config.HMM_MAX_ITER,
              tol=config.HMM_TOL, seed=seed)
    nb2 = fit_hmm(y, n_states=2, family="nb", **kw)
    nb3 = fit_hmm(y, n_states=3, family="nb", **kw)
    po2 = fit_hmm(y, n_states=2, family="poisson", **kw)
    gauss = _gaussian_crosscheck(y, n_restarts=config.HMM_N_RESTARTS, seed=seed)
    switches = persistent_switches(quarters, nb2.filtered[:, 1])
    return {"ticker": company.ticker, "label": company.label,
            "quarters": quarters, "y": y, "note": note,
            "nb2": nb2, "nb3": nb3, "po2": po2, "gauss": gauss,
            "switches": switches}


def analyze_all(companies: list[CompanyPatents]) -> list[dict]:
    return [analyze_company(c) for c in companies]


# --------------------------------------------------------------------------- #
# output tables
# --------------------------------------------------------------------------- #
def probabilities_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        nb2: HMMFit = res["nb2"]
        for i, q in enumerate(res["quarters"]):
            rows.append({
                "ticker": res["ticker"], "quarter": str(q),
                "patent_filing_count": int(res["y"][i]),
                "p_high_filtered": float(nb2.filtered[i, 1]),
                "p_high_smoothed": float(nb2.smoothed[i, 1]),
                "viterbi_state": int(nb2.viterbi_path[i]),
                "caveat": POINT_IN_TIME_CAVEAT if i == 0 else "",
            })
    return pd.DataFrame(rows)


def parameters_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        for key in ("nb2", "nb3", "po2"):
            fit: HMMFit = res[key]
            spread = (max(fit.restart_logliks) - min(fit.restart_logliks))
            at_best = sum(1 for l in fit.restart_logliks
                          if abs(l - fit.log_likelihood) < 1e-3)
            rows.append({
                "ticker": res["ticker"], "model": key,
                "family": fit.family, "n_states": fit.n_states,
                "mus": " | ".join(f"{m:.1f}" for m in fit.mus),
                "dispersion_r": None if fit.r is None else round(fit.r, 3),
                "transmat": " | ".join(f"{p:.3f}" for p in fit.transmat.ravel()),
                "expected_durations_q": " | ".join(
                    f"{d:.1f}" for d in fit.expected_durations),
                "log_likelihood": round(fit.log_likelihood, 2),
                "n_params": fit.n_params, "bic": round(fit.bic, 1),
                "n_obs": fit.n_obs, "em_converged": fit.converged,
                "restarts_at_best_logl": f"{at_best}/{len(fit.restart_logliks)}",
                "restart_logl_spread": round(spread, 3),
                "data_note": res["note"],
            })
    return pd.DataFrame(rows)


def switches_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        if not res["switches"]:
            rows.append({"ticker": res["ticker"], "quarter": None,
                         "direction": "none", "p_high_at_switch": None,
                         "criterion": f">=0.5 for >= {config.HMM_SWITCH_MIN_PERSIST} "
                                      "consecutive quarters (filtered)"})
        for s in res["switches"]:
            rows.append({"ticker": res["ticker"], **s,
                         "criterion": f">=0.5 for >= {config.HMM_SWITCH_MIN_PERSIST} "
                                      "consecutive quarters (filtered)"})
    return pd.DataFrame(rows)


def implementation_agreement(results: list[dict]) -> pd.DataFrame:
    """Do NB / Poisson / Gaussian implementations agree on the regime path?"""
    rows = []
    for res in results:
        nb_v = res["nb2"].viterbi_path
        po_v = res["po2"].viterbi_path
        row = {"ticker": res["ticker"],
               "nb_vs_poisson_viterbi_agree_pct":
                   round(100.0 * float((nb_v == po_v).mean()), 1)}
        if res["gauss"] is not None:
            _, g_v = res["gauss"]
            row["nb_vs_gaussian_viterbi_agree_pct"] = \
                round(100.0 * float((nb_v == g_v).mean()), 1)
        else:
            row["nb_vs_gaussian_viterbi_agree_pct"] = None
        row["nb_minus_poisson_bic"] = round(res["nb2"].bic - res["po2"].bic, 1)
        row["bic_2state_minus_3state"] = round(res["nb2"].bic - res["nb3"].bic, 1)
        rows.append(row)
    return pd.DataFrame(rows)
