"""Methodological measurement: filing -> publication -> grant LAG.

This is the whitepaper's FIRST concrete methodological finding (research
question c). For each company we measure, per patent and in MONTHS:
    filing -> publication lag   (when does a filing become public?)
    filing -> grant lag         (when is it confirmed?)

Why it matters for look-ahead bias: if a momentum model keys off filing
dates, the filing information was NOT public at filing time -- on average it
became visible ~`filing->publication` months later. A point-in-time
backtest may therefore not legitimately "know" a filing on its filing date.
Quantifying that lag is the precondition for an honest backtest design.

Lags are computed only on patents that carry BOTH endpoints of the pair
(same patent), so they are exact per-patent differences, not cohort averages.
The count of usable patents (and the publication-match rate) is reported so
the reader can judge coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.base import CompanyPatents

_DAYS_PER_MONTH = 30.4375  # mean Gregorian month, for day->month conversion


def _months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    delta_days = (later - earlier).dt.days
    return delta_days / _DAYS_PER_MONTH


def company_lag_table(company: CompanyPatents) -> pd.DataFrame:
    """Per-patent lag rows for one company (only patents with the needed dates)."""
    rows = [{
        "patent_id": r.patent_id,
        "filing_date": r.filing_date,
        "publication_date": r.publication_date,
        "grant_date": r.grant_date,
    } for r in company.records]
    df = pd.DataFrame(rows)
    for col in ("filing_date", "publication_date", "grant_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["ticker"] = company.ticker
    df["label"] = company.label
    df["filing_to_publication_months"] = _months_between(
        df["publication_date"], df["filing_date"])
    df["filing_to_grant_months"] = _months_between(
        df["grant_date"], df["filing_date"])
    # Negative or absurd values signal data problems; keep them visible but
    # flag so the summary can exclude them transparently.
    return df


def _distribution(series: pd.Series) -> dict:
    s = series.dropna()
    s = s[(s >= 0) & (s < 600)]  # drop impossible lags (>50y) and negatives
    if s.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan,
                "min": np.nan, "max": np.nan, "p25": np.nan, "p75": np.nan,
                "std": np.nan}
    return {
        "n": int(s.size),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "p75": round(float(s.quantile(0.75)), 2),
        "std": round(float(s.std()), 2),
    }


def lag_summary(companies: list[CompanyPatents]) -> pd.DataFrame:
    """One summary row per (company, lag type): mean/median/min/max/IQR/std."""
    out_rows = []
    for c in companies:
        tbl = company_lag_table(c)
        for lag_name, col in (
            ("filing_to_publication", "filing_to_publication_months"),
            ("filing_to_grant", "filing_to_grant_months"),
        ):
            stats = _distribution(tbl[col])
            out_rows.append({
                "ticker": c.ticker,
                "label": c.label,
                "lag_type": lag_name,
                "unit": "months",
                "n_patents_total": c.n_patents,
                "n_used": stats["n"],
                "publication_match_rate": (
                    round(c.n_publication_matched / c.n_patents, 3)
                    if c.n_patents else np.nan),
                "mean": stats["mean"],
                "median": stats["median"],
                "p25": stats["p25"],
                "p75": stats["p75"],
                "min": stats["min"],
                "max": stats["max"],
                "std": stats["std"],
            })
    return pd.DataFrame(out_rows)


def cross_company_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    """Is the lag consistent across companies, or variable? (research q. c)

    Returns, per lag type, the spread of company MEAN lags: the min/max mean,
    the range, and the coefficient of variation across companies.
    """
    rows = []
    for lag_type, grp in summary.groupby("lag_type"):
        means = grp["mean"].dropna()
        if means.empty:
            continue
        mu = float(means.mean())
        rows.append({
            "lag_type": lag_type,
            "n_companies": int(means.size),
            "mean_of_company_means": round(mu, 2),
            "min_company_mean": round(float(means.min()), 2),
            "max_company_mean": round(float(means.max()), 2),
            "range_months": round(float(means.max() - means.min()), 2),
            "cv_across_companies": (
                round(float(means.std() / mu), 3) if mu else np.nan),
        })
    return pd.DataFrame(rows)
