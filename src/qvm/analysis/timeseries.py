"""NAIVE BASELINE patent-count time series (yearly + quarterly).

================================ READ THIS ================================
The series produced here are the NAIVE BASELINE for research question (b).
They are RAW counts of patents by date. NO signal processing, NO
normalization, NO smoothing, NO change-point detection. When signal
processing is added later it will be benchmarked AGAINST these baseline
series. Keep this module free of any signal-processing logic.
===========================================================================

For each company we build, at YEARLY and QUARTERLY granularity, THREE
independent baseline series, one per date:
  - filing date       (innovation moment; secret at the time)
  - publication date  (~18 months later; first public)
  - grant date        (~2-3 years later; confirmed)

INCOMPLETE-TAIL CAVEAT: recent FILING counts undercount, because filings from
the last ~18 months are not yet published and so are absent from a publication
table. We flag (do not drop) the trailing `config.INCOMPLETE_TRAILING_YEARS`
periods on the filing series via an `incomplete` column, so plots can mark them.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..data.base import CompanyPatents

DATE_DIMENSIONS = ("filing_date", "publication_date", "grant_date")


def _records_dataframe(company: CompanyPatents) -> pd.DataFrame:
    rows = [{
        "filing_date": r.filing_date,
        "publication_date": r.publication_date,
        "grant_date": r.grant_date,
    } for r in company.records]
    df = pd.DataFrame(rows)
    for col in DATE_DIMENSIONS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _counts(company: CompanyPatents, granularity: str) -> pd.DataFrame:
    """granularity in {'year','quarter'} -> tidy counts for one company."""
    df = _records_dataframe(company)
    period = "Y" if granularity == "year" else "Q"
    frames = []
    for dim in DATE_DIMENSIONS:
        valid = df[df[dim].notna()]
        if valid.empty:
            continue
        p = valid[dim].dt.to_period(period)
        counts = p.value_counts().sort_index()
        frames.append(pd.DataFrame({
            "ticker": company.ticker,
            "label": company.label,
            "granularity": granularity,
            "date_dimension": dim,
            "period_start": counts.index.to_timestamp(),
            "year": [pp.year for pp in counts.index],
            "patent_count": counts.values,
            "series_kind": "naive_baseline",
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    _flag_incomplete(out)
    return out


def _flag_incomplete(df: pd.DataFrame) -> None:
    """Mark the trailing N years of the FILING series as incomplete (in place)."""
    df["incomplete"] = False
    filing = df[df["date_dimension"] == "filing_date"]
    if filing.empty:
        return
    max_year = int(filing["year"].max())
    cutoff = max_year - (config.INCOMPLETE_TRAILING_YEARS - 1)
    mask = (df["date_dimension"] == "filing_date") & (df["year"] >= cutoff)
    df.loc[mask, "incomplete"] = True


def naive_baseline(company: CompanyPatents) -> pd.DataFrame:
    """Yearly + quarterly naive-baseline series for one company, stacked."""
    parts = [_counts(company, g) for g in ("year", "quarter")]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_all_baselines(companies: list[CompanyPatents]) -> pd.DataFrame:
    frames = [naive_baseline(c) for c in companies]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
