"""CHANNEL ALIGNMENT GRID — the input format of the latent-state fusion model.

One row per (company, calendar quarter), 2009Q1 onward, fusing:
  patent channel    : patent_filing_count from the existing pipeline
                      (trailing ~2 years flagged incomplete, not hidden)
  financial channel : revenue, revenue_yoy_growth, gross_margin,
                      operating_margin — AS-FILED values (point-in-time), each
                      with knowable_at = the date its filing became public.

CALENDAR ALIGNMENT (the fiscal-year trap): NVDA/MRVL fiscal years end late
January, MU late Aug/early Sep — only AMD is calendar. Fiscal labels (fy/fp)
would shift NVDA ~1 year, so every fiscal period is mapped to the calendar
quarter whose end is NEAREST its XBRL 'end' date (a quarter ending 2024-01-28
covers Nov-Jan and lands in calendar 2023Q4). Fiscal quarters are ~91 days
apart, so the mapping is unambiguous; the original end date is kept in the
fiscal_period_end column for transparency.

NO signal processing here — this is alignment plumbing only. The
naive_baseline series remain untouched as the question-(b) comparison bar.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from .. import config
from ..data.base import CompanyPatents
from ..data.financial_base import CompanyFinancials, FinancialObservation

logger = logging.getLogger(__name__)


def calendar_quarter(end_iso: str) -> pd.Period:
    """Fiscal period end -> calendar quarter with the NEAREST quarter end."""
    end = dt.date.fromisoformat(end_iso)
    p = pd.Period(end, freq="Q")
    candidates = [p - 1, p, p + 1]
    return min(candidates, key=lambda q: abs((q.end_time.date() - end).days))


def _per_quarter(fin: CompanyFinancials, concept: str) -> dict[pd.Period, FinancialObservation]:
    """concept observations keyed by calendar quarter (collision-checked)."""
    out: dict[pd.Period, FinancialObservation] = {}
    for obs in fin.quarterly(concept):
        q = calendar_quarter(obs.end)
        if q in out:
            logger.warning("[%s] %s: two fiscal periods map to %s (%s, %s) — "
                           "keeping the first", fin.ticker, concept, q,
                           out[q].end, obs.end)
            continue
        out[q] = obs
    return out


def _patent_quarters(company: CompanyPatents) -> tuple[dict[pd.Period, int], int]:
    """Filing counts per calendar quarter + first incomplete-tail year.

    Any grid quarter from the cutoff year ON is incomplete — including
    quarters with zero recorded filings (they are the most undercounted of
    all, so they must not pass as complete zeros).
    """
    dates = pd.to_datetime(
        [r.filing_date for r in company.records if r.filing_date], errors="coerce")
    dates = dates[dates.notna()]
    counts = dates.to_period("Q").value_counts().sort_index()
    if counts.empty:
        return {}, 0
    cutoff_year = counts.index.max().year - (config.INCOMPLETE_TRAILING_YEARS - 1)
    return dict(counts), cutoff_year


def build_channels(companies: list[CompanyPatents],
                   financials: dict[str, CompanyFinancials]) -> pd.DataFrame:
    """The fusion-ready grid: one row per (company, calendar quarter)."""
    start_q = pd.Period(config.CHANNELS_START_QUARTER, freq="Q")
    rows = []
    for c in companies:
        fin = financials.get(c.ticker)
        pat_counts, pat_cutoff_year = _patent_quarters(c)
        rev = _per_quarter(fin, "revenue") if fin else {}
        cor = _per_quarter(fin, "cost_of_revenue") if fin else {}
        gp = _per_quarter(fin, "gross_profit") if fin else {}
        op = _per_quarter(fin, "operating_income") if fin else {}

        last_q = max([q for q in list(pat_counts) + list(rev) if q >= start_q],
                     default=start_q)
        q = start_q
        while q <= last_q:
            row = {
                "ticker": c.ticker, "label": c.label, "quarter": str(q),
                "period_start": q.start_time.date().isoformat(),
                "patent_filing_count": pat_counts.get(q, 0),
                "patent_incomplete": bool(pat_counts) and q.year >= pat_cutoff_year,
                "fiscal_period_end": None, "revenue": None,
                "revenue_yoy_growth": None, "gross_margin": None,
                "operating_margin": None, "revenue_is_derived_q4": None,
                "revenue_knowable_at": None, "gross_margin_knowable_at": None,
                "operating_margin_knowable_at": None,
            }
            r = rev.get(q)
            if r is not None:
                row.update({
                    "fiscal_period_end": r.end, "revenue": r.value,
                    "revenue_is_derived_q4": r.derived_q4,
                    "revenue_knowable_at": r.filed,
                })
                r_prev = rev.get(q - 4)
                if r_prev is not None and r_prev.value:
                    row["revenue_yoy_growth"] = r.value / r_prev.value - 1.0
                # gross margin: prefer the GrossProfit tag, else Rev - CoR
                g, co = gp.get(q), cor.get(q)
                if g is not None and r.value:
                    row["gross_margin"] = g.value / r.value
                    row["gross_margin_knowable_at"] = max(r.filed, g.filed)
                elif co is not None and r.value:
                    row["gross_margin"] = (r.value - co.value) / r.value
                    row["gross_margin_knowable_at"] = max(r.filed, co.filed)
                o = op.get(q)
                if o is not None and r.value:
                    row["operating_margin"] = o.value / r.value
                    row["operating_margin_knowable_at"] = max(r.filed, o.filed)
            rows.append(row)
            q += 1
    return pd.DataFrame(rows)


def financials_detail_table(financials: dict[str, CompanyFinancials]) -> pd.DataFrame:
    """Observation-level dump (as_filed + latest + provenance) for the record."""
    rows = []
    for ticker, fin in financials.items():
        for o in fin.observations:
            rows.append({
                "ticker": ticker, "concept": o.concept, "tag": o.tag,
                "start": o.start, "end": o.end,
                "calendar_quarter": str(calendar_quarter(o.end)),
                "value_as_filed": o.value, "filed_as_filed": o.filed,
                "value_latest": o.value_latest, "filed_latest": o.filed_latest,
                "restated": o.restated, "n_filings": o.n_filings,
                "form": o.form, "fiscal_label": o.fiscal_label,
                "derived_q4": o.derived_q4,
            })
    return pd.DataFrame(rows).sort_values(
        ["ticker", "concept", "end"]).reset_index(drop=True)


def channels_summary(channels: pd.DataFrame) -> pd.DataFrame:
    """Per-company coverage summary for the console report."""
    rows = []
    for ticker, g in channels.groupby("ticker", sort=False):
        fin = g[g["revenue"].notna()]
        rows.append({
            "ticker": ticker,
            "quarters_total": len(g),
            "quarters_with_revenue": len(fin),
            "first_fin_quarter": fin["quarter"].min() if not fin.empty else None,
            "last_fin_quarter": fin["quarter"].max() if not fin.empty else None,
            "quarters_with_gross_margin": int(g["gross_margin"].notna().sum()),
            "quarters_with_op_margin": int(g["operating_margin"].notna().sum()),
            "filed_date_coverage": (f"{fin['revenue_knowable_at'].notna().sum()}"
                                    f"/{len(fin)}" if not fin.empty else "0/0"),
        })
    return pd.DataFrame(rows)
