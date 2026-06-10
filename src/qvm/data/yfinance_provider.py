"""YFinanceProvider — independent SANITY CHECK for the SEC EDGAR channel.

yfinance only exposes the last ~4-5 quarters of income-statement data, so it
cannot replace EDGAR (no history, no filed dates => no point-in-time). Its job
is cross-validation: if Yahoo's recent quarters disagree materially with what
we extracted from XBRL, our tag selection or Q4 derivation is broken.

No filed dates exist here, so observations carry the period end as a
placeholder 'filed' and must NEVER feed the point-in-time channel grid.
"""

from __future__ import annotations

import logging

from .financial_base import CompanyFinancials, FinancialObservation, FinancialProvider

logger = logging.getLogger(__name__)

# yfinance income-statement row label -> our canonical concept
_ROW_TO_CONCEPT = {
    "Total Revenue": "revenue",
    "Cost Of Revenue": "cost_of_revenue",
    "Gross Profit": "gross_profit",
    "Operating Income": "operating_income",
}


class YFinanceProvider(FinancialProvider):
    """Quarterly income statement from Yahoo Finance (recent quarters only)."""

    def get_company_financials(self, ticker: str) -> CompanyFinancials:
        import yfinance as yf

        stmt = yf.Ticker(ticker).quarterly_income_stmt  # rows=items, cols=period ends
        fin = CompanyFinancials(ticker=ticker.upper(), cik=None, source="yfinance")
        if stmt is None or stmt.empty:
            fin.notes.append("yfinance returned no quarterly income statement")
            return fin
        for row_label, concept in _ROW_TO_CONCEPT.items():
            if row_label not in stmt.index:
                fin.notes.append(f"{concept}: row {row_label!r} missing")
                continue
            for col, val in stmt.loc[row_label].items():
                if val != val or val is None:  # NaN
                    continue
                end = col.date().isoformat()
                fin.observations.append(FinancialObservation(
                    concept=concept, tag=f"yfinance:{row_label}",
                    start="", end=end, value=float(val), filed=end,
                    value_latest=float(val), filed_latest=end,
                    n_filings=1, form="yfinance"))
        return fin


def cross_check(sec_fin: CompanyFinancials, yf_fin: CompanyFinancials,
                tolerance_pct: float = 2.0, match_days: int = 7) -> list[dict]:
    """Compare yfinance quarters against SEC 'latest' values by period end.

    yfinance reports current (possibly restated) figures, so the comparison
    target is value_latest, not the as-filed value. Period ends are matched
    within `match_days` (yfinance sometimes uses month-end approximations).
    Returns one record per (concept, quarter) with pct difference and a flag.
    """
    import datetime as dt

    out = []
    for concept in ("revenue", "cost_of_revenue", "gross_profit", "operating_income"):
        sec_obs = sec_fin.quarterly(concept)
        for y in yf_fin.quarterly(concept):
            y_end = dt.date.fromisoformat(y.end)
            best = min(sec_obs, key=lambda o: abs(
                (dt.date.fromisoformat(o.end) - y_end).days), default=None)
            if best is None:
                continue
            gap = abs((dt.date.fromisoformat(best.end) - y_end).days)
            if gap > match_days:
                out.append({"ticker": sec_fin.ticker, "concept": concept,
                            "yf_end": y.end, "sec_end": None, "pct_diff": None,
                            "status": "no SEC quarter within match window"})
                continue
            diff = (100.0 * (y.value - best.value_latest) / best.value_latest
                    if best.value_latest else None)
            out.append({
                "ticker": sec_fin.ticker, "concept": concept,
                "yf_end": y.end, "sec_end": best.end,
                "yf_value": y.value, "sec_value_latest": best.value_latest,
                "pct_diff": diff,
                "status": ("ok" if diff is not None and abs(diff) <= tolerance_pct
                           else "MISMATCH"),
            })
    return out
