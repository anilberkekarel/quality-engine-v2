"""Abstract contract for the FINANCIAL channel (V1 DataProvider pattern).

Mirrors base.py: plain-data carriers plus an abstract FinancialProvider.
SECEdgarProvider implements it today; YFinanceProvider cross-checks it;
FMPProvider is a slot for a future premium source. Analysis code only ever
sees these dataclasses, never a source's raw payload.

POINT-IN-TIME DISCIPLINE (the reason this module exists): every observation
carries the 'filed' date of the filing it came from — the date the number
became PUBLIC. A fiscal quarter is typically filed twice (its own 10-Q, then
next year's comparative 10-Q/10-K), and restatements add more. We keep TWO
values per period:
  value  / filed         : from the EARLIEST filing  -> "as_filed" series.
                           This is what a real-time observer knew; the fusion
                           model must train on THIS.
  value_latest / filed_latest : from the LATEST filing -> "latest" series,
                           kept separately for restatement diagnostics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FinancialObservation:
    """One flow value (revenue, cost, income) for one fiscal period."""

    concept: str            # canonical: revenue | cost_of_revenue | gross_profit | operating_income
    tag: str                # XBRL tag the value came from (methodology record)
    start: str              # ISO period start
    end: str                # ISO period end  -> THE alignment key (never fy/fp)
    value: float            # as-filed value (earliest filing)
    filed: str              # ISO date that value became public == knowable_at
    value_latest: float     # value per the most recent filing
    filed_latest: str
    n_filings: int          # how many filings reported this period
    form: str               # 10-Q / 10-K / 'derived(10-K)'
    fiscal_label: str = ""  # XBRL fy/fp as filed, e.g. 'FY2024 Q4' — display only
    derived_q4: bool = False  # True if computed as FY - (Q1+Q2+Q3)

    @property
    def restated(self) -> bool:
        return self.value_latest != self.value


@dataclass
class CompanyFinancials:
    """All extracted quarterly observations for one company + provenance."""

    ticker: str
    cik: int | None
    source: str                                  # 'sec_edgar' / 'yfinance' / ...
    observations: list[FinancialObservation] = field(default_factory=list)
    # concept -> [{tag, n_periods, first_end, last_end}] in priority order;
    # documents exactly which tag covered which years (whitepaper methodology).
    tag_provenance: dict[str, list[dict]] = field(default_factory=dict)
    # one record per (concept, fiscal year) Q4 derivation / FY-sum check.
    q4_sanity: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def quarterly(self, concept: str) -> list[FinancialObservation]:
        """Quarterly observations for one concept, sorted by period end."""
        return sorted((o for o in self.observations if o.concept == concept),
                      key=lambda o: o.end)


class FinancialProvider(ABC):
    """Abstract contract for a source of company quarterly financials."""

    @abstractmethod
    def get_company_financials(self, ticker: str) -> CompanyFinancials:
        """Return quarterly flow observations (with filed dates) for a ticker.

        Implementations must populate tag_provenance and q4_sanity so the
        caller can audit tag selection and Q4 derivation quality.
        """
        ...
