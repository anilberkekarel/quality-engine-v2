"""The data layer's abstract contract.

Mirrors the V1 design: a plain-data carrier (PatentRecord) plus an abstract
PatentProvider interface. A concrete provider (PatentsViewProvider) implements
the contract today; another source (ODP bulk, Google Patents, ...) can be
slotted in later without touching analysis or viz code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PatentRecord:
    """One granted patent, carrying the THREE dates the study turns on.

    Why three dates (research question c): we need to know which date is the
    stronger predictor and how look-ahead bias distorts that comparison.
      - filing_date      : when the invention happened (but was then secret)
      - publication_date : ~18 months later, first time it is public (PGPub)
      - grant_date       : when finally granted (~2-3 years), fully confirmed

    publication_date is Optional: not every granted patent has a matching
    pre-grant publication (e.g. non-publication requests, or PGPub data gaps).
    Such records are still valid for filing/grant series; they are simply
    excluded from filing->publication lag stats, and that exclusion is reported.
    """

    patent_id: str
    assignee_id: str | None       # matched assignee key (harmonized name in BQ)
    assignee_org: str | None      # the organization string as recorded
    filing_date: str | None       # ISO YYYY-MM-DD
    publication_date: str | None  # ISO YYYY-MM-DD, may be None (pre-grant pub)
    grant_date: str | None        # ISO YYYY-MM-DD, None if not (yet) granted
    application_id: str | None    # application_number; the per-invention key
    priority_date: str | None = None  # ISO YYYY-MM-DD, earliest priority (bonus)


@dataclass
class CompanyPatents:
    """All patents matched to one study company, plus the matching provenance.

    The provenance fields exist for methodological transparency: the whitepaper
    must be able to state exactly how a company's patent set was assembled.
    """

    ticker: str
    label: str
    name_variants: list[str]                   # assignee-name filters applied
    records: list[PatentRecord] = field(default_factory=list)
    matched_assignee_ids: dict[str, int] = field(default_factory=dict)
    # ^ disambiguated assignee_id -> patent count, so we can SEE whether the
    #   match pulled one clean entity or a scatter of near-duplicates.
    n_publication_matched: int = 0             # how many got a publication_date

    @property
    def n_patents(self) -> int:
        return len(self.records)


class PatentProvider(ABC):
    """Abstract contract for a source of company patent records."""

    @abstractmethod
    def get_company_patents(
        self,
        ticker: str,
        label: str,
        name_variants: list[str],
        start_date: str,
        end_date: str,
    ) -> CompanyPatents:
        """Return all patents (with the three dates) for one company.

        Implementations must perform assignee disambiguation using
        ``name_variants`` and populate ``matched_assignee_ids`` so the caller
        can audit the match quality.
        """
        ...
