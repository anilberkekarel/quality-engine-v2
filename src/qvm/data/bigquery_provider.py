"""Concrete PatentProvider over Google BigQuery `patents-public-data`.

Same ABC contract as the (now-retired) PatentSearch provider — only the source
changed. The V1 DataProvider pattern earns its keep here.

KEY SCHEMA FACTS (verified in the BigQuery console):
  * Table `patents-public-data.patents.publications` has ONE ROW PER
    PUBLICATION EVENT. A single invention (application_number) appears as
    multiple rows: a pre-grant application publication ("A" doc, grant_date=0)
    and the granted patent ("B" doc, grant_date>0). Both rows share the same
    filing_date and application_number.
  * filing_date / publication_date / grant_date / priority_date are INTEGER
    yyyymmdd (e.g. 20180315). grant_date == 0 means "not (yet) granted".
  * assignee_harmonized is REPEATED -> must UNNEST; fields .name, .country_code.
  * country_code exists on BOTH the main table and inside assignee_harmonized
    -> ALWAYS alias the table `p` and write p.country_code (else "ambiguous").

To recover the THREE dates per invention we collapse the per-publication rows
to ONE record per application_number:
    filing_date       = the application's filing date
    publication_date  = earliest PRE-GRANT publication (min pub date among
                        grant_date==0 rows) -> the ~18-month "first public"
    grant_date        = grant date (min positive grant_date), or None
This makes the filing/publication/grant series and the lag stats consistent.

COST: one narrow query (only the needed columns, US + assignee pre-filter),
guarded by maximum_bytes_billed and preceded by a dry-run estimate that is
logged. No SELECT *.
"""

from __future__ import annotations

import logging

import pandas as pd

from .. import config
from .base import CompanyPatents, PatentProvider, PatentRecord

logger = logging.getLogger(__name__)

# Columns we pull. Deliberately minimal (bytes scanned == columns referenced).
_SELECT_COLUMNS = (
    "p.application_number AS application_number",
    "a.name AS assignee_name",
    "p.country_code AS country_code",
    "p.filing_date AS filing_date",
    "p.publication_date AS publication_date",
    "p.grant_date AS grant_date",
    "p.priority_date AS priority_date",
)


def _parse_yyyymmdd(value) -> str | None:
    """INTEGER/float yyyymmdd -> 'YYYY-MM-DD'. 0/NaN/invalid -> None."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n <= config.DATE_NOT_GRANTED_SENTINEL:  # 0 (or negative) => no date
        return None
    y, m, d = n // 10000, (n // 100) % 100, n % 100
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1700 <= y <= 2100):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def build_query() -> tuple[str, list]:
    """Build the single parameterized query for all study companies.

    Returns (sql, query_parameters). One LIKE param per company; we SELECT the
    matched assignee name and assign each row to a company in pandas by prefix.
    """
    from google.cloud import bigquery

    like_clauses, params = [], []
    for i, spec in enumerate(config.COMPANIES):
        pname = f"like{i}"
        like_clauses.append(f"a.name LIKE @{pname}")
        params.append(bigquery.ScalarQueryParameter(pname, "STRING", spec["name_like"]))
    country_param = bigquery.ScalarQueryParameter("country", "STRING", config.PATENT_COUNTRY)
    params.append(country_param)

    sql = (
        "SELECT\n  " + ",\n  ".join(_SELECT_COLUMNS) + "\n"
        f"FROM `{config.BIGQUERY_TABLE}` AS p,\n"
        "  UNNEST(p.assignee_harmonized) AS a\n"
        "WHERE p.country_code = @country\n"
        "  AND (" + " OR ".join(like_clauses) + ")\n"
    )
    return sql, params


def _assign_company(assignee_name: str) -> dict | None:
    """Map a harmonized assignee name to its company spec via prefix match."""
    if not isinstance(assignee_name, str):
        return None
    up = assignee_name.upper()
    for spec in config.COMPANIES:
        prefix = spec["name_like"].rstrip("%").upper()
        if up.startswith(prefix):
            return spec
    return None


def rows_to_companies(df: pd.DataFrame) -> list[CompanyPatents]:
    """PURE collapse: per-publication rows -> one PatentRecord per application.

    Network-free and fully unit-testable. Expects columns: application_number,
    assignee_name, country_code, filing_date, publication_date, grant_date,
    priority_date (raw INTEGER yyyymmdd for the date columns).
    """
    # tag each row with its company (drop rows that match no prefix -> tail noise)
    df = df.copy()
    df["_spec"] = df["assignee_name"].map(_assign_company)
    df = df[df["_spec"].notna()]
    df["_ticker"] = df["_spec"].map(lambda s: s["ticker"])

    companies: list[CompanyPatents] = []
    for spec in config.COMPANIES:
        sub = df[df["_ticker"] == spec["ticker"]]
        records: list[PatentRecord] = []
        name_audit: dict[str, int] = {}
        n_pub = 0
        # collapse per application_number
        for app_id, g in sub.groupby("application_number", sort=False):
            filing = _earliest(g["filing_date"])
            # pre-grant publication = earliest pub among NON-granted rows
            nongrant = g[g["grant_date"].fillna(0).astype("int64") <= config.DATE_NOT_GRANTED_SENTINEL]
            pub = _earliest(nongrant["publication_date"]) if not nongrant.empty else None
            grant = _earliest_positive(g["grant_date"])
            priority = _earliest(g["priority_date"])
            # representative harmonized name for this application (most common)
            org = g["assignee_name"].mode()
            org = org.iloc[0] if not org.empty else None
            if pub:
                n_pub += 1
            if org:
                name_audit[org] = name_audit.get(org, 0) + 1
            records.append(PatentRecord(
                patent_id=str(app_id),
                assignee_id=(org.upper() if isinstance(org, str) else None),
                assignee_org=org,
                filing_date=filing,
                publication_date=pub,
                grant_date=grant,
                application_id=str(app_id),
                priority_date=priority,
            ))
        companies.append(CompanyPatents(
            ticker=spec["ticker"], label=spec["label"],
            name_variants=[spec["name_like"]], records=records,
            matched_assignee_ids=dict(sorted(
                name_audit.items(), key=lambda kv: kv[1], reverse=True)),
            n_publication_matched=n_pub,
        ))
    return companies


def _earliest(series: pd.Series) -> str | None:
    parsed = [d for d in (_parse_yyyymmdd(v) for v in series) if d]
    return min(parsed) if parsed else None


def _earliest_positive(series: pd.Series) -> str | None:
    # grant dates only; _parse already drops 0/None
    return _earliest(series)


class BigQueryPatentProvider(PatentProvider):
    """Fetches harmonized company patents (three dates) from BigQuery."""

    def __init__(self, project: str | None = None, max_bytes_billed: int | None = None):
        from google.cloud import bigquery
        import os
        self.project = project or os.environ.get(config.GCP_PROJECT_ENV_VAR)
        # project=None lets the client use the ADC default project.
        self.client = bigquery.Client(project=self.project)
        self.max_bytes_billed = max_bytes_billed or config.MAX_BYTES_BILLED

    def fetch_all(self) -> list[CompanyPatents]:
        """Run ONE query for all study companies and return collapsed records."""
        from google.cloud import bigquery
        sql, params = build_query()

        # 1) dry run -> estimate + log scan size, abort if over guardrail
        dry = self.client.query(
            sql, job_config=bigquery.QueryJobConfig(
                query_parameters=params, dry_run=True, use_query_cache=False))
        gib = dry.total_bytes_processed / 1024**3
        logger.info("BigQuery dry-run: ~%.2f GiB will be scanned (guard %.0f GiB)",
                    gib, self.max_bytes_billed / 1024**3)
        if dry.total_bytes_processed > self.max_bytes_billed:
            raise RuntimeError(
                f"Query would scan {gib:.1f} GiB, over the "
                f"{self.max_bytes_billed/1024**3:.0f} GiB guardrail. Aborting.")

        # 2) real run, billed-bytes capped as a hard backstop
        job = self.client.query(
            sql, job_config=bigquery.QueryJobConfig(
                query_parameters=params,
                maximum_bytes_billed=self.max_bytes_billed))
        df = job.result().to_dataframe(create_bqstorage_client=False)
        logger.info("BigQuery returned %d publication rows (%.2f GiB billed)",
                    len(df), (job.total_bytes_billed or 0) / 1024**3)

        companies = rows_to_companies(df)
        for c in companies:
            floor = next((s.get("expect_min_patents", 0)
                          for s in config.COMPANIES if s["ticker"] == c.ticker), 0)
            flag = "  <-- BELOW EXPECTED, CHECK ASSIGNEE MATCH" if c.n_patents < floor else ""
            logger.info("[%s] %d applications | %d with pre-grant pub%s",
                        c.label, c.n_patents, c.n_publication_matched, flag)
        return companies

    # -- ABC contract (per-company convenience; fetch_all is the efficient path) -
    def get_company_patents(self, ticker, label, name_variants,
                            start_date, end_date) -> CompanyPatents:
        for c in self.fetch_all():
            if c.ticker == ticker:
                return c
        return CompanyPatents(ticker=ticker, label=label,
                              name_variants=name_variants, records=[])
