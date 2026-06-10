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


def _spec_name_likes(spec: dict) -> list[str]:
    """A spec carries one prefix (`name_like`) or several (`name_likes`)."""
    return spec.get("name_likes") or [spec["name_like"]]


def build_query(specs: list[dict] | None = None) -> tuple[str, list]:
    """Build the single parameterized query for a set of company specs
    (default: the four study companies; the Step-5 universe passes its own).

    Returns (sql, query_parameters). LIKE params per company; we SELECT the
    matched assignee name and assign each row to a company in pandas by prefix.
    """
    from google.cloud import bigquery

    specs = specs if specs is not None else config.COMPANIES
    like_clauses, params = [], []
    for i, spec in enumerate(specs):
        likes = []
        for j, nl in enumerate(_spec_name_likes(spec)):
            pname = f"like{i}_{j}" if j else f"like{i}"
            likes.append(f"a.name LIKE @{pname}")
            params.append(bigquery.ScalarQueryParameter(pname, "STRING", nl))
        clause = "(" + " OR ".join(likes) + ")"
        # exclusions: false-positive prefixes the LIKE would otherwise capture
        for j, excl in enumerate(spec.get("exclude_name_like", [])):
            ename = f"excl{i}_{j}"
            clause += f" AND a.name NOT LIKE @{ename}"
            params.append(bigquery.ScalarQueryParameter(ename, "STRING", excl["prefix"]))
        # allowlist: a match must also hit one of the curated entity prefixes
        includes = spec.get("include_name_like", [])
        if includes:
            inc_names = []
            for j, inc in enumerate(includes):
                iname = f"inc{i}_{j}"
                inc_names.append(f"a.name LIKE @{iname}")
                params.append(bigquery.ScalarQueryParameter(iname, "STRING", inc))
            clause += " AND (" + " OR ".join(inc_names) + ")"
        like_clauses.append(f"({clause})")
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


def _is_excluded(spec: dict, name_upper: str) -> bool:
    """True if the spec's filters reject this assignee name.

    Mirrors the SQL: rejected when an exclude_name_like prefix matches, OR
    when an include_name_like allowlist exists and NO allowlist prefix matches.
    """
    if any(name_upper.startswith(e["prefix"].rstrip("%").upper())
           for e in spec.get("exclude_name_like", [])):
        return True
    includes = spec.get("include_name_like", [])
    return bool(includes) and not any(
        name_upper.startswith(inc.rstrip("%").upper()) for inc in includes)


def _assign_company(assignee_name: str, specs: list[dict] | None = None) -> dict | None:
    """Map a harmonized assignee name to its company spec via prefix match.

    Mirrors the SQL exactly: prefix LIKE minus exclude_name_like prefixes.
    Returns None for both never-matched names and excluded names; callers that
    need the excluded set for auditing use `_assign_company_audited`.
    """
    spec, excluded = _assign_company_audited(assignee_name, specs)
    return None if excluded else spec


def _assign_company_audited(assignee_name: str,
                            specs: list[dict] | None = None
                            ) -> tuple[dict | None, bool]:
    """Like _assign_company but flags exclusion: (spec, was_excluded).

    (spec, True) means the name matched spec's prefix but an exclusion rule
    rejected it — exactly the rows the audit CSV must report.
    """
    if not isinstance(assignee_name, str):
        return None, False
    up = assignee_name.upper()
    for spec in (specs if specs is not None else config.COMPANIES):
        for nl in _spec_name_likes(spec):
            if up.startswith(nl.rstrip("%").upper()):
                return spec, _is_excluded(spec, up)
    return None, False


def rows_to_companies(df: pd.DataFrame,
                      specs: list[dict] | None = None) -> list[CompanyPatents]:
    """PURE collapse: per-publication rows -> one PatentRecord per application.

    Network-free and fully unit-testable. Expects columns: application_number,
    assignee_name, country_code, filing_date, publication_date, grant_date,
    priority_date (raw INTEGER yyyymmdd for the date columns).
    """
    specs = specs if specs is not None else config.COMPANIES
    # tag each row with its company (drop rows that match no prefix -> tail
    # noise; rows an exclusion rule rejected are counted for the audit trail)
    df = df.copy()
    assigned = df["assignee_name"].map(lambda n: _assign_company_audited(n, specs))
    df["_spec"] = assigned.map(lambda t: t[0])
    df["_excluded"] = assigned.map(lambda t: t[1])
    excluded_rows = df[df["_excluded"]]
    df = df[df["_spec"].notna() & ~df["_excluded"]]
    df["_ticker"] = df["_spec"].map(lambda s: s["ticker"])

    companies: list[CompanyPatents] = []
    for spec in specs:
        sub = df[df["_ticker"] == spec["ticker"]]
        # excluded applications for THIS company (unique apps, not pub rows)
        excl_sub = excluded_rows[excluded_rows["_spec"].map(
            lambda s: s is not None and s["ticker"] == spec["ticker"])]
        excl_audit = {
            name: int(g["application_number"].nunique())
            for name, g in excl_sub.groupby("assignee_name", sort=False)
        } if not excl_sub.empty else {}
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
            name_variants=_spec_name_likes(spec), records=records,
            matched_assignee_ids=dict(sorted(
                name_audit.items(), key=lambda kv: kv[1], reverse=True)),
            n_publication_matched=n_pub,
            excluded_assignees=dict(sorted(
                excl_audit.items(), key=lambda kv: kv[1], reverse=True)),
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

    def fetch_all(self, specs: list[dict] | None = None) -> list[CompanyPatents]:
        """Run ONE query for a company-spec set and return collapsed records.

        Always a SINGLE consolidated query — scan cost is column-based, so 4
        or 50 companies cost the same bytes; per-company queries would
        multiply it.
        """
        from google.cloud import bigquery
        specs = specs if specs is not None else config.COMPANIES
        sql, params = build_query(specs)

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

        companies = rows_to_companies(df, specs)
        for c in companies:
            floor = next((s.get("expect_min_patents", 0)
                          for s in specs if s["ticker"] == c.ticker), 0)
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
