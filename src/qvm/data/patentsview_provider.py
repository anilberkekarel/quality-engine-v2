"""Concrete PatentProvider over the PatentSearch API (search.patentsview.org/api/v1).

Design notes
------------
* All endpoints/fields come from config.py -> migration-resilient.
* Two endpoints are queried per company:
    - /patent/      -> granted patents (filing_date + grant_date in one row)
    - /publication/ -> pre-grant publications (publication_date)
  They are joined on the application id to attach a publication_date to each
  granted patent. Patents with no matching PGPub keep publication_date = None.
* Assignee matching: we filter on the disambiguated assignee_organization text
  and then AUDIT the disambiguated assignee_id values that came back
  (CompanyPatents.matched_assignee_ids), so match quality is inspectable.
* Cohort definition: we fetch by FILING-date window [START, END]. Binning by
  grant/publication then naturally exposes the lag and the right-censoring of
  recent filings (central to research question c). Documented, not hidden.
"""

from __future__ import annotations

import logging
import os
import time

import requests

from .. import config
from .base import CompanyPatents, PatentProvider, PatentRecord

logger = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    """Raised when no PatentSearch API key can be found."""


def _load_api_key() -> str:
    """Find the API key in the env var or the repo-root .apikey file."""
    key = os.environ.get(config.API_KEY_ENV_VAR)
    if key:
        return key.strip()
    key_path = os.path.join(config.REPO_ROOT, config.API_KEY_FILE)
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if content:
            return content
    raise MissingAPIKeyError(
        "No PatentSearch API key found.\n"
        f"  Set ${config.API_KEY_ENV_VAR}, or create a one-line file "
        f"'{config.API_KEY_FILE}' at the repo root.\n"
        "  Request a free key: "
        "https://patentsview-support.atlassian.net/servicedesk/customer/portals"
    )


def _first(value):
    """Return a scalar from a value the API may nest as a list/dict.

    Group fields (e.g. 'application') come back as a list of dicts or a dict.
    For one-to-one-ish groups we take the first element.
    """
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _get(obj, dotted_key: str):
    """Resolve a possibly-dotted, possibly-nested field from a result dict.

    'application.filing_date' -> obj['application'][0]['filing_date'].
    Returns None if any hop is missing.
    """
    head, _, tail = dotted_key.partition(".")
    val = obj.get(head)
    if not tail:
        return val
    val = _first(val)
    if not isinstance(val, dict):
        return None
    return _get(val, tail)


class PatentsViewProvider(PatentProvider):
    """Fetches company patents (with three dates) from PatentSearch."""

    def __init__(self, api_key: str | None = None, request_delay: float = 0.4):
        self.api_key = api_key or _load_api_key()
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({config.API_KEY_HEADER: self.api_key})

    # -- low-level paged query -------------------------------------------------
    def _query_all(
        self,
        path: str,
        query: dict,
        fields: list[str],
        result_key: str,
        sort_field: str,
    ) -> list[dict]:
        """Run a query and follow the sort cursor until all rows are pulled."""
        url = config.API_BASE_URL + path
        rows: list[dict] = []
        after = None
        page = 0
        while True:
            options = {"size": config.PAGE_SIZE}
            if after is not None:
                options["after"] = after
            payload = {
                "q": query,
                "f": fields,
                "s": [{sort_field: "asc"}],
                "o": options,
            }
            resp = self.session.post(url, json=payload, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"{path} returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
            body = resp.json()
            batch = body.get(result_key) or []
            rows.extend(batch)
            page += 1
            total = body.get("total_hits", body.get("count"))
            logger.info("    page %d: +%d rows (running %d / total %s)",
                        page, len(batch), len(rows), total)
            if len(batch) < config.PAGE_SIZE:
                break
            after = _get(batch[-1], sort_field)
            if after is None:
                logger.warning("    cursor field %s missing; stopping early",
                               sort_field)
                break
            time.sleep(self.request_delay)
        return rows

    # -- query builders --------------------------------------------------------
    @staticmethod
    def _assignee_filter(name_variants: list[str], filing_field: str,
                         start_date: str, end_date: str) -> dict:
        """Build the q-object: (org matches ANY variant) AND filing in window."""
        org_field = config.PATENT_FIELDS["assignee_org"]
        name_clauses = [{"_text_phrase": {org_field: v}} for v in name_variants]
        org_clause = name_clauses[0] if len(name_clauses) == 1 else {"_or": name_clauses}
        return {
            "_and": [
                org_clause,
                {"_gte": {filing_field: start_date}},
                {"_lte": {filing_field: end_date}},
            ]
        }

    # -- public contract -------------------------------------------------------
    def get_company_patents(
        self,
        ticker: str,
        label: str,
        name_variants: list[str],
        start_date: str,
        end_date: str,
    ) -> CompanyPatents:
        logger.info("[%s] fetching granted patents ...", label)
        patents = self._fetch_patents(name_variants, start_date, end_date)
        logger.info("[%s] %d granted patents", label, len(patents))

        logger.info("[%s] fetching pre-grant publications ...", label)
        pub_date_by_app = self._fetch_publication_dates(
            name_variants, start_date, end_date)
        logger.info("[%s] %d publications with an application id",
                    label, len(pub_date_by_app))

        records: list[PatentRecord] = []
        assignee_audit: dict[str, int] = {}
        n_pub_matched = 0
        for p in patents:
            app_id = _get(p, config.PATENT_FIELDS["application_id"])
            pub_date = pub_date_by_app.get(app_id) if app_id else None
            if pub_date:
                n_pub_matched += 1
            # audit ALL disambiguated assignee ids on the patent
            for a in (p.get("assignees") or []):
                aid = a.get("assignee_id")
                if aid:
                    assignee_audit[aid] = assignee_audit.get(aid, 0) + 1
            assignee = _first(p.get("assignees"))
            records.append(PatentRecord(
                patent_id=p.get("patent_id"),
                assignee_id=(assignee or {}).get("assignee_id"),
                assignee_org=(assignee or {}).get("assignee_organization"),
                filing_date=_get(p, config.PATENT_FIELDS["filing_date"]),
                publication_date=pub_date,
                grant_date=_get(p, config.PATENT_FIELDS["grant_date"]),
                application_id=app_id,
            ))

        return CompanyPatents(
            ticker=ticker,
            label=label,
            name_variants=name_variants,
            records=records,
            matched_assignee_ids=dict(sorted(
                assignee_audit.items(), key=lambda kv: kv[1], reverse=True)),
            n_publication_matched=n_pub_matched,
        )

    # -- internals -------------------------------------------------------------
    def _fetch_patents(self, name_variants, start_date, end_date) -> list[dict]:
        query = self._assignee_filter(
            name_variants, config.PATENT_FIELDS["filing_date"],
            start_date, end_date)
        fields = sorted(set(config.PATENT_FIELDS.values()))
        return self._query_all(
            config.PATENT_PATH, query, fields,
            config.PATENT_RESULT_KEY, config.PATENT_SORT_FIELD)

    def _fetch_publication_dates(self, name_variants, start_date,
                                 end_date) -> dict[str, str]:
        """Return {application_id: publication_date} for the assignee."""
        query = self._assignee_filter(
            name_variants, config.PUBLICATION_FIELDS["filing_date"],
            start_date, end_date)
        fields = sorted(set(config.PUBLICATION_FIELDS.values()))
        rows = self._query_all(
            config.PUBLICATION_PATH, query, fields,
            config.PUBLICATION_RESULT_KEY, config.PUBLICATION_SORT_FIELD)
        out: dict[str, str] = {}
        for r in rows:
            app_id = _get(r, config.PUBLICATION_FIELDS["application_id"])
            pub_date = _get(r, config.PUBLICATION_FIELDS["publication_date"])
            if app_id and pub_date:
                out[app_id] = pub_date
        return out
