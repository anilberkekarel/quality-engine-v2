"""Single source of truth for the data layer.

MIGRATION-RESILIENT BY DESIGN. Every source identifier, field name, and tunable
parameter lives HERE so that when a source moves we patch one file, not the
provider logic.

=============================== ACTIVE SOURCE ===============================
Google BigQuery public dataset:  `patents-public-data.patents.publications`
WHY we switched here from the PatentSearch API:
  - The PatentSearch / USPTO ODP migration (2026-03) left the search API
    unreliable, and ODP API keys require a US-identity (ID.me/SSN) we lack.
  - BigQuery needs only a Google Cloud login (no special key), the assignee
    field is already HARMONIZED, all three dates sit in one table, and queries
    are fully reproducible. Categorically better for a research artifact.
See BIGQUERY_* constants below. The legacy PatentSearch constants are kept for
reference (and the old provider still compiles) but are NOT used.
============================================================================

--- legacy PatentSearch API status (kept for reference, 2026-06-05) ---
- OLD endpoint  api.patentsview.org   : CLOSED 2025-05-01.
- search.patentsview.org/api/v1       : unreliable mid-ODP-migration; needs key.
"""

from __future__ import annotations

import os

# ===========================================================================
# ACTIVE SOURCE: Google BigQuery
# ===========================================================================
# Fully-qualified table (change here only if the public dataset is renamed).
BIGQUERY_TABLE = "patents-public-data.patents.publications"

# GCP billing/quota project. Queries are billed to whatever project the client
# resolves. Set env QVM_GCP_PROJECT to pin it; otherwise the BigQuery client
# falls back to the Application-Default-Credentials default project.
GCP_PROJECT_ENV_VAR = "QVM_GCP_PROJECT"

# Cost guardrail: abort any query estimated to scan more than this (bytes).
# Keeps us safely inside the 1 TB/month free tier during development.
MAX_BYTES_BILLED = 80 * 1024**3  # 80 GiB

# Only US-origin patents (p.country_code on the MAIN table -> ALWAYS alias `p`
# to avoid the "country_code is ambiguous" clash with assignee_harmonized).
PATENT_COUNTRY = "US"

# Dates in this table are INTEGER yyyymmdd (e.g. 20180315). grant_date == 0
# means "not (yet) granted". The provider parses these to ISO dates.
DATE_NOT_GRANTED_SENTINEL = 0

# Figures are restricted to this start year for readability. The RAW CSV keeps
# full history (transparency), but the table reaches back to ~1921 — those very
# old records (mostly AMD, ~0.9% of rows) are almost certainly harmonization
# artifacts (AMD was founded 1969), so we don't let them stretch the x-axis.
PLOT_START_YEAR = 2005

# Trailing years to flag as INCOMPLETE in the filing series: recent filings are
# not yet published (~18mo secrecy), so the last counts undercount — a data
# artifact, not a real decline. Marked, not hidden (evidence for question c).
INCOMPLETE_TRAILING_YEARS = 2

# ---------------------------------------------------------------------------
# ENDPOINT  (change here only)
# ---------------------------------------------------------------------------
API_BASE_URL = "https://search.patentsview.org/api/v1"

# Resource paths under the base URL.
#   /patent/       -> GRANTED patents; carries filing date AND grant date.
#   /publication/  -> PRE-GRANT publications (PGPubs); carries publication date.
# These are SEPARATE datasets; see analysis/lag.py for how we join them.
PATENT_PATH = "/patent/"
PUBLICATION_PATH = "/publication/"

API_KEY_HEADER = "X-Api-Key"
API_KEY_ENV_VAR = "PATENTSVIEW_API_KEY"
API_KEY_FILE = ".apikey"  # repo-root fallback, gitignored

# ---------------------------------------------------------------------------
# FIELD NAMES  (the API's vocabulary -> our internal vocabulary)
# Centralised so a schema rename is a one-line fix. If a request 400s on an
# unknown field, this is the first place to look.
# ---------------------------------------------------------------------------

# Fields requested from the GRANTED-PATENT endpoint.
PATENT_FIELDS = {
    "patent_id": "patent_id",                       # granted patent number
    "grant_date": "patent_date",                    # date patent was granted
    "filing_date": "application.filing_date",       # date application was filed
    "application_id": "application.application_id",  # join key to publications
    "assignee_org": "assignees.assignee_organization",
    "assignee_id": "assignees.assignee_id",         # disambiguated assignee id
}

# Fields requested from the PRE-GRANT PUBLICATION endpoint.
PUBLICATION_FIELDS = {
    "document_number": "document_number",           # PGPub number
    "publication_date": "publication_date",         # date published (~18mo)
    "filing_date": "application.filing_date",
    "application_id": "application.application_id",  # join key to granted patents
    "assignee_org": "assignees.assignee_organization",
    "assignee_id": "assignees.assignee_id",
}

# JSON keys the API wraps results in (plural of the endpoint resource).
PATENT_RESULT_KEY = "patents"
PUBLICATION_RESULT_KEY = "publications"

# Pagination: PatentSearch caps a page at 1000 rows and paginates by a sort
# cursor ("after"). We sort by the resource id for a stable cursor.
PAGE_SIZE = 1000
PATENT_SORT_FIELD = "patent_id"
PUBLICATION_SORT_FIELD = "document_number"

# ---------------------------------------------------------------------------
# STUDY UNIVERSE  (the four test companies)
# Each entry documents the assignee-name variations we accept and WHY the
# company is in the study. This block IS the methodological-transparency record
# the whitepaper will cite for assignee matching.
# ---------------------------------------------------------------------------

# `name_like` is matched against assignee_harmonized.name with a SQL LIKE.
# CRITICAL: we use a PREFIX match ('NVIDIA%'), NOT a contains ('%NVIDIA%').
# Verified in console: '%NVIDIA%' pulls false positives (INVIDIATO COSMO L,
# CONVIDIA) that merely share letters; 'NVIDIA%' (begins-with) excludes them.
# This is the approved "Option 3". The harmonized field already collapses
# ~99% of NVIDIA into "NVIDIA CORP"; the prefix handles the ~1% tail
# (typos like NVIDIA CORPORTION, subsidiaries like NVIDIA TECHNOLOGY UK LTD).
COMPANIES = [
    {
        "ticker": "NVDA",
        "label": "NVIDIA",
        # Role: large, patent-rich, lived a 2020-23 revenue jump. POSITIVE case.
        "name_like": "NVIDIA%",
        "expect_min_patents": 5000,  # ~11.9k verified; far below => match broke
    },
    {
        "ticker": "AMD",
        "label": "AMD",
        # Role: cyclical recovery. MEDIUM case. Use AMD's full legal name;
        # bare 'AMD%' would over-match unrelated assignees.
        "name_like": "ADVANCED MICRO DEVICES%",
        "expect_min_patents": 3000,
    },
    {
        "ticker": "MRVL",
        "label": "Marvell",
        # Role: mid-cap, connectivity/networking. DOMAIN-RELEVANT case.
        "name_like": "MARVELL%",
        "expect_min_patents": 1000,
    },
    {
        "ticker": "MU",
        "label": "Micron",
        # Role: memory, EXTREMELY cyclical. CONTROL case. Revenue is driven by
        # the supply/demand commodity cycle, not patents. A momentum signal here
        # would be a FALSE POSITIVE -> the study's internal control.
        "name_like": "MICRON%",
        "expect_min_patents": 3000,
    },
]

# ---------------------------------------------------------------------------
# TEMPORAL SCOPE
# ---------------------------------------------------------------------------
# Historical depth. Data updates quarterly; coverage runs to ~2025 Q3.
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# ---------------------------------------------------------------------------
# OUTPUT LOCATIONS
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_REPO_ROOT, "_cache")    # raw pulls, gitignored
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")  # whitepaper-grade artifacts
REPO_ROOT = _REPO_ROOT
