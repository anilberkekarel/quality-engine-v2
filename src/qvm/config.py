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
#
# `exclude_name_like` (optional) lists prefixes that the company's `name_like`
# would otherwise capture but that belong to UNRELATED companies. Each entry
# carries the reason — this is the whitepaper's false-positive audit trail.
# `include_name_like` (optional) is the stricter tool for a polluted net: when
# present, a `name_like` match must ALSO hit one of these prefixes (a curated
# allowlist of the company's real entities), otherwise it is excluded.
# Both filters are applied in SQL for live pulls AND re-applied in pandas when
# loading the cache, so cached pulls from before the rule are cleaned
# identically.
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
        # SEC's ticker map only carries the SUCCESSOR entity (Marvell
        # Technology, Inc., CIK 1835632, created in the 2021 Inphi merger),
        # whose XBRL history starts FY2020. The pre-2021 history sits under
        # the predecessor's CIK, fetched additionally and merged per period.
        "sec_additional_ciks": [
            {"cik": 1058057,
             "reason": "Marvell Technology Group Ltd — predecessor entity "
                       "(Bermuda), delisted at the 2021 reorganization"},
        ],
    },
    {
        "ticker": "MU",
        "label": "Micron",
        # Role: memory, EXTREMELY cyclical. CONTROL case. Revenue is driven by
        # the supply/demand commodity cycle, not patents. A momentum signal here
        # would be a FALSE POSITIVE -> the study's internal control.
        "name_like": "MICRON%",
        "expect_min_patents": 3000,
        # The MICRON% net catches a LONG tail of unrelated small companies
        # (~120 names, ~1.1k apps): two sizable ones found in the Step-1 audit
        # (MICRONAS GmbH, MICRONIC Laser — kept below with documented reasons)
        # plus a Step-2 scatter (MICRON DEVICES LLC, MICRON OPTICS, MICRONEL
        # AG, MICRONIT BV, MICRONET*, MICRONIX*, ...). The control case must
        # be clean, so MU additionally uses an ALLOWLIST: a match must also
        # hit one of the `include_name_like` prefixes below — Micron
        # Technology Inc, its known subsidiaries, and their typo variants
        # (curated from the full 199-name audit dump). Ambiguous bare names
        # (MICRON CORP, MICRON CO LTD, MICRON INC...) are NOT allowlisted:
        # Micron Technology never filed under them, ~17 apps, likely Japanese
        # 'Micron Corp' etc. Everything dropped lands in the audit CSV.
        "include_name_like": [
            "MICRON TECH%",        # MICRON TECHNOLOGY INC + the typo tail
                                   # (TECHNOLGY, TECHONOLOGY, TECHNOLOGY INCV,
                                   # ...) + TECH INC / HOLDING / LICENSING
            "MICRON TEHNOLOGY%",   # typos that break the TECH prefix
            "MICRON TECNOLOGY%",
            "MICRON TCHNOLOGY%",
            "MICRON TEDHNOLOGY%",
            "MICRONG TECHNOLOGY%",
            "MICRONTECHNOLOGY%",   # concatenation typos
            "MICRONBTECHNOLOGY%",
            "MICRON ELECTRONIC%",  # Micron Electronics Inc — PC subsidiary
            "MICRON ELETRONICS%",  # its typos
            "MICRON ELECTONICS%",
            "MICRON SEMICONDUCTOR%",  # Micron Semiconductor Inc (Boise sub)
            "MICRON COMMUNICATION%",  # RFID subsidiary
            "MICRON DISPLAY%",        # FED display subsidiary
            "MICRON QUANTUM%",        # flash subsidiary (Micron Quantum Devices)
            "MICRONPC%",              # MicronPC LLC
            "MICRON PC%",
            "MICRON CUSTOM MANUFACTURING%",  # MCMS Inc spin-off
            "MICRON MEMORY%",         # Micron Memory Japan (ex-Elpida)
        ],
        "exclude_name_like": [
            {
                "prefix": "MICRONAS%",
                "reason": "Micronas GmbH/Intermetall — German semiconductor "
                          "company (Hall sensors), unrelated to Micron Technology",
            },
            {
                "prefix": "MICRONIC%",
                "reason": "Micronic Laser Systems / Mydata AB — Swedish "
                          "lithography company (and MICRONICS* entities), "
                          "unrelated to Micron Technology",
            },
        ],
    },
]

# ===========================================================================
# FINANCIAL CHANNEL (Step 2): SEC EDGAR XBRL primary, yfinance sanity, FMP slot
# ===========================================================================
# SEC EDGAR is free and key-less but REQUIRES a User-Agent identifying the
# requester (fair-access policy) and <=10 requests/second. We make ~5 requests
# total and cache every response, so we are far inside both limits.
SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_USER_AGENT = "QVM-Research anilberke.karel@gmail.com"
SEC_MIN_REQUEST_INTERVAL_S = 0.15  # >=0.1s between calls => <=10 req/s

# us-gaap tag priority per canonical concept. Companies switch tags across
# years (ASC 606 moved most from SalesRevenueNet/Revenues to RevenueFromContract...),
# so extraction merges PER PERIOD in this priority order and logs, per company,
# which tag covered which years (whitepaper methodology record).
CONCEPT_TAG_PRIORITY = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        # AMD 2009-2011 quarterly revenue lives ONLY here (verified in its
        # companyfacts); pro-forma tags (BusinessAcquisitionsProFormaRevenue)
        # are deliberately NOT listed.
        "SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
}

# Flow-period classification by duration in days. XBRL 10-Q facts also carry
# 6- and 9-month YTD cumulatives — anything outside these windows is skipped.
QUARTER_DURATION_DAYS = (70, 100)
ANNUAL_DURATION_DAYS = (340, 380)

# Refiled periods sometimes shift a boundary by a day (NVDA fiscal Q2-2011
# appears as ..2010-07-31 and ..2010-08-01 in different filings). Periods whose
# start AND end both differ by at most this many days are the SAME period.
PERIOD_MATCH_TOLERANCE_DAYS = 5

# Q4 is NOT filed separately (the 10-K carries the full FY): for flow concepts
# Q4 := FY - (Q1+Q2+Q3). Derived Q4s failing these plausibility screens are
# flagged in the q4 sanity report (negative revenue, or far off its siblings).
Q4_MAX_RATIO_TO_SIBLINGS = 3.0  # |Q4| vs max(|Q1..Q3|)

# FISCAL != CALENDAR YEAR (NVDA/MRVL end late Jan, MU late Aug/early Sep; only
# AMD is calendar). Alignment NEVER uses XBRL fy/fp labels — each period's
# 'end' date is mapped to the calendar quarter whose end is NEAREST (a fiscal
# quarter ending 2024-01-28 belongs to calendar 2023Q4, not 2024Q1's label).
# fy/fp would shift NVIDIA ~1 year and silently break the fusion alignment.

# Channel-grid scope (Step 2 alignment table for the fusion model).
CHANNELS_START_QUARTER = "2009Q1"

# FMP (Financial Modeling Prep) — future premium source; skeleton only.
# Key goes in .env (gitignored) as FMP_API_KEY=...
FMP_API_KEY_ENV_VAR = "FMP_API_KEY"

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
