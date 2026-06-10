# QVM-V2 — Multi-Channel Latent-State Fusion Research Engine

> ⚠️ **Status: work in progress — early-stage research project.** This repo is
> at **Step 2**: data layers (patent + financial channels) + channel alignment
> infrastructure. There is **no signal processing, no model, and no forecasting
> yet** — those are *planned*, not done. The end goal is an arXiv-quality
> working paper.

A research codebase, separate from V1
[*quality-engine*](https://github.com/anilberkekarel/quality-engine) (which did
financial quality scoring). V2's framing: a company's hidden **structural
quality state** is not directly observable; patent-filing rhythm, margins and
revenue growth are noisy **sensors** of that state. Step 1's finding motivates
the fusion: a naive patent count could NOT separate NVIDIA (structural rise)
from Micron (commodity cycle) — single-channel is not enough. Research
questions:

> **(a)** In semiconductor companies, are changes in patent-filing rhythm an
> *early indicator* of future revenue jumps, or just cyclical noise?
>
> **(b)** Do signal-processing methods capture momentum better than a naive
> patent count? *(naive baseline established in Step 1 — kept untouched as the
> comparison bar.)*
>
> **(c)** Filing-date vs grant-date: which forecasts better, and how does
> look-ahead bias affect the comparison?

**Implemented so far: STEP 1** (patent data layer + raw visualization + date-lag
measurement) and **STEP 2** (assignee-match cleanup of the Micron control case,
SEC EDGAR XBRL financial channel with point-in-time `filed` dates, and the
quarterly channel-alignment grid `outputs/channels.csv` that the fusion model
will consume). The latent-state model itself is Step 3-4.

---

## Data source: Google BigQuery

Table: **`patents-public-data.patents.publications`**.

We switched here from the PatentSearch API because that API became unreliable
during the USPTO ODP migration (2026-03) and ODP keys require a US identity
(ID.me/SSN). BigQuery needs only a Google login, the assignee field is already
**harmonized**, all three dates live in one table, and queries are fully
reproducible. The retired PatentSearch provider remains in
`src/qvm/data/patentsview_provider.py` for reference but is unused.

### BigQuery auth (one-time)

Two options (the client auto-detects either via Application Default Credentials):

1. **gcloud login (recommended, no secret file):**
   ```bash
   gcloud auth application-default login
   gcloud config set project <YOUR_PROJECT_ID>
   ```
2. **Service-account key:** create a SA with the *BigQuery Job User* +
   *BigQuery Data Viewer* roles, download its JSON key, then
   `export GOOGLE_APPLICATION_CREDENTIALS=/path/key.json`.

Pin the billing/quota project with `export QVM_GCP_PROJECT=<YOUR_PROJECT_ID>`
(optional; falls back to the ADC default project).

### Cost

One narrow, parameterized query (only the needed columns; US + assignee
pre-filter). `run.py` does a **dry-run first**, logs the estimated scan size,
and aborts if it exceeds `config.MAX_BYTES_BILLED` (80 GiB) — comfortably
inside the 1 TB/month BigQuery free tier. No `SELECT *`.

---

## Run

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python run.py              # query BigQuery (needs auth), caches raw data
./venv/bin/python run.py --use-cache  # reuse _cache/raw_patents.csv, no query
```

`run.py` puts `src/` on the path itself — no `PYTHONPATH` needed.

---

## What it produces

Raw pull (gitignored):
- `_cache/raw_patents.csv` — one row per **application**, three dates each.

Whitepaper artifacts (committed, in `outputs/`):
- `naive_baseline_series.csv` — **naive-baseline** yearly + quarterly series
  (filing / publication / grant), tagged `series_kind=naive_baseline`, with an
  `incomplete` flag on the trailing filing years.
- `lag_summary.csv` — filing→publication & filing→grant lag per company
  (mean / median / IQR / min / max / std) — **methodological finding #1**.
- `lag_cross_company.csv` — is the lag consistent across companies?
- `assignee_match_audit.csv` — harmonized names matched per company, **plus an
  `excluded` section**: names the prefix would have captured but that an
  `exclude_name_like` rule or the `include_name_like` allowlist rejected, with
  counts and reasons. The `MICRON%` net needed both: MICRONAS GmbH / MICRONIC
  Laser (documented exclusions) plus a ~120-name tail of unrelated small
  companies — MU now requires a match to hit a curated allowlist of Micron
  Technology entities and their typo variants.
- `fig1_raw_filing_companies.png` / `fig1_normalized_filing_companies.png` —
  four-company filing rhythm, raw and normalized (Micron's volume dominates raw).
- `fig2_three_dates_NVDA.png` — NVIDIA's three date series overlaid (lag visible).

Step 2 artifacts (committed, in `outputs/`):
- **`channels.csv`** — the fusion model's input grid: one row per (company,
  calendar quarter) from 2009Q1 — patent_filing_count (+incomplete flag),
  revenue, revenue_yoy_growth, gross_margin, operating_margin, each financial
  value **as originally filed** with its `*_knowable_at` (SEC `filed`) date.
- `financials_quarterly.csv` — observation-level financial record: as-filed vs
  latest value, both filed dates, XBRL tag, form, derived-Q4 flag.
- `q4_derivation_sanity.csv` — per (concept, fiscal year): could Q4 be derived,
  deviation between derived and directly-filed Q4 facts where both exist.
- `yfinance_crosscheck.csv` — recent quarters cross-validated against Yahoo
  Finance (sanity only; yfinance has no filed dates and is never a source).

### Financial channel: SEC EDGAR XBRL (Step 2)

One cached `companyfacts` request per CIK (User-Agent with contact info, ≤10
req/s per SEC policy). Methodology, all enforced in code and logged per company:
- **Tag variants:** revenue is merged per period over the priority list
  `Revenues → RevenueFromContractWithCustomer... → SalesRevenueNet →
  SalesRevenueGoodsNet` (AMD's 2009-2011 quarters live only in the last one);
  same idea for cost of revenue. The console report prints exactly which tag
  covered which years.
- **Q4 derivation:** 10-Ks carry full-FY values only, so for flow concepts
  Q4 := FY − (Q1+Q2+Q3). Directly-filed Q4 facts (which often first surface in
  comparative filings 1-2 years later) are demoted to cross-checks.
- **Point-in-time:** every observation keeps the earliest `filed` date
  (as-filed value) and the latest (restated value) separately. The channel grid
  uses **as-filed**; e.g. Marvell's 2015-16 restatement saga stays visible.
- **Fiscal ≠ calendar year:** NVDA/MRVL fiscal years end late January, MU late
  Aug/early Sep. Alignment maps each XBRL period `end` to the **nearest
  calendar quarter end** — never the fy/fp labels, which would shift NVIDIA ~1
  year.
- **Entity history:** MRVL's ticker maps only to the 2021 successor entity;
  the predecessor CIK (Marvell Technology Group Ltd) is fetched and merged
  (`sec_additional_ciks` in config).
- `FMPProvider` is a skeleton for a possible premium source (key slot:
  `FMP_API_KEY` in a gitignored `.env`); `YFinanceProvider` is sanity-only.

---

## Architecture (mirrors V1)

```
src/qvm/
  config.py                       # source ids, fields, companies, tags, dates (single source)
  data/
    base.py                       # PatentRecord + CompanyPatents + PatentProvider (ABC)
    bigquery_provider.py          # ACTIVE patents: BigQuery patents-public-data
    patentsview_provider.py       # retired (kept for reference, unused)
    financial_base.py             # FinancialObservation + CompanyFinancials + FinancialProvider (ABC)
    sec_edgar_provider.py         # ACTIVE financials: SEC EDGAR XBRL companyfacts
    yfinance_provider.py          # sanity cross-check only (no filed dates)
    fmp_provider.py               # skeleton for a future premium source
  analysis/
    timeseries.py                 # NAIVE BASELINE yearly+quarterly counts (isolated)
    lag.py                        # filing/publication/grant lag measurement
    channels.py                   # channel-alignment grid (fusion model input)
  viz/
    plots.py                      # whitepaper-grade figures
run.py                            # orchestration + CSV/PNG output + console report
tests/test_pipeline_synthetic.py  # network-free patent-pipeline validation
tests/test_financials_synthetic.py# network-free financial-extraction validation
```

### Study universe & assignee matching

| Ticker | `name_like` (prefix) | Role |
|--------|----------------------|------|
| NVDA | `NVIDIA%` | positive case — large, patent-rich, 2020-23 revenue jump |
| AMD  | `ADVANCED MICRO DEVICES%` | medium case — cyclical recovery |
| MRVL | `MARVELL%` | domain case — mid-cap connectivity/networking |
| MU   | `MICRON%` | **CONTROL** — extreme commodity cycle; a momentum signal here would be a **false positive** |

Matching uses a **prefix** LIKE (`'NVIDIA%'`), *not* a contains (`'%NVIDIA%'`):
verified in console, contains pulls letter-coincidence false positives
(INVIDIATO COSMO L, CONVIDIA). `assignee_match_audit.csv` reports exactly which
harmonized names were folded in.

### Key schema handling (documented for transparency)

- **One row per publication event.** A pre-grant publication ("A" doc,
  `grant_date=0`) and the granted patent ("B" doc, `grant_date>0`) are separate
  rows sharing an `application_number`. We **collapse to one record per
  application**: filing date; earliest pre-grant publication date; grant date
  (or none).
- **Dates are INTEGER `yyyymmdd`** (e.g. `20180315`); `grant_date=0` means not
  (yet) granted. Parsed to ISO; sentinels dropped.
- **`country_code` is ambiguous** (exists on the table *and* inside
  `assignee_harmonized`) — we always alias the table `p` and write
  `p.country_code`.
- **Incomplete tail:** recent filings (~last 18 months) are not yet published,
  so the trailing filing-year counts undercount. Flagged (`incomplete` column,
  shaded/dashed in plots), never silently dropped — this *is* the look-ahead
  evidence for question (c).

---

## Roadmap — done vs planned

**Done (Step 1):**
- [x] BigQuery data layer (`patents-public-data`), ADC auth, A+B doc collapse
- [x] Naive-baseline annual/quarterly filing/publication/grant series
- [x] Filing→grant / filing→publication lag measurement (finding #1)
- [x] Whitepaper-grade figures + assignee-match audit

**Done (Step 2, this checkpoint):**
- [x] Assignee-match hardening: the Micron control now uses a curated entity
      allowlist on top of the prefix match (46,268 → 45,184 applications;
      MICRONAS/MICRONIC plus a ~120-name unrelated tail dropped), audit CSV
      gained an `excluded` section, all four companies re-audited
- [x] Financial channel: SEC EDGAR XBRL quarterly revenue / gross margin /
      operating margin with point-in-time `filed` dates, Q4 derivation,
      tag-variant handling, predecessor-CIK merge (MRVL)
- [x] Channel-alignment grid `outputs/channels.csv` (2009Q1→, calendar-quarter
      aligned by XBRL period end — the fusion model's input format)
- [x] yfinance cross-validation of recent quarters; FMP provider skeleton

**Planned (not started):**
- [ ] Latent-state fusion model (HMM-family) over the aligned channels (Step 3-4)
- [ ] Signal processing (normalization, change-point / acceleration detection)
      benchmarked against the untouched naive baseline (b)
- [ ] Point-in-time backtest honouring the measured lags + `knowable_at` dates (c)
- [ ] Write-up / working paper

This is a research project under active development; interfaces and outputs may change.
