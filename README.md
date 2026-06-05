# QVM-V2 — Patent-Rhythm Research Engine

> ⚠️ **Status: work in progress — early-stage research project.** This repo is
> at **Step 1 only**: the data layer + raw visualization + a first
> methodological measurement. There is **no signal processing, no model, and no
> forecasting yet** — those are *planned*, not done. The end goal is an
> arXiv-quality working paper.

A research codebase, separate from V1
[*quality-engine*](https://github.com/anilberkekarel/quality-engine) (which did
financial quality scoring). V2 asks a **forecasting / research** question:

> **(a)** In semiconductor companies, are changes in patent-filing rhythm an
> *early indicator* of future revenue jumps, or just cyclical noise?
>
> **(b)** Do signal-processing methods capture momentum better than a naive
> patent count? *(naive baseline established here; signal processing later.)*
>
> **(c)** Filing-date vs grant-date: which forecasts better, and how does
> look-ahead bias affect the comparison?

**This repository currently implements STEP 1 only: the data layer + raw
visualization + the first methodological measurement (date lags).** No signal
processing, no model, no forecasting yet — by design.

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
- `assignee_match_audit.csv` — harmonized names matched per company (how each
  patent set was assembled — transparency).
- `fig1_raw_filing_companies.png` / `fig1_normalized_filing_companies.png` —
  four-company filing rhythm, raw and normalized (Micron's volume dominates raw).
- `fig2_three_dates_NVDA.png` — NVIDIA's three date series overlaid (lag visible).

---

## Architecture (mirrors V1)

```
src/qvm/
  config.py                       # source ids, fields, companies, dates (single source)
  data/
    base.py                       # PatentRecord + CompanyPatents + PatentProvider (ABC)
    bigquery_provider.py          # ACTIVE: BigQuery patents-public-data
    patentsview_provider.py       # retired (kept for reference, unused)
  analysis/
    timeseries.py                 # NAIVE BASELINE yearly+quarterly counts (isolated)
    lag.py                        # filing/publication/grant lag measurement
  viz/
    plots.py                      # whitepaper-grade figures
run.py                            # orchestration + CSV/PNG output + console report
tests/test_pipeline_synthetic.py  # network-free end-to-end validation
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

**Done (Step 1, this checkpoint):**
- [x] BigQuery data layer (`patents-public-data`), ADC auth, A+B doc collapse
- [x] Naive-baseline annual/quarterly filing/publication/grant series
- [x] Filing→grant / filing→publication lag measurement (finding #1)
- [x] Whitepaper-grade figures + assignee-match audit

**Planned (not started):**
- [ ] Signal processing (normalization, change-point / acceleration detection)
- [ ] Financial overlay (revenue series) to test the lead/lag hypothesis (a)
- [ ] Forecasting model + point-in-time backtest honouring the measured lags (c)
- [ ] Assignee-match hardening (e.g. drop MICRONAS/MICRONIC from the Micron control)
- [ ] Write-up / working paper

This is a research project under active development; interfaces and outputs may change.
