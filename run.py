"""QVM-V2 entry point — STEP 1: data layer + raw visualization only.

Source: Google BigQuery `patents-public-data.patents.publications`.

Pipeline:
  1. Fetch (or load cached) harmonized patents for the four study companies,
     collapsed to one record per application with three dates each.
  2. Build the NAIVE-BASELINE yearly + quarterly count series.
  3. Measure filing->publication and filing->grant LAG (whitepaper finding #1).
  4. Render whitepaper-grade figures (filing rhythm raw+normalized; 3-date overlay).
  5. Write raw + summary CSVs to _cache/ and outputs/.

NO signal processing, NO model, NO forecasting in this step by design.

Usage:
  python run.py                 # query BigQuery (needs Google auth), then cache
  python run.py --use-cache     # skip BigQuery, load _cache/raw_patents.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import argparse
import logging

import pandas as pd

from qvm import config
from qvm.data.base import CompanyPatents, PatentRecord
from qvm.analysis import timeseries, lag
from qvm.viz import plots

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("qvm.run")

RAW_CSV = os.path.join(config.CACHE_DIR, "raw_patents.csv")
BASELINE_CSV = os.path.join(config.OUTPUT_DIR, "naive_baseline_series.csv")
LAG_SUMMARY_CSV = os.path.join(config.OUTPUT_DIR, "lag_summary.csv")
LAG_CONSISTENCY_CSV = os.path.join(config.OUTPUT_DIR, "lag_cross_company.csv")
ASSIGNEE_AUDIT_CSV = os.path.join(config.OUTPUT_DIR, "assignee_match_audit.csv")

_RAW_FIELDS = ["ticker", "label", "name_like", "patent_id", "assignee_id",
               "assignee_org", "filing_date", "publication_date", "grant_date",
               "application_id", "priority_date"]


# --------------------------------------------------------------------------- #
# raw-data round-trip
# --------------------------------------------------------------------------- #
def save_raw(companies: list[CompanyPatents]) -> None:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    rows = []
    for c in companies:
        nl = c.name_variants[0] if c.name_variants else ""
        for r in c.records:
            rows.append({
                "ticker": c.ticker, "label": c.label, "name_like": nl,
                "patent_id": r.patent_id, "assignee_id": r.assignee_id,
                "assignee_org": r.assignee_org, "filing_date": r.filing_date,
                "publication_date": r.publication_date, "grant_date": r.grant_date,
                "application_id": r.application_id, "priority_date": r.priority_date,
            })
    pd.DataFrame(rows, columns=_RAW_FIELDS).to_csv(RAW_CSV, index=False)
    logger.info("wrote raw patents -> %s (%d rows)", RAW_CSV, len(rows))


def load_raw() -> list[CompanyPatents]:
    df = pd.read_csv(RAW_CSV, dtype=str)
    g = lambda v: v if isinstance(v, str) and v else None
    companies = []
    for (ticker, label), grp in df.groupby(["ticker", "label"], sort=False):
        nl = grp["name_like"].iloc[0]
        records, audit, n_pub = [], {}, 0
        for _, r in grp.iterrows():
            pub = g(r.get("publication_date"))
            aid = g(r.get("assignee_id"))
            org = g(r.get("assignee_org"))
            if pub:
                n_pub += 1
            if org:
                audit[org] = audit.get(org, 0) + 1
            records.append(PatentRecord(
                patent_id=r["patent_id"], assignee_id=aid, assignee_org=org,
                filing_date=g(r.get("filing_date")),
                publication_date=pub, grant_date=g(r.get("grant_date")),
                application_id=g(r.get("application_id")),
                priority_date=g(r.get("priority_date"))))
        companies.append(CompanyPatents(
            ticker=ticker, label=label, name_variants=[nl], records=records,
            matched_assignee_ids=dict(sorted(audit.items(),
                                             key=lambda kv: kv[1], reverse=True)),
            n_publication_matched=n_pub))
    return companies


def fetch_live() -> list[CompanyPatents]:
    from qvm.data.bigquery_provider import BigQueryPatentProvider
    return BigQueryPatentProvider().fetch_all()


# --------------------------------------------------------------------------- #
def write_assignee_audit(companies: list[CompanyPatents]) -> None:
    rows = []
    for c in companies:
        for name, count in c.matched_assignee_ids.items():
            rows.append({"ticker": c.ticker, "label": c.label,
                         "harmonized_name": name, "application_count": count})
    pd.DataFrame(rows).to_csv(ASSIGNEE_AUDIT_CSV, index=False)
    logger.info("wrote assignee audit -> %s", ASSIGNEE_AUDIT_CSV)


def main():
    ap = argparse.ArgumentParser(description="QVM-V2 step 1: data + raw viz (BigQuery).")
    ap.add_argument("--use-cache", action="store_true",
                    help="load _cache/raw_patents.csv instead of querying BigQuery")
    args = ap.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    if args.use_cache:
        if not os.path.exists(RAW_CSV):
            logger.error("no cache at %s; run without --use-cache first", RAW_CSV)
            return 1
        logger.info("loading cached raw patents from %s", RAW_CSV)
        companies = load_raw()
    else:
        try:
            companies = fetch_live()
        except Exception as e:  # auth / network / quota
            logger.error("\nBigQuery fetch failed: %s\n", e)
            logger.error("Check Google auth (see README 'BigQuery auth') and "
                         "that the project has BigQuery enabled.")
            return 2
        save_raw(companies)

    baseline = timeseries.build_all_baselines(companies)
    baseline.to_csv(BASELINE_CSV, index=False)
    logger.info("wrote naive-baseline series -> %s (%d rows)", BASELINE_CSV, len(baseline))

    lag_sum = lag.lag_summary(companies)
    lag_sum.to_csv(LAG_SUMMARY_CSV, index=False)
    consistency = lag.cross_company_consistency(lag_sum)
    consistency.to_csv(LAG_CONSISTENCY_CSV, index=False)
    logger.info("wrote lag summary -> %s", LAG_SUMMARY_CSV)

    write_assignee_audit(companies)

    order = [c["ticker"] for c in config.COMPANIES]
    raw_path, norm_path = plots.plot_filing_companies(baseline, config.OUTPUT_DIR,
                                                      company_order=order)
    fig2 = plots.plot_three_dates(baseline, config.OUTPUT_DIR, ticker="NVDA")
    logger.info("wrote figures -> %s | %s | %s", raw_path, norm_path, fig2)

    _print_summary(companies, baseline, lag_sum, consistency)
    return 0


def _print_summary(companies, baseline, lag_sum, consistency):
    print("\n" + "=" * 72)
    print("QVM-V2 STEP 1 — SUMMARY (source: BigQuery patents-public-data)")
    print("=" * 72)
    yearly = baseline[(baseline["granularity"] == "year")
                      & (baseline["date_dimension"] == "filing_date")]
    for c in companies:
        rate = (c.n_publication_matched / c.n_patents) if c.n_patents else 0
        print(f"\n{c.label} ({c.ticker}): {c.n_patents} applications | "
              f"{c.n_publication_matched} with pre-grant pub ({rate:.0%})")
        cy = yearly[yearly["ticker"] == c.ticker]
        complete = cy[~cy["incomplete"]] if "incomplete" in cy else cy
        if not complete.empty:
            peak = complete.loc[complete["patent_count"].idxmax()]
            print(f"  filing peak year (complete): {int(peak['year'])} "
                  f"= {int(peak['patent_count'])} filings")
        print(f"  top harmonized names: {list(c.matched_assignee_ids.items())[:3]}")
    print("\n--- LAG (months) ---")
    if not lag_sum.empty:
        print(lag_sum[["label", "lag_type", "n_used", "mean", "median",
                       "p25", "p75", "publication_match_rate"]].to_string(index=False))
    print("\n--- cross-company lag consistency ---")
    if not consistency.empty:
        print(consistency.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
