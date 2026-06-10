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
from qvm.analysis import timeseries, lag, channels
from qvm.viz import plots

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("qvm.run")

RAW_CSV = os.path.join(config.CACHE_DIR, "raw_patents.csv")
BASELINE_CSV = os.path.join(config.OUTPUT_DIR, "naive_baseline_series.csv")
LAG_SUMMARY_CSV = os.path.join(config.OUTPUT_DIR, "lag_summary.csv")
LAG_CONSISTENCY_CSV = os.path.join(config.OUTPUT_DIR, "lag_cross_company.csv")
ASSIGNEE_AUDIT_CSV = os.path.join(config.OUTPUT_DIR, "assignee_match_audit.csv")
CHANNELS_CSV = os.path.join(config.OUTPUT_DIR, "channels.csv")
FIN_DETAIL_CSV = os.path.join(config.OUTPUT_DIR, "financials_quarterly.csv")
Q4_SANITY_CSV = os.path.join(config.OUTPUT_DIR, "q4_derivation_sanity.csv")
YF_CROSSCHECK_CSV = os.path.join(config.OUTPUT_DIR, "yfinance_crosscheck.csv")
REGIME_PROBS_CSV = os.path.join(config.OUTPUT_DIR, "regime_probabilities.csv")
HMM_PARAMS_CSV = os.path.join(config.OUTPUT_DIR, "hmm_parameters.csv")
REGIME_SWITCHES_CSV = os.path.join(config.OUTPUT_DIR, "regime_switches.csv")

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
    from qvm.data.bigquery_provider import _is_excluded
    df = pd.read_csv(RAW_CSV, dtype=str)
    g = lambda v: v if isinstance(v, str) and v else None
    spec_by_ticker = {s["ticker"]: s for s in config.COMPANIES}
    companies = []
    for (ticker, label), grp in df.groupby(["ticker", "label"], sort=False):
        nl = grp["name_like"].iloc[0]
        # The cache may predate an exclude_name_like rule -> re-apply it here
        # so cached and live pulls produce the identical universe. The rows we
        # drop are counted into excluded_assignees for the audit CSV.
        spec = spec_by_ticker.get(ticker, {})
        n_before = len(grp)
        upper = grp["assignee_org"].fillna("").str.upper()
        excl_mask = upper.map(lambda u: _is_excluded(spec, u))
        excluded = {
            name: int(cnt) for name, cnt in
            grp.loc[excl_mask, "assignee_org"].value_counts().items()
        }
        grp = grp[~excl_mask]
        if excluded:
            logger.info("[%s] cache exclusion filter: %d -> %d applications "
                        "(%d excluded across %d assignee names)", label,
                        n_before, len(grp), n_before - len(grp), len(excluded))
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
            n_publication_matched=n_pub,
            excluded_assignees=excluded))
    return companies


def fetch_live() -> list[CompanyPatents]:
    from qvm.data.bigquery_provider import BigQueryPatentProvider
    return BigQueryPatentProvider().fetch_all()


# --------------------------------------------------------------------------- #
def write_assignee_audit(companies: list[CompanyPatents]) -> None:
    """Audit CSV: every harmonized name we MATCHED, then every name our LIKE
    would have captured but an exclude_name_like rule REJECTED (with reason).
    """
    spec_by_ticker = {s["ticker"]: s for s in config.COMPANIES}

    def _reason(spec: dict, name: str) -> str:
        up = (name or "").upper()
        for e in spec.get("exclude_name_like", []):
            if up.startswith(e["prefix"].rstrip("%").upper()):
                return f"excluded by {e['prefix']}: {e['reason']}"
        if spec.get("include_name_like"):
            return ("not in the curated entity allowlist (include_name_like) "
                    "— unrelated company caught by the broad prefix")
        return ""

    rows = []
    for c in companies:
        for name, count in c.matched_assignee_ids.items():
            rows.append({"ticker": c.ticker, "label": c.label,
                         "section": "matched", "harmonized_name": name,
                         "application_count": count, "reason": ""})
        spec = spec_by_ticker.get(c.ticker, {})
        for name, count in c.excluded_assignees.items():
            rows.append({"ticker": c.ticker, "label": c.label,
                         "section": "excluded", "harmonized_name": name,
                         "application_count": count,
                         "reason": _reason(spec, name)})
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

    # ---- STEP 2: financial channel (SEC EDGAR) + channel alignment grid ----
    financials = run_financials()
    if financials:
        ch = channels.build_channels(companies, financials)
        ch.to_csv(CHANNELS_CSV, index=False)
        logger.info("wrote channel grid -> %s (%d rows)", CHANNELS_CSV, len(ch))
        detail = channels.financials_detail_table(financials)
        detail.to_csv(FIN_DETAIL_CSV, index=False)
        sanity = pd.DataFrame([{"ticker": t, **r}
                               for t, f in financials.items() for r in f.q4_sanity])
        sanity.to_csv(Q4_SANITY_CSV, index=False)
        logger.info("wrote financial detail -> %s | q4 sanity -> %s",
                    FIN_DETAIL_CSV, Q4_SANITY_CSV)
        _print_step2_summary(financials, ch, sanity)
        run_yfinance_crosscheck(financials)
        fig3 = plots.plot_channels_eyetest(ch, config.OUTPUT_DIR,
                                           company_order=order)
        logger.info("wrote figure -> %s", fig3)

    # ---- STEP 3: patent-only NB-HMM baseline (single-channel regimes) ----
    run_regimes(companies)
    return 0


def run_regimes(companies) -> None:
    from qvm.analysis import regimes
    results = regimes.analyze_all(companies)

    regimes.probabilities_table(results).to_csv(REGIME_PROBS_CSV, index=False)
    params = regimes.parameters_table(results)
    params.to_csv(HMM_PARAMS_CSV, index=False)
    switches = regimes.switches_table(results)
    switches.to_csv(REGIME_SWITCHES_CSV, index=False)
    logger.info("wrote regime outputs -> %s | %s | %s",
                REGIME_PROBS_CSV, HMM_PARAMS_CSV, REGIME_SWITCHES_CSV)
    for res in results:
        logger.info("wrote figure -> %s", plots.plot_regimes(res, config.OUTPUT_DIR))

    agree = regimes.implementation_agreement(results)
    print("\n" + "=" * 72)
    print("QVM-V2 STEP 3 — PATENT-ONLY NB-HMM BASELINE (single-channel regimes)")
    print("=" * 72)
    print(f"\nNOTE: {regimes.POINT_IN_TIME_CAVEAT}")
    print("\n--- parameters (2-state NB; states ordered low->high) ---")
    nb2 = params[params["model"] == "nb2"]
    print(nb2[["ticker", "mus", "dispersion_r", "expected_durations_q",
               "restarts_at_best_logl", "restart_logl_spread",
               "em_converged"]].to_string(index=False))
    print("\n--- model comparison (BIC; negative delta favours first) ---")
    print(agree.to_string(index=False))
    print("\n--- persistent filtered switches (>=0.5 for >=2 quarters) ---")
    print(switches[["ticker", "quarter", "direction",
                    "p_high_at_switch"]].to_string(index=False))
    for res in results:
        print(f"  [{res['ticker']}] {res['note']}")
    print("=" * 72)


def run_financials() -> dict:
    """Fetch (or load cached) SEC EDGAR quarterly financials per company."""
    from qvm.data.sec_edgar_provider import SECEdgarProvider
    provider = SECEdgarProvider()
    out = {}
    for spec in config.COMPANIES:
        try:
            out[spec["ticker"]] = provider.get_company_financials(spec["ticker"])
        except Exception as e:  # network / SEC outage: patents still usable
            logger.error("SEC EDGAR fetch failed for %s: %s", spec["ticker"], e)
            return {}
    return out


def run_yfinance_crosscheck(financials: dict) -> None:
    """Sanity-only: recent yfinance quarters vs SEC latest values."""
    from qvm.data.yfinance_provider import YFinanceProvider, cross_check
    try:
        yf_provider = YFinanceProvider()
        records = []
        for ticker, sec_fin in financials.items():
            records.extend(cross_check(sec_fin, yf_provider.get_company_financials(ticker)))
    except Exception as e:  # yfinance is best-effort; never block the pipeline
        logger.warning("yfinance cross-check unavailable: %s", e)
        return
    df = pd.DataFrame(records)
    df.to_csv(YF_CROSSCHECK_CSV, index=False)
    bad = df[df["status"] == "MISMATCH"] if not df.empty else df
    print(f"\n--- yfinance cross-check: {len(df)} comparisons, "
          f"{len(bad)} mismatches (>2%) -> {os.path.basename(YF_CROSSCHECK_CSV)} ---")
    if not bad.empty:
        print(bad[["ticker", "concept", "yf_end", "sec_end", "yf_value",
                   "sec_value_latest", "pct_diff"]].to_string(index=False))


def _print_step2_summary(financials: dict, ch: pd.DataFrame, sanity: pd.DataFrame):
    print("\n" + "=" * 72)
    print("QVM-V2 STEP 2 — FINANCIAL CHANNEL + ALIGNMENT GRID (source: SEC EDGAR)")
    print("=" * 72)
    print("\n--- channel coverage (calendar quarters, %s onward) ---"
          % config.CHANNELS_START_QUARTER)
    print(channels.channels_summary(ch).to_string(index=False))

    print("\n--- revenue tag provenance (which us-gaap tag covered which years) ---")
    for ticker, fin in financials.items():
        for concept in ("revenue", "cost_of_revenue", "gross_profit", "operating_income"):
            for p in fin.tag_provenance.get(concept, []):
                print(f"  {ticker:5s} {concept:17s} {p['tag']:55s} "
                      f"{p['n_periods']:3d} periods  {p['first_end']} .. {p['last_end']}")
        for note in fin.notes:
            print(f"  {ticker:5s} NOTE: {note}")

    if not sanity.empty:
        flagged = sanity[sanity["status"] != "ok"]
        derived = sanity[sanity["derived"]]
        print(f"\n--- Q4 derivation sanity: {len(derived)} fiscal years derived, "
              f"{len(flagged)} flagged ---")
        if not flagged.empty:
            print(flagged[["ticker", "concept", "fy_end", "n_quarters_found",
                           "status"]].to_string(index=False))

    restated = [(t, sum(o.restated for o in f.observations))
                for t, f in financials.items()]
    print("\n--- restatements (as_filed != latest): "
          + ", ".join(f"{t}={n}" for t, n in restated) + " ---")
    print("=" * 72)


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
