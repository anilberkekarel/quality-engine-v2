"""End-to-end pipeline check on SYNTHETIC BigQuery-shaped data (no auth needed).

Validates: yyyymmdd parsing, per-publication -> per-application collapse,
assignee prefix routing (incl. false-positive rejection), yearly+quarterly
naive baseline with incomplete-tail flagging, lag, and figures.

Run:  ./venv/bin/python tests/test_pipeline_synthetic.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import datetime as dt
import tempfile

import pandas as pd

from qvm.data.bigquery_provider import (
    _parse_yyyymmdd, _assign_company, rows_to_companies, build_query)
from qvm.analysis import timeseries, lag
from qvm.viz import plots


def _ymd(d: dt.date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def _company_rows(harmonized_name, n, base_year=2010, span=15):
    """Emit per-publication rows: each application yields an A-row + B-row."""
    rows = []
    for i in range(n):
        year = base_year + int((i / n) * span)
        month = 1 + (i % 12)
        filing = dt.date(year, month, 1)
        pub = filing + dt.timedelta(days=545)   # ~18 months pre-grant pub
        grant = filing + dt.timedelta(days=910)  # ~30 months grant
        app = f"US-{harmonized_name[:4]}-{i:06d}"
        # A-row: pre-grant publication, grant_date = 0
        rows.append(dict(application_number=app, assignee_name=harmonized_name,
                         country_code="US", filing_date=_ymd(filing),
                         publication_date=_ymd(pub), grant_date=0,
                         priority_date=_ymd(filing)))
        # B-row: granted document, grant_date > 0
        rows.append(dict(application_number=app, assignee_name=harmonized_name,
                         country_code="US", filing_date=_ymd(filing),
                         publication_date=_ymd(grant), grant_date=_ymd(grant),
                         priority_date=_ymd(filing)))
    return rows


def test_parser():
    assert _parse_yyyymmdd(20180315) == "2018-03-15"
    assert _parse_yyyymmdd(0) is None          # not granted sentinel
    assert _parse_yyyymmdd(None) is None
    assert _parse_yyyymmdd(20189999) is None   # invalid month/day
    print("OK  yyyymmdd parser (valid / sentinel / invalid)")


def test_assign_and_query():
    assert _assign_company("NVIDIA CORP")["ticker"] == "NVDA"
    assert _assign_company("ADVANCED MICRO DEVICES INC")["ticker"] == "AMD"
    assert _assign_company("CONVIDIA CORP") is None       # false positive rejected
    assert _assign_company("INVIDIATO COSMO L") is None   # false positive rejected
    # exclude_name_like: unrelated companies inside the MICRON% net
    assert _assign_company("MICRON TECHNOLOGY INC")["ticker"] == "MU"
    assert _assign_company("MICRONAS GMBH") is None           # excluded (German)
    assert _assign_company("MICRONIC LASER SYSTEMS AB") is None  # excluded (Swedish)
    # include_name_like allowlist: small unrelated companies rejected,
    # Micron-entity typos kept
    assert _assign_company("MICRON DEVICES LLC") is None      # FL medical devices
    assert _assign_company("MICRON OPTICS INC") is None       # Atlanta fiber sensing
    assert _assign_company("MICRON ELETRONICS INC")["ticker"] == "MU"  # typo of the PC sub
    assert _assign_company("MICRON TEHNOLOGY INC")["ticker"] == "MU"   # typo of the parent
    sql, params = build_query()
    assert "p.country_code = @country" in sql and "UNNEST(p.assignee_harmonized)" in sql
    assert "p.country_code AS country_code" in sql  # aliasing avoids ambiguity
    assert len([p for p in params if p.name.startswith("like")]) == 4
    assert "NOT LIKE @excl" in sql  # exclusions pushed into SQL for live pulls
    print("OK  assignee routing + exclusions + parameterized query build")


def main():
    test_parser()
    test_assign_and_query()

    rows = []
    rows += _company_rows("NVIDIA CORP", 1800)
    rows += _company_rows("ADVANCED MICRO DEVICES INC", 1200)
    rows += _company_rows("MARVELL ASIA PTE LTD", 700)
    rows += _company_rows("MICRON TECHNOLOGY INC", 1500)
    rows += _company_rows("CONVIDIA CORP", 50)  # noise that must be dropped
    rows += _company_rows("MICRONAS GMBH", 40)  # excluded false positive
    df = pd.DataFrame(rows)

    companies = rows_to_companies(df)
    by = {c.ticker: c for c in companies}
    # collapse worked: 1800 applications, not 3600 publication rows
    assert by["NVDA"].n_patents == 1800, by["NVDA"].n_patents
    assert by["MU"].n_patents == 1500  # MICRONAS not counted...
    assert by["MU"].excluded_assignees == {"MICRONAS GMBH": 40}  # ...but audited
    # noise rejected: only the 4 real companies present
    assert all(c.ticker in {"NVDA", "AMD", "MRVL", "MU"} for c in companies)
    # three dates recovered per application
    r = by["NVDA"].records[0]
    assert r.filing_date and r.publication_date and r.grant_date
    print(f"OK  collapse: NVDA={by['NVDA'].n_patents} apps "
          f"(from {len(df[df.assignee_name=='NVIDIA CORP'])} pub rows), noise dropped")

    baseline = timeseries.build_all_baselines(companies)
    assert set(baseline["granularity"]) == {"year", "quarter"}
    assert (baseline["series_kind"] == "naive_baseline").all()
    inc = baseline[(baseline.granularity == "year")
                   & (baseline.date_dimension == "filing_date") & baseline.incomplete]
    assert not inc.empty, "incomplete trailing years not flagged"
    print(f"OK  baseline: yearly+quarterly, {inc['year'].nunique()} incomplete filing years flagged")

    lag_sum = lag.lag_summary(companies)
    fg = lag_sum[lag_sum.lag_type == "filing_to_grant"]["mean"].mean()
    fp = lag_sum[lag_sum.lag_type == "filing_to_publication"]["mean"].mean()
    assert 28 < fg < 32 and 16 < fp < 20, (fp, fg)
    print(f"OK  lag: filing->pub ~{fp:.1f}mo, filing->grant ~{fg:.1f}mo")

    with tempfile.TemporaryDirectory() as tmp:
        raw_p, norm_p = plots.plot_filing_companies(baseline, tmp,
                                                    company_order=["NVDA", "AMD", "MRVL", "MU"])
        f2 = plots.plot_three_dates(baseline, tmp, ticker="NVDA")
        for p in (raw_p, norm_p, f2):
            assert os.path.exists(p) and os.path.getsize(p) > 5000
        print("OK  figures: fig1 raw + fig1 normalized + fig2 three-dates rendered")

    print("\nALL SYNTHETIC PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
