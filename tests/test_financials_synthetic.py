"""Financial-channel checks on SYNTHETIC XBRL companyfacts data (no network).

Validates: duration classification, per-period tag-priority merge, fuzzy
period matching (refiled boundaries), as_filed vs latest (restatements),
Q4 derivation + direct-Q4 demotion, calendar-quarter alignment of offset
fiscal years (the NVDA late-January trap), and the channel grid build.

Run:  ./venv/bin/python tests/test_financials_synthetic.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd

from qvm.analysis.channels import build_channels, calendar_quarter
from qvm.data.base import CompanyPatents, PatentRecord
from qvm.data.sec_edgar_provider import (
    _classify_duration, derive_q4, extract_company, extract_concept_periods)


def _fact(start, end, val, filed, form="10-Q", fy=2024, fp="Q1"):
    return {"start": start, "end": end, "val": val, "filed": filed,
            "form": form, "fy": fy, "fp": fp, "accn": filed}


def test_duration():
    assert _classify_duration("2024-01-01", "2024-03-31") == "quarter"
    assert _classify_duration("2024-01-01", "2024-12-31") == "annual"
    assert _classify_duration("2024-01-01", "2024-06-30") is None  # 6mo YTD
    assert _classify_duration("2024-01-01", "2024-09-30") is None  # 9mo YTD
    print("OK  duration classification (quarter / annual / YTD skipped)")


# An offset fiscal year like NVIDIA's: FY ends late January.
# Q1 Feb-Apr, Q2 May-Jul (REFILED with end shifted one day), Q3 Aug-Oct,
# FY in the 10-K, PLUS a direct Q4 fact that surfaces in a LATER filing.
_GAAP = {
    "Revenues": {"units": {"USD": [
        _fact("2023-01-30", "2023-04-30", 100.0, "2023-05-20", fp="Q1"),
        _fact("2023-05-01", "2023-07-30", 110.0, "2023-08-20", fp="Q2"),
        # same Q2 refiled a year later with boundary moved one day AND restated
        _fact("2023-05-01", "2023-07-31", 111.0, "2024-08-20", fp="Q2"),
        _fact("2023-07-31", "2023-10-29", 120.0, "2023-11-20", fp="Q3"),
        _fact("2023-01-30", "2024-01-28", 500.0, "2024-02-21", form="10-K", fp="FY"),
        # direct Q4 fact, first published TWO YEARS later in a comparative
        _fact("2023-10-30", "2024-01-28", 169.0, "2026-02-21", fp="Q4"),
    ]}},
    "GrossProfit": {"units": {"USD": [
        _fact("2023-01-30", "2023-04-30", 60.0, "2023-05-20", fp="Q1"),
        _fact("2023-05-01", "2023-07-30", 66.0, "2023-08-20", fp="Q2"),
        _fact("2023-07-31", "2023-10-29", 72.0, "2023-11-20", fp="Q3"),
        _fact("2023-01-30", "2024-01-28", 300.0, "2024-02-21", form="10-K", fp="FY"),
    ]}},
}


def test_extraction_and_q4():
    quarterly, annual, prov = extract_concept_periods([_GAAP], "revenue")
    # fuzzy match folded the refiled Q2 into ONE period (not five quarters)
    assert len(quarterly) == 4, sorted(quarterly)  # Q1 Q2 Q3 + direct Q4
    assert len(annual) == 1
    q2 = next(o for o in quarterly.values() if o["start"] == "2023-05-01")
    assert q2["value"] == 110.0 and q2["filed"] == "2023-08-20"   # as filed
    assert q2["value_latest"] == 111.0                            # restated
    assert prov[0]["tag"] == "Revenues" and prov[0]["n_periods"] == 5

    derived, sanity, drops = derive_q4(quarterly, annual, "revenue")
    assert len(derived) == 1 and len(drops) == 1
    q4 = derived[0]
    assert q4["value"] == 500.0 - (100 + 110 + 120) == 170.0      # as-filed math
    assert q4["value_latest"] == 500.0 - (100 + 111 + 120)        # uses restated Q2
    assert q4["filed"] == "2024-02-21"  # knowable at the 10-K, NOT 2026
    rec = next(r for r in sanity if r["derived"])
    assert rec["deviation_pct"] is not None  # direct (169) vs derived compared
    print("OK  fuzzy merge + as_filed/latest + Q4 derived (direct Q4 demoted)")


def test_company_and_channels():
    fin = extract_company([_GAAP], "TEST", cik=1)
    rev = fin.quarterly("revenue")
    assert [o.end for o in rev] == ["2023-04-30", "2023-07-30", "2023-10-29",
                                    "2024-01-28"]
    assert rev[-1].derived_q4 and rev[-1].value == 170.0

    # the fiscal-year trap: end 2024-01-28 must land in calendar 2023Q4
    assert str(calendar_quarter("2024-01-28")) == "2023Q4"
    # a Feb-Apr fiscal quarter (ends Apr 30) is mostly Q1 -> calendar Q1
    assert str(calendar_quarter("2023-04-30")) == "2023Q1"
    assert str(calendar_quarter("2024-12-31")) == "2024Q4"

    pats = CompanyPatents(
        ticker="TEST", label="Test", name_variants=["TEST%"],
        records=[PatentRecord(patent_id=str(i), assignee_id="T", assignee_org="T",
                              filing_date=f"202{y}-0{m}-15", publication_date=None,
                              grant_date=None, application_id=str(i))
                 for i, (y, m) in enumerate([(3, 2), (3, 2), (3, 6), (4, 2)])])
    ch = build_channels([pats], {"TEST": fin})
    row = ch[ch.quarter == "2023Q4"].iloc[0]
    assert row["revenue"] == 170.0 and row["revenue_is_derived_q4"]
    assert row["revenue_knowable_at"] == "2024-02-21"
    assert row["gross_margin"] is not None  # GrossProfit tag path
    q1 = ch[ch.quarter == "2023Q1"].iloc[0]
    assert q1["patent_filing_count"] == 2 and q1["revenue"] == 100.0
    assert pd.isna(ch[ch.quarter == "2022Q4"].iloc[0]["revenue"])  # pre-history
    # trailing patent years flagged incomplete even where counts are zero
    assert ch[ch.quarter == "2023Q3"].iloc[0]["patent_incomplete"]
    print("OK  extract_company + calendar alignment + channel grid")


def main():
    test_duration()
    test_extraction_and_q4()
    test_company_and_channels()
    print("\nALL FINANCIAL SYNTHETIC CHECKS PASSED")


if __name__ == "__main__":
    main()
