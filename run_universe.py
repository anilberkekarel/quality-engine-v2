"""QVM-V2 STEP 5a entry point — UNIVERSE PLUMBING ONLY (no model).

Scales the data layer to the ~50-company semiconductor universe defined in
qvm.universe_registry (SIC 3674 + size/category/depth screens, delisted
members included). The four case companies keep their Step 1-4 outputs
untouched — this script writes universe_* artifacts only.

Pipeline:
  1. registry -> outputs/universe_registry.csv (included + excluded, reasons)
  2. patents: ONE consolidated BigQuery query (dry-run guarded), vectorized
     per-application collapse, semi-automatic assignee audit (>=20-app names
     tagged REVIEW — researcher decides, nothing auto-excluded)
  3. financials: SEC companyfacts per registry CIK list (cached, rate-limited)
     — unresolved revenue mappings reported loudly, never silently empty
  4. ma_events.csv — manually curated M&A annotation layer (non-exhaustive)
  5. data-quality table + channels_universe.csv

Usage:
  python run_universe.py               # live BigQuery + SEC (both cached)
  python run_universe.py --use-cache   # reuse _cache/universe_raw_patents.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import argparse
import logging

import numpy as np
import pandas as pd

from qvm import config
from qvm.universe_registry import (EXCLUDED_SCREEN, REATTRIBUTIONS, UNIVERSE,
                                   apply_reattributions, universe_specs)
from qvm.data.base import CompanyPatents, PatentRecord
from qvm.data.bigquery_provider import _assign_company_audited, build_query
from qvm.analysis import channels

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("qvm.universe")

UNIVERSE_RAW_CSV = os.path.join(config.CACHE_DIR, "universe_raw_patents.csv")
REGISTRY_CSV = os.path.join(config.OUTPUT_DIR, "universe_registry.csv")
AUDIT_CSV = os.path.join(config.OUTPUT_DIR, "universe_assignee_audit.csv")
MA_CSV = os.path.join(config.OUTPUT_DIR, "ma_events.csv")
QUALITY_CSV = os.path.join(config.OUTPUT_DIR, "universe_data_quality.csv")
CHANNELS_CSV = os.path.join(config.OUTPUT_DIR, "channels_universe.csv")
REATTRIBUTION_CSV = os.path.join(config.OUTPUT_DIR, "reattribution_log.csv")

REVIEW_THRESHOLD = 20  # >=20-app assignee names are tagged REVIEW (user decides)


# --------------------------------------------------------------------------- #
# M&A annotation layer (Görev D) — NOT a model input.
# --------------------------------------------------------------------------- #
MA_EVENTS = [  # (ticker, date, event) — manually curated, non-exhaustive
    # acquirer-side rows mirror every in-universe closing: the Step-5b ±2Q
    # robustness exclusion must blank BOTH sides of a deal window.
    ("TXN",  "2011-09", "acquired National Semiconductor (NSM exits)"),
    ("QCOM", "2011-05", "acquired Atheros (ATHR exits; patent composition)"),
    ("ATHR", "2011-05", "acquired by Qualcomm (exits; QCOM member by amendment)"),
    ("MCHP", "2012-08", "acquired Standard Microsystems"),
    ("MU",   "2013-07", "acquired Elpida Memory (patent composition)"),
    ("LSI",  "2014-05", "acquired by Avago (exits)"),
    ("IRF",  "2015-01", "acquired by Infineon (exits; acquirer outside universe)"),
    ("QRVO", "2015-01", "formed by RFMD + TriQuint merger (RFMD/TQNT exit)"),
    ("CODE", "2015-03", "Spansion merged into Cypress (exits)"),
    ("CY",   "2015-03", "Spansion merged in (CODE exits; composition)"),
    ("ALTR", "2015-12", "acquired by Intel (exits)"),
    ("INTC", "2015-12", "acquired Altera (ALTR exits; patent composition)"),
    ("FSL",  "2015-12", "acquired by NXP (exits)"),
    ("NXPI", "2015-12", "acquired Freescale (FSL exits; patent composition)"),
    ("OVTI", "2016-01", "acquired by Chinese consortium (exits)"),
    ("PMCS", "2016-01", "acquired by Microsemi (exits)"),
    ("MSCC", "2016-01", "acquired PMC-Sierra (PMCS exits; composition)"),
    ("BRCM", "2016-02", "acquired by Avago; Avago renamed Broadcom (exits)"),
    ("AVGO", "2016-02", "acquired Broadcom Corp (BRCM exits; composition)"),
    ("AVGO", "2014-05", "acquired LSI (LSI exits; patent composition)"),
    ("ATML", "2016-04", "acquired by Microchip (exits)"),
    ("MCHP", "2016-04", "acquired Atmel (ATML exits; patent composition)"),
    ("FCS",  "2016-09", "acquired by onsemi (exits)"),
    ("ON",   "2016-09", "acquired Fairchild (FCS exits; patent composition)"),
    ("ISIL", "2017-02", "acquired by Renesas (exits; acquirer outside universe)"),
    ("LLTC", "2017-03", "acquired by Analog Devices (exits)"),
    ("ADI",  "2017-03", "acquired Linear (LLTC exits; patent composition)"),
    ("MXL",  "2017-05", "acquired Exar"),
    ("MSCC", "2018-05", "acquired by Microchip (exits)"),
    ("MCHP", "2018-05", "acquired Microsemi (MSCC exits; composition)"),
    ("MRVL", "2018-07", "acquired Cavium (CAVM exits; patent composition)"),
    ("IDTI", "2019-03", "acquired by Renesas (exits; acquirer outside universe)"),
    ("NVDA", "2020-04", "acquired Mellanox (MLNX exits; patent composition)"),
    ("CY",   "2020-04", "acquired by Infineon (exits; acquirer outside universe)"),
    ("ALGM", "2020-10", "IPO (financial series starts; patents reach back)"),
    ("MRVL", "2021-04", "acquired Inphi (IPHI exits; patent composition)"),
    ("SWKS", "2021-07", "acquired Silicon Labs infrastructure & automotive unit"),
    ("SLAB", "2021-07", "sold infrastructure & automotive unit to Skyworks"),
    ("MXIM", "2021-08", "acquired by Analog Devices (exits)"),
    ("ADI",  "2021-08", "acquired Maxim (MXIM exits; patent composition)"),
    ("WOLF", "2021-03", "CreeLED divested to SMART Global (patent composition)"),
    ("WOLF", "2021-10", "Cree renamed Wolfspeed (same CIK; assignee names change)"),
    ("AMD",  "2022-02", "acquired Xilinx (XLNX exits; patent composition)"),
    ("MTSI", "2023-08", "acquired Wolfspeed RF business (composition)"),
]


# --------------------------------------------------------------------------- #
# Görev A: registry CSV
# --------------------------------------------------------------------------- #
def write_registry(quality: dict[str, dict] | None = None) -> pd.DataFrame:
    sic_names = {}
    sic_path = os.path.join(config.CACHE_DIR, "sic3674_filers.json")
    if os.path.exists(sic_path):
        import json
        sic_names = {int(k): v for k, v in json.load(open(sic_path)).items()}
    rows = []
    for e in universe_specs():
        q = (quality or {}).get(e["ticker"], {})
        rows.append({
            "included": True, "ticker": e["ticker"], "label": e["label"],
            "ciks": ";".join(str(c) for c in e["ciks"]),
            "assignee_prefixes": ";".join(e.get("name_likes")
                                          or [e.get("name_like", "")]),
            "exclusions": ";".join(x["prefix"] for x in
                                   e.get("exclude_name_like", [])),
            "allowlist": ";".join(e.get("include_name_like", [])),
            "is_case_company": bool(e.get("case")),
            "delisted": e.get("delisted", ""),
            "first_fin_quarter": q.get("fin_first", ""),
            "last_fin_quarter": q.get("fin_last", ""),
            "flags": " | ".join(e.get("flags", [])),
            "notes": e.get("notes", ""),
        })
    for cik, (cat, reason) in EXCLUDED_SCREEN.items():
        rows.append({"included": False, "ticker": "", "label":
                     sic_names.get(cik, f"CIK {cik}"), "ciks": str(cik),
                     "assignee_prefixes": "", "exclusions": "",
                     "allowlist": "", "is_case_company": False,
                     "delisted": "", "first_fin_quarter": "",
                     "last_fin_quarter": "",
                     "flags": f"excluded: {cat}", "notes": reason})
    df = pd.DataFrame(rows)
    df.to_csv(REGISTRY_CSV, index=False)
    logger.info("wrote registry -> %s (%d included, %d screen-excluded)",
                REGISTRY_CSV, int(df["included"].sum()),
                int((~df["included"]).sum()))
    return df


# --------------------------------------------------------------------------- #
# Görev B: patents — one consolidated query + vectorized collapse + audit
# --------------------------------------------------------------------------- #
def _specs_missing_from_cache(df: pd.DataFrame, specs: list[dict]) -> list[dict]:
    """Specs whose assignee prefixes match NOTHING in the cache — i.e. members
    added to the registry after the cached pull (QCOM amendment)."""
    names = pd.Series(df["assignee_name"].dropna().unique()).str.upper()
    missing = []
    for s in specs:
        prefixes = [p.rstrip("%").upper()
                    for p in (s.get("name_likes") or [s.get("name_like", "")])]
        if not names.str.startswith(tuple(prefixes)).any():
            missing.append(s)
    return missing


def fetch_universe_rows(specs: list[dict], use_cache: bool) -> pd.DataFrame:
    if use_cache:
        if not os.path.exists(UNIVERSE_RAW_CSV):
            raise SystemExit(f"no cache at {UNIVERSE_RAW_CSV}; run live first")
        logger.info("loading cached universe patents from %s", UNIVERSE_RAW_CSV)
        df = pd.read_csv(UNIVERSE_RAW_CSV)
        missing = _specs_missing_from_cache(df, specs)
        if missing:
            logger.info("cache lacks rows for %s — fetching just those specs",
                        [s["ticker"] for s in missing])
            add = _fetch_live_rows(missing, cache_path=None)
            df = pd.concat([df, add], ignore_index=True)
            df.to_csv(UNIVERSE_RAW_CSV, index=False)
            logger.info("appended %d rows to %s", len(add), UNIVERSE_RAW_CSV)
        return df
    return _fetch_live_rows(specs, cache_path=UNIVERSE_RAW_CSV)


def _fetch_live_rows(specs: list[dict], cache_path: str | None) -> pd.DataFrame:
    from google.cloud import bigquery
    from qvm.data.bigquery_provider import BigQueryPatentProvider
    prov = BigQueryPatentProvider()
    sql, params = build_query(specs)
    dry = prov.client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params, dry_run=True, use_query_cache=False))
    gib = dry.total_bytes_processed / 1024**3
    logger.info("BigQuery dry-run (universe, ONE query): ~%.2f GiB "
                "(guard %.0f GiB)", gib, prov.max_bytes_billed / 1024**3)
    if dry.total_bytes_processed > prov.max_bytes_billed:
        raise SystemExit(f"dry-run {gib:.1f} GiB exceeds guard — STOPPING; "
                         "ask before raising MAX_BYTES_BILLED")
    job = prov.client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params, maximum_bytes_billed=prov.max_bytes_billed))
    df = job.result().to_dataframe(create_bqstorage_client=False)
    logger.info("BigQuery returned %d publication rows (%.2f GiB billed)",
                len(df), (job.total_bytes_billed or 0) / 1024**3)
    if cache_path:
        df.to_csv(cache_path, index=False)
    return df


def assign_rows(df: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Vectorized company assignment: classify each UNIQUE assignee name once."""
    uniq = pd.Series(df["assignee_name"].dropna().unique())
    assignment = {n: _assign_company_audited(n, specs) for n in uniq}
    df = df.copy()
    df["_ticker"] = df["assignee_name"].map(
        lambda n: (assignment.get(n, (None, False))[0] or {}).get("ticker")
        if not assignment.get(n, (None, False))[1] else None)
    df["_excl_ticker"] = df["assignee_name"].map(
        lambda n: (assignment.get(n, (None, False))[0] or {}).get("ticker")
        if assignment.get(n, (None, False))[1] else None)
    return df


def _dates_from_int(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")
    v = v.where(v > config.DATE_NOT_GRANTED_SENTINEL)
    return pd.to_datetime(v.astype("Int64").astype(str),
                          format="%Y%m%d", errors="coerce")


def collapse_to_companies(df: pd.DataFrame,
                          specs: list[dict]) -> list[CompanyPatents]:
    """Vectorized version of rows_to_companies (the 4-company loop version
    does not scale to ~1M publication rows)."""
    d = df[df["_ticker"].notna()].copy()
    for col in ("filing_date", "publication_date", "grant_date", "priority_date"):
        d["_" + col] = _dates_from_int(d[col])
    d["_grant_int"] = pd.to_numeric(d["grant_date"], errors="coerce").fillna(0)
    keys = ["_ticker", "application_number"]
    base = d.groupby(keys, sort=False).agg(
        filing=("_filing_date", "min"), grant=("_grant_date", "min"),
        priority=("_priority_date", "min"), org=("assignee_name", "first"))
    # pre-grant publication = earliest pub among NON-granted rows only
    pre = d[d["_grant_int"] <= config.DATE_NOT_GRANTED_SENTINEL]
    pub = pre.groupby(keys, sort=False)["_publication_date"].min().rename("pub")
    base = base.join(pub).reset_index()

    fmt = lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None
    out = []
    for spec in specs:
        sub = base[base["_ticker"] == spec["ticker"]]
        records = [PatentRecord(
            patent_id=str(r.application_number), assignee_id=None,
            assignee_org=r.org, filing_date=fmt(r.filing),
            publication_date=fmt(r.pub), grant_date=fmt(r.grant),
            application_id=str(r.application_number),
            priority_date=fmt(r.priority)) for r in sub.itertuples()]
        out.append(CompanyPatents(
            ticker=spec["ticker"], label=spec["label"],
            name_variants=spec.get("name_likes") or [spec.get("name_like", "")],
            records=records,
            n_publication_matched=int(sub["pub"].notna().sum())))
    return out


def write_audit(df: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Semi-automatic audit: matched + excluded names with app counts.
    Every matched name with >= REVIEW_THRESHOLD applications is tagged REVIEW
    — the researcher decides; this script excludes nothing on its own."""
    spec_by = {s["ticker"]: s for s in specs}
    rows = []
    for section, col in (("matched", "_ticker"), ("excluded", "_excl_ticker")):
        sub = df[df[col].notna()]
        counts = (sub.groupby([col, "assignee_name"])["application_number"]
                  .nunique().reset_index(name="application_count")
                  .rename(columns={col: "t"}))  # itertuples drops _-prefixed names
        for r in counts.itertuples():
            ticker, name, n = r.t, r.assignee_name, r.application_count
            reason = ""
            if section == "excluded":
                spec = spec_by.get(ticker, {})
                up = name.upper()
                for e in spec.get("exclude_name_like", []):
                    if up.startswith(e["prefix"].rstrip("%").upper()):
                        reason = f"excluded by {e['prefix']}: {e['reason']}"
                        break
                else:
                    reason = "not in entity allowlist"
            rows.append({
                "ticker": ticker, "section": section, "harmonized_name": name,
                "application_count": int(n),
                "review_status": ("REVIEW" if section == "matched"
                                  and n >= REVIEW_THRESHOLD else ""),
                "reason": reason})
    audit = pd.DataFrame(rows).sort_values(
        ["ticker", "section", "application_count"],
        ascending=[True, True, False]).reset_index(drop=True)
    audit.to_csv(AUDIT_CSV, index=False)
    n_rev = int((audit["review_status"] == "REVIEW").sum())
    logger.info("wrote assignee audit -> %s (%d names, %d tagged REVIEW)",
                AUDIT_CSV, len(audit), n_rev)
    return audit


# --------------------------------------------------------------------------- #
# Görev C: financials
# --------------------------------------------------------------------------- #
def fetch_universe_financials(specs: list[dict]) -> tuple[dict, list[dict]]:
    from qvm.data.sec_edgar_provider import SECEdgarProvider
    provider = SECEdgarProvider()
    fins, unresolved = {}, []
    for spec in specs:
        try:
            fin = provider.get_financials_for_ciks(spec["ticker"], spec["ciks"])
        except Exception as e:
            unresolved.append({"ticker": spec["ticker"],
                               "problem": f"companyfacts fetch failed: {e}"})
            continue
        if not fin.quarterly("revenue"):
            # never silently empty (the AMD lesson): list candidate tags
            import json
            cands = set()
            for c in spec["ciks"]:
                p = os.path.join(config.CACHE_DIR,
                                 f"sec_companyfacts_{spec['ticker']}_CIK{c:010d}.json")
                if os.path.exists(p):
                    gaap = json.load(open(p)).get("facts", {}).get("us-gaap", {})
                    cands |= {t for t in gaap if "REVENUE" in t.upper()
                              or t.upper().startswith("SALES")}
            unresolved.append({"ticker": spec["ticker"],
                               "problem": "UNRESOLVED revenue mapping",
                               "candidate_tags": sorted(cands)})
        fins[spec["ticker"]] = fin
    return fins, unresolved


# --------------------------------------------------------------------------- #
# Görev E: quality table + universe channel grid
# --------------------------------------------------------------------------- #
def quality_table(companies: list[CompanyPatents], fins: dict,
                  specs: list[dict]) -> pd.DataFrame:
    spec_by = {s["ticker"]: s for s in specs}
    rows = []
    for c in companies:
        fin = fins.get(c.ticker)
        fdates = pd.to_datetime([r.filing_date for r in c.records
                                 if r.filing_date], errors="coerce")
        pat_q = pd.Series(fdates).dropna().dt.to_period("Q")
        rev = fin.quarterly("revenue") if fin else []
        fin_first = str(channels.calendar_quarter(rev[0].end)) if rev else ""
        fin_last = str(channels.calendar_quarter(rev[-1].end)) if rev else ""
        ends = sorted({channels.calendar_quarter(o.end) for o in rev})
        missing = ((ends[-1] - ends[0]).n + 1 - len(ends)) if ends else None
        flags = list(spec_by[c.ticker].get("flags", []))
        if not pat_q.empty and rev:
            if pat_q.min() < pd.Period(fin_first, freq="Q") - 8:
                flags.append("fin series starts >2y after patents (XBRL "
                             "start or late IPO)")
            if pat_q.max() < pd.Period(fin_last, freq="Q") - 8:
                flags.append("patent series ends >2y before financials — "
                             "check assignee continuity / name change")
        rows.append({
            "ticker": c.ticker, "label": c.label,
            "patent_applications": c.n_patents,
            "patent_first_quarter": str(pat_q.min()) if not pat_q.empty else "",
            "patent_last_quarter": str(pat_q.max()) if not pat_q.empty else "",
            "fin_first_quarter": fin_first, "fin_last_quarter": fin_last,
            "fin_quarters": len(rev), "fin_missing_internal_quarters": missing,
            "restated_observations": (sum(o.restated for o in fin.observations)
                                      if fin else None),
            "delisted": spec_by[c.ticker].get("delisted", ""),
            "flags": " | ".join(flags),
        })
    df = pd.DataFrame(rows)
    df.to_csv(QUALITY_CSV, index=False)
    logger.info("wrote data-quality table -> %s", QUALITY_CSV)
    return df


def main():
    ap = argparse.ArgumentParser(description="QVM-V2 Step 5a: universe plumbing.")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    specs = universe_specs()
    logger.info("universe: %d companies (%d delisted)", len(specs),
                sum(1 for s in specs if s.get("delisted")))

    raw = fetch_universe_rows(specs, args.use_cache)
    raw = assign_rows(raw, specs)
    audit = write_audit(raw, specs)  # name-level audit BEFORE reattribution
    raw, change_log = apply_reattributions(raw)
    log_df = pd.DataFrame(change_log)
    log_df.to_csv(REATTRIBUTION_CSV, index=False)
    logger.info("wrote M&A reattribution log -> %s (%d rules, %d applications "
                "moved/dropped)", REATTRIBUTION_CSV, len(log_df),
                int(log_df["applications_moved"].sum()))
    companies = collapse_to_companies(raw, specs)
    for c in companies:
        logger.info("  [%s] %d applications", c.ticker, c.n_patents)

    fins, unresolved = fetch_universe_financials(specs)
    pd.DataFrame(MA_EVENTS, columns=["ticker", "date", "event"]).assign(
        note="manually curated, non-exhaustive; annotation layer, NOT a model "
             "input").to_csv(MA_CSV, index=False)
    logger.info("wrote M&A annotations -> %s (%d events)", MA_CSV, len(MA_EVENTS))

    quality = quality_table(companies, fins, specs)
    write_registry({r["ticker"]: {"fin_first": r["fin_first_quarter"],
                                  "fin_last": r["fin_last_quarter"]}
                    for r in quality.to_dict("records")})

    grid = channels.build_channels(companies, fins)
    grid.to_csv(CHANNELS_CSV, index=False)
    logger.info("wrote universe channel grid -> %s (%d rows)",
                CHANNELS_CSV, len(grid))

    print("\n" + "=" * 76)
    print("QVM-V2 STEP 5a — UNIVERSE PLUMBING SUMMARY")
    print("=" * 76)
    print(quality.to_string(index=False, max_colwidth=58))
    if unresolved:
        print("\n--- UNRESOLVED financial mappings (fix before Step 5b) ---")
        for u in unresolved:
            print(f"  {u}")
    rev = audit[audit["review_status"] == "REVIEW"]
    print(f"\nREVIEW queue: {len(rev)} assignee names >= {REVIEW_THRESHOLD} apps "
          f"across {rev['ticker'].nunique()} companies -> {AUDIT_CSV}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
