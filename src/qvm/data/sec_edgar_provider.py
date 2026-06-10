"""Concrete FinancialProvider over SEC EDGAR XBRL companyfacts.

Free, key-less, primary-source. ONE request per company returns every XBRL
concept the company ever filed (data.sec.gov/api/xbrl/companyfacts). We cache
the raw JSON in _cache/ and re-extract from cache on later runs.

SEC fair-access policy: a User-Agent with real contact info is REQUIRED and
rate must stay <=10 req/s — see config.SEC_USER_AGENT / SEC_MIN_REQUEST_INTERVAL_S.

EXTRACTION PIPELINE (pure functions, unit-testable without network):
  1. For each canonical concept, walk its us-gaap tag-priority list
     (config.CONCEPT_TAG_PRIORITY); merge PER PERIOD so a company that
     switched tags mid-history (ASC 606) still yields one continuous series.
     Which tag covered which years is logged into tag_provenance.
  2. Classify each fact by period duration: ~quarter (70-100d) or ~annual
     (340-380d); 6/9-month YTD cumulatives are skipped.
  3. A period usually appears in MULTIPLE filings (own 10-Q, then comparative
     columns of later filings, plus restatements). Keep the EARLIEST filing's
     value as the point-in-time "as_filed" value and the LATEST separately.
  4. Q4 TRAP: 10-Ks report the full fiscal year; Q4 is never filed alone.
     Derive Q4 = FY - (Q1+Q2+Q3) from the three quarters inside each annual
     period; knowable_at(Q4) = max(filed of FY,Q1..Q3) (in practice the 10-K
     filing date). Every derivation is sanity-screened and recorded.
  5. FISCAL != CALENDAR: alignment downstream uses each period's 'end' date
     (mapped to the nearest calendar quarter), NEVER the fy/fp labels —
     NVDA/MRVL fiscal years end late January, MU late Aug/early Sep.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time

from .. import config
from .financial_base import CompanyFinancials, FinancialObservation, FinancialProvider

logger = logging.getLogger(__name__)

_DAY = dt.timedelta(days=1)


def _d(iso: str) -> dt.date:
    return dt.date.fromisoformat(iso)


# --------------------------------------------------------------------------- #
# pure extraction helpers
# --------------------------------------------------------------------------- #
def _classify_duration(start: str, end: str) -> str | None:
    days = (_d(end) - _d(start)).days
    lo, hi = config.QUARTER_DURATION_DAYS
    if lo <= days <= hi:
        return "quarter"
    lo, hi = config.ANNUAL_DURATION_DAYS
    if lo <= days <= hi:
        return "annual"
    return None  # YTD cumulative or junk


def _collapse_filings(entries: list[dict]) -> dict:
    """All filings of ONE (start,end) period -> as_filed + latest values."""
    entries = sorted(entries, key=lambda e: (e["filed"], e.get("accn", "")))
    first, last = entries[0], entries[-1]
    return {
        "value": float(first["val"]), "filed": first["filed"],
        "value_latest": float(last["val"]), "filed_latest": last["filed"],
        "n_filings": len({e["filed"] for e in entries}),
        "form": first.get("form", ""),
        "fiscal_label": f"FY{first.get('fy', '?')} {first.get('fp', '?')}",
    }


def _fuzzy_key(target: dict, start: str, end: str) -> tuple | None:
    """Existing period key whose start AND end are both within tolerance.

    Refilings sometimes shift a fiscal boundary by a day (NVDA fiscal Q2-2011
    is filed as ..2010-07-31 and ..2010-08-01); exact keying would duplicate
    the quarter and silently break Q4 derivation.
    """
    tol = config.PERIOD_MATCH_TOLERANCE_DAYS
    s, e = _d(start), _d(end)
    for ks, ke in target:
        if abs((_d(ks) - s).days) <= tol and abs((_d(ke) - e).days) <= tol:
            return (ks, ke)
    return None


def extract_concept_periods(gaap_facts_list: list[dict], concept: str
                            ) -> tuple[dict, dict, list[dict]]:
    """One concept -> ({(start,end): obs_dict} quarterly, same annual, provenance).

    `gaap_facts_list` holds one us-gaap dict per SEC entity — successor +
    predecessor CIKs of the same company are merged here, period by period.
    Walks the tag priority list; a period already covered by a higher-priority
    tag is NOT overwritten by a lower one (per-period merge). Same-tag entries
    for the same (fuzzy-matched) period pool their filings, so as_filed /
    latest are computed across entities and refilings alike.
    """
    quarterly: dict[tuple, dict] = {}
    annual: dict[tuple, dict] = {}
    for tag_idx, tag in enumerate(config.CONCEPT_TAG_PRIORITY[concept]):
        usd = [e for gaap in gaap_facts_list
               for e in gaap.get(tag, {}).get("units", {}).get("USD", [])]
        for e in usd:
            if not e.get("start") or not e.get("end") or e.get("val") is None:
                continue
            kind = _classify_duration(e["start"], e["end"])
            if kind is None:
                continue
            target = quarterly if kind == "quarter" else annual
            key = _fuzzy_key(target, e["start"], e["end"]) or (e["start"], e["end"])
            slot = target.get(key)
            if slot is None:
                target[key] = {"tag": tag, "tag_idx": tag_idx, "entries": [e]}
            elif slot["tag_idx"] == tag_idx:
                slot["entries"].append(e)
            # else: a higher-priority tag already covers this period -> skip

    def _collapse(d: dict) -> dict:
        out = {}
        for (start, end), slot in d.items():
            obs = _collapse_filings(slot["entries"])
            obs.update({"tag": slot["tag"], "start": start, "end": end})
            out[(start, end)] = obs
        return out

    quarterly, annual = _collapse(quarterly), _collapse(annual)
    by_tag: dict[str, list[str]] = {}
    for obs in list(quarterly.values()) + list(annual.values()):
        by_tag.setdefault(obs["tag"], []).append(obs["end"])
    provenance = [{"tag": tag, "n_periods": len(ends),
                   "first_end": min(ends), "last_end": max(ends)}
                  for tag in config.CONCEPT_TAG_PRIORITY[concept]
                  if (ends := by_tag.get(tag))]
    return quarterly, annual, provenance


def derive_q4(quarterly: dict, annual: dict, concept: str
              ) -> tuple[list[dict], list[dict], list[tuple]]:
    """FY - (Q1+Q2+Q3) per fiscal year -> (derived obs, sanity records, drops).

    Q4 is ALWAYS the derived value when FY + three early quarters exist —
    uniform methodology AND the earliest knowable_at (the 10-K date). Some
    fiscal years also carry a DIRECTLY filed Q4 fact, but those often surface
    first in a comparative filing 1-2 YEARS later (seen in MRVL 2017), so they
    are demoted to a cross-check: deviation vs the derived value is recorded
    and the direct fact's period key is returned in `drops` for removal.
    A direct Q4 is kept as data only when derivation is impossible.
    """
    derived, sanity, drops = [], [], []
    for (a_start, a_end), fy in sorted(annual.items()):
        inside = sorted(
            ((key, obs) for key, obs in quarterly.items()
             if _d(key[0]) >= _d(a_start) - 10 * _DAY
             and _d(key[1]) <= _d(a_end) + 10 * _DAY),
            key=lambda kv: kv[1]["end"])
        # a quarter ending AT the FY end is a directly-filed Q4
        direct_q4 = None
        if inside and abs((_d(inside[-1][1]["end"]) - _d(a_end)).days) \
                <= config.PERIOD_MATCH_TOLERANCE_DAYS:
            direct_q4 = inside[-1]
        early = inside[:-1] if direct_q4 else inside
        rec = {"concept": concept, "fy_start": a_start, "fy_end": a_end,
               "fy_value": fy["value"], "n_quarters_found": len(inside),
               "derived": False, "deviation_pct": None, "status": "ok"}
        if len(early) != 3:
            rec["status"] = (f"cannot derive Q4 ({len(early)} early quarters "
                             f"inside FY)" + (": direct Q4 fact kept"
                                              if direct_q4 else ""))
            sanity.append(rec)
            continue
        early_obs = [obs for _, obs in early]
        q123 = sum(o["value"] for o in early_obs)
        q123_latest = sum(o["value_latest"] for o in early_obs)
        q4_start = (_d(early_obs[-1]["end"]) + _DAY).isoformat()
        q4 = {
            "tag": fy["tag"], "start": q4_start, "end": a_end,
            "value": fy["value"] - q123,
            # public only once ALL components are: in practice the 10-K date
            "filed": max([fy["filed"]] + [o["filed"] for o in early_obs]),
            "value_latest": fy["value_latest"] - q123_latest,
            "filed_latest": max([fy["filed_latest"]]
                                + [o["filed_latest"] for o in early_obs]),
            "n_filings": fy["n_filings"], "form": f"derived({fy['form']})",
            "fiscal_label": fy["fiscal_label"].replace(" FY", " Q4(derived)"),
        }
        rec["derived"] = True
        # plausibility screen (a failed screen usually means a misclassified
        # YTD fact slipped in or the FY/quarter tags disagree)
        sib_max = max(abs(o["value"]) for o in early_obs)
        if concept == "revenue" and q4["value"] < 0:
            rec["status"] = "DERIVED Q4 NEGATIVE (revenue) — check tags"
        elif sib_max and abs(q4["value"]) > config.Q4_MAX_RATIO_TO_SIBLINGS * sib_max:
            rec["status"] = "derived Q4 implausibly large vs Q1-Q3"
        if direct_q4 is not None:
            key, dobs = direct_q4
            drops.append(key)  # derived replaces the direct fact in the series
            if q4["value_latest"]:
                rec["deviation_pct"] = (100.0 * (dobs["value_latest"] - q4["value_latest"])
                                        / abs(q4["value_latest"]))
                if abs(rec["deviation_pct"]) > 0.5 and rec["status"] == "ok":
                    rec["status"] = "direct Q4 fact deviates from derived Q4"
        derived.append(q4)
        sanity.append(rec)
    return derived, sanity, drops


def extract_company(gaap_facts_list: list[dict], ticker: str,
                    cik: int | None) -> CompanyFinancials:
    """Pure: us-gaap facts dicts (one per SEC entity) -> CompanyFinancials."""
    fin = CompanyFinancials(ticker=ticker, cik=cik, source="sec_edgar")
    for concept in config.CONCEPT_TAG_PRIORITY:
        quarterly, annual, prov = extract_concept_periods(gaap_facts_list, concept)
        fin.tag_provenance[concept] = prov
        derived, sanity, drops = derive_q4(quarterly, annual, concept)
        fin.q4_sanity.extend(sanity)
        for key in drops:  # direct Q4 facts replaced by derived values
            quarterly.pop(key, None)
        all_q = list(quarterly.values()) + derived
        for obs in sorted(all_q, key=lambda o: o["end"]):
            fin.observations.append(FinancialObservation(
                concept=concept, derived_q4=obs.get("form", "").startswith("derived"),
                **{k: obs[k] for k in ("tag", "start", "end", "value", "filed",
                                       "value_latest", "filed_latest",
                                       "n_filings", "form", "fiscal_label")}))
        if not prov:
            fin.notes.append(f"{concept}: no usable us-gaap tag found")
    return fin


# --------------------------------------------------------------------------- #
# the provider (network + cache around the pure core)
# --------------------------------------------------------------------------- #
class SECEdgarProvider(FinancialProvider):
    """Fetches and extracts XBRL companyfacts, one cached request per company."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._last_request = 0.0
        self._cik_map: dict[str, int] | None = None

    def _get_json(self, url: str, cache_name: str) -> dict:
        path = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        import requests
        wait = config.SEC_MIN_REQUEST_INTERVAL_S - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        logger.info("SEC EDGAR GET %s", url)
        resp = requests.get(url, headers={"User-Agent": config.SEC_USER_AGENT},
                            timeout=60)
        self._last_request = time.time()
        resp.raise_for_status()
        data = resp.json()
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("cached -> %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
        return data

    def cik_for(self, ticker: str) -> int:
        if self._cik_map is None:
            raw = self._get_json(config.SEC_TICKER_CIK_URL, "sec_company_tickers.json")
            self._cik_map = {row["ticker"].upper(): int(row["cik_str"])
                             for row in raw.values()}
        try:
            return self._cik_map[ticker.upper()]
        except KeyError:
            raise KeyError(f"ticker {ticker!r} not in SEC company_tickers.json")

    def get_company_financials(self, ticker: str) -> CompanyFinancials:
        cik = self.cik_for(ticker)
        # predecessor entities (e.g. Marvell Technology GROUP pre-2021) are
        # separate CIKs whose facts must be merged for full history
        spec = next((s for s in config.COMPANIES
                     if s["ticker"] == ticker.upper()), {})
        extra = spec.get("sec_additional_ciks", [])
        gaap_list, entities = [], []
        for c in [cik] + [x["cik"] for x in extra]:
            facts = self._get_json(
                config.SEC_COMPANYFACTS_URL.format(cik=c),
                f"sec_companyfacts_{ticker.upper()}_CIK{c:010d}.json")
            gaap_list.append(facts.get("facts", {}).get("us-gaap", {}))
            entities.append(f"{facts.get('entityName', '?')} (CIK {c:010d})")
        fin = extract_company(gaap_list, ticker.upper(), cik)
        fin.notes.extend(f"entity: {e}" for e in entities)
        logger.info("[%s] %s: %d quarterly observations across %d concepts",
                    ticker, " + ".join(entities), len(fin.observations),
                    len(config.CONCEPT_TAG_PRIORITY))
        return fin
