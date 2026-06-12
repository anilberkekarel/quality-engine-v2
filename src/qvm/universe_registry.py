"""STEP 5a — universe registry: the single source of truth for the ~50-company
semiconductor universe. NO MODELING HERE; plumbing only.

SELECTION RULE (documented, applied symmetrically; no ad-hoc picks):
  1. SEC EDGAR companies classified SIC 3674 (Semiconductors & Related
     Devices) that have filed a 10-K — pulled from the EDGAR company browse,
     which RETAINS DELISTED filers (Xilinx, Altera, Inphi, Cypress, Maxim,
     Freescale, ... enter with their series up to delisting; survivorship is
     thereby partially mitigated. Pre-2009 delistings are out of scope —
     documented limitation, not hidden: XBRL starts ~2009).
  2. Size screen: peak annual revenue >= $500M in any XBRL revenue frame
     2009-2023 (union of the four revenue tag variants). $500M (not $1B) so
     that mid-caps like Inphi/Lattice/MACOM stay in; sensitivity: $1B would
     give ~36 companies, $500M gives ~51.
  3. Category screen — the study is about semiconductor DEVICE companies
     (the four cases' peer group). SIC 3674 also contains solar/PV system
     makers, equipment/materials/OSAT firms, optical-module makers, and
     foreign 20-F filers (no quarterly 10-Qs -> no quarterly grid). These are
     EXCLUDED with per-company category labels recorded in the registry CSV
     (transparency), not silently dropped.
  4. Depth screen: >= 8 quarterly XBRL observations (drops e.g. Qnity, the
     2025 spin-off). Later IPOs (Allegro 2020) stay, with depth recorded.
  5. The four case companies are pinned (they pass every screen anyway) and
     their assignee specs are IMPORTED from config.COMPANIES so Step 1-4
     cleanups stay the single source of truth.

KNOWN RULE CONSEQUENCES (documented): Qualcomm is SIC 3663 (radio/TV comms
equipment), NOT 3674 -> outside the rule, outside the universe. Texas
Instruments pre-2009 history, National Semi pre-2009 era etc. limited by
XBRL's ~2009 start.

Entity continuity (the MRVL lesson): successor tickers carry predecessor
CIKs in `ciks` (successor first). True mergers-of-equals (Qorvo = RFMD +
TriQuint) are NOT pre-merger-merged — predecessors enter as separate
delisted members; merging their financials would fabricate a pre-merger
company. Same-CIK renames (Cree->Wolfspeed, IDT->'Renesas Electronics
America') are flagged, not split.

Assignee prefixes are CANDIDATES feeding the semi-automatic audit; every
matched name with >=20 applications is tagged REVIEW in the audit CSV and
DECIDED BY THE RESEARCHER — this module never silently excludes beyond the
explicitly listed exclusion rules. (<10-app ambiguous names = residual
noise, documented, kept — the Step-2 symmetric rule.)
"""

from __future__ import annotations

from . import config

# --------------------------------------------------------------------------- #
# the included universe (category: semiconductor device, >=$500M, >=8 quarters)
# Fields: ticker, label, ciks (successor first), name_likes (prefix candidates),
# optional exclude_name_like / include_name_like (same semantics as
# config.COMPANIES), delisted ("" = still listed; else "YYYY-MM acquirer"),
# flags (continuity / composition warnings), notes.
# --------------------------------------------------------------------------- #
UNIVERSE: list[dict] = [
    # ---- the four case companies (assignee spec imported from config) ----
    {"ticker": "NVDA", "label": "NVIDIA", "ciks": [1045810], "case": True,
     "delisted": "", "flags": ["M&A: Mellanox 2020 (patent composition)"]},
    {"ticker": "AMD", "label": "AMD", "ciks": [2488], "case": True,
     "delisted": "", "flags": ["M&A: Xilinx 2022 (patent composition)"]},
    {"ticker": "MRVL", "label": "Marvell", "ciks": [1835632, 1058057],
     "case": True, "delisted": "",
     "flags": ["predecessor CIK merged (Marvell Technology Group)",
               "M&A: Cavium 2018, Inphi 2021 (patent composition)"]},
    {"ticker": "MU", "label": "Micron", "ciks": [723125], "case": True,
     "delisted": "", "flags": ["M&A: Elpida 2013 (patent composition)"]},

    # ---- listed members ----
    # DOCUMENTED AMENDMENT (Step 5b Görev 0, decided 2026-06-12, BEFORE any
    # prediction was run): Qualcomm is SIC 3663 (radio/TV comms equipment), so
    # the SIC-3674 selection rule excludes it — yet it is a semiconductor
    # DEVICE company by any economic definition, a top-3 patent filer in the
    # peer group, and the acquirer in the ATHR reattribution chain. Excluding
    # the largest fabless designer on an SIC technicality would bias the
    # universe more than the exception biases the rule. Recorded as the single
    # SIC exception; the selection rule itself is unchanged.
    {"ticker": "QCOM", "label": "Qualcomm", "ciks": [804328],
     "name_likes": ["QUALCOMM%"], "delisted": "",
     "flags": ["SIC exception: 3663, added by documented amendment (Görev 0, "
               "pre-prediction)",
               "M&A: Atheros 2011 (patent composition; ATHR member exits)"],
     "notes": "amendment decision logged before the Step 5b prediction task "
              "ran — not survivorship- or result-driven"},
    {"ticker": "INTC", "label": "Intel", "ciks": [50863],
     "name_likes": ["INTEL CORP%", "INTEL IP%", "INTEL AMERICAS%",
                    "INTEL MOBILE%", "INTEL DEUTSCHLAND%"],
     "delisted": "",
     "notes": "bare INTEL% would catch INTELLECTUAL/INTELLIGENT/INTELLON -> "
              "entity-specific prefixes; audit shows what each catches"},
    {"ticker": "TXN", "label": "Texas Instruments", "ciks": [97476],
     "name_likes": ["TEXAS INSTRUMENTS%"], "delisted": "",
     "flags": ["M&A: National Semiconductor 2011 (patent composition)"]},
    {"ticker": "AVGO", "label": "Broadcom (Avago lineage)",
     "ciks": [1730168, 1441634],
     "name_likes": ["AVAGO%", "BROADCOM INTERNATIONAL%"],
     "delisted": "",
     "flags": ["predecessor CIK merged (Avago Technologies Ltd)",
               "BROADCOM% assignee names pre-2016 belong to BRCM (separate "
               "member); post-2018 'BROADCOM INC' filings belong HERE — "
               "REVIEW the split in the audit",
               "M&A: LSI 2014, Broadcom Corp 2016 (patent composition)"]},
    {"ticker": "NXPI", "label": "NXP Semiconductors", "ciks": [1413447],
     "name_likes": ["NXP%"], "delisted": "",
     "flags": ["M&A: Freescale 2015 (patent composition)",
               "filed 20-F as foreign private issuer until ~2017 -> quarterly "
               "XBRL only from 2018 (structural, not a bug)"]},
    {"ticker": "ADI", "label": "Analog Devices", "ciks": [6281],
     "name_likes": ["ANALOG DEVICES%"], "delisted": "",
     "flags": ["M&A: Linear 2017, Maxim 2021 (patent composition)"]},
    {"ticker": "MCHP", "label": "Microchip", "ciks": [827054],
     "name_likes": ["MICROCHIP TECH%"], "delisted": "",
     "notes": "harmonized name is MICROCHIP TECH INC (BQ probe); prefix also "
              "avoids GENESIS MICROCHIP and MICROCHIPS INC (different cos)",
     "flags": ["M&A: SMSC 2012, Atmel 2016, Microsemi 2018 (composition)"]},
    {"ticker": "ON", "label": "onsemi", "ciks": [1097864],
     "name_likes": ["SEMICONDUCTOR COMPONENTS%", "ON SEMICONDUCTOR%"],
     "delisted": "",
     "flags": ["patents filed by operating sub SEMICONDUCTOR COMPONENTS "
               "INDUSTRIES LLC — both prefixes needed (continuity)",
               "M&A: Fairchild 2016 (patent composition)"]},
    {"ticker": "SWKS", "label": "Skyworks", "ciks": [4127],
     "name_likes": ["SKYWORKS%"], "delisted": "",
     "notes": "SKYWORKS PANASONIC FILTER SOLUTIONS JAPAN (42 apps) KEPT — "
              "consolidated JV (Skyworks majority owner since 2016), same "
              "consolidation principle as subsidiaries (Görev 0 decision)"},
    {"ticker": "QRVO", "label": "Qorvo", "ciks": [1604778],
     "name_likes": ["QORVO%"], "delisted": "",
     # panel membership starts at the merger closing; QRVO companyfacts
     # comparatives reaching back to 2013 are the RFMD continuation and would
     # double-count the separate RFMD member in the 2013-2014 cross-sections
     "member_start": "2015Q1",
     "notes": "QORVO BIOTECHNOLOGIES LLC (26 apps) KEPT — wholly-owned "
              "subsidiary (Omnia diagnostics), consolidation principle: "
              "consolidated subsidiaries' filings belong to the parent "
              "regardless of business line (Görev 0 decision)",
     "flags": ["member starts AT the merger closing (2015-01-01) — "
               "RFMD/TQNT series END there (M&A attribution rule)",
               "formed 2015 from RFMD + TriQuint (both separate delisted "
               "members); patent series pre-2015 lives under those names — "
               "continuity flag, NOT merged",
               "QRVO companyfacts comparatives reach back to 2013 = RFMD "
               "continuation (accounting acquirer) -> 2013Q2-2014Q3 overlaps "
               "the RFMD member series (composition note)"]},
    {"ticker": "DIOD", "label": "Diodes", "ciks": [29002],
     "name_likes": ["DIODES INC%"], "delisted": "",
     "notes": "bare DIODES% would catch generic 'DIODES ...' names"},
    {"ticker": "CRUS", "label": "Cirrus Logic", "ciks": [772406],
     "name_likes": ["CIRRUS LOGIC%"], "delisted": ""},
    {"ticker": "MPWR", "label": "Monolithic Power", "ciks": [1280452],
     "name_likes": ["MONOLITHIC POWER%"], "delisted": ""},
    {"ticker": "SYNA", "label": "Synaptics", "ciks": [817720],
     "name_likes": ["SYNAPTICS%"], "delisted": ""},
    {"ticker": "WOLF", "label": "Wolfspeed (ex-Cree)", "ciks": [895419],
     "name_likes": ["CREE%", "WOLFSPEED%"],
     "exclude_name_like": [
         {"prefix": "CREED%",
          "reason": "Creed & Co Ltd — unrelated company sharing the prefix"}],
     "delisted": "",
     "flags": ["same CIK renamed Cree->Wolfspeed 2021",
               "CREELED INC (224 apps) = LED unit divested to SMART Global "
               "2021 — REVIEW whether post-2021 CreeLED filings stay"]},
    {"ticker": "MXL", "label": "MaxLinear", "ciks": [1288469],
     "name_likes": ["MAXLINEAR%"], "delisted": ""},
    {"ticker": "SLAB", "label": "Silicon Labs", "ciks": [1038074],
     "name_likes": ["SILICON LAB%"], "delisted": "",
     "notes": "harmonized name is SILICON LAB INC (BQ probe)"},
    {"ticker": "ALGM", "label": "Allegro MicroSystems", "ciks": [866291],
     "name_likes": ["ALLEGRO MICROSYSTEMS%"], "delisted": "",
     "flags": ["IPO 2020 -> short financial series (patents reach back)"]},
    {"ticker": "MX", "label": "MagnaChip", "ciks": [1325702],
     "name_likes": ["MAGNACHIP%"], "delisted": ""},
    {"ticker": "SMTC", "label": "Semtech", "ciks": [88941],
     "name_likes": ["SEMTECH%"], "delisted": ""},
    {"ticker": "MTSI", "label": "MACOM", "ciks": [1493594],
     "name_likes": ["MACOM%", "M A COM%", "M/A-COM%"],
     "exclude_name_like": [
         {"prefix": "MACOMBER%",
          "reason": "Macomber Steel / Macomber Inc — unrelated companies"}],
     "delisted": "",
     "flags": ["legacy M/A-COM (Tyco-era) entity names match — REVIEW "
               "whether pre-2009 M/A-COM heritage counts as MACOM"]},
    {"ticker": "LSCC", "label": "Lattice", "ciks": [855658],
     "name_likes": ["LATTICE SEMICONDUCTOR%"], "delisted": ""},
    {"ticker": "POWI", "label": "Power Integrations", "ciks": [833640],
     "name_likes": ["POWER INTEGRATIONS%"], "delisted": ""},
    {"ticker": "AOSL", "label": "Alpha & Omega", "ciks": [1387467],
     "name_likes": ["ALPHA & OMEGA SEMICONDUCTOR%",
                    "ALPHA AND OMEGA SEMICONDUCTOR%"], "delisted": ""},

    # ---- delisted members (series end naturally at delisting) ----
    {"ticker": "BRCM", "label": "Broadcom Corp (legacy)", "ciks": [1054374],
     "name_likes": ["BROADCOM%"],
     "exclude_name_like": [
         {"prefix": "BROADCOM INTERNATIONAL%",
          "reason": "post-2018 Avago-lineage entity (member AVGO), not "
                    "legacy Broadcom Corp"}],
     "delisted": "2016-02 acquired by Avago (renamed Broadcom)",
     "flags": ["BROADCOM INC assignee names post-2018 belong to AVGO — "
               "REVIEW the audit split"]},
    {"ticker": "XLNX", "label": "Xilinx", "ciks": [743988],
     "name_likes": ["XILINX%"], "delisted": "2022-02 acquired by AMD"},
    {"ticker": "ALTR", "label": "Altera", "ciks": [768251],
     "name_likes": ["ALTERA%"], "delisted": "2015-12 acquired by Intel"},
    {"ticker": "MXIM", "label": "Maxim Integrated", "ciks": [743316],
     "name_likes": ["MAXIM INTEGRATED%"],
     "delisted": "2021-08 acquired by Analog Devices",
     "notes": "bare MAXIM% would catch persons/unrelated MAXIM* companies"},
    {"ticker": "CY", "label": "Cypress", "ciks": [791915],
     "name_likes": ["CYPRESS SEMICONDUCTOR%"],
     "delisted": "2020-04 acquired by Infineon",
     "flags": ["M&A: Spansion merged in 2015 (separate member; composition)"]},
    {"ticker": "CODE", "label": "Spansion", "ciks": [1322705],
     "name_likes": ["SPANSION%"],
     "delisted": "2015-03 merged into Cypress"},
    {"ticker": "FSL", "label": "Freescale", "ciks": [1392522],
     "name_likes": ["FREESCALE%"],
     "delisted": "2015-12 acquired by NXP",
     "flags": ["predecessor CIK 1272547 (pre-2011-IPO bond filer) has NO "
               "companyfacts on EDGAR — pre-IPO quarters unavailable, "
               "documented limitation"]},
    {"ticker": "MSCC", "label": "Microsemi", "ciks": [310568],
     "name_likes": ["MICROSEMI%"],
     "delisted": "2018-05 acquired by Microchip",
     "flags": ["M&A: PMC-Sierra 2016 (separate member; composition)"]},
    {"ticker": "ATML", "label": "Atmel", "ciks": [872448],
     "name_likes": ["ATMEL%"], "delisted": "2016-04 acquired by Microchip"},
    {"ticker": "FCS", "label": "Fairchild", "ciks": [1036960],
     "name_likes": ["FAIRCHILD SEMICONDUCTOR%"],
     "delisted": "2016-09 acquired by onsemi"},
    {"ticker": "NSM", "label": "National Semiconductor", "ciks": [70530],
     "name_likes": ["NAT SEMICONDUCTOR%", "NATIONAL SEMICONDUCT%"],
     "delisted": "2011-09 acquired by Texas Instruments",
     "notes": "harmonized name is NAT SEMICONDUCTOR CORP (BQ probe)"},
    {"ticker": "LSI", "label": "LSI Corp", "ciks": [703360],
     "name_likes": ["LSI CORP%", "LSI LOGIC%"],
     "delisted": "2014-05 acquired by Avago",
     "notes": "bare LSI% would catch LSI INDUSTRIES etc."},
    {"ticker": "LLTC", "label": "Linear Technology", "ciks": [791907],
     "name_likes": ["LINEAR TECH%"],
     "delisted": "2017-03 acquired by Analog Devices",
     "notes": "harmonized names are LINEAR TECHN INC / LINEAR TECH CORP "
              "(BQ probe) — TECH% prefix covers both"},
    {"ticker": "MLNX", "label": "Mellanox", "ciks": [1356104],
     "name_likes": ["MELLANOX%"],
     "delisted": "2020-04 acquired by NVIDIA"},
    {"ticker": "IRF", "label": "International Rectifier", "ciks": [316793],
     "name_likes": ["INT RECTIFIER%", "INTERNATIONAL RECTIF%"],
     "delisted": "2015-01 acquired by Infineon",
     "notes": "harmonized name is INT RECTIFIER CORP (BQ probe)"},
    {"ticker": "RFMD", "label": "RF Micro Devices", "ciks": [911160],
     "name_likes": ["RF MICRO DEVICES%"],
     "delisted": "2015-01 merged into Qorvo"},
    {"ticker": "TQNT", "label": "TriQuint", "ciks": [913885],
     "name_likes": ["TRIQUINT%"],
     "delisted": "2015-01 merged into Qorvo"},
    {"ticker": "CAVM", "label": "Cavium", "ciks": [1175609],
     "name_likes": ["CAVIUM%"],
     "delisted": "2018-07 acquired by Marvell"},
    {"ticker": "ATHR", "label": "Atheros", "ciks": [1140486],
     "name_likes": ["ATHEROS%"],
     "delisted": "2011-05 acquired by Qualcomm"},
    {"ticker": "OVTI", "label": "OmniVision", "ciks": [1106851],
     "name_likes": ["OMNIVISION%"],
     "delisted": "2016-01 acquired (Chinese consortium)"},
    {"ticker": "IDTI", "label": "Integrated Device Technology", "ciks": [703361],
     "name_likes": ["INTEGRATED DEVICE TECH%"],
     "delisted": "2019-03 acquired by Renesas",
     "flags": ["same CIK now named 'Renesas Electronics America' — rename, "
               "not a new entity"]},
    {"ticker": "ISIL", "label": "Intersil", "ciks": [1096325],
     "name_likes": ["INTERSIL%"],
     "delisted": "2017-02 acquired by Renesas"},
    {"ticker": "PMCS", "label": "PMC-Sierra", "ciks": [767920],
     "name_likes": ["PMC SIERRA%", "PMC-SIERRA%"],
     "delisted": "2016-01 acquired by Microsemi"},
    {"ticker": "IPHI", "label": "Inphi", "ciks": [1160958],
     "name_likes": ["INPHI%"],
     "delisted": "2021-04 acquired by Marvell"},
]

# --------------------------------------------------------------------------- #
# screen survivors EXCLUDED by the category screen — recorded for transparency
# (cik -> (category, short reason)). Categories: solar_pv, equipment_materials,
# optical_modules, foreign_20f (no 10-Q quarterly), duplicate_coissuer,
# insufficient_depth, ip_licensing.
# --------------------------------------------------------------------------- #
EXCLUDED_SCREEN: dict[int, tuple[str, str]] = {
    1855447: ("solar_pv", "Tigo Energy (its $145B frame fact is an XBRL filing error)"),
    1274494: ("solar_pv", "First Solar"),
    1481513: ("foreign_20f", "JinkoSolar (also solar)"),
    1375877: ("foreign_20f", "Canadian Solar (also solar)"),
    1342803: ("foreign_20f", "Suntech (also solar)"),
    1419612: ("solar_pv", "SolarEdge"),
    1382158: ("foreign_20f", "Trina Solar (also solar)"),
    867773:  ("solar_pv", "SunPower"),
    945436:  ("solar_pv", "SunEdison"),
    1385424: ("foreign_20f", "LDK Solar (also solar)"),
    1371541: ("foreign_20f", "Hanwha Q CELLS (also solar)"),
    1394029: ("foreign_20f", "Yingli (also solar)"),
    1463101: ("solar_pv", "Enphase"),
    1820721: ("solar_pv", "Array Technologies"),
    1796898: ("foreign_20f", "Maxeon (also solar)"),
    1852131: ("solar_pv", "Nextpower"),
    1396247: ("foreign_20f", "China Sunergy (also solar)"),
    1477641: ("foreign_20f", "Daqo New Energy (also solar)"),
    1394954: ("equipment_materials", "GT Advanced (PV/sapphire equipment)"),
    6951:    ("equipment_materials", "Applied Materials (fab equipment)"),
    1047127: ("equipment_materials", "Amkor (OSAT packaging services)"),
    1275014: ("equipment_materials", "Ultra Clean (fab subsystems)"),
    56978:   ("equipment_materials", "Kulicke & Soffa (assembly equipment)"),
    1652535: ("equipment_materials", "Ichor (fluid delivery subsystems)"),
    1102934: ("equipment_materials", "CMC Materials (CMP slurries)"),
    810136:  ("equipment_materials", "Photronics (photomasks)"),
    1039399: ("equipment_materials", "FormFactor (probe cards)"),
    1352341: ("equipment_materials", "Verigy (test equipment)"),
    1487990: ("equipment_materials", "Aeroflex (test/microelectronics mix)"),
    1039065: ("equipment_materials", "OSI Systems (security/inspection systems)"),
    1585854: ("equipment_materials", "SunEdison Semiconductor (silicon wafers)"),
    1616533: ("equipment_materials", "Penguin Solutions / SMART Global (memory modules)"),
    1111928: ("optical_modules", "IPG Photonics (fiber lasers)"),
    1094739: ("optical_modules", "Finisar"),
    912093:  ("optical_modules", "Viavi (test + optical)"),
    1110647: ("optical_modules", "Oclaro"),
    1651235: ("optical_modules", "Acacia Communications"),
    932787:  ("foreign_20f", "STMicroelectronics"),
    928876:  ("foreign_20f", "Tower Semiconductor"),
    1267482: ("foreign_20f", "SMIC"),
    1329394: ("foreign_20f", "Silicon Motion"),
    1342338: ("foreign_20f", "Himax"),
    1287950: ("foreign_20f", "Spreadtrum"),
    1973239: ("foreign_20f", "ARM Holdings"),
    1649338: ("duplicate_coissuer", "Broadcom Pte — AVGO lineage co-registrant"),
    1649345: ("duplicate_coissuer", "Broadcom Cayman LP — AVGO lineage co-registrant"),
    1272547: ("duplicate_coissuer", "Freescale Inc — merged into FSL member as predecessor CIK"),
    2058873: ("insufficient_depth", "Qnity Electronics (2025 spin-off, <8 quarters)"),
}


# --------------------------------------------------------------------------- #
# M&A PATENT-ATTRIBUTION RULE (Step 5b Görev 0, pre-registered):
# attribution SPLITS AT THE CLOSING DATE — an application FILED on/after the
# closing under the target's assignee names belongs to the acquirer (if a
# universe member) or to nobody (acquirer outside the universe -> the target's
# series ends naturally; post-close filings dropped). In-universe double
# counting is thereby impossible. Rules are applied in effective-date order,
# so chains compose (PMC-Sierra -> Microsemi 2016 -> Microchip 2018).
# `prefix` (optional) scopes a rule to one assignee prefix instead of the
# whole member (CreeLED divestiture). Filing date is the split key — it is
# the only date that exists at attribution time.
# --------------------------------------------------------------------------- #
REATTRIBUTIONS: list[dict] = [
    {"from": "ATHR", "to": "QCOM", "effective": "2011-05-24",
     "reason": "Qualcomm closing (member by documented amendment)"},
    {"from": "NSM", "to": "TXN", "effective": "2011-09-23", "reason": "TI closing"},
    {"from": "LSI", "to": "AVGO", "effective": "2014-05-06", "reason": "Avago closing"},
    {"from": "IRF", "to": None, "effective": "2015-01-13",
     "reason": "Infineon (non-member): post-close filings dropped"},
    {"from": "RFMD", "to": "QRVO", "effective": "2015-01-01",
     "reason": "Qorvo merger close"},
    {"from": "TQNT", "to": "QRVO", "effective": "2015-01-01",
     "reason": "Qorvo merger close"},
    {"from": "CODE", "to": "CY", "effective": "2015-03-12",
     "reason": "Cypress merger close"},
    {"from": "FSL", "to": "NXPI", "effective": "2015-12-07", "reason": "NXP closing"},
    {"from": "ALTR", "to": "INTC", "effective": "2015-12-28",
     "reason": "Intel closing"},
    {"from": "OVTI", "to": None, "effective": "2016-01-28",
     "reason": "Chinese consortium (non-member)"},
    {"from": "PMCS", "to": "MSCC", "effective": "2016-01-15",
     "reason": "Microsemi closing"},
    {"from": "BRCM", "to": "AVGO", "effective": "2016-02-01",
     "reason": "Avago closing (renamed Broadcom)"},
    {"from": "ATML", "to": "MCHP", "effective": "2016-04-04",
     "reason": "Microchip closing"},
    {"from": "FCS", "to": "ON", "effective": "2016-09-19", "reason": "onsemi closing"},
    {"from": "ISIL", "to": None, "effective": "2017-02-24",
     "reason": "Renesas (non-member)"},
    {"from": "LLTC", "to": "ADI", "effective": "2017-03-10", "reason": "ADI closing"},
    {"from": "MSCC", "to": "MCHP", "effective": "2018-05-29",
     "reason": "Microchip closing (chains ex-PMC-Sierra forward)"},
    {"from": "CAVM", "to": "MRVL", "effective": "2018-07-06",
     "reason": "Marvell closing"},
    {"from": "IDTI", "to": None, "effective": "2019-03-30",
     "reason": "Renesas (non-member)"},
    {"from": "CY", "to": None, "effective": "2020-04-16",
     "reason": "Infineon (non-member)"},
    {"from": "MLNX", "to": "NVDA", "effective": "2020-04-27",
     "reason": "NVIDIA closing"},
    {"from": "WOLF", "to": None, "effective": "2021-03-01", "prefix": "CREELED%",
     "reason": "CreeLED divested to SMART Global (non-member); pre-close "
               "CREELED filings stay with WOLF"},
    {"from": "IPHI", "to": "MRVL", "effective": "2021-04-20",
     "reason": "Marvell closing"},
    {"from": "MXIM", "to": "ADI", "effective": "2021-08-26", "reason": "ADI closing"},
    {"from": "XLNX", "to": "AMD", "effective": "2022-02-14", "reason": "AMD closing"},
]


def apply_reattributions(df, ticker_col: str = "_ticker"):
    """Apply the M&A attribution rule to assigned publication rows.

    Expects columns: ticker_col (member assignment from assignee prefixes),
    assignee_name, filing_date (INTEGER yyyymmdd). Rules run in effective-date
    order on the EVOLVING ticker column, so chains compose (a PMC-Sierra-named
    2019 filing goes PMCS -> MSCC -> MCHP). `to=None` drops the rows (acquirer
    outside the universe). Returns (df, change_log) where change_log counts
    moved/dropped applications per rule — the Görev 0 transparency record.
    """
    import pandas as pd
    df = df.copy()
    filing = pd.to_numeric(df["filing_date"], errors="coerce").fillna(0)
    change_log = []
    for rule in sorted(REATTRIBUTIONS, key=lambda r: r["effective"]):
        eff = int(rule["effective"].replace("-", ""))
        mask = (df[ticker_col] == rule["from"]) & (filing >= eff)
        if rule.get("prefix"):
            pre = rule["prefix"].rstrip("%").upper()
            mask &= df["assignee_name"].fillna("").str.upper().str.startswith(pre)
        n_apps = df.loc[mask, "application_number"].nunique()
        if rule["to"] is None:
            df = df[~mask]
            filing = filing[df.index]
        else:
            df.loc[mask, ticker_col] = rule["to"]
        change_log.append({"from": rule["from"], "to": rule["to"] or "DROPPED",
                           "effective": rule["effective"],
                           "applications_moved": int(n_apps),
                           "reason": rule["reason"]})
    return df, change_log


def universe_specs() -> list[dict]:
    """Registry entries as company specs for the BigQuery/audit machinery.

    The four case companies take their assignee spec (name_like, exclusions,
    allowlist) STRAIGHT from config.COMPANIES — one source of truth.
    """
    case_by_ticker = {s["ticker"]: s for s in config.COMPANIES}
    specs = []
    for entry in UNIVERSE:
        spec = dict(entry)
        if entry.get("case"):
            base = case_by_ticker[entry["ticker"]]
            for key in ("name_like", "name_likes", "exclude_name_like",
                        "include_name_like", "expect_min_patents"):
                if key in base:
                    spec[key] = base[key]
        specs.append(spec)
    return specs
