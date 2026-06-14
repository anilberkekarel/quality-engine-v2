"""Emit LaTeX table bodies for Appendix C straight from the output CSVs.

Numbers are read from outputs/scoreboard.csv and
outputs/scoreboard_size_robustness.csv -- nothing is hand-entered.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
sb = pd.read_csv(os.path.join(OUT, "scoreboard.csv"))
sz = pd.read_csv(os.path.join(OUT, "scoreboard_size_robustness.csv"))

NAME = {"D1": "D1 (revenue momentum)", "D2": "D2 (prior-year quintile)",
        "A_feas": "A feasible (patent-only)", "A_orac": "A oracle (patent-only)",
        "B": "B (financial-only)", "C": "C (naive joint)",
        "Cw": "Cw (tempered joint)", "F0": "F0 (fundamentals)",
        "F0+A": "F0+A", "F0+B": "F0+B", "F0+A+B": "F0+A+B", "F0+Cw": "F0+Cw"}
T1 = ["D1", "D2", "A_feas", "A_orac", "B", "C", "Cw"]
T2 = ["F0", "F0+A", "F0+B", "F0+A+B", "F0+Cw"]


def g(comp, target, col="mean"):
    r = sb[(sb.competitor == comp) & (sb.target == target)]
    return None if r.empty else r[col].iloc[0]


def fnum(v, plus=True):
    if v is None:
        return "--"
    return (f"{v:+.3f}" if plus else f"{v:.3f}")


print("% ===== Table C1: T3 gross-margin-change target (IC) =====")
for c in T1:
    print(f"{NAME[c]} & {fnum(g(c,'t3'))} & {g(c,'t3','nw_t'):+.2f} & {int(g(c,'t3','n_quarters'))} \\\\")
print("\\midrule")
for c in T2:
    print(f"{NAME[c]} & {fnum(g(c,'t3'))} & {g(c,'t3','nw_t'):+.2f} & {int(g(c,'t3','n_quarters'))} \\\\")

print("\n% ===== Table C2: net-income-direction target (AUC) =====")
for c in T1:
    print(f"{NAME[c]} & {fnum(g(c,'ni_dir(app)'),plus=False)} & {g(c,'ni_dir(app)','nw_t'):+.2f} & {int(g(c,'ni_dir(app)','n_quarters'))} \\\\")

print("\n% ===== Table C3: M&A +-2Q exclusion robustness (T1 h=4 IC) =====")
for c in T1 + T2:
    base, ex = g(c, "t1_h4"), g(c, "t1_h4_ma_excl")
    sep = "\\midrule\n" if c == "F0" else ""
    print(f"{sep}{NAME[c]} & {fnum(base)} & {fnum(ex)} \\\\")

print("\n% ===== Table C4: latest vs as-filed target (T1 h=4 IC) =====")
for c in T1:
    af, lt = g(c, "t1_h4"), g(c, "t1_h4_latest(app)")
    ratio = f"{lt/af:.2f}" if (af and af > 0) else "--"
    print(f"{NAME[c]} & {fnum(af)} & {fnum(lt)} & {ratio} \\\\")

print("\n% ===== Table C5: size-confound robustness (post-hoc) =====")


def sg(section, comp, target):
    r = sz[(sz.section == section) & (sz.competitor == comp) & (sz.target == target)]
    return (r["mean"].iloc[0], r["nw_t"].iloc[0]) if not r.empty else (None, None)


print("% -- panel A: diagnostics")
for lbl, sec, comp, tgt in [
        ("size $\\to$ growth, h=4", "diag", "size", "t1_h4"),
        ("size $\\to$ growth, h=8", "diag", "size", "t1_h8"),
        ("corr(A oracle, size)", "diag", "corr(A_orac,size)", "—"),
        ("corr(A feasible, size)", "diag", "corr(A_feas,size)", "—")]:
    m, t = sg(sec, comp, tgt)
    print(f"{lbl} & {m:+.3f} & {t:+.2f} \\\\")
print("% -- panel B: raw vs size-residualized A feature (T1 IC)")
for lbl, comp, tgt in [("A oracle, h=4", "A_orac", "t1_h4"),
                       ("A oracle, h=8", "A_orac", "t1_h8"),
                       ("A feasible, h=4", "A_feas", "t1_h4"),
                       ("A feasible, h=8", "A_feas", "t1_h8")]:
    rm, rt = sg("tier1_raw", comp, tgt)
    sm, st = sg("tier1_resid_size", comp, tgt)
    print(f"{lbl} & {rm:+.3f} ({rt:+.2f}) & {sm:+.3f} ({st:+.2f}) \\\\")
print("% -- panel C: Tier-2 marginal contribution over fundamentals+size (h=4)")
for lbl, comp in [("F0s+A feasible $-$ F0s", "F0s+A_feas - F0s"),
                  ("F0s+A oracle $-$ F0s", "F0s+A_orac - F0s"),
                  ("F0s $-$ F0", "F0s - F0")]:
    m, t = sg("tier2_marginal", comp, "t1_h4")
    print(f"{lbl} & {m:+.3f} & {t:+.2f} \\\\")
