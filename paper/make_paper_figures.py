"""Regenerate paper Figures 9 (IC time series) and 10 (headline marginal
contributions) as vector PDF, reusing the EXACT scoreboard plotting code
(run_scoreboard.make_figures) but feeding it data read from the committed
outputs/ic_timeseries.csv -- so the figure DATA is unchanged and only the
styling (larger fonts, vector format) differs.

The headline marginal contributions are reconstructed from the per-quarter IC
series exactly as run_scoreboard.main does (paired per-quarter differences,
Newey-West t with lag = horizon).
"""
import os
import sys

os.environ.setdefault("QVM_FIG_EXT", "pdf")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pandas as pd
from qvm import config
from qvm.analysis import scoreboard as sb
from run_scoreboard import make_figures

ts = pd.read_csv(os.path.join(config.OUTPUT_DIR, "ic_timeseries.csv"))


def pq_ic(comp):
    """Per-quarter t1_h4 IC series for a competitor, time-ordered."""
    s = ts[(ts.competitor == comp) & (ts.target == "t1_h4")]
    return s.set_index("quarter")["value"].sort_index()


def paired(a, b):
    d = (pq_ic(a) - pq_ic(b)).dropna().sort_index()
    return sb.nw_tstat(d.to_numpy(), 4)


# same order/keys as run_scoreboard.main -> identical bar layout & colors
headline = {}
for a, b, name in (("F0+A", "F0", "F0+A - F0"),
                   ("F0+A+B", "F0", "F0+A+B - F0"),
                   ("F0+Cw", "F0", "F0+Cw - F0")):
    headline[name] = paired(a, b)
headline["A_oracle - A_feasible"] = paired("A_orac", "A_feas")
headline["Cw - C (naive)"] = paired("Cw", "C")

print("Reconstructed headline (marginal IC, NW t, n):")
for k, (m, t, n) in headline.items():
    print(f"  {k:22s} {m:+.4f}  t={t:+.2f}  n={n}")

make_figures(ts, None, headline, None)
print("wrote fig7_ic_timeseries / fig8_headline ->", config.OUTPUT_DIR,
      "(ext=%s)" % os.environ["QVM_FIG_EXT"])
