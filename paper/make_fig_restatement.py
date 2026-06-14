"""Generate paper Figure 2 (restatement inflation) from REAL scoreboard data.

Source: outputs/scoreboard.csv — the pre-registered Step-5b scoreboard.
We plot the two momentum baselines' 4Q-ahead Spearman IC under the as-filed
(point-in-time) target vs the latest (restated) target. The gap is the
restatement-inflation finding of Section 3.4.

No numbers are hand-entered; everything is read from the CSV.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, "..", "outputs", "scoreboard.csv")
# vector PDF for the paper (arXiv-preferred); QVM_FIG_EXT=png to override
_EXT = os.environ.get("QVM_FIG_EXT", "pdf")
OUT = os.path.join(HERE, "figures", f"fig02_restatement.{_EXT}")

df = pd.read_csv(SB)

# D1 = revenue-momentum baseline; D2 = its prior-year quintile.
labels = {"D1": "Revenue\nmomentum", "D2": "Prior-year\nquintile"}
asfiled, latest = [], []
for comp in ("D1", "D2"):
    af = df[(df.competitor == comp) & (df.target == "t1_h4")]["mean"].iloc[0]
    lt = df[(df.competitor == comp) & (df.target == "t1_h4_latest(app)")]["mean"].iloc[0]
    asfiled.append(af)
    latest.append(lt)
    print(f"{comp}: as-filed IC={af:+.4f}  latest IC={lt:+.4f}  ratio={lt/af:.2f}x")

x = range(len(labels))
w = 0.36
fig, ax = plt.subplots(figsize=(7.2, 4.6))
b1 = ax.bar([i - w / 2 for i in x], asfiled, w, label="As-filed (point-in-time)",
            color="#4c72b0")
b2 = ax.bar([i + w / 2 for i in x], latest, w, label="Latest (restated)",
            color="#dd8452")
for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 0.0015, f"{h:+.3f}",
                ha="center", va="bottom", fontsize=11)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels([labels[c] for c in ("D1", "D2")], fontsize=12)
ax.set_ylabel("Spearman IC (4Q-ahead revenue-growth rank)")
ax.set_title("Restatements inflate momentum backtests")
ax.legend(frameon=False, fontsize=11, loc="upper left")
ax.set_ylim(0, max(latest) * 1.25)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("wrote", OUT)
