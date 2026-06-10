"""Whitepaper-grade visualizations of the RAW patent rhythm (no signal proc).

Figures
  Fig1a  four companies' FILING-date YEARLY series, raw counts, one chart.
  Fig1b  same, each company NORMALIZED to its own max (0-1) so the RHYTHMS are
         comparable despite Micron's far larger volume (an outlier-scale fix,
         echoing V1's normalization step).
  Fig2   NVIDIA's three date series (filing / publication / grant) overlaid,
         to SEE the inter-date lag.
  Fig3   the two-sensor EYE TEST (pre-model): per company, quarterly patent
         filings vs gross margin vs revenue YoY growth on one time axis —
         do the channels break together (NVDA) or decouple (MU control)?
  Fig4   (per company) NB-HMM regimes: counts with smoothed high-regime
         shading + filtered (causal) probability lower panel.

The trailing INCOMPLETE_TRAILING_YEARS of the filing series are drawn dashed
and shaded ("incomplete: filings not yet published") — a data artifact, not a
real decline (evidence for research question c).

Style targets arXiv-placeable quality: clean grid, labeled axes, titles,
sourced caption, consistent colorblind-friendly palette, high-DPI PNG.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

# Stable colors for companies (Fig1) and for date dimensions (Fig2).
_COMPANY_COLOR = {
    "NVDA": "#76b900",  # NVIDIA green
    "AMD":  "#ed1c24",
    "MRVL": "#1f77b4",
    "MU":   "#7f7f7f",  # control -> neutral grey
}
_DIM_STYLE = {
    "filing_date":      {"color": "#1f77b4", "label": "Filing (innovation, secret)"},
    "publication_date": {"color": "#ff7f0e", "label": "Publication (~18mo, public)"},
    "grant_date":       {"color": "#2ca02c", "label": "Grant (~2-3y, confirmed)"},
}

_CAPTION = ("Source: BigQuery patents-public-data.patents.publications "
            "(US, assignee_harmonized). RAW counts — NAIVE BASELINE, no signal processing.")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _incomplete_cutoff_year(filing_df) -> int | None:
    inc = filing_df[filing_df["incomplete"]]
    return int(inc["year"].min()) if not inc.empty else None


def _plot_company_line(ax, comp, color, normalize=False):
    """Plot one company's yearly filing series: solid (complete) + dashed (tail)."""
    comp = comp.sort_values("year")
    y = comp["patent_count"].astype(float).values
    if normalize:
        complete = comp[~comp["incomplete"]]["patent_count"]
        denom = complete.max() if not complete.empty else comp["patent_count"].max()
        y = y / denom if denom else y
    years = comp["year"].values
    incomplete = comp["incomplete"].values
    label = comp["label"].iloc[0]
    # solid over complete portion (include first incomplete point for continuity)
    last_complete = None
    for i, inc in enumerate(incomplete):
        if not inc:
            last_complete = i
    if last_complete is None:
        ax.plot(years, y, color=color, lw=1.9, marker="o", ms=3, label=label)
        return
    ax.plot(years[:last_complete + 1], y[:last_complete + 1],
            color=color, lw=1.9, marker="o", ms=3, label=label)
    if last_complete < len(years) - 1:
        ax.plot(years[last_complete:], y[last_complete:],
                color=color, lw=1.5, ls="--", marker="o", ms=3, alpha=0.8)


def _filing_yearly(baseline):
    return baseline[(baseline["granularity"] == "year")
                    & (baseline["date_dimension"] == "filing_date")]


def plot_filing_companies(baseline, out_dir, company_order=None):
    """Fig1a (raw) + Fig1b (normalized). Returns (raw_path, norm_path)."""
    filing = _filing_yearly(baseline)
    tickers = company_order or list(dict.fromkeys(filing["ticker"]))
    cutoff = _incomplete_cutoff_year(filing)
    paths = []
    for normalize, tag, ylab, title in (
        (False, "raw", "Patents filed per year",
         "Filing-date patent rhythm — raw counts"),
        (True, "normalized", "Filings per year (normalized to own max)",
         "Filing-date patent rhythm — normalized (rhythms comparable)"),
    ):
        fig, ax = plt.subplots(figsize=(11, 6))
        for ticker in tickers:
            comp = filing[filing["ticker"] == ticker]
            if comp.empty:
                continue
            _plot_company_line(ax, comp, _COMPANY_COLOR.get(ticker, "#333"),
                               normalize=normalize)
        ax.set_xlim(left=config.PLOT_START_YEAR)
        if cutoff is not None:
            xmax = filing["year"].max()
            ax.axvspan(cutoff - 0.5, xmax + 0.5, color="grey", alpha=0.12)
            ax.text(cutoff, ax.get_ylim()[1] * 0.97,
                    " incomplete\n (not yet published)", fontsize=7.5,
                    va="top", ha="left", color="#555", style="italic")
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=9, title="Company (— complete  -- incomplete)")
        fig.text(0.01, -0.02, _CAPTION, fontsize=7, style="italic", color="#555")
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig1_{tag}_filing_companies.png")
        _ensure_dir(path)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def plot_channels_eyetest(channels, out_dir, company_order=None):
    """Fig3: per-company panel — patent filings (left axis) vs gross margin &
    revenue YoY growth (right axis), quarterly, shared time axis.

    Financial values are AS-FILED (point-in-time); the incomplete patent tail
    is shaded and dashed exactly as in Fig1. Returns the PNG path.
    """
    import pandas as pd

    ch = channels.copy()
    ch["t"] = pd.to_datetime(ch["period_start"])
    tickers = company_order or list(dict.fromkeys(ch["ticker"]))
    fig, axes = plt.subplots(len(tickers), 1, figsize=(11, 3.0 * len(tickers)),
                             sharex=True)
    for ax, ticker in zip(axes, tickers):
        g = ch[ch["ticker"] == ticker].sort_values("t")
        label = g["label"].iloc[0]
        color = _COMPANY_COLOR.get(ticker, "#333")
        complete = g[~g["patent_incomplete"]]
        tail = g[g["patent_incomplete"]]
        ax.plot(complete["t"], complete["patent_filing_count"], color=color,
                lw=1.8, label="Patent filings / quarter")
        if not tail.empty:
            bridge = pd.concat([complete.tail(1), tail])
            ax.plot(bridge["t"], bridge["patent_filing_count"], color=color,
                    lw=1.4, ls="--", alpha=0.75)
            ax.axvspan(tail["t"].iloc[0], g["t"].iloc[-1],
                       color="grey", alpha=0.12)
        ax.set_ylabel("Filings / quarter", fontsize=8.5)
        ax.set_title(f"{label} ({ticker})", fontsize=10, loc="left")

        axr = ax.twinx()
        axr.grid(False)
        axr.spines.top.set_visible(False)
        fin = g[g["gross_margin"].notna()]
        axr.plot(fin["t"], 100 * fin["gross_margin"], color="#ff7f0e",
                 lw=1.5, label="Gross margin (as filed)")
        yoy = g[g["revenue_yoy_growth"].notna()]
        axr.plot(yoy["t"], 100 * yoy["revenue_yoy_growth"], color="#9467bd",
                 lw=1.1, alpha=0.85, label="Revenue YoY growth")
        axr.axhline(0, color="#999", lw=0.7, ls=":")
        axr.set_ylabel("%", fontsize=8.5)
        if ax is axes[0]:
            lines = ax.get_lines()[:1] + axr.get_lines()[:2]
            ax.legend(lines, [l.get_label() for l in lines], frameon=False,
                      fontsize=8, loc="upper left", ncol=3)
    axes[-1].set_xlabel("Calendar quarter")
    fig.suptitle("Two-sensor eye test — patent rhythm vs financial channel "
                 "(pre-model, raw aligned series)", fontweight="bold", y=0.995)
    fig.text(0.01, -0.01, _CAPTION + " Financials: SEC EDGAR XBRL, as-filed "
             "values, calendar-quarter aligned by period end. Shaded: "
             "incomplete patent tail (18-month publication secrecy).",
             fontsize=7, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = os.path.join(out_dir, "fig3_channels_eyetest.png")
    _ensure_dir(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regimes(res: dict, out_dir: str) -> str:
    """Fig4 (one per company): quarterly counts with smoothed high-regime
    periods shaded, plus the FILTERED (causal) probability in a lower panel.

    `res` is one element of qvm.analysis.regimes.analyze_all output.
    Returns the PNG path.
    """
    import numpy as np

    quarters, y, fit = res["quarters"], res["y"], res["nb2"]
    t = quarters.to_timestamp()
    color = _COMPANY_COLOR.get(res["ticker"], "#333")

    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08})
    # shade smoothed high-regime periods (retrospective view)
    in_high = fit.smoothed[:, 1] >= 0.5
    start = None
    for i in range(len(in_high) + 1):
        if i < len(in_high) and in_high[i] and start is None:
            start = i
        elif (i == len(in_high) or not in_high[i]) and start is not None:
            ax.axvspan(t[start], t[min(i, len(t) - 1)], color=color, alpha=0.14)
            start = None
    ax.plot(t, y, color=color, lw=1.7, marker="o", ms=2.6,
            label="Patent filings / quarter (complete only)")
    mu0, mu1 = fit.mus
    ax.axhline(mu0, color="#888", lw=0.9, ls=":", label=f"$\\mu_0$={mu0:.0f} (low)")
    ax.axhline(mu1, color="#444", lw=0.9, ls="--", label=f"$\\mu_1$={mu1:.0f} (high)")
    ax.set_ylabel("Filings / quarter")
    ax.set_title(f"{res['label']} ({res['ticker']}) — NB-HMM regimes "
                 f"(shaded: smoothed P(high)$\\geq$0.5)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    axp.plot(t, fit.filtered[:, 1], color="#d62728", lw=1.5,
             label="Filtered P(high | data up to t) — causal")
    axp.plot(t, fit.smoothed[:, 1], color="#555", lw=1.0, ls="--", alpha=0.8,
             label="Smoothed P(high | all data) — retrospective")
    axp.axhline(0.5, color="#999", lw=0.8, ls=":")
    axp.set_ylim(-0.04, 1.04)
    axp.set_ylabel("P(high regime)")
    axp.set_xlabel("Quarter")
    axp.legend(frameon=False, fontsize=8, loc="center left")

    fig.text(0.01, -0.02, _CAPTION + f" Model: 2-state NB-HMM, {res['note']}. "
             "Caveat: parameters are full-sample — filtered probs are causal "
             "in observations only (true OOS = expanding window, Step 5).",
             fontsize=7, style="italic", color="#555")
    path = os.path.join(out_dir, f"fig4_regimes_{res['ticker']}.png")
    _ensure_dir(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation_filtered(results: list, markers: dict, out_dir: str) -> str:
    """Fig5 (whitepaper centerpiece candidate): per company, the three
    models' FILTERED P(high) curves overlaid + data-derived event markers.

    A = patent-only (Step 3, reused), B = financial-only, C = joint.
    Returns the PNG path.
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.9 * len(results)),
                             sharex=True)
    for ax, res in zip(axes, results):
        t = res["quarters"].to_timestamp()
        a_filt = res["step3"]["nb2"].filtered[:, 1]
        ax.plot(t, a_filt, color="#999", lw=1.2, ls="--",
                label="A patent-only (Step 3)")
        ax.plot(t, res["B"].filtered[:, 1], color="#1f77b4", lw=1.2, ls=":",
                label="B financial-only")
        ax.plot(t, res["C"].filtered[:, 1],
                color=_COMPANY_COLOR.get(res["ticker"], "#333"), lw=2.0,
                label="C joint (3 channels)")
        ax.axhline(0.5, color="#bbb", lw=0.7)
        for q, text in markers.get(res["ticker"], []):
            import pandas as pd
            tq = pd.Period(q, freq="Q").to_timestamp()
            if t[0] <= tq <= t[-1]:
                ax.axvline(tq, color="#d62728", lw=1.0, ls="-.", alpha=0.8)
                ax.text(tq, 1.06, f" {text}", fontsize=7, color="#d62728",
                        ha="left", va="bottom")
        ax.set_ylim(-0.05, 1.18)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("P(high) filtered")
        ax.set_title(f"{res['label']} ({res['ticker']})", fontsize=10, loc="left")
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8, loc="center left", ncol=3)
    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Ablation — filtered P(high regime): patent-only vs "
                 "financial-only vs joint", fontweight="bold", y=0.995)
    fig.text(0.01, -0.01, _CAPTION + " Models: 2-state HMMs (NB counts / "
             "Gaussian financials), full-sample parameters — DESCRIPTIVE; "
             "real-time version is Step 5. Event markers derived by rule "
             "(MU: as-filed gross margin -15pp YoY episode starts; NVDA: "
             "first revenue YoY >= +100%).",
             fontsize=7, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = os.path.join(out_dir, "fig5_ablation_filtered.png")
    _ensure_dir(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_state_profiles(results: list, out_dir: str) -> str:
    """Fig6 (compact): state-conditional channel means of the JOINT model —
    4 companies x 2 states x 3 channels, one glance at "does 'high' mean the
    same thing everywhere?". Returns the PNG path.
    """
    import numpy as np

    channels = [("patent_count", "Patent filings / q", "mus", "{:.0f}"),
                ("gross_margin", "Gross margin", "means", "{:.1%}"),
                ("revenue_yoy", "Revenue YoY", "means", "{:+.1%}")]
    tickers = [res["ticker"] for res in results]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (name, title, key, fmt) in zip(axes, channels):
        mat = np.array([[res["C"].channel_params[name][key][k]
                         for k in (0, 1)] for res in results], dtype=float)
        # color: within-company scaling, so the low->high DIRECTION pops
        norm = (mat - mat.min(axis=1, keepdims=True)) / np.maximum(
            mat.max(axis=1, keepdims=True) - mat.min(axis=1, keepdims=True),
            1e-12)
        ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(2):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1], ["low state", "high state"], fontsize=8.5)
        ax.set_yticks(range(len(tickers)), tickers, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Joint-model state profiles — is the 'high' state the same "
                 "thing in every company?", fontweight="bold")
    fig.text(0.01, -0.04, "States ordered by patent mean (state 1 = high "
             "patent tempo). Color: within-company scaling, green = the "
             "larger value. A high-patent state whose financial cells are "
             "red = patents and financials DISAGREE in that company. "
             + _CAPTION, fontsize=7, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, "fig6_state_profiles.png")
    _ensure_dir(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_three_dates(baseline, out_dir, ticker="NVDA"):
    """Fig2: one company's three date series overlaid (yearly). Returns path."""
    yearly = baseline[(baseline["granularity"] == "year")
                      & (baseline["ticker"] == ticker)]
    label = yearly["label"].iloc[0] if not yearly.empty else ticker
    fig, ax = plt.subplots(figsize=(11, 6))
    for dim, style in _DIM_STYLE.items():
        s = yearly[yearly["date_dimension"] == dim].sort_values("year")
        if s.empty:
            continue
        ax.plot(s["year"], s["patent_count"], color=style["color"],
                label=style["label"], lw=1.9, marker="o", ms=3.5)
    ax.set_xlim(left=config.PLOT_START_YEAR)
    ax.set_title(f"{label} ({ticker}) — three date series (lag is visible)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Patents per year")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, -0.02, _CAPTION, fontsize=7, style="italic", color="#555")
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig2_three_dates_{ticker}.png")
    _ensure_dir(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
