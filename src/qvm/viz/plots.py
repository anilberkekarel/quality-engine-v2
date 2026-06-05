"""Whitepaper-grade visualizations of the RAW patent rhythm (no signal proc).

Figures
  Fig1a  four companies' FILING-date YEARLY series, raw counts, one chart.
  Fig1b  same, each company NORMALIZED to its own max (0-1) so the RHYTHMS are
         comparable despite Micron's far larger volume (an outlier-scale fix,
         echoing V1's normalization step).
  Fig2   NVIDIA's three date series (filing / publication / grant) overlaid,
         to SEE the inter-date lag.

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
