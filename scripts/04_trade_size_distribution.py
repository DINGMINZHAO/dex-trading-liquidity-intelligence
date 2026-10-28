import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
INPUT_PATH = (
    DATA_ROOT / "processed/swaps_pilot_20260901_20260904.parquet"
)

FIGURE_PATH = ROOT / "reports/figures/02_trade_size_distribution.png"
SUMMARY_PATH = DATA_ROOT / "analytics/trade_size_summary_pilot.csv"
BUCKET_PATH = DATA_ROOT / "analytics/trade_size_buckets_pilot.csv"

FEES = ["0.05%", "0.30%"]

# Python / Matplotlib-like standard palette
COLORS = {
    "0.05%": "#1f77b4",  # blue
    "0.30%": "#ff7f0e",  # orange
}

EDGES = [
    0, 1, 10, 100, 1_000, 10_000,
    100_000, 1_000_000, np.inf,
]

LABELS = [
    "< $1",
    "$1–10",
    "$10–100",
    "$100–1K",
    "$1K–10K",
    "$10K–100K",
    "$100K–1M",
    "≥ $1M",
]

START = pd.Timestamp("2026-09-01", tz="UTC")
END = pd.Timestamp("2026-09-04", tz="UTC")

def read_data():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Pilot data not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    required = {
        "event_id",
        "fee_tier",
        "trade_size_usd",
        "timestamp_utc",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty")

    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs detected")

    if set(df["fee_tier"]) != set(FEES):
        raise ValueError("Unexpected fee tiers in dataset")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    if not df["timestamp_utc"].between(
        START, END, inclusive="left"
    ).all():
        raise ValueError("Events outside pilot window")

    size = pd.to_numeric(
        df["trade_size_usd"], errors="coerce"
    )

    if not np.isfinite(size).all() or (size < 0).any():
        raise ValueError("Invalid trade size values")

    df["trade_size_usd"] = size

    return df

def calculate_metrics(df):
    summaries = []

    for tier in FEES:
        trades = df.loc[
            df["fee_tier"] == tier,
            "trade_size_usd",
        ]

        n = len(trades)
        total = trades.sum()

        top_n = max(1, math.ceil(n * 0.01))
        largest = trades.nlargest(top_n)

        summaries.append({
            "fee_tier": tier,
            "swap_events": n,
            "volume_usd": total,
            "median_trade_usd": trades.median(),
            "p90_trade_usd": trades.quantile(0.90),
            "p99_trade_usd": trades.quantile(0.99),
            "under_1_usd_event_pct": 100 * (trades < 1).mean(),
            "top_1pct_events_volume_share_pct": (
                100 * largest.sum() / total if total > 0 else np.nan
            ),
        })

    df = df.copy()

    df["size_bucket"] = pd.cut(
        df["trade_size_usd"],
        bins=EDGES,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )

    buckets = (
        df.groupby(
            ["fee_tier", "size_bucket"],
            observed=False,
        )
        .agg(
            swap_events=("event_id", "size"),
            volume_usd=("trade_size_usd", "sum"),
        )
        .reset_index()
    )

    event_totals = df.groupby("fee_tier").size()
    volume_totals = df.groupby("fee_tier")["trade_size_usd"].sum()

    buckets["event_share_pct"] = (
        100
        * buckets["swap_events"]
        / buckets["fee_tier"].map(event_totals)
    )

    buckets["volume_share_pct"] = (
        100
        * buckets["volume_usd"]
        / buckets["fee_tier"].map(volume_totals)
    )

    return pd.DataFrame(summaries), buckets

def style_axis(ax):
    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")

    ax.tick_params(
        axis="both",
        colors="#333333",
        length=0,
        pad=7,
    )

def create_figure(buckets):
    FIGURE_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13.5, 8.8),
        sharex=True,
    )

    x = np.arange(len(LABELS))
    width = 0.36

    panels = [
        (
            "event_share_pct",
            "Share of swap events (%)",
            "A  |  Share of swap events",
        ),
        (
            "volume_share_pct",
            "Share of trading volume (%)",
            "B  |  Share of trading volume",
        ),
    ]

    for ax, (column, ylabel, heading) in zip(
        axes, panels
    ):
        for i, tier in enumerate(FEES):
            part = (
                buckets[buckets["fee_tier"] == tier]
                .set_index("size_bucket")
            )

            values = (
                part.reindex(LABELS)[column]
                .to_numpy(dtype=float)
            )

            ax.bar(
                x + (i - 0.5) * width,
                values,
                width=width,
                color=COLORS[tier],
                label=tier,
                zorder=3,
            )

        ax.set_title(
            heading,
            loc="left",
            fontweight="bold",
            pad=12,
        )
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_ylim(bottom=0)

        style_axis(ax)

    axes[0].legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        fontsize=10,
    )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        LABELS,
        rotation=20,
        ha="right",
    )
    axes[1].set_xlabel(
        "Swap size in USD (logarithmic buckets)",
        labelpad=12,
    )

    fig.text(
        0.075, 0.965,
        "Uniswap V3 | Trade Size Distribution",
        fontsize=20,
        fontweight="bold",
        color="#1f1f1f",
    )

    fig.text(
        0.075, 0.93,
        "USDC/WETH · 0.05% vs 0.30% fee tier · Sep 1–3, 2026 (UTC)",
        fontsize=11,
        color="#555555",
    )

    fig.text(
        0.075, 0.032,
        "Source: Uniswap V3 Subgraph / The Graph  |  "
        "Unit: pool-level Swap events  |  Zero-leg events retained",
        fontsize=8.8,
        color="#666666",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.98,
        top=0.86,
        bottom=0.16,
        hspace=0.32,
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

def main():
    df = read_data()
    summary, buckets = calculate_metrics(df)

    SUMMARY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    summary.to_csv(SUMMARY_PATH, index=False)
    buckets.to_csv(BUCKET_PATH, index=False)

    create_figure(buckets)

    print("\n=== Trade Size Summary: Sep 1–3, 2026 (UTC) ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\n=== Bucket Shares: Events (%) / Volume (%) ===")
    print(
        buckets[
            [
                "fee_tier",
                "size_bucket",
                "event_share_pct",
                "volume_share_pct",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(f"\nPNG: {FIGURE_PATH}")
    print(f"Metrics: {SUMMARY_PATH}")

if __name__ == "__main__":
    main()