
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))

INPUT = (
    DATA_ROOT / "processed"
    / "swaps_pilot_20260901_20260904.parquet"
)

FIGURE = ROOT / "reports/figures/04_observed_execution_cost.png"

EVENT_OUTPUT = (
    DATA_ROOT / "analytics"
    / "observed_execution_cost_pilot.parquet"
)

SUMMARY_OUTPUT = (
    DATA_ROOT / "analytics"
    / "observed_execution_cost_summary_pilot.csv"
)

START = pd.Timestamp("2026-09-01", tz="UTC")
END = pd.Timestamp("2026-09-04", tz="UTC")

FEES = ["0.05%", "0.30%"]
DIRECTIONS = ["BUY_WETH", "SELL_WETH"]

FEE_BPS = {
    "0.05%": 5.0,
    "0.30%": 30.0,
}

COLORS = {
    "0.05%": "#4A7BB7",
    "0.30%": "#F7834D",
}

SIZE_BINS = [100, 1_000, 10_000, 100_000, np.inf]
SIZE_LABELS = [
    "$100–1K",
    "$1K–10K",
    "$10K–100K",
    "≥ $100K",
]

MIN_GROUP_COUNT = 10


def spot_usdc_per_weth(sqrt_price_x96):
    if pd.isna(sqrt_price_x96):
        return np.nan

    value = int(sqrt_price_x96)

    if value <= 0:
        raise ValueError("Nonpositive sqrtPriceX96")

    ratio = value / (2 ** 96)

    # Token0: USDC (6 decimals)
    # Token1: WETH (18 decimals)
    return 1e12 / (ratio * ratio)


def load_and_calculate():
    df = pd.read_parquet(INPUT)

    required = {
        "event_id",
        "fee_tier",
        "pool_address",
        "timestamp_utc",
        "block_number",
        "log_index",
        "trade_direction",
        "trade_size_usd",
        "execution_price_usdc_per_weth",
        "sqrt_price_x96_after",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty or df["event_id"].duplicated().any():
        raise ValueError(
            "Empty dataset or duplicate events"
        )

    if set(df["fee_tier"]) != set(FEES):
        raise ValueError("Unexpected fee tiers")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    if not df["timestamp_utc"].between(
        START, END, inclusive="left"
    ).all():
        raise ValueError("Events outside pilot window")

    if df[["block_number", "log_index"]].isna().any().any():
        raise ValueError(
            "Missing block/log index prevents reliable ordering"
        )

    if df.duplicated(
        ["pool_address", "block_number", "log_index"]
    ).any():
        raise ValueError(
            "Duplicate pool/block/log combination"
        )

    sizes = df["trade_size_usd"].to_numpy()

    if not np.isfinite(sizes).all() or (sizes < 0).any():
        raise ValueError("Invalid trade sizes")

    df = df.sort_values(
        ["pool_address", "block_number", "log_index"]
    ).reset_index(drop=True)

    for _, part in df.groupby("pool_address"):
        if not part["timestamp_utc"].is_monotonic_increasing:
            raise ValueError(
                "Event ordering inconsistent with timestamps"
            )

    # Recover pool spot prices after each swap.
    df["spot_after_usdc_per_weth"] = (
        df["sqrt_price_x96_after"]
        .map(spot_usdc_per_weth)
    )

    # Previous swap's post-state is the next swap's
    # pre-state, within the same pool.
    df["spot_before_usdc_per_weth"] = (
        df.groupby("pool_address")[
            "spot_after_usdc_per_weth"
        ].shift(1)
    )

    df["observed_cost_bps"] = np.nan

    valid = (
        df["trade_direction"].isin(DIRECTIONS)
        & df["spot_before_usdc_per_weth"].notna()
        & df["execution_price_usdc_per_weth"].notna()
    )

    valid &= (
        df["execution_price_usdc_per_weth"] > 0
    )

    buy = valid & df["trade_direction"].eq("BUY_WETH")
    sell = valid & df["trade_direction"].eq("SELL_WETH")

    ratio = (
        df["execution_price_usdc_per_weth"]
        / df["spot_before_usdc_per_weth"]
    )

    df.loc[buy, "observed_cost_bps"] = (
        10000 * (ratio[buy] - 1)
    )

    df.loc[sell, "observed_cost_bps"] = (
        10000 * (1 - ratio[sell])
    )

    if not np.isfinite(
        df.loc[valid, "observed_cost_bps"]
    ).all():
        raise ValueError("Non-finite execution costs")

    if not np.isfinite(
        df["spot_after_usdc_per_weth"]
    ).all():
        raise ValueError("Non-finite spot prices")

    df["fee_baseline_bps"] = (
        df["fee_tier"].map(FEE_BPS)
    )

    df["above_fee_baseline_bps"] = (
        df["observed_cost_bps"]
        - df["fee_baseline_bps"]
    )

    prior = df["spot_before_usdc_per_weth"]
    after = df["spot_after_usdc_per_weth"]

    df["spot_direction_violation"] = (
        (buy & (after < prior * (1 - 1e-10)))
        | (sell & (after > prior * (1 + 1e-10)))
    )

    # Research sample: swaps >= $100.
    eligible = df.loc[
        valid & df["trade_size_usd"].ge(100)
    ].copy()

    eligible["size_bucket"] = pd.cut(
        eligible["trade_size_usd"],
        bins=SIZE_BINS,
        labels=SIZE_LABELS,
        right=False,
    )

    summary = (
        eligible.groupby(
            [
                "trade_direction",
                "fee_tier",
                "size_bucket",
            ],
            observed=True,
        )
        .agg(
            events=("event_id", "size"),
            median_cost_bps=(
                "observed_cost_bps", "median"
            ),
            p25_cost_bps=(
                "observed_cost_bps",
                lambda x: x.quantile(0.25),
            ),
            p75_cost_bps=(
                "observed_cost_bps",
                lambda x: x.quantile(0.75),
            ),
            negative_cost_pct=(
                "observed_cost_bps",
                lambda x: 100 * (x < 0).mean(),
            ),
        )
        .reset_index()
    )

    return df, eligible, summary


def create_figure(summary):
    FIGURE.parent.mkdir(
        parents=True, exist_ok=True
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6.6),
        sharey=True,
    )

    positions = np.arange(len(SIZE_LABELS))

    panels = [
        ("BUY_WETH", "A  |  Buy WETH"),
        ("SELL_WETH", "B  |  Sell WETH"),
    ]

    for ax, (direction, title) in zip(axes, panels):
        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
            fontsize=13,
            pad=14,
        )

        for tier, shift in [
            ("0.05%", -0.12),
            ("0.30%", 0.12),
        ]:
            part = summary[
                (summary["trade_direction"] == direction)
                & (summary["fee_tier"] == tier)
            ].set_index("size_bucket")

            for idx, size in enumerate(SIZE_LABELS):
                if size not in part.index:
                    continue

                item = part.loc[size]

                if int(item["events"]) < MIN_GROUP_COUNT:
                    continue

                med = float(item["median_cost_bps"])
                low = float(item["p25_cost_bps"])
                high = float(item["p75_cost_bps"])

                ax.errorbar(
                    idx + shift,
                    med,
                    yerr=[
                        [max(0, med - low)],
                        [max(0, high - med)],
                    ],
                    fmt="o",
                    color=COLORS[tier],
                    markersize=7,
                    capsize=4,
                    linewidth=1.7,
                    zorder=4,
                )

            ax.axhline(
                FEE_BPS[tier],
                color=COLORS[tier],
                linestyle="--",
                alpha=0.55,
                linewidth=1.0,
            )

        ax.set_xticks(
            positions,
            SIZE_LABELS,
            rotation=16,
        )

        ax.set_xlabel(
            "Swap size (USD)",
            labelpad=12,
        )

        ax.grid(
            axis="y",
            color="#E0E0E0",
            linewidth=0.8,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#B0B0B0")
        ax.spines["bottom"].set_color("#B0B0B0")

        ax.tick_params(
            length=0,
            pad=7,
            colors="#333333",
        )

    axes[0].set_ylabel(
        "Observed execution cost (bps)"
    )

    handles = [
        Line2D(
            [0], [0],
            color=COLORS[tier],
            marker="o",
            linewidth=1.5,
            label=tier,
        )
        for tier in FEES
    ]

    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.88),
        frameon=False,
        ncol=2,
    )

    fig.text(
        0.07, 0.95,
        "Uniswap V3 | Observed Execution Cost",
        fontsize=19,
        fontweight="bold",
        color="#1f1f1f",
    )

    fig.text(
        0.07, 0.90,
        "USDC/WETH · Sep 1–3, 2026 (UTC) "
        "· Median and interquartile range",
        fontsize=10.5,
        color="#555555",
    )

    fig.text(
        0.07, 0.058,
        "Source: Uniswap V3 Subgraph / The Graph  |  "
        "Min swap size $100; min 10 events per plotted group",
        fontsize=8.6,
        color="#666666",
    )

    fig.text(
        0.07, 0.025,
        "Previous swap post-price as reference; "
        "includes fee and price impact, excludes gas. "
        "Dashed: fee rates (5 / 30 bps).",
        fontsize=8.6,
        color="#666666",
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.80,
        bottom=0.19,
        wspace=0.13,
    )

    fig.savefig(
        FIGURE,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    df, eligible, summary = load_and_calculate()

    EVENT_OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_parquet(
        EVENT_OUTPUT,
        index=False,
        compression="zstd",
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    create_figure(summary)

    print("\n=== Observed Execution Cost Pilot ===")
    print(f"All swap events: {len(df):,}")
    print(f"Eligible events (>= $100): {len(eligible):,}")

    print(
        "Zero-leg swaps: "
        f"{(df['trade_direction'] == 'UNDETERMINED_ZERO_LEG').sum():,}"
    )

    print(
        "Missing pre-swap reference: "
        f"{df['spot_before_usdc_per_weth'].isna().sum():,}"
    )

    print(
        "Spot movement direction violations: "
        f"{df['spot_direction_violation'].sum():,}"
    )

    print("\n=== Cost by Size Bucket (bps) ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print(f"\nFigure: {FIGURE}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Event data: {EVENT_OUTPUT}")


if __name__ == "__main__":
    main()
