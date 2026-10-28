
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

INPUT = (
    DATA_ROOT / "analytics"
    / "observed_execution_cost_pilot.parquet"
)

FIGURE = ROOT / "reports/figures/05_price_impact_analysis.png"

SUMMARY = (
    DATA_ROOT / "analytics"
    / "price_impact_summary_pilot.csv"
)

EVENTS = (
    DATA_ROOT / "analytics"
    / "price_impact_events_pilot.parquet"
)

COLORS = {
    "0.05%": "#4A7BB7",
    "0.30%": "#F7834D",
}

FEE_RATES = {
    "0.05%": 0.0005,
    "0.30%": 0.003,
}

FEES = ["0.05%", "0.30%"]
DIRECTIONS = ["BUY_WETH", "SELL_WETH"]

BINS = [100, 1_000, 10_000, 100_000, np.inf]
LABELS = ["$100–1K", "$1K–10K", "$10K–100K", "≥ $100K"]

MIN_EVENTS = 10


def calculate():
    df = pd.read_parquet(INPUT)

    required = {
        "event_id", "fee_tier", "trade_direction",
        "trade_size_usd",
        "execution_price_usdc_per_weth",
        "spot_before_usdc_per_weth",
        "spot_after_usdc_per_weth",
        "observed_cost_bps",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty or df["event_id"].duplicated().any():
        raise ValueError("Empty or duplicated events")

    if set(df["fee_tier"]) != set(FEES):
        raise ValueError("Unexpected fee tier")

    df = df[
        df["trade_direction"].isin(DIRECTIONS)
        & (df["trade_size_usd"] >= 100)
    ].copy()

    df = df.dropna(subset=[
        "execution_price_usdc_per_weth",
        "spot_before_usdc_per_weth",
        "spot_after_usdc_per_weth",
        "observed_cost_bps",
    ])

    if df.empty:
        raise ValueError("No eligible events")

    effective = df["execution_price_usdc_per_weth"].to_numpy(
        dtype=float
    )

    before = df["spot_before_usdc_per_weth"].to_numpy(
        dtype=float
    )

    after = df["spot_after_usdc_per_weth"].to_numpy(
        dtype=float
    )

    fee = df["fee_tier"].map(FEE_RATES).to_numpy(
        dtype=float
    )

    is_buy = df["trade_direction"].eq("BUY_WETH").to_numpy()

    if (
        not np.isfinite(effective).all()
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or (effective <= 0).any()
        or (before <= 0).any()
        or (after <= 0).any()
    ):
        raise ValueError("Invalid price observations")

    # Remove the nominal input-token fee from
    # the effective execution price.
    no_fee_price = np.where(
        is_buy,
        effective * (1 - fee),
        effective / (1 - fee),
    )

    # Average execution-price deviation, excluding
    # the nominal swap fee.
    df["fee_adjusted_penalty_bps"] = np.where(
        is_buy,
        10000 * (no_fee_price / before - 1),
        10000 * (1 - no_fee_price / before),
    )

    # End-of-swap pool spot-price movement.
    df["terminal_spot_impact_bps"] = np.where(
        is_buy,
        10000 * (after / before - 1),
        10000 * (1 - after / before),
    )

    # Exact additive difference under these
    # execution-cost definitions.
    df["fee_contribution_bps"] = (
        df["observed_cost_bps"]
        - df["fee_adjusted_penalty_bps"]
    )

    metrics = [
        "fee_adjusted_penalty_bps",
        "terminal_spot_impact_bps",
        "fee_contribution_bps",
    ]

    if not np.isfinite(df[metrics].to_numpy()).all():
        raise ValueError("Non-finite calculated metrics")

    df["size_bucket"] = pd.cut(
        df["trade_size_usd"],
        bins=BINS,
        labels=LABELS,
        right=False,
    )

    records = []

    for (direction, tier, bucket), part in df.groupby(
        ["trade_direction", "fee_tier", "size_bucket"],
        observed=True,
    ):
        record = {
            "trade_direction": direction,
            "fee_tier": tier,
            "size_bucket": str(bucket),
            "events": len(part),
        }

        for metric in metrics:
            values = part[metric]
            record[f"{metric}_median"] = values.median()
            record[f"{metric}_p25"] = values.quantile(0.25)
            record[f"{metric}_p75"] = values.quantile(0.75)

        records.append(record)

    summary = pd.DataFrame(records)

    return df, summary


def style_axis(ax, title, ylabel):
    ax.set_title(
        title, loc="left",
        fontsize=12, fontweight="bold", pad=13,
    )

    ax.set_ylabel(ylabel, labelpad=9)
    ax.axhline(0, color="#888888", linewidth=0.9)

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
    ax.tick_params(length=0, colors="#333333", pad=7)


def create_figure(summary):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        sharey="row",
    )

    panels = [
        (
            0, 0, "BUY_WETH",
            "fee_adjusted_penalty_bps",
            "A | Buy WETH · Fee-adjusted penalty",
        ),
        (
            0, 1, "SELL_WETH",
            "fee_adjusted_penalty_bps",
            "B | Sell WETH · Fee-adjusted penalty",
        ),
        (
            1, 0, "BUY_WETH",
            "terminal_spot_impact_bps",
            "C | Buy WETH · Terminal spot movement",
        ),
        (
            1, 1, "SELL_WETH",
            "terminal_spot_impact_bps",
            "D | Sell WETH · Terminal spot movement",
        ),
    ]

    x = np.arange(len(LABELS))

    for row, col, direction, metric, title in panels:
        ax = axes[row, col]

        for tier, shift in [
            ("0.05%", -0.10),
            ("0.30%", 0.10),
        ]:
            part = summary[
                (summary["trade_direction"] == direction)
                & (summary["fee_tier"] == tier)
                & (summary["events"] >= MIN_EVENTS)
            ].set_index("size_bucket")

            positions = []
            medians = []
            lower = []
            upper = []

            for i, bucket in enumerate(LABELS):
                if bucket not in part.index:
                    continue

                record = part.loc[bucket]

                med = record[f"{metric}_median"]
                p25 = record[f"{metric}_p25"]
                p75 = record[f"{metric}_p75"]

                positions.append(i + shift)
                medians.append(med)
                lower.append(max(0, med - p25))
                upper.append(max(0, p75 - med))

            if positions:
                ax.errorbar(
                    positions,
                    medians,
                    yerr=[lower, upper],
                    fmt="o-",
                    color=COLORS[tier],
                    linewidth=1.8,
                    markersize=6,
                    capsize=3,
                    label=tier,
                )

        ylabel = (
            "Fee-adjusted penalty (bps)"
            if row == 0
            else "Terminal spot movement (bps)"
        )

        style_axis(ax, title, ylabel)

        ax.set_xticks(x)
        ax.set_xticklabels(
            LABELS,
            rotation=15,
            ha="right",
        )

        ax.set_xlim(-0.4, len(LABELS) - 0.6)
        ax.set_xlabel("Swap size (USD)", labelpad=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.895),
        frameon=False,
        ncol=2,
    )

    fig.text(
        0.065, 0.968,
        "Uniswap V3 | Price Impact Analysis",
        fontsize=19,
        fontweight="bold",
        color="#1f1f1f",
    )

    fig.text(
        0.065, 0.937,
        "USDC/WETH · Sep 1–3, 2026 (UTC) "
        "· Median and interquartile range",
        fontsize=10.5,
        color="#555555",
    )

    fig.text(
        0.065, 0.029,
        "Source: Uniswap V3 Subgraph / The Graph  |  "
        "Swaps ≥ $100  |  Minimum 10 events per plotted group  |  "
        "Gas excluded",
        fontsize=8.5,
        color="#666666",
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.83,
        bottom=0.13,
        hspace=0.42,
        wspace=0.18,
    )

    fig.savefig(
        FIGURE,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    df, summary = calculate()

    EVENTS.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        EVENTS, index=False, compression="zstd"
    )

    summary.to_csv(SUMMARY, index=False)

    create_figure(summary)

    print("\n=== Price Impact Analysis ===")
    print(f"Eligible swap events: {len(df):,}")

    print("\n=== Median Impacts by Trade Size (bps) ===")

    cols = [
        "trade_direction",
        "fee_tier",
        "size_bucket",
        "events",
        "fee_adjusted_penalty_bps_median",
        "terminal_spot_impact_bps_median",
    ]

    print(
        summary[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.3f}",
        )
    )

    print("\n=== Validation ===")

    print(
        "Negative terminal movements:",
        int((df["terminal_spot_impact_bps"] < -1e-8).sum()),
    )

    print(
        "Negative fee-adjusted penalties:",
        int((df["fee_adjusted_penalty_bps"] < -1e-8).sum()),
    )

    print(f"\nFigure: {FIGURE}")
    print(f"Summary: {SUMMARY}")


if __name__ == "__main__":
    main()
