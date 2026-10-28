
import os
from datetime import timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))

INPUT = (
    DATA_ROOT
    / "processed/swaps_pilot_20260901_20260904.parquet"
)

FIGURE = ROOT / "reports/figures/03_intraday_activity.png"

ANALYTICS = DATA_ROOT / "analytics"

START = pd.Timestamp("2026-09-01", tz="UTC")
END = pd.Timestamp("2026-09-04", tz="UTC")

FEES = ["0.05%", "0.30%"]
DIRECTIONS = ["BUY_WETH", "SELL_WETH"]

# Project research palette
PALETTE = {
    "red": "#D9412B",
    "orange": "#F7834D",
    "yellow": "#FED081",
    "pale_yellow": "#FFFDBB",
    "pale_cyan": "#D2EDF4",
    "light_blue": "#91C4DE",
    "blue": "#4A7BB7",
}

COLORS = {
    "0.05%": PALETTE["blue"],
    "0.30%": PALETTE["orange"],
    "BUY_WETH": PALETTE["light_blue"],
    "SELL_WETH": PALETTE["red"],
}


# ============================================================
# Data Validation
# ============================================================

def load_data():
    df = pd.read_parquet(INPUT)

    required = {
        "event_id",
        "fee_tier",
        "timestamp_utc",
        "trade_size_usd",
        "trade_direction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty or df["event_id"].duplicated().any():
        raise ValueError(
            "Empty dataset or duplicate Swap events"
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    if not df["timestamp_utc"].between(
        START,
        END,
        inclusive="left",
    ).all():
        raise ValueError("Events outside pilot window")

    if set(df["fee_tier"]) != set(FEES):
        raise ValueError("Unexpected fee tiers")

    valid_directions = set(DIRECTIONS) | {
        "UNDETERMINED_ZERO_LEG"
    }

    if not set(df["trade_direction"]).issubset(
        valid_directions
    ):
        raise ValueError("Unexpected trade directions")

    volume = df["trade_size_usd"].to_numpy()

    if not np.isfinite(volume).all():
        raise ValueError("Non-finite trading volume")

    if (volume < 0).any():
        raise ValueError("Negative trading volume")

    return df


# ============================================================
# Hourly Trading Activity
# ============================================================

def hourly_activity(df):
    data = df.copy()

    data["hour"] = data["timestamp_utc"].dt.floor("h")

    hourly = (
        data.groupby(["fee_tier", "hour"])
        .agg(
            swap_events=("event_id", "size"),
            volume_usd=("trade_size_usd", "sum"),
        )
    )

    hours = pd.date_range(
        START,
        END,
        freq="h",
        inclusive="left",
    )

    full_index = pd.MultiIndex.from_product(
        [FEES, hours],
        names=["fee_tier", "hour"],
    )

    hourly = (
        hourly.reindex(full_index, fill_value=0)
        .reset_index()
    )

    for source, target in [
        ("swap_events", "event_activity_index"),
        ("volume_usd", "volume_activity_index"),
    ]:
        mean = hourly.groupby("fee_tier")[
            source
        ].transform("mean")

        if (mean <= 0).any():
            raise ValueError("Zero hourly mean")

        hourly[target] = hourly[source] / mean

    return hourly


# ============================================================
# Buy/Sell Direction Analysis
# ============================================================

def direction_analysis(df):
    classified = df[
        df["trade_direction"].isin(DIRECTIONS)
    ].copy()

    direction = (
        classified.groupby(
            ["fee_tier", "trade_direction"]
        )
        .agg(
            swap_events=("event_id", "size"),
            volume_usd=("trade_size_usd", "sum"),
        )
    )

    index = pd.MultiIndex.from_product(
        [FEES, DIRECTIONS],
        names=["fee_tier", "trade_direction"],
    )

    direction = (
        direction.reindex(index, fill_value=0)
        .reset_index()
    )

    for source, target in [
        ("swap_events", "event_share_pct"),
        ("volume_usd", "volume_share_pct"),
    ]:
        total = direction.groupby("fee_tier")[
            source
        ].transform("sum")

        if (total <= 0).any():
            raise ValueError("No classified activity")

        direction[target] = (
            100 * direction[source] / total
        )

    return direction


# ============================================================
# Visualization Styling
# ============================================================

def style_axis(ax, title, ylabel):
    ax.set_title(
        title,
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    ax.set_ylabel(ylabel, labelpad=10)

    ax.grid(
        axis="y",
        color="#E0E0E0",
        linewidth=0.8,
        alpha=0.85,
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


# ============================================================
# Research Figure
# ============================================================

def create_figure(hourly, direction):
    FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9.5),
    )

    # --------------------------------------------------------
    # Panel A & B: Hourly Activity
    # --------------------------------------------------------

    line_panels = [
        (
            axes[0, 0],
            "event_activity_index",
            "A  |  Hourly Swap Activity",
        ),
        (
            axes[0, 1],
            "volume_activity_index",
            "B  |  Hourly Trading Volume",
        ),
    ]

    for ax, column, title in line_panels:

        for tier in FEES:
            part = hourly[
                hourly["fee_tier"] == tier
            ]

            ax.plot(
                part["hour"],
                part[column],
                label=tier,
                color=COLORS[tier],
                linewidth=2.0,
                alpha=0.95,
            )

        ax.axhline(
            1,
            color="#999999",
            linestyle="--",
            linewidth=1,
            alpha=0.75,
        )

        ax.xaxis.set_major_locator(
            mdates.HourLocator(
                byhour=[0, 12],
                tz=timezone.utc,
            )
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%b %d\n%H:%M",
                tz=timezone.utc,
            )
        )

        ax.set_xlim(START, END)
        ax.set_ylim(bottom=0)

        style_axis(
            ax,
            title,
            "Activity index (mean = 1)",
        )

        ax.legend(
            frameon=False,
            ncol=2,
            loc="upper right",
            fontsize=10,
            handlelength=2,
            columnspacing=1.5,
        )

    # --------------------------------------------------------
    # Panel C & D: Buy/Sell Composition
    # --------------------------------------------------------

    bar_panels = [
        (
            axes[1, 0],
            "event_share_pct",
            "C  |  Buy/Sell Event Share",
        ),
        (
            axes[1, 1],
            "volume_share_pct",
            "D  |  Buy/Sell Volume Share",
        ),
    ]

    for ax, column, title in bar_panels:

        bottom = np.zeros(len(FEES))

        for trade_direction in DIRECTIONS:

            part = (
                direction[
                    direction["trade_direction"]
                    == trade_direction
                ]
                .set_index("fee_tier")
                .reindex(FEES)
            )

            values = part[column].to_numpy()

            bars = ax.bar(
                FEES,
                values,
                bottom=bottom,
                width=0.48,
                color=COLORS[trade_direction],
                edgecolor="white",
                linewidth=0.5,
            )

            text_color = (
                "#253323"
                if trade_direction == "BUY_WETH"
                else "white"
            )

            for bar, value, base in zip(
                bars, values, bottom
            ):
                if value >= 5:
                    ax.text(
                        bar.get_x()
                        + bar.get_width() / 2,
                        base + value / 2,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                        fontsize=11,
                    )

            bottom += values

        style_axis(
            ax,
            title,
            "Share of classified swaps (%)",
        )

        ax.set_ylim(0, 105)

    # --------------------------------------------------------
    # Shared Buy/Sell Legend
    # --------------------------------------------------------

    direction_handles = [
        Patch(
            facecolor=COLORS["BUY_WETH"],
            label="Buy WETH",
        ),
        Patch(
            facecolor=COLORS["SELL_WETH"],
            label="Sell WETH",
        ),
    ]

    fig.legend(
        handles=direction_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.080),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.5,
        columnspacing=3,
    )

    # --------------------------------------------------------
    # Figure Titles
    # --------------------------------------------------------

    fig.text(
        0.07,
        0.965,
        "Uniswap V3 | Intraday Trading Activity",
        fontsize=20,
        fontweight="bold",
        color="#1f1f1f",
    )

    fig.text(
        0.07,
        0.93,
        "USDC/WETH · 0.05% vs 0.30% fee tier "
        "· Sep 1–3, 2026 (UTC)",
        fontsize=11,
        color="#555555",
    )

    # --------------------------------------------------------
    # Source Note
    # --------------------------------------------------------

    fig.text(
        0.07,
        0.025,
        "Source: Uniswap V3 Subgraph / The Graph  |  "
        "Hourly index normalized by pool mean  |  "
        "Zero-leg events excluded from direction shares",
        fontsize=8.5,
        color="#666666",
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.865,
        bottom=0.18,
        wspace=0.27,
        hspace=0.43,
    )

    fig.savefig(
        FIGURE,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    df = load_data()

    hourly = hourly_activity(df)
    direction = direction_analysis(df)

    ANALYTICS.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly.to_csv(
        ANALYTICS / "hourly_activity_pilot.csv",
        index=False,
    )

    direction.to_csv(
        ANALYTICS / "direction_summary_pilot.csv",
        index=False,
    )

    create_figure(hourly, direction)

    print("\n=== Hourly Activity Summary ===")

    print(
        hourly.groupby("fee_tier").agg(
            total_events=("swap_events", "sum"),
            total_volume_usd=("volume_usd", "sum"),
            peak_event_index=(
                "event_activity_index", "max"
            ),
            peak_volume_index=(
                "volume_activity_index", "max"
            ),
        ).to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    print("\n=== Buy/Sell Direction Summary ===")

    print(
        direction.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\n=== Directional Volume Imbalance ===")

    for tier in FEES:
        part = direction[
            direction["fee_tier"] == tier
        ].set_index("trade_direction")

        buy = part.loc["BUY_WETH", "volume_usd"]
        sell = part.loc["SELL_WETH", "volume_usd"]

        imbalance = (
            (buy - sell) / (buy + sell)
        )

        print(f"{tier}: {imbalance:+.2%}")

    excluded = (
        df["trade_direction"]
        == "UNDETERMINED_ZERO_LEG"
    ).sum()

    print(f"\nZero-leg events: {excluded}")
    print(f"Figure: {FIGURE}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
