
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))

INPUT = (
    DATA_ROOT / "analytics"
    / "matched_quotes_multiblock_pilot.csv"
)

FIGURE = (
    ROOT / "reports/figures"
    / "07_multiblock_matched_execution.png"
)

SUMMARY = (
    DATA_ROOT / "analytics"
    / "matched_quotes_multiblock_summary.csv"
)

ORDER_SIZES = [1000, 5000, 10000, 20000]

PALETTE = {
    "blue": "#4A7BB7",
    "orange": "#F7834D",
    "red": "#D9412B",
    "yellow": "#FED081",
    "pale_yellow": "#FFFDBB",
    "pale_cyan": "#D2EDF4",
    "light_blue": "#91C4DE",
}

START = pd.Timestamp("2026-09-01", tz="UTC")
END = pd.Timestamp("2026-09-04", tz="UTC")


def load_data():
    df = pd.read_csv(INPUT)

    required = {
        "target_utc",
        "block_number",
        "input_usdc",
        "low_status",
        "high_status",
        "low_advantage_usd",
        "winner",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["target_utc"] = pd.to_datetime(
        df["target_utc"], utc=True
    )

    if not df["target_utc"].between(
        START, END, inclusive="left"
    ).all():
        raise ValueError("Snapshots outside study window")

    if set(df["input_usdc"]) != set(ORDER_SIZES):
        raise ValueError("Unexpected order sizes")

    if df.duplicated(
        ["input_usdc", "block_number"]
    ).any():
        raise ValueError("Duplicate matched quotes")

    if len(df) != 48:
        raise ValueError(
            f"Expected 48 quotes, found {len(df)}"
        )

    df["comparable"] = (
        df["low_status"].eq("OK_SINGLE_RANGE")
        & df["high_status"].eq("OK_SINGLE_RANGE")
    )

    valid = df["comparable"]

    if df.loc[valid, "low_advantage_usd"].isna().any():
        raise ValueError("Missing comparable quote advantages")

    df["advantage_bps"] = (
        df["low_advantage_usd"]
        / df["input_usdc"]
        * 10000
    )

    for _, row in df.iterrows():
        if row["comparable"]:
            if not np.isfinite(row["advantage_bps"]):
                raise ValueError("Invalid advantage")

            expected = (
                "0.05%" if row["advantage_bps"] > 0
                else "0.30%" if row["advantage_bps"] < 0
                else "TIE"
            )

            if row["winner"] != expected:
                raise ValueError("Winner inconsistency")

        elif row["winner"] != "NOT_COMPARABLE":
            raise ValueError("Invalid excluded quote")

    reference = None

    for amount in ORDER_SIZES:
        part = df[df["input_usdc"] == amount]

        if len(part) != 12:
            raise ValueError("Expected 12 snapshots per size")

        current = (
            part.sort_values("target_utc")[
                ["target_utc", "block_number"]
            ].reset_index(drop=True)
        )

        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(
                "Snapshot mismatch across order sizes"
            )

    return df


def calculate_summary(df):
    records = []

    for amount in ORDER_SIZES:
        part = df[df["input_usdc"] == amount]
        valid = part[part["comparable"]]

        records.append({
            "input_usdc": amount,
            "snapshots": len(part),
            "comparable": len(valid),
            "coverage_pct": 100 * len(valid) / len(part),
            "low_wins": (valid["winner"] == "0.05%").sum(),
            "high_wins": (valid["winner"] == "0.30%").sum(),
            "ties": (valid["winner"] == "TIE").sum(),
            "median_advantage_usd": (
                valid["low_advantage_usd"].median()
            ),
            "median_advantage_bps": (
                valid["advantage_bps"].median()
            ),
        })

    return pd.DataFrame(records)


def style_axis(ax, title):
    ax.set_title(
        title,
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    ax.set_ylabel(
        "0.05% advantage (bps of input)",
        labelpad=10,
    )

    ax.axhline(
        0,
        color="#777777",
        linewidth=1,
        zorder=2,
    )

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


def create_figure(df, summary):
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
        2, 2,
        figsize=(15, 9.5),
    )

    letters = ["A", "B", "C", "D"]

    for ax, letter, amount in zip(
        axes.flat, letters, ORDER_SIZES
    ):
        part = (
            df[df["input_usdc"] == amount]
            .sort_values("target_utc")
            .reset_index(drop=True)
        )

        stats = summary[
            summary["input_usdc"] == amount
        ].iloc[0]

        positions = np.arange(len(part))

        for i, row in part.iterrows():
            if not row["comparable"]:
                ax.scatter(
                    i,
                    0,
                    marker="x",
                    s=85,
                    linewidths=2,
                    color="#9CA3AF",
                    zorder=5,
                )
                continue

            value = row["advantage_bps"]

            color = (
                PALETTE["blue"]
                if value >= 0
                else PALETTE["orange"]
            )

            ax.bar(
                i,
                value,
                width=0.72,
                color=color,
                zorder=3,
            )

        style_axis(
            ax,
            f"{letter}  |  ${amount:,} USDC",
        )

        ax.set_xlim(-0.7, len(part) - 0.3)

        ticks = np.arange(0, len(part), 2)

        labels = [
            part.iloc[i]["target_utc"].strftime(
                "%b %d\n%H:%M"
            )
            for i in ticks
        ]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=9)

        comparable = part.loc[
            part["comparable"], "advantage_bps"
        ]

        if not comparable.empty:
            maximum = max(0, comparable.max())
            minimum = min(0, comparable.min())
            spread = max(maximum - minimum, 1)

            ax.set_ylim(
                minimum - spread * 0.14,
                maximum + spread * 0.32,
            )

        ax.text(
            0.98,
            0.95,
            (
                f"Comparable: {int(stats['comparable'])}/12\n"
                f"Median: {stats['median_advantage_bps']:+.2f} bps"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
                "pad": 3,
            },
        )

    handles = [
        Patch(
            facecolor=PALETTE["blue"],
            label="0.05% better",
        ),
        Patch(
            facecolor=PALETTE["orange"],
            label="0.30% better",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="#9CA3AF",
            markersize=8,
            markeredgewidth=2,
            label="Not comparable",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.906),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.075,
        0.968,
        "Uniswap V3 | Matched Execution Across Historical States",
        fontsize=19,
        fontweight="bold",
        color="#1f1f1f",
    )

    fig.text(
        0.075,
        0.935,
        "USDC → WETH · Sep 1–3, 2026 (UTC) "
        "· 12 snapshots at 6-hour intervals",
        fontsize=11,
        color="#555555",
    )

    fig.text(
        0.075,
        0.055,
        "Positive values favor the 0.05% pool; "
        "negative values favor the 0.30% pool. "
        "Each panel uses its own vertical scale.",
        fontsize=9,
        color="#555555",
    )

    fig.text(
        0.075,
        0.029,
        "Source: Uniswap V3 Subgraph / The Graph  |  "
        "Simulated single-range quotes  |  "
        "Gas, routing and MEV excluded",
        fontsize=8.5,
        color="#666666",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.97,
        top=0.84,
        bottom=0.14,
        wspace=0.26,
        hspace=0.48,
    )

    fig.savefig(
        FIGURE,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    df = load_data()
    summary = calculate_summary(df)

    SUMMARY.parent.mkdir(
        parents=True, exist_ok=True
    )

    summary.to_csv(SUMMARY, index=False)

    create_figure(df, summary)

    print("\n=== Matched Execution: Multi-block Summary ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print(f"\nFigure: {FIGURE}")
    print(f"Summary: {SUMMARY}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
