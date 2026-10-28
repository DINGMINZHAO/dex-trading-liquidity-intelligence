
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

DATA = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
SOURCE = DATA / "processed/swaps_2026q3"
ANALYTICS = DATA / "analytics"
FIGURES = ROOT / "reports/figures"

FEES = ["0.05%", "0.30%"]
DIRECTIONS = ["BUY_WETH", "SELL_WETH"]
COLORS = {"0.05%": "#4A7BB7", "0.30%": "#F7834D"}
FEE_RATES = {"0.05%": 0.0005, "0.30%": 0.003}
BINS = [100, 1000, 10000, 100000, np.inf]
LABELS = ["$100–1K", "$1K–10K", "$10K–100K", "≥$100K"]

COLUMNS = [
    "event_id", "fee_tier", "pool_address", "timestamp_utc",
    "block_number", "log_index", "trade_direction",
    "trade_size_usd", "execution_price_usdc_per_weth",
    "sqrt_price_x96_after"
]


def spot_price(value):
    if pd.isna(value):
        return np.nan
    x = int(value)
    if x <= 0:
        raise ValueError("Invalid sqrtPriceX96")
    ratio = x / (2 ** 96)
    return 1e12 / (ratio * ratio)


def load_data():
    paths = sorted(SOURCE.glob("pool_*/*.parquet"))
    if len(paths) != 184:
        raise ValueError(f"Expected 184 partitions, got {len(paths)}")

    df = pd.concat(
        (pd.read_parquet(path, columns=COLUMNS) for path in paths),
        ignore_index=True,
    )

    if len(df) != 581454:
        raise ValueError(f"Unexpected number of events: {len(df)}")

    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs")

    if df[["block_number", "log_index"]].isna().any().any():
        raise ValueError("Missing event ordering fields")

    if df.duplicated(
        ["pool_address", "block_number", "log_index"]
    ).any():
        raise ValueError("Duplicate pool/block/log index")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values(
        ["pool_address", "block_number", "log_index"]
    ).reset_index(drop=True)

    for _, group in df.groupby("pool_address"):
        if not group["timestamp_utc"].is_monotonic_increasing:
            raise ValueError("Timestamp ordering error")

    return df


def calculate(df):
    df["spot_after"] = (
        df["sqrt_price_x96_after"].map(spot_price)
    )

    df["spot_before"] = (
        df.groupby("pool_address")["spot_after"].shift(1)
    )

    df["month"] = df["timestamp_utc"].dt.strftime("%Y-%m")

    valid = (
        df["trade_direction"].isin(DIRECTIONS)
        & df["spot_before"].gt(0)
        & df["execution_price_usdc_per_weth"].gt(0)
    )

    eligible = df.loc[
        valid & df["trade_size_usd"].ge(100)
    ].copy()

    if eligible.empty:
        raise ValueError("No eligible Swap events")

    before = eligible["spot_before"].to_numpy(dtype=float)
    after = eligible["spot_after"].to_numpy(dtype=float)
    executed = eligible[
        "execution_price_usdc_per_weth"
    ].to_numpy(dtype=float)

    fee = eligible["fee_tier"].map(FEE_RATES).to_numpy()
    buy = eligible["trade_direction"].eq("BUY_WETH").to_numpy()

    if not (
        np.isfinite(before).all()
        and np.isfinite(after).all()
        and np.isfinite(executed).all()
        and (before > 0).all()
        and (after > 0).all()
    ):
        raise ValueError("Invalid reference prices")

    eligible["observed_cost_bps"] = np.where(
        buy,
        10000 * (executed / before - 1),
        10000 * (1 - executed / before)
    )

    no_fee = np.where(
        buy,
        executed * (1 - fee),
        executed / (1 - fee)
    )

    eligible["fee_adjusted_penalty_bps"] = np.where(
        buy,
        10000 * (no_fee / before - 1),
        10000 * (1 - no_fee / before)
    )

    eligible["terminal_spot_impact_bps"] = np.where(
        buy,
        10000 * (after / before - 1),
        10000 * (1 - after / before)
    )

    metrics = [
        "observed_cost_bps",
        "fee_adjusted_penalty_bps",
        "terminal_spot_impact_bps"
    ]

    if not np.isfinite(eligible[metrics].to_numpy()).all():
        raise ValueError("Non-finite execution metrics")

    eligible["size_bucket"] = pd.cut(
        eligible["trade_size_usd"],
        bins=BINS, labels=LABELS, right=False
    )

    return eligible


def summarize(df, keys):
    return (
        df.groupby(keys, observed=True)
        .agg(
            events=("event_id", "size"),
            median_cost_bps=("observed_cost_bps", "median"),
            p25_cost_bps=("observed_cost_bps",
                          lambda x: x.quantile(0.25)),
            p75_cost_bps=("observed_cost_bps",
                          lambda x: x.quantile(0.75)),
            median_fee_adjusted_bps=(
                "fee_adjusted_penalty_bps", "median"
            ),
            p25_fee_adjusted_bps=(
                "fee_adjusted_penalty_bps",
                lambda x: x.quantile(0.25)
            ),
            p75_fee_adjusted_bps=(
                "fee_adjusted_penalty_bps",
                lambda x: x.quantile(0.75)
            ),
            median_terminal_impact_bps=(
                "terminal_spot_impact_bps", "median"
            ),
            p25_terminal_impact_bps=(
                "terminal_spot_impact_bps",
                lambda x: x.quantile(0.25)
            ),
            p75_terminal_impact_bps=(
                "terminal_spot_impact_bps",
                lambda x: x.quantile(0.75)
            )
        )
        .reset_index()
    )


def draw_panel(ax, summary, direction, metric, title):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)

    for label, shift in [("0.05%", -0.12), ("0.30%", 0.12)]:
        part = summary[
            (summary["fee_tier"] == label)
            & (summary["trade_direction"] == direction)
        ].set_index("size_bucket")

        x, y, lower, upper = [], [], [], []

        for i, bucket in enumerate(LABELS):
            if bucket not in part.index:
                continue

            row = part.loc[bucket]
            if int(row["events"]) < 10:
                continue

            med = float(row[f"median_{metric}_bps"])
            p25 = float(row[f"p25_{metric}_bps"])
            p75 = float(row[f"p75_{metric}_bps"])

            x.append(i + shift)
            y.append(med)
            lower.append(max(0, med - p25))
            upper.append(max(0, p75 - med))

        if x:
            ax.errorbar(
                x, y, yerr=[lower, upper],
                fmt="o-", color=COLORS[label],
                linewidth=1.6, markersize=5,
                capsize=3, label=label
            )

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_ylabel("Basis points (bps)")
    ax.grid(axis="y", color="#E0E0E0", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figures(summary):
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, direction, heading in zip(
        axes, DIRECTIONS, ["A | Buy WETH", "B | Sell WETH"]
    ):
        draw_panel(
            ax, summary, direction, "cost", heading
        )

        for label in FEES:
            ax.axhline(
                FEE_RATES[label] * 10000,
                color=COLORS[label], linestyle="--",
                linewidth=1, alpha=0.5
            )

    axes[0].legend(frameon=False, ncol=2)

    fig.suptitle(
        "Uniswap V3 | Q3 Observed Execution Cost",
        fontsize=19, fontweight="bold", y=0.98
    )
    fig.text(
        0.5, 0.92,
        "USDC/WETH | July–September 2026 (UTC) | Median and IQR",
        ha="center", color="#555555"
    )
    fig.text(
        0.5, 0.025,
        "Pool-level swaps ≥$100 | Dashed lines: nominal fee rates | Gas excluded",
        ha="center", fontsize=9, color="#666666"
    )
    fig.subplots_adjust(top=0.83, bottom=0.19, wspace=0.24)

    path1 = FIGURES / "10_q3_observed_execution_cost.png"
    fig.savefig(path1, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for col, direction in enumerate(DIRECTIONS):
        draw_panel(
            axes[0, col], summary, direction,
            "fee_adjusted",
            f"{'A' if col == 0 else 'B'} | "
            f"{'Buy' if col == 0 else 'Sell'} WETH · Fee-adjusted"
        )
        draw_panel(
            axes[1, col], summary, direction,
            "terminal_impact",
            f"{'C' if col == 0 else 'D'} | "
            f"{'Buy' if col == 0 else 'Sell'} WETH · Terminal impact"
        )

    axes[0, 0].legend(frameon=False, ncol=2)

    fig.suptitle(
        "Uniswap V3 | Q3 Price Impact Analysis",
        fontsize=19, fontweight="bold", y=0.99
    )
    fig.text(
        0.5, 0.943,
        "USDC/WETH | July–September 2026 (UTC) | Median and IQR",
        ha="center", color="#555555"
    )
    fig.text(
        0.5, 0.025,
        "Pool-level swaps ≥$100 | Previous swap post-price reference | Gas excluded",
        ha="center", fontsize=9, color="#666666"
    )
    fig.subplots_adjust(
        top=0.87, bottom=0.13, hspace=0.42, wspace=0.25
    )

    path2 = FIGURES / "11_q3_price_impact_analysis.png"
    fig.savefig(path2, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return path1, path2


def main():
    ANALYTICS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    eligible = calculate(df)

    overall = summarize(
        eligible, ["trade_direction", "fee_tier", "size_bucket"]
    )
    monthly = summarize(
        eligible,
        ["month", "trade_direction", "fee_tier", "size_bucket"]
    )

    overall.to_csv(
        ANALYTICS / "q3_execution_quality_summary.csv",
        index=False
    )
    monthly.to_csv(
        ANALYTICS / "q3_execution_quality_monthly.csv",
        index=False
    )

    eligible.to_parquet(
        ANALYTICS / "q3_execution_events.parquet",
        index=False, compression="zstd"
    )

    fig1, fig2 = create_figures(overall)

    print("\n=== Q3 EXECUTION QUALITY ===")
    print(f"All Swap events: {len(df):,}")
    print(f"Eligible events >=$100: {len(eligible):,}")
    print(f"Excluded events: {len(df) - len(eligible):,}")
    print(
        "Negative observed costs:",
        int((eligible["observed_cost_bps"] < 0).sum())
    )
    print(
        "Negative terminal impacts:",
        int((eligible["terminal_spot_impact_bps"] < -1e-8).sum())
    )

    print("\n=== BUY WETH: Q3 MEDIAN COST BY SIZE ===")
    print(overall[
        overall["trade_direction"] == "BUY_WETH"
    ][[
        "fee_tier", "size_bucket", "events",
        "median_cost_bps", "median_fee_adjusted_bps"
    ]].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"
    ))

    print("\n=== MONTHLY MEDIAN COST: $10K–100K BUY WETH ===")
    print(monthly[
        (monthly["trade_direction"] == "BUY_WETH")
        & (monthly["size_bucket"] == "$10K–100K")
    ][[
        "month", "fee_tier", "events",
        "median_cost_bps", "median_fee_adjusted_bps"
    ]].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"
    ))

    print(f"\nFigure: {fig1}")
    print(f"Figure: {fig2}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
