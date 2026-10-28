
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

REGIMES = ["Low", "Medium", "High"]
FEES = ["0.05%", "0.30%"]
COLORS = {"0.05%": "#4A7BB7", "0.30%": "#F7834D"}

START = pd.Timestamp("2026-07-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")


def spot_price(value):
    x = int(value)

    if x <= 0:
        raise ValueError("Invalid sqrtPriceX96")

    ratio = x / (2 ** 96)
    return 1e12 / (ratio * ratio)


def calculate_volatility():
    rows = []

    for day in pd.date_range(
        START, END, freq="D", inclusive="left"
    ):
        path = (
            SOURCE / "pool_0500"
            / f"{day.strftime('%Y-%m-%d')}.parquet"
        )

        df = pd.read_parquet(
            path,
            columns=[
                "timestamp_utc", "block_number",
                "log_index", "sqrt_price_x96_after"
            ]
        )

        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"], utc=True
        )

        df = df.sort_values([
            "block_number", "log_index"
        ])

        df["hour"] = df["timestamp_utc"].dt.floor("h")

        last = df.groupby("hour", sort=True).tail(1)

        rows.extend(
            (row.hour, spot_price(row.sqrt_price_x96_after))
            for row in last.itertuples()
        )

    observed = pd.DataFrame(rows, columns=["hour", "price"])

    if observed["hour"].duplicated().any():
        raise RuntimeError("Duplicate hourly prices")

    hours = pd.date_range(
        START, END, freq="h", inclusive="left"
    )

    raw = observed.set_index("hour")[
        "price"
    ].sort_index().reindex(hours)

    missing_hours = int(raw.isna().sum())
    prices = raw.ffill()

    hourly = pd.DataFrame({
        "price": prices,
        "observed": raw.notna().astype(int),
    }, index=hours)

    hourly["log_return"] = np.log(
        hourly["price"]
    ).diff()

    hourly["date"] = hourly.index.floor("D")

    daily = (
        hourly.groupby("date")
        .agg(
            realized_vol_pct=(
                "log_return",
                lambda s: 100 * np.sqrt(
                    np.square(s.dropna()).sum()
                )
            ),
            observed_hours=("observed", "sum"),
        )
        .reset_index()
    )

    if len(daily) != 92:
        raise RuntimeError("Expected 92 daily volatility values")

    if daily["observed_hours"].min() < 12:
        print(
            "WARNING: Some days have fewer than "
            "12 observed hourly prices"
        )

    q1, q2 = daily["realized_vol_pct"].quantile([
        1 / 3, 2 / 3
    ])

    daily["regime"] = np.select(
        [
            daily["realized_vol_pct"] <= q1,
            daily["realized_vol_pct"] >= q2,
        ],
        ["Low", "High"],
        default="Medium",
    )

    daily["regime"] = pd.Categorical(
        daily["regime"],
        categories=REGIMES,
        ordered=True,
    )

    return daily, float(q1), float(q2), missing_hours


def calculate_market(daily):
    market = pd.read_csv(
        ANALYTICS / "q3_liquidity_daily.csv"
    )

    market["date"] = pd.to_datetime(
        market["date"], utc=True
    )

    merged = market.merge(
        daily[["date", "realized_vol_pct", "regime"]],
        on="date",
        validate="many_to_one",
    )

    if len(merged) != 184:
        raise RuntimeError("Market/regime alignment failed")

    summary = (
        merged.groupby(
            ["regime", "fee_tier"], observed=True
        )
        .agg(
            days=("date", "nunique"),
            median_daily_volume_usd=(
                "volume_usd_swaps", "median"
            ),
            median_daily_turnover=("turnover", "median"),
            median_volume_share_pct=(
                "volume_share_pct", "median"
            ),
        )
        .reset_index()
    )

    return merged, summary


def calculate_execution(daily):
    path = ANALYTICS / "q3_execution_events.parquet"

    df = pd.read_parquet(
        path,
        columns=[
            "timestamp_utc", "fee_tier",
            "trade_direction", "trade_size_usd",
            "observed_cost_bps",
            "fee_adjusted_penalty_bps",
            "terminal_spot_impact_bps",
        ],
    )

    df = df[
        df["trade_direction"].eq("BUY_WETH")
        & df["trade_size_usd"].ge(10000)
        & df["trade_size_usd"].lt(100000)
    ].copy()

    df["date"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    ).dt.floor("D")

    df = df.merge(
        daily[["date", "regime"]],
        on="date",
        validate="many_to_one",
    )

    if df.empty:
        raise RuntimeError("No matched execution observations")

    summary = (
        df.groupby(
            ["regime", "fee_tier"], observed=True
        )
        .agg(
            events=("observed_cost_bps", "size"),
            median_cost_bps=(
                "observed_cost_bps", "median"
            ),
            median_fee_adjusted_bps=(
                "fee_adjusted_penalty_bps", "median"
            ),
            median_terminal_impact_bps=(
                "terminal_spot_impact_bps", "median"
            ),
        )
        .reset_index()
    )

    return summary


def style(ax):
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, pad=6)


def create_figure(daily, market, summary, execution, q1, q2):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    x = np.arange(3)
    width = 0.34

    ax = axes[0, 0]
    ax.plot(
        daily["date"], daily["realized_vol_pct"],
        color="#4A7BB7", linewidth=1.8
    )

    for threshold in [q1, q2]:
        ax.axhline(
            threshold, color="#F7834D",
            linestyle="--", alpha=0.7
        )

    ax.set_title(
        "A | Daily Realized Volatility",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Daily realized volatility (%)")
    style(ax)

    ax = axes[0, 1]
    high = market[market["fee_tier"] == "0.30%"]

    values = [
        high.loc[
            high["regime"] == regime,
            "volume_share_pct"
        ].to_numpy()
        for regime in REGIMES
    ]

    bp = ax.boxplot(
        values, tick_labels=REGIMES,
        patch_artist=True, showfliers=False,
        widths=0.5
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("#FED081")

    ax.set_title(
        "B | 0.30% Pool Volume Share by Regime",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Daily volume share (%)")
    style(ax)

    ax = axes[1, 0]

    for i, fee in enumerate(FEES):
        part = (
            execution[execution["fee_tier"] == fee]
            .set_index("regime")
            .reindex(REGIMES)
        )

        ax.bar(
            x + (i - 0.5) * width,
            part["median_fee_adjusted_bps"],
            width, color=COLORS[fee], label=fee
        )

    ax.set_xticks(x, REGIMES)
    ax.set_title(
        "C | Buy WETH Fee-adjusted Execution Penalty",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Median penalty (bps), $10K–100K")
    ax.legend(frameon=False, ncol=2)
    style(ax)

    ax = axes[1, 1]

    for i, fee in enumerate(FEES):
        part = (
            summary[summary["fee_tier"] == fee]
            .set_index("regime")
            .reindex(REGIMES)
        )

        ax.bar(
            x + (i - 0.5) * width,
            part["median_daily_turnover"],
            width, color=COLORS[fee], label=fee
        )

    ax.set_xticks(x, REGIMES)
    ax.set_title(
        "D | Capital Turnover by Regime",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Median daily volume / TVL (x)")
    style(ax)

    fig.suptitle(
        "Uniswap V3 | Q3 Volatility Regimes and Market Dynamics",
        fontsize=19, fontweight="bold", y=0.99
    )

    fig.text(
        0.5, 0.945,
        "USDC/WETH | July 1 – September 30, 2026 (UTC)",
        ha="center", color="#555555"
    )

    fig.text(
        0.5, 0.018,
        "Volatility from 0.05% pool hourly closing prices"
        " | Regimes defined by Q3 terciles"
        " | Descriptive comparisons, not causal effects",
        ha="center", fontsize=8.5, color="#666666"
    )

    fig.tight_layout(
        rect=[0.025, 0.05, 0.99, 0.91],
        h_pad=2.8, w_pad=2.3
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "18_q3_volatility_regimes.png"

    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return output


def main():
    ANALYTICS.mkdir(parents=True, exist_ok=True)

    daily, q1, q2, missing = calculate_volatility()
    market, market_summary = calculate_market(daily)
    execution = calculate_execution(daily)

    daily.to_csv(
        ANALYTICS / "q3_volatility_regimes_daily.csv",
        index=False
    )
    market_summary.to_csv(
        ANALYTICS / "q3_volatility_market_summary.csv",
        index=False
    )
    execution.to_csv(
        ANALYTICS / "q3_volatility_execution_summary.csv",
        index=False
    )

    daily["month"] = daily["date"].dt.strftime("%Y-%m")

    month_counts = pd.crosstab(
        daily["month"], daily["regime"]
    )

    month_counts.to_csv(
        ANALYTICS / "q3_volatility_month_counts.csv"
    )

    figure = create_figure(
        daily, market, market_summary, execution, q1, q2
    )

    print("\n=== Q3 VOLATILITY REGIMES ===")
    print(f"Days: {len(daily)}")
    print(f"Hours without fresh Swap price: {missing}")
    print(f"Low/Medium threshold: {q1:.3f}%")
    print(f"Medium/High threshold: {q2:.3f}%")
    print("\nDays per month and regime:")
    print(month_counts.to_string())

    print("\n=== MARKET BY VOLATILITY REGIME ===")
    print(market_summary.to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"
    ))

    print("\n=== EXECUTION BY VOLATILITY REGIME ===")
    print(execution.to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"
    ))

    print(f"\nFigure: {figure}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
