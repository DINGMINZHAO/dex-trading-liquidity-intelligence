
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
INPUT = DATA / "analytics/q3_liquidity_daily.csv"
ANALYTICS = DATA / "analytics"
FIGURES = ROOT / "reports/figures"

FEE_RATES = {"0.05%": 0.0005, "0.30%": 0.003}
COLORS = {"0.05%": "#4A7BB7", "0.30%": "#F7834D"}
FEES = list(FEE_RATES)
MONTHS = ["2026-07", "2026-08", "2026-09"]


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "date", "fee_tier", "volume_usd_swaps",
        "tvl_usd", "turnover", "volume_gap_pct"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {sorted(required - set(df.columns))}"
        )

    df["date"] = pd.to_datetime(df["date"], utc=True)

    expected = {
        (day, fee)
        for day in pd.date_range(
            "2026-07-01", "2026-09-30", tz="UTC"
        )
        for fee in FEES
    }

    actual = set(zip(df["date"], df["fee_tier"]))

    if (
        len(df) != 184
        or df.duplicated(["date", "fee_tier"]).any()
        or actual != expected
    ):
        raise ValueError("Q3 pool-day coverage mismatch")

    numeric = [
        "volume_usd_swaps", "tvl_usd",
        "turnover", "volume_gap_pct"
    ]

    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite source data")

    if (
        (df["volume_usd_swaps"] < 0).any()
        or (df["tvl_usd"] <= 0).any()
        or df["volume_gap_pct"].abs().max() > 2
    ):
        raise ValueError("Invalid volume, TVL, or reconciliation")

    expected_turnover = df["volume_usd_swaps"] / df["tvl_usd"]

    if not np.allclose(
        df["turnover"], expected_turnover,
        rtol=1e-8, atol=1e-10
    ):
        raise ValueError("Turnover calculation mismatch")

    return df.sort_values(
        ["fee_tier", "date"]
    ).reset_index(drop=True)


def calculate(df):
    df = df.copy()

    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["fee_rate"] = df["fee_tier"].map(FEE_RATES)

    df["gross_fee_est_usd"] = (
        df["volume_usd_swaps"] * df["fee_rate"]
    )

    df["gross_fee_tvl_bps"] = (
        10000 * df["gross_fee_est_usd"] / df["tvl_usd"]
    )

    for col in ["gross_fee_est_usd", "gross_fee_tvl_bps"]:
        df[f"{col}_ma7"] = (
            df.groupby("fee_tier")[col]
            .transform(
                lambda x: x.rolling(
                    7, min_periods=7
                ).mean()
            )
        )

    daily_total = df.groupby("date")[
        "gross_fee_est_usd"
    ].transform("sum")

    df["gross_fee_share_pct"] = (
        100 * df["gross_fee_est_usd"] / daily_total
    )

    monthly = (
        df.groupby(["month", "fee_tier"], as_index=False)
        .agg(
            swap_volume_usd=("volume_usd_swaps", "sum"),
            gross_fee_est_usd=("gross_fee_est_usd", "sum"),
            mean_tvl_usd=("tvl_usd", "mean"),
            median_daily_turnover=("turnover", "median"),
            median_fee_tvl_bps=("gross_fee_tvl_bps", "median"),
            mean_fee_tvl_bps=("gross_fee_tvl_bps", "mean"),
        )
    )

    monthly["gross_fee_share_pct"] = (
        100 * monthly["gross_fee_est_usd"]
        / monthly.groupby("month")[
            "gross_fee_est_usd"
        ].transform("sum")
    )

    monthly["volume_share_pct"] = (
        100 * monthly["swap_volume_usd"]
        / monthly.groupby("month")[
            "swap_volume_usd"
        ].transform("sum")
    )

    monthly["tvl_share_pct"] = (
        100 * monthly["mean_tvl_usd"]
        / monthly.groupby("month")[
            "mean_tvl_usd"
        ].transform("sum")
    )

    return df, monthly


def style(ax):
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#333333", length=0, pad=6)


def plot_results(daily, monthly):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    panels = [
        (
            axes[0, 0],
            "gross_fee_est_usd_ma7",
            1000,
            "A | Daily Gross Fee Estimate",
            "USD thousands / day"
        ),
        (
            axes[0, 1],
            "gross_fee_tvl_bps_ma7",
            1,
            "B | Daily Gross Fee / TVL",
            "bps / day"
        ),
    ]

    for ax, field, scale, title, ylabel in panels:
        for fee in FEES:
            part = daily[
                daily["fee_tier"] == fee
            ].sort_values("date")

            ax.plot(
                part["date"],
                part[field] / scale,
                color=COLORS[fee],
                linewidth=2,
                label=fee
            )

        ax.set_title(
            title, loc="left", fontweight="bold", pad=12
        )
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=2)
        ax.xaxis.set_major_locator(
            mdates.MonthLocator()
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %Y")
        )
        style(ax)

    x = np.arange(len(MONTHS))
    width = 0.34

    bar_panels = [
        (
            axes[1, 0],
            "gross_fee_est_usd",
            1e6,
            "C | Monthly Gross Fee Estimate",
            "USD millions"
        ),
        (
            axes[1, 1],
            "median_fee_tvl_bps",
            1,
            "D | Median Daily Gross Fee / TVL",
            "bps / day"
        ),
    ]

    for ax, field, scale, title, ylabel in bar_panels:
        for i, fee in enumerate(FEES):
            part = (
                monthly[monthly["fee_tier"] == fee]
                .set_index("month")
                .reindex(MONTHS)
            )

            values = part[field].to_numpy() / scale

            bars = ax.bar(
                x + (i - 0.5) * width,
                values,
                width,
                color=COLORS[fee],
                label=fee
            )

            for bar, val in zip(bars, values):
                ax.annotate(
                    f"{val:.2f}",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height()
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9
                )

        ax.set_title(
            title, loc="left", fontweight="bold", pad=12
        )
        ax.set_xticks(x, ["Jul", "Aug", "Sep"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(
            0,
            max(
                monthly[field].max() / scale * 1.2,
                0.1
            )
        )
        ax.legend(frameon=False, ncol=2)
        style(ax)

    fig.suptitle(
        "Uniswap V3 | Q3 Fee Economics and Capital Efficiency",
        fontsize=19,
        fontweight="bold",
        y=0.985
    )

    fig.text(
        0.5, 0.945,
        "USDC/WETH | July 1 – September 30, 2026 (UTC)",
        ha="center", color="#555555", fontsize=11
    )

    fig.text(
        0.5, 0.025,
        "Gross fee estimate = Swap USD volume × pool fee rate"
        " | 7-day moving averages in A/B"
        " | Not realized LP or protocol revenue",
        ha="center", fontsize=8.8, color="#666666"
    )

    fig.tight_layout(
        rect=[0.03, 0.06, 0.99, 0.91],
        h_pad=3, w_pad=2
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "13_q3_fee_economics.png"

    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return path


def main():
    ANALYTICS.mkdir(parents=True, exist_ok=True)

    source = load_data()
    daily, monthly = calculate(source)

    daily.to_csv(
        ANALYTICS / "q3_fee_economics_daily.csv",
        index=False
    )
    monthly.to_csv(
        ANALYTICS / "q3_fee_economics_monthly.csv",
        index=False
    )

    figure = plot_results(daily, monthly)

    print("\n=== Q3 FEE ECONOMICS ===")
    print(f"Validated pool-days: {len(daily)}/184")

    print("\n=== MONTHLY SUMMARY ===")
    columns = [
        "month",
        "fee_tier",
        "gross_fee_est_usd",
        "gross_fee_share_pct",
        "median_fee_tvl_bps",
        "mean_fee_tvl_bps",
    ]

    print(
        monthly[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:,.3f}"
        )
    )

    print("\n=== QUARTER TOTAL ===")

    totals = (
        monthly.groupby("fee_tier")[
            "gross_fee_est_usd"
        ].sum()
    )

    for fee, amount in totals.items():
        print(f"{fee}: ${amount:,.2f}")

    print(f"\nFigure: {figure}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
