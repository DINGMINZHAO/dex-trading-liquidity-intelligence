
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

DATA = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
SOURCE = DATA / "processed/swaps_2026q3"
ANALYTICS = DATA / "analytics"
FIGURES = ROOT / "reports/figures"

FEES = {"0.05%": 500, "0.30%": 3000}
MONTHS = ["2026-07", "2026-08", "2026-09"]
COLORS = {"0.05%": "#4A7BB7", "0.30%": "#F7834D"}


def collect_activity():
    origin_rows = []
    size_arrays = {}
    checks = {}

    days = pd.date_range(
        "2026-07-01", "2026-09-30", tz="UTC"
    )

    for day in days:
        date = day.strftime("%Y-%m-%d")
        month = date[:7]

        for fee, code in FEES.items():
            path = (
                SOURCE / f"pool_{code:04d}"
                / f"{date}.parquet"
            )

            if not path.exists():
                raise FileNotFoundError(path)

            df = pd.read_parquet(
                path,
                columns=["origin", "trade_size_usd"]
            )

            if df.empty:
                raise ValueError(f"Empty partition: {path}")

            values = pd.to_numeric(
                df["trade_size_usd"], errors="coerce"
            ).to_numpy(dtype=float)

            if (
                not np.isfinite(values).all()
                or (values < 0).any()
            ):
                raise ValueError(f"Invalid volumes: {path}")

            key = (month, fee)
            size_arrays.setdefault(key, []).append(values)

            df["origin"] = (
                df["origin"].astype("string").str.lower()
            )

            valid = df["origin"].str.fullmatch(
                r"0x[0-9a-f]{40}"
            ).fillna(False)

            group = (
                df.loc[valid]
                .groupby("origin")
                .agg(
                    origin_volume_usd=("trade_size_usd", "sum"),
                    origin_swap_events=("trade_size_usd", "size")
                )
                .reset_index()
            )

            group["month"] = month
            group["fee_tier"] = fee
            origin_rows.append(group)

            stats = checks.setdefault(key, {
                "events": 0,
                "volume_usd": 0.0,
                "valid_origin_events": 0,
            })

            stats["events"] += len(df)
            stats["volume_usd"] += float(values.sum())
            stats["valid_origin_events"] += int(valid.sum())

    activity = (
        pd.concat(origin_rows, ignore_index=True)
        .groupby(
            ["month", "fee_tier", "origin"],
            as_index=False
        )
        .agg(
            origin_volume_usd=("origin_volume_usd", "sum"),
            origin_swap_events=("origin_swap_events", "sum")
        )
    )

    return activity, size_arrays, checks


def calculate_kpis(activity, size_arrays, checks):
    base = pd.read_csv(
        ANALYTICS / "q3_monthly_market_summary.csv"
    )

    required = {
        "month", "fee_tier", "swap_events",
        "volume_usd", "volume_share_pct"
    }

    if not required.issubset(base.columns):
        raise ValueError("Missing market summary columns")

    if len(base) != 6:
        raise ValueError("Expected six pool-month rows")

    base = base.copy()
    base["event_share_pct"] = (
        100 * base["swap_events"]
        / base.groupby("month")["swap_events"].transform("sum")
    )

    origin_sets = {}
    output = []

    for month in MONTHS:
        for fee in FEES:
            key = (month, fee)

            subset = activity[
                (activity["month"] == month)
                & (activity["fee_tier"] == fee)
            ]

            origins = set(subset["origin"])
            origin_sets[key] = origins

            values = np.concatenate(size_arrays[key])
            stats = checks[key]

            reference = base[
                (base["month"] == month)
                & (base["fee_tier"] == fee)
            ].iloc[0]

            if (
                len(values) != int(reference["swap_events"])
                or stats["events"] != len(values)
                or not np.isclose(
                    stats["volume_usd"],
                    float(reference["volume_usd"]),
                    rtol=1e-9, atol=1.0
                )
            ):
                raise ValueError(
                    f"Market reconciliation failed: {key}"
                )

            volumes = subset["origin_volume_usd"]
            n_origins = len(origins)

            top_n = max(1, math.ceil(n_origins * 0.01))

            concentration = (
                100 * volumes.nlargest(top_n).sum()
                / volumes.sum()
                if n_origins and volumes.sum() > 0
                else np.nan
            )

            output.append({
                "month": month,
                "fee_tier": fee,
                "swap_events": stats["events"],
                "volume_usd": stats["volume_usd"],
                "volume_share_pct": float(
                    reference["volume_share_pct"]
                ),
                "event_share_pct": float(
                    reference["event_share_pct"]
                ),
                "median_swap_usd": float(np.median(values)),
                "unique_origins": n_origins,
                "origin_coverage_pct": (
                    100 * stats["valid_origin_events"]
                    / stats["events"]
                ),
                "swaps_per_origin": (
                    stats["valid_origin_events"] / n_origins
                    if n_origins else np.nan
                ),
                "top_1pct_origin_volume_share_pct": concentration,
            })

    kpis = pd.DataFrame(output)

    kpis["prev_month_reappearance_pct"] = np.nan

    for fee in FEES:
        for i in range(1, len(MONTHS)):
            current = (MONTHS[i], fee)
            previous = (MONTHS[i - 1], fee)

            old = origin_sets[previous]
            new = origin_sets[current]

            value = (
                100 * len(old & new) / len(old)
                if old else np.nan
            )

            mask = (
                (kpis["month"] == current[0])
                & (kpis["fee_tier"] == fee)
            )

            kpis.loc[
                mask, "prev_month_reappearance_pct"
            ] = value

    overlap = []

    for month in MONTHS:
        low = origin_sets[(month, "0.05%")]
        high = origin_sets[(month, "0.30%")]

        shared = len(low & high)
        union = len(low | high)

        overlap.append({
            "month": month,
            "unique_origins_either_pool": union,
            "origins_using_both_pools": shared,
            "both_pool_share_pct": (
                100 * shared / union if union else np.nan
            ),
        })

    return kpis, pd.DataFrame(overlap)


def style(ax, title, ylabel):
    ax.set_title(
        title, loc="left", fontweight="bold", pad=12
    )
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, pad=7)


def create_figure(kpis):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    x = np.arange(len(MONTHS))
    width = 0.34

    high = (
        kpis[kpis["fee_tier"] == "0.30%"]
        .set_index("month").reindex(MONTHS)
    )

    ax = axes[0, 0]
    ax.plot(
        x, high["volume_share_pct"],
        "o-", color="#F7834D",
        linewidth=2.2, label="Volume share"
    )
    ax.plot(
        x, high["event_share_pct"],
        "s-", color="#4A7BB7",
        linewidth=2.2, label="Event share"
    )
    style(
        ax, "A | 0.30% Pool Market Share",
        "Share of two pools (%)"
    )
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for i, fee in enumerate(FEES):
        part = (
            kpis[kpis["fee_tier"] == fee]
            .set_index("month").reindex(MONTHS)
        )
        ax.bar(
            x + (i - 0.5) * width,
            part["unique_origins"],
            width, color=COLORS[fee], label=fee
        )
    style(
        ax, "B | Monthly Active Transaction Origins",
        "Unique origin addresses"
    )
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for i, fee in enumerate(FEES):
        part = (
            kpis[kpis["fee_tier"] == fee]
            .set_index("month").reindex(MONTHS)
        )
        ax.bar(
            x + (i - 0.5) * width,
            part["top_1pct_origin_volume_share_pct"],
            width, color=COLORS[fee], label=fee
        )
    style(
        ax, "C | Top 1% Origin Volume Concentration",
        "Share of attributed volume (%)"
    )
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for fee in FEES:
        part = (
            kpis[kpis["fee_tier"] == fee]
            .set_index("month").reindex(MONTHS)
        )
        ax.plot(
            x[1:],
            part["prev_month_reappearance_pct"].to_numpy()[1:],
            "o-", linewidth=2.2,
            color=COLORS[fee], label=fee
        )
    style(
        ax, "D | Previous-Month Origin Reappearance",
        "Previous-month origins seen again (%)"
    )
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)

    for ax in axes.flat:
        ax.set_xticks(x, ["Jul", "Aug", "Sep"])
        ax.set_xlim(-0.5, 2.5)

    fig.suptitle(
        "Uniswap V3 | Q3 Product and Business Analytics",
        fontsize=19, fontweight="bold", y=0.99
    )
    fig.text(
        0.5, 0.945,
        "USDC/WETH | July 1 – September 30, 2026 (UTC)",
        ha="center", color="#555555"
    )
    fig.text(
        0.5, 0.020,
        "Source: The Graph / Uniswap V3 Subgraph | "
        "Origin addresses are not unique users | "
        "Activity reappearance is not user retention",
        ha="center", fontsize=8.5, color="#666666"
    )

    fig.tight_layout(
        rect=[0.025, 0.055, 0.99, 0.91],
        h_pad=2.6, w_pad=2.0
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "17_q3_product_business_analytics.png"

    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return path


def main():
    ANALYTICS.mkdir(parents=True, exist_ok=True)

    activity, arrays, checks = collect_activity()
    kpis, overlap = calculate_kpis(activity, arrays, checks)

    kpis.to_csv(
        ANALYTICS / "q3_product_kpis_monthly.csv",
        index=False
    )
    overlap.to_csv(
        ANALYTICS / "q3_origin_cross_pool_overlap.csv",
        index=False
    )

    figure = create_figure(kpis)

    print("\n=== Q3 PRODUCT KPI SUMMARY ===")

    columns = [
        "month", "fee_tier",
        "swap_events", "volume_share_pct",
        "event_share_pct", "unique_origins",
        "origin_coverage_pct",
        "top_1pct_origin_volume_share_pct",
        "prev_month_reappearance_pct"
    ]

    print(
        kpis[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    print("\n=== CROSS-POOL ORIGIN OVERLAP ===")
    print(
        overlap.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    print(f"\nFigure: {figure}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
