
import json
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
PARTITIONS = DATA / "processed/swaps_2026q3"
MANIFEST = DATA / "_system/swaps_2026q3_manifest.json"
DAILY_REF = DATA / "processed/pool_daily_sep2026.parquet"
ANALYTICS = DATA / "analytics"
FIGURES = ROOT / "reports/figures"

POOLS = {
    "0.05%": (500, "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"),
    "0.30%": (3000, "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"),
}

COLORS = {"0.05%": "#4A7BB7", "0.30%": "#F7834D"}
EDGES = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, np.inf]
LABELS = [
    "<$1", "$1–10", "$10–100", "$100–1K",
    "$1K–10K", "$10K–100K", "$100K–1M", "≥$1M"
]

COLUMNS = [
    "event_id", "transaction_hash", "log_index",
    "timestamp_utc", "fee_tier", "pool_address",
    "trade_size_usd", "trade_direction"
]


def load_validate():
    if not MANIFEST.exists() or not DAILY_REF.exists():
        raise FileNotFoundError("Manifest or September daily reference missing")

    manifest = json.loads(MANIFEST.read_text())
    records = manifest.get("partitions", {})

    reference = pd.read_parquet(DAILY_REF)
    reference["day"] = pd.to_datetime(
        reference["date"], utc=True
    ).dt.strftime("%Y-%m-%d")

    ref = {
        (r.day, r.fee_tier): float(r.volume_usd)
        for r in reference.itertuples()
    }

    days = pd.date_range("2026-07-01", "2026-09-30", tz="UTC")
    seen = set()
    daily_rows = []
    bucket_rows = []
    reconciliation = []

    for day_ts in days:
        day = day_ts.strftime("%Y-%m-%d")

        for label, (fee, address) in POOLS.items():
            path = PARTITIONS / f"pool_{fee:04d}" / f"{day}.parquet"

            if not path.exists():
                raise FileNotFoundError(f"Missing partition: {path}")

            df = pd.read_parquet(path, columns=COLUMNS)

            if df.empty:
                raise ValueError(f"Empty partition: {day} {label}")

            ids = df["event_id"].tolist()

            if (
                df["event_id"].isna().any()
                or len(ids) != len(set(ids))
                or any(event_id in seen for event_id in ids)
            ):
                raise ValueError(f"Duplicate/null event IDs: {day} {label}")

            seen.update(ids)

            ts = pd.to_datetime(df["timestamp_utc"], utc=True)
            volumes = pd.to_numeric(df["trade_size_usd"], errors="coerce")
            values = volumes.to_numpy(dtype=float)

            if not (
                (ts >= day_ts).all()
                and (ts < day_ts + pd.Timedelta(days=1)).all()
                and df["fee_tier"].eq(label).all()
                and df["pool_address"].str.lower().eq(address).all()
                and np.isfinite(values).all()
                and (values >= 0).all()
            ):
                raise ValueError(f"Data validation failed: {day} {label}")

            valid_directions = {
                "BUY_WETH", "SELL_WETH", "UNDETERMINED_ZERO_LEG"
            }
            if not df["trade_direction"].isin(valid_directions).all():
                raise ValueError(f"Invalid directions: {day} {label}")

            tx_logs = df.dropna(subset=["log_index"])
            if tx_logs.duplicated(["transaction_hash", "log_index"]).any():
                raise ValueError(f"Duplicate transaction/log: {day} {label}")

            volume_usd = float(values.sum())
            key = f"{day}/{fee}"

            if key not in records:
                raise ValueError(f"Missing manifest entry: {key}")

            if int(records[key]["events"]) != len(df):
                raise ValueError(f"Manifest count mismatch: {key}")

            manifest_volume = float(records[key]["volume_usd"])
            if abs(manifest_volume - volume_usd) > 0.02:
                raise ValueError(f"Manifest volume mismatch: {key}")

            buy = df["trade_direction"].eq("BUY_WETH")
            sell = df["trade_direction"].eq("SELL_WETH")

            daily_rows.append({
                "date": day,
                "month": day[:7],
                "fee_tier": label,
                "swap_events": len(df),
                "volume_usd": volume_usd,
                "buy_events": int(buy.sum()),
                "sell_events": int(sell.sum()),
                "buy_volume_usd": float(volumes[buy].sum()),
                "sell_volume_usd": float(volumes[sell].sum()),
            })

            counts, _ = np.histogram(values, bins=EDGES)
            bucket_volumes, _ = np.histogram(
                values, bins=EDGES, weights=values
            )

            for bucket, count, amount in zip(
                LABELS, counts, bucket_volumes
            ):
                bucket_rows.append({
                    "fee_tier": label,
                    "size_bucket": bucket,
                    "swap_events": int(count),
                    "volume_usd": float(amount),
                })

            if (day, label) in ref:
                reference_volume = ref[(day, label)]
                gap = (
                    100 * (volume_usd / reference_volume - 1)
                    if reference_volume > 0 else np.nan
                )
                reconciliation.append({
                    "date": day,
                    "fee_tier": label,
                    "volume_gap_pct": gap,
                })

    daily = pd.DataFrame(daily_rows)
    buckets = (
        pd.DataFrame(bucket_rows)
        .groupby(["fee_tier", "size_bucket"], as_index=False)
        .agg(swap_events=("swap_events", "sum"),
             volume_usd=("volume_usd", "sum"))
    )

    recon = pd.DataFrame(reconciliation)
    if len(recon) != 60 or recon["volume_gap_pct"].isna().any():
        raise ValueError("Incomplete September volume reconciliation")

    max_gap = recon["volume_gap_pct"].abs().max()
    if max_gap > 2:
        raise ValueError(f"Volume reconciliation failed: {max_gap:.3f}%")

    print("\n=== Q3 DATA VALIDATION ===")
    print(f"Pool-day partitions: {len(daily)}/184")
    print(f"Unique Swap events: {len(seen):,}")
    print("Duplicate events: 0")
    print(f"September max volume gap: {max_gap:.4f}%")
    print("VALIDATION: PASS")

    return daily, buckets, recon


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, pad=6)


def plot_activity(daily):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)

    for ax, label in zip(axes, POOLS):
        part = daily[daily["fee_tier"] == label].copy()
        x = pd.to_datetime(part["date"], utc=True)
        y = part["volume_usd"] / 1e6

        ax.plot(x, y, color=COLORS[label], linewidth=1.8)
        ax.fill_between(x, y, color=COLORS[label], alpha=0.12)

        for boundary in ["2026-08-01", "2026-09-01"]:
            ax.axvline(
                pd.Timestamp(boundary, tz="UTC"),
                color="#999999", linestyle="--", alpha=0.6
            )

        ax.set_title(f"{label} Pool", loc="left", fontweight="bold")
        ax.set_ylabel("Daily volume (USD millions)")
        ax.set_ylim(bottom=0)
        style(ax)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date (UTC)")

    fig.suptitle(
        "Uniswap V3 | Q3 Daily Trading Volume",
        fontsize=19, fontweight="bold", y=0.98
    )
    fig.text(
        0.5, 0.935,
        "USDC/WETH | July 1 – September 30, 2026 (UTC) | Independent vertical scales",
        ha="center", color="#555555", fontsize=10
    )
    fig.text(
        0.5, 0.025,
        "Source: The Graph / Uniswap V3 Subgraph | Pool-level Swap USD volume",
        ha="center", fontsize=9, color="#666666"
    )

    fig.subplots_adjust(top=0.87, bottom=0.11, hspace=0.34)
    path = FIGURES / "08_q3_daily_trading_volume.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sizes(buckets):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    x = np.arange(len(LABELS))
    width = 0.36

    for ax, metric, heading in zip(
        axes,
        ["event_share_pct", "volume_share_pct"],
        ["A | Share of Swap Events", "B | Share of Trading Volume"]
    ):
        for i, label in enumerate(POOLS):
            part = buckets[buckets["fee_tier"] == label]
            part = part.set_index("size_bucket").reindex(LABELS)

            ax.bar(
                x + (i - 0.5) * width,
                part[metric],
                width=width,
                color=COLORS[label],
                label=label
            )

        ax.set_title(heading, loc="left", fontweight="bold")
        ax.set_ylabel("Share (%)")
        ax.set_ylim(bottom=0)
        style(ax)

    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABELS, rotation=20, ha="right")
    axes[1].set_xlabel("Swap size (USD)")

    fig.suptitle(
        "Uniswap V3 | Q3 Trade Size Distribution",
        fontsize=19, fontweight="bold", y=0.98
    )
    fig.text(
        0.5, 0.935,
        "USDC/WETH | July 1 – September 30, 2026 (UTC)",
        ha="center", fontsize=10, color="#555555"
    )
    fig.text(
        0.5, 0.025,
        "Source: The Graph / Uniswap V3 Subgraph | Pool-level Swap events",
        ha="center", fontsize=9, color="#666666"
    )

    fig.subplots_adjust(top=0.87, bottom=0.16, hspace=0.30)
    path = FIGURES / "09_q3_trade_size_distribution.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ANALYTICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    daily, buckets, recon = load_validate()

    monthly = (
        daily.groupby(["month", "fee_tier"], as_index=False)
        .agg(
            swap_events=("swap_events", "sum"),
            volume_usd=("volume_usd", "sum"),
            buy_events=("buy_events", "sum"),
            sell_events=("sell_events", "sum"),
            buy_volume_usd=("buy_volume_usd", "sum"),
            sell_volume_usd=("sell_volume_usd", "sum")
        )
    )

    monthly["volume_share_pct"] = (
        100 * monthly["volume_usd"]
        / monthly.groupby("month")["volume_usd"].transform("sum")
    )

    buckets["event_share_pct"] = (
        100 * buckets["swap_events"]
        / buckets.groupby("fee_tier")["swap_events"].transform("sum")
    )
    buckets["volume_share_pct"] = (
        100 * buckets["volume_usd"]
        / buckets.groupby("fee_tier")["volume_usd"].transform("sum")
    )

    daily.to_csv(ANALYTICS / "q3_daily_market_activity.csv", index=False)
    monthly.to_csv(ANALYTICS / "q3_monthly_market_summary.csv", index=False)
    buckets.to_csv(ANALYTICS / "q3_trade_size_distribution.csv", index=False)
    recon.to_csv(ANALYTICS / "q3_september_reconciliation.csv", index=False)

    fig1 = plot_activity(daily)
    fig2 = plot_sizes(buckets)

    print("\n=== Q3 MONTHLY MARKET SUMMARY ===")
    print(monthly[
        ["month", "fee_tier", "swap_events", "volume_usd", "volume_share_pct"]
    ].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=== Q3 TOTALS BY POOL ===")
    print(daily.groupby("fee_tier").agg(
        events=("swap_events", "sum"),
        volume_usd=("volume_usd", "sum")
    ).to_string(float_format=lambda x: f"{x:,.2f}"))

    print(f"\nFigure 1: {fig1}")
    print(f"Figure 2: {fig2}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
