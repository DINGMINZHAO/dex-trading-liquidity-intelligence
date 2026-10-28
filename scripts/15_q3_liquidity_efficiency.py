
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
ANALYTICS = DATA / "analytics"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "reports/figures"
SYSTEM = DATA / "_system"

SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
URL = f"https://gateway.thegraph.com/api/subgraphs/id/{SUBGRAPH}"

POOLS = {
    "low": (
        "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        500, "0.05%"
    ),
    "high": (
        "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        3000, "0.30%"
    ),
}

COLORS = {
    "0.05%": "#4A7BB7",
    "0.30%": "#F7834D",
}

QUERY = """
query Daily($start: Int!, $end: Int!) {
  _meta {
    block { number }
    hasIndexingErrors
  }

  lowPool: pool(
    id: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
  ) {
    id feeTier
    token0 { id decimals }
    token1 { id decimals }
  }

  highPool: pool(
    id: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
  ) {
    id feeTier
    token0 { id decimals }
    token1 { id decimals }
  }

  low: poolDayDatas(
    first: 100
    orderBy: date
    orderDirection: asc
    where: {
      pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
      date_gte: $start
      date_lt: $end
    }
  ) {
    date volumeUSD tvlUSD txCount
  }

  high: poolDayDatas(
    first: 100
    orderBy: date
    orderDirection: asc
    where: {
      pool: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
      date_gte: $start
      date_lt: $end
    }
  ) {
    date volumeUSD tvlUSD txCount
  }
}
"""


def fetch_jul_aug():
    key = os.getenv("GRAPH_API_KEY")
    if not key:
        raise RuntimeError("Missing GRAPH_API_KEY")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"

    session.mount(
        "https://",
        HTTPAdapter(max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        ))
    )

    try:
        response = session.post(
            URL,
            json={
                "query": QUERY,
                "variables": {
                    "start": int(pd.Timestamp(
                        "2026-07-01", tz="UTC"
                    ).timestamp()),
                    "end": int(pd.Timestamp(
                        "2026-09-01", tz="UTC"
                    ).timestamp()),
                },
            },
            timeout=90,
        )
        response.raise_for_status()
        result = response.json()
    finally:
        session.close()

    if result.get("errors") or not result.get("data"):
        raise RuntimeError(
            f"Subgraph error: {result.get('errors')}"
        )

    data = result["data"]

    if data["_meta"]["hasIndexingErrors"]:
        raise RuntimeError("Subgraph indexing error")

    rows = []

    for prefix, (address, fee, label) in POOLS.items():
        pool = data[f"{prefix}Pool"]

        if (
            not pool
            or pool["id"].lower() != address
            or int(pool["feeTier"]) != fee
            or pool["token0"]["id"].lower()
            != "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
            or pool["token1"]["id"].lower()
            != "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
            or int(pool["token0"]["decimals"]) != 6
            or int(pool["token1"]["decimals"]) != 18
        ):
            raise RuntimeError(
                f"Pool metadata mismatch: {label}"
            )

        for item in data[prefix]:
            rows.append({
                "date": pd.to_datetime(
                    int(item["date"]), unit="s", utc=True
                ),
                "fee_tier": label,
                "pool_address": address,
                "volume_usd": float(item["volumeUSD"]),
                "tvl_usd": float(item["tvlUSD"]),
                "tx_count_subgraph": int(item["txCount"]),
            })

    return pd.DataFrame(rows), data["_meta"]["block"]["number"]


def main():
    for path in [ANALYTICS, PROCESSED, FIGURES, SYSTEM]:
        path.mkdir(parents=True, exist_ok=True)

    existing = PROCESSED / "pool_daily_q3_2026.parquet"

    if existing.exists():
        daily = pd.read_parquet(existing)
        indexed_block = "reused_existing_parquet"
        print("Reusing Q3 pool daily Parquet")
    else:
        earlier, indexed_block = fetch_jul_aug()

        september = pd.read_parquet(
            PROCESSED / "pool_daily_sep2026.parquet"
        )

        daily = pd.concat(
            [earlier, september], ignore_index=True
        )

    daily["date"] = pd.to_datetime(
        daily["date"], utc=True
    )

    dates = pd.date_range(
        "2026-07-01", "2026-09-30", tz="UTC"
    )

    expected = pd.MultiIndex.from_product(
        [dates, ["0.05%", "0.30%"]],
        names=["date", "fee_tier"],
    )

    daily = daily.set_index(
        ["date", "fee_tier"]
    ).sort_index()

    if (
        len(daily) != 184
        or not daily.index.is_unique
        or not daily.index.equals(expected.sort_values())
    ):
        raise RuntimeError(
            "Missing or duplicate Q3 pool-day data"
        )

    daily = daily.reset_index()

    for field in ["volume_usd", "tvl_usd"]:
        values = daily[field].to_numpy(dtype=float)

        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
        ):
            raise ValueError(f"Invalid {field}")

    swap = pd.read_csv(
        ANALYTICS / "q3_daily_market_activity.csv"
    )

    swap["date"] = pd.to_datetime(
        swap["date"], utc=True
    )

    merged = daily.merge(
        swap[[
            "date", "fee_tier",
            "volume_usd", "swap_events"
        ]],
        on=["date", "fee_tier"],
        how="inner",
        validate="one_to_one",
        suffixes=("_subgraph", "_swaps"),
    )

    if len(merged) != 184:
        raise RuntimeError(
            "Pool-day and Swap-day alignment failed"
        )

    merged["volume_gap_pct"] = 100 * (
        merged["volume_usd_swaps"]
        / merged["volume_usd_subgraph"] - 1
    )

    maximum = merged["volume_gap_pct"].abs().max()

    if maximum > 2:
        raise RuntimeError(
            f"Daily volume reconciliation gap: {maximum:.4f}%"
        )

    merged["turnover"] = (
        merged["volume_usd_swaps"] / merged["tvl_usd"]
    )

    merged["month"] = (
        merged["date"].dt.strftime("%Y-%m")
    )

    merged["volume_share_pct"] = (
        100 * merged["volume_usd_swaps"]
        / merged.groupby("date")[
            "volume_usd_swaps"
        ].transform("sum")
    )

    merged["tvl_share_pct"] = (
        100 * merged["tvl_usd"]
        / merged.groupby("date")[
            "tvl_usd"
        ].transform("sum")
    )

    monthly = (
        merged.groupby(
            ["month", "fee_tier"], as_index=False
        )
        .agg(
            swap_events=("swap_events", "sum"),
            volume_usd=("volume_usd_swaps", "sum"),
            mean_tvl_usd=("tvl_usd", "mean"),
            median_daily_turnover=("turnover", "median"),
            mean_daily_turnover=("turnover", "mean"),
        )
    )

    monthly["volume_share_pct"] = (
        100 * monthly["volume_usd"]
        / monthly.groupby("month")[
            "volume_usd"
        ].transform("sum")
    )

    monthly["mean_tvl_share_pct"] = (
        100 * monthly["mean_tvl_usd"]
        / monthly.groupby("month")[
            "mean_tvl_usd"
        ].transform("sum")
    )

    daily.to_parquet(
        existing, index=False, compression="zstd"
    )

    merged.to_csv(
        ANALYTICS / "q3_liquidity_daily.csv",
        index=False,
    )

    monthly.to_csv(
        ANALYTICS / "q3_liquidity_monthly.csv",
        index=False,
    )

    manifest = {
        "source": "Uniswap V3 Subgraph via The Graph",
        "subgraph_id": SUBGRAPH,
        "start_utc": "2026-07-01",
        "end_exclusive_utc": "2026-10-01",
        "row_count": len(daily),
        "indexed_block": indexed_block,
        "max_volume_gap_pct": float(maximum),
        "validated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    (SYSTEM / "pool_daily_q3_2026_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 9.5)
    )

    for idx, label in enumerate(["0.05%", "0.30%"]):
        part = merged[
            merged["fee_tier"] == label
        ].sort_values("date")

        ax = axes[0, idx]

        ax.plot(
            part["date"],
            part["tvl_usd"] / 1e6,
            color=COLORS[label],
            linewidth=2,
        )

        letter = "A" if idx == 0 else "B"

        ax.set_title(
            f"{letter} | {label} Pool TVL",
            loc="left",
            fontweight="bold",
        )

        ax.set_ylabel("TVL (USD millions)")

        axes[1, 0].plot(
            part["date"],
            part["turnover"],
            color=COLORS[label],
            linewidth=1.7,
            label=label,
        )

    axes[1, 0].set_title(
        "C | Daily trading volume / TVL",
        loc="left",
        fontweight="bold",
    )

    axes[1, 0].set_ylabel("Daily turnover (x)")
    axes[1, 0].legend(frameon=False)

    high = merged[
        merged["fee_tier"] == "0.30%"
    ].sort_values("date")

    axes[1, 1].plot(
        high["date"],
        high["volume_share_pct"]
        .rolling(7, min_periods=1).mean(),
        label="Trading volume share",
        color=COLORS["0.30%"],
        linewidth=2,
    )

    axes[1, 1].plot(
        high["date"],
        high["tvl_share_pct"]
        .rolling(7, min_periods=1).mean(),
        label="TVL share",
        color=COLORS["0.05%"],
        linewidth=2,
    )

    axes[1, 1].set_title(
        "D | 0.30% Pool share (7-day moving average)",
        loc="left",
        fontweight="bold",
    )

    axes[1, 1].set_ylabel(
        "Share of the two pools (%)"
    )

    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("Date (UTC)")

        for boundary in ["2026-08-01", "2026-09-01"]:
            ax.axvline(
                pd.Timestamp(boundary, tz="UTC"),
                color="#999999",
                linestyle="--",
                alpha=0.5,
            )

    fig.suptitle(
        "Uniswap V3 | Q3 Liquidity and Capital Efficiency",
        fontsize=19,
        fontweight="bold",
        y=0.99,
    )

    fig.text(
        0.5, 0.945,
        "USDC/WETH | July 1 – September 30, 2026 (UTC)",
        ha="center",
        color="#555555",
    )

    fig.text(
        0.5, 0.015,
        "Source: The Graph | TVL: daily snapshot | "
        "Turnover: daily Swap volume / daily TVL",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    fig.tight_layout(
        rect=[0.025, 0.045, 0.99, 0.91],
        h_pad=2.5,
    )

    output = (
        FIGURES / "12_q3_liquidity_capital_efficiency.png"
    )

    fig.savefig(
        output, dpi=250, bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nQ3 pool-day rows: {len(daily)}/184")
    print(
        "Max daily volume reconciliation gap: "
        f"{maximum:.5f}%"
    )

    print("\n=== MONTHLY LIQUIDITY SUMMARY ===")

    cols = [
        "month", "fee_tier", "volume_usd",
        "mean_tvl_usd", "median_daily_turnover",
        "volume_share_pct", "mean_tvl_share_pct"
    ]

    print(
        monthly[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.3f}",
        )
    )

    print(f"\nFigure: {output}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
