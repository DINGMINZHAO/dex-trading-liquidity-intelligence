
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))

OUTPUT = DATA / "processed/q3_lp_activity/hourly_transition.parquet"
CSV = DATA / "analytics/q3_hourly_liquidity_validation.csv"
FIGURE = ROOT / "reports/figures/16_hourly_liquidity_validation.png"

POOL = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

WINDOWS = {
    "August transition": ("2026-08-18", "2026-08-24"),
    "September reversal": ("2026-09-01", "2026-09-05"),
}

QUERY = """
query Hours($a: Int!, $b: Int!, $c: Int!, $d: Int!) {
  _meta { hasIndexingErrors }

  aug: poolHourDatas(
    first: 500
    orderBy: periodStartUnix
    orderDirection: asc
    where: {
      pool: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
      periodStartUnix_gte: $a
      periodStartUnix_lt: $b
    }
  ) {
    periodStartUnix liquidity tick tvlUSD volumeUSD
  }

  sep: poolHourDatas(
    first: 500
    orderBy: periodStartUnix
    orderDirection: asc
    where: {
      pool: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
      periodStartUnix_gte: $c
      periodStartUnix_lt: $d
    }
  ) {
    periodStartUnix liquidity tick tvlUSD volumeUSD
  }
}
"""


def timestamp(value):
    return int(pd.Timestamp(value, tz="UTC").timestamp())


def fetch():
    if OUTPUT.exists():
        print("REUSED hourly data")
        return pd.read_parquet(OUTPUT)

    key = os.getenv("GRAPH_API_KEY")
    if not key:
        raise RuntimeError("Missing GRAPH_API_KEY")

    url = f"https://gateway.thegraph.com/api/subgraphs/id/{SUBGRAPH}"

    variables = {
        "a": timestamp("2026-08-18"),
        "b": timestamp("2026-08-24"),
        "c": timestamp("2026-09-01"),
        "d": timestamp("2026-09-05"),
    }

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"query": QUERY, "variables": variables},
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()

    if result.get("errors") or not result.get("data"):
        raise RuntimeError(result.get("errors", "No GraphQL data"))

    data = result["data"]

    if data["_meta"]["hasIndexingErrors"]:
        raise RuntimeError("Subgraph indexing errors")

    rows = []

    for alias, name in [
        ("aug", "August transition"),
        ("sep", "September reversal"),
    ]:
        for item in data[alias]:
            rows.append({
                "window": name,
                "hour": pd.to_datetime(
                    int(item["periodStartUnix"]),
                    unit="s", utc=True
                ),
                "active_liquidity": float(int(item["liquidity"])),
                "tick": (
                    int(item["tick"])
                    if item["tick"] is not None else np.nan
                ),
                "tvl_usd": float(item["tvlUSD"]),
                "volume_usd": float(item["volumeUSD"]),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No hourly data returned")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression="zstd")
    return df


def calculate(df):
    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"], utc=True)

    if df.duplicated(["window", "hour"]).any():
        raise RuntimeError("Duplicate hourly snapshots")

    for name, (start, end) in WINDOWS.items():
        part = df[df["window"] == name]

        if part.empty:
            raise RuntimeError(f"Missing window: {name}")

        if not part["hour"].between(
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"),
            inclusive="left",
        ).all():
            raise RuntimeError(f"Invalid timestamps: {name}")

    numeric = [
        "active_liquidity", "tick", "tvl_usd", "volume_usd"
    ]

    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite hourly measurements")

    if (
        (df["active_liquidity"] <= 0).any()
        or (df["tvl_usd"] <= 0).any()
        or (df["volume_usd"] < 0).any()
    ):
        raise RuntimeError("Invalid liquidity, TVL or volume")

    df = df.sort_values(["window", "hour"]).reset_index(drop=True)

    df["weth_price_usdc"] = (
        1e12 * np.exp(-df["tick"] * np.log1p(0.0001))
    )

    for field in ["active_liquidity", "weth_price_usdc"]:
        df[f"{field}_change_pct"] = (
            df.groupby("window")[field].pct_change() * 100
        )

    for name in WINDOWS:
        mask = df["window"] == name
        part = df.loc[mask]

        base_liq = part["active_liquidity"].iloc[:24].median()
        base_tvl = part["tvl_usd"].iloc[:24].median()

        df.loc[mask, "liquidity_index"] = (
            part["active_liquidity"] / base_liq * 100
        )
        df.loc[mask, "tvl_index"] = (
            part["tvl_usd"] / base_tvl * 100
        )

    return df


def plot(df):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for row, name in enumerate(WINDOWS):
        part = df[df["window"] == name]

        ax = axes[row, 0]
        ax.plot(
            part["hour"], part["liquidity_index"],
            color="#F7834D", linewidth=2,
            label="Active liquidity"
        )
        ax.plot(
            part["hour"], part["tvl_index"],
            color="#4A7BB7", linewidth=2,
            label="TVL"
        )
        ax.set_title(
            f"{'A' if row == 0 else 'C'} | {name}: Liquidity",
            loc="left", fontweight="bold"
        )
        ax.set_ylabel("Index (first 24 observations = 100)")
        ax.legend(frameon=False)

        ax = axes[row, 1]
        ax.plot(
            part["hour"], part["weth_price_usdc"],
            color="#4A7BB7", linewidth=2
        )
        ax.set_ylabel("WETH price (USDC)")

        volume_ax = ax.twinx()
        volume_ax.bar(
            part["hour"],
            part["volume_usd"] / 1e6,
            width=0.025,
            color="#F7834D",
            alpha=0.3,
        )
        volume_ax.set_ylabel("Hourly volume (USD M)")
        volume_ax.spines["top"].set_visible(False)

        ax.set_title(
            f"{'B' if row == 0 else 'D'} | Price and Volume",
            loc="left", fontweight="bold"
        )

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=4, maxticks=7)
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %d", tz=mdates.UTC)
        )
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Uniswap V3 | Hourly Liquidity Transition Diagnostics",
        fontsize=19, fontweight="bold", y=0.99
    )
    fig.text(
        0.5, 0.945,
        "USDC/WETH 0.30% Pool | August 18–23 and September 1–4, 2026 (UTC)",
        ha="center", color="#555555"
    )
    fig.text(
        0.5, 0.018,
        "Source: The Graph | Hour-end snapshots | WETH price derived from tick"
        " | Temporal association does not establish causality",
        ha="center", fontsize=8.5, color="#666666"
    )

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.025, 0.055, 0.985, 0.91],
                     h_pad=2.5, w_pad=2.3)
    fig.savefig(FIGURE, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    df = calculate(fetch())
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)

    for name, (start, end) in WINDOWS.items():
        part = df[df["window"] == name]
        expected = int(
            (pd.Timestamp(end, tz="UTC")
             - pd.Timestamp(start, tz="UTC")).total_seconds() / 3600
        )

        print(f"\n=== {name.upper()} ===")
        print(f"Hourly snapshots: {len(part)}/{expected}")

        top = (
            part.assign(
                magnitude=part[
                    "active_liquidity_change_pct"
                ].abs()
            )
            .nlargest(6, "magnitude")
        )

        print(top[[
            "hour",
            "active_liquidity_change_pct",
            "weth_price_usdc",
            "weth_price_usdc_change_pct",
            "volume_usd",
        ]].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}"
        ))

        if len(part) < expected:
            print(
                "Note: Missing hours limit event-time inference."
            )

    plot(df)
    print(f"\nFigure: {FIGURE}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
