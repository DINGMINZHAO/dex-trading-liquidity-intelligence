
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))

OUT = DATA / "processed/q3_lp_activity"
ANALYTICS = DATA / "analytics"
FIGURES = ROOT / "reports/figures"

SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
URL = f"https://gateway.thegraph.com/api/subgraphs/id/{SUBGRAPH}"

POOLS = {
    "0.05%": ("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", 500),
    "0.30%": ("0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8", 3000),
}

START = pd.Timestamp("2026-08-10", tz="UTC")
END = pd.Timestamp("2026-09-11", tz="UTC")
COLORS = {"0.05%": "#4A7BB7", "0.30%": "#F7834D"}


def session_factory():
    key = os.getenv("GRAPH_API_KEY")
    if not key:
        raise RuntimeError("GRAPH_API_KEY missing")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )))
    return session


def request(session, query, variables=None):
    response = session.post(
        URL,
        json={"query": query, "variables": variables or {}},
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()

    if result.get("errors") or not result.get("data"):
        raise RuntimeError(f"GraphQL error: {result.get('errors')}")

    data = result["data"]
    if data.get("_meta", {}).get("hasIndexingErrors"):
        raise RuntimeError("Subgraph indexing error")
    return data


DAILY_QUERY = """
query($start: Int!, $end: Int!) {
  _meta { hasIndexingErrors }

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
    date liquidity tick
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
    date liquidity tick
  }
}
"""


def get_daily_liquidity(session):
    path = OUT / "q3_active_liquidity.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        print("REUSED: Q3 active liquidity")
    else:
        data = request(session, DAILY_QUERY, {
            "start": int(pd.Timestamp(
                "2026-07-01", tz="UTC"
            ).timestamp()),
            "end": int(pd.Timestamp(
                "2026-10-01", tz="UTC"
            ).timestamp()),
        })

        rows = []
        for alias, fee in [("low", "0.05%"), ("high", "0.30%")]:
            for record in data[alias]:
                rows.append({
                    "date": pd.to_datetime(
                        int(record["date"]), unit="s", utc=True
                    ),
                    "fee_tier": fee,
                    "liquidity_raw": record["liquidity"],
                    "tick": (
                        int(record["tick"])
                        if record["tick"] is not None else None
                    ),
                })

        df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["date"], utc=True)

    expected = {
        (day, fee)
        for day in pd.date_range(
            "2026-07-01", "2026-09-30", tz="UTC"
        )
        for fee in POOLS
    }

    if (
        len(df) != 184
        or df.duplicated(["date", "fee_tier"]).any()
        or set(zip(df["date"], df["fee_tier"])) != expected
    ):
        raise RuntimeError("Daily active liquidity coverage mismatch")

    df["active_liquidity"] = df["liquidity_raw"].map(
        lambda x: float(int(x))
    )

    if (
        not np.isfinite(df["active_liquidity"]).all()
        or (df["active_liquidity"] <= 0).any()
    ):
        raise RuntimeError("Invalid active liquidity")

    if not path.exists():
        df.to_parquet(path, index=False, compression="zstd")

    return df


def get_events(session, kind, label, address, fee):
    path = OUT / f"{kind}_{fee}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        print(f"REUSED: {kind} {label}: {len(df):,}")
        return df

    cursor = ""
    rows = []
    page = 0

    while True:
        query = f"""
        {{
          _meta {{ hasIndexingErrors }}

          events: {kind}(
            first: 1000
            orderBy: id
            orderDirection: asc
            where: {{
              pool: "{address}"
              timestamp_gte: {int(START.timestamp())}
              timestamp_lt: {int(END.timestamp())}
              id_gt: {json.dumps(cursor)}
            }}
          ) {{
            id
            timestamp
            pool {{ id }}
            amount
            amountUSD
            tickLower
            tickUpper
            owner
          }}
        }}
        """

        batch = request(session, query)["events"]
        ids = [r["id"] for r in batch]

        if (
            ids != sorted(ids)
            or len(ids) != len(set(ids))
            or (cursor and ids and ids[0] <= cursor)
        ):
            raise RuntimeError(f"Pagination error: {kind} {label}")

        for event in batch:
            timestamp = pd.to_datetime(
                int(event["timestamp"]), unit="s", utc=True
            )

            if (
                not START <= timestamp < END
                or event["pool"]["id"].lower() != address
            ):
                raise RuntimeError("Event window/pool mismatch")

            amount = int(event["amount"])
            if amount < 0:
                raise RuntimeError("Negative liquidity amount")

            amount_usd = (
                float(event["amountUSD"])
                if event["amountUSD"] is not None
                else np.nan
            )

            if not (np.isnan(amount_usd) or
                    (np.isfinite(amount_usd) and amount_usd >= 0)):
                raise RuntimeError("Invalid event USD amount")

            rows.append({
                "event_id": event["id"],
                "timestamp_utc": timestamp,
                "date": timestamp.floor("D"),
                "fee_tier": label,
                "event_type": kind,
                "owner": event["owner"],
                "liquidity_amount": str(amount),
                "nonzero_liquidity": amount > 0,
                "amount_usd": amount_usd,
                "tick_lower": int(event["tickLower"]),
                "tick_upper": int(event["tickUpper"]),
            })

        page += 1

        if len(batch) < 1000:
            break

        cursor = ids[-1]

        if page > 2000:
            raise RuntimeError("Pagination safety limit exceeded")

    columns = [
        "event_id", "timestamp_utc", "date", "fee_tier",
        "event_type", "owner", "liquidity_amount",
        "nonzero_liquidity", "amount_usd",
        "tick_lower", "tick_upper"
    ]
    df = pd.DataFrame(rows, columns=columns)

    if df["event_id"].duplicated().any():
        raise RuntimeError("Duplicate liquidity events")

    df.to_parquet(path, index=False, compression="zstd")
    print(f"SAVED: {kind} {label}: {len(df):,} events")
    return df


def analyze(active, events):
    market = pd.read_csv(
        ANALYTICS / "q3_liquidity_daily.csv"
    )
    market["date"] = pd.to_datetime(market["date"], utc=True)

    df = active.merge(
        market[[
            "date", "fee_tier", "tvl_usd",
            "volume_usd_swaps", "volume_share_pct"
        ]],
        on=["date", "fee_tier"],
        validate="one_to_one",
    )

    if len(df) != 184:
        raise RuntimeError("Market and liquidity alignment failed")

    df["month"] = df["date"].dt.strftime("%Y-%m")

    for fee in POOLS:
        mask = df["fee_tier"] == fee
        july = mask & (df["month"] == "2026-07")
        baseline = df.loc[july, "active_liquidity"].median()

        df.loc[mask, "active_liquidity_index"] = (
            100 * df.loc[mask, "active_liquidity"] / baseline
        )

    focus_days = pd.date_range(
        START, END - pd.Timedelta(days=1), tz="UTC"
    )

    counts = pd.DataFrame({"date": focus_days})

    for fee, (_, code) in POOLS.items():
        for kind in ("mints", "burns"):
            part = events[
                (events["fee_tier"] == fee)
                & (events["event_type"] == kind)
                & events["nonzero_liquidity"]
            ]

            series = part.groupby("date").size()
            counts[f"{kind}_{code}"] = (
                counts["date"].map(series).fillna(0).astype(int)
            )

    monthly = (
        df.groupby(["month", "fee_tier"], as_index=False)
        .agg(
            median_active_index=("active_liquidity_index", "median"),
            mean_tvl_usd=("tvl_usd", "mean"),
            volume_usd=("volume_usd_swaps", "sum"),
        )
    )

    df.to_csv(
        ANALYTICS / "q3_active_liquidity_daily.csv", index=False
    )
    counts.to_csv(
        ANALYTICS / "q3_lp_event_daily_focus.csv", index=False
    )
    monthly.to_csv(
        ANALYTICS / "q3_active_liquidity_monthly.csv", index=False
    )

    return df, counts, monthly


def plot(df, counts):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    ax = axes[0, 0]
    for fee in POOLS:
        part = df[df["fee_tier"] == fee].sort_values("date")
        ax.plot(
            part["date"], part["active_liquidity_index"],
            label=fee, color=COLORS[fee], linewidth=2
        )

    ax.set_title(
        "A | Q3 Active Liquidity Index",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Index (July median = 100)")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    high = df[
        (df["fee_tier"] == "0.30%")
        & (df["date"] >= START)
        & (df["date"] < END)
    ].sort_values("date").copy()

    baseline = high[
        high["date"] < pd.Timestamp("2026-08-17", tz="UTC")
    ]

    for field, name, color in [
        ("active_liquidity", "Active liquidity", "#F7834D"),
        ("tvl_usd", "TVL", "#4A7BB7"),
    ]:
        base = baseline[field].median()
        ax.plot(
            high["date"],
            100 * high[field] / base,
            label=name, color=color, linewidth=2
        )

    ax.set_title(
        "B | 0.30% Pool: Liquidity vs TVL",
        loc="left", fontweight="bold"
    )
    ax.set_ylabel("Index (Aug 10–16 median = 100)")
    ax.legend(frameon=False)

    x = mdates.date2num(counts["date"])

    for ax, (fee, (_, code)), letter in zip(
        axes[1], POOLS.items(), ["C", "D"]
    ):
        ax.bar(
            x - 0.21, counts[f"mints_{code}"],
            width=0.42, color="#4A7BB7", label="Mint"
        )
        ax.bar(
            x + 0.21, counts[f"burns_{code}"],
            width=0.42, color="#D9412B", label="Burn"
        )
        ax.set_title(
            f"{letter} | {fee} Pool LP Events",
            loc="left", fontweight="bold"
        )
        ax.set_ylabel("Nonzero-liquidity events / day")
        ax.legend(frameon=False, ncol=2)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
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
        "Uniswap V3 | Active Liquidity and LP Activity",
        fontsize=19, fontweight="bold", y=0.99
    )
    fig.text(
        0.5, 0.945,
        "USDC/WETH | Q3 2026 snapshots | "
        "Mint/Burn focus: Aug 10 – Sep 10 (UTC)",
        ha="center", color="#555555"
    )
    fig.text(
        0.5, 0.018,
        "Source: The Graph / Uniswap V3 Subgraph | "
        "Active liquidity changes include tick crossings | "
        "Mint/Burn events are not unique LPs",
        ha="center", fontsize=8.5, color="#666666"
    )

    fig.tight_layout(rect=[0.025, 0.05, 0.99, 0.91],
                     h_pad=2.8, w_pad=2.3)

    path = FIGURES / "14_q3_active_liquidity_lp_activity.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ANALYTICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    session = session_factory()

    try:
        active = get_daily_liquidity(session)

        frames = []
        for fee, (address, code) in POOLS.items():
            for kind in ("mints", "burns"):
                frames.append(
                    get_events(session, kind, fee, address, code)
                )
    finally:
        session.close()

    events = pd.concat(frames, ignore_index=True)
    daily, counts, monthly = analyze(active, events)
    figure = plot(daily, counts)

    print("\n=== ACTIVE LIQUIDITY: MONTHLY SUMMARY ===")
    print(monthly.to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"
    ))

    print("\n=== MINT/BURN FOCUS WINDOW ===")
    for fee, (_, code) in POOLS.items():
        print(
            f"{fee}: Mint={counts[f'mints_{code}'].sum():,}, "
            f"Burn={counts[f'burns_{code}'].sum():,}"
        )

    print("\n=== 0.30% POOL: BEFORE VS AFTER ===")

    high = daily[daily["fee_tier"] == "0.30%"]

    for name, start, end in [
        ("Before", "2026-08-10", "2026-08-17"),
        ("After", "2026-08-24", "2026-08-31"),
    ]:
        period = high[
            (high["date"] >= pd.Timestamp(start, tz="UTC"))
            & (high["date"] < pd.Timestamp(end, tz="UTC"))
        ]

        print(
            f"{name}: Active index="
            f"{period['active_liquidity_index'].median():.2f}, "
            f"TVL=${period['tvl_usd'].median()/1e6:.2f}M, "
            f"Volume share="
            f"{period['volume_share_pct'].median():.2f}%"
        )

    print(f"\nFigure: {figure}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
