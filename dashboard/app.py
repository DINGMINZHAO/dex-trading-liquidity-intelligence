"""DEX Intelligence — V1 refined.

Streamlit research dashboard using the existing Q3 2026 analysis artifacts.
Run: python -m streamlit run dashboard/app.py
"""

from datetime import date
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_ROOT = Path(os.getenv("DEX_DATA_ROOT", ROOT / "data"))
ANALYTICS = DATA_ROOT / "analytics"
SWAPS = DATA_ROOT / "processed" / "swaps_2026q3"

START, END = date(2026, 7, 1), date(2026, 9, 30)
POOLS = ["0.05%", "0.30%"]
FEES = {"0.05%": 500, "0.30%": 3000}
BLUE, RED = "#356CDB", "#E45454"
COLORS = {"0.05%": BLUE, "0.30%": RED}
INK, MUTED, GRID = "#202D42", "#738094", "#E8EDF4"

SIZE_BINS = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf]
SIZE_BUCKETS = ["<$1", "$1–10", "$10–100", "$100–1K",
                "$1K–10K", "$10K–100K", "$100K–1M", "≥$1M"]
SIZE_SHORT = ["<1", "1–10", "10–100", "100–1K",
              "1K–10K", "10K–100K", "100K–1M", "1M+"]
EXEC_BINS = [100, 1_000, 10_000, 100_000, np.inf]
EXEC_BUCKETS = ["$100–1K", "$1K–10K", "$10K–100K", "≥$100K"]

st.set_page_config(
    page_title="DEX Intelligence | Uniswap V3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The original V1 composition: simple sidebar, KPI row, two-column charts.
# Only a restrained set of CSS rules; no forced chart/card heights or text overlays.
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F7F9FC; }
[data-testid="stHeader"] { background: rgba(247,249,252,.94); }
[data-testid="stSidebar"] { background: #F1F5FA; border-right: 1px solid #E3EAF3; }
.block-container { max-width: 1520px; padding-top: 1.55rem; padding-bottom: 2.7rem; }
[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #E4EAF2;
  border-radius: 10px; padding: 14px 17px;
  min-height: 112px;
}
[data-testid="stMetricLabel"] { color: #64748B; font-size: .83rem; }
[data-testid="stMetricValue"] { color: #202D42; font-weight: 700; letter-spacing: -.7px; }
[data-testid="stVerticalBlockBorderWrapper"] {
  background: white; border: 1px solid #E4EAF2 !important;
  border-radius: 11px !important; box-shadow: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 13px 16px 5px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPlotlyChart"] { margin: 0; }
.dash-kicker { color: #667995; font-weight: 700; letter-spacing: 1.7px;
  font-size: 10px; margin-bottom: 3px; }
.dash-title { font-size: 27px; font-weight: 760; color: #202D42;
  line-height: 1.22; margin: 0; letter-spacing: -.6px; }
.dash-description { color: #8190A4; font-size: 12px; margin: 6px 0 10px; }
.module-title { font-size: 17px; font-weight: 720; color: #202D42;
  margin: 17px 0 5px; }
.chart-heading { font-size: 14px; font-weight: 690; color: #26364E;
  line-height: 1.3; padding: 0; margin: 0; }
.chart-desc { color: #8B98AA; font-size: 11px; margin: 4px 0 0; }
.brand { font-size: 17px; font-weight: 760; color: #202D42;
  line-height: 1.25; margin: 0; }
.brand-sub { font-size: 11px; color: #8591A2; margin: 4px 0 17px; }
.small-note { color: #7A889C; font-size: 11px; line-height: 1.55; }
[data-testid="stAlert"] { border-radius: 9px; }
@media (max-width: 900px) {
  .dash-title { font-size: 22px; }
  .block-container { padding-top: 1rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def csv_data(filename):
    path = ANALYTICS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing data: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def execution_data():
    path = ANALYTICS / "q3_execution_events.parquet"
    columns = ["timestamp_utc", "fee_tier", "trade_direction", "trade_size_usd",
               "observed_cost_bps", "fee_adjusted_penalty_bps",
               "terminal_spot_impact_bps"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing data: {path}")
    df = pd.read_parquet(path, columns=columns)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["day"] = df["timestamp_utc"].dt.date
    df["month"] = df["timestamp_utc"].dt.strftime("%Y-%m")
    df["size_bucket"] = pd.cut(df["trade_size_usd"], bins=EXEC_BINS,
                               labels=EXEC_BUCKETS, right=False)
    return df


@st.cache_data(show_spinner=False)
def trade_size_summary(start, end, tier_tuple):
    """Read only one numeric column per partition, without loading full events."""
    if start == START and end == END:
        path = ANALYTICS / "q3_trade_size_distribution.csv"
        if path.is_file():
            summary = pd.read_csv(path)
            return summary[summary["fee_tier"].isin(tier_tuple)].rename(columns={
                "size_bucket": "bucket", "swap_events": "events",
                "volume_usd": "volume", "event_share_pct": "event_pct",
                "volume_share_pct": "volume_pct",
            })

    rows = []
    for day in pd.date_range(start, end):
        for tier in tier_tuple:
            path = SWAPS / f"pool_{FEES[tier]:04d}" / f"{day:%Y-%m-%d}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing partition: {path}")
            values = pd.read_parquet(path, columns=["trade_size_usd"])["trade_size_usd"]
            a = values.to_numpy(dtype=float)
            if not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f"Invalid swap sizes: {path.name}")
            count, _ = np.histogram(a, SIZE_BINS)
            volume, _ = np.histogram(a, SIZE_BINS, weights=a)
            rows.extend((tier, label, int(n), float(v)) for label, n, v
                        in zip(SIZE_BUCKETS, count, volume))
    out = (pd.DataFrame(rows, columns=["fee_tier", "bucket", "events", "volume"])
           .groupby(["fee_tier", "bucket"], as_index=False).sum())
    for name, result in [("events", "event_pct"), ("volume", "volume_pct")]:
        denominator = out.groupby("fee_tier")[name].transform("sum")
        out[result] = np.where(denominator > 0, 100 * out[name] / denominator, 0)
    return out


def load_dateframe(name):
    df = csv_data(name).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def in_dates(df, name="date"):
    current = pd.to_datetime(df[name], utc=True).dt.date
    return df.loc[current.between(start_date, end_date)].copy()


def fmt_usd(value):
    value = float(value)
    if value >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if value >= 1e3:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


def plot_style(fig, height=345, percent=False, legend=True, horizontal=False):
    # Chart titles live OUTSIDE the figure, so there is no text collision.
    has_legend = legend and len(fig.data) > 1
    fig.update_layout(
        template="plotly_white", height=height, autosize=True,
        margin=dict(l=55 if not horizontal else 75, r=18,
                    t=53 if has_legend else 22, b=47),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11, color=MUTED),
        showlegend=has_legend,
        legend=dict(orientation="h", x=1, xanchor="right",
                    y=1.13, yanchor="top", font=dict(size=11),
                    bgcolor="rgba(255,255,255,0)"),
        hovermode="x unified", bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, zeroline=False,
                     tickfont=dict(size=10), automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10), automargin=True)
    if percent:
        fig.update_yaxes(ticksuffix="%")
    return fig


def panel(title, caption, fig, **kwargs):
    with st.container(border=True):
        st.markdown(f'<div class="chart-heading">{title}</div>', unsafe_allow_html=True)
        if caption:
            st.markdown(f'<div class="chart-desc">{caption}</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_style(fig, **kwargs), use_container_width=True, theme=None,
            config={"displaylogo": False, "scrollZoom": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )


def pool_lines(df, field, divisor=1, x="date"):
    fig = go.Figure()
    for tier in tiers:
        part = df[df["fee_tier"].eq(tier)].sort_values(x)
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part[x], y=part[field] / divisor, name=tier,
            mode="lines", line=dict(color=COLORS[tier], width=2.4),
            hovertemplate="%{x}<br>%{y:,.3f}<extra>" + tier + "</extra>",
        ))
    return fig


# ────────── Sidebar ──────────
with st.sidebar:
    st.markdown('<div class="brand">DEX INTELLIGENCE</div>'
                '<div class="brand-sub">Uniswap V3 · Research dashboard</div>',
                unsafe_allow_html=True)
    page = st.radio("Explore", ["Market Overview", "Execution Intelligence",
                               "Liquidity & Economics", "Product & Research"])
    st.divider()
    period = st.date_input("Date range · UTC", value=(START, END),
                           min_value=START, max_value=END)
    if not isinstance(period, (list, tuple)) or len(period) != 2:
        st.info("Select a start and end date.")
        st.stop()
    start_date, end_date = period
    if start_date > end_date:
        st.error("Start date must precede end date.")
        st.stop()
    tiers = st.multiselect("Fee tiers", POOLS, default=POOLS)
    if not tiers:
        st.warning("Select at least one fee tier.")
        st.stop()
    st.caption("Ethereum Mainnet · Q3 2026 · UTC")

st.markdown('<div class="dash-kicker">ETHEREUM MAINNET / UNISWAP V3</div>'
            '<div class="dash-title">DEX Trading & Liquidity Intelligence</div>'
            '<div class="dash-description">USDC / WETH &nbsp;·&nbsp; '
            f'{start_date:%b %d} – {end_date:%b %d, %Y} &nbsp;·&nbsp; Historical research</div>',
            unsafe_allow_html=True)

try:
    full_market = load_dateframe("q3_liquidity_daily.csv")
    window = in_dates(full_market)
    market = window[window["fee_tier"].isin(tiers)].copy()
    if market.empty:
        st.info("No data for the selected filters.")
        st.stop()

    # ────────── Market ──────────
    if page == "Market Overview":
        total_volume = market["volume_usd_swaps"].sum()
        total_events = int(market["swap_events"].sum())
        full_volume = window["volume_usd_swaps"].sum()
        high_volume = window.loc[window["fee_tier"].eq("0.30%"),
                                 "volume_usd_swaps"].sum()
        high_share = 100 * high_volume / full_volume if full_volume > 0 else np.nan

        a, b, c, d = st.columns(4)
        a.metric("Swap volume", fmt_usd(total_volume))
        b.metric("Swap events", f"{total_events:,}")
        c.metric("Days", f"{market['date'].nunique():,}")
        d.metric("0.30% volume share", f"{high_share:.2f}%",
                 help="Share of both studied pools, regardless of pool filter")

        st.markdown('<div class="module-title">Trading activity</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            totals = (market.groupby("date", as_index=False)["volume_usd_swaps"]
                      .sum().sort_values("date"))
            fig = go.Figure(go.Scatter(
                x=totals["date"], y=totals["volume_usd_swaps"] / 1e6,
                mode="lines", name="Volume", line=dict(color=BLUE, width=2.5),
                fill="tozeroy", fillcolor="rgba(53,108,219,0.08)"))
            panel("Daily volume", "USD millions · selected pools", fig)
        with right:
            high = (window[window["fee_tier"].eq("0.30%")]
                    .sort_values("date"))
            fig = go.Figure(go.Scatter(
                x=high["date"], y=high["volume_share_pct"],
                mode="lines", name="0.30% share", line=dict(color=RED, width=2.5),
                fill="tozeroy", fillcolor="rgba(228,84,84,0.07)"))
            panel("0.30% market share", "Share of two studied pools", fig, percent=True)

        st.markdown('<div class="module-title">Trade sizes</div>', unsafe_allow_html=True)
        mix = trade_size_summary(start_date, end_date, tuple(tiers))
        left, right = st.columns(2)
        for holder, field, title, cap in [
            (left, "volume_pct", "Volume composition", "% of selected pool volume"),
            (right, "event_pct", "Swap composition", "% of selected pool events"),
        ]:
            fig = go.Figure()
            for tier in tiers:
                part = mix[mix["fee_tier"].eq(tier)].set_index("bucket").reindex(SIZE_BUCKETS)
                fig.add_trace(go.Bar(x=SIZE_SHORT, y=part[field], name=tier,
                                     marker_color=COLORS[tier]))
            fig.update_layout(barmode="group")
            fig.update_xaxes(tickangle=-27)
            with holder:
                panel(title, cap, fig, percent=True)
        st.caption("Market shares cover the two selected USDC/WETH fee-tier pools, not all DEXs.")

    # ────────── Execution ──────────
    elif page == "Execution Intelligence":
        left, right = st.columns(2)
        with left:
            direction = st.selectbox("Direction", ["BUY_WETH", "SELL_WETH"],
                                     format_func=lambda x: "Buy WETH" if x == "BUY_WETH" else "Sell WETH")
        with right:
            size_filter = st.selectbox("Trade size", ["All ≥$100"] + EXEC_BUCKETS)

        events = execution_data()
        selected_events = events.loc[
            events["day"].between(start_date, end_date)
            & events["fee_tier"].isin(tiers)
            & events["trade_direction"].eq(direction)
        ].copy()
        if size_filter != "All ≥$100":
            selected_events = selected_events.loc[selected_events["size_bucket"].eq(size_filter)]
        if selected_events.empty:
            st.warning("No matching swaps.")
            st.stop()
        a, b, c, d = st.columns(4)
        a.metric("Eligible swaps", f"{len(selected_events):,}")
        for widget, tier in [(b, "0.05%"), (c, "0.30%")]:
            values = selected_events.loc[selected_events["fee_tier"].eq(tier), "observed_cost_bps"]
            widget.metric(f"{tier} median cost", f"{values.median():.2f} bps" if len(values) else "—")
        d.metric("Direction", "Buy" if direction == "BUY_WETH" else "Sell")

        def cost_chart(column):
            fig = go.Figure()
            for tier in tiers:
                part = selected_events[selected_events["fee_tier"].eq(tier)]
                stats = (part.groupby("size_bucket", observed=True)[column]
                         .agg(median="median", p25=lambda s: s.quantile(.25),
                              p75=lambda s: s.quantile(.75), events="size")
                         .reindex(EXEC_BUCKETS))
                stats = stats.loc[stats["events"].fillna(0).ge(10)]
                if stats.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=stats.index.astype(str), y=stats["median"],
                    name=tier, mode="lines+markers",
                    line=dict(color=COLORS[tier], width=2.3), marker=dict(size=7),
                    error_y=dict(type="data", symmetric=False,
                                 array=stats["p75"] - stats["median"],
                                 arrayminus=stats["median"] - stats["p25"],
                                 thickness=1.2, width=4)))
            return fig

        st.markdown('<div class="module-title">Execution costs</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            panel("Observed cost", "Median and IQR · basis points", cost_chart("observed_cost_bps"))
        with right:
            panel("Fee-adjusted penalty", "Median and IQR · basis points",
                  cost_chart("fee_adjusted_penalty_bps"))
        left, right = st.columns(2)
        with left:
            panel("Terminal price impact", "Median and IQR · basis points",
                  cost_chart("terminal_spot_impact_bps"))
        with right:
            grouped = (selected_events.groupby(["month", "fee_tier"], as_index=False)
                       ["observed_cost_bps"].median())
            fig = go.Figure()
            for tier in tiers:
                subset = grouped[grouped["fee_tier"].eq(tier)]
                fig.add_trace(go.Scatter(x=subset["month"], y=subset["observed_cost_bps"],
                                         name=tier, mode="lines+markers",
                                         line=dict(color=COLORS[tier], width=2.5)))
            panel("Monthly median cost", "Selected events · basis points", fig)
        st.caption("Reference: previous Swap post-price. Gas, routing and MEV excluded; pools are not matched quotes.")

    # ────────── Liquidity ──────────
    elif page == "Liquidity & Economics":
        fees = in_dates(load_dateframe("q3_fee_economics_daily.csv"))
        fees = fees[fees["fee_tier"].isin(tiers)]
        active = in_dates(load_dateframe("q3_active_liquidity_daily.csv"))
        active = active[active["fee_tier"].isin(tiers)]
        last_date = market["date"].max()
        tvl = market.loc[market["date"].eq(last_date), "tvl_usd"].sum()
        combined_day = market.groupby("date").agg(volume=("volume_usd_swaps", "sum"),
                                                   tvl=("tvl_usd", "sum"))
        turnover = (combined_day["volume"] / combined_day["tvl"]).median()
        a, b, c, d = st.columns(4)
        a.metric("Latest TVL", fmt_usd(tvl))
        b.metric("Gross fee estimate", fmt_usd(fees["gross_fee_est_usd"].sum()))
        c.metric("Median turnover", f"{turnover:.3f}×")
        d.metric("Pool-day records", f"{len(market):,}")
        st.markdown('<div class="module-title">Liquidity and efficiency</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            panel("Total value locked", "USD millions · daily snapshot",
                  pool_lines(market, "tvl_usd", 1e6))
        with right:
            panel("Daily turnover", "Trading volume / TVL", pool_lines(market, "turnover"))
        left, right = st.columns(2)
        with left:
            panel("Gross fees", "USD thousands · nominal estimates",
                  pool_lines(fees, "gross_fee_est_usd", 1e3))
        with right:
            panel("Active liquidity", "Within-pool July median = 100",
                  pool_lines(active, "active_liquidity_index"))
        st.caption("Gross fees are nominal estimates, not realized LP returns. Active liquidity is a relative index, not USD.")

    # ────────── Product and Research ──────────
    else:
        product_tab, research_tab = st.tabs(["Address activity", "Volatility & regression"])
        with product_tab:
            st.caption("These address metrics use full calendar months, regardless of the date filter.")
            product = csv_data("q3_product_kpis_monthly.csv")
            product = product[product["fee_tier"].isin(tiers)].copy()
            left, right = st.columns(2)
            with left:
                fig = go.Figure()
                for tier in tiers:
                    part = product[product["fee_tier"].eq(tier)]
                    fig.add_trace(go.Bar(x=part["month"], y=part["unique_origins"],
                                         name=tier, marker_color=COLORS[tier]))
                fig.update_layout(barmode="group")
                panel("Active origins", "Unique transaction-origin addresses", fig)
            with right:
                fig = go.Figure()
                for tier in tiers:
                    part = product[product["fee_tier"].eq(tier)]
                    fig.add_trace(go.Scatter(x=part["month"],
                                             y=part["top_1pct_origin_volume_share_pct"],
                                             name=tier, mode="lines+markers",
                                             line=dict(color=COLORS[tier], width=2.5)))
                panel("Trading concentration", "Volume share from top 1% origins", fig, percent=True)
            left, right = st.columns(2)
            with left:
                fig = go.Figure()
                for tier in tiers:
                    part = product[product["fee_tier"].eq(tier)]
                    fig.add_trace(go.Scatter(x=part["month"],
                                             y=part["prev_month_reappearance_pct"],
                                             name=tier, mode="lines+markers",
                                             line=dict(color=COLORS[tier], width=2.5)))
                panel("Origin reappearance", "Previous-month addresses seen again", fig, percent=True)
            with right:
                if "0.30%" in tiers:
                    high = product[product["fee_tier"].eq("0.30%")]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=high["month"], y=high["volume_share_pct"],
                                             name="Volume", mode="lines+markers",
                                             line=dict(color=RED, width=2.5)))
                    fig.add_trace(go.Scatter(x=high["month"], y=high["event_share_pct"],
                                             name="Events", mode="lines+markers",
                                             line=dict(color=BLUE, width=2.5)))
                    panel("0.30% pool position", "Volume vs event share", fig, percent=True)
                else:
                    st.info("Select the 0.30% pool to see its volume and event shares.")
            st.caption("Origin addresses are not unique users; reappearance is not customer retention.")

        with research_tab:
            vol = load_dateframe("q3_volatility_regimes_daily.csv")
            vol_filtered = in_dates(vol)
            left, right = st.columns(2)
            with left:
                fig = go.Figure(go.Scatter(x=vol_filtered["date"],
                                            y=vol_filtered["realized_vol_pct"],
                                            mode="lines", name="Realized vol",
                                            line=dict(color=BLUE, width=2.5)))
                panel("Daily realized volatility", "0.05% pool hourly close · %", fig, percent=True)
            with right:
                high = window[window["fee_tier"].eq("0.30%")]
                merged = high.merge(vol_filtered[["date", "regime"]], on="date",
                                    validate="one_to_one")
                fig = go.Figure()
                for regime, color in [("Low", BLUE), ("Medium", "#91A3BF"), ("High", RED)]:
                    values = merged.loc[merged["regime"].eq(regime), "volume_share_pct"]
                    if len(values):
                        fig.add_trace(go.Box(y=values, name=regime, marker_color=color,
                                             boxpoints=False))
                panel("Volume share by regime", "0.30% pool · Q3 terciles", fig, percent=True)
            models = csv_data("q3_regression_robustness.csv")
            models = models[models["outcome"].eq("share_pct")].copy()
            fig = go.Figure(go.Scatter(
                x=models["vol_coefficient"], y=models["model"],
                mode="markers", name="Coefficient", marker=dict(color=BLUE, size=9),
                error_x=dict(type="data", symmetric=False,
                             array=models["ci_upper"] - models["vol_coefficient"],
                             arrayminus=models["vol_coefficient"] - models["ci_lower"])))
            fig.add_vline(x=0, line_dash="dash", line_color="#A8B3C1")
            panel("Regression robustness", "Full Q3 · HAC 95% confidence intervals",
                  fig, height=365, legend=False, horizontal=True)
            st.caption("Full-Q3 regression estimates remain fixed when filters change. Association is not causation.")

except (FileNotFoundError, KeyError, ValueError, OSError) as error:
    st.error(f"Could not load the research data: {error}")
    st.info("Check that scripts 13–22 have produced their Q3 CSV and Parquet files.")
    st.stop()

st.caption("DEX Intelligence · Historical research · Uniswap V3 / The Graph · UTC")
