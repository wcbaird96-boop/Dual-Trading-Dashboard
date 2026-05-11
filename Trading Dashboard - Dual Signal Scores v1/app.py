import streamlit as st
import yfinance as yf
import yfinance.cache as yf_cache
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

from backtest import backtest_signals_no_lookahead
from indicators import (
    calculate_adx,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_stochastic,
    calculate_vwap,
)
from levels import (
    build_level_snapshot,
    calculate_actionable_threshold_pct,
    calculate_atr,
    calculate_cluster_tolerance_pct,
    create_display_zone,
)
from signals import (
    generate_bullish_continuation_score,
    generate_intraday_trading_score,
    generate_sell_exhaustion_score,
    generate_semi_auto_trend_score,
    generate_signal,
)
from vcp import apply_vcp_to_signal, create_vcp_evidence_chart, detect_vcp


def configure_yfinance_cache() -> None:
    """Point yfinance caches to a writable directory inside the workspace."""
    cache_dir = Path(__file__).resolve().parent / ".yfinance-cache"
    cache_dir.mkdir(exist_ok=True)
    yf_cache.set_cache_location(str(cache_dir))
    yf.set_tz_cache_location(str(cache_dir))


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)
        if {"Open", "High", "Low", "Close"}.issubset(first_level):
            df.columns = first_level
        else:
            try:
                df = df.xs(df.columns.levels[1][0], axis=1, level=1)
            except Exception:
                df = df.copy()
    return df


def get_series(df: pd.DataFrame, key: str) -> pd.Series:
    """Return a clean series for an OHLC or volume column."""
    if key not in df.columns:
        raise KeyError(key)
    series = df[key]
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0]
    return series


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate a simple moving average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate an exponential moving average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_max_drawdown(close: pd.Series) -> float:
    """Calculate maximum drawdown as a percentage."""
    cumulative_max = close.cummax()
    drawdowns = close / cumulative_max - 1
    return float(drawdowns.min() * 100)


def calculate_annualized_volatility(close: pd.Series, ticker: str) -> float:
    """Calculate the latest 20-bar annualized volatility."""
    daily_vol = close.pct_change().rolling(window=20, min_periods=20).std()
    annual_factor = 365 if "-USD" in ticker.upper() else 252
    latest_daily_vol = daily_vol.iloc[-1] if not daily_vol.empty else np.nan
    if pd.isna(latest_daily_vol):
        return 0.0
    return float(latest_daily_vol * np.sqrt(annual_factor) * 100)


def format_money(value: float | None) -> str:
    """Format a price value for display."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_percent(value: float | None, digits: int = 2) -> str:
    """Format a percentage in decimal form."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def format_percent_points(value: float | None, digits: int = 2) -> str:
    """Format a percentage that is already expressed in percentage points."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}%"


def format_level_metric(level: float | None, distance_pct: float | None, distance_atr: float | None):
    """Prepare display strings for support and resistance metrics."""
    if level is None:
        return "N/A", "N/A"

    delta_parts = []
    if distance_pct is not None:
        delta_parts.append(f"{distance_pct * 100:.1f}%")
    if distance_atr is not None:
        delta_parts.append(f"{distance_atr:.2f} ATR")
    return format_money(level), " | ".join(delta_parts) if delta_parts else "N/A"


def build_level_table(levels: list[dict]) -> pd.DataFrame:
    """Convert level dictionaries into a display table."""
    rows = []
    for item in levels:
        rows.append({
            "Type": item["type"],
            "Level": format_money(item["level"]),
            "Source": item["source_label"],
            "Level Score": f"{item.get('level_score', 0.0):.2f}",
            "Touches": int(item.get("touches", item.get("hits", 0))),
            "Last Touch": str(item.get("most_recent_touch_date", "N/A")).split(" ")[0],
            "Distance %": format_percent_points(item.get("distance_pct")),
            "Distance ATR": f"{item['distance_atr']:.2f}" if item.get("distance_atr") is not None else "N/A",
            "Dynamic": "Yes" if item.get("dynamic_level") else "No",
        })
    return pd.DataFrame(rows)


def build_score_component_table(components: dict, max_points: dict | None = None) -> pd.DataFrame:
    """Convert score components into a compact table."""
    max_points = max_points or {}
    rows = []
    for label, value in components.items():
        display_label = label.replace("_", " ").title()
        max_value = max_points.get(label)
        rows.append({
            "Rule": display_label,
            "Points": value,
            "Max": max_value if max_value is not None else "",
        })
    return pd.DataFrame(rows)


def build_intraday_score_audit(score: dict) -> tuple[pd.DataFrame, bool]:
    """Show whether the reported intraday score matches its component totals."""
    daily_component_sum = int(sum(score["daily"]["components"].values()))
    entry_component_sum = int(sum(score["entry"]["components"].values()))
    reported_daily = int(score["daily"]["score"])
    reported_entry = int(score["entry"]["score"])
    reported_total = int(score["total_score"])
    expected_total = daily_component_sum + entry_component_sum

    rows = [
        {"Check": "Daily component sum", "Value": daily_component_sum, "Expected": reported_daily},
        {"Check": "Entry component sum", "Value": entry_component_sum, "Expected": reported_entry},
        {"Check": "Total score", "Value": reported_total, "Expected": expected_total},
    ]
    audit_passed = (
        daily_component_sum == reported_daily
        and entry_component_sum == reported_entry
        and reported_total == expected_total
    )
    return pd.DataFrame(rows), audit_passed


def build_chart_x_axis(index: pd.Index) -> tuple[list[str] | pd.Index, bool]:
    """Use a compact categorical axis for intraday data so non-trading gaps do not distort the chart."""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        median_gap = index.to_series().diff().dropna().median()
        if pd.notna(median_gap) and median_gap < pd.Timedelta(days=1):
            return [timestamp.strftime("%m-%d %H:%M") for timestamp in index], True
    return index, False


def break_intraday_session_lines(values, index: pd.Index):
    """Avoid drawing indicator lines across overnight or weekend session gaps."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) <= 1:
        return values

    median_gap = index.to_series().diff().dropna().median()
    if pd.isna(median_gap) or median_gap >= pd.Timedelta(days=1):
        return values

    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values, index=index)
    session = pd.Series(index.date, index=index)
    session_start = session.ne(session.shift())
    series.loc[session_start] = np.nan
    return series


def render_emphasis_card(title: str, value: str, accent: str) -> None:
    """Render a wider summary card for long text values."""
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {accent};
            background: rgba(15, 23, 42, 0.78);
            padding: 0.9rem 1rem;
            border-radius: 0.75rem;
            min-height: 92px;
        ">
            <div style="
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #cbd5e1;
                margin-bottom: 0.45rem;
            ">
                {title}
            </div>
            <div style="
                font-size: 1.08rem;
                font-weight: 700;
                line-height: 1.35;
                color: #f8fafc;
                white-space: normal;
                word-break: break-word;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_volume_profile_overlay(
    fig: go.Figure,
    close: pd.Series,
    volume: pd.Series,
    buckets: int = 400,
    max_width: float = 0.22,
) -> None:
    """Add a right-anchored horizontal volume profile to the price panel."""
    profile_df = pd.DataFrame({"close": close, "volume": volume}).dropna()
    profile_df = profile_df[profile_df["volume"] > 0]
    if profile_df.empty or profile_df["close"].nunique() < 2:
        return

    price_low = float(profile_df["close"].min())
    price_high = float(profile_df["close"].max())
    if price_low == price_high:
        return

    histogram, edges = np.histogram(
        profile_df["close"],
        bins=buckets,
        range=(price_low, price_high),
        weights=profile_df["volume"],
    )
    max_volume = float(np.nanmax(histogram)) if histogram.size else 0.0
    if max_volume <= 0:
        return

    for idx, bucket_volume in enumerate(histogram):
        if bucket_volume <= 0:
            continue
        width = max_width * float(bucket_volume / max_volume)
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=1.0 - width,
            x1=1.0,
            y0=float(edges[idx]),
            y1=float(edges[idx + 1]),
            fillcolor="rgba(56, 189, 248, 0.18)",
            line=dict(width=0),
            layer="below",
        )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=0.99,
        text="Volume Profile",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=10, color="#93c5fd"),
        bgcolor="rgba(15, 23, 42, 0.65)",
        borderpad=3,
    )


def create_price_chart(
    df: pd.DataFrame,
    ticker: str,
    sma_period: int | None = None,
    ema_period: int | None = None,
    show_ema9: bool = False,
    show_vwap: bool = False,
    show_bollinger_bands: bool = False,
    show_rsi: bool = False,
    show_macd: bool = False,
    show_stochastic: bool = False,
    show_obv: bool = False,
    show_adx: bool = False,
    show_volume_profile: bool = False,
    volume_profile_buckets: int = 400,
    actionable_levels=None,
    historical_levels=None,
):
    """Create a candlestick chart with optional technical indicator panels."""
    close_series = get_series(df, "Close")
    open_series = get_series(df, "Open")
    high_series = get_series(df, "High")
    low_series = get_series(df, "Low")
    volume_series = get_series(df, "Volume") if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    x_values, compact_intraday_axis = build_chart_x_axis(df.index)

    def line_y(values):
        return break_intraday_session_lines(values, df.index)

    indicator_panels = []
    if show_rsi:
        indicator_panels.append("RSI")
    if show_macd:
        indicator_panels.append("MACD")
    if show_stochastic:
        indicator_panels.append("Stochastic")
    if show_obv:
        indicator_panels.append("OBV")
    if show_adx:
        indicator_panels.append("ADX")

    total_rows = 2 + len(indicator_panels)
    row_heights = [0.54, 0.16] + [0.15] * len(indicator_panels)

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.06,
        specs=[[{"secondary_y": False}] for _ in range(total_rows)],
    )

    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="Price",
            increasing_line_color="lightgreen",
            decreasing_line_color="lightcoral",
        ),
        row=1,
        col=1,
    )

    if sma_period and sma_period > 0:
        sma = calculate_sma(close_series, sma_period)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=line_y(sma),
                mode="lines",
                name=f"SMA {sma_period}",
                line=dict(color="orange", width=2),
            ),
            row=1,
            col=1,
        )

    if ema_period and ema_period > 0:
        ema = calculate_ema(close_series, ema_period)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=line_y(ema),
                mode="lines",
                name=f"EMA {ema_period}",
                line=dict(color="royalblue", width=2),
            ),
            row=1,
            col=1,
        )

    if show_ema9:
        ema9 = calculate_ema(close_series, 9)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=line_y(ema9),
                mode="lines",
                name="EMA 9",
                line=dict(color="#38bdf8", width=1.8),
            ),
            row=1,
            col=1,
        )

    if show_vwap:
        vwap = calculate_vwap(high_series, low_series, close_series, volume_series)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=line_y(vwap),
                mode="lines",
                name="VWAP",
                line=dict(color="#facc15", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    if show_bollinger_bands:
        bollinger = calculate_bollinger_bands(close_series, period=20, standard_deviations=2.0)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=line_y(bollinger["bb_upper"]),
                mode="lines",
                name="BB Upper",
                line=dict(color="rgba(250, 204, 21, 0.8)", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=line_y(bollinger["bb_middle"]),
                mode="lines",
                name="BB Middle",
                line=dict(color="rgba(250, 204, 21, 0.55)", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=line_y(bollinger["bb_lower"]),
                mode="lines",
                name="BB Lower",
                line=dict(color="rgba(250, 204, 21, 0.8)", width=1),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=volume_series,
            name="Volume",
            marker_color="gray",
            opacity=0.4,
        ),
        row=2,
        col=1,
    )

    if show_volume_profile:
        add_volume_profile_overlay(
            fig,
            close=close_series,
            volume=volume_series,
            buckets=volume_profile_buckets,
        )

    panel_row = 3
    if show_rsi:
        rsi = calculate_rsi(close_series, period=14)
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(rsi), mode="lines", name="RSI 14", line=dict(color="#22c55e", width=2)),
            row=panel_row,
            col=1,
        )
        fig.add_hline(y=70, line_color="rgba(239, 68, 68, 0.7)", line_dash="dot", row=panel_row, col=1)
        fig.add_hline(y=30, line_color="rgba(34, 197, 94, 0.7)", line_dash="dot", row=panel_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=panel_row, col=1)
        panel_row += 1

    if show_macd:
        macd = calculate_macd(close_series)
        histogram_colors = np.where(macd["macd_histogram"] >= 0, "#22c55e", "#ef4444")
        fig.add_trace(
            go.Bar(x=x_values, y=macd["macd_histogram"], name="MACD Hist", marker_color=histogram_colors, opacity=0.55),
            row=panel_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(macd["macd"]), mode="lines", name="MACD", line=dict(color="#38bdf8", width=2)),
            row=panel_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(macd["macd_signal"]), mode="lines", name="MACD Signal", line=dict(color="#f97316", width=1.5)),
            row=panel_row,
            col=1,
        )
        fig.update_yaxes(title_text="MACD", row=panel_row, col=1)
        panel_row += 1

    if show_stochastic:
        stochastic = calculate_stochastic(high_series, low_series, close_series)
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(stochastic["stoch_k"]), mode="lines", name="Stoch %K", line=dict(color="#a78bfa", width=2)),
            row=panel_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(stochastic["stoch_d"]), mode="lines", name="Stoch %D", line=dict(color="#f59e0b", width=1.5)),
            row=panel_row,
            col=1,
        )
        fig.add_hline(y=80, line_color="rgba(239, 68, 68, 0.7)", line_dash="dot", row=panel_row, col=1)
        fig.add_hline(y=20, line_color="rgba(34, 197, 94, 0.7)", line_dash="dot", row=panel_row, col=1)
        fig.update_yaxes(title_text="Stoch", range=[0, 100], row=panel_row, col=1)
        panel_row += 1

    if show_obv:
        obv = calculate_obv(close_series, volume_series)
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(obv), mode="lines", name="OBV", line=dict(color="#14b8a6", width=2)),
            row=panel_row,
            col=1,
        )
        fig.update_yaxes(title_text="OBV", row=panel_row, col=1)
        panel_row += 1

    if show_adx:
        adx = calculate_adx(high_series, low_series, close_series)
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(adx["adx14"]), mode="lines", name="ADX 14", line=dict(color="#e879f9", width=2)),
            row=panel_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(adx["plus_di14"]), mode="lines", name="+DI 14", line=dict(color="#22c55e", width=1.5)),
            row=panel_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_values, y=line_y(adx["minus_di14"]), mode="lines", name="-DI 14", line=dict(color="#ef4444", width=1.5)),
            row=panel_row,
            col=1,
        )
        fig.add_hline(y=20, line_color="rgba(148, 163, 184, 0.55)", line_dash="dot", row=panel_row, col=1)
        fig.add_hline(y=25, line_color="rgba(250, 204, 21, 0.65)", line_dash="dot", row=panel_row, col=1)
        fig.update_yaxes(title_text="ADX/DI", row=panel_row, col=1)
        panel_row += 1

    for level in actionable_levels or []:
        zone = create_display_zone(level, atr=0.0)
        fill_color = "rgba(34, 197, 94, 0.18)" if level["type"] == "Support" else "rgba(239, 68, 68, 0.16)"
        line_color = "green" if level["type"] == "Support" else "red"
        fig.add_hrect(
            y0=zone["lower"],
            y1=zone["upper"],
            line_width=0,
            fillcolor=fill_color,
            row=1,
            col=1,
        )
        fig.add_hline(
            y=zone["level"],
            line_dash="solid",
            line_color=line_color,
            line_width=2,
            annotation_text=f"{level['type']} {zone['level']:.2f}",
            row=1,
            col=1,
        )

    for level in historical_levels or []:
        zone = create_display_zone(level, atr=0.0)
        line_color = "rgba(34, 197, 94, 0.45)" if level["type"] == "Support" else "rgba(239, 68, 68, 0.45)"
        fig.add_hline(
            y=zone["level"],
            line_dash="dot",
            line_color=line_color,
            line_width=1,
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        template="plotly_dark",
        height=760 + 140 * len(indicator_panels),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=30, l=40, r=40),
    )

    if compact_intraday_axis:
        tick_step = max(1, len(x_values) // 8)
        tick_values = x_values[::tick_step]
        fig.update_xaxes(type="category", rangeslider_visible=False, tickmode="array", tickvals=tick_values, ticktext=tick_values)
    else:
        fig.update_xaxes(type="date", rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def load_data(ticker: str, start_date, end_date, interval: str) -> pd.DataFrame:
    """Download ticker data and return a normalized DataFrame."""
    configure_yfinance_cache()
    raw_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if raw_df.empty:
        return raw_df

    df = normalize_data(raw_df)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def trim_to_recent_trading_sessions(df: pd.DataFrame, sessions: int = 5) -> pd.DataFrame:
    """Keep the most recent distinct trading sessions in an intraday DataFrame."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    session_dates = pd.Series(df.index.date, index=df.index)
    recent_sessions = session_dates.drop_duplicates().tail(sessions).tolist()
    return df[session_dates.isin(recent_sessions)]


def load_intraday_trading_sessions(ticker: str, end_date, interval: str, sessions: int = 5) -> pd.DataFrame:
    """Load enough intraday data to display the requested number of trading sessions."""
    lookback_days = 7 if interval == "1m" else 10
    start_date = end_date - timedelta(days=lookback_days)
    df = load_data(ticker, start_date, end_date, interval)
    return trim_to_recent_trading_sessions(df, sessions=sessions)


def parse_ticker_list(raw_tickers: str) -> list[str]:
    """Parse comma, newline, or space separated tickers."""
    cleaned = raw_tickers.replace(",", " ").replace("\n", " ").replace("\t", " ")
    tickers = []
    seen = set()
    for item in cleaned.split(" "):
        ticker = item.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def normalize_sparkline(close: pd.Series, points: int = 60) -> list[float]:
    """Return a compact normalized trend line for Streamlit line chart cells."""
    clean = close.dropna()
    if clean.empty:
        return []
    sampled = clean.iloc[np.linspace(0, len(clean) - 1, min(points, len(clean))).astype(int)]
    first = float(sampled.iloc[0])
    if first == 0:
        return [float(value) for value in sampled]
    return [float((value / first - 1) * 100) for value in sampled]


def build_watchlist_row(
    ticker: str,
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None,
    benchmark_df: pd.DataFrame | None,
    benchmark_ticker: str,
) -> dict:
    """Build one sortable watchlist row from daily and intraday data."""
    latest_close = float(daily_df["Close"].iloc[-1])
    intraday_score = generate_intraday_trading_score(
        daily_df=daily_df,
        intraday_df=intraday_df if intraday_df is not None and not intraday_df.empty else None,
        benchmark_df=benchmark_df,
        ticker=ticker,
        benchmark_ticker=benchmark_ticker,
    )
    reliability_sell_score = generate_sell_exhaustion_score(
        daily_df=daily_df,
        intraday_df=intraday_df if intraday_df is not None and not intraday_df.empty else None,
        ticker=ticker,
    )
    return_score = generate_semi_auto_trend_score(
        daily_df=daily_df,
        intraday_df=intraday_df if intraday_df is not None and not intraday_df.empty else None,
        benchmark_df=benchmark_df,
        ticker=ticker,
        benchmark_ticker=benchmark_ticker,
    )

    six_month_return = 0.0
    first_close = float(daily_df["Close"].iloc[0])
    if first_close:
        six_month_return = (latest_close / first_close - 1) * 100

    rs_3m = intraday_score["relative_strength"].get("3m", {}).get("relative_return")
    rs_6m = intraday_score["relative_strength"].get("6m", {}).get("relative_return")
    daily_features = intraday_score["daily"]["features"]
    entry_features = intraday_score["entry"]["features"]
    distance_from_20ema = None
    if daily_features.get("ema20") not in {None, 0}:
        distance_from_20ema = (latest_close / daily_features["ema20"] - 1) * 100

    return {
        "Trend": normalize_sparkline(daily_df["Close"]),
        "Ticker": ticker,
        "Last": latest_close,
        "Intraday Buy": int(intraday_score["total_score"]),
        "Intraday Sell": int(reliability_sell_score["total_score"]),
        "Intraday Action": intraday_score["action"],
        "HR Buy": int(intraday_score["total_score"]),
        "HR Daily": int(intraday_score["daily"]["score"]),
        "HR Entry": int(intraday_score["entry"]["score"]),
        "HR Sell": int(reliability_sell_score["total_score"]),
        "HR Signal": "YES" if intraday_score["intraday_long_signal"] else "NO",
        "HR Action": intraday_score["action"],
        "Return Buy": int(return_score["total_score"]),
        "Return Daily": int(return_score["daily"]["score"]),
        "Return Entry": int(return_score["entry"]["score"]),
        "Return Sell": int(return_score["exit"]["score"]),
        "Return Signal": "YES" if return_score["semi_auto_long_signal"] else "NO",
        "Return Action": return_score["action"],
        "RS 3M %": None if rs_3m is None else rs_3m * 100,
        "RS 6M %": None if rs_6m is None else rs_6m * 100,
        "Dist 20EMA %": distance_from_20ema,
        "Intraday RSI": entry_features.get("rsi14"),
        "Volume Ratio": entry_features.get("volume_ratio_20"),
        "ATR %": None if entry_features.get("atr_pct") is None else entry_features["atr_pct"] * 100,
        "6M Return %": six_month_return,
    }


def build_watchlist_table(
    tickers: list[str],
    intraday_interval: str,
    max_symbols: int,
    benchmark_ticker: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Scan tickers and return a sortable watchlist table."""
    rows = []
    skipped = []
    end = datetime.now() + timedelta(days=1)
    daily_start = datetime.now() - timedelta(days=430)
    benchmark_df = load_data(benchmark_ticker, daily_start, end, "1d")

    progress = st.progress(0, text="Preparing watchlist scan...")
    selected_tickers = tickers[:int(max_symbols)]

    for idx, ticker in enumerate(selected_tickers):
        progress.progress((idx + 1) / max(1, len(selected_tickers)), text=f"Scanning {ticker}...")
        try:
            daily_df = load_data(ticker, daily_start, end, "1d")
            if daily_df.empty or len(daily_df) < 200:
                skipped.append(f"{ticker}: not enough daily data")
                continue
            intraday_df = load_intraday_trading_sessions(ticker, end, intraday_interval, sessions=5)
            rows.append(build_watchlist_row(ticker, daily_df, intraday_df, benchmark_df, benchmark_ticker))
        except Exception as exc:
            skipped.append(f"{ticker}: {exc}")

    progress.empty()
    if not rows:
        return pd.DataFrame(), skipped

    table = pd.DataFrame(rows).sort_values(["Intraday Buy", "Return Buy", "Intraday Sell"], ascending=[False, False, True]).reset_index(drop=True)
    return table, skipped


def create_six_month_trend_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create a compact 6-month trend chart with EMA context."""
    close = get_series(df, "Close")
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name="Close", line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=ema20, mode="lines", name="EMA 20", line=dict(color="#facc15", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=ema50, mode="lines", name="EMA 50", line=dict(color="#a78bfa", width=1.5)))
    fig.update_layout(
        title=f"{ticker} 6-Month Trend",
        template="plotly_dark",
        height=260,
        margin=dict(t=45, b=20, l=35, r=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def render_watchlist_dashboard() -> None:
    """Render the sortable multi-stock dashboard and detail drill-in."""
    st.title("Trading Dashboard: Dual Signal Scanner")

    default_tickers = "AMD, MSFT, NVDA, AAPL, TSLA, META, GOOGL, AMZN, SPY, QQQ"
    raw_tickers = st.sidebar.text_area("Watchlist tickers", value=default_tickers, height=180)
    benchmark_ticker = st.sidebar.text_input("Relative strength benchmark", value="SPY").upper()
    intraday_interval = st.sidebar.selectbox("Detail / entry interval", options=["5m", "1m"], index=0)
    max_symbols = st.sidebar.number_input("Max symbols to scan", min_value=1, value=40, step=10)
    show_detail_volume_profile = st.sidebar.toggle("Show detail volume profile", value=False)
    run_scan = st.sidebar.button("Scan Watchlist", type="primary")

    tickers = parse_ticker_list(raw_tickers)
    if not tickers:
        st.info("Add ticker symbols in the sidebar to scan a watchlist.")
        return

    if run_scan or "watchlist_table" not in st.session_state:
        table, skipped = build_watchlist_table(
            tickers,
            intraday_interval=intraday_interval,
            max_symbols=max_symbols,
            benchmark_ticker=benchmark_ticker,
        )
        st.session_state["watchlist_table"] = table
        st.session_state["watchlist_skipped"] = skipped
        st.session_state["watchlist_interval"] = intraday_interval
        st.session_state["benchmark_ticker"] = benchmark_ticker

    table = st.session_state.get("watchlist_table", pd.DataFrame())
    skipped = st.session_state.get("watchlist_skipped", [])
    active_interval = st.session_state.get("watchlist_interval", intraday_interval)

    if table.empty:
        st.warning("No watchlist rows were built. Check ticker symbols or try fewer symbols.")
        if skipped:
            with st.expander("Skipped symbols"):
                for item in skipped:
                    st.write(f"- {item}")
        return

    st.caption(
        "Sort the table by either score family. HR Buy/Sell is the higher-reliability intraday setup and exhaustion model; "
        "Return Buy/Sell is the semi-auto trend/momentum model."
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Trend": st.column_config.LineChartColumn("6M Trend"),
            "Last": st.column_config.NumberColumn("Last", format="$%.2f"),
            "Intraday Buy": st.column_config.NumberColumn("Intraday Buy", format="%d"),
            "Intraday Sell": st.column_config.NumberColumn("Intraday Sell", format="%d"),
            "HR Buy": st.column_config.NumberColumn("HR Buy", format="%d"),
            "HR Daily": st.column_config.NumberColumn("HR Daily", format="%d"),
            "HR Entry": st.column_config.NumberColumn("HR Entry", format="%d"),
            "HR Sell": st.column_config.NumberColumn("HR Sell", format="%d"),
            "Return Buy": st.column_config.NumberColumn("Return Buy", format="%d"),
            "Return Daily": st.column_config.NumberColumn("Return Daily", format="%d"),
            "Return Entry": st.column_config.NumberColumn("Return Entry", format="%d"),
            "Return Sell": st.column_config.NumberColumn("Return Sell", format="%d"),
            "RS 3M %": st.column_config.NumberColumn("RS 3M", format="%.1f%%"),
            "RS 6M %": st.column_config.NumberColumn("RS 6M", format="%.1f%%"),
            "Dist 20EMA %": st.column_config.NumberColumn("Dist 20EMA", format="%.1f%%"),
            "Intraday RSI": st.column_config.NumberColumn("Intraday RSI", format="%.1f"),
            "Volume Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2fx"),
            "ATR %": st.column_config.NumberColumn("ATR", format="%.1f%%"),
            "6M Return %": st.column_config.NumberColumn("6M Return", format="%.1f%%"),
        },
    )

    if skipped:
        with st.expander("Skipped symbols"):
            for item in skipped:
                st.write(f"- {item}")

    st.divider()
    st.caption("Click a ticker button or use the dropdown to open the 5-trading-day candle detail view.")
    button_cols = st.columns(8)
    for idx, row in table.head(32).iterrows():
        with button_cols[idx % len(button_cols)]:
            if st.button(str(row["Ticker"]), key=f"open_{row['Ticker']}"):
                st.session_state["selected_watchlist_ticker"] = row["Ticker"]

    ticker_options = table["Ticker"].tolist()
    default_ticker = st.session_state.get("selected_watchlist_ticker", ticker_options[0])
    default_index = ticker_options.index(default_ticker) if default_ticker in ticker_options else 0
    selected_ticker = st.selectbox("Open ticker detail", options=ticker_options, index=default_index)
    st.session_state["selected_watchlist_ticker"] = selected_ticker
    detail_interval = st.radio("Candle interval", options=["5m", "1m"], index=0 if active_interval == "5m" else 1, horizontal=True)

    end = datetime.now() + timedelta(days=1)
    detail_benchmark = st.session_state.get("benchmark_ticker", benchmark_ticker)
    daily_df = load_data(selected_ticker, datetime.now() - timedelta(days=430), end, "1d")
    benchmark_df = load_data(detail_benchmark, datetime.now() - timedelta(days=430), end, "1d")
    detail_df = load_intraday_trading_sessions(selected_ticker, end, detail_interval, sessions=5)

    if daily_df.empty or detail_df.empty:
        st.warning(f"Could not load detail charts for {selected_ticker}.")
        return

    detail_score = generate_intraday_trading_score(
        daily_df=daily_df,
        intraday_df=detail_df,
        benchmark_df=benchmark_df if not benchmark_df.empty else None,
        ticker=selected_ticker,
        benchmark_ticker=detail_benchmark,
    )
    detail_reliability_sell = generate_sell_exhaustion_score(
        daily_df=daily_df,
        intraday_df=detail_df,
        ticker=selected_ticker,
    )
    detail_return_score = generate_semi_auto_trend_score(
        daily_df=daily_df,
        intraday_df=detail_df,
        benchmark_df=benchmark_df if not benchmark_df.empty else None,
        ticker=selected_ticker,
        benchmark_ticker=detail_benchmark,
    )

    score_cols = st.columns(7)
    score_cols[0].metric("Intraday Buy", f"{detail_score['total_score']}/20", detail_score["score_band"])
    score_cols[1].metric("Intraday Sell", f"{detail_reliability_sell['total_score']}/20", detail_reliability_sell["score_band"])
    score_cols[2].metric("HR Buy", f"{detail_score['total_score']}/20", detail_score["score_band"])
    score_cols[3].metric("HR Sell", f"{detail_reliability_sell['total_score']}/20", detail_reliability_sell["score_band"])
    score_cols[4].metric("Return Buy", f"{detail_return_score['total_score']}/20", detail_return_score["action"])
    score_cols[5].metric("Return Sell", f"{detail_return_score['exit']['score']}/{detail_return_score['exit']['max_score']}", detail_return_score["exit"]["interpretation"])
    score_cols[6].metric("Return Signal", "Yes" if detail_return_score["semi_auto_long_signal"] else "No")

    rule_cols = st.columns(4)
    rule_cols[0].metric("Daily >= 3", "Yes" if detail_score["rules"]["daily_score_at_least_3"] else "No")
    rule_cols[1].metric("Entry >= 9", "Yes" if detail_score["rules"]["entry_score_at_least_9"] else "No")
    rule_cols[2].metric("VWAP Confirmed", "Yes" if detail_score["rules"]["vwap_confirmed"] else "No")
    rule_cols[3].metric("Not ATR Extended", "Yes" if detail_score["rules"]["not_extended_by_atr"] else "No")

    with st.expander("Higher Reliability Buy Score Breakdown", expanded=False):
        st.markdown("Daily / market bias")
        st.table(build_score_component_table(
            detail_score["daily"]["components"],
            {
                "daily_trend": 2,
                "relative_strength": 1,
                "daily_momentum": 1,
                "trend_quality": 1,
                "not_extended": 1,
            },
        ))
        st.markdown("Intraday execution")
        st.table(build_score_component_table(
            detail_score["entry"]["components"],
            {
                "vwap": 3,
                "ema_structure": 2,
                "pullback_quality": 2,
                "rsi": 2,
                "macd": 2,
                "volume": 2,
                "atr_location": 1,
            },
        ))
        st.markdown("Score audit")
        audit_table, audit_passed = build_intraday_score_audit(detail_score)
        st.table(audit_table)
        if audit_passed:
            st.success("Score audit passed: component totals match the displayed total score.")
        else:
            st.error("Score audit failed: displayed score does not match the component totals.")

    with st.expander("Higher Reliability Sell / Exhaustion Breakdown", expanded=False):
        st.markdown("Daily exhaustion")
        st.table(build_score_component_table(
            detail_reliability_sell["daily"]["components"],
            {"adx_aging": 2, "di_weakening": 2, "rsi_exhaustion": 2, "macd_weakening": 2, "price_vs_ema20": 2},
        ))
        st.markdown("Intraday exit trigger")
        st.table(build_score_component_table(
            detail_reliability_sell["intraday"]["components"],
            {"vwap_loss": 2, "ema_structure": 2, "rsi_exit": 2, "macd_exit": 2, "distribution_volume": 2},
        ))
        st.write(f"ATR modifier: +{detail_reliability_sell['atr_modifier']['modifier']}")
        for check in detail_reliability_sell["atr_modifier"]["checks"]:
            st.write(f"- {check}")

    with st.expander("Higher Return Score Breakdown", expanded=False):
        st.markdown("Daily systematic score")
        st.table(build_score_component_table(
            detail_return_score["daily"]["components"],
            {
                "long_term_trend": 2,
                "intermediate_trend": 2,
                "relative_strength": 2,
                "momentum_persistence": 1,
                "adx_di_filter": 1,
                "volume_trend": 1,
                "atr_tradability": 1,
            },
        ))
        st.markdown("Entry trigger")
        st.table(build_score_component_table(
            detail_return_score["entry"]["components"],
            {
                "pullback_entry": 2,
                "reversal_confirmation": 1,
                "price_reclaim": 2,
                "vwap_confirmation": 2,
                "macd_confirmation": 1,
                "volume_confirmation": 1,
                "atr_entry_quality": 1,
            },
        ))
        st.markdown("Trend-following exit")
        st.table(build_score_component_table(detail_return_score["exit"]["components"]))

    st.plotly_chart(create_six_month_trend_chart(daily_df, selected_ticker), use_container_width=True)
    st.plotly_chart(
        create_price_chart(
            detail_df,
            selected_ticker,
            ema_period=20,
            show_ema9=True,
            show_vwap=True,
            show_rsi=True,
            show_macd=True,
            show_adx=False,
            show_volume_profile=show_detail_volume_profile,
            volume_profile_buckets=400,
        ),
        use_container_width=True,
    )


def main():
    st.set_page_config(page_title="Trading Dashboard - Dual Signal Scores", layout="wide")
    app_mode = st.sidebar.radio(
        "App Mode",
        options=["Watchlist Dashboard", "Single Stock Analysis"],
        index=0,
    )

    if app_mode == "Watchlist Dashboard":
        render_watchlist_dashboard()
        return

    st.title("Trading Dashboard: Dual Signal Single Stock Analysis")
    st.markdown("Single-stock charting uses a fixed 10-day, 5-minute candle window for intraday review.")

    st.sidebar.header("Market Selection")
    ticker = st.sidebar.text_input(
        "Ticker symbol",
        value="AMD",
        help="Examples: AMD, MSFT, SPY, BTC-USD, SNDK",
    ).upper()
    chart_end_date = datetime.now()
    chart_start_date = chart_end_date - timedelta(days=10)
    interval = "5m"
    st.sidebar.caption("Single-stock chart fixed to 10 days of 5-minute candles.")

    st.sidebar.divider()
    st.sidebar.header("Chart Overlays")
    sma_enabled = st.sidebar.checkbox("Show SMA", value=True)
    sma_period = st.sidebar.slider("SMA period", min_value=5, max_value=200, value=50, disabled=not sma_enabled) if sma_enabled else 50
    ema_enabled = st.sidebar.checkbox("Show EMA", value=True)
    ema_period = st.sidebar.slider("EMA period", min_value=5, max_value=200, value=20, disabled=not ema_enabled) if ema_enabled else 20
    show_ema9 = st.sidebar.checkbox("Show EMA 9", value=True)
    show_vwap = st.sidebar.checkbox("Show VWAP", value=True)
    show_bollinger_bands = st.sidebar.checkbox("Show Bollinger Bands", value=False)
    show_rsi = st.sidebar.checkbox("Show RSI panel", value=True)
    show_macd = st.sidebar.checkbox("Show MACD panel", value=True)
    show_stochastic = st.sidebar.checkbox("Show Stochastic panel", value=False)
    show_obv = st.sidebar.checkbox("Show OBV panel", value=False)
    show_adx = st.sidebar.checkbox("Show ADX / DI panel", value=True)
    show_volume_profile = st.sidebar.toggle("Show volume profile", value=False)

    st.sidebar.divider()
    st.sidebar.header("Support / Resistance")
    show_zones = st.sidebar.checkbox("Show actionable zones", value=True)
    show_historical_levels = st.sidebar.checkbox("Show Historical Levels", value=False)
    zone_count = st.sidebar.slider("Max actionable levels", min_value=1, max_value=8, value=4)
    swing_sensitivity = st.sidebar.slider("Swing sensitivity (N)", min_value=2, max_value=15, value=5)
    include_dynamic_levels = st.sidebar.checkbox("Include dynamic EMA20 / SMA50 levels", value=True)

    st.sidebar.divider()
    st.sidebar.header("Signal Engine")
    use_transparent_signals = st.sidebar.checkbox("Use regime-based rule engine", value=True)
    show_backtest = st.sidebar.checkbox("Show historical backtest", value=True)
    show_intraday_trading_score = st.sidebar.checkbox("Show dual signal scores", value=True)
    show_bullish_continuation_score = st.sidebar.checkbox("Show legacy bullish checklist", value=False)
    show_sell_exhaustion_score = st.sidebar.checkbox("Show legacy sell / exhaustion checklist", value=False)
    checklist_intraday_interval = st.sidebar.selectbox(
        "Checklist intraday interval",
        options=["5m", "1m", "15m"],
        index=0,
        help="Used only for the Stage 2 entry score.",
    )

    st.sidebar.divider()
    st.sidebar.header("VCP Detector")
    enable_vcp_detection = st.sidebar.toggle("Enable VCP Detection", value=False)
    show_vcp_annotations = st.sidebar.checkbox("Show VCP Annotations", value=True)
    show_vcp_diagnostics = st.sidebar.checkbox("Show VCP diagnostics even when no VCP is detected", value=False)
    vcp_min_base_length = st.sidebar.slider("VCP base min length", min_value=20, max_value=80, value=20)
    vcp_max_base_length = st.sidebar.slider("VCP base max length", min_value=40, max_value=180, value=120)

    try:
        with st.spinner(f"Loading data for {ticker}..."):
            df = load_data(ticker, chart_start_date, chart_end_date + timedelta(days=1), interval)
            analysis_df = load_data(ticker, datetime.now() - timedelta(days=430), chart_end_date + timedelta(days=1), "1d")

        if df.empty:
            st.error(f"No 5-minute data found for {ticker}. Please verify the ticker or try during market hours.")
            return

        if analysis_df.empty:
            st.error(f"No daily data found for {ticker}. Please verify the ticker.")
            return

        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            st.error("Downloaded 5-minute data is missing required OHLC columns.")
            return

        if not {"Open", "High", "Low", "Close"}.issubset(analysis_df.columns):
            st.error("Downloaded daily data is missing required OHLC columns.")
            return

        close_series = get_series(df, "Close")
        volume_series = get_series(df, "Volume") if "Volume" in df.columns else None
        analysis_close_series = get_series(analysis_df, "Close")
        latest_close = float(close_series.iloc[-1])
        start_close = float(close_series.iloc[0])
        end_close = float(close_series.iloc[-1])
        bars_used = int(len(close_series))
        period_return = ((end_close / start_close) - 1) * 100 if start_close else 0.0
        volatility = calculate_annualized_volatility(analysis_close_series, ticker)
        max_drawdown = calculate_max_drawdown(close_series)
        average_volume = float(volume_series.rolling(window=20, min_periods=1).mean().iloc[-1]) if volume_series is not None else 0.0

        atr = calculate_atr(analysis_df, period=14)
        cluster_tolerance_pct = calculate_cluster_tolerance_pct(atr, latest_close)
        actionable_threshold_pct = calculate_actionable_threshold_pct(atr, latest_close)

        level_snapshot = build_level_snapshot(
            analysis_df,
            current_price=latest_close,
            atr=atr,
            max_levels=zone_count,
            show_all_historical=show_historical_levels,
            sensitivity=swing_sensitivity,
            include_dynamic_levels=include_dynamic_levels,
        )
        support_levels = level_snapshot["actionable_supports"]
        resistance_levels = level_snapshot["actionable_resistances"]
        historical_supports = level_snapshot["historical_supports"]
        historical_resistances = level_snapshot["historical_resistances"]
        historical_levels = historical_supports + historical_resistances if show_historical_levels else []

        signal_result = generate_signal(
            df=analysis_df,
            ticker=ticker,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            max_levels=zone_count,
            swing_sensitivity=swing_sensitivity,
        )

        vcp_result = None
        if enable_vcp_detection:
            vcp_result = detect_vcp(
                df=analysis_df,
                ticker=ticker,
                swing_sensitivity=swing_sensitivity,
                min_base_length=vcp_min_base_length,
                max_base_length=vcp_max_base_length,
                actionable_resistances=resistance_levels,
            )
            signal_result = apply_vcp_to_signal(signal_result, vcp_result)

        bullish_continuation_score = None
        sell_exhaustion_score = None
        intraday_trading_score = None
        dual_reliability_sell_score = None
        dual_return_score = None
        if show_intraday_trading_score:
            with st.spinner("Loading daily benchmark context for intraday score..."):
                score_end = datetime.now() + timedelta(days=1)
                score_benchmark_df = load_data("SPY", datetime.now() - timedelta(days=430), score_end, "1d")
            if not analysis_df.empty:
                intraday_trading_score = generate_intraday_trading_score(
                    daily_df=analysis_df,
                    intraday_df=df,
                    benchmark_df=score_benchmark_df if not score_benchmark_df.empty else None,
                    ticker=ticker,
                    benchmark_ticker="SPY",
                )
                dual_reliability_sell_score = generate_sell_exhaustion_score(
                    daily_df=analysis_df,
                    intraday_df=df,
                    ticker=ticker,
                )
                dual_return_score = generate_semi_auto_trend_score(
                    daily_df=analysis_df,
                    intraday_df=df,
                    benchmark_df=score_benchmark_df if not score_benchmark_df.empty else None,
                    ticker=ticker,
                    benchmark_ticker="SPY",
                )

        if show_bullish_continuation_score or show_sell_exhaustion_score:
            with st.spinner("Loading daily and intraday data for checklist scoring..."):
                checklist_end = datetime.now() + timedelta(days=1)
                daily_bias_df = load_data(ticker, datetime.now() - timedelta(days=183), checklist_end, "1d")
                intraday_start = datetime.now() - timedelta(days=2 if checklist_intraday_interval == "1m" else 5)
                intraday_df = load_data(ticker, intraday_start, checklist_end, checklist_intraday_interval)

            nearest_resistance_for_checklist = signal_result["nearest_levels"].get("resistance")
            if show_bullish_continuation_score and not daily_bias_df.empty:
                bullish_continuation_score = generate_bullish_continuation_score(
                    daily_df=daily_bias_df,
                    intraday_df=intraday_df if not intraday_df.empty else None,
                    nearest_resistance=nearest_resistance_for_checklist,
                    ticker=ticker,
                )
            if show_sell_exhaustion_score and not daily_bias_df.empty:
                sell_exhaustion_score = generate_sell_exhaustion_score(
                    daily_df=daily_bias_df,
                    intraday_df=intraday_df if not intraday_df.empty else None,
                    ticker=ticker,
                )

        volatility_percentile = signal_result["features"]["annual_vol_percentile"]
        volatility_threshold_percentile = signal_result["features"]["volatility_threshold_percentile"]

        if interval != "1d":
            st.info("Current and visual analysis uses the selected visible data range. Daily data is preferred for the signal engine, VCP detector, and historical tests.")

        if signal_result["high_volatility"]:
            st.warning(
                f"High-volatility regime active. Current 20-day annualized volatility is {volatility:.2f}% "
                f"at the {volatility_percentile * 100:.0f}th percentile for the selected period."
            )

        if level_snapshot["price_discovery_mode"]:
            st.info("No overhead actionable resistance detected - price discovery mode.")

        if level_snapshot["price_extended"]:
            st.warning("Price may be extended above short-term trend.")

        if use_transparent_signals:
            st.divider()
            st.subheader("Signal Summary")

            headline_cols = st.columns([2.2, 2.0, 2.0, 1.2, 1.2, 1.2])
            with headline_cols[0]:
                render_emphasis_card("Regime", signal_result["regime_label"], "#2563eb")
            with headline_cols[1]:
                render_emphasis_card("Current Bias", signal_result["current_bias"], "#059669")
            with headline_cols[2]:
                render_emphasis_card("Signal Type", signal_result["signal_style"], "#7c3aed")
            headline_cols[3].metric("Trade Trigger", signal_result["trade_signal"])
            headline_cols[4].metric("Legacy Score", f"{signal_result['score']:+.2f}")
            headline_cols[5].metric("Setup Type", signal_result.get("setup_type", "Standard"))

            st.caption(signal_result["signal_strength_note"])

            stat_cols = st.columns(4)
            stat_cols[0].metric("Signal Strength", f"{signal_result['signal_strength']:.1f}%")
            stat_cols[1].metric("Annual Volatility", f"{volatility:.2f}%")
            stat_cols[2].metric("Volatility Percentile", f"{volatility_percentile * 100:.0f}%" if volatility_percentile is not None else "N/A")
            stat_cols[3].metric("High-Vol Threshold", f"{volatility_threshold_percentile * 100:.0f}%")

            if intraday_trading_score is not None and dual_reliability_sell_score is not None and dual_return_score is not None:
                st.markdown("#### Dual Signal Scores")
                dual_cols = st.columns(7)
                dual_cols[0].metric("Intraday Buy", f"{intraday_trading_score['total_score']}/20", intraday_trading_score["score_band"])
                dual_cols[1].metric("Intraday Sell", f"{dual_reliability_sell_score['total_score']}/20", dual_reliability_sell_score["score_band"])
                dual_cols[2].metric("HR Buy", f"{intraday_trading_score['total_score']}/20", intraday_trading_score["score_band"])
                dual_cols[3].metric("HR Sell", f"{dual_reliability_sell_score['total_score']}/20", dual_reliability_sell_score["score_band"])
                dual_cols[4].metric("Return Buy", f"{dual_return_score['total_score']}/20", dual_return_score["action"])
                dual_cols[5].metric("Return Sell", f"{dual_return_score['exit']['score']}/{dual_return_score['exit']['max_score']}", dual_return_score["exit"]["interpretation"])
                dual_cols[6].metric("Return Signal", "Yes" if dual_return_score["semi_auto_long_signal"] else "No")

                gate_cols = st.columns(4)
                gate_cols[0].metric("Daily >= 3", "Yes" if intraday_trading_score["rules"]["daily_score_at_least_3"] else "No")
                gate_cols[1].metric("Entry >= 9", "Yes" if intraday_trading_score["rules"]["entry_score_at_least_9"] else "No")
                gate_cols[2].metric("VWAP Confirmed", "Yes" if intraday_trading_score["rules"]["vwap_confirmed"] else "No")
                gate_cols[3].metric("Not ATR Extended", "Yes" if intraday_trading_score["rules"]["not_extended_by_atr"] else "No")

                with st.expander("Higher Reliability Buy Score Breakdown", expanded=False):
                    st.markdown("Daily / market bias")
                    st.table(build_score_component_table(
                        intraday_trading_score["daily"]["components"],
                        {"daily_trend": 2, "relative_strength": 1, "daily_momentum": 1, "trend_quality": 1, "not_extended": 1},
                    ))
                    st.markdown("Intraday execution")
                    st.table(build_score_component_table(
                        intraday_trading_score["entry"]["components"],
                        {"vwap": 3, "ema_structure": 2, "pullback_quality": 2, "rsi": 2, "macd": 2, "volume": 2, "atr_location": 1},
                    ))
                    st.markdown("Score audit")
                    audit_table, audit_passed = build_intraday_score_audit(intraday_trading_score)
                    st.table(audit_table)
                    if audit_passed:
                        st.success("Score audit passed: component totals match the displayed total score.")
                    else:
                        st.error("Score audit failed: displayed score does not match the component totals.")
                    st.markdown("Daily rationale")
                    for explanation in intraday_trading_score["daily"]["explanations"]:
                        st.write(f"- {explanation}")
                    st.markdown("Intraday rationale")
                    for explanation in intraday_trading_score["entry"]["explanations"]:
                        st.write(f"- {explanation}")

                with st.expander("Higher Reliability Sell / Exhaustion Breakdown", expanded=False):
                    st.markdown("Daily exhaustion")
                    st.table(build_score_component_table(
                        dual_reliability_sell_score["daily"]["components"],
                        {"adx_aging": 2, "di_weakening": 2, "rsi_exhaustion": 2, "macd_weakening": 2, "price_vs_ema20": 2},
                    ))
                    st.markdown("Intraday exit trigger")
                    st.table(build_score_component_table(
                        dual_reliability_sell_score["intraday"]["components"],
                        {"vwap_loss": 2, "ema_structure": 2, "rsi_exit": 2, "macd_exit": 2, "distribution_volume": 2},
                    ))
                    st.write(f"ATR modifier: +{dual_reliability_sell_score['atr_modifier']['modifier']}")
                    for check in dual_reliability_sell_score["atr_modifier"]["checks"]:
                        st.write(f"- {check}")

                with st.expander("Higher Return Score Breakdown", expanded=False):
                    st.markdown("Daily systematic score")
                    st.table(build_score_component_table(
                        dual_return_score["daily"]["components"],
                        {
                            "long_term_trend": 2,
                            "intermediate_trend": 2,
                            "relative_strength": 2,
                            "momentum_persistence": 1,
                            "adx_di_filter": 1,
                            "volume_trend": 1,
                            "atr_tradability": 1,
                        },
                    ))
                    st.markdown("Entry trigger")
                    st.table(build_score_component_table(
                        dual_return_score["entry"]["components"],
                        {
                            "pullback_entry": 2,
                            "reversal_confirmation": 1,
                            "price_reclaim": 2,
                            "vwap_confirmation": 2,
                            "macd_confirmation": 1,
                            "volume_confirmation": 1,
                            "atr_entry_quality": 1,
                        },
                    ))
                    st.markdown("Trend-following exit")
                    st.table(build_score_component_table(dual_return_score["exit"]["components"]))

            if bullish_continuation_score is not None or sell_exhaustion_score is not None:
                buy_score = bullish_continuation_score["total_score"] if bullish_continuation_score is not None else 0
                sell_score = sell_exhaustion_score["total_score"] if sell_exhaustion_score is not None else 0
                directional_score = buy_score - sell_score
                if directional_score >= 8:
                    directional_label = "Strong buy bias"
                elif directional_score >= 3:
                    directional_label = "Buy bias"
                elif directional_score <= -8:
                    directional_label = "Strong sell / exit bias"
                elif directional_score <= -3:
                    directional_label = "Sell / protect gains bias"
                else:
                    directional_label = "Mixed / neutral"

                st.markdown("#### Directional Checklist Score")
                directional_cols = st.columns(4)
                directional_cols[0].metric("Signed Score", f"{directional_score:+d}", directional_label)
                directional_cols[1].metric("Buy Checklist", f"{buy_score}/20")
                directional_cols[2].metric("Exit Checklist", f"{sell_score}/20")
                directional_cols[3].metric("Scale", "-20 sell to +20 buy")

            if bullish_continuation_score is not None:
                st.markdown("#### Bullish Continuation Checklist")
                checklist_cols = st.columns(5)
                checklist_cols[0].metric(
                    "Total Checklist Score",
                    f"{bullish_continuation_score['total_score']}/{bullish_continuation_score['max_score']}",
                    bullish_continuation_score["score_band"],
                )
                checklist_cols[1].metric(
                    "Daily Bias",
                    f"{bullish_continuation_score['daily']['score']}/10",
                    bullish_continuation_score["daily"]["interpretation"],
                )
                checklist_cols[2].metric(
                    "Intraday Entry",
                    f"{bullish_continuation_score['intraday']['score']}/10",
                    bullish_continuation_score["intraday"]["interpretation"],
                )
                checklist_cols[3].metric("ATR Penalty", str(bullish_continuation_score["atr_quality"]["penalty"]))
                checklist_cols[4].metric("Action", bullish_continuation_score["recommended_action"])

                minimum_rules = bullish_continuation_score["minimum_rules"]
                rule_cols = st.columns(3)
                rule_cols[0].metric("Daily >= 7", "Yes" if minimum_rules["daily_score_at_least_7"] else "No")
                rule_cols[1].metric("Intraday >= 7", "Yes" if minimum_rules["intraday_score_at_least_7"] else "No")
                rule_cols[2].metric("ATR Not Chasing", "Yes" if minimum_rules["not_chasing_by_atr"] else "No")

                daily_feature_cols = st.columns(5)
                daily_features = bullish_continuation_score["daily"]["features"]
                daily_feature_cols[0].metric("Daily ADX", f"{daily_features['adx14']:.1f}" if daily_features.get("adx14") is not None else "N/A")
                daily_feature_cols[1].metric("+DI / -DI", f"{daily_features['plus_di14']:.1f} / {daily_features['minus_di14']:.1f}" if daily_features.get("plus_di14") is not None and daily_features.get("minus_di14") is not None else "N/A")
                daily_feature_cols[2].metric("Daily RSI", f"{daily_features['rsi14']:.1f}" if daily_features.get("rsi14") is not None else "N/A")
                daily_feature_cols[3].metric("Daily MACD Hist", f"{daily_features['macd_histogram']:.2f}" if daily_features.get("macd_histogram") is not None else "N/A")
                daily_feature_cols[4].metric("Daily EMA20", format_money(daily_features.get("ema20")))

                intraday_feature_cols = st.columns(5)
                intraday_features = bullish_continuation_score["intraday"]["features"]
                intraday_feature_cols[0].metric("Intraday VWAP", format_money(intraday_features.get("vwap")))
                intraday_feature_cols[1].metric("EMA 9 / 20", f"{intraday_features['ema9']:.2f} / {intraday_features['ema20']:.2f}" if intraday_features.get("ema9") is not None and intraday_features.get("ema20") is not None else "N/A")
                intraday_feature_cols[2].metric("Intraday RSI", f"{intraday_features['rsi14']:.1f}" if intraday_features.get("rsi14") is not None else "N/A")
                intraday_feature_cols[3].metric("Intraday MACD Hist", f"{intraday_features['macd_histogram']:.2f}" if intraday_features.get("macd_histogram") is not None else "N/A")
                intraday_feature_cols[4].metric("Volume Ratio", f"{intraday_features['volume_ratio_20']:.2f}x" if intraday_features.get("volume_ratio_20") is not None else "N/A")

                with st.expander("Bullish Checklist Score Breakdown", expanded=False):
                    st.markdown("Daily bias components")
                    st.table(build_score_component_table(
                        bullish_continuation_score["daily"]["components"],
                        {"adx": 2, "di_direction": 2, "rsi": 2, "macd": 2, "ema20_slope": 1, "price_vs_ema20": 1},
                    ))
                    st.markdown("Intraday entry components")
                    st.table(build_score_component_table(
                        bullish_continuation_score["intraday"]["components"],
                        {"vwap": 2, "ema_structure": 3, "rsi": 2, "macd": 2, "volume": 1},
                    ))
                    st.markdown("ATR checks")
                    for check in bullish_continuation_score["atr_quality"]["checks"]:
                        st.write(f"- {check}")
                    st.markdown("Daily rationale")
                    for explanation in bullish_continuation_score["daily"]["explanations"]:
                        st.write(f"- {explanation}")
                    st.markdown("Intraday rationale")
                    for explanation in bullish_continuation_score["intraday"]["explanations"]:
                        st.write(f"- {explanation}")

            if sell_exhaustion_score is not None:
                st.markdown("#### Sell / Exhaustion Checklist")
                exit_cols = st.columns(5)
                exit_cols[0].metric(
                    "Total Exit Score",
                    f"{sell_exhaustion_score['total_score']}/{sell_exhaustion_score['max_score']}",
                    sell_exhaustion_score["score_band"],
                )
                exit_cols[1].metric(
                    "Daily Exhaustion",
                    f"{sell_exhaustion_score['daily']['score']}/10",
                    sell_exhaustion_score["daily"]["interpretation"],
                )
                exit_cols[2].metric(
                    "Intraday Exit",
                    f"{sell_exhaustion_score['intraday']['score']}/10",
                    sell_exhaustion_score["intraday"]["interpretation"],
                )
                exit_cols[3].metric("ATR Modifier", f"+{sell_exhaustion_score['atr_modifier']['modifier']}")
                exit_cols[4].metric("Action", sell_exhaustion_score["recommended_action"])

                exit_daily_features = sell_exhaustion_score["daily"]["features"]
                exit_daily_cols = st.columns(5)
                exit_daily_cols[0].metric("Daily ADX", f"{exit_daily_features['adx14']:.1f}" if exit_daily_features.get("adx14") is not None else "N/A")
                exit_daily_cols[1].metric("+DI / -DI", f"{exit_daily_features['plus_di14']:.1f} / {exit_daily_features['minus_di14']:.1f}" if exit_daily_features.get("plus_di14") is not None and exit_daily_features.get("minus_di14") is not None else "N/A")
                exit_daily_cols[2].metric("Daily RSI", f"{exit_daily_features['rsi14']:.1f}" if exit_daily_features.get("rsi14") is not None else "N/A")
                exit_daily_cols[3].metric("Daily MACD Hist", f"{exit_daily_features['macd_histogram']:.2f}" if exit_daily_features.get("macd_histogram") is not None else "N/A")
                exit_daily_cols[4].metric("Bearish Divergence", "Yes" if sell_exhaustion_score["daily"]["bearish_divergence"] else "No")

                exit_intraday_features = sell_exhaustion_score["intraday"]["features"]
                exit_intraday_cols = st.columns(5)
                exit_intraday_cols[0].metric("Intraday VWAP", format_money(exit_intraday_features.get("vwap")))
                exit_intraday_cols[1].metric("EMA 9 / 20", f"{exit_intraday_features['ema9']:.2f} / {exit_intraday_features['ema20']:.2f}" if exit_intraday_features.get("ema9") is not None and exit_intraday_features.get("ema20") is not None else "N/A")
                exit_intraday_cols[2].metric("Intraday RSI", f"{exit_intraday_features['rsi14']:.1f}" if exit_intraday_features.get("rsi14") is not None else "N/A")
                exit_intraday_cols[3].metric("Intraday MACD Hist", f"{exit_intraday_features['macd_histogram']:.2f}" if exit_intraday_features.get("macd_histogram") is not None else "N/A")
                exit_intraday_cols[4].metric("Volume Ratio", f"{exit_intraday_features['volume_ratio_20']:.2f}x" if exit_intraday_features.get("volume_ratio_20") is not None else "N/A")

                with st.expander("Sell Checklist Score Breakdown", expanded=False):
                    st.markdown("Daily exhaustion components")
                    st.table(build_score_component_table(
                        sell_exhaustion_score["daily"]["components"],
                        {"adx_trend_aging": 2, "di_weakening": 2, "rsi_exhaustion": 2, "macd_weakening": 2, "price_vs_ema20": 2},
                    ))
                    st.markdown("Intraday exit components")
                    st.table(build_score_component_table(
                        sell_exhaustion_score["intraday"]["components"],
                        {"vwap_loss": 2, "ema_structure": 2, "rsi_exit": 2, "macd_exit": 2, "distribution_volume": 2},
                    ))
                    st.markdown("ATR exit modifier")
                    for check in sell_exhaustion_score["atr_modifier"]["checks"]:
                        st.write(f"- {check}")
                    st.markdown("Daily exhaustion rationale")
                    for explanation in sell_exhaustion_score["daily"]["explanations"]:
                        st.write(f"- {explanation}")
                    st.markdown("Intraday exit rationale")
                    for explanation in sell_exhaustion_score["intraday"]["explanations"]:
                        st.write(f"- {explanation}")

            support_value, support_delta = format_level_metric(
                signal_result["nearest_levels"]["support"],
                signal_result["nearest_levels"]["distance_to_support_pct"],
                signal_result["nearest_levels"]["distance_to_support_atr"],
            )
            resistance_value, resistance_delta = format_level_metric(
                signal_result["nearest_levels"]["resistance"],
                signal_result["nearest_levels"]["distance_to_resistance_pct"],
                signal_result["nearest_levels"]["distance_to_resistance_atr"],
            )

            level_cols = st.columns(4)
            level_cols[0].metric("Nearest Support", support_value, support_delta)
            level_cols[1].metric(
                "Nearest Resistance",
                "Price discovery" if level_snapshot["price_discovery_mode"] else resistance_value,
                "No overhead actionable resistance" if level_snapshot["price_discovery_mode"] else resistance_delta,
            )
            ema20 = signal_result["features"]["ema20"]
            level_cols[2].metric("EMA20", format_money(ema20))
            level_cols[3].metric("Actionable Threshold", f"{actionable_threshold_pct:.2f}%")

            st.markdown("#### Trade Setup Interpretation")
            setup = signal_result["trade_setup"]
            setup_cols = st.columns(4)
            setup_cols[0].metric("Market Regime", setup["market_regime"])
            setup_cols[1].metric("Current Bias", setup["current_bias"])
            setup_cols[2].metric(
                "Confirmation Level",
                format_money(setup["confirmation_level"]),
                setup["confirmation_label"],
            )
            setup_cols[3].metric(
                "Invalidation Level",
                format_money(setup["invalidation_level"]),
                setup["invalidation_label"],
            )

            setup_cols = st.columns(5)
            setup_cols[0].metric("Nearest Support", format_money(setup["nearest_support"]))
            setup_cols[1].metric(
                "Nearest Resistance",
                "Price discovery" if setup["nearest_resistance"] is None else format_money(setup["nearest_resistance"]),
            )
            setup_cols[2].metric("Potential Upside", format_percent(setup["potential_upside_pct"]))
            setup_cols[3].metric("Downside Risk", format_percent(setup["downside_risk_pct"]))
            setup_cols[4].metric(
                "Risk / Reward",
                f"{setup['risk_reward_ratio']:.2f}" if setup["risk_reward_ratio"] is not None else "N/A",
            )

            st.info(f"{ticker}: {setup['narrative']}")

            st.markdown("#### Explanation")
            for explanation in signal_result["explanations"]:
                st.write(f"- {explanation}")

            with st.expander("Score Breakdown"):
                for label, value in signal_result["score_components"].items():
                    st.write(f"**{label.replace('_', ' ').title()}:** {value:+.2f}")

            with st.expander("Indicator Snapshot", expanded=True):
                features = signal_result["features"]
                indicator_cols = st.columns(5)
                indicator_cols[0].metric("RSI 14", f"{features['rsi14']:.1f}" if features["rsi14"] is not None else "N/A")
                indicator_cols[1].metric("MACD Hist", f"{features['macd_histogram']:.2f}" if features["macd_histogram"] is not None else "N/A")
                indicator_cols[2].metric("BB %B", f"{features['bb_percent_b']:.2f}" if features["bb_percent_b"] is not None else "N/A")
                indicator_cols[3].metric("Stoch %K", f"{features['stoch_k']:.1f}" if features["stoch_k"] is not None else "N/A")
                indicator_cols[4].metric("OBV 5-Bar Change", f"{features['obv_change_5']:,.0f}" if features["obv_change_5"] is not None else "N/A")

        chart = create_price_chart(
            df,
            ticker,
            sma_period=sma_period if sma_enabled else None,
            ema_period=ema_period if ema_enabled else None,
            show_ema9=show_ema9,
            show_vwap=show_vwap,
            show_bollinger_bands=show_bollinger_bands,
            show_rsi=show_rsi,
            show_macd=show_macd,
            show_stochastic=show_stochastic,
            show_obv=show_obv,
            show_adx=show_adx,
            show_volume_profile=show_volume_profile,
            volume_profile_buckets=400,
            actionable_levels=support_levels + resistance_levels if show_zones else None,
            historical_levels=historical_levels,
        )
        st.caption(
            f"Chart data: {bars_used:,} bars, {interval} candles, "
            f"{df.index.min().strftime('%Y-%m-%d %H:%M')} to {df.index.max().strftime('%Y-%m-%d %H:%M')}. "
            "Intraday charts use a compact trading-bar axis to remove overnight and weekend gaps."
        )
        st.plotly_chart(chart, use_container_width=True)

        with st.expander("Support/Resistance Methodology"):
            st.write("Levels are automatically detected from swing highs and swing lows.")
            st.write("Nearby swing points are clustered into zones using ATR-based tolerance so volatile stocks get wider grouping than quiet stocks.")
            st.write("Clustering answers whether prices are basically the same zone. Actionable filtering separately answers whether the zone is close enough to matter right now.")
            st.write("Support is only below current price. Resistance is only above current price.")
            st.write("Level strength is based on touches, recency, relative volume near the zone, and a penalty for repeated violations.")
            st.write(f"Current/visual analysis uses swing sensitivity N = {swing_sensitivity}, cluster tolerance {cluster_tolerance_pct:.2f}%, and actionable threshold {actionable_threshold_pct:.2f}%.")

        twenty_bar_return = None
        if len(close_series) > 20:
            prior_close = close_series.shift(20).iloc[-1]
            if pd.notna(prior_close) and prior_close != 0:
                twenty_bar_return = (close_series.iloc[-1] / prior_close - 1) * 100

        st.divider()
        st.subheader("Key Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Latest Close", format_money(latest_close))
        metric_cols[1].metric("20-Bar Return", f"{twenty_bar_return:.2f}%" if twenty_bar_return is not None else "N/A")
        metric_cols[2].metric("Annual Volatility", f"{volatility:.2f}%")
        metric_cols[3].metric("Volatility Percentile", f"{volatility_percentile * 100:.0f}%" if volatility_percentile is not None else "N/A")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Period Return", f"{period_return:.2f}%")
        metric_cols[1].metric("Average Volume (20)", f"{average_volume:,.0f}")
        metric_cols[2].metric("Cluster Tolerance", f"{cluster_tolerance_pct:.2f}%")
        metric_cols[3].metric("Max Drawdown", f"{max_drawdown:.2f}%")

        period_cols = st.columns(3)
        period_cols[0].metric("Start Price", format_money(start_close))
        period_cols[1].metric("End Price", format_money(end_close))
        period_cols[2].metric("Bars Used", str(bars_used))

        if show_backtest:
            st.divider()
            st.subheader("Historical Backtest")
            st.caption("Historical backtest uses daily data only and only information available up to each historical date. No look-ahead data is used.")
            st.caption("Exploratory backtest only. It does not include trading costs, slippage, taxes, or execution risk.")
            st.caption("When enabled, VCP is treated as a supporting factor, not a standalone trade instruction.")

            with st.spinner("Running backtest..."):
                backtest_results = backtest_signals_no_lookahead(
                    df=analysis_df,
                    ticker=ticker,
                    max_levels=zone_count,
                    lookback_periods=(5, 10, 20),
                    swing_sensitivity=swing_sensitivity,
                    enable_vcp_detection=enable_vcp_detection,
                    vcp_min_base_length=vcp_min_base_length,
                    vcp_max_base_length=vcp_max_base_length,
                )

            backtest_cols = st.columns(5)
            backtest_cols[0].metric("Total Signals", int(backtest_results["total_signals"]))
            backtest_cols[1].metric("BUY Signals", int(backtest_results["buy_signals"]))
            backtest_cols[2].metric("SELL Signals", int(backtest_results["sell_signals"]))
            backtest_cols[3].metric("BUY Win Rate", f"{backtest_results['buy_win_rate']:.1f}%")
            backtest_cols[4].metric("SELL Win Rate", f"{backtest_results['sell_win_rate']:.1f}%")

            avg_cols = st.columns(3)
            avg_cols[0].metric("Avg Forward Return (5)", f"{backtest_results['average_forward_return_5']:.2f}%")
            avg_cols[1].metric("Avg Forward Return (10)", f"{backtest_results['average_forward_return_10']:.2f}%")
            avg_cols[2].metric("Avg Forward Return (20)", f"{backtest_results['average_forward_return_20']:.2f}%")

            median_cols = st.columns(3)
            median_cols[0].metric("Median Forward Return (5)", f"{backtest_results['median_forward_return_5']:.2f}%")
            median_cols[1].metric("Median Forward Return (10)", f"{backtest_results['median_forward_return_10']:.2f}%")
            median_cols[2].metric("Median Forward Return (20)", f"{backtest_results['median_forward_return_20']:.2f}%")

            risk_cols = st.columns(2)
            risk_cols[0].metric("Signal Equity Max Drawdown", f"{backtest_results['max_drawdown']:.2f}%")
            risk_cols[1].metric("Buy-and-Hold Return", f"{backtest_results['buy_and_hold_return']:.2f}%")

            consistency_cols = st.columns(3)
            consistency_cols[0].metric("Backtest Start Price", format_money(backtest_results["start_price"]))
            consistency_cols[1].metric("Backtest End Price", format_money(backtest_results["end_price"]))
            consistency_cols[2].metric("Backtest Bars Used", str(int(backtest_results["bars_used"])))

        st.divider()
        st.subheader("Actionable Support/Resistance Levels")
        actionable_rows = support_levels + resistance_levels
        if actionable_rows:
            st.table(build_level_table(actionable_rows))
        else:
            st.info("No actionable support or resistance levels were detected near current price.")

        st.subheader("Historical Support/Resistance Levels")
        historical_rows = historical_supports + historical_resistances
        if historical_rows:
            st.table(build_level_table(historical_rows))
        else:
            st.caption("Enable 'Show Historical Levels' in the sidebar to inspect distant zones.")

        if enable_vcp_detection and vcp_result is not None:
            st.divider()
            st.subheader("VCP Pattern Detector")
            st.caption("Current/visual analysis uses the selected visible data range. VCP is a setup condition, not a trade recommendation.")

            if vcp_result["status"] == "No Clear VCP":
                st.write("No Clear VCP")
                for reason in vcp_result.get("failed_reasons", []):
                    st.write(f"- {reason}")
            else:
                st.success(vcp_result["status"])

            vcp_cols = st.columns(5)
            vcp_cols[0].metric("VCP Status", vcp_result["status"])
            vcp_cols[1].metric("VCP Score", f"{vcp_result['score']:.2f}")
            vcp_cols[2].metric("Pivot Price", format_money(vcp_result.get("pivot")))
            vcp_cols[3].metric("Current Close", format_money(vcp_result.get("current_close")))
            vcp_cols[4].metric("Distance to Pivot %", format_percent(vcp_result.get("distance_to_pivot_pct")))

            vcp_cols = st.columns(5)
            vcp_cols[0].metric("Base Length", str(vcp_result.get("base_length", "N/A")))
            vcp_cols[1].metric("Base High", format_money(vcp_result.get("base_high")))
            vcp_cols[2].metric("Base Low", format_money(vcp_result.get("base_low")))
            vcp_cols[3].metric("Base Depth %", format_percent(vcp_result.get("base_depth_pct")))
            vcp_cols[4].metric("Pullback Sequence", ", ".join(f"{value * 100:.1f}%" for value in vcp_result.get("pullbacks", [])) or "N/A")

            vcp_cols = st.columns(5)
            vcp_cols[0].metric("ATR Contraction Ratio", f"{vcp_result['atr_contraction_ratio']:.2f}" if vcp_result.get("atr_contraction_ratio") is not None else "N/A")
            vcp_cols[1].metric("Volume Contraction Ratio", f"{vcp_result['volume_contraction_ratio']:.2f}" if vcp_result.get("volume_contraction_ratio") is not None else "N/A")
            vcp_cols[2].metric("Breakout Trigger", format_money(vcp_result.get("breakout_trigger_price")))
            vcp_cols[3].metric("Breakout Volume Requirement", f"{vcp_result['breakout_volume_requirement']:,.0f}" if vcp_result.get("breakout_volume_requirement") is not None else "N/A")
            vcp_cols[4].metric("Volume Confirmed", "Yes" if vcp_result.get("breakout_volume_confirmed") else "No")

            st.markdown("#### VCP Explanation")
            for explanation in vcp_result.get("explanations", []):
                st.write(f"- {explanation}")
            st.write("- VCP is a setup condition, not a trade recommendation.")

            show_vcp_chart = vcp_result["status"] in {"Strong VCP Candidate", "Possible VCP Candidate", "VCP Breakout Confirmed"} or show_vcp_diagnostics
            if show_vcp_chart:
                st.subheader("VCP Evidence Chart")
                vcp_chart = create_vcp_evidence_chart(
                    vcp_result,
                    support_levels=support_levels,
                    resistance_levels=resistance_levels,
                    show_annotations=show_vcp_annotations,
                )
                st.plotly_chart(vcp_chart, use_container_width=True)
                st.markdown("#### Why this was flagged as VCP")
                st.write("- Current/visual analysis is based on the selected visible data range.")
                for explanation in vcp_result.get("explanations", []):
                    st.write(f"- {explanation}")

    except Exception as exc:
        st.error(f"Error loading dashboard: {exc}")
        import traceback
        st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
