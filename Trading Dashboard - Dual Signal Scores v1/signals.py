"""
Explainable, regime-based signal engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import (
    calculate_adx,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_stochastic,
    calculate_vwap,
)
from levels import build_level_snapshot, calculate_atr_series, get_nearest_levels


STRONG_LEVEL_MIN_HITS = 2
VOLATILITY_ALERT_PERCENTILE = 0.80
TRENDING_UP = "Trending Up"
TRENDING_DOWN = "Trending Down"
RANGING = "Ranging"
HIGH_VOLATILITY = "High Volatility"
NEUTRAL = "Neutral"


def annualization_factor_for_ticker(ticker: str) -> int:
    """Use stock or crypto annualization conventions."""
    return 365 if "-USD" in ticker.upper() else 252


def calculate_slope_series(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Calculate slope as a lookback return."""
    return series / series.shift(lookback) - 1


def calculate_sma_slope(sma_series: pd.Series, lookback: int = 5) -> float:
    """Return the latest SMA slope."""
    slope_series = calculate_slope_series(sma_series, lookback=lookback)
    latest_value = slope_series.iloc[-1] if not slope_series.empty else np.nan
    return 0.0 if pd.isna(latest_value) else float(latest_value)


def calculate_ema_slope(ema_series: pd.Series, lookback: int = 5) -> float:
    """Return the latest EMA slope."""
    slope_series = calculate_slope_series(ema_series, lookback=lookback)
    latest_value = slope_series.iloc[-1] if not slope_series.empty else np.nan
    return 0.0 if pd.isna(latest_value) else float(latest_value)


def _expanding_percentile(series: pd.Series, percentile: float, min_periods: int = 20) -> pd.Series:
    """Calculate an expanding percentile threshold using only history available so far."""
    return series.expanding(min_periods=min_periods).apply(
        lambda values: float(np.nanquantile(values, percentile)) if np.isfinite(values).any() else np.nan,
        raw=True,
    )


def _percentile_rank_last(values: np.ndarray) -> float:
    """Return the percentile rank of the latest finite value in the array."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan

    latest_value = finite_values[-1]
    return float(np.mean(finite_values <= latest_value))


def _expanding_percentile_rank(series: pd.Series, min_periods: int = 20) -> pd.Series:
    """Calculate the expanding percentile rank of the latest value."""
    return series.expanding(min_periods=min_periods).apply(_percentile_rank_last, raw=True)


def _safe_float(value) -> float | None:
    """Return a float when the value is available."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _latest_feature_dict(row: pd.Series) -> dict:
    """Convert the common feature columns into a display-safe dictionary."""
    fields = [
        "close",
        "ema9",
        "ema9_slope",
        "ema20",
        "ema20_slope",
        "sma50",
        "sma200",
        "rsi14",
        "rsi2",
        "rsi4",
        "rsi14_change_3",
        "macd",
        "macd_signal",
        "macd_histogram",
        "macd_histogram_change_3",
        "adx14",
        "adx14_slope",
        "plus_di14",
        "minus_di14",
        "di_spread",
        "di_spread_change_3",
        "vwap",
        "atr14",
        "volume",
        "average_volume_20",
        "volume_ratio_20",
        "atr_pct",
    ]
    return {field: _safe_float(row.get(field)) for field in fields}


def _score_daily_bias(row: pd.Series) -> dict:
    """Score the 6-month daily bullish continuation bias out of 10."""
    components = {}
    explanations = []

    adx = row.get("adx14")
    adx_slope = row.get("adx14_slope")
    if pd.isna(adx) or adx < 18:
        components["adx"] = 0
        explanations.append("Daily ADX is below 18, so trend strength is weak.")
    elif adx < 25:
        components["adx"] = 1
        explanations.append("Daily ADX is between 18 and 25, so a trend may be developing.")
    elif pd.notna(adx_slope) and adx_slope > 0:
        components["adx"] = 2
        explanations.append("Daily ADX is above 25 and rising.")
    else:
        components["adx"] = 1
        explanations.append("Daily ADX is strong but not clearly rising.")

    plus_di = row.get("plus_di14")
    minus_di = row.get("minus_di14")
    di_spread_change = row.get("di_spread_change_3")
    if pd.isna(plus_di) or pd.isna(minus_di) or plus_di <= minus_di:
        components["di_direction"] = 0
        explanations.append("Daily +DI is not above -DI.")
    elif pd.notna(di_spread_change) and di_spread_change > 0:
        components["di_direction"] = 2
        explanations.append("Daily +DI is above -DI and the spread is widening.")
    else:
        components["di_direction"] = 1
        explanations.append("Daily +DI is above -DI.")

    rsi = row.get("rsi14")
    rsi_change = row.get("rsi14_change_3")
    if pd.isna(rsi) or rsi < 40:
        components["rsi"] = 0
        explanations.append("Daily RSI is below 40.")
    elif rsi < 50:
        components["rsi"] = 2 if pd.notna(rsi_change) and rsi_change > 0 else 1
        explanations.append("Daily RSI is in the 40-50 pullback zone" + (" and turning up." if components["rsi"] == 2 else "."))
    elif rsi <= 65:
        components["rsi"] = 2
        explanations.append("Daily RSI is in the healthy 50-65 momentum zone.")
    elif rsi <= 72:
        components["rsi"] = 1
        explanations.append("Daily RSI is bullish but slightly extended.")
    else:
        components["rsi"] = 0
        explanations.append("Daily RSI is above 72 and stretched.")

    macd = row.get("macd")
    macd_signal = row.get("macd_signal")
    macd_hist = row.get("macd_histogram")
    macd_hist_change = row.get("macd_histogram_change_3")
    if pd.isna(macd) or pd.isna(macd_signal) or pd.isna(macd_hist):
        components["macd"] = 0
        explanations.append("Daily MACD is unavailable.")
    elif macd > macd_signal and (macd > 0 or macd_hist > 0) and (pd.isna(macd_hist_change) or macd_hist_change >= 0):
        components["macd"] = 2
        explanations.append("Daily MACD is bullish or clearly strengthening.")
    elif pd.notna(macd_hist_change) and macd_hist_change > 0:
        components["macd"] = 1
        explanations.append("Daily MACD histogram is improving.")
    else:
        components["macd"] = 0
        explanations.append("Daily MACD is bearish or histogram is falling.")

    if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] > 0:
        components["ema20_slope"] = 1
        explanations.append("Daily 20 EMA is upward sloping.")
    else:
        components["ema20_slope"] = 0
        explanations.append("Daily 20 EMA is flat or down.")

    if pd.notna(row.get("ema20")) and row["close"] > row["ema20"]:
        components["price_vs_ema20"] = 1
        explanations.append("Daily price is above the 20 EMA.")
    else:
        components["price_vs_ema20"] = 0
        explanations.append("Daily price is below the 20 EMA.")

    score = int(sum(components.values()))
    if score >= 8:
        interpretation = "Strong long candidate"
    elif score >= 6:
        interpretation = "Tradable, needs clean intraday trigger"
    else:
        interpretation = "Usually skip for bullish continuation"

    return {
        "score": score,
        "max_score": 10,
        "components": components,
        "interpretation": interpretation,
        "features": _latest_feature_dict(row),
        "explanations": explanations,
    }


def _score_intraday_entry(row: pd.Series, previous_row: pd.Series | None = None) -> dict:
    """Score the intraday bullish entry trigger out of 10."""
    components = {}
    explanations = []
    previous_row = previous_row if previous_row is not None else pd.Series(dtype=float)

    close = row.get("close")
    vwap = row.get("vwap")
    previous_close = previous_row.get("close")
    previous_vwap = previous_row.get("vwap")
    recent_above_vwap = bool(pd.notna(close) and pd.notna(vwap) and close > vwap)
    reclaiming_vwap = bool(
        pd.notna(previous_close)
        and pd.notna(previous_vwap)
        and previous_close <= previous_vwap
        and pd.notna(close)
        and pd.notna(vwap)
        and close > vwap
    )
    if reclaiming_vwap:
        components["vwap"] = 1
        explanations.append("Intraday price is reclaiming VWAP.")
    elif recent_above_vwap:
        components["vwap"] = 2
        explanations.append("Intraday price is above and holding VWAP.")
    else:
        components["vwap"] = 0
        explanations.append("Intraday price is below VWAP.")

    ema9 = row.get("ema9")
    ema20 = row.get("ema20")
    ema9_slope = row.get("ema9_slope")
    ema20_slope = row.get("ema20_slope")
    price_above_ema9 = bool(pd.notna(close) and pd.notna(ema9) and close > ema9)
    price_above_ema20 = bool(pd.notna(close) and pd.notna(ema20) and close > ema20)
    ema_crossing = bool(
        pd.notna(previous_row.get("ema9"))
        and pd.notna(previous_row.get("ema20"))
        and previous_row["ema9"] <= previous_row["ema20"]
        and pd.notna(ema9)
        and pd.notna(ema20)
        and ema9 > ema20
    )
    ema_base = 0
    if pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20 and price_above_ema9 and price_above_ema20:
        ema_base = 2
    elif ema_crossing or price_above_ema9 or price_above_ema20:
        ema_base = 1

    pullback_hold = bool(
        ema_base > 0
        and pd.notna(row.get("low"))
        and (
            (pd.notna(ema9) and row["low"] <= ema9 and close >= ema9)
            or (pd.notna(ema20) and row["low"] <= ema20 and close >= ema20)
        )
        and bool(row.get("green_candle", False))
    )
    ema_bonus = 1 if pullback_hold and pd.notna(ema9_slope) and pd.notna(ema20_slope) and ema9_slope >= 0 and ema20_slope >= 0 else 0
    components["ema_structure"] = min(3, ema_base + ema_bonus)
    if components["ema_structure"] == 0:
        explanations.append("Intraday EMA structure is not bullish.")
    elif components["ema_structure"] == 1:
        explanations.append("Intraday EMA crossover or price reclaim is forming.")
    elif components["ema_structure"] == 2:
        explanations.append("Intraday 9 EMA is above 20 EMA and price is above both.")
    else:
        explanations.append("Intraday EMA structure is bullish and a pullback held the 9/20 EMA area.")

    rsi = row.get("rsi14")
    rsi_change = row.get("rsi14_change_3")
    if pd.isna(rsi) or rsi < 35:
        components["rsi"] = 0
        explanations.append("Intraday RSI is below 35.")
    elif 40 <= rsi <= 55 and pd.notna(rsi_change) and rsi_change > 0:
        components["rsi"] = 2
        explanations.append("Intraday RSI is turning up from the preferred 40-55 entry zone.")
    elif 55 < rsi <= 68:
        components["rsi"] = 1
        explanations.append("Intraday RSI shows momentum is intact.")
    elif rsi > 75:
        components["rsi"] = 0
        explanations.append("Intraday RSI is above 75, so the entry may be chasing.")
    else:
        components["rsi"] = 1
        explanations.append("Intraday RSI is acceptable but not ideal.")

    macd = row.get("macd")
    macd_signal = row.get("macd_signal")
    macd_hist_change = row.get("macd_histogram_change_3")
    if pd.notna(macd) and pd.notna(macd_signal) and macd > macd_signal:
        components["macd"] = 2
        explanations.append("Intraday MACD is above signal.")
    elif pd.notna(macd_hist_change) and macd_hist_change > 0:
        components["macd"] = 1
        explanations.append("Intraday MACD histogram is improving.")
    else:
        components["macd"] = 0
        explanations.append("Intraday MACD is bearish or not improving.")

    volume_ratio = row.get("volume_ratio_20")
    if pd.notna(volume_ratio) and volume_ratio >= 0.90:
        components["volume"] = 1
        explanations.append("Intraday volume participation is healthy.")
    else:
        components["volume"] = 0
        explanations.append("Intraday volume participation is weak.")

    score = int(sum(components.values()))
    if score >= 8:
        interpretation = "Strong intraday trigger"
    elif score >= 7:
        interpretation = "Acceptable intraday trigger"
    elif score >= 5:
        interpretation = "Watch for cleaner execution"
    else:
        interpretation = "Entry trigger is weak"

    return {
        "score": score,
        "max_score": 10,
        "components": components,
        "interpretation": interpretation,
        "features": _latest_feature_dict(row),
        "explanations": explanations,
    }


def _score_atr_quality(row: pd.Series, nearest_resistance: float | None = None) -> dict:
    """Apply ATR quality and chasing penalties from 0 to -2."""
    close = row.get("close")
    atr = row.get("atr14")
    penalties = []
    raw_penalty = 0

    if pd.isna(close) or pd.isna(atr) or close <= 0 or atr <= 0:
        return {
            "penalty": 0,
            "max_penalty": -2,
            "hard_stop": False,
            "checks": ["ATR quality could not be evaluated."],
        }

    ema9 = row.get("ema9")
    ema20 = row.get("ema20")
    vwap = row.get("vwap")

    if pd.notna(ema9) and close > ema9 + 0.5 * atr:
        raw_penalty -= 1
        penalties.append("Entry is more than 0.5 ATR above the 9 EMA.")

    extended_vs_vwap = bool(pd.notna(vwap) and close > vwap + 0.75 * atr)
    extended_vs_ema20 = bool(pd.notna(ema20) and close > ema20 + 0.75 * atr)
    if extended_vs_vwap or extended_vs_ema20:
        raw_penalty -= 2
        penalties.append("Entry is more than 0.75 ATR above VWAP or the 20 EMA.")

    candle_range = row.get("high") - row.get("low") if pd.notna(row.get("high")) and pd.notna(row.get("low")) else np.nan
    candle_body = row.get("close") - row.get("open") if pd.notna(row.get("open")) and pd.notna(row.get("close")) else np.nan
    if pd.notna(candle_range) and pd.notna(candle_body) and candle_body > 0.75 * atr and candle_range > atr:
        raw_penalty -= 2
        penalties.append("Latest candle is sharp enough that the entry may be very extended.")

    if atr / close < 0.0025:
        raw_penalty -= 1
        penalties.append("ATR is very small relative to price, so the move may not justify the trade.")

    if nearest_resistance is not None and pd.notna(nearest_resistance):
        room_to_resistance = float(nearest_resistance) - close
        if room_to_resistance > 0 and room_to_resistance < atr:
            raw_penalty -= 1
            penalties.append("Less than 1 ATR is available before nearby resistance.")

    capped_penalty = max(-2, raw_penalty)
    if not penalties:
        penalties.append("ATR quality checks passed; entry is not obviously stretched.")

    return {
        "penalty": int(capped_penalty),
        "max_penalty": -2,
        "hard_stop": bool(extended_vs_vwap or extended_vs_ema20),
        "checks": penalties,
    }


def _detect_bearish_divergence(feature_frame: pd.DataFrame, lookback: int = 30) -> bool:
    """Detect a simple price higher-high with RSI lower-high divergence."""
    if len(feature_frame) < max(lookback, 10):
        return False

    window = feature_frame.tail(lookback)
    price = window["close"]
    rsi = window["rsi14"]
    if price.isna().all() or rsi.isna().all():
        return False

    midpoint = max(2, len(window) // 2)
    first_half = window.iloc[:midpoint]
    second_half = window.iloc[midpoint:]
    if first_half.empty or second_half.empty:
        return False

    first_price_high = first_half["close"].max()
    second_price_high = second_half["close"].max()
    first_rsi_high = first_half["rsi14"].max()
    second_rsi_high = second_half["rsi14"].max()

    return bool(
        pd.notna(first_price_high)
        and pd.notna(second_price_high)
        and pd.notna(first_rsi_high)
        and pd.notna(second_rsi_high)
        and second_price_high > first_price_high
        and second_rsi_high < first_rsi_high
    )


def _score_daily_exhaustion(row: pd.Series, bearish_divergence: bool = False) -> dict:
    """Score daily sell/exhaustion risk out of 10 for long positions."""
    components = {}
    explanations = []

    adx = row.get("adx14")
    adx_slope = row.get("adx14_slope")
    if pd.notna(adx) and pd.notna(adx_slope) and adx > 25 and adx_slope < 0:
        components["adx_trend_aging"] = 2
        explanations.append("Daily ADX is falling after a strong trend.")
    elif pd.notna(adx) and pd.notna(adx_slope) and adx >= 35 and adx_slope <= 0:
        components["adx_trend_aging"] = 1
        explanations.append("Daily ADX is very high and flattening.")
    else:
        components["adx_trend_aging"] = 0
        explanations.append("Daily ADX does not show trend aging.")

    plus_di = row.get("plus_di14")
    minus_di = row.get("minus_di14")
    di_spread_change = row.get("di_spread_change_3")
    if pd.notna(plus_di) and pd.notna(minus_di) and plus_di < minus_di:
        components["di_weakening"] = 2
        explanations.append("Daily +DI has crossed below -DI.")
    elif pd.notna(plus_di) and pd.notna(minus_di) and plus_di > minus_di and pd.notna(di_spread_change) and di_spread_change < 0:
        components["di_weakening"] = 1
        explanations.append("Daily +DI is narrowing toward -DI.")
    else:
        components["di_weakening"] = 0
        explanations.append("Daily +DI remains in control.")

    rsi = row.get("rsi14")
    rsi_change = row.get("rsi14_change_3")
    if bearish_divergence:
        components["rsi_exhaustion"] = 2
        explanations.append("Bearish daily RSI divergence is present.")
    elif pd.notna(rsi) and rsi > 70 and pd.notna(rsi_change) and rsi_change < 0:
        components["rsi_exhaustion"] = 2
        explanations.append("Daily RSI is above 70 and rolling over.")
    elif pd.notna(rsi) and rsi > 70:
        components["rsi_exhaustion"] = 1
        explanations.append("Daily RSI is above 70 and extended.")
    else:
        components["rsi_exhaustion"] = 0
        explanations.append("Daily RSI does not show exhaustion.")

    macd = row.get("macd")
    macd_signal = row.get("macd_signal")
    macd_hist = row.get("macd_histogram")
    macd_hist_change = row.get("macd_histogram_change_3")
    if pd.notna(macd) and pd.notna(macd_signal) and pd.notna(macd_hist) and (macd < macd_signal or macd_hist < 0):
        components["macd_weakening"] = 2
        explanations.append("Daily MACD has crossed bearish or histogram is negative.")
    elif pd.notna(macd_hist_change) and macd_hist_change < 0:
        components["macd_weakening"] = 1
        explanations.append("Daily MACD histogram is shrinking.")
    else:
        components["macd_weakening"] = 0
        explanations.append("Daily MACD remains constructive.")

    close = row.get("close")
    ema20 = row.get("ema20")
    ema20_slope = row.get("ema20_slope")
    atr = row.get("atr14")
    if pd.notna(close) and pd.notna(ema20) and close < ema20:
        components["price_vs_ema20"] = 2
        explanations.append("Daily price has closed below the 20 EMA.")
    elif pd.notna(close) and pd.notna(ema20) and pd.notna(atr) and atr > 0 and close > ema20 + atr:
        components["price_vs_ema20"] = 1
        explanations.append("Daily price is extended more than 1 ATR above the 20 EMA.")
    elif pd.notna(ema20_slope) and ema20_slope <= 0:
        components["price_vs_ema20"] = 1
        explanations.append("Daily 20 EMA is no longer rising cleanly.")
    else:
        components["price_vs_ema20"] = 0
        explanations.append("Daily price remains supported by a rising 20 EMA.")

    score = int(sum(components.values()))
    if score <= 3:
        interpretation = "Trend still healthy"
    elif score <= 5:
        interpretation = "Early warning"
    elif score <= 7:
        interpretation = "Move may be aging; tighten stops"
    else:
        interpretation = "Strong sell/trim warning"

    return {
        "score": score,
        "max_score": 10,
        "components": components,
        "interpretation": interpretation,
        "features": _latest_feature_dict(row),
        "bearish_divergence": bearish_divergence,
        "explanations": explanations,
    }


def _score_intraday_exit(row: pd.Series, previous_row: pd.Series | None = None, bearish_divergence: bool = False) -> dict:
    """Score intraday exit trigger risk out of 10 for long positions."""
    components = {}
    explanations = []
    previous_row = previous_row if previous_row is not None else pd.Series(dtype=float)

    close = row.get("close")
    vwap = row.get("vwap")
    previous_close = previous_row.get("close")
    previous_vwap = previous_row.get("vwap")
    lost_vwap = bool(pd.notna(close) and pd.notna(vwap) and close < vwap)
    failed_reclaim = bool(
        lost_vwap
        and pd.notna(previous_close)
        and pd.notna(previous_vwap)
        and previous_close < previous_vwap
    )
    if failed_reclaim:
        components["vwap_loss"] = 2
        explanations.append("Intraday price lost VWAP and failed to reclaim it.")
    elif lost_vwap:
        components["vwap_loss"] = 1
        explanations.append("Intraday price is below VWAP.")
    else:
        components["vwap_loss"] = 0
        explanations.append("Intraday price is holding above VWAP.")

    ema9 = row.get("ema9")
    ema20 = row.get("ema20")
    ema9_slope = row.get("ema9_slope")
    previous_ema9 = previous_row.get("ema9")
    previous_ema20 = previous_row.get("ema20")
    bearish_ema_cross = bool(
        pd.notna(previous_ema9)
        and pd.notna(previous_ema20)
        and previous_ema9 >= previous_ema20
        and pd.notna(ema9)
        and pd.notna(ema20)
        and ema9 < ema20
    )
    if bearish_ema_cross:
        components["ema_structure"] = 2
        explanations.append("Intraday 9 EMA crossed below 20 EMA.")
    elif pd.notna(ema9_slope) and ema9_slope <= 0:
        components["ema_structure"] = 1
        explanations.append("Intraday 9 EMA is flattening or price is chopping around EMAs.")
    else:
        components["ema_structure"] = 0
        explanations.append("Intraday EMA structure remains constructive.")

    rsi = row.get("rsi14")
    rsi_change = row.get("rsi14_change_3")
    previous_rsi = previous_row.get("rsi14")
    if bearish_divergence:
        components["rsi_exit"] = 2
        explanations.append("Bearish intraday RSI divergence is present.")
    elif pd.notna(rsi) and pd.notna(previous_rsi) and previous_rsi > 70 and rsi < 50:
        components["rsi_exit"] = 2
        explanations.append("Intraday RSI broke below 50 after an overbought move.")
    elif pd.notna(rsi) and rsi > 75 and pd.notna(rsi_change) and rsi_change < 0:
        components["rsi_exit"] = 1
        explanations.append("Intraday RSI is over 75 and rolling over.")
    else:
        components["rsi_exit"] = 0
        explanations.append("Intraday RSI is not flashing an exit trigger.")

    macd = row.get("macd")
    macd_signal = row.get("macd_signal")
    macd_hist_change = row.get("macd_histogram_change_3")
    if pd.notna(macd) and pd.notna(macd_signal) and macd < macd_signal:
        components["macd_exit"] = 2
        explanations.append("Intraday MACD has crossed bearish.")
    elif pd.notna(macd_hist_change) and macd_hist_change < 0:
        components["macd_exit"] = 1
        explanations.append("Intraday MACD histogram is fading.")
    else:
        components["macd_exit"] = 0
        explanations.append("Intraday MACD remains stable or rising.")

    volume_ratio = row.get("volume_ratio_20")
    red_candle = bool(pd.notna(row.get("close")) and pd.notna(row.get("open")) and row["close"] < row["open"])
    near_high_rejection = bool(pd.notna(row.get("high")) and pd.notna(row.get("close")) and row["high"] > row["close"])
    if red_candle and near_high_rejection and pd.notna(volume_ratio) and volume_ratio >= 1.5 and lost_vwap:
        components["distribution_volume"] = 2
        explanations.append("High-volume rejection is appearing with VWAP weakness.")
    elif red_candle and pd.notna(volume_ratio) and volume_ratio >= 1.2:
        components["distribution_volume"] = 1
        explanations.append("Heavy red candle volume is present.")
    else:
        components["distribution_volume"] = 0
        explanations.append("Pullback volume does not show clear distribution.")

    score = int(sum(components.values()))
    if score <= 3:
        interpretation = "No exit trigger"
    elif score <= 5:
        interpretation = "Watch closely"
    elif score <= 7:
        interpretation = "Trim or tighten stop"
    else:
        interpretation = "Strong exit trigger"

    return {
        "score": score,
        "max_score": 10,
        "components": components,
        "interpretation": interpretation,
        "features": _latest_feature_dict(row),
        "bearish_divergence": bearish_divergence,
        "explanations": explanations,
    }


def _score_exit_atr_modifier(row: pd.Series) -> dict:
    """Apply ATR modifiers for exit/exhaustion scoring, capped at +2."""
    close = row.get("close")
    open_price = row.get("open")
    low = row.get("low")
    ema20 = row.get("ema20")
    vwap = row.get("vwap")
    atr = row.get("atr14")
    checks = []
    modifier = 0

    if pd.isna(close) or pd.isna(atr) or atr <= 0:
        return {"modifier": 0, "max_modifier": 2, "checks": ["ATR exit modifier could not be evaluated."]}

    if pd.notna(ema20) and close > ema20 + 1.5 * atr:
        modifier += 2
        checks.append("Price is more than 1.5 ATR above the 20 EMA.")
    elif pd.notna(ema20) and close > ema20 + atr:
        modifier += 1
        checks.append("Price is more than 1 ATR above the 20 EMA after a fast move.")

    if pd.notna(open_price) and pd.notna(low) and close < open_price and (open_price - low) > atr:
        modifier += 2
        checks.append("Large red reversal candle is greater than 1 ATR.")

    broke_vwap_by_atr = bool(pd.notna(vwap) and close < vwap - 0.5 * atr)
    broke_ema20_by_atr = bool(pd.notna(ema20) and close < ema20 - 0.5 * atr)
    if broke_vwap_by_atr or broke_ema20_by_atr:
        modifier += 1
        checks.append("Price broke below VWAP or 20 EMA by more than 0.5 ATR.")

    capped_modifier = min(2, modifier)
    if not checks:
        checks.append("ATR exit modifier is neutral.")

    return {"modifier": int(capped_modifier), "max_modifier": 2, "checks": checks}


def _classify_exit_total(total_score: int) -> str:
    """Map the final exit score into action bands."""
    if total_score <= 6:
        return "Hold / trend intact"
    if total_score <= 10:
        return "Watchlist warning"
    if total_score <= 13:
        return "Tighten stop, consider partial trim"
    if total_score <= 16:
        return "Trim or exit most of position"
    return "Strong exit signal"


def generate_sell_exhaustion_score(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None = None,
    ticker: str = "",
) -> dict:
    """Generate a two-stage sell/exhaustion score for long exits."""
    daily_features = build_feature_frame(daily_df, ticker or "TICKER")
    daily_row = daily_features.iloc[-1]
    daily_divergence = _detect_bearish_divergence(daily_features, lookback=30)
    daily_score = _score_daily_exhaustion(daily_row, bearish_divergence=daily_divergence)

    intraday_score = {
        "score": 0,
        "max_score": 10,
        "components": {},
        "interpretation": "Intraday data unavailable",
        "features": {},
        "bearish_divergence": False,
        "explanations": ["Intraday data was not available, so exit trigger timing could not be scored."],
    }
    atr_modifier = {
        "modifier": 0,
        "max_modifier": 2,
        "checks": ["ATR exit modifier could not be evaluated without intraday data."],
    }

    if intraday_df is not None and not intraday_df.empty and len(intraday_df) >= 20:
        intraday_features = build_feature_frame(intraday_df, ticker or "TICKER")
        intraday_row = intraday_features.iloc[-1]
        previous_row = intraday_features.iloc[-2] if len(intraday_features) >= 2 else None
        intraday_divergence = _detect_bearish_divergence(intraday_features, lookback=30)
        intraday_score = _score_intraday_exit(
            intraday_row,
            previous_row=previous_row,
            bearish_divergence=intraday_divergence,
        )
        atr_modifier = _score_exit_atr_modifier(intraday_row)

    total_score = int(daily_score["score"] + intraday_score["score"] + atr_modifier["modifier"])
    total_score = min(20, max(0, total_score))
    action = _classify_exit_total(total_score)

    return {
        "total_score": total_score,
        "max_score": 20,
        "score_band": action,
        "recommended_action": action,
        "daily": daily_score,
        "intraday": intraday_score,
        "atr_modifier": atr_modifier,
    }


def _window_return(close: pd.Series, bars: int) -> float | None:
    """Return lookback performance when enough history exists."""
    if len(close) <= bars:
        return None
    prior = close.iloc[-bars]
    latest = close.iloc[-1]
    if pd.isna(prior) or prior == 0 or pd.isna(latest):
        return None
    return float(latest / prior - 1)


def _calculate_relative_strength(stock_close: pd.Series, benchmark_close: pd.Series | None = None) -> dict:
    """Calculate stock returns and relative returns versus a benchmark."""
    windows = {"3m": 63, "6m": 126, "12m": 252}
    result = {}
    benchmark_close = benchmark_close.dropna() if benchmark_close is not None else pd.Series(dtype=float)
    for label, bars in windows.items():
        stock_return = _window_return(stock_close.dropna(), bars)
        benchmark_return = _window_return(benchmark_close, bars) if not benchmark_close.empty else None
        relative_return = None
        if stock_return is not None and benchmark_return is not None:
            relative_return = stock_return - benchmark_return
        result[label] = {
            "stock_return": stock_return,
            "benchmark_return": benchmark_return,
            "relative_return": relative_return,
        }
    return result


def _score_semi_auto_daily(row: pd.Series, relative_strength: dict) -> dict:
    """Score daily semi-auto trend eligibility out of 10."""
    components = {}
    explanations = []

    if pd.notna(row.get("sma200")) and row["close"] > row["sma200"]:
        components["long_term_trend"] = 2
        explanations.append("Close is above the 200-day SMA.")
    else:
        components["long_term_trend"] = 0
        explanations.append("Close is not above the 200-day SMA.")

    if pd.notna(row.get("ema20")) and pd.notna(row.get("sma50")) and row["ema20"] > row["sma50"]:
        components["intermediate_trend"] = 2
        explanations.append("20 EMA is above the 50 SMA.")
    else:
        components["intermediate_trend"] = 0
        explanations.append("20 EMA is not above the 50 SMA.")

    rs_3m = relative_strength.get("3m", {}).get("relative_return")
    rs_6m = relative_strength.get("6m", {}).get("relative_return")
    if (rs_3m is not None and rs_3m > 0) or (rs_6m is not None and rs_6m > 0):
        components["relative_strength"] = 2
        explanations.append("3-month or 6-month relative strength is above the benchmark.")
    else:
        components["relative_strength"] = 0
        explanations.append("Relative strength is not above the benchmark on the 3-month or 6-month window.")

    if pd.notna(row.get("close")) and pd.notna(row.get("close_21")) and row["close"] > row["close_21"]:
        components["momentum_persistence"] = 1
        explanations.append("Close is above the close from roughly one month ago.")
    else:
        components["momentum_persistence"] = 0
        explanations.append("Close is not above the close from roughly one month ago.")

    if (
        pd.notna(row.get("adx14"))
        and pd.notna(row.get("plus_di14"))
        and pd.notna(row.get("minus_di14"))
        and row["adx14"] > 20
        and row["plus_di14"] > row["minus_di14"]
    ):
        components["adx_di_filter"] = 1
        explanations.append("ADX is above 20 and +DI is above -DI.")
    else:
        components["adx_di_filter"] = 0
        explanations.append("ADX/+DI trend-quality filter is not confirmed.")

    if (
        pd.notna(row.get("average_volume_20"))
        and pd.notna(row.get("average_volume_20_change_20"))
        and row["average_volume_20_change_20"] > 0
    ) or (pd.notna(row.get("volume_ratio_20")) and row["volume_ratio_20"] > 1.2):
        components["volume_trend"] = 1
        explanations.append("20-day average volume is rising or current volume is above average.")
    else:
        components["volume_trend"] = 0
        explanations.append("Volume trend is not confirming.")

    atr_pct = row.get("atr_pct")
    if pd.notna(atr_pct) and 0.01 <= atr_pct <= 0.08:
        components["atr_tradability"] = 1
        explanations.append("ATR is large enough to trade but not extreme.")
    else:
        components["atr_tradability"] = 0
        explanations.append("ATR is either too small or too extreme for the default rules.")

    score = int(sum(components.values()))
    if score >= 8:
        interpretation = "Strong candidate"
    elif score >= 6:
        interpretation = "Watchlist only unless intraday setup is excellent"
    else:
        interpretation = "Avoid for bullish semi-auto entries"

    return {
        "score": score,
        "max_score": 10,
        "components": components,
        "interpretation": interpretation,
        "features": _latest_feature_dict(row),
        "relative_strength": relative_strength,
        "explanations": explanations,
    }


def _score_semi_auto_entry(row: pd.Series, previous_row: pd.Series | None = None) -> dict:
    """Score pullback/reclaim entry trigger out of 10."""
    previous_row = previous_row if previous_row is not None else pd.Series(dtype=float)
    components = {}
    explanations = []

    rsi2 = row.get("rsi2")
    rsi4 = row.get("rsi4")
    if (pd.notna(rsi2) and rsi2 < 10) or (pd.notna(rsi4) and rsi4 < 30):
        components["pullback_entry"] = 2
        explanations.append("RSI(2) or RSI(4) is in pullback territory inside the uptrend.")
    else:
        components["pullback_entry"] = 0
        explanations.append("Short RSI pullback trigger is not active.")

    previous_rsi2 = previous_row.get("rsi2")
    previous_rsi4 = previous_row.get("rsi4")
    if (
        pd.notna(rsi2)
        and pd.notna(previous_rsi2)
        and rsi2 > previous_rsi2
    ) or (
        pd.notna(rsi4)
        and pd.notna(previous_rsi4)
        and rsi4 > previous_rsi4
    ) or (
        pd.notna(previous_rsi4)
        and pd.notna(rsi4)
        and previous_rsi4 < 30
        and rsi4 >= 30
    ):
        components["reversal_confirmation"] = 1
        explanations.append("Short RSI is turning back up or reclaiming the pullback threshold.")
    else:
        components["reversal_confirmation"] = 0
        explanations.append("Short RSI has not confirmed reversal yet.")

    close = row.get("close")
    ema9 = row.get("ema9")
    ema20 = row.get("ema20")
    previous_close = previous_row.get("close")
    previous_ema9 = previous_row.get("ema9")
    previous_ema20 = previous_row.get("ema20")
    reclaim_ema9 = bool(pd.notna(previous_close) and pd.notna(previous_ema9) and previous_close <= previous_ema9 and pd.notna(close) and pd.notna(ema9) and close > ema9)
    reclaim_ema20 = bool(pd.notna(previous_close) and pd.notna(previous_ema20) and previous_close <= previous_ema20 and pd.notna(close) and pd.notna(ema20) and close > ema20)
    if reclaim_ema9 or reclaim_ema20 or (pd.notna(close) and pd.notna(ema9) and pd.notna(ema20) and close > ema9 and close > ema20):
        components["price_reclaim"] = 2
        explanations.append("Price reclaimed or is holding above the 9 EMA / 20 EMA.")
    else:
        components["price_reclaim"] = 0
        explanations.append("Price has not reclaimed the 9 EMA / 20 EMA.")

    vwap = row.get("vwap")
    previous_vwap = previous_row.get("vwap")
    reclaim_vwap = bool(pd.notna(previous_close) and pd.notna(previous_vwap) and previous_close <= previous_vwap and pd.notna(close) and pd.notna(vwap) and close > vwap)
    if reclaim_vwap or (pd.notna(close) and pd.notna(vwap) and close > vwap):
        components["vwap_confirmation"] = 2
        explanations.append("Intraday price is above or reclaiming VWAP.")
    else:
        components["vwap_confirmation"] = 0
        explanations.append("VWAP confirmation is missing.")

    if (
        pd.notna(row.get("macd"))
        and pd.notna(row.get("macd_signal"))
        and row["macd"] > row["macd_signal"]
    ) or (pd.notna(row.get("macd_histogram_change_3")) and row["macd_histogram_change_3"] > 0):
        components["macd_confirmation"] = 1
        explanations.append("Intraday MACD histogram is improving or bullish.")
    else:
        components["macd_confirmation"] = 0
        explanations.append("MACD is not confirming yet.")

    if bool(row.get("green_candle", False)) and pd.notna(row.get("volume_ratio_20")) and row["volume_ratio_20"] >= 1.0:
        components["volume_confirmation"] = 1
        explanations.append("Green reclaim candle has at least average relative volume.")
    else:
        components["volume_confirmation"] = 0
        explanations.append("Volume confirmation is missing.")

    atr = row.get("atr14")
    extended = bool(
        pd.notna(close)
        and pd.notna(atr)
        and atr > 0
        and (
            (pd.notna(vwap) and close > vwap + 0.75 * atr)
            or (pd.notna(ema20) and close > ema20 + 0.75 * atr)
        )
    )
    if not extended:
        components["atr_entry_quality"] = 1
        explanations.append("Entry is not more than 0.75 ATR above VWAP / 20 EMA.")
    else:
        components["atr_entry_quality"] = 0
        explanations.append("Entry is extended more than 0.75 ATR above VWAP / 20 EMA.")

    score = int(sum(components.values()))
    if score >= 8:
        action = "Valid entry"
    elif score >= 6:
        action = "Small starter only"
    else:
        action = "Wait"

    return {
        "score": score,
        "max_score": 10,
        "components": components,
        "interpretation": action,
        "features": _latest_feature_dict(row),
        "extended_by_atr": extended,
        "explanations": explanations,
    }


def _score_semi_auto_exit(row: pd.Series, previous_row: pd.Series | None = None) -> dict:
    """Score trend-following exit risk."""
    previous_row = previous_row if previous_row is not None else pd.Series(dtype=float)
    components = {}
    explanations = []

    close = row.get("close")
    ema20 = row.get("ema20")
    ema9 = row.get("ema9")
    previous_ema9 = previous_row.get("ema9")
    previous_ema20 = previous_row.get("ema20")
    if pd.notna(close) and pd.notna(ema20) and close < ema20:
        components["close_below_20ema"] = 2
        explanations.append("Close is below the 20 EMA.")
    else:
        components["close_below_20ema"] = 0
        explanations.append("Close remains above the 20 EMA.")

    if pd.notna(previous_ema9) and pd.notna(previous_ema20) and pd.notna(ema9) and pd.notna(ema20) and previous_ema9 >= previous_ema20 and ema9 < ema20:
        components["ema9_cross_below_20ema"] = 2
        explanations.append("9 EMA crossed below 20 EMA.")
    else:
        components["ema9_cross_below_20ema"] = 0
        explanations.append("9 EMA has not crossed below 20 EMA.")

    if "macd_histogram" in row.index and row.get("macd_histogram_3bar_sum") is not None and pd.notna(row.get("macd_histogram_3bar_sum")) and row["macd_histogram_3bar_sum"] < 0:
        components["macd_hist_negative"] = 1
        explanations.append("MACD histogram has been negative over recent bars.")
    else:
        components["macd_hist_negative"] = 0
        explanations.append("MACD histogram is not persistently negative.")

    if pd.notna(row.get("rsi14")) and row["rsi14"] < 50:
        components["rsi14_loses_50"] = 1
        explanations.append("RSI(14) lost 50.")
    else:
        components["rsi14_loses_50"] = 0
        explanations.append("RSI(14) remains above 50.")

    if pd.notna(row.get("plus_di14")) and pd.notna(row.get("minus_di14")) and row["plus_di14"] < row["minus_di14"]:
        components["di_cross_bearish"] = 2
        explanations.append("+DI crossed below -DI.")
    else:
        components["di_cross_bearish"] = 0
        explanations.append("+DI remains above -DI.")

    if pd.notna(row.get("adx14")) and pd.notna(row.get("adx14_slope")) and row["adx14"] > 25 and row["adx14_slope"] < 0:
        components["adx_falling_high"] = 1
        explanations.append("ADX is falling after a high reading.")
    else:
        components["adx_falling_high"] = 0
        explanations.append("ADX is not showing a high-reading rollover.")

    red_candle = bool(pd.notna(row.get("open")) and pd.notna(close) and close < row["open"])
    if red_candle and pd.notna(row.get("volume_ratio_20")) and row["volume_ratio_20"] >= 1.2:
        components["distribution_day"] = 1
        explanations.append("High-volume red candle / distribution day is present.")
    else:
        components["distribution_day"] = 0
        explanations.append("No high-volume red distribution day.")

    trailing_stop = row.get("trailing_stop_25atr")
    if pd.notna(close) and pd.notna(trailing_stop) and close < trailing_stop:
        components["trailing_stop_break"] = 2
        explanations.append("Close is below the 2.5 ATR trailing stop.")
    else:
        components["trailing_stop_break"] = 0
        explanations.append("2.5 ATR trailing stop is intact.")

    score = int(sum(components.values()))
    if score <= 3:
        action = "Hold"
    elif score <= 5:
        action = "Tighten stop"
    elif score <= 7:
        action = "Trim"
    else:
        action = "Exit"

    return {
        "score": score,
        "max_score": 12,
        "components": components,
        "interpretation": action,
        "features": _latest_feature_dict(row),
        "explanations": explanations,
    }


def generate_semi_auto_trend_score(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None = None,
    benchmark_df: pd.DataFrame | None = None,
    ticker: str = "",
    benchmark_ticker: str = "SPY",
) -> dict:
    """Generate the research-style long-only trend-following/pullback score."""
    daily_features = build_feature_frame(daily_df, ticker or "TICKER")
    daily_row = daily_features.iloc[-1]
    benchmark_close = benchmark_df["Close"] if benchmark_df is not None and not benchmark_df.empty else None
    relative_strength = _calculate_relative_strength(daily_df["Close"], benchmark_close)
    daily_score = _score_semi_auto_daily(daily_row, relative_strength)

    entry_score = {
        "score": 0,
        "max_score": 10,
        "components": {},
        "interpretation": "Intraday data unavailable",
        "features": {},
        "extended_by_atr": False,
        "explanations": ["Intraday data was not available, so entry timing could not be scored."],
    }
    if intraday_df is not None and not intraday_df.empty and len(intraday_df) >= 20:
        intraday_features = build_feature_frame(intraday_df, ticker or "TICKER")
        intraday_row = intraday_features.iloc[-1]
        previous_intraday_row = intraday_features.iloc[-2] if len(intraday_features) >= 2 else None
        entry_score = _score_semi_auto_entry(intraday_row, previous_row=previous_intraday_row)

    previous_daily_row = daily_features.iloc[-2] if len(daily_features) >= 2 else None
    exit_score = _score_semi_auto_exit(daily_row, previous_row=previous_daily_row)

    close_above_200 = bool(pd.notna(daily_row.get("sma200")) and daily_row["close"] > daily_row["sma200"])
    extended_above_20 = bool(
        pd.notna(daily_row.get("close"))
        and pd.notna(daily_row.get("ema20"))
        and pd.notna(daily_row.get("atr14"))
        and daily_row["atr14"] > 0
        and daily_row["close"] > daily_row["ema20"] + daily_row["atr14"]
    )

    semi_auto_signal = (
        daily_score["score"] >= 7
        and entry_score["score"] >= 7
        and close_above_200
        and not extended_above_20
        and not entry_score.get("extended_by_atr", False)
    )
    if semi_auto_signal:
        action = "Semi-auto long signal"
    elif daily_score["score"] >= 7 and entry_score["score"] < 7:
        action = "Eligible watchlist - wait for entry"
    elif exit_score["score"] >= 8:
        action = "Exit / avoid new entry"
    else:
        action = "No long setup"

    total_score = int(daily_score["score"] + entry_score["score"])

    return {
        "action": action,
        "total_score": total_score,
        "max_score": 20,
        "semi_auto_long_signal": semi_auto_signal,
        "daily": daily_score,
        "entry": entry_score,
        "exit": exit_score,
        "relative_strength": relative_strength,
        "rules": {
            "daily_score_at_least_7": daily_score["score"] >= 7,
            "entry_score_at_least_7": entry_score["score"] >= 7,
            "price_above_200sma": close_above_200,
            "not_more_than_1atr_above_20ema": not extended_above_20,
            "entry_not_more_than_075atr_extended": not entry_score.get("extended_by_atr", False),
        },
        "benchmark_ticker": benchmark_ticker,
    }


def _score_intraday_daily_bias(row: pd.Series, relative_strength: dict) -> dict:
    """Score the daily/market intraday-trading bias filter out of 6."""
    components = {}
    explanations = []

    if (
        pd.notna(row.get("ema20"))
        and pd.notna(row.get("sma50"))
        and row["close"] > row["ema20"]
        and row["close"] > row["sma50"]
    ):
        components["daily_trend"] = 2
        explanations.append("Price is above both the daily 20 EMA and 50 SMA.")
    else:
        components["daily_trend"] = 0
        explanations.append("Price is not above both the daily 20 EMA and 50 SMA.")

    rs_3m = relative_strength.get("3m", {}).get("relative_return")
    rs_6m = relative_strength.get("6m", {}).get("relative_return")
    if (rs_3m is not None and rs_3m > 0) or (rs_6m is not None and rs_6m > 0):
        components["relative_strength"] = 1
        explanations.append("Stock is outperforming the benchmark recently.")
    else:
        components["relative_strength"] = 0
        explanations.append("Recent relative strength is not above the benchmark.")

    if (
        pd.notna(row.get("macd"))
        and pd.notna(row.get("macd_signal"))
        and row["macd"] > row["macd_signal"]
    ) or (pd.notna(row.get("macd_histogram_change_3")) and row["macd_histogram_change_3"] > 0):
        components["daily_momentum"] = 1
        explanations.append("Daily MACD is positive or improving.")
    else:
        components["daily_momentum"] = 0
        explanations.append("Daily MACD is not improving.")

    if (
        pd.notna(row.get("plus_di14"))
        and pd.notna(row.get("minus_di14"))
        and row["plus_di14"] > row["minus_di14"]
    ) or (pd.notna(row.get("adx14")) and row["adx14"] > 20):
        components["trend_quality"] = 1
        explanations.append("+DI is above -DI or ADX is above 20.")
    else:
        components["trend_quality"] = 0
        explanations.append("Trend-quality filter is not confirmed.")

    extended = bool(
        pd.notna(row.get("close"))
        and pd.notna(row.get("ema20"))
        and pd.notna(row.get("atr14"))
        and row["atr14"] > 0
        and row["close"] > row["ema20"] + row["atr14"]
    )
    if not extended:
        components["not_extended"] = 1
        explanations.append("Daily price is not more than about 1 ATR above the 20 EMA.")
    else:
        components["not_extended"] = 0
        explanations.append("Daily price is more than about 1 ATR above the 20 EMA.")

    score = int(sum(components.values()))
    if score >= 5:
        interpretation = "Strong long bias"
    elif score >= 3:
        interpretation = "Acceptable, but needs excellent intraday setup"
    else:
        interpretation = "Avoid long unless pure scalp/reversal"

    return {
        "score": score,
        "max_score": 6,
        "components": components,
        "interpretation": interpretation,
        "features": _latest_feature_dict(row),
        "relative_strength": relative_strength,
        "explanations": explanations,
    }


def _score_intraday_execution(row: pd.Series, previous_row: pd.Series | None = None) -> dict:
    """Score intraday execution quality out of 14."""
    previous_row = previous_row if previous_row is not None else pd.Series(dtype=float)
    components = {}
    explanations = []

    close = row.get("close")
    vwap = row.get("vwap")
    previous_close = previous_row.get("close")
    previous_vwap = previous_row.get("vwap")
    above_vwap = bool(pd.notna(close) and pd.notna(vwap) and close > vwap)
    reclaim_vwap = bool(
        pd.notna(previous_close)
        and pd.notna(previous_vwap)
        and previous_close <= previous_vwap
        and above_vwap
    )
    if above_vwap or reclaim_vwap:
        components["vwap"] = 3
        explanations.append("Price is above VWAP or reclaiming VWAP and holding.")
    else:
        components["vwap"] = 0
        explanations.append("VWAP confirmation is missing.")

    ema9 = row.get("ema9")
    ema20 = row.get("ema20")
    ema9_slope = row.get("ema9_slope")
    ema20_slope = row.get("ema20_slope")
    ema_structure_ok = bool(
        pd.notna(ema9)
        and pd.notna(ema20)
        and ema9 > ema20
        and pd.notna(ema9_slope)
        and pd.notna(ema20_slope)
        and ema9_slope > 0
        and ema20_slope > 0
    )
    if ema_structure_ok:
        components["ema_structure"] = 2
        explanations.append("9 EMA is above 20 EMA and both are rising.")
    else:
        components["ema_structure"] = 0
        explanations.append("9/20 EMA structure is not fully bullish.")

    pullback_holds = bool(
        pd.notna(row.get("low"))
        and pd.notna(close)
        and (
            (pd.notna(vwap) and row["low"] <= vwap and close >= vwap)
            or (pd.notna(ema9) and row["low"] <= ema9 and close >= ema9)
            or (pd.notna(ema20) and row["low"] <= ema20 and close >= ema20)
        )
    )
    if pullback_holds:
        components["pullback_quality"] = 2
        explanations.append("Pullback held VWAP, 9 EMA, or 20 EMA.")
    else:
        components["pullback_quality"] = 0
        explanations.append("Pullback did not clearly hold VWAP/EMA structure.")

    rsi = row.get("rsi14")
    rsi_change = row.get("rsi14_change_3")
    if pd.notna(rsi) and 40 <= rsi <= 55 and pd.notna(rsi_change) and rsi_change > 0:
        components["rsi"] = 2
        explanations.append("RSI cooled into 40-55 and is turning up.")
    else:
        components["rsi"] = 0
        explanations.append("RSI is not in the preferred 40-55 turn-up zone.")

    if (
        pd.notna(row.get("macd"))
        and pd.notna(row.get("macd_signal"))
        and row["macd"] > row["macd_signal"]
    ) or (pd.notna(row.get("macd_histogram_change_3")) and row["macd_histogram_change_3"] > 0):
        components["macd"] = 2
        explanations.append("MACD histogram is improving or a bullish cross is active.")
    else:
        components["macd"] = 0
        explanations.append("MACD is not confirming.")

    if bool(row.get("green_candle", False)) and pd.notna(row.get("volume_ratio_20")) and row["volume_ratio_20"] >= 1.0:
        components["volume"] = 2
        explanations.append("Green reclaim/breakout candle has above-average volume.")
    else:
        components["volume"] = 0
        explanations.append("Volume confirmation is missing.")

    atr = row.get("atr14")
    extended = bool(
        pd.notna(close)
        and pd.notna(atr)
        and atr > 0
        and (
            (pd.notna(vwap) and close > vwap + 0.75 * atr)
            or (pd.notna(ema20) and close > ema20 + 0.75 * atr)
        )
    )
    if not extended:
        components["atr_location"] = 1
        explanations.append("Entry is not overly extended from VWAP/EMA.")
    else:
        components["atr_location"] = 0
        explanations.append("Entry is overly extended from VWAP/EMA by ATR.")

    score = int(sum(components.values()))
    if score >= 11:
        interpretation = "Strong intraday entry"
    elif score >= 8:
        interpretation = "Tradable, smaller size or tighter stop"
    else:
        interpretation = "Wait"

    return {
        "score": score,
        "max_score": 14,
        "components": components,
        "interpretation": interpretation,
        "features": _latest_feature_dict(row),
        "above_or_reclaiming_vwap": above_vwap or reclaim_vwap,
        "extended_by_atr": extended,
        "explanations": explanations,
    }


def generate_intraday_trading_score(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None = None,
    benchmark_df: pd.DataFrame | None = None,
    ticker: str = "",
    benchmark_ticker: str = "SPY",
) -> dict:
    """Generate the 6-point daily filter plus 14-point intraday execution score."""
    daily_features = build_feature_frame(daily_df, ticker or "TICKER")
    daily_row = daily_features.iloc[-1]
    benchmark_close = benchmark_df["Close"] if benchmark_df is not None and not benchmark_df.empty else None
    relative_strength = _calculate_relative_strength(daily_df["Close"], benchmark_close)
    daily_bias = _score_intraday_daily_bias(daily_row, relative_strength)

    intraday_entry = {
        "score": 0,
        "max_score": 14,
        "components": {},
        "interpretation": "Intraday data unavailable",
        "features": {},
        "above_or_reclaiming_vwap": False,
        "extended_by_atr": False,
        "explanations": ["Intraday data was not available, so execution could not be scored."],
    }
    if intraday_df is not None and not intraday_df.empty and len(intraday_df) >= 20:
        intraday_features = build_feature_frame(intraday_df, ticker or "TICKER")
        intraday_row = intraday_features.iloc[-1]
        previous_intraday_row = intraday_features.iloc[-2] if len(intraday_features) >= 2 else None
        intraday_entry = _score_intraday_execution(intraday_row, previous_intraday_row)

    total_score = int(daily_bias["score"] + intraday_entry["score"])
    if total_score >= 17:
        total_action = "Best long setup"
    elif total_score >= 14:
        total_action = "Acceptable long setup"
    elif total_score >= 11:
        total_action = "Watch only or small starter"
    else:
        total_action = "Pass"

    valid_signal = (
        daily_bias["score"] >= 3
        and intraday_entry["score"] >= 9
        and intraday_entry["above_or_reclaiming_vwap"]
        and not intraday_entry["extended_by_atr"]
    )

    if valid_signal and total_score >= 17:
        action = "Best long setup"
    elif valid_signal:
        action = "Valid intraday long"
    elif daily_bias["score"] < 3:
        action = "Pass - weak daily/market bias"
    elif intraday_entry["score"] < 9:
        action = "Wait - intraday setup not strong enough"
    elif not intraday_entry["above_or_reclaiming_vwap"]:
        action = "Wait - VWAP not confirmed"
    elif intraday_entry["extended_by_atr"]:
        action = "Pass - ATR extension"
    else:
        action = total_action

    return {
        "total_score": total_score,
        "max_score": 20,
        "score_band": total_action,
        "action": action,
        "intraday_long_signal": valid_signal,
        "daily": daily_bias,
        "entry": intraday_entry,
        "relative_strength": relative_strength,
        "rules": {
            "daily_score_at_least_3": daily_bias["score"] >= 3,
            "entry_score_at_least_9": intraday_entry["score"] >= 9,
            "vwap_confirmed": intraday_entry["above_or_reclaiming_vwap"],
            "not_extended_by_atr": not intraday_entry["extended_by_atr"],
        },
        "benchmark_ticker": benchmark_ticker,
    }


def _classify_bullish_total(total_score: int) -> str:
    """Map the final 20-point score into the requested bands."""
    if total_score >= 16:
        return "Strong setup"
    if total_score >= 13:
        return "Acceptable, not ideal"
    if total_score >= 10:
        return "Watchlist only or smaller size"
    return "Pass"


def generate_bullish_continuation_score(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None = None,
    nearest_resistance: float | None = None,
    ticker: str = "",
) -> dict:
    """Generate the requested two-stage bullish continuation checklist score."""
    daily_features = build_feature_frame(daily_df, ticker or "TICKER")
    daily_row = daily_features.iloc[-1]
    daily_score = _score_daily_bias(daily_row)

    intraday_score = {
        "score": 0,
        "max_score": 10,
        "components": {},
        "interpretation": "Intraday data unavailable",
        "features": {},
        "explanations": ["Intraday data was not available, so entry timing could not be scored."],
    }
    atr_quality = {
        "penalty": 0,
        "max_penalty": -2,
        "hard_stop": False,
        "checks": ["ATR quality could not be evaluated without intraday data."],
    }

    if intraday_df is not None and not intraday_df.empty and len(intraday_df) >= 20:
        intraday_features = build_feature_frame(intraday_df, ticker or "TICKER")
        intraday_row = intraday_features.iloc[-1]
        previous_row = intraday_features.iloc[-2] if len(intraday_features) >= 2 else None
        intraday_score = _score_intraday_entry(intraday_row, previous_row=previous_row)
        atr_quality = _score_atr_quality(intraday_row, nearest_resistance=nearest_resistance)

    total_score = int(daily_score["score"] + intraday_score["score"] + atr_quality["penalty"])
    total_score = max(0, total_score)

    recommended_action = _classify_bullish_total(total_score)
    if daily_score["score"] < 7:
        recommended_action = "Pass - daily score below 7"
    elif intraday_score["score"] < 7:
        recommended_action = "Wait - intraday score below 7"
    elif atr_quality["hard_stop"]:
        recommended_action = "Pass - ATR chasing filter triggered"

    return {
        "total_score": total_score,
        "max_score": 20,
        "score_band": _classify_bullish_total(total_score),
        "recommended_action": recommended_action,
        "daily": daily_score,
        "intraday": intraday_score,
        "atr_quality": atr_quality,
        "minimum_rules": {
            "daily_score_at_least_7": daily_score["score"] >= 7,
            "intraday_score_at_least_7": intraday_score["score"] >= 7,
            "not_chasing_by_atr": not atr_quality["hard_stop"],
        },
    }


def build_feature_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build the full rule-based feature frame."""
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    high = df["High"]
    low = df["Low"]
    open_price = df["Open"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    sma50 = close.rolling(window=50, min_periods=50).mean()
    sma200 = close.rolling(window=200, min_periods=200).mean()
    daily_return = close.pct_change()
    daily_vol = daily_return.rolling(window=20, min_periods=20).std()
    annual_factor = annualization_factor_for_ticker(ticker)
    annual_vol = daily_vol * np.sqrt(annual_factor)
    atr14 = calculate_atr_series(df, period=14)
    bollinger = calculate_bollinger_bands(close, period=20, standard_deviations=2.0)
    macd = calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
    stochastic = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    obv = calculate_obv(close, volume)
    adx = calculate_adx(high, low, close, period=14)
    vwap = calculate_vwap(high, low, close, volume)

    features = pd.DataFrame(index=df.index)
    features["open"] = open_price
    features["close"] = close
    features["high"] = high
    features["low"] = low
    features["volume"] = volume
    features["daily_return"] = daily_return
    features["5_period_return"] = close / close.shift(5) - 1
    features["20_period_return"] = close / close.shift(20) - 1
    features["close_21"] = close.shift(21)
    features["ema9"] = ema9
    features["ema9_slope"] = calculate_slope_series(ema9, lookback=3)
    features["ema20"] = ema20
    features["ema20_slope"] = calculate_slope_series(ema20, lookback=5)
    features["sma50"] = sma50
    features["sma50_slope"] = calculate_slope_series(sma50, lookback=5)
    features["sma200"] = sma200
    features["sma200_slope"] = calculate_slope_series(sma200, lookback=21)
    features["daily_vol"] = daily_vol
    features["annual_vol"] = annual_vol
    features["annual_vol_75th"] = _expanding_percentile(annual_vol, percentile=0.75, min_periods=20)
    features["annual_vol_80th"] = _expanding_percentile(annual_vol, percentile=VOLATILITY_ALERT_PERCENTILE, min_periods=20)
    features["annual_vol_percentile"] = _expanding_percentile_rank(annual_vol, min_periods=20)
    features["volatility_threshold_percentile"] = VOLATILITY_ALERT_PERCENTILE
    features["atr14"] = atr14
    features["atr_pct"] = atr14 / close.replace(0, np.nan)
    features["average_volume_20"] = volume.rolling(window=20, min_periods=20).mean()
    features["average_volume_20_change_20"] = features["average_volume_20"] / features["average_volume_20"].shift(20) - 1
    features["rsi14"] = calculate_rsi(close, period=14)
    features["rsi2"] = calculate_rsi(close, period=2)
    features["rsi4"] = calculate_rsi(close, period=4)
    features["bb_middle"] = bollinger["bb_middle"]
    features["bb_upper"] = bollinger["bb_upper"]
    features["bb_lower"] = bollinger["bb_lower"]
    features["bb_percent_b"] = bollinger["bb_percent_b"]
    features["bb_width"] = bollinger["bb_width"]
    features["macd"] = macd["macd"]
    features["macd_signal"] = macd["macd_signal"]
    features["macd_histogram"] = macd["macd_histogram"]
    features["macd_histogram_3bar_sum"] = macd["macd_histogram"].rolling(window=3, min_periods=3).sum()
    features["stoch_k"] = stochastic["stoch_k"]
    features["stoch_d"] = stochastic["stoch_d"]
    features["obv"] = obv
    features["obv_change_5"] = obv - obv.shift(5)
    features["adx14"] = adx["adx14"]
    features["adx14_slope"] = adx["adx14"] - adx["adx14"].shift(3)
    features["plus_di14"] = adx["plus_di14"]
    features["minus_di14"] = adx["minus_di14"]
    features["di_spread"] = adx["di_spread"]
    features["di_spread_change_3"] = adx["di_spread"] - adx["di_spread"].shift(3)
    features["vwap"] = vwap
    features["price_above_vwap"] = close > vwap
    features["rsi14_change_3"] = features["rsi14"] - features["rsi14"].shift(3)
    features["macd_histogram_change_3"] = features["macd_histogram"] - features["macd_histogram"].shift(3)
    features["volume_ratio_20"] = volume / features["average_volume_20"].replace(0, np.nan)
    features["green_candle"] = close > open_price
    features["highest_close_20"] = close.rolling(window=20, min_periods=20).max()
    features["trailing_stop_25atr"] = features["highest_close_20"] - 2.5 * atr14
    features["rolling_high_20"] = close.rolling(window=20, min_periods=20).max()
    features["rolling_low_20"] = close.rolling(window=20, min_periods=20).min()
    features["close_is_20_high"] = close >= features["rolling_high_20"]
    features["close_is_20_low"] = close <= features["rolling_low_20"]
    features["high_volatility"] = features["annual_vol_percentile"] >= VOLATILITY_ALERT_PERCENTILE
    return features


def detect_regime(row: pd.Series) -> dict:
    """Detect the current market regime from a fully prepared row."""
    sma50_available = pd.notna(row.get("sma50"))

    is_trending_up = (
        pd.notna(row.get("close"))
        and pd.notna(row.get("ema20"))
        and row["close"] > row["ema20"]
        and pd.notna(row.get("ema20_slope"))
        and row["ema20_slope"] > 0.01
        and (not sma50_available or row["close"] > row["sma50"])
        and pd.notna(row.get("20_period_return"))
        and row["20_period_return"] > 0.05
    )

    is_trending_down = (
        pd.notna(row.get("close"))
        and pd.notna(row.get("ema20"))
        and row["close"] < row["ema20"]
        and pd.notna(row.get("ema20_slope"))
        and row["ema20_slope"] < -0.01
        and (not sma50_available or row["close"] < row["sma50"])
        and pd.notna(row.get("20_period_return"))
        and row["20_period_return"] < -0.05
    )

    high_volatility = bool(pd.notna(row.get("annual_vol_percentile")) and row.get("high_volatility", False))

    is_ranging = (
        not is_trending_up
        and not is_trending_down
        and pd.notna(row.get("20_period_return"))
        and -0.05 <= row["20_period_return"] <= 0.05
        and bool(row.get("price_between_levels", False))
    )

    if is_trending_up:
        regime = TRENDING_UP
    elif is_trending_down:
        regime = TRENDING_DOWN
    elif is_ranging:
        regime = RANGING
    elif high_volatility:
        regime = HIGH_VOLATILITY
    else:
        regime = NEUTRAL

    if high_volatility and regime not in {HIGH_VOLATILITY, NEUTRAL}:
        label = f"{regime} + {HIGH_VOLATILITY}"
    else:
        label = regime

    return {
        "regime": regime,
        "label": label,
        "high_volatility": high_volatility,
        "is_trending_up": is_trending_up,
        "is_trending_down": is_trending_down,
        "is_ranging": is_ranging,
    }


def _normalize_nearest_levels(nearest_levels: dict | None) -> dict:
    """Provide a stable nearest-level structure."""
    nearest_levels = nearest_levels or {}
    return {
        "support": nearest_levels.get("support"),
        "support_strength": nearest_levels.get("support_strength", 0),
        "resistance": nearest_levels.get("resistance"),
        "resistance_strength": nearest_levels.get("resistance_strength", 0),
        "distance_to_support_atr": nearest_levels.get("distance_to_support_atr"),
        "distance_to_resistance_atr": nearest_levels.get("distance_to_resistance_atr"),
        "distance_to_support_pct": nearest_levels.get("distance_to_support_pct"),
        "distance_to_resistance_pct": nearest_levels.get("distance_to_resistance_pct"),
        "price_between_levels": nearest_levels.get("price_between_levels", False),
    }


def _append_level_context(row: pd.Series, nearest_levels: dict) -> pd.Series:
    """Attach nearest level metadata to the working row."""
    context = row.copy()
    for key, value in _normalize_nearest_levels(nearest_levels).items():
        context[key] = value
    return context


def _classify_signal(score: float) -> dict:
    """Map score to both a trade trigger and a more descriptive bias label."""
    if score >= 3:
        return {"trade_signal": "BUY", "bias": "BUY"}
    if score <= -3:
        return {"trade_signal": "SELL", "bias": "SELL"}
    if score >= 1:
        return {"trade_signal": "HOLD", "bias": "WATCH BULLISH"}
    if score <= -1:
        return {"trade_signal": "HOLD", "bias": "WATCH BEARISH"}
    return {"trade_signal": "HOLD", "bias": "NEUTRAL"}


def _pick_closer_level(current_price: float, candidates: list[tuple[str, float | None]], prefer_below: bool | None = None):
    """Pick the level closest to current price, optionally restricting side."""
    filtered = []
    for label, level in candidates:
        if level is None or pd.isna(level):
            continue
        level_value = float(level)
        if prefer_below is True and level_value >= current_price:
            continue
        if prefer_below is False and level_value <= current_price:
            continue
        filtered.append((label, level_value))

    if not filtered:
        fallback = []
        for label, level in candidates:
            if level is None or pd.isna(level):
                continue
            fallback.append((label, float(level)))
        filtered = fallback

    if not filtered:
        return None, None

    label, level_value = min(filtered, key=lambda item: abs(current_price - item[1]))
    return label, level_value


def _format_level(level: float | None) -> str:
    """Format a price level for display."""
    if level is None or pd.isna(level):
        return "N/A"
    return f"${level:,.2f}"


def _build_trade_setup(row: pd.Series, regime_label: str, current_bias: str, nearest_levels: dict) -> dict:
    """Create a rule-based trade setup interpretation payload."""
    close = float(row["close"])
    ema20 = None if pd.isna(row.get("ema20")) else float(row["ema20"])
    atr = None if pd.isna(row.get("atr14")) else float(row["atr14"])
    support = nearest_levels.get("support")
    resistance = nearest_levels.get("resistance")
    recent_high = None if pd.isna(row.get("rolling_high_20")) else float(row["rolling_high_20"])
    recent_low = None if pd.isna(row.get("rolling_low_20")) else float(row["rolling_low_20"])

    setup_direction = "neutral"
    if current_bias in {"BUY", "WATCH BULLISH"}:
        setup_direction = "bullish"
    elif current_bias in {"SELL", "WATCH BEARISH"}:
        setup_direction = "bearish"

    confirmation_label = "None"
    confirmation_level = None
    invalidation_label = "None"
    invalidation_level = None

    if setup_direction == "bullish":
        confirmation_label, confirmation_level = _pick_closer_level(
            close,
            [("Nearest Resistance", resistance), ("Recent 20-Bar High", recent_high)],
            prefer_below=False,
        )
        invalidation_label, invalidation_level = _pick_closer_level(
            close,
            [("Nearest Support", support), ("EMA20", ema20)],
            prefer_below=True,
        )
    elif setup_direction == "bearish":
        breakdown_level = None
        if support is not None:
            breakdown_level = float(support - 0.5 * atr) if atr and atr > 0 else float(support)

        confirmation_label, confirmation_level = _pick_closer_level(
            close,
            [("Nearest Support Breakdown", breakdown_level), ("Recent 20-Bar Low", recent_low)],
            prefer_below=True,
        )
        invalidation_label, invalidation_level = _pick_closer_level(
            close,
            [("Nearest Resistance", resistance), ("EMA20", ema20)],
            prefer_below=False,
        )

    potential_upside_pct = None
    if resistance is not None and close != 0:
        potential_upside_pct = float(resistance / close - 1)

    downside_risk_pct = None
    if support is not None and support != 0:
        downside_risk_pct = float(close / support - 1)

    risk_reward_ratio = None
    if potential_upside_pct is not None and downside_risk_pct is not None and downside_risk_pct > 0:
        risk_reward_ratio = float(potential_upside_pct / downside_risk_pct)

    if setup_direction == "bullish":
        narrative = (
            f"Price bias is bullish within a {regime_label.lower()} regime. "
            f"Nearest support is {_format_level(support)} and confirmation sits at {_format_level(confirmation_level)}. "
        )
        if resistance is not None:
            narrative += f"A move toward resistance around {_format_level(resistance)} remains the upside objective. "
        else:
            narrative += "No overhead resistance is visible, so the setup remains in price discovery mode. "
        narrative += "Avoid aggressive BUY unless price confirms strength."
    elif setup_direction == "bearish":
        narrative = (
            f"Price bias is bearish within a {regime_label.lower()} regime. "
            f"Nearest resistance is {_format_level(resistance)} and confirmation sits at {_format_level(confirmation_level)}. "
        )
        if support is not None:
            narrative += f"A break toward support around {_format_level(support)} keeps downside pressure active. "
        else:
            narrative += "There is no clear nearby support, so downside targets are less defined. "
        narrative += "Avoid aggressive SELL unless price confirms weakness."
    else:
        narrative = (
            f"Price is in a {regime_label.lower()} regime with a neutral bias. "
            f"Support is {_format_level(support)} and resistance is {_format_level(resistance)}. "
            "Wait for clearer confirmation before taking an aggressive directional view."
        )

    return {
        "market_regime": regime_label,
        "current_bias": current_bias,
        "confirmation_level": confirmation_level,
        "confirmation_label": confirmation_label,
        "invalidation_level": invalidation_level,
        "invalidation_label": invalidation_label,
        "nearest_support": support,
        "nearest_resistance": resistance,
        "potential_upside_pct": potential_upside_pct,
        "downside_risk_pct": downside_risk_pct,
        "risk_reward_ratio": risk_reward_ratio,
        "narrative": narrative,
    }


def evaluate_signal_row(row: pd.Series, nearest_levels: dict | None = None) -> dict:
    """Evaluate one row of features with explicit nearest support and resistance."""
    row = _append_level_context(row, nearest_levels or {})
    regime_info = detect_regime(row)
    regime = regime_info["regime"]
    high_volatility = regime_info["high_volatility"]

    atr = row.get("atr14")
    atr_is_valid = pd.notna(atr) and atr and atr > 0
    avg_volume = row.get("average_volume_20")
    avg_volume_is_valid = pd.notna(avg_volume) and avg_volume > 0

    support_distance_atr = row.get("distance_to_support_atr")
    resistance_distance_atr = row.get("distance_to_resistance_atr")
    strong_support_near = bool(
        atr_is_valid
        and row.get("support") is not None
        and row.get("support_strength", 0) >= STRONG_LEVEL_MIN_HITS
        and support_distance_atr is not None
        and 0 <= support_distance_atr <= 0.75
    )
    strong_resistance_near = bool(
        atr_is_valid
        and row.get("resistance") is not None
        and row.get("resistance_strength", 0) >= STRONG_LEVEL_MIN_HITS
        and resistance_distance_atr is not None
        and 0 <= resistance_distance_atr <= 0.75
    )
    support_near = bool(
        atr_is_valid
        and row.get("support") is not None
        and support_distance_atr is not None
        and 0 <= support_distance_atr <= 0.75
    )
    resistance_near = bool(
        atr_is_valid
        and row.get("resistance") is not None
        and resistance_distance_atr is not None
        and 0 <= resistance_distance_atr <= 0.75
    )

    breakout_threshold_atr = 1.0 if regime == RANGING else 0.5
    breakdown_threshold_atr = 1.0 if regime == RANGING else 0.5

    breakout_active = bool(
        atr_is_valid
        and row.get("resistance") is not None
        and row["close"] > row["resistance"] + breakout_threshold_atr * atr
    )
    breakdown_active = bool(
        atr_is_valid
        and row.get("support") is not None
        and row["close"] < row["support"] - breakdown_threshold_atr * atr
    )
    volume_confirmed = bool(
        avg_volume_is_valid and pd.notna(row.get("volume")) and row["volume"] > 1.2 * avg_volume
    )

    score_components = {
        "trend_position": 0,
        "ema_slope": 0,
        "recent_return": 0,
        "range_location": 0,
        "rsi": 0,
        "high_20_low_20": 0,
        "volatility": 0,
        "support_resistance": 0,
        "breakout": 0,
        "breakdown": 0,
        "vcp": 0,
    }
    explanations = []

    if regime == TRENDING_UP:
        if pd.notna(row.get("ema20")) and row["close"] > row["ema20"]:
            score_components["trend_position"] += 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] > 0:
            score_components["ema_slope"] += 1
        if pd.notna(row.get("5_period_return")) and row["5_period_return"] > 0:
            score_components["recent_return"] += 1
        if bool(row.get("close_is_20_high", False)):
            score_components["high_20_low_20"] += 2
        if high_volatility:
            score_components["volatility"] -= 1
        if strong_resistance_near:
            score_components["support_resistance"] -= 2

    elif regime == TRENDING_DOWN:
        if pd.notna(row.get("ema20")) and row["close"] < row["ema20"]:
            score_components["trend_position"] -= 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] < 0:
            score_components["ema_slope"] -= 1
        if pd.notna(row.get("5_period_return")) and row["5_period_return"] < 0:
            score_components["recent_return"] -= 1
        if bool(row.get("close_is_20_low", False)):
            score_components["high_20_low_20"] -= 2
        if high_volatility:
            score_components["volatility"] += 1
        if strong_support_near:
            score_components["support_resistance"] += 2

    elif regime == RANGING:
        if support_near:
            score_components["range_location"] += 2
        if resistance_near:
            score_components["range_location"] -= 2
        if pd.notna(row.get("rsi14")) and row["rsi14"] < 35:
            score_components["rsi"] += 1
        if pd.notna(row.get("rsi14")) and row["rsi14"] > 65:
            score_components["rsi"] -= 1

    if breakout_active:
        if volume_confirmed:
            score_components["breakout"] += 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] > 0:
            score_components["breakout"] += 1
        if bool(row.get("close_is_20_high", False)):
            score_components["breakout"] += 1

    if breakdown_active:
        if volume_confirmed:
            score_components["breakdown"] -= 2
        if pd.notna(row.get("ema20_slope")) and row["ema20_slope"] < 0:
            score_components["breakdown"] -= 1
        if bool(row.get("close_is_20_low", False)):
            score_components["breakdown"] -= 1

    raw_score = float(sum(score_components.values()))
    final_score = raw_score
    if high_volatility:
        final_score *= 0.7
        explanations.append("High volatility reduces signal reliability")

    rounded_score = round(final_score, 2)
    classified_signal = _classify_signal(rounded_score)
    trade_signal = classified_signal["trade_signal"]
    current_bias = classified_signal["bias"]

    signal_strength = min(95, 50 + abs(rounded_score) * 10)
    if high_volatility:
        signal_strength = min(signal_strength, 70)

    if regime == TRENDING_UP:
        explanations.insert(0, "Trending Up regime favors long trend-following setups")
    elif regime == TRENDING_DOWN:
        explanations.insert(0, "Trending Down regime favors defensive or short-biased setups")
    elif regime == RANGING:
        explanations.insert(0, "Ranging regime favors mean-reversion near support and resistance")
    elif regime == HIGH_VOLATILITY:
        explanations.insert(0, "High volatility regime is active without a strong directional trend")
    else:
        explanations.insert(0, "Neutral regime with mixed evidence")

    if score_components["breakout"] > 0:
        explanations.append("Resistance breakout conditions are active")
    if score_components["breakdown"] < 0:
        explanations.append("Support breakdown conditions are active")
    if score_components["support_resistance"] < 0:
        explanations.append("Price is close to strong resistance")
    if score_components["support_resistance"] > 0:
        explanations.append("Price is close to strong support")
    if score_components["range_location"] > 0:
        explanations.append("Price is trading close to support inside the range")
    if score_components["range_location"] < 0:
        explanations.append("Price is trading close to resistance inside the range")
    if score_components["rsi"] > 0:
        explanations.append("RSI is oversold for a range environment")
    if score_components["rsi"] < 0:
        explanations.append("RSI is overbought for a range environment")
    if score_components["high_20_low_20"] > 0:
        explanations.append("Price is printing a fresh 20-bar high")
    if score_components["high_20_low_20"] < 0:
        explanations.append("Price is printing a fresh 20-bar low")

    if score_components["breakout"] != 0 or score_components["breakdown"] != 0:
        signal_style = "Breakout"
    elif regime in {TRENDING_UP, TRENDING_DOWN}:
        signal_style = "Trend-Following"
    elif regime == RANGING and (score_components["range_location"] != 0 or score_components["rsi"] != 0):
        signal_style = "Mean-Reversion"
    else:
        signal_style = "Neutral"

    trade_setup = _build_trade_setup(row, regime_info["label"], current_bias, _normalize_nearest_levels(nearest_levels or {}))

    annual_vol = None if pd.isna(row.get("annual_vol")) else float(row["annual_vol"])
    annual_vol_75th = None if pd.isna(row.get("annual_vol_75th")) else float(row["annual_vol_75th"])
    annual_vol_80th = None if pd.isna(row.get("annual_vol_80th")) else float(row["annual_vol_80th"])
    annual_vol_percentile = None if pd.isna(row.get("annual_vol_percentile")) else float(row["annual_vol_percentile"])

    return {
        "signal": current_bias,
        "trade_signal": trade_signal,
        "current_bias": current_bias,
        "score": rounded_score,
        "raw_score": raw_score,
        "signal_strength": round(signal_strength, 1),
        "signal_strength_note": "Signal Strength is a rule-based score, not a calibrated probability.",
        "regime": regime,
        "regime_label": regime_info["label"],
        "high_volatility": high_volatility,
        "signal_style": signal_style,
        "score_components": score_components,
        "features": {
            "ema9": None if pd.isna(row.get("ema9")) else float(row["ema9"]),
            "ema9_slope": None if pd.isna(row.get("ema9_slope")) else float(row["ema9_slope"]),
            "ema20": None if pd.isna(row.get("ema20")) else float(row["ema20"]),
            "ema20_slope": None if pd.isna(row.get("ema20_slope")) else float(row["ema20_slope"]),
            "sma50": None if pd.isna(row.get("sma50")) else float(row["sma50"]),
            "sma50_slope": None if pd.isna(row.get("sma50_slope")) else float(row["sma50_slope"]),
            "rsi14": None if pd.isna(row.get("rsi14")) else float(row["rsi14"]),
            "bb_middle": None if pd.isna(row.get("bb_middle")) else float(row["bb_middle"]),
            "bb_upper": None if pd.isna(row.get("bb_upper")) else float(row["bb_upper"]),
            "bb_lower": None if pd.isna(row.get("bb_lower")) else float(row["bb_lower"]),
            "bb_percent_b": None if pd.isna(row.get("bb_percent_b")) else float(row["bb_percent_b"]),
            "bb_width": None if pd.isna(row.get("bb_width")) else float(row["bb_width"]),
            "macd": None if pd.isna(row.get("macd")) else float(row["macd"]),
            "macd_signal": None if pd.isna(row.get("macd_signal")) else float(row["macd_signal"]),
            "macd_histogram": None if pd.isna(row.get("macd_histogram")) else float(row["macd_histogram"]),
            "stoch_k": None if pd.isna(row.get("stoch_k")) else float(row["stoch_k"]),
            "stoch_d": None if pd.isna(row.get("stoch_d")) else float(row["stoch_d"]),
            "obv": None if pd.isna(row.get("obv")) else float(row["obv"]),
            "obv_change_5": None if pd.isna(row.get("obv_change_5")) else float(row["obv_change_5"]),
            "adx14": None if pd.isna(row.get("adx14")) else float(row["adx14"]),
            "adx14_slope": None if pd.isna(row.get("adx14_slope")) else float(row["adx14_slope"]),
            "plus_di14": None if pd.isna(row.get("plus_di14")) else float(row["plus_di14"]),
            "minus_di14": None if pd.isna(row.get("minus_di14")) else float(row["minus_di14"]),
            "di_spread": None if pd.isna(row.get("di_spread")) else float(row["di_spread"]),
            "di_spread_change_3": None if pd.isna(row.get("di_spread_change_3")) else float(row["di_spread_change_3"]),
            "vwap": None if pd.isna(row.get("vwap")) else float(row["vwap"]),
            "price_above_vwap": bool(row.get("price_above_vwap", False)),
            "rsi14_change_3": None if pd.isna(row.get("rsi14_change_3")) else float(row["rsi14_change_3"]),
            "macd_histogram_change_3": None if pd.isna(row.get("macd_histogram_change_3")) else float(row["macd_histogram_change_3"]),
            "volume_ratio_20": None if pd.isna(row.get("volume_ratio_20")) else float(row["volume_ratio_20"]),
            "5_period_return": None if pd.isna(row.get("5_period_return")) else float(row["5_period_return"]),
            "20_period_return": None if pd.isna(row.get("20_period_return")) else float(row["20_period_return"]),
            "annual_vol": annual_vol,
            "annual_vol_75th": annual_vol_75th,
            "annual_vol_80th": annual_vol_80th,
            "annual_vol_percentile": annual_vol_percentile,
            "volatility_threshold_percentile": VOLATILITY_ALERT_PERCENTILE,
            "atr14": None if pd.isna(row.get("atr14")) else float(row["atr14"]),
            "average_volume_20": None if pd.isna(row.get("average_volume_20")) else float(row["average_volume_20"]),
            "rolling_high_20": None if pd.isna(row.get("rolling_high_20")) else float(row["rolling_high_20"]),
            "rolling_low_20": None if pd.isna(row.get("rolling_low_20")) else float(row["rolling_low_20"]),
        },
        "nearest_levels": _normalize_nearest_levels(nearest_levels or {}),
        "trade_setup": trade_setup,
        "explanations": explanations,
    }


def generate_signal(
    df: pd.DataFrame,
    ticker: str,
    support_levels=None,
    resistance_levels=None,
    max_levels: int = 4,
    swing_sensitivity: int = 5,
) -> dict:
    """Generate the latest signal for a ticker using the full explainable ruleset."""
    features = build_feature_frame(df, ticker)
    latest_row = features.iloc[-1]

    if support_levels is None or resistance_levels is None:
        snapshot = build_level_snapshot(
            df,
            current_price=float(df["Close"].iloc[-1]),
            atr=latest_row.get("atr14"),
            max_levels=max_levels,
            sensitivity=swing_sensitivity,
            include_dynamic_levels=True,
        )
        support_levels = snapshot["actionable_supports"]
        resistance_levels = snapshot["actionable_resistances"]

    nearest_levels = get_nearest_levels(
        current_price=float(df["Close"].iloc[-1]),
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        atr=latest_row.get("atr14"),
    )

    result = evaluate_signal_row(latest_row, nearest_levels=nearest_levels)
    result["support_levels"] = support_levels
    result["resistance_levels"] = resistance_levels
    return result
