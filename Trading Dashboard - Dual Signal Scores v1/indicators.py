"""
Reusable technical indicator calculations for the trading dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate true range from high, low, and close series."""
    previous_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR with Wilder-style smoothing."""
    true_range = calculate_true_range(high, low, close)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Calculate ADX, +DI, and -DI with Wilder-style smoothing."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr = calculate_atr(high, low, close, period=period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame(
        {
            "adx14": adx,
            "plus_di14": plus_di,
            "minus_di14": minus_di,
            "di_spread": plus_di - minus_di,
        },
        index=close.index,
    )


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate VWAP, resetting by session when the index is intraday datetime data."""
    typical_price = (high + low + close) / 3
    clean_volume = volume.fillna(0)

    if isinstance(close.index, pd.DatetimeIndex):
        session_key = close.index.date
        cumulative_volume = clean_volume.groupby(session_key).cumsum()
        cumulative_value = (typical_price * clean_volume).groupby(session_key).cumsum()
    else:
        cumulative_volume = clean_volume.cumsum()
        cumulative_value = (typical_price * clean_volume).cumsum()

    return cumulative_value / cumulative_volume.replace(0, np.nan)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate a simple rolling RSI."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    both_flat = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask(both_flat, 50)
    return rsi


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    standard_deviations: float = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Band middle, upper, lower, width, and percent-b."""
    middle = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    upper = middle + standard_deviations * rolling_std
    lower = middle - standard_deviations * rolling_std
    band_range = upper - lower

    percent_b = (close - lower) / band_range.replace(0, np.nan)
    width = band_range / middle.replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_percent_b": percent_b,
            "bb_width": width,
        },
        index=close.index,
    )


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculate MACD line, signal line, and histogram."""
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal

    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": signal,
            "macd_histogram": histogram,
        },
        index=close.index,
    )


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Calculate stochastic oscillator %K and %D."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    price_range = highest_high - lowest_low
    percent_k = 100 * (close - lowest_low) / price_range.replace(0, np.nan)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame(
        {
            "stoch_k": percent_k,
            "stoch_d": percent_d,
        },
        index=close.index,
    )


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate on-balance volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()
