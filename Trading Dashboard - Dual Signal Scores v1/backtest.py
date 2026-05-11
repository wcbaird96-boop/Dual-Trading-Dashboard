"""
Historical backtesting for the explainable regime-based signal engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals import build_feature_frame, evaluate_signal_row
from levels import build_level_snapshot
from vcp import apply_vcp_to_signal, detect_vcp


def _empty_backtest_result() -> dict:
    """Return a consistent empty backtest payload."""
    return {
        "total_signals": 0,
        "buy_signals": 0,
        "sell_signals": 0,
        "buy_win_rate": 0.0,
        "sell_win_rate": 0.0,
        "average_forward_return_5": 0.0,
        "average_forward_return_10": 0.0,
        "average_forward_return_20": 0.0,
        "median_forward_return_5": 0.0,
        "median_forward_return_10": 0.0,
        "median_forward_return_20": 0.0,
        "max_drawdown": 0.0,
        "buy_and_hold_return": 0.0,
        "start_price": 0.0,
        "end_price": 0.0,
        "bars_used": 0,
        "signal_history": pd.DataFrame(),
    }


def backtest_signals_no_lookahead(
    df: pd.DataFrame,
    ticker: str,
    max_levels: int = 4,
    lookback_periods: tuple[int, ...] = (5, 10, 20),
    relevance_atr: float = 5.0,
    relevance_pct: float = 0.25,
    swing_sensitivity: int = 5,
    enable_vcp_detection: bool = False,
    vcp_min_base_length: int = 20,
    vcp_max_base_length: int = 120,
) -> dict:
    """
    Backtest signals using only information available up to each date.
    """
    required_history = max(55, max(lookback_periods) + 1)
    if len(df) < required_history:
        return _empty_backtest_result()

    feature_frame = build_feature_frame(df, ticker)
    close = df["Close"]

    directional_returns = {period: [] for period in lookback_periods}
    buy_returns = {period: [] for period in lookback_periods}
    sell_returns = {period: [] for period in lookback_periods}

    history_rows = []
    active_position = 0.0

    for idx in range(required_history - 1, len(df)):
        hist_df = df.iloc[: idx + 1]
        row = feature_frame.iloc[idx]
        level_snapshot = build_level_snapshot(
            hist_df,
            current_price=float(close.iloc[idx]),
            atr=row.get("atr14"),
            max_levels=max_levels,
            relevance_atr=relevance_atr,
            relevance_pct=relevance_pct,
            show_all_historical=False,
            sensitivity=swing_sensitivity,
            include_dynamic_levels=True,
        )
        nearest_levels = level_snapshot["nearest_actionable_levels"]

        signal_result = evaluate_signal_row(row, nearest_levels=nearest_levels)
        if enable_vcp_detection:
            vcp_result = detect_vcp(
                hist_df,
                ticker=ticker,
                swing_sensitivity=swing_sensitivity,
                min_base_length=vcp_min_base_length,
                max_base_length=vcp_max_base_length,
                actionable_resistances=level_snapshot["actionable_resistances"],
            )
            signal_result = apply_vcp_to_signal(signal_result, vcp_result)
        signal = signal_result["trade_signal"]

        if signal == "BUY":
            active_position = 1.0
        elif signal == "SELL":
            active_position = -1.0

        history_row = {
            "date": df.index[idx],
            "close": float(close.iloc[idx]),
            "signal": signal,
            "bias": signal_result["current_bias"],
            "score": float(signal_result["score"]),
            "regime": signal_result["regime_label"],
            "signal_style": signal_result["signal_style"],
            "position": active_position,
        }

        for period in lookback_periods:
            forward_value = np.nan
            if idx + period < len(df):
                raw_forward_return = close.iloc[idx + period] / close.iloc[idx] - 1
                if signal == "BUY":
                    forward_value = raw_forward_return * 100
                    directional_returns[period].append(forward_value)
                    buy_returns[period].append(forward_value)
                elif signal == "SELL":
                    forward_value = -raw_forward_return * 100
                    directional_returns[period].append(forward_value)
                    sell_returns[period].append(forward_value)
            history_row[f"forward_return_{period}"] = forward_value

        history_rows.append(history_row)

    signal_history = pd.DataFrame(history_rows).set_index("date")

    buy_signal_count = int((signal_history["signal"] == "BUY").sum())
    sell_signal_count = int((signal_history["signal"] == "SELL").sum())
    total_signal_count = buy_signal_count + sell_signal_count

    buy_win_rate = 0.0
    if buy_returns[lookback_periods[0]]:
        buy_win_rate = 100 * sum(value > 0 for value in buy_returns[lookback_periods[0]]) / len(buy_returns[lookback_periods[0]])

    sell_win_rate = 0.0
    if sell_returns[lookback_periods[0]]:
        sell_win_rate = 100 * sum(value > 0 for value in sell_returns[lookback_periods[0]]) / len(sell_returns[lookback_periods[0]])

    close_returns = close.pct_change().fillna(0.0)
    aligned_positions = signal_history["position"].reindex(df.index).ffill().fillna(0.0)
    strategy_returns = aligned_positions.shift(1).fillna(0.0) * close_returns
    strategy_equity = (1 + strategy_returns).cumprod()
    strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1
    max_drawdown = float(strategy_drawdown.min() * 100) if not strategy_drawdown.empty else 0.0

    start_close = float(close.iloc[0])
    end_close = float(close.iloc[-1])
    buy_and_hold_return = 0.0
    if start_close != 0:
        buy_and_hold_return = float((end_close / start_close - 1) * 100)

    return {
        "total_signals": total_signal_count,
        "buy_signals": buy_signal_count,
        "sell_signals": sell_signal_count,
        "buy_win_rate": float(buy_win_rate),
        "sell_win_rate": float(sell_win_rate),
        "average_forward_return_5": float(np.mean(directional_returns[5])) if directional_returns[5] else 0.0,
        "average_forward_return_10": float(np.mean(directional_returns[10])) if directional_returns[10] else 0.0,
        "average_forward_return_20": float(np.mean(directional_returns[20])) if directional_returns[20] else 0.0,
        "median_forward_return_5": float(np.median(directional_returns[5])) if directional_returns[5] else 0.0,
        "median_forward_return_10": float(np.median(directional_returns[10])) if directional_returns[10] else 0.0,
        "median_forward_return_20": float(np.median(directional_returns[20])) if directional_returns[20] else 0.0,
        "max_drawdown": max_drawdown,
        "buy_and_hold_return": buy_and_hold_return,
        "start_price": start_close,
        "end_price": end_close,
        "bars_used": int(len(close)),
        "signal_history": signal_history,
    }
