"""
Rule-based Volatility Contraction Pattern (VCP) detection.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from levels import identify_swing_points
from signals import build_feature_frame


def _safe_pct(numerator: float, denominator: float) -> float | None:
    """Return a simple percentage ratio if valid."""
    if denominator == 0 or denominator is None or pd.isna(denominator):
        return None
    return float(numerator / denominator - 1)


def _classify_signal(score: float) -> tuple[str, str]:
    """Map score into trade trigger and bias label."""
    if score >= 3:
        return "BUY", "BUY"
    if score <= -3:
        return "SELL", "SELL"
    if score >= 1:
        return "HOLD", "WATCH BULLISH"
    if score <= -1:
        return "HOLD", "WATCH BEARISH"
    return "HOLD", "NEUTRAL"


def _score_prior_uptrend(df: pd.DataFrame, feature_frame: pd.DataFrame) -> tuple[float, list[str]]:
    """Score prior uptrend quality up to 20 points."""
    latest = feature_frame.iloc[-1]
    close = float(latest["close"])
    score = 0.0
    bullets = []

    if pd.notna(latest.get("sma50")) and close > latest["sma50"]:
        score += 5
        bullets.append("Prior uptrend confirmed: price is above SMA50.")

    if pd.notna(latest.get("sma50_slope")) and latest["sma50_slope"] > 0:
        score += 5
        bullets.append("SMA50 is rising.")

    sma200 = df["Close"].rolling(window=200, min_periods=200).mean().iloc[-1]
    if pd.notna(sma200) and pd.notna(latest.get("sma50")) and latest["sma50"] > sma200:
        score += 5
        bullets.append("SMA50 is above SMA200.")

    fifty_two_week_high = float(df["High"].tail(252).max()) if len(df) >= 20 else float(df["High"].max())
    if fifty_two_week_high > 0 and close >= 0.75 * fifty_two_week_high:
        score += 5
        bullets.append("Price is within 25% of the 52-week high.")

    return score, bullets


def _extract_pullbacks(base_df: pd.DataFrame, swing_sensitivity: int) -> tuple[list[float], list[dict]]:
    """Extract pullback percentages from swing highs to following swing lows."""
    sensitivity = max(2, min(swing_sensitivity, max(2, len(base_df) // 6)))
    swing_highs, swing_lows = identify_swing_points(base_df, sensitivity=sensitivity)
    pullbacks = []
    annotations = []

    low_idx = 0
    for high in swing_highs:
        while low_idx < len(swing_lows) and swing_lows[low_idx]["date"] <= high["date"]:
            low_idx += 1
        if low_idx >= len(swing_lows):
            break

        low = swing_lows[low_idx]
        pullback_pct = low["price"] / high["price"] - 1
        pullbacks.append(float(pullback_pct))
        annotations.append({
            "high_date": high["date"],
            "high_price": high["price"],
            "low_date": low["date"],
            "low_price": low["price"],
            "pullback_pct": float(pullback_pct),
        })

    return pullbacks, annotations


def _score_pullback_contraction(pullbacks: list[float]) -> tuple[float, list[str]]:
    """Score whether pullbacks are contracting."""
    if len(pullbacks) < 2:
        return 0.0, ["Not enough pullbacks to confirm contraction."]

    score = 8.0
    contraction_count = 0

    for idx in range(1, len(pullbacks)):
        if abs(pullbacks[idx]) < abs(pullbacks[idx - 1]):
            contraction_count += 1

    if contraction_count >= 1:
        score += 9
    if contraction_count >= 2:
        score += 8

    if all(abs(pullbacks[idx]) <= abs(pullbacks[idx - 1]) * 1.15 for idx in range(1, len(pullbacks))):
        score = min(25.0, score + 4)

    bullets = [f"Pullbacks are contracting: {', '.join(f'{value * 100:.1f}%' for value in pullbacks[:3])}."]
    return min(score, 25.0), bullets


def _score_atr_contraction(base_features: pd.DataFrame) -> tuple[float, float | None, list[str]]:
    """Score ATR percentage contraction inside the base."""
    atr_pct = (base_features["atr14"] / base_features["close"]) * 100
    atr_pct = atr_pct.dropna()
    if atr_pct.empty:
        return 0.0, None, ["ATR contraction could not be measured."]

    start_atr_pct = float(atr_pct.iloc[0])
    current_atr_pct = float(atr_pct.iloc[-1])
    if start_atr_pct == 0:
        return 0.0, None, ["ATR contraction could not be measured."]

    ratio = float(current_atr_pct / start_atr_pct)
    if ratio < 0.60:
        score = 20.0
    elif ratio < 0.80:
        score = 15.0
    elif ratio < 1.00:
        score = 8.0
    else:
        score = 0.0

    return score, ratio, [f"ATR contraction ratio is {ratio:.2f}."]


def _score_volume_contraction(base_df: pd.DataFrame) -> tuple[float, float | None, list[str]]:
    """Score whether recent volume has dried up inside the base."""
    if "Volume" not in base_df.columns or len(base_df) < 20:
        return 0.0, None, ["Volume contraction could not be measured."]

    recent_window = min(10, len(base_df))
    prior_window = min(30, max(0, len(base_df) - recent_window))
    recent = base_df["Volume"].tail(recent_window)
    prior = base_df["Volume"].iloc[-(recent_window + prior_window):-recent_window] if prior_window > 0 else pd.Series(dtype=float)

    if prior.empty or prior.mean() == 0:
        return 0.0, None, ["Volume contraction could not be measured."]

    ratio = float(recent.mean() / prior.mean())
    if ratio < 0.60:
        score = 20.0
    elif ratio < 0.80:
        score = 15.0
    elif ratio < 1.00:
        score = 8.0
    else:
        score = 0.0

    return score, ratio, [f"Recent volume is {ratio * 100:.0f}% of prior volume."]


def _score_near_pivot(current_close: float, pivot: float) -> tuple[float, float | None, list[str]]:
    """Score how tightly price is trading near pivot."""
    distance_to_pivot_pct = _safe_pct(pivot, current_close)
    if distance_to_pivot_pct is None:
        return 0.0, None, ["Pivot distance could not be measured."]

    if current_close > pivot:
        score = 10.0
    elif 0 <= distance_to_pivot_pct <= 0.05:
        score = 10.0
    elif distance_to_pivot_pct <= 0.10:
        score = 6.0
    else:
        score = 0.0

    return score, float(distance_to_pivot_pct), [f"Price is {distance_to_pivot_pct * 100:.1f}% from the pivot."]


def _score_clean_base(base_df: pd.DataFrame, pivot: float, base_low: float, base_depth_pct: float) -> tuple[float, int, list[str]]:
    """Score how clean the base is, penalizing failed breakouts and erratic bars."""
    failed_breakouts = int(((base_df["High"] > pivot) & (base_df["Close"] < pivot)).sum())
    erratic_threshold = float((base_df["High"] - base_df["Low"]).median() * 2) if len(base_df) > 0 else 0.0
    erratic_bars = int(((base_df["High"] - base_df["Low"]) > erratic_threshold).sum()) if erratic_threshold > 0 else 0
    base_breaks = int((base_df["Close"] < base_low).sum())

    penalties = failed_breakouts + base_breaks + (1 if erratic_bars > max(2, len(base_df) // 10) else 0)
    if base_depth_pct < -0.35 or base_depth_pct > -0.03:
        penalties += 1

    score = max(0.0, 5.0 - penalties)
    bullets = [f"Base cleanliness penalty count: {penalties}."]
    return score, penalties, bullets


def _score_window(
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    window_length: int,
    swing_sensitivity: int,
    actionable_resistances: list[dict] | None = None,
) -> dict:
    """Score one candidate base window."""
    base_df = df.tail(window_length).copy()
    base_features = feature_frame.loc[base_df.index].copy()
    current_close = float(base_df["Close"].iloc[-1])
    current_volume = float(base_df["Volume"].iloc[-1]) if "Volume" in base_df.columns else 0.0

    base_high = float(base_df["High"].max())
    base_low = float(base_df["Low"].min())
    base_depth_pct = float(base_low / base_high - 1) if base_high else 0.0

    pivot = base_high
    pivot_source = "Base High"
    if actionable_resistances:
        top_resistances = [
            level for level in actionable_resistances
            if abs(level["level"] / base_high - 1) <= 0.03
        ]
        if top_resistances:
            strongest = max(top_resistances, key=lambda item: item.get("level_score", 0.0))
            pivot = float(strongest["level"])
            pivot_source = strongest["source_label"]

    prior_score, prior_bullets = _score_prior_uptrend(df, feature_frame)
    pullbacks, pullback_annotations = _extract_pullbacks(base_df, swing_sensitivity=swing_sensitivity)
    pullback_score, pullback_bullets = _score_pullback_contraction(pullbacks)
    atr_score, atr_contraction_ratio, atr_bullets = _score_atr_contraction(base_features)
    volume_score, volume_contraction_ratio, volume_bullets = _score_volume_contraction(base_df)
    near_pivot_score, distance_to_pivot_pct, pivot_bullets = _score_near_pivot(current_close, pivot)
    clean_score, clean_penalties, clean_bullets = _score_clean_base(base_df, pivot, base_low, base_depth_pct)

    average_volume_20 = base_features["average_volume_20"].iloc[-1] if "average_volume_20" in base_features else np.nan
    breakout_volume_requirement = float(1.5 * average_volume_20) if pd.notna(average_volume_20) else None
    breakout_volume_confirmed = bool(
        breakout_volume_requirement is not None
        and current_close > pivot
        and current_volume > breakout_volume_requirement
    )

    total_score = float(prior_score + pullback_score + atr_score + volume_score + near_pivot_score + clean_score)
    all_bullets = prior_bullets + pullback_bullets + atr_bullets + volume_bullets + pivot_bullets + clean_bullets

    return {
        "base_df": base_df,
        "base_features": base_features,
        "base_length": window_length,
        "base_high": base_high,
        "base_low": base_low,
        "base_depth_pct": base_depth_pct,
        "pivot": pivot,
        "pivot_source": pivot_source,
        "pullbacks": pullbacks,
        "pullback_annotations": pullback_annotations,
        "prior_uptrend_score": prior_score,
        "pullback_score": pullback_score,
        "atr_score": atr_score,
        "volume_score": volume_score,
        "near_pivot_score": near_pivot_score,
        "clean_score": clean_score,
        "total_score": total_score,
        "atr_contraction_ratio": atr_contraction_ratio,
        "volume_contraction_ratio": volume_contraction_ratio,
        "distance_to_pivot_pct": distance_to_pivot_pct,
        "breakout_trigger_price": pivot,
        "breakout_volume_requirement": breakout_volume_requirement,
        "breakout_volume_confirmed": breakout_volume_confirmed,
        "explanations": all_bullets,
        "clean_penalties": clean_penalties,
        "current_close": current_close,
        "volume_expanding_downward": bool(volume_contraction_ratio is not None and volume_contraction_ratio > 1.0 and current_close < pivot),
        "base_too_loose": bool(base_depth_pct < -0.35),
    }


def detect_vcp(
    df: pd.DataFrame,
    ticker: str,
    swing_sensitivity: int = 5,
    min_base_length: int = 20,
    max_base_length: int = 120,
    actionable_resistances: list[dict] | None = None,
) -> dict:
    """Detect a current/visual VCP setup using the selected visible data range."""
    if len(df) < min_base_length:
        return {
            "status": "No Clear VCP",
            "score": 0.0,
            "explanations": ["Not enough data to evaluate a VCP base."],
            "current_visual_analysis": True,
        }

    feature_frame = build_feature_frame(df, ticker)
    best_candidate = None

    for window_length in range(min_base_length, min(max_base_length, len(df)) + 1):
        candidate = _score_window(
            df,
            feature_frame,
            window_length=window_length,
            swing_sensitivity=swing_sensitivity,
            actionable_resistances=actionable_resistances,
        )

        if best_candidate is None or candidate["total_score"] > best_candidate["total_score"]:
            best_candidate = candidate

    result = best_candidate or {
        "total_score": 0.0,
        "explanations": ["No candidate base qualified."],
    }

    status = "No Clear VCP"
    if result.get("breakout_volume_confirmed", False):
        status = "VCP Breakout Confirmed"
    elif result.get("current_close") is not None and result.get("pivot") is not None and result["current_close"] > result["pivot"]:
        status = "Potential Breakout Without Volume Confirmation"
    elif result["total_score"] >= 75:
        status = "Strong VCP Candidate"
    elif result["total_score"] >= 60:
        status = "Possible VCP Candidate"

    failed_reasons = []
    if result.get("prior_uptrend_score", 0.0) < 10:
        failed_reasons.append("Prior uptrend is weak.")
    if result.get("pullback_score", 0.0) < 10:
        failed_reasons.append("Pullback contraction is weak or incomplete.")
    if result.get("atr_score", 0.0) < 8:
        failed_reasons.append("ATR contraction is weak.")
    if result.get("volume_score", 0.0) < 8:
        failed_reasons.append("Volume contraction is weak.")
    if result.get("near_pivot_score", 0.0) == 0:
        failed_reasons.append("Price is too far from pivot.")

    result.update({
        "status": status,
        "score": round(float(result["total_score"]), 2),
        "current_visual_analysis": True,
        "breakout_status": status == "VCP Breakout Confirmed",
        "failed_reasons": failed_reasons,
    })
    return result


def apply_vcp_to_signal(signal_result: dict, vcp_result: dict) -> dict:
    """Use VCP as a supporting factor without overriding the core signal engine."""
    adjusted = copy.deepcopy(signal_result)
    status = vcp_result.get("status", "No Clear VCP")
    score_delta = 0.0
    setup_type = "None"

    if status == "Strong VCP Candidate":
        score_delta = 1.0
        setup_type = "VCP Watch"
    elif status == "VCP Breakout Confirmed":
        score_delta = 2.0
        setup_type = "VCP Breakout Watch"

    if score_delta:
        adjusted["raw_score"] = float(adjusted.get("raw_score", adjusted["score"])) + score_delta
        adjusted["score"] = round(float(adjusted["score"]) + score_delta, 2)
        trade_signal, current_bias = _classify_signal(adjusted["score"])
        adjusted["trade_signal"] = trade_signal
        adjusted["signal"] = current_bias
        adjusted["current_bias"] = current_bias
        adjusted["signal_strength"] = round(min(70 if adjusted.get("high_volatility") else 95, 50 + abs(adjusted["score"]) * 10), 1)
        adjusted["score_components"]["vcp"] = score_delta
        adjusted["explanations"].append(f"Setup Type: {setup_type}")

    if vcp_result.get("volume_expanding_downward"):
        adjusted["explanations"].append("VCP caution: volume is expanding while price remains below pivot.")
    if vcp_result.get("base_too_loose"):
        adjusted["explanations"].append("VCP caution: base depth is loose for a classic contraction setup.")

    adjusted["explanations"].append("VCP is a setup condition, not a trade recommendation.")
    adjusted["vcp_status"] = status
    adjusted["setup_type"] = setup_type if setup_type != "None" else "Standard"
    return adjusted


def create_vcp_evidence_chart(
    vcp_result: dict,
    support_levels: list[dict] | None = None,
    resistance_levels: list[dict] | None = None,
    show_annotations: bool = True,
) -> go.Figure:
    """Create a chart explaining why a VCP candidate was flagged."""
    base_df = vcp_result["base_df"]
    base_features = vcp_result["base_features"]
    atr_pct = (base_features["atr14"] / base_features["close"]) * 100

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.20, 0.25],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=base_df.index,
            open=base_df["Open"],
            high=base_df["High"],
            low=base_df["Low"],
            close=base_df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_shape(
        type="rect",
        x0=base_df.index[0],
        x1=base_df.index[-1],
        y0=vcp_result["base_low"],
        y1=vcp_result["base_high"],
        fillcolor="rgba(59, 130, 246, 0.12)",
        line=dict(color="rgba(59, 130, 246, 0.35)", width=1),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=vcp_result["pivot"],
        line_dash="dash",
        line_color="gold",
        annotation_text=f"Pivot {vcp_result['pivot']:.2f}",
        row=1,
        col=1,
    )

    if show_annotations:
        for annotation in vcp_result.get("pullback_annotations", []):
            fig.add_trace(
                go.Scatter(
                    x=[annotation["high_date"], annotation["low_date"]],
                    y=[annotation["high_price"], annotation["low_price"]],
                    mode="markers+lines+text",
                    text=["", f"{annotation['pullback_pct'] * 100:.1f}%"],
                    textposition="bottom center",
                    marker=dict(size=8, color=["tomato", "limegreen"]),
                    line=dict(color="rgba(148, 163, 184, 0.6)", width=1),
                    name="Pullback",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    for level in (support_levels or []):
        if level["level"] >= vcp_result["base_low"] * 0.95 and level["level"] <= vcp_result["base_high"] * 1.05:
            fig.add_hrect(
                y0=level.get("zone_lower_bound", level["level"]),
                y1=level.get("zone_upper_bound", level["level"]),
                fillcolor="rgba(34, 197, 94, 0.12)",
                line_width=0,
                row=1,
                col=1,
            )

    for level in (resistance_levels or []):
        if level["level"] >= vcp_result["base_low"] * 0.95 and level["level"] <= vcp_result["base_high"] * 1.05:
            fig.add_hrect(
                y0=level.get("zone_lower_bound", level["level"]),
                y1=level.get("zone_upper_bound", level["level"]),
                fillcolor="rgba(239, 68, 68, 0.10)",
                line_width=0,
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=base_df.index,
            y=atr_pct,
            mode="lines",
            name="ATR14%",
            line=dict(color="royalblue", width=2),
        ),
        row=2,
        col=1,
    )

    if vcp_result.get("atr_contraction_ratio") is not None:
        fig.add_annotation(
            x=base_df.index[-1],
            y=float(atr_pct.iloc[-1]),
            text=f"ATR ratio {vcp_result['atr_contraction_ratio']:.2f}",
            showarrow=False,
            bgcolor="rgba(30, 41, 59, 0.8)",
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=base_df.index,
            y=base_df["Volume"] if "Volume" in base_df.columns else pd.Series(0.0, index=base_df.index),
            name="Volume",
            marker_color="gray",
            opacity=0.6,
        ),
        row=3,
        col=1,
    )

    recent_avg = None
    prior_avg = None
    if "Volume" in base_df.columns and len(base_df) >= 20:
        recent_avg = float(base_df["Volume"].tail(min(10, len(base_df))).mean())
        prior_slice = base_df["Volume"].iloc[-40:-10] if len(base_df) >= 40 else base_df["Volume"].iloc[:-10]
        if not prior_slice.empty:
            prior_avg = float(prior_slice.mean())
            fig.add_hline(y=recent_avg, line_color="green", line_dash="dot", row=3, col=1)
            fig.add_hline(y=prior_avg, line_color="orange", line_dash="dot", row=3, col=1)

    if vcp_result.get("breakout_volume_confirmed") and "Volume" in base_df.columns:
        fig.add_trace(
            go.Scatter(
                x=[base_df.index[-1]],
                y=[base_df["Volume"].iloc[-1]],
                mode="markers",
                marker=dict(color="gold", size=12, symbol="star"),
                name="Breakout Volume",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title="VCP Evidence Chart",
        template="plotly_dark",
        height=900,
        margin=dict(t=60, b=30, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(type="date", rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ATR14%", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    return fig
