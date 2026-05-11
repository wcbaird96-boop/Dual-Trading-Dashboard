"""
Support and resistance utilities for the trading dashboard.
"""

from __future__ import annotations

from collections import Counter
from statistics import median

import numpy as np
import pandas as pd


RECENT_SUPPORT_WINDOWS = (20, 50, 100)


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate the true range series."""
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the ATR series using a simple rolling mean."""
    true_range = calculate_true_range(df)
    return true_range.rolling(window=period, min_periods=period).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value."""
    atr_series = calculate_atr_series(df, period=period)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return 0.0
    return float(atr_series.iloc[-1])


def calculate_atr_percent(atr: float | None, close: float) -> float:
    """Calculate ATR as a percentage of price."""
    if atr is None or pd.isna(atr) or close == 0:
        return 0.0
    return float((atr / close) * 100)


def calculate_cluster_tolerance_pct(atr: float | None, close: float) -> float:
    """Calculate ATR-normalized clustering tolerance."""
    atr_pct = calculate_atr_percent(atr, close)
    tolerance = max(0.5 * atr_pct, 0.75)
    return float(min(tolerance, 5.0))


def calculate_actionable_threshold_pct(atr: float | None, close: float) -> float:
    """Calculate ATR-normalized actionable distance threshold."""
    if atr is None or pd.isna(atr) or close == 0:
        return 25.0
    return float(min((5 * atr / close) * 100, 25.0))


def calculate_zone_width(price: float, atr: float, price_pct: float = 0.01) -> float:
    """Calculate display zone width based on ATR and price percentage."""
    atr_value = 0.0 if pd.isna(atr) else atr
    atr_based = 0.5 * atr_value
    pct_based = price_pct * price
    return max(atr_based, pct_based)


def create_support_zone(support_level: float, atr: float, strength_hits: int = 1) -> dict:
    """Create a support zone as a band around the support level."""
    zone_width = calculate_zone_width(support_level, atr)
    adjusted_width = zone_width * (1.0 + 0.1 * strength_hits)
    return {
        "level": support_level,
        "lower": support_level - adjusted_width,
        "upper": support_level + adjusted_width / 2,
        "width": adjusted_width,
        "type": "support",
        "strength": strength_hits,
    }


def create_resistance_zone(resistance_level: float, atr: float, strength_hits: int = 1) -> dict:
    """Create a resistance zone as a band around the resistance level."""
    zone_width = calculate_zone_width(resistance_level, atr)
    adjusted_width = zone_width * (1.0 + 0.1 * strength_hits)
    return {
        "level": resistance_level,
        "lower": resistance_level - adjusted_width / 2,
        "upper": resistance_level + adjusted_width,
        "width": adjusted_width,
        "type": "resistance",
        "strength": strength_hits,
    }


def create_display_zone(level: dict, atr: float) -> dict:
    """Create a chartable zone from a level dictionary."""
    if level.get("dynamic_level", False):
        if level["type"] == "Support":
            return create_support_zone(level["level"], atr, strength_hits=max(1, int(level.get("touches", 1))))
        return create_resistance_zone(level["level"], atr, strength_hits=max(1, int(level.get("touches", 1))))

    return {
        "level": float(level["level"]),
        "lower": float(level.get("zone_lower_bound", level["level"])),
        "upper": float(level.get("zone_upper_bound", level["level"])),
        "width": float(level.get("zone_upper_bound", level["level"]) - level.get("zone_lower_bound", level["level"])),
        "type": level["type"].lower(),
        "strength": int(level.get("touches", 1)),
    }


def is_price_in_zone(price: float, zone: dict) -> bool:
    """Check whether a price is inside a zone."""
    return zone["lower"] <= price <= zone["upper"]


def distance_to_zone(price: float, zone: dict) -> float:
    """Calculate distance from the price to a zone."""
    if is_price_in_zone(price, zone):
        return 0.0
    if price > zone["upper"]:
        return price - zone["upper"]
    return zone["lower"] - price


def distance_to_zone_pct(price: float, zone: dict) -> float:
    """Calculate distance to a zone as a percent of price."""
    if price == 0:
        return 0.0
    return float(distance_to_zone(price, zone) / price)


def identify_swing_points(df: pd.DataFrame, sensitivity: int = 5):
    """Identify swing highs and swing lows using N bars on each side."""
    highs = []
    lows = []
    high_values = df["High"]
    low_values = df["Low"]
    volume_values = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    for idx in range(sensitivity, len(df) - sensitivity):
        current_high = high_values.iloc[idx]
        current_low = low_values.iloc[idx]

        left_highs = high_values.iloc[idx - sensitivity:idx]
        right_highs = high_values.iloc[idx + 1:idx + sensitivity + 1]
        left_lows = low_values.iloc[idx - sensitivity:idx]
        right_lows = low_values.iloc[idx + 1:idx + sensitivity + 1]

        if current_high > left_highs.max() and current_high > right_highs.max():
            highs.append({
                "date": df.index[idx],
                "price": float(current_high),
                "volume": float(volume_values.iloc[idx]),
                "source": "Swing High",
                "candidate_type": "resistance",
            })

        if current_low < left_lows.min() and current_low < right_lows.min():
            lows.append({
                "date": df.index[idx],
                "price": float(current_low),
                "volume": float(volume_values.iloc[idx]),
                "source": "Swing Low",
                "candidate_type": "support",
            })

    return highs, lows


def cluster_swing_points(points: list[dict], tolerance_pct: float) -> list[dict]:
    """Cluster nearby swing points into zones using ATR-normalized tolerance."""
    if not points:
        return []

    sorted_points = sorted(points, key=lambda item: item["price"])
    clusters = []

    for point in sorted_points:
        matched_cluster = None
        for cluster in clusters:
            cluster_mid = float(np.median(cluster["prices"]))
            if cluster_mid == 0:
                continue
            distance_pct = abs(point["price"] / cluster_mid - 1) * 100
            if distance_pct <= tolerance_pct:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            matched_cluster = {
                "prices": [],
                "dates": [],
                "volumes": [],
                "candidate_types": [],
                "sources": [],
            }
            clusters.append(matched_cluster)

        matched_cluster["prices"].append(float(point["price"]))
        matched_cluster["dates"].append(point["date"])
        matched_cluster["volumes"].append(float(point["volume"]))
        matched_cluster["candidate_types"].append(point["candidate_type"])
        matched_cluster["sources"].append(point["source"])

    return clusters


def _calculate_violation_penalty(df: pd.DataFrame, zone_mid: float, atr: float | None) -> int:
    """Count rough zone violations when price slices through the zone repeatedly."""
    close = df["Close"]
    if close.empty:
        return 0

    if atr is None or pd.isna(atr) or atr <= 0:
        tolerance = zone_mid * 0.005
    else:
        tolerance = max(0.25 * atr, zone_mid * 0.0025)

    relative_position = np.where(close > zone_mid + tolerance, 1, np.where(close < zone_mid - tolerance, -1, 0))
    sign_changes = 0
    previous = relative_position[0]
    for current in relative_position[1:]:
        if current != 0 and previous != 0 and current != previous:
            sign_changes += 1
        if current != 0:
            previous = current
    return int(max(0, sign_changes - 1))


def _score_cluster(df: pd.DataFrame, cluster: dict, overall_average_volume: float, atr: float | None) -> dict:
    """Score a clustered zone using touches, recency, volume, and violations."""
    prices = cluster["prices"]
    dates = cluster["dates"]
    volumes = cluster["volumes"]

    zone_mid = float(median(prices))
    zone_lower_bound = float(min(prices))
    zone_upper_bound = float(max(prices))
    touches = int(len(prices))
    most_recent_touch_date = max(dates)
    bars_since_touch = int(len(df.loc[df.index > most_recent_touch_date]))
    recency_score = float(1 / (1 + bars_since_touch / 20)) if bars_since_touch >= 0 else 0.0

    average_touch_volume = float(np.mean(volumes)) if volumes else 0.0
    if overall_average_volume > 0:
        volume_score = float(min(3.0, average_touch_volume / overall_average_volume))
    else:
        volume_score = 0.0

    violation_penalty = _calculate_violation_penalty(df, zone_mid, atr)
    level_score = 2.0 * touches + 1.5 * recency_score + 1.0 * volume_score - 1.0 * violation_penalty

    candidate_type_counts = Counter(cluster["candidate_types"])
    zone_type_candidate = candidate_type_counts.most_common(1)[0][0]

    return {
        "level": zone_mid,
        "zone_mid": zone_mid,
        "zone_lower_bound": zone_lower_bound,
        "zone_upper_bound": zone_upper_bound,
        "touches": touches,
        "hits": touches,
        "most_recent_touch_date": most_recent_touch_date,
        "recency_score": recency_score,
        "volume_score": volume_score,
        "violation_penalty": float(violation_penalty),
        "level_score": round(level_score, 2),
        "zone_type_candidate": zone_type_candidate,
        "source_label": "Swing Zone",
        "dynamic_level": False,
    }


def build_clustered_zones(
    df: pd.DataFrame,
    sensitivity: int = 5,
    cluster_tolerance_pct: float | None = None,
) -> dict:
    """Build clustered swing-based zones from visible data."""
    atr = calculate_atr(df, period=14)
    current_close = float(df["Close"].iloc[-1]) if not df.empty else 0.0
    if cluster_tolerance_pct is None:
        cluster_tolerance_pct = calculate_cluster_tolerance_pct(atr, current_close)

    swing_highs, swing_lows = identify_swing_points(df, sensitivity=sensitivity)
    overall_average_volume = float(df["Volume"].mean()) if "Volume" in df.columns and not df["Volume"].empty else 0.0

    support_clusters = cluster_swing_points(swing_lows, cluster_tolerance_pct)
    resistance_clusters = cluster_swing_points(swing_highs, cluster_tolerance_pct)

    support_zones = [_score_cluster(df, cluster, overall_average_volume, atr) for cluster in support_clusters]
    resistance_zones = [_score_cluster(df, cluster, overall_average_volume, atr) for cluster in resistance_clusters]

    return {
        "support_zones": support_zones,
        "resistance_zones": resistance_zones,
        "cluster_tolerance_pct": cluster_tolerance_pct,
        "atr": atr,
    }


def _build_recent_support_candidates(df: pd.DataFrame) -> list[dict]:
    """Build recent support candidates from the lowest lows over key lookback windows."""
    candidates = []
    low = df["Low"]

    for window in RECENT_SUPPORT_WINDOWS:
        if len(df) >= window:
            level = float(low.tail(window).min())
            candidates.append({
                "level": level,
                "zone_mid": level,
                "zone_lower_bound": level,
                "zone_upper_bound": level,
                "touches": 1,
                "hits": 1,
                "most_recent_touch_date": df.index[-1],
                "recency_score": 1.0,
                "volume_score": 0.0,
                "violation_penalty": 0.0,
                "level_score": 3.5,
                "zone_type_candidate": "support",
                "source_label": f"Recent Support {window} bars",
                "dynamic_level": False,
            })

    return candidates


def _build_dynamic_candidates(df: pd.DataFrame, include_dynamic_levels: bool = True) -> list[dict]:
    """Build dynamic support and resistance candidates from EMA20 and SMA50."""
    if not include_dynamic_levels:
        return []

    close = df["Close"]
    current_price = float(close.iloc[-1])
    candidates = []

    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    if pd.notna(ema20):
        candidates.append({
            "level": float(ema20),
            "zone_mid": float(ema20),
            "zone_lower_bound": float(ema20),
            "zone_upper_bound": float(ema20),
            "touches": 1,
            "hits": 1,
            "most_recent_touch_date": df.index[-1],
            "recency_score": 1.0,
            "volume_score": 0.0,
            "violation_penalty": 0.0,
            "level_score": 2.0,
            "zone_type_candidate": "support" if current_price >= ema20 else "resistance",
            "source_label": "Dynamic Level: EMA20",
            "dynamic_level": True,
        })

    sma50 = close.rolling(window=50, min_periods=50).mean().iloc[-1]
    if pd.notna(sma50):
        candidates.append({
            "level": float(sma50),
            "zone_mid": float(sma50),
            "zone_lower_bound": float(sma50),
            "zone_upper_bound": float(sma50),
            "touches": 1,
            "hits": 1,
            "most_recent_touch_date": df.index[-1],
            "recency_score": 1.0,
            "volume_score": 0.0,
            "violation_penalty": 0.0,
            "level_score": 2.0,
            "zone_type_candidate": "support" if current_price >= sma50 else "resistance",
            "source_label": "Dynamic Level: SMA50",
            "dynamic_level": True,
        })

    return candidates


def _classify_zone(level: dict, current_price: float, actionable_threshold_pct: float) -> dict:
    """Classify a clustered zone as support or resistance relative to current price."""
    level_value = float(level["level"])

    if level_value <= current_price:
        level_type = "Support"
        distance_pct = float(abs(level_value / current_price - 1) * 100)
    else:
        level_type = "Resistance"
        distance_pct = float(abs(level_value / current_price - 1) * 100)

    is_actionable = bool(distance_pct <= actionable_threshold_pct)
    classified = dict(level)
    classified["type"] = level_type
    classified["distance_pct"] = distance_pct
    classified["is_actionable"] = is_actionable
    classified["distance_atr"] = None
    return classified


def build_level_snapshot(
    df: pd.DataFrame,
    current_price: float,
    atr: float | None = None,
    max_levels: int = 4,
    relevance_atr: float | None = None,
    relevance_pct: float | None = None,
    show_all_historical: bool = False,
    sensitivity: int = 5,
    include_dynamic_levels: bool = True,
) -> dict:
    """Build actionable and historical support/resistance views for the current price."""
    current_atr = atr if atr is not None else calculate_atr(df, period=14)
    cluster_tolerance_pct = calculate_cluster_tolerance_pct(current_atr, current_price)
    actionable_threshold_pct = calculate_actionable_threshold_pct(current_atr, current_price)

    clustered = build_clustered_zones(df, sensitivity=sensitivity, cluster_tolerance_pct=cluster_tolerance_pct)
    static_candidates = clustered["support_zones"] + clustered["resistance_zones"] + _build_recent_support_candidates(df)
    dynamic_candidates = _build_dynamic_candidates(df, include_dynamic_levels=include_dynamic_levels)
    all_candidates = static_candidates + dynamic_candidates

    classified_levels = [
        _classify_zone(level, current_price=current_price, actionable_threshold_pct=actionable_threshold_pct)
        for level in all_candidates
    ]

    for item in classified_levels:
        if current_atr and current_atr > 0:
            item["distance_atr"] = float(abs(current_price - item["level"]) / current_atr)

    all_supports = sorted(
        [item for item in classified_levels if item["type"] == "Support"],
        key=lambda item: (-item["level_score"], -item["level"]),
    )
    all_resistances = sorted(
        [item for item in classified_levels if item["type"] == "Resistance"],
        key=lambda item: (-item["level_score"], item["level"]),
    )

    actionable_supports = [item for item in all_supports if item["is_actionable"]][:max_levels]
    actionable_resistances = [item for item in all_resistances if item["is_actionable"]][:max_levels]
    historical_supports = [item for item in all_supports if not item["is_actionable"]]
    historical_resistances = [item for item in all_resistances if not item["is_actionable"]]

    if not show_all_historical:
        historical_supports = []
        historical_resistances = []

    nearest_actionable_levels = get_nearest_levels(
        current_price=current_price,
        support_levels=actionable_supports,
        resistance_levels=actionable_resistances,
        atr=current_atr,
    )

    ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
    price_extended = bool(
        pd.notna(ema20)
        and (
            current_price > ema20 * 1.20
            or (current_atr and current_atr > 0 and current_price > ema20 + 2 * current_atr)
        )
    )

    return {
        "actionable_supports": actionable_supports,
        "actionable_resistances": actionable_resistances,
        "historical_supports": historical_supports,
        "historical_resistances": historical_resistances,
        "all_supports": all_supports,
        "all_resistances": all_resistances,
        "nearest_actionable_levels": nearest_actionable_levels,
        "price_discovery_mode": len(actionable_resistances) == 0,
        "price_extended": price_extended,
        "cluster_tolerance_pct": cluster_tolerance_pct,
        "actionable_threshold_pct": actionable_threshold_pct,
        "swing_sensitivity": sensitivity,
    }


def get_nearest_levels(
    current_price: float,
    support_levels,
    resistance_levels,
    atr: float | None = None,
) -> dict:
    """Find the nearest support and resistance plus normalized distances."""
    support_candidates = [level for level in support_levels if level["level"] < current_price]
    resistance_candidates = [level for level in resistance_levels if level["level"] > current_price]

    nearest_support = max(support_candidates, key=lambda item: item["level"], default=None)
    nearest_resistance = min(resistance_candidates, key=lambda item: item["level"], default=None)

    support_level = nearest_support["level"] if nearest_support else None
    resistance_level = nearest_resistance["level"] if nearest_resistance else None

    atr_value = None
    if atr is not None and not pd.isna(atr) and atr > 0:
        atr_value = float(atr)

    distance_to_support_atr = None
    if support_level is not None and atr_value:
        distance_to_support_atr = float((current_price - support_level) / atr_value)

    distance_to_resistance_atr = None
    if resistance_level is not None and atr_value:
        distance_to_resistance_atr = float((resistance_level - current_price) / atr_value)

    distance_to_support_pct = None
    if support_level is not None and support_level != 0:
        distance_to_support_pct = float(current_price / support_level - 1)

    distance_to_resistance_pct = None
    if resistance_level is not None and current_price != 0:
        distance_to_resistance_pct = float(resistance_level / current_price - 1)

    return {
        "support": support_level,
        "support_strength": nearest_support["hits"] if nearest_support else 0,
        "resistance": resistance_level,
        "resistance_strength": nearest_resistance["hits"] if nearest_resistance else 0,
        "distance_to_support_atr": distance_to_support_atr,
        "distance_to_resistance_atr": distance_to_resistance_atr,
        "distance_to_support_pct": distance_to_support_pct,
        "distance_to_resistance_pct": distance_to_resistance_pct,
        "price_between_levels": bool(
            support_level is not None
            and resistance_level is not None
            and support_level <= current_price <= resistance_level
        ),
    }
