# Trading Dashboard - Dual Signal Scores v1

Streamlit dashboard for stocks and crypto with a fully rule-based, explainable signal engine. Version 4.5 adds richer signal interpretation, volatility percentile logic, and a more actionable trade setup panel without using machine learning.

This copy adds side-by-side higher-reliability and higher-return scoring in a separate folder, preserving the prior intraday version.

The single-stock analysis chart uses a fixed 10-day, 5-minute window. Intraday candles are plotted on a compact trading-bar axis so overnight and weekend gaps do not visually stretch the chart, and indicator lines break between sessions instead of drawing fake overnight diagonals. The higher-reliability buy breakdown also includes a score audit that verifies the displayed daily, entry, and total scores against the underlying component sums.

## Main Watchlist Scores

The main table exposes two signal families:

- `Intraday Buy`: intraday-weighted long score, using the 6-point daily/market filter plus 14-point intraday execution score.
- `Intraday Sell`: intraday-aware sell/exhaustion score, where higher values mean protect gains or reduce exposure.
- `HR Buy`: higher-reliability intraday long score, using the 6-point daily/market filter plus 14-point intraday execution score.
- `HR Sell`: higher-reliability sell/exhaustion score, where higher values mean protect gains or reduce exposure.
- `Return Buy`: higher-return semi-auto trend/momentum score, using daily trend/relative-strength eligibility plus pullback/reclaim entry timing.
- `Return Sell`: higher-return trend-following exit score.

## Higher Reliability Buy Model

This version treats the daily chart as a filter, not the main score:

- Stage 1 daily / market bias: 6 points
- Stage 2 intraday execution quality: 14 points
- Total score: 20 points

Recommended signal gates:

- Daily bias score at least `3`
- Intraday execution score at least `9`
- Price above or reclaiming VWAP
- Entry not overly extended by ATR

Total score interpretation:

- 17-20: best long setup
- 14-16: acceptable long setup
- 11-13: watch only or small starter
- below 11: pass

## Daily / Market Bias Score

Daily bias is a 6-point filter:

- Price above daily 20 EMA and 50 SMA: 2
- Relative strength above benchmark recently: 1
- Daily MACD improving or positive: 1
- +DI above -DI or ADX above 20: 1
- Price not more than about 1 ATR above daily 20 EMA: 1

Interpretation:

- 5-6: strong long bias
- 3-4: acceptable, but needs excellent intraday setup
- 0-2: avoid long unless pure scalp/reversal

## Intraday Execution Score

Intraday execution is a 14-point score:

- Price above VWAP or reclaims VWAP and holds: 3
- 9 EMA above 20 EMA and both rising: 2
- Pullback holds VWAP, 9 EMA, or 20 EMA: 2
- RSI cools into 40-55 and turns up: 2
- MACD histogram improving or bullish cross: 2
- Green reclaim/breakout candle has above-average volume: 2
- Entry is not overly extended from VWAP/EMA by ATR: 1

Interpretation:

- 11-14: strong intraday entry
- 8-10: tradable, smaller size or tighter stop
- 0-7: wait

## Higher Reliability Sell / Exhaustion Model

The sell score is a risk-of-exhaustion model for existing long positions:

- Daily exhaustion score: 10 points
- Intraday exit trigger score: 10 points
- ATR modifier: capped at +2, with the final score capped at 20

Higher values mean protect gains, tighten stops, trim, or exit. They are not intended as standalone short-sale signals.

## Single Stock Analysis Chart

The `Single Stock Analysis` mode now uses a fixed price chart window:

- 10 days
- 5-minute candles
- VWAP enabled by default

The single-stock page intentionally separates data by purpose:

- 10-day, 5-minute data is used for the chart and intraday execution score.
- daily data is used for regime scoring, support/resistance, VCP, volatility, and backtest calculations.
- VWAP resets by trading session when intraday timestamps are available.

The watchlist dashboard detail view is intentionally unchanged and still uses its 5-day `5m` / `1m` candle workflow.

## Higher Return Model

The higher-return model emphasizes:

- 200-day SMA trend filter
- 20 EMA / 50 SMA structure
- 3-month, 6-month, and 12-month relative strength versus a benchmark, default `SPY`
- RSI(2) and RSI(4) pullback timing
- VWAP only for intraday execution
- ATR tradability, extension filters, and trailing-stop context
- volume confirmation
- MACD as confirmation only
- ADX / +DI / -DI as a lower-weight trend-quality filter

The practical entry rule is:

- Daily score must be at least `7`
- Entry score must be at least `7`
- Price must be above the 200-day SMA
- No entry if price is more than 1 ATR above the 20 EMA
- No entry if the intraday entry is more than 0.75 ATR above VWAP / 20 EMA

## Daily Systematic Score

Daily score is a 10-point trade eligibility score:

- Close above 200 SMA: 2
- 20 EMA above 50 SMA: 2
- 3-month or 6-month relative strength above benchmark: 2
- Close above one-month-ago close: 1
- ADX above 20 and +DI above -DI: 1
- 20-day average volume rising or breakout volume above average: 1
- ATR large enough to trade but not extreme: 1

Interpretation:

- 8-10: strong candidate
- 6-7: watchlist only unless intraday setup is excellent
- 0-5: avoid for bullish semi-auto entries

## Entry Trigger Score

Entry score is a 10-point score for buying today:

- RSI(2) below 10 or RSI(4) below 30 inside an uptrend: 2
- RSI turns back up or reclaims threshold: 1
- Price reclaims 9 EMA or 20 EMA: 2
- Intraday price above or reclaiming VWAP: 2
- Intraday MACD histogram improving or bullish: 1
- Green reclaim candle with at least average relative volume: 1
- Entry not more than 0.75 ATR above VWAP / 20 EMA: 1

Interpretation:

- 8-10: valid entry
- 6-7: small starter only
- 0-5: wait

## Trend-Failure Exit Score

Exit score is used for larger swing/trend-following exits:

- Close below 20 EMA: 2
- 9 EMA crosses below 20 EMA: 2
- MACD histogram negative over recent bars: 1
- RSI(14) loses 50: 1
- +DI crosses below -DI: 2
- ADX falling after a high reading: 1
- High-volume red candle / distribution day: 1
- Close below 2.5 ATR trailing stop: 2

Interpretation:

- 0-3: hold
- 4-5: tighten stop
- 6-7: trim
- 8+: exit

## Watchlist Dashboard

Use `App Mode -> Watchlist Dashboard` to scan many ticker symbols at once.

The watchlist dashboard supports:

- comma, space, or newline separated ticker lists
- sortable table columns for:
  - 6-month trend sparkline
  - Intraday Buy
  - Intraday Sell
  - Intraday Action
  - HR Buy
  - HR Sell
  - Return Buy
  - Return Sell
  - relative strength
  - intraday RSI
  - intraday volume ratio
  - distance from 20 EMA
  - ATR percentage
  - 6-month return
- a mini 6-month trend sparkline for each ticker
- quick ticker buttons plus a dropdown to open a detail view
- 5-trading-session candle detail charts using either `5m` or `1m` candles
- optional 400-bucket horizontal volume profile overlay on detail and single-stock price charts
- a 6-month trend chart for the selected ticker

Sort descending by `Intraday Buy` for the intraday-weighted long model, by `Return Buy` for trend/momentum pullback candidates, or by `Intraday Sell`, `HR Sell`, / `Return Sell` for exit candidates.

The watchlist scan limit uses a free numeric input rather than a capped slider, so larger ticker lists can be screened when the data source and local runtime can handle the load. Detail price charts load the most recent 5 trading sessions rather than the most recent 5 calendar days. VCP detection is off by default and can be enabled from the sidebar.

## What Changed

- Added a clearly defined market regime detector:
  - `Trending Up`
  - `Trending Down`
  - `Ranging`
  - `High Volatility` as an overlay or standalone regime when no directional structure is present
- Rebuilt signal scoring around explicit rule weights instead of opaque heuristics
- Renamed `Confidence` to `Signal Strength`
- Added the note: `This is rule-based signal strength, not a calibrated probability.`
- Added signal style classification:
  - `Trend-Following`
  - `Mean-Reversion`
  - `Breakout`
  - `Neutral`
- Updated the backtest to generate signals historically with only data available up to each date
- Added buy-and-hold comparison and signal-based equity drawdown
- Added actionable-vs-historical level separation for parabolic names and price-discovery mode
- Added volatility percentile-based high-volatility detection and watch-state bias labels
- Added a trade setup interpretation panel with confirmation, invalidation, and simple risk/reward

## Signal Engine Rules

The dashboard uses fixed indicator settings for the signal engine, even if the chart overlays use different display periods:

- Returns:
  - `daily_return = close.pct_change()`
  - `5_period_return = close / close.shift(5) - 1`
  - `20_period_return = close / close.shift(20) - 1`
- EMA:
  - `EMA20 = close.ewm(span=20, adjust=False).mean()`
  - `EMA20 slope = EMA20 / EMA20.shift(5) - 1`
- SMA:
  - `SMA50 = close.rolling(50).mean()`
  - `SMA50 slope = SMA50 / SMA50.shift(5) - 1`
- Volatility:
  - `daily_vol = daily_return.rolling(20).std()`
  - `annual_vol = daily_vol * sqrt(252)` for stocks
  - `annual_vol = daily_vol * sqrt(365)` for tickers containing `-USD`
- ATR:
  - `true_range = max(high-low, abs(high-prev_close), abs(low-prev_close))`
  - `ATR14 = true_range.rolling(14).mean()`
- Support and resistance distance:
  - `distance_to_support_atr = (close - nearest_support) / ATR`
  - `distance_to_resistance_atr = (nearest_resistance - close) / ATR`
  - `distance_to_support_pct = close / nearest_support - 1`
  - `distance_to_resistance_pct = nearest_resistance / close - 1`

## Regime Detection

`Trending Up`

- `close > EMA20`
- `EMA20 slope over 5 periods > +1%`
- `close > SMA50` when SMA50 is available
- `20_period_return > +5%`

`Trending Down`

- `close < EMA20`
- `EMA20 slope over 5 periods < -1%`
- `close < SMA50` when SMA50 is available
- `20_period_return < -5%`

`High Volatility`

- `annual_vol` is elevated relative to the ticker's own selected-period history
- current 20-day annualized volatility percentile is shown in the dashboard
- the warning banner appears only when volatility is in the high percentile regime

`Ranging`

- not `Trending Up`
- not `Trending Down`
- `20_period_return` between `-5%` and `+5%`
- price is between the nearest support and nearest resistance

## Scoring Model

Base score starts at `0`.

`Trending Up`

- `+2` if `close > EMA20`
- `+1` if `EMA20 slope > 0`
- `+1` if `5_period_return > 0`
- `+2` if `close` makes a 20-bar high
- `-1` if annual volatility is above its 75th percentile
- `-2` if price is within `0.75 ATR` of strong resistance

`Trending Down`

- `-2` if `close < EMA20`
- `-1` if `EMA20 slope < 0`
- `-1` if `5_period_return < 0`
- `-2` if `close` makes a 20-bar low
- `+1` if annual volatility is above its 75th percentile
- `+2` if price is within `0.75 ATR` of strong support

`Ranging`

- `+2` if price is within `0.75 ATR` above support
- `-2` if price is within `0.75 ATR` below resistance
- `+1` if `RSI < 35`
- `-1` if `RSI > 65`
- breakout rewards are suppressed unless price closes outside the zone by more than `1 ATR`

`High Volatility`

- current 20-day annualized volatility must be above the ticker's own high percentile threshold
- the current implementation uses the `80th percentile`
- final score is reduced by `30%`
- signal strength is capped at `70%`
- explanation includes: `High volatility reduces signal reliability`

Breakout and breakdown logic:

- Resistance breakout:
  - if `close > nearest_resistance + 0.5 ATR`
  - in a ranging regime, breakout reward requires more than `1 ATR`
  - `+2` if `volume > 1.2 * average_volume_20`
  - `+1` if `EMA20 slope > 0`
  - `+1` if `close` is a 20-bar high
- Support breakdown:
  - if `close < nearest_support - 0.5 ATR`
  - in a ranging regime, breakdown reward requires more than `1 ATR`
  - `-2` if `volume > 1.2 * average_volume_20`
  - `-1` if `EMA20 slope < 0`
  - `-1` if `close` is a 20-bar low

Strong support and resistance are currently defined as levels with at least `2` clustered swing hits.

## Final Signal Mapping

- `score >= +3`: `BUY`
- `score <= -3`: `SELL`
- `+1 <= score < +3`: `WATCH BULLISH`
- `-3 < score <= -1`: `WATCH BEARISH`
- otherwise: `NEUTRAL`

Signal strength:

- `signal_strength = min(95, 50 + abs(score) * 10)`
- `High Volatility` caps the result at `70`
- `Signal Strength is a rule-based score, not a calibrated probability.`

## Backtest

The dashboard backtest is historical and no-look-ahead:

1. Generate a signal on each date using only information available up to that date.
2. For each `BUY`, compute forward returns over `5`, `10`, and `20` bars.
3. For each `SELL`, compute inverse forward returns over `5`, `10`, and `20` bars.
4. Report:
   - total signals
   - BUY signals
   - SELL signals
   - BUY win rate
   - SELL win rate
   - average forward return over `5`, `10`, and `20` bars
   - median forward return over `5`, `10`, and `20` bars
   - max drawdown of the signal-based equity curve
   - buy-and-hold return

## UI Updates

The dashboard now shows:

- detected regime near the top
- current bias and trade trigger separately
- signal strength
- score
- annualized volatility and volatility percentile
- explanation list
- signal style
- high-volatility warning when active
- actionable levels separate from historical levels
- price discovery and extended-move warnings
- trade setup interpretation with confirmation and invalidation levels
- optional indicator overlays/panels:
  - Bollinger Bands
  - RSI
  - MACD
  - Stochastic oscillator
  - OBV
- an Indicator Snapshot panel with the latest RSI14, MACD histogram, Bollinger %B, Stochastic %K, and OBV 5-bar change

## Technical Indicator Extension Points

Indicator calculations live in `indicators.py` so they can be reused by the single-symbol dashboard, the signal engine, and a future multi-symbol summary dashboard.

Current reusable indicators:

- `calculate_rsi`
- `calculate_atr`
- `calculate_adx`
- `calculate_vwap`
- `calculate_bollinger_bands`
- `calculate_macd`
- `calculate_stochastic`
- `calculate_obv`

## Bullish Continuation Checklist Score

The dashboard includes a separate two-stage score for long continuation trades. This score is designed to keep daily chart quality separate from intraday execution quality.

Stage 1: Daily bias score, 10 points, using a 6-month daily chart.

- ADX(14): 0 to 2 points
- +DI vs -DI: 0 to 2 points
- RSI(14): 0 to 2 points
- MACD daily: 0 to 2 points
- 20 EMA daily slope: 0 to 1 point
- Price vs 20 EMA daily: 0 to 1 point

Daily interpretation:

- 8-10: strong long candidate
- 6-7: tradable, but needs cleaner intraday trigger
- 0-5: usually skip for bullish continuation trades

Stage 2: Intraday entry score, 10 points, using the selected checklist intraday interval.

- VWAP position: 0 to 2 points
- EMA structure using 9 EMA, 20 EMA, and price relation: 0 to 3 points
- RSI: 0 to 2 points
- MACD: 0 to 2 points
- Volume participation: 0 to 1 point

ATR is used as a quality and chasing filter, not a direction signal.

- Entry more than 0.5 ATR above 9 EMA: -1
- Entry more than 0.75 ATR above VWAP or 20 EMA: -2 and triggers the chasing filter
- Very sharp extended candle: -2
- ATR too small relative to price: -1
- Less than 1 ATR available before nearby resistance: -1
- ATR penalty is capped at -2

Final checklist score:

- 16-20: strong setup
- 13-15: acceptable, but not ideal
- 10-12: watchlist only or smaller size
- Below 10: pass

Recommended default gates:

- Require daily score of at least 7
- Require intraday score of at least 7
- Do not enter when the ATR chasing filter is triggered

## Sell / Exhaustion Checklist Score

The dashboard also includes a two-stage exit score for existing long positions. A higher score means the move may be getting tired and the trader should consider tightening stops, trimming, or exiting.

Stage 1: Daily exhaustion score, 10 points.

- ADX trend aging: 0 to 2 points
- +DI / -DI weakening: 0 to 2 points
- RSI exhaustion or bearish divergence: 0 to 2 points
- MACD weakening: 0 to 2 points
- Price vs 20 EMA: 0 to 2 points

Daily exit interpretation:

- 0-3: trend still healthy
- 4-5: early warning
- 6-7: move may be aging; tighten stops
- 8-10: strong sell/trim warning

Stage 2: Intraday exit trigger score, 10 points.

- VWAP loss: 0 to 2 points
- 9 EMA / 20 EMA structure: 0 to 2 points
- Intraday RSI exit signal or bearish divergence: 0 to 2 points
- Intraday MACD weakening: 0 to 2 points
- Volume / distribution pressure: 0 to 2 points

ATR is used as an exit quality modifier and is capped at +2.

- Price more than 1 ATR above 20 EMA after a fast move: +1
- Price more than 1.5 ATR above 20 EMA: +2
- Large red reversal candle greater than 1 ATR: +2
- Price breaks below VWAP or 20 EMA by more than 0.5 ATR: +1

Combined exit score:

- 0-6: hold / trend intact
- 7-10: watchlist warning
- 11-13: tighten stop, consider partial trim
- 14-16: trim or exit most of position
- 17-20: strong exit signal

## Directional Checklist Score

The dashboard shows a signed score that consolidates the buy and exit checklists:

`Directional Score = Buy Checklist Score - Exit Checklist Score`

- Positive scores favor long entries.
- Negative scores favor protecting gains, trimming, or exiting.
- The displayed scale runs from `-20` for strongest sell/exit pressure to `+20` for strongest buy pressure.

The main data flow is:

1. `app.py` downloads and normalizes OHLCV data with `load_data`.
2. `signals.py` builds reusable feature columns with `build_feature_frame`.
3. `app.py` renders current metrics, chart overlays, indicator panels, support/resistance, VCP, and backtest output.
4. `backtest.py` reuses `build_feature_frame` so historical tests and current signals are based on the same indicator calculations.

A future many-stock dashboard should reuse `load_data`, `build_level_snapshot`, `generate_signal`, and the feature values returned in `signal_result["features"]`.

## Level Relevance

Support and resistance are now classified relative to the current price:

- only levels below the current price are treated as support
- only levels above the current price are treated as resistance
- distant levels are excluded from the actionable view unless they are within the configured distance filters
- default actionable filters are `5 ATR` or `25%`
- a sidebar toggle can reveal all historical levels

Additional candidates are included for parabolic and trend-driven charts:

- recent support from the last `20`, `50`, and `100` bars
- `EMA20`
- `SMA50`

If no resistance exists above the current price, the dashboard shows:

- `No overhead resistance detected - price discovery mode.`

If the price is more than `20%` above `EMA20`, the dashboard warns that the stock may be extended.

## Trade Setup Interpretation

The trade setup panel summarizes the current rule-based view:

- market regime
- current bias
- confirmation level
- invalidation level
- nearest support
- nearest resistance
- potential upside to resistance
- downside risk to support
- simple risk/reward ratio when both support and resistance exist

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

For Windows double-click launching, use [Open Trading Dashboard.bat](/C:/Users/kiyam/Trading%20Dashboard/Open%20Trading%20Dashboard.bat). You can place a shortcut to that file on your desktop and open the dashboard by double-clicking it.

## Smoke Test

Run the requested ticker validation:

```bash
python test_decision_engine.py
```

This checks the current engine against:

- `AMD`
- `MSFT`
- `SPY`
- `BTC-USD`
- `SNDK`

## Notes

- The signal engine is fully rule-based and explainable.
- No machine learning is used.
- Daily bars are the intended calibration for the regime rules. Weekly and monthly views remain available for charting, but the signal model is less reliable there.
- The backtest is exploratory and does not include trading costs, slippage, taxes, or execution risk.
