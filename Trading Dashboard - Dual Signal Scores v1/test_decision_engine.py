from datetime import datetime, timedelta

from app import load_data
from backtest import backtest_signals_no_lookahead
from levels import build_level_snapshot, calculate_atr
from signals import generate_signal


SYMBOLS = ["AMD", "MSFT", "SPY", "BTC-USD", "SNDK"]


def main():
    end = datetime.now()
    start = end - timedelta(days=365)

    print("=" * 80)
    print("SIGNAL INTERPRETATION ENGINE SMOKE TEST")
    print("=" * 80)

    skipped_symbols = []

    for symbol in SYMBOLS:
        print(f"\nTesting {symbol}")
        print("-" * 80)

        df = load_data(symbol, start, end, "1d")
        if df.empty:
            skipped_symbols.append(symbol)
            print(f"Skipped {symbol}: no data returned from Yahoo Finance for the selected range.")
            continue

        atr = calculate_atr(df, period=14)
        level_snapshot = build_level_snapshot(
            df,
            current_price=float(df["Close"].iloc[-1]),
            atr=atr,
            max_levels=4,
            relevance_atr=5.0,
            relevance_pct=0.25,
            show_all_historical=False,
        )

        signal = generate_signal(
            df=df,
            ticker=symbol,
            support_levels=level_snapshot["actionable_supports"],
            resistance_levels=level_snapshot["actionable_resistances"],
            max_levels=4,
        )

        assert signal["trade_signal"] in {"BUY", "SELL", "HOLD"}
        assert signal["current_bias"] in {"BUY", "SELL", "WATCH BULLISH", "WATCH BEARISH", "NEUTRAL"}
        assert signal["regime"] in {"Trending Up", "Trending Down", "Ranging", "High Volatility", "Neutral"}
        assert 0 <= signal["signal_strength"] <= 95
        assert signal["trade_setup"]["market_regime"]
        if signal["high_volatility"]:
            assert signal["signal_strength"] <= 70

        backtest = backtest_signals_no_lookahead(
            df=df,
            ticker=symbol,
            max_levels=4,
            lookback_periods=(5, 10, 20),
            relevance_atr=5.0,
            relevance_pct=0.25,
        )

        print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
        print(f"Regime: {signal['regime_label']}")
        print(f"Current bias: {signal['current_bias']} | Trade trigger: {signal['trade_signal']} | Score: {signal['score']:+.2f}")
        if signal["features"]["annual_vol_percentile"] is not None:
            print(
                f"Volatility: {signal['features']['annual_vol'] * 100:.2f}% annualized "
                f"({signal['features']['annual_vol_percentile'] * 100:.0f}th percentile)"
            )
        print(f"Signal type: {signal['signal_style']}")
        print(f"Nearest support: {signal['nearest_levels']['support']}")
        print(f"Nearest resistance: {signal['nearest_levels']['resistance']}")
        print(
            "Backtest: "
            f"{backtest['total_signals']} signals, "
            f"BUY win rate {backtest['buy_win_rate']:.1f}%, "
            f"SELL win rate {backtest['sell_win_rate']:.1f}%, "
            f"buy-and-hold {backtest['buy_and_hold_return']:.2f}%"
        )

    if skipped_symbols:
        print("\nSkipped symbols due to unavailable data:")
        for symbol in skipped_symbols:
            print(f"- {symbol}")

    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
