from app import load_data, select_strong_levels, calculate_max_drawdown, calculate_annualized_volatility, get_nearest_levels, get_series
from datetime import datetime, timedelta
import os

os.chdir(r'c:\Users\kiyam\Trading Dashboard')

symbols = ['AAPL', 'AMD', 'SPY', 'BTC-USD']
end = datetime.now()
start = end - timedelta(days=365)

for symbol in symbols:
    df = load_data(symbol, start, end, '1d')
    if df.empty:
        raise SystemExit(f'No data for {symbol}')
    support, resistance = select_strong_levels(df, max_levels=4)
    close = float(get_series(df, 'Close').iloc[-1])
    nearest = get_nearest_levels(close, support, resistance)
    vol = calculate_annualized_volatility(get_series(df, 'Close'), '1d')
    mdd = calculate_max_drawdown(get_series(df, 'Close'))
    print(symbol, 'rows', len(df), 'close', f'{close:.2f}', 'vol%', f'{vol:.2f}', 'mdd%', f'{mdd:.2f}', 'sup', nearest['support'], 'res', nearest['resistance'])
