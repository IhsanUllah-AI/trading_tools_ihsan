import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone, timedelta
import math

# common.py
@st.cache_data(ttl=60, show_spinner="Fetching market data...")
def fetch_candles_from_binance(symbol="BTCUSDT", interval="5m", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    
    minutes_back = limit * interval_minutes.get(interval, 5)
    start_time = int((datetime.now() - timedelta(minutes=minutes_back)).timestamp() * 1000)
    
    all_data = []
    while limit > 0:
        request_limit = min(limit, 1000)
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": request_limit,
            "startTime": start_time
        }
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_data.extend(data)
            start_time = data[-1][0] + 1
            limit -= request_limit
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None

    rows = []
    for d in all_data:
        rows.append({
            "opentime": pd.to_datetime(d[0], unit='ms', utc=True),
            "closetime": pd.to_datetime(d[6], unit='ms', utc=True),
            "open": float(d[1]),
            "high": float(d[2]),
            "low": float(d[3]),
            "close": float(d[4]),
            "volume": float(d[5]),
            "number_of_trades": d[8],
            "quote_asset_volume": float(d[7]),
            "taker_buy_base_volume": float(d[9]),
            "taker_buy_quote_volume": float(d[10])
        })

    df = pd.DataFrame(rows).sort_values("closetime").reset_index(drop=True)
    df = calculate_indicators(df, interval)
    now_utc = datetime.now(timezone.utc)
    df = df[df["closetime"] <= now_utc].reset_index(drop=True)
    return df

def fetch_current_price(symbol="BTCUSDT"):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return float(data['price'])
    except Exception as e:
        st.error(f"Error fetching current price for {symbol}: {e}")
        return None

def calculate_indicators(df, interval, rsi_period=14, ema_period=20, ema_long_period=50, atr_period=14, bb_period=20,
                        macd_fast=12, macd_slow=26, macd_signal=9,
                        ichimoku_tenkan=9, ichimoku_kijun=26, ichimoku_senkou_b=52):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=ema_long_period, adjust=False).mean()
    
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)

    df['atr'] = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    df['inverseATR'] = 1 / df['atr']

    df['bb_sma'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std(ddof=0)
    df['bb_upper'] = df['bb_sma'] + (bb_std * 2)
    df['bb_lower'] = df['bb_sma'] - (bb_std * 2)
    df['bb_percentB'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    high_tenkan = df['high'].rolling(window=ichimoku_tenkan).max()
    low_tenkan = df['low'].rolling(window=ichimoku_tenkan).min()
    df['Tenkan_sen'] = (high_tenkan + low_tenkan) / 2
    
    high_kijun = df['high'].rolling(window=ichimoku_kijun).max()
    low_kijun = df['low'].rolling(window=ichimoku_kijun).min()
    df['Kijun_sen'] = (high_kijun + low_kijun) / 2
    
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(ichimoku_kijun)
    df['Senkou_Span_A_unshifted'] = (df['Tenkan_sen'] + df['Kijun_sen']) / 2
    
    high_senkou_b = df['high'].rolling(window=ichimoku_senkou_b).max()
    low_senkou_b = df['low'].rolling(window=ichimoku_senkou_b).min()
    df['Senkou_Span_B'] = ((high_senkou_b + low_senkou_b) / 2).shift(ichimoku_kijun)
    df['Senkou_Span_B_unshifted'] = (high_senkou_b + low_senkou_b) / 2

    lookback = 20
    df['support'] = df['low'].rolling(window=lookback).min()
    df['resistance'] = df['high'].rolling(window=lookback).max()
    
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    
    df['volume_spike'] = df['volume'] > df['avg_volume'] * 1.5
    
    df['volume_spike_ema'] = df['volume_spike'].ewm(span=20, adjust=False).mean()
    
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    swing_window_dict = {
        '1m': 65,
        '3m': 50,
        '5m': 50,
        '15m': 70,
        '30m': 85,
        '1h': 100,
        '2h': 100,
        '4h': 90,
        '6h': 90,
        '8h': 90,
        '12h': 90,
        '1d': 60,
        '3d': 60,
        '1w': 60,
        '1M': 60
    }
    swing_window = swing_window_dict.get(interval, 50)
    
    min_price_change = df['atr'] * 0.5

    df['hh'] = False
    df['hl'] = False
    df['lh'] = False
    df['ll'] = False
    for i in range(swing_window, len(df)):
        local_window = 5
        is_local_high = df['high'].iloc[i] == df['high'].iloc[max(0, i-local_window):i+local_window+1].max()
        is_local_low = df['low'].iloc[i] == df['low'].iloc[max(0, i-local_window):i+local_window+1].min()
        
        if is_local_high and df['high'].iloc[i] > df['high'].iloc[i-swing_window:i].max():
            price_diff = df['high'].iloc[i] - df['high'].iloc[i-swing_window:i].max()
            if price_diff >= min_price_change.iloc[i]:
                df['hh'].iloc[i] = True
        
        if is_local_low and df['low'].iloc[i] > df['low'].iloc[i-swing_window:i].min():
            price_diff = df['low'].iloc[i] - df['low'].iloc[i-swing_window:i].min()
            if price_diff >= min_price_change.iloc[i]:
                df['hl'].iloc[i] = True
        
        if is_local_high and df['high'].iloc[i] < df['high'].iloc[i-swing_window:i].max():
            price_diff = df['high'].iloc[i-swing_window:i].max() - df['high'].iloc[i]
            if price_diff >= min_price_change.iloc[i]:
                df['lh'].iloc[i] = True
        
        if is_local_low and df['low'].iloc[i] < df['low'].iloc[i-swing_window:i].min():
            price_diff = df['low'].iloc[i-swing_window:i].min() - df['low'].iloc[i]
            if price_diff >= min_price_change.iloc[i]:
                df['ll'].iloc[i] = True

    # Gann-specific indicators
    lookback = 10
    if len(df) > lookback * 2 + 1:
        df['pivot_low'] = (df['low'] == df['low'].rolling(window=lookback*2+1, center=True).min()) & (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        df['pivot_high'] = (df['high'] == df['high'].rolling(window=lookback*2+1, center=True).max()) & (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    else:
        df['pivot_low'] = False
        df['pivot_high'] = False
    
    gann_columns = ['gann_1x1', 'gann_2x1', 'gann_1x2', 'gann_4x1', 'gann_1x4', 
                    'gann_8x1', 'gann_3x1', 'gann_1x3', 'gann_1x8']
    for col in gann_columns:
        df[col] = np.nan
    
    df['pivot_low_price'] = np.nan
    df['pivot_high_price'] = np.nan
    df['gann_price_scale'] = np.nan
    
    lookback = min(50, len(df))
    if len(df) > 0:
        df['gann_significant_low'] = df['low'].tail(lookback).min()
        df['gann_significant_high'] = df['high'].tail(lookback).max()
    else:
        df['gann_significant_low'] = 0
        df['gann_significant_high'] = 0
    
    return df