# tools/fib.py
# (Unchanged, but ensure 'degree': '' is added in main.py for fib trades)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import streamlit as st

from common import fetch_current_price

def compute_fib_levels(df, window=20, fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0]):
    messages = []
    if len(df) < window:
        adjusted_window = len(df)
        messages.append(('warning', f"Window size adjusted to {adjusted_window} (available data points)"))
        window = adjusted_window
    
    latest = df.iloc[-1]
    tail = df.tail(window)

    high = tail["high"].max()
    low = tail["low"].min()
    close = latest["close"]

    move = high - low
    if move == 0:
        return "sideways", {}, high, low, close, latest, messages
    
    pos = (close - low) / move

    if pos > 0.6:
        trend = "uptrend"
        levels = {f"support_{int(r*100)}": high - move * r for r in fib_ratios}
    elif pos < 0.4:
        trend = "downtrend"
        levels = {f"resistance_{int(r*100)}": low + move * r for r in fib_ratios}
    else:
        trend = "sideways"
        levels = {}

    return trend, levels, high, low, close, latest, messages

def generate_signals(df, trend, levels, latest, high, low, symbol, fib_threshold=0.01):
    messages = []
    signals = []
    confidence = 0
    buy_signals = []
    sell_signals = []
    buy_descriptions = []
    sell_descriptions = []

    entry_price = None
    stop_loss = None
    take_profit = None
    trade_action = None
    fib_signal_triggered = False

    current_price = fetch_current_price(symbol)
    if current_price is None:
        current_price = latest['close']

    # RSI Signals (weight 0.25)
    rsi = latest['rsi']
    if rsi < 30:
        buy_signals.append(("RSI Oversold", "BUY", "green", 0.25))
        buy_descriptions.append("RSI < 30: Potential oversold condition, suggesting a buy.")
    elif rsi > 70:
        sell_signals.append(("RSI Overbought", "SELL", "red", 0.25))
        sell_descriptions.append("RSI > 70: Potential overbought condition, suggesting a sell.")

    # Bollinger Bands Signals (weight 0.20)
    bb_percent = latest['bb_percentB']
    if bb_percent < 0:
        buy_signals.append(("Price Below Lower BB", "BUY", "green", 0.20))
        buy_descriptions.append("Price below lower Bollinger Band: Suggests oversold, potential buy.")
    elif bb_percent > 1:
        sell_signals.append(("Price Above Upper BB", "SELL", "red", 0.20))
        sell_descriptions.append("Price above upper Bollinger Band: Suggests overbought, potential sell.")

    # EMA Signal (weight 0.15)
    if current_price > latest['ema']:
        buy_signals.append(("Price Above EMA", "BUY", "blue", 0.15))
        buy_descriptions.append("Price above 20-period EMA: Bullish momentum, favoring buys.")
    else:
        sell_signals.append(("Price Below EMA", "SELL", "orange", 0.15))
        sell_descriptions.append("Price below 20-period EMA: Bearish momentum, favoring sells.")

    # Fibonacci Signals with Indicator Confirmation (weight 0.40)
    if trend != "sideways" and levels:
        fib_levels = list(levels.values())
        fib_keys = list(levels.keys())

        closest_idx = np.argmin(np.abs(np.array(fib_levels) - current_price))
        closest_level = fib_levels[closest_idx]
        level_name = fib_keys[closest_idx]
        distance_percent = abs(current_price - closest_level) / closest_level * 100

        messages.append(('info', f"**Fibonacci Debug Info for {symbol}:**"))
        messages.append(('info', f"Current Price: {current_price:.4f}"))
        messages.append(('info', f"Closest Fibonacci Level: {level_name} at {closest_level:.4f} ({distance_percent:.2f}% away)"))
        messages.append(('info', f"RSI: {rsi:.2f} (Buy: <30, Sell: >70)"))
        messages.append(('info', f"Bollinger %B: {bb_percent*100:.2f}% (Buy: <0, Sell: >100)"))
        messages.append(('info', f"Price vs EMA: {'Above' if current_price > latest['ema'] else 'Below'} (Buy: Above, Sell: Below)"))

        has_buy_indicator = rsi < 30 or bb_percent < 0 or current_price > latest['ema']
        has_sell_indicator = rsi > 70 or bb_percent > 1 or current_price < latest['ema']

        if abs(current_price - closest_level) / closest_level <= fib_threshold:
            if trend == "uptrend" and has_buy_indicator:
                buy_signals.append((f"Near {level_name} Support", "BUY", "green", 0.40))
                buy_descriptions.append(f"Price near {level_name} support with indicator confirmation: Strong buy signal.")
                entry_price = current_price
                stop_loss = entry_price * 0.995
                take_profit = entry_price * 1.01
                fib_signal_triggered = True
                trade_action = "BUY"
            elif trend == "downtrend" and has_sell_indicator:
                sell_signals.append((f"Near {level_name} Resistance", "SELL", "red", 0.40))
                sell_descriptions.append(f"Price near {level_name} resistance with indicator confirmation: Strong sell signal.")
                entry_price = current_price
                stop_loss = entry_price * 1.005
                take_profit = entry_price * 0.99
                fib_signal_triggered = True
                trade_action = "SELL"
            else:
                messages.append(('warning', f"Price is within {fib_threshold*100}% of {level_name}, but no confirming indicators for {trend}."))
        else:
            messages.append(('warning', f"Price is {distance_percent:.2f}% from closest Fibonacci level ({level_name}), outside {fib_threshold*100}% threshold."))

    # Fallback to ATR
    atr = latest['atr']
    if fib_signal_triggered and entry_price is not None:
        if trade_action == "BUY" and (stop_loss >= entry_price or take_profit <= entry_price):
            stop_loss = entry_price - (1.5 * atr)
            take_profit = entry_price + (3 * atr)
        elif trade_action == "SELL" and (stop_loss <= entry_price or take_profit >= entry_price):
            stop_loss = entry_price + (1.5 * atr)
            take_profit = entry_price - (3 * atr)

    signals = buy_signals + sell_signals
    signal_descriptions = buy_descriptions + sell_descriptions

    buy_confidence = sum(weight for _, _, _, weight in buy_signals)
    sell_confidence = sum(weight for _, _, _, weight in sell_signals)

    if buy_confidence > 0 and sell_confidence > 0:
        conflict_penalty = min(buy_confidence, sell_confidence) * 0.5
        confidence = max(buy_confidence, sell_confidence) - conflict_penalty
        trade_action = "BUY" if buy_confidence > sell_confidence and fib_signal_triggered and trade_action == "BUY" else "SELL" if sell_confidence > buy_confidence and fib_signal_triggered and trade_action == "SELL" else None
    else:
        confidence = buy_confidence + sell_confidence
        trade_action = "BUY" if buy_confidence > 0 and fib_signal_triggered else "SELL" if sell_confidence > 0 and fib_signal_triggered else None

    confidence = min(confidence, 1.0)

    # Apply 38% confidence threshold
    if confidence < 0.38:
        trade_action = None
        entry_price = None
        stop_loss = None
        take_profit = None

    return signals, confidence, entry_price, stop_loss, take_profit, signal_descriptions, trade_action, fib_signal_triggered, messages

def create_chart(df, trend, levels, window, entry_price=None, stop_loss=None, take_profit=None):
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'Price with Fibonacci Levels (Based on {window} candles)', 'Volume', 'RSI'),
        row_width=[0.2, 0.2, 0.6]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['closetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['ema'],
            name='EMA (20)',
            line=dict(color='orange', width=1.5)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['bb_upper'],
            name='BB Upper',
            line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['bb_lower'],
            name='BB Lower',
            line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.1)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    colors = ['#FF6B6B', '#FF9E6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C']
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    for i, (ratio, color) in enumerate(zip(fib_ratios, colors)):
        if i < len(levels):
            level_value = list(levels.values())[i]
            if not np.isnan(level_value):
                level_name = list(levels.keys())[i]
                fig.add_hline(
                    y=level_value, 
                    line_dash="dash", 
                    line_color=color,
                    annotation_text=f"{level_name}: {level_value:.2f}",
                    annotation_position="top right",
                    row=1, col=1
                )
    
    if entry_price is not None:
        fig.add_hline(
            y=entry_price, 
            line_dash="solid", 
            line_color="blue",
            annotation_text=f"Entry (Current Price): {entry_price:.2f}",
            row=1, col=1
        )
    
    if stop_loss is not None:
        fig.add_hline(
            y=stop_loss, 
            line_dash="solid", 
            line_color="red",
            annotation_text=f"Stop Loss: {stop_loss:.2f}",
            row=1, col=1
        )
    
    if take_profit is not None:
        fig.add_hline(
            y=take_profit, 
            line_dash="solid", 
            line_color="green",
            annotation_text=f"Take Profit: {take_profit:.2f}",
            row=1, col=1
        )
    
    colors_volume = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['closetime'],
            y=df['volume'],
            name='Volume',
            marker_color=colors_volume
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['closetime'],
            y=df['rsi'],
            name='RSI',
            line=dict(color='purple', width=1.5)
        ),
        row=3, col=1
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        height=800,
        title=f"Market Analysis - {trend.capitalize()}",
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def compute_fib_results(df, tool_params, symbol, interval):
    window = tool_params["window"]
    fib_threshold = tool_params["fib_threshold"]
    
    trend, levels, high, low, close, latest, fib_level_messages = compute_fib_levels(df, window=window)
    
    signals, confidence, entry_price, stop_loss, take_profit, signal_descriptions, trade_action, fib_signal_triggered, signal_messages = generate_signals(
        df, trend, levels, latest, high, low, symbol, fib_threshold
    )
    
    messages = fib_level_messages + signal_messages
    
    current_price = fetch_current_price(symbol) or close
    
    fig = create_chart(df, trend, levels, window, entry_price, stop_loss, take_profit)
    
    if levels:
        is_support = any('support' in key for key in levels.keys())
        level_type = "Support" if is_support else "Resistance"
        
        fib_df = pd.DataFrame.from_dict(levels, orient='index', columns=['Price Level'])
        fib_df.index.name = 'Fibonacci Level'
        fib_df['Type'] = fib_df.index.map(lambda x: 'Support' if 'support' in x else 'Resistance')
        fib_df['Distance %'] = ((current_price - fib_df['Price Level']) / fib_df['Price Level']) * 100
        fib_df['Strength'] = fib_df.index.map(lambda x: 'Strong' if any(r in x for r in ['0618', '0500', '0786']) else 'Moderate' if any(r in x for r in ['0382', '0236']) else 'Weak')
        
        closest_idx = np.argmin(np.abs(fib_df['Distance %']))
        closest_level = fib_df.iloc[closest_idx]
        closest_level_dict = closest_level.to_dict()
        closest_level_dict['name'] = closest_level.name
    else:
        is_support = None
        level_type = None
        fib_df = None
        closest_level_dict = None

    if levels:
        if is_support:
            fib_explanation_md = f"""
<div class="fib-explanation">
<h4>📈 Uptrend Fibonacci Support Levels Analysis</h4>
<p>In an <strong>uptrend</strong>, support levels are where price might bounce up.</p>

<p><strong>Current Price: {current_price:.4f}</strong></p>

<ul>
<li><strong>23.6% Support ({levels.get('support_23', 0):.4f})</strong> - Shallow pullback. Buy with confirmation.</li>
<li><strong>38.2% Support ({levels.get('support_38', 0):.4f})</strong> - Common pullback. Good buy spot.</li>
<li><strong>50.0% Support ({levels.get('support_50', 0):.4f})</strong> - Psychological level. Buy if holds.</li>
<li><strong>61.8% Support ({levels.get('support_61', 0):.4f})</strong> - Golden ratio. Strong buy signal.</li>
<li><strong>78.6% Support ({levels.get('support_78', 0):.4f})</strong> - Deep pullback. Risky buy.</li>
<li><strong>100% Support ({levels.get('support_100', 0):.4f})</strong> - Trend reversal if hit.</li>
</ul>

<p><strong>Trading Recommendations:</strong></p>
<ul>
<li>Price is <strong>{abs(fib_df.loc[closest_level.name, 'Distance %']):.2f}%</strong> from {closest_level.name}</li>
<li>Wait for <strong>indicator confirmation</strong> (RSI < 30, BB %B < 0, or Price > EMA)</li>
<li>Stronger supports (61.8%, 50%) offer better risk-reward</li>
<li>Breaking below 78.6% may signal trend end</li>
</ul>
</div>
"""
        else:
            fib_explanation_md = f"""
<div class="fib-explanation">
<h4>📉 Downtrend Fibonacci Resistance Levels Analysis</h4>
<p>In a <strong>downtrend</strong>, resistance levels are where price might fall back.</p>

<p><strong>Current Price: {current_price:.4f}</strong></p>

<ul>
<li><strong>23.6% Resistance ({levels.get('resistance_23', 0):.4f})</strong> - Shallow bounce. Sell with confirmation.</li>
<li><strong>38.2% Resistance ({levels.get('resistance_38', 0):.4f})</strong> - Common bounce. Good sell spot.</li>
<li><strong>50.0% Resistance ({levels.get('resistance_50', 0):.4f})</strong> - Psychological level. Sell if rejects.</li>
<li><strong>61.8% Resistance ({levels.get('resistance_61', 0):.4f})</strong> - Golden ratio. Strong sell signal.</li>
<li><strong>78.6% Resistance ({levels.get('resistance_78', 0):.4f})</strong> - Deep bounce. Risky sell.</li>
<li><strong>100% Resistance ({levels.get('resistance_100', 0):.4f})</strong> - Trend reversal if hit.</li>
</ul>

<p><strong>Trading Recommendations:</strong></p>
<ul>
<li>Price is <strong>{abs(fib_df.loc[closest_level.name, 'Distance %']):.2f}%</strong> from {closest_level.name}</li>
<li>Wait for <strong>indicator confirmation</strong> (RSI > 70, BB %B > 1, or Price < EMA)</li>
<li>Stronger resistances (61.8%, 50%) offer better risk-reward</li>
<li>Breaking above 78.6% may signal trend end</li>
</ul>
</div>
"""
    else:
        fib_explanation_md = ""

    rsi_value = latest['rsi']
    ema_rel = "Above" if current_price > latest['ema'] else "Below"
    atr = latest['atr']
    bb_value = latest['bb_percentB'] * 100
    bb_status = "Overbought" if bb_value > 100 else "Oversold" if bb_value < 0 else "Neutral"

    price_change = ((latest['close'] - latest['open']) / latest['open']) * 100

    price_stats = {
        'open': latest['open'],
        'high': latest['high'],
        'low': latest['low'],
        'close': latest['close'],
        'change': price_change
    }

    volume_stats = {
        'volume': latest['volume'],
        'quote_volume': latest['quote_asset_volume'],
        'trades': latest['number_of_trades']
    }

    results = {
        'trend': trend,
        'levels': levels,
        'high': high,
        'low': low,
        'close': close,
        'latest': latest,
        'signals': signals,
        'confidence': confidence,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'signal_descriptions': signal_descriptions,
        'trade_action': trade_action,
        'fib_signal_triggered': fib_signal_triggered,
        'current_price': current_price,
        'fig': fig,
        'fib_df': fib_df,
        'closest_level': closest_level_dict,
        'level_type': level_type,
        'is_support': is_support,
        'fib_explanation_md': fib_explanation_md,
        'rsi_value': rsi_value,
        'ema_rel': ema_rel,
        'atr': atr,
        'bb_value': bb_value,
        'bb_status': bb_status,
        'price_stats': price_stats,
        'volume_stats': volume_stats,
        'df': df,
        'messages': messages,
        'window': window
    }

    return results

def display_fib_results(results):
    # Display messages
    for typ, msg in results['messages']:
        if typ == 'info':
            st.write(msg)
        elif typ == 'warning':
            st.warning(msg)
    
    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Price", f"{results['current_price']:.4f}")
        st.caption("Latest market price via Binance API.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Swing High", f"{results['high']:.4f}")
        st.caption("Highest price in the window.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Swing Low", f"{results['low']:.4f}")
        st.caption("Lowest price in the window.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        trend_class = f"trend-{results['trend']}"
        st.markdown(f"Trend: **<span class='{trend_class}'>{results['trend'].upper()}</span>**", unsafe_allow_html=True)
        st.caption("Market direction based on price position.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trading Signals
    st.subheader("📈 Trading Signals")
    if results['signals']:
        confidence_class = "confidence-high" if results['confidence'] > 0.6 else "confidence-medium" if results['confidence'] >= 0.38 else "confidence-low"
        st.markdown(f"Confidence: **<span class='{confidence_class}'>{results['confidence']:.0%}</span>**", unsafe_allow_html=True)
        st.caption("Strength of signals based on indicators. Conflicts reduce confidence.")
        
        if results['entry_price'] is not None and results['stop_loss'] is not None and results['take_profit'] is not None and results['trade_action'] is not None:
            if results['trade_action'] == "BUY":
                risk_reward_ratio = abs((results['take_profit'] - results['entry_price']) / (results['entry_price'] - results['stop_loss']))
            else:
                risk_reward_ratio = abs((results['entry_price'] - results['take_profit']) / (results['stop_loss'] - results['entry_price']))
            
            st.markdown('<div class="trade-signal">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 {results['trade_action']} Trade Setup")
            st.markdown(f"**Entry Price (Current):** {results['entry_price']:.4f}")
            st.caption("Price to enter the trade.")
            st.markdown(f"**Stop Loss:** {results['stop_loss']:.4f}")
            st.caption("Exit to limit losses.")
            st.markdown(f"**Take Profit:** {results['take_profit']:.4f}")
            st.caption("Exit to lock in profits.")
            st.markdown(f"**Risk/Reward Ratio:** {risk_reward_ratio:.2f}:1")
            st.caption("Potential profit vs. risk.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if results['fib_signal_triggered'] and results['confidence'] < 0.38:
                st.warning("Confidence is low")
            else:
                st.warning("No trade setup generated. Conflicting signals or no Fibonacci confirmation.")
        
        signal_cols = st.columns(3)
        for i, (sig_tuple, desc) in enumerate(zip(results['signals'], results['signal_descriptions'])):
            signal, action, color, _ = sig_tuple
            with signal_cols[i % 3]:
                if action == "BUY":
                    st.markdown(f'<div class="signal-buy">✅ {signal}</div>', unsafe_allow_html=True)
                elif action == "SELL":
                    st.markdown(f'<div class="signal-sell">❌ {signal}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="signal-neutral">➡️ {signal}</div>', unsafe_allow_html=True)
                st.caption(desc)
    else:
        st.info("No strong trading signals detected.")
    
    st.caption(f"Last Completed Candle (UTC): {results['latest']['closetime']} | Data from {results['df']['closetime'].min()} to {results['df']['closetime'].max()}")
    
    st.plotly_chart(results['fig'], use_container_width=True)
    
    # Support & Resistance
    st.subheader("📊 Support & Resistance Levels")
    if results['levels']:
        st.markdown(f"**{results['level_type']} Levels based on {results['window']} candles:**")
        st.caption(f"{results['level_type']} levels from recent swing high/low.")
        
        st.dataframe(
            results['fib_df'].style.format({
                'Price Level': '{:.8f}',
                'Distance %': '{:.2f}%'
            }), 
            use_container_width=True
        )
        
        st.info(f"Closest {results['level_type']} level: {results['closest_level']['name']} at {results['closest_level']['Price Level']:.4f} ({results['closest_level']['Distance %']:.2f}% from current price)")
        
        st.subheader("📚 Fibonacci Level Analysis")
        st.markdown(results['fib_explanation_md'], unsafe_allow_html=True)
        
        with st.expander("📚 Understanding Fibonacci Levels"):
            st.markdown("""
            ### Fibonacci Retracement Levels
            
            Fibonacci levels indicate potential support/resistance based on Fibonacci ratios.
            
            **Common Levels:**
            - **23.6%**: Shallow level - quick pullback/bounce
            - **38.2%**: Moderate level - common reaction point
            - **50.0%**: Psychological level - often tested
            - **61.8%**: Golden ratio - strongest level
            - **78.6%**: Deep level - trend may weaken
            - **100%**: Full retracement - potential reversal
            
            **Interpretation:**
            - In **uptrend**, levels are **support** (buy zones)
            - In **downtrend**, levels are **resistance** (sell zones)
            - Trade setups require price within threshold and indicator confirmation
            """)
    else:
        st.info("Market is in sideways movement - No strong Fibonacci levels identified")
    
    with st.expander("📚 Understanding Window Size & RSI"):
        st.markdown(f"""
        ## Window Size vs. Fetched Data
        
        Fetched **{len(results['df'])} candles** with a window size of **{results['window']} candles**.
        
        ### What This Means:
        - Fibonacci levels use the **most recent {results['window']} candles**
        - Chart shows **{len(results['df'])} candles** of historical data
        - RSI uses all data with standard 30-70 levels
        
        ### RSI (Relative Strength Index)
        RSI measures momentum on a 0-100 scale.
        
        **RSI Levels:**
        - **>70**: Overbought (potential sell)
        - **<30**: Oversold (potential buy)
        - **30-70**: Neutral
        
        ### Window Size Strategy:
        - **Small (10-50 candles)**: Short-term levels
        - **Medium (50-200 candles)**: Balanced view
        - **Large (200+ candles)**: Long-term significant levels
        """)
    
    st.subheader("📈 Technical Indicators")
    indicator_col1, indicator_col2, indicator_col3, indicator_col4 = st.columns(4)
    
    with indicator_col1:
        rsi_color = "green" if results['rsi_value'] < 30 else "red" if results['rsi_value'] > 70 else "gray"
        st.metric("RSI", f"{results['rsi_value']:.2f}", delta=None, delta_color="normal")
        st.markdown(f"<span style='color:{rsi_color}'>30-70 range</span>", unsafe_allow_html=True)
        st.caption("Momentum indicator.")
    
    with indicator_col2:
        st.metric("EMA (20)", f"{results['latest']['ema']:.4f}")
        st.caption(f"Price is {results['ema_rel']} EMA")
        st.caption("Trend indicator.")
    
    with indicator_col3:
        st.metric("ATR", f"{results['atr']:.4f}")
        st.caption("Average True Range")
        st.caption("Measures volatility.")
    
    with indicator_col4:
        st.metric("BB %B", f"{results['bb_value']:.2f}%")
        st.caption(results['bb_status'])
        st.caption("Bollinger Bands position.")
    
    with st.expander("View Market Statistics"):
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write("**Price Statistics**")
            st.write(f"Open: {results['price_stats']['open']:.4f}")
            st.write(f"High: {results['price_stats']['high']:.4f}")
            st.write(f"Low: {results['price_stats']['low']:.4f}")
            st.write(f"Close: {results['price_stats']['close']:.4f}")
            st.write(f"Change: {results['price_stats']['change']:.2f}%")
            
        with stats_col2:
            st.write("**Volume Statistics**")
            st.write(f"Volume: {results['volume_stats']['volume']:.2f}")
            st.write(f"Quote Volume: {results['volume_stats']['quote_volume']:.2f}")
            st.write(f"Trades: {results['volume_stats']['trades']}")
    
    with st.expander("View Raw Data"):
        st.dataframe(results['df'], use_container_width=True)