import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from common import fetch_current_price, calculate_indicators

def detect_ichimoku_signals(df, current_price, interval):
    if len(df) < 52:
        return {'trend': 'none', 'confidence': 'low', 'signals': [], 'cloud_bullish': False, 'reasons_not_met': [], 'ichimoku_values': {}}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    # Trend determination using cloud at current candle (26 candles back)
    if len(df) >= 26:
        cloud_idx = -26
        cloud_span_a = df['Senkou_Span_A_unshifted'].iloc[cloud_idx] if pd.notnull(df['Senkou_Span_A_unshifted'].iloc[cloud_idx]) else last['close']
        cloud_span_b = df['Senkou_Span_B_unshifted'].iloc[cloud_idx] if pd.notnull(df['Senkou_Span_B_unshifted'].iloc[cloud_idx]) else last['close']
        cloud_top = max(cloud_span_a, cloud_span_b)
        cloud_bottom = min(cloud_span_a, cloud_span_b)
        cloud_bullish = cloud_span_a > cloud_span_b
    else:
        cloud_top = last['close']
        cloud_bottom = last['close']
        cloud_bullish = False

    if last['close'] > cloud_top:
        trend = 'uptrend'
    elif last['close'] < cloud_bottom:
        trend = 'downtrend'
    else:
        trend = 'sideways'

    # Future cloud color (26 candles ahead)
    future_cloud_bullish = False
    if pd.notnull(last['Senkou_Span_A']) and pd.notnull(last['Senkou_Span_B']):
        future_cloud_bullish = last['Senkou_Span_A'] > last['Senkou_Span_B']

    # TK Configuration (current position)
    bull_config = last['Tenkan_sen'] > last['Kijun_sen']
    bear_config = last['Tenkan_sen'] < last['Kijun_sen']

    # TK distance check
    tk_diff = abs(last['Tenkan_sen'] - last['Kijun_sen']) / last['atr'] if pd.notnull(last['atr']) and last['atr'] > 0 else 0
    tk_not_close = tk_diff >= 0.5

    # Chikou confirmation
    if len(df) > 26:
        close_26_ago = df['close'].iloc[-27]
        chikou_bull = current_price > close_26_ago
        chikou_bear = current_price < close_26_ago
    else:
        chikou_bull = False
        chikou_bear = False

    # Volume trend
    volume_rising = False
    volume_sell_increasing = False
    if len(df) > 5:
        recent_volumes = df['volume'].tail(5)
        volume_rising = recent_volumes.is_monotonic_increasing
        sell_volumes = df['volume'].tail(5)[df['close'].tail(5) < df['open'].tail(5)]
        if len(sell_volumes) >= 2:
            volume_sell_increasing = sell_volumes.is_monotonic_increasing

    # Momentum filters
    rsi_bull = last['rsi'] > 50 if pd.notnull(last['rsi']) else False
    rsi_bear = last['rsi'] < 50 if pd.notnull(last['rsi']) else False
    macd_bull = last['macd'] > 0 if pd.notnull(last['macd']) else False
    macd_bear = last['macd'] < 0 if pd.notnull(last['macd']) else False

    # Three Ichimoku confirmations
    bull_three_met = bull_config and tk_not_close and chikou_bull and trend == 'uptrend' and cloud_bullish
    bear_three_met = bear_config and tk_not_close and chikou_bear and trend == 'downtrend' and not cloud_bullish

    # Confidence score
    score = 0
    if bull_three_met or bear_three_met:
        score = 3
        if future_cloud_bullish if bull_three_met else not future_cloud_bullish:
            score += 1
        if volume_rising if bull_three_met else volume_sell_increasing:
            score += 1
        if (rsi_bull or macd_bull) if bull_three_met else (rsi_bear or macd_bear):
            score += 1

    confidence = 'high' if score >= 5 else 'medium' if score >= 4 else 'low'

    # Signals
    signals = []
    reasons_not_met = []
    if not (bull_config or bear_config):
        reasons_not_met.append("Current Tenkan-sen and Kijun-sen configuration not met.")
    if not tk_not_close:
        reasons_not_met.append("Tenkan-sen and Kijun-sen are too close.")
    if not (chikou_bull or chikou_bear):
        reasons_not_met.append("Chikou Span condition not met.")
    if not ((trend == 'uptrend' and cloud_bullish) or (trend == 'downtrend' and not cloud_bullish)):
        reasons_not_met.append("Price vs Cloud condition not met.")
    if len(reasons_not_met) >= 2:
        reasons_not_met.append("Multiple Ichimoku conditions not met.")

    atr = last['atr'] if pd.notnull(last['atr']) else 0
    reason_parts = []
    if bull_three_met and confidence in ['high', 'medium']:
        reason_parts.append("Tenkan above Kijun with distance, Price above Green Cloud, Chikou confirmation")
        if future_cloud_bullish:
            reason_parts.append("Future Green Cloud")
        if volume_rising:
            reason_parts.append("Rising Volume")
        if rsi_bull or macd_bull:
            reason_parts.append("Momentum (RSI or MACD) confirmed")
        reason = "Bullish Ichimoku: " + ", ".join(reason_parts)
        sl = current_price - 1.5 * atr
        tp = current_price + 3 * atr
        signals.append({
            'type': 'Buy',
            'entry_price': current_price,
            'sl': sl,
            'tp': tp,
            'reason': reason
        })
    elif bear_three_met and confidence in ['high', 'medium']:
        reason_parts.append("Tenkan below Kijun with distance, Price below Red Cloud, Chikou confirmation")
        if not future_cloud_bullish:
            reason_parts.append("Future Red Cloud")
        if volume_sell_increasing:
            reason_parts.append("Sell Volume Increasing")
        if rsi_bear or macd_bear:
            reason_parts.append("Momentum (RSI or MACD) confirmed")
        reason = "Bearish Ichimoku: " + ", ".join(reason_parts)
        sl = current_price + 1.5 * atr
        tp = current_price - 3 * atr
        signals.append({
            'type': 'Sell',
            'entry_price': current_price,
            'sl': sl,
            'tp': tp,
            'reason': reason
        })

    ichimoku_values = {
        'Tenkan_sen': last['Tenkan_sen'],
        'Kijun_sen': last['Kijun_sen'],
        'Senkou_Span_A': last['Senkou_Span_A'] if pd.notnull(last['Senkou_Span_A']) else None,
        'Senkou_Span_B': last['Senkou_Span_B'] if pd.notnull(last['Senkou_Span_B']) else None,
        'close': last['close']
    }

    return {
        'trend': trend,
        'confidence': confidence,
        'signals': signals,
        'cloud_bullish': cloud_bullish,
        'reasons_not_met': reasons_not_met,
        'ichimoku_values': ichimoku_values
    }

def create_ichimoku_chart(raw_df, df_live, trend, indicators_to_show, show_ema, show_bb, interval):
    num_subplots = len(indicators_to_show) + 1
    row_heights = [0.6] + [0.4 / len(indicators_to_show)] * len(indicators_to_show) if indicators_to_show else [1.0]
    
    subplot_titles = [f'Ichimoku Cloud Analysis - {trend.capitalize()}']
    subplot_titles.extend(indicators_to_show)
    
    fig = make_subplots(
        rows=num_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    fig.add_trace(
        go.Candlestick(
            x=raw_df['closetime'],
            open=raw_df['open'],
            high=raw_df['high'],
            low=raw_df['low'],
            close=raw_df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['ema'],
                name='EMA (20)',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
    
    if show_bb:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['bb_upper'],
                name='BB Upper',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['bb_lower'],
                name='BB Lower',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(200, 200, 200, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Ichimoku Lines
    fig.add_trace(
        go.Scatter(
            x=df_live['closetime'],
            y=df_live['Tenkan_sen'],
            name='Tenkan-sen',
            line=dict(color='#0000FF', width=1.5)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_live['closetime'],
            y=df_live['Kijun_sen'],
            name='Kijun-sen',
            line=dict(color='#FF0000', width=1.5)
        ),
        row=1, col=1
    )
    
    # Chikou Span
    ichimoku_kijun = 26
    if len(df_live) > ichimoku_kijun:
        chikou_x = df_live['closetime'][:-ichimoku_kijun]
        chikou_y = df_live['close'][ichimoku_kijun:]
        fig.add_trace(
            go.Scatter(
                x=chikou_x,
                y=chikou_y,
                name='Chikou Span',
                line=dict(color='#800080', width=1.5)
            ),
            row=1, col=1
        )
    
    # Calculate time shift for cloud
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    period_minutes = interval_minutes.get(interval, 5)
    shift_timedelta = timedelta(minutes=period_minutes * ichimoku_kijun)
    
    # Shift cloud x-axis
    cloud_closetime = df_live['closetime'] + shift_timedelta
    
    # Plot Senkou Span lines
    fig.add_trace(
        go.Scatter(
            x=cloud_closetime,
            y=df_live['Senkou_Span_A'],
            name='Senkou Span A',
            line=dict(color='#00FF00', width=1, dash='dot')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=cloud_closetime,
            y=df_live['Senkou_Span_B'],
            name='Senkou Span B',
            line=dict(color='#FF0000', width=1, dash='dot')
        ),
        row=1, col=1
    )
    
    # Single cloud fill with dynamic color
    df_cloud = df_live[['closetime', 'Senkou_Span_A', 'Senkou_Span_B']].copy()
    df_cloud['closetime'] = cloud_closetime
    df_cloud['upper'] = df_cloud[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1)
    df_cloud['lower'] = df_cloud[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)
    df_cloud['color'] = np.where(df_cloud['Senkou_Span_A'] >= df_cloud['Senkou_Span_B'], 'rgba(144, 238, 144, 0.15)', 'rgba(255, 182, 193, 0.15)')
    
    # Split cloud into segments
    segments = []
    current_segment = {'closetime': [], 'upper': [], 'lower': [], 'color': df_cloud['color'].iloc[0]}
    for i in range(len(df_cloud)):
        if i > 0 and (df_cloud['color'].iloc[i] != current_segment['color'] or pd.isna(df_cloud['upper'].iloc[i]) or pd.isna(df_cloud['lower'].iloc[i])):
            if current_segment['closetime']:
                segments.append(current_segment)
            current_segment = {'closetime': [], 'upper': [], 'lower': [], 'color': df_cloud['color'].iloc[i] if not pd.isna(df_cloud['upper'].iloc[i]) else None}
        if not pd.isna(df_cloud['upper'].iloc[i]) and not pd.isna(df_cloud['lower'].iloc[i]):
            current_segment['closetime'].append(df_cloud['closetime'].iloc[i])
            current_segment['upper'].append(df_cloud['upper'].iloc[i])
            current_segment['lower'].append(df_cloud['lower'].iloc[i])
    if current_segment['closetime']:
        segments.append(current_segment)
    
    # Add cloud segments
    cloud_legend_added = False
    for segment in segments:
        if segment['color']:
            fig.add_trace(
                go.Scatter(
                    x=segment['closetime'],
                    y=segment['upper'],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=segment['closetime'],
                    y=segment['lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor=segment['color'],
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Cloud',
                    showlegend=not cloud_legend_added
                ),
                row=1, col=1
            )
            cloud_legend_added = True
    
    current_row = 2
    if "Volume" in indicators_to_show:
        colors_volume = ['red' if row['open'] > row['close'] else 'green' for _, row in raw_df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=raw_df['closetime'],
                y=raw_df['volume'],
                name='Volume',
                marker_color=colors_volume
            ),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1
    
    if "RSI" in indicators_to_show:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['rsi'],
                name='RSI',
                line=dict(color='purple', width=1.5)
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1, annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1
    
    if "MACD" in indicators_to_show:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd'],
                name='MACD Line',
                line=dict(color='blue', width=1.5)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd_signal'],
                name='Signal Line',
                line=dict(color='orange', width=1.5)
            ),
            row=current_row, col=1
        )
        histogram_colors = ['green' if val >= 0 else 'red' for val in df_live['macd_histogram']]
        fig.add_trace(
            go.Bar(
                x=df_live['closetime'],
                y=df_live['macd_histogram'],
                name='MACD Histogram',
                marker_color=histogram_colors,
                opacity=0.5
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    fig.update_layout(
        height=900,
        title=f"Ichimoku Cloud Analysis - {trend.capitalize()}",
        yaxis_title='Price',
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode='zoom',
        hovermode='x unified'
    )
    fig.update_xaxes(
        title_text="Date",
        row=num_subplots, col=1,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    if len(raw_df) > 0:
        initial_start_idx = max(0, len(raw_df) - 200)
        initial_start = raw_df['closetime'].iloc[initial_start_idx]
        initial_end = raw_df['closetime'].iloc[-1] + shift_timedelta
        for r in range(1, num_subplots + 1):
            fig.update_xaxes(range=[initial_start, initial_end], row=r, col=1, matches='x')
    
    return fig

def compute_ichimoku_results(df, params, symbol, interval):
    tenkan_period = params.get('tenkan_period', 9)
    kijun_period = params.get('kijun_period', 26)
    senkou_b_period = params.get('senkou_b_period', 52)
    show_ema = params.get('show_ema', True)
    show_bb = params.get('show_bb', True)
    show_volume = params.get('show_volume', True)
    show_rsi = params.get('show_rsi', True)
    show_macd = params.get('show_macd', True)

    raw_df = df.copy()
    df = calculate_indicators(
        df,
        interval,  # Fixed: Added interval parameter
        ichimoku_tenkan=tenkan_period,
        ichimoku_kijun=kijun_period,
        ichimoku_senkou_b=senkou_b_period
    )

    current_price = fetch_current_price(symbol) or df['close'].iloc[-1]
    ichimoku_data = detect_ichimoku_signals(df, current_price, interval)

    indicators_to_show = []
    if show_volume:
        indicators_to_show.append("Volume")
    if show_rsi:
        indicators_to_show.append("RSI")
    if show_macd:
        indicators_to_show.append("MACD")

    fig = create_ichimoku_chart(
        raw_df, df, ichimoku_data['trend'], indicators_to_show, show_ema, show_bb, interval
    )

    return {
        'trend': ichimoku_data['trend'],
        'confidence': ichimoku_data['confidence'],
        'signals': ichimoku_data['signals'],
        'cloud_bullish': ichimoku_data['cloud_bullish'],
        'reasons_not_met': ichimoku_data['reasons_not_met'],
        'ichimoku_values': ichimoku_data['ichimoku_values'],
        'raw_df': raw_df,
        'df': df,
        'fig': fig,
        'symbol': symbol,
        'interval': interval,
        'show_ema': show_ema,
        'show_bb': show_bb,
        'show_volume': show_volume,
        'show_rsi': show_rsi,
        'show_macd': show_macd
    }

def display_ichimoku_results(results):
    trend = results['trend']
    confidence = results['confidence']
    signals = results['signals']
    reasons_not_met = results['reasons_not_met']
    ichimoku_values = results['ichimoku_values']
    raw_df = results['raw_df']
    df = results['df']
    fig = results['fig']
    symbol = results['symbol']
    interval = results['interval']

    st.subheader("📊 Ichimoku Summary")
    st.markdown(f"Trend: <span class='trend-{trend}'>{trend.capitalize()}</span>", unsafe_allow_html=True)
    st.markdown(f"Confidence: <span class='confidence-{confidence}'>{confidence.capitalize()}</span>", unsafe_allow_html=True)
    st.info(f"Fetched {len(df)} candles for analysis.")

    st.subheader("Current Ichimoku Values")
    ichimoku_df = pd.DataFrame([ichimoku_values])
    st.dataframe(ichimoku_df.style.format({
        'Tenkan_sen': '{:.4f}',
        'Kijun_sen': '{:.4f}',
        'Senkou_Span_A': '{:.4f}',
        'Senkou_Span_B': '{:.4f}',
        'close': '{:.4f}'
    }), use_container_width=True)

    st.subheader("🎯 Trading Signals")
    is_consolidating = trend == 'sideways'
    if signals:
        for sig in signals:
            color = 'buy' if sig['type'] == 'Buy' else 'sell'
            st.markdown(f'<div class="signal-{color}">{sig["type"]} Signal: {sig["reason"]}<br>Entry ~ {sig["entry_price"]:.2f}, SL: {sig["sl"]:.2f}, TP: {sig["tp"]:.2f}</div>', unsafe_allow_html=True)
    elif is_consolidating:
        st.markdown('<div class="signal-neutral">Market in sideways trend (inside cloud). Await breakout.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-neutral">No strong signal. Monitor for TK crossover or cloud breakout.</div>', unsafe_allow_html=True)
        if reasons_not_met:
            st.warning("\n".join(reasons_not_met))

    st.subheader("Signal Logic Explanation")
    st.markdown("""
    **Bullish Entry (Buy):**
    - Conversion Line (Tenkan-sen) above Base Line (Kijun-sen) with sufficient distance.
    - Price is above the Cloud 26 candles ago, and Cloud is green (Senkou Span A > Senkou Span B).
    - Lagging Line (Chikou Span) is above the price 26 periods ago.
    - Future Cloud (26 periods ahead) is green (optional for higher confidence).
    - Volume is rising (optional).
    - Momentum: RSI > 50 or MACD > 0 (optional).

    **Bearish Entry (Sell):**
    - Conversion Line (Tenkan-sen) below Base Line (Kijun-sen) with sufficient distance.
    - Price is below the Cloud 26 candles ago, and Cloud is red (Senkou Span A < Senkou Span B).
    - Lagging Line (Chikou Span) is below the price 26 periods ago.
    - Future Cloud is red (optional for higher confidence).
    - Volume increasing on sell candles (optional).
    - Momentum: RSI < 50 or MACD < 0 (optional).

    **SL and TP Calculation:**
    - Stop Loss (SL): 1.5 x ATR from entry price.
    - Take Profit (TP): 3 x ATR from entry price.

    Entries require three Ichimoku confirmations (TK config, price vs cloud, Chikou). Confidence: 3 for core conditions, +1 for future cloud, +1 for volume, +1 for momentum. High (>=5), medium (>=4), low (<4). Entries on high/medium confidence.
    """)

    st.plotly_chart(fig, use_container_width=True, key=f"chart_{symbol}_{interval}")

    st.subheader("📈 Technical Indicators")
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi_value = latest['rsi']
        st.metric("RSI", f"{rsi_value:.2f}")
        st.caption("30-70 range")
    with col2:
        st.metric("EMA (20)", f"{latest['ema']:.4f}")
        ema_rel = "Above" if latest['close'] > latest['ema'] else "Below"
        st.caption(f"Price {ema_rel} EMA")
    with col3:
        st.metric("ATR", f"{latest['atr']:.4f}")
        st.caption("Volatility")
    with col4:
        bb_value = latest['bb_percentB'] * 100
        st.metric("BB %B", f"{bb_value:.2f}%")
        st.caption("Bands position")

    with st.expander("View Raw Data"):
        st.dataframe(raw_df.tail(20), use_container_width=True)