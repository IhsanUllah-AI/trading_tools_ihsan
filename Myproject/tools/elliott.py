import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from common import fetch_current_price, calculate_indicators

def find_pivots(df, delta):
    pivots = []
    if len(df) == 0:
        return pivots

    last_pivot_idx = 0
    last_pivot_price = df['close'].iloc[0]
    last_pivot_type = None
    direction = 0

    for i in range(1, len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if high > last_pivot_price:
            direction = 1
            last_pivot_price = df['low'].iloc[0]
            last_pivot_type = 'L'
            pivots.append((0, last_pivot_price, 'L'))
            last_pivot_idx = 0
            break
        elif low < last_pivot_price:
            direction = -1
            last_pivot_price = df['high'].iloc[0]
            last_pivot_type = 'H'
            pivots.append((0, last_pivot_price, 'H'))
            last_pivot_idx = 0
            break

    if direction == 0:
        return pivots

    for i in range(last_pivot_idx + 1, len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        reversal_amount = last_pivot_price * delta

        if direction == 1:
            if high > last_pivot_price:
                last_pivot_price = high
                last_pivot_idx = i
            if low < last_pivot_price - reversal_amount:
                pivots.append((last_pivot_idx, df['high'].iloc[last_pivot_idx], 'H'))
                direction = -1
                last_pivot_price = low
                last_pivot_idx = i
                last_pivot_type = 'H'
        elif direction == -1:
            if low < last_pivot_price:
                last_pivot_price = low
                last_pivot_idx = i
            if high > last_pivot_price + reversal_amount:
                pivots.append((last_pivot_idx, df['low'].iloc[last_pivot_idx], 'L'))
                direction = 1
                last_pivot_price = high
                last_pivot_idx = i
                last_pivot_type = 'L'

    if direction == 1:
        pivots.append((last_pivot_idx, df['high'].iloc[last_pivot_idx], 'H?'))
    elif direction == -1:
        pivots.append((last_pivot_idx, df['low'].iloc[last_pivot_idx], 'L?'))

    return pivots

def detect_elliott_waves_from_pivots(pivots, df):
    if len(pivots) < 3:
        return {'historical_patterns': [], 'partial_waves': {}, 'confidence': 'low', 'direction': 'none', 'fib_historical': {}, 'fib_partial': {}, 'signals': []}

    historical_patterns = []
    partial_waves = {}
    confidence = 'low'
    direction = 'none'
    fib_historical = {}
    fib_partial = {}
    signals = []

    historical_completes = find_all_complete_impulses(pivots)
    for historical_complete in historical_completes:
        if historical_complete['waves']:
            pattern = {'impulse': historical_complete['waves'], 'fib_impulse': historical_complete['fib']}
            if 'confidence' in historical_complete and historical_complete['confidence'] > confidence:
                confidence = historical_complete['confidence']
            if 'direction' in historical_complete:
                direction = historical_complete['direction']

            start_idx = historical_complete['start_idx'] + 6
            fib_abc = {}
            abc = {}
            if len(pivots) - start_idx >= 3:
                abc_data = detect_abc_correction(pivots[start_idx:start_idx+3], historical_complete['direction'], historical_complete['waves']['5']['price'])
                if abc_data['abc']:
                    abc = abc_data['abc']
                    fib_abc = abc_data['fib_abc']
                    confidence = 'high' if confidence == 'medium' else confidence
            pattern['abc'] = abc
            pattern['fib_abc'] = fib_abc
            historical_patterns.append(pattern)

    if historical_patterns:
        last_historical = historical_completes[-1]
        confidence = last_historical['confidence']
        direction = last_historical['direction']
        last_pattern = historical_patterns[-1]
        fib_historical = {**last_pattern['fib_impulse'], **last_pattern['fib_abc']}

    if len(pivots) >= 5:
        partial_data = detect_partial_up_to_wave4(pivots[-5:], df)
        if partial_data['partial_waves']:
            partial_waves = partial_data['partial_waves']
            confidence = max(confidence, partial_data['confidence']) if isinstance(confidence, str) else partial_data['confidence']
            direction = partial_data['direction']
            fib_partial.update(partial_data['fib'])
            signals = partial_data['signals']

    if not partial_waves and len(pivots) >= 3:
        partial_data = detect_partial_up_to_wave2(pivots[-3:], df)
        if partial_data['partial_waves']:
            partial_waves = partial_data['partial_waves']
            confidence = max(confidence, partial_data['confidence']) if isinstance(confidence, str) else partial_data['confidence']
            direction = partial_data['direction']
            fib_partial.update(partial_data['fib'])
            signals = partial_data['signals']

    return {'historical_patterns': historical_patterns, 'partial_waves': partial_waves, 'confidence': confidence, 'direction': direction, 'fib_historical': fib_historical, 'fib_partial': fib_partial, 'signals': signals}

def find_all_complete_impulses(pivots):
    completes = []
    i = 0
    while i <= len(pivots) - 6:
        candidate_pivots = pivots[i:i+6]
        complete_data = detect_complete_impulse(candidate_pivots)
        if complete_data['waves']:
            complete_data['start_idx'] = i
            completes.append(complete_data)
            i += 6
        else:
            i += 1
    return completes

def detect_complete_impulse(last_pivots):
    types = [p[2].replace('?', '') for p in last_pivots]
    if types != ['L', 'H', 'L', 'H', 'L', 'H'] and types != ['H', 'L', 'H', 'L', 'H', 'L']:
        return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    is_bullish = types[0] == 'L' and last_pivots[-1][1] > last_pivots[0][1]
    is_bearish = types[0] == 'H' and last_pivots[-1][1] < last_pivots[0][1]

    if not (is_bullish or is_bearish):
        return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    direction = 'bullish' if is_bullish else 'bearish'
    p0, p1, p2, p3, p4, p5 = [p[1] for p in last_pivots]

    w1 = abs(p1 - p0)
    w2 = abs(p2 - p1)
    w3 = abs(p3 - p2)
    w4 = abs(p4 - p3)
    w5 = abs(p5 - p4)

    if is_bullish:
        if p2 <= p0 or p4 <= p1:
            return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}
    else:
        if p2 >= p0 or p4 >= p1:
            return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if w3 <= min(w1, w5):
        return {'waves': {}, 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if is_bullish:
        w2_retr = (p1 - p2) / (p1 - p0)
        w3_ext = w3 / w1
        w4_retr = (p3 - p4) / (p3 - p2)
        w5_ext = w5 / w1
    else:
        w2_retr = (p2 - p1) / (p0 - p1)
        w3_ext = w3 / w1
        w4_retr = (p4 - p3) / (p2 - p3)
        w5_ext = w5 / w1

    fib = {
        'Wave 2 Retrace': w2_retr,
        'Wave 3 Extension': w3_ext,
        'Wave 4 Retrace': w4_retr,
        'Wave 5 Extension': w5_ext
    }

    score = 0
    if 0.382 <= w2_retr <= 0.618:
        score += 2
    elif 0.236 <= w2_retr <= 0.786:
        score += 1

    if 1.0 <= w3_ext <= 2.618:
        score += 2 if abs(w3_ext - 1.618) < 0.3 else 1

    if 0.236 <= w4_retr <= 0.382:
        score += 1

    if abs(w5_ext - 1.0) < 0.2 or abs(w5_ext - 0.618) < 0.2 or abs(w5_ext - 1.618) < 0.2:
        score += 1

    confidence = 'high' if score >= 4 else 'medium' if score >= 3 else 'low'

    waves = {}
    labels = ['0', '1', '2', '3', '4', '5']
    for j, label in enumerate(labels):
        idx, price, ptype = last_pivots[j]
        waves[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}

    return {'waves': waves, 'confidence': confidence, 'direction': direction, 'fib': fib}

def detect_partial_up_to_wave2(last_pivots, df):
    types = [p[2].replace('?', '') for p in last_pivots]
    if types != ['L', 'H', 'L'] and types != ['H', 'L', 'H']:
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    is_bullish = types[0] == 'L' and last_pivots[1][1] > last_pivots[0][1]
    is_bearish = types[0] == 'H' and last_pivots[1][1] < last_pivots[0][1]

    if not (is_bullish or is_bearish):
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    direction = 'bullish' if is_bullish else 'bearish'
    p0, p1, p2 = [p[1] for p in last_pivots]

    w1 = abs(p1 - p0)
    w2 = abs(p2 - p1)

    if is_bullish:
        if p2 <= p0:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}
    else:
        if p2 >= p0:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if is_bullish:
        w2_retr = (p1 - p2) / (p1 - p0)
    else:
        w2_retr = (p2 - p1) / (p0 - p1)

    fib = {'Wave 2 Retrace': w2_retr}

    score = 0
    if 0.382 <= w2_retr <= 0.618:
        score += 2
    elif 0.236 <= w2_retr <= 0.786:
        score += 1

    last_idx = last_pivots[-1][0]
    rsi_at_w2 = df['rsi'].iloc[last_idx]
    if is_bullish and rsi_at_w2 > 30:
        score += 1
    elif is_bearish and rsi_at_w2 < 70:
        score += 1

    confidence = 'high' if score >= 4 else 'medium' if score >= 3 else 'low'

    partial_waves = {}
    labels = ['0?', '1?', '2?']
    for j, label in enumerate(labels):
        idx, price, ptype = last_pivots[j]
        partial_waves[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}

    signals = []
    if confidence != 'low':
        atr = df['atr'].iloc[-1]
        current_price = df['close'].iloc[-1]
        if is_bullish:
            sl = p2 - atr
            tp = p2 + 1.0 * w1
            signals.append({'action': 'Buy', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 2?, expect Wave 3 up', 'degree': 'Minor'})
        else:
            sl = p2 + atr
            tp = p2 - 1.0 * w1
            signals.append({'action': 'Sell', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 2?, expect Wave 3 down', 'degree': 'Minor'})

    return {'partial_waves': partial_waves, 'signals': signals, 'confidence': confidence, 'direction': direction, 'fib': fib}

def detect_partial_up_to_wave4(last_pivots, df):
    types = [p[2].replace('?', '') for p in last_pivots]
    if types != ['L', 'H', 'L', 'H', 'L'] and types != ['H', 'L', 'H', 'L', 'H']:
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    is_bullish = types[0] == 'L' and last_pivots[3][1] > last_pivots[1][1]
    is_bearish = types[0] == 'H' and last_pivots[3][1] < last_pivots[1][1]

    if not (is_bullish or is_bearish):
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    direction = 'bullish' if is_bullish else 'bearish'
    p0, p1, p2, p3, p4 = [p[1] for p in last_pivots]

    w1 = abs(p1 - p0)
    w2 = abs(p2 - p1)
    w3 = abs(p3 - p2)
    w4 = abs(p4 - p3)

    if is_bullish:
        if p2 <= p0 or p4 <= p1:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}
    else:
        if p2 >= p0 or p4 >= p1:
            return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if w3 <= w1:
        return {'partial_waves': {}, 'signals': [], 'confidence': 'low', 'direction': 'none', 'fib': {}}

    if is_bullish:
        w2_retr = (p1 - p2) / (p1 - p0)
        w3_ext = w3 / w1
        w4_retr = (p3 - p4) / (p3 - p2)
    else:
        w2_retr = (p2 - p1) / (p0 - p1)
        w3_ext = w3 / w1
        w4_retr = (p4 - p3) / (p2 - p3)

    fib = {
        'Wave 2 Retrace': w2_retr,
        'Wave 3 Extension': w3_ext,
        'Wave 4 Retrace': w4_retr
    }

    score = 0
    if 0.382 <= w2_retr <= 0.618:
        score += 2
    elif 0.236 <= w2_retr <= 0.786:
        score += 1

    if 1.0 <= w3_ext <= 2.618:
        score += 2 if abs(w3_ext - 1.618) < 0.3 else 1

    if 0.236 <= w4_retr <= 0.382:
        score += 1

    last_idx = last_pivots[-1][0]
    rsi_at_w4 = df['rsi'].iloc[last_idx]
    if is_bullish and rsi_at_w4 > 30:
        score += 1
    elif is_bearish and rsi_at_w4 < 70:
        score += 1

    confidence = 'high' if score >= 4 else 'medium' if score >= 3 else 'low'

    partial_waves = {}
    labels = ['0?', '1?', '2?', '3?', '4?']
    for j, label in enumerate(labels):
        idx, price, ptype = last_pivots[j]
        partial_waves[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}

    signals = []
    if confidence != 'low':
        atr = df['atr'].iloc[-1]
        current_price = df['close'].iloc[-1]
        if is_bullish:
            sl = p4 - atr
            tp = p4 + 0.38 * w1
            signals.append({'action': 'Buy', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 4?, expect Wave 5 up', 'degree': 'Minor'})
        else:
            sl = p4 + atr
            tp = p4 - 0.38 * w1
            signals.append({'action': 'Sell', 'entry_price': current_price, 'sl': sl, 'tp': tp, 'reason': 'After Wave 4?, expect Wave 5 down', 'degree': 'Minor'})

    return {'partial_waves': partial_waves, 'signals': signals, 'confidence': confidence, 'direction': direction, 'fib': fib}

def detect_abc_correction(last_pivots, direction, wave5_price):
    types = [p[2].replace('?', '') for p in last_pivots]
    pa, pb, pc = [p[1] for p in last_pivots]

    add_abc = False
    fib_abc = {}
    if direction == 'bullish' and types == ['L', 'H', 'L']:
        add_abc = True
        a_len = wave5_price - pa
        b_retr = (pb - pa) / a_len
        c_len = pb - pc
        c_ext = c_len / a_len
    elif direction == 'bearish' and types == ['H', 'L', 'H']:
        add_abc = True
        a_len = pa - wave5_price
        b_retr = (pa - pb) / a_len
        c_len = pc - pb
        c_ext = c_len / a_len

    if add_abc:
        score = 0
        if 0.382 < b_retr < 0.786:
            score += 1
        if abs(c_ext - 1.0) < 0.2 or abs(c_ext - 1.618) < 0.2:
            score += 1

        if score > 0:
            fib_abc = {'B Retrace': b_retr, 'C Extension': c_ext}
            abc = {}
            labels = ['A', 'B', 'C']
            for j, label in enumerate(labels):
                idx, price, ptype = last_pivots[j]
                abc[label] = {'index': idx, 'price': price, 'type': ptype.replace('?', '')}
            return {'abc': abc, 'fib_abc': fib_abc}

    return {'abc': {}, 'fib_abc': {}}

def create_elliott_chart(raw_df, df_live, historical_patterns, partial_waves, trend, indicators_to_show, show_ema, show_bb):
    num_subplots = len(indicators_to_show) + 1
    row_heights = [0.6] + [0.4 / len(indicators_to_show)] * len(indicators_to_show) if indicators_to_show else [1.0]
    
    subplot_titles = [f'Elliott Wave Analysis - {trend.capitalize()}']
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
                name=f'EMA (20)',
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
    
    price_range = raw_df['high'].max() - raw_df['low'].min()
    offset = price_range * 0.05
    
    colors_wave = {'0': 'gray', '1': 'blue', '2': 'red', '3': 'blue', '4': 'red', '5': 'blue', 
                   'A': 'green', 'B': 'red', 'C': 'green', '0?': 'gray', '1?': 'blue', '2?': 'red', '3?': 'blue', '4?': 'red'}
    for hist_idx, pattern in enumerate(historical_patterns):
        prefix = f"H{hist_idx+1}-"
        wave_points = []
        for label in ['0', '1', '2', '3', '4', '5']:
            if label in pattern['impulse']:
                idx = pattern['impulse'][label]['index']
                price = pattern['impulse'][label]['price']
                ptype = pattern['impulse'][label]['type']
                offset_price = price + offset if ptype == 'H' else price - offset
                wave_points.append((df_live['closetime'].iloc[idx], offset_price))
                fig.add_annotation(
                    x=df_live['closetime'].iloc[idx],
                    y=offset_price,
                    text=f"{prefix}{label}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=colors_wave.get(label, 'gray'),
                    opacity=0.6,
                    ax=0,
                    ay=-30 if ptype == 'H' else 30,
                    row=1, col=1
                )
        
        if wave_points:
            x_vals, y_vals = zip(*wave_points)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=f'{prefix}Impulse',
                    line=dict(color='blue', width=2, dash='solid'),
                    opacity=0.5,
                    showlegend=(hist_idx == 0)
                ),
                row=1, col=1
            )

        if pattern['abc']:
            abc_prefix = f"{prefix}ABC-"
            abc_points = []
            for label in ['A', 'B', 'C']:
                if label in pattern['abc']:
                    idx = pattern['abc'][label]['index']
                    price = pattern['abc'][label]['price']
                    ptype = pattern['abc'][label]['type']
                    offset_price = price + offset if ptype == 'H' else price - offset
                    abc_points.append((df_live['closetime'].iloc[idx], offset_price))
                    fig.add_annotation(
                        x=df_live['closetime'].iloc[idx],
                        y=offset_price,
                        text=f"{abc_prefix}{label}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=colors_wave.get(label, 'green'),
                        opacity=0.6,
                        ax=0,
                        ay=-30 if ptype == 'H' else 30,
                        row=1, col=1
                    )
            
            if abc_points:
                x_vals, y_vals = zip(*abc_points)
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name=f'{abc_prefix}',
                        line=dict(color='green', width=2, dash='solid'),
                        opacity=0.5,
                        showlegend=(hist_idx == 0)
                    ),
                    row=1, col=1
                )
    
    partial_points = []
    for label in ['0?', '1?', '2?', '3?', '4?']:
        if label in partial_waves:
            idx = partial_waves[label]['index']
            price = partial_waves[label]['price']
            ptype = partial_waves[label]['type']
            offset_price = price + offset if ptype == 'H' else price - offset
            partial_points.append((df_live['closetime'].iloc[idx], offset_price))
            fig.add_annotation(
                x=df_live['closetime'].iloc[idx],
                y=offset_price,
                text=label,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors_wave.get(label, 'gray'),
                ax=0,
                ay=-30 if ptype == 'H' else 30,
                row=1, col=1
            )
    
    if partial_points:
        x_vals, y_vals = zip(*partial_points)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name='Current Partial',
                line=dict(color='yellow', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
    
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
    
    if "MACD" in indicators_to_show and all(col in df_live.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
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
        title=f"Elliott Wave Analysis - {trend.capitalize()}",
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode='zoom',
        hovermode='x unified'
    )
    fig.update_xaxes(
        title_text="Date", 
        row=num_subplots, col=1,
        rangeslider_visible=True,
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
        initial_end = raw_df['closetime'].iloc[-1]
        fig.update_xaxes(range=[initial_start, initial_end], row=1, col=1)
    
    return fig

def compute_elliott_results(df, params, symbol, interval):
    thresholds = params.get('thresholds', {})
    selected_degrees = params.get('selected_degrees', ['Minor'])
    use_smoothing = params.get('use_smoothing', False)
    smooth_period = params.get('smooth_period', 3)
    show_ema = params.get('show_ema', True)
    show_bb = params.get('show_bb', True)
    show_volume = params.get('show_volume', True)
    show_rsi = params.get('show_rsi', True)
    show_macd = params.get('show_macd', True)

    raw_df = df.copy()
    if use_smoothing:
        df['high'] = df['high'].ewm(span=smooth_period, adjust=False).mean()
        df['low'] = df['low'].ewm(span=smooth_period, adjust=False).mean()
        df['close'] = df['close'].ewm(span=smooth_period, adjust=False).mean()

    df = calculate_indicators(df, interval)  # Fixed: Added interval parameter

    wave_data_by_degree = {}
    for degree in selected_degrees:
        delta = thresholds.get(degree, 0.005)
        pivots = find_pivots(df, delta)
        wave_data = detect_elliott_waves_from_pivots(pivots, df)
        wave_data['pivots'] = pivots
        wave_data_by_degree[degree] = wave_data

    return {
        'wave_data_by_degree': wave_data_by_degree,
        'raw_df': raw_df,
        'df': df,
        'show_ema': show_ema,
        'show_bb': show_bb,
        'show_volume': show_volume,
        'show_rsi': show_rsi,
        'show_macd': show_macd,
        'symbol': symbol,
        'interval': interval,
        'signals': wave_data_by_degree.get('Minor', {}).get('signals', [])
    }

def display_elliott_results(results):
    wave_data_by_degree = results['wave_data_by_degree']
    raw_df = results['raw_df']
    df = results['df']
    show_ema = results['show_ema']
    show_bb = results['show_bb']
    show_volume = results['show_volume']
    show_rsi = results['show_rsi']
    show_macd = results['show_macd']
    symbol = results['symbol']
    interval = results['interval']
    signals = results['signals']

    tabs = st.tabs(list(wave_data_by_degree.keys()))
    for i, degree in enumerate(wave_data_by_degree):
        with tabs[i]:
            wave_data = wave_data_by_degree[degree]
            historical_patterns = wave_data['historical_patterns']
            partial_waves = wave_data['partial_waves']
            confidence = wave_data['confidence']
            trend = wave_data['direction']
            fib_historical = wave_data['fib_historical']
            fib_partial = wave_data['fib_partial']
            pivots = wave_data['pivots']

            st.subheader(f"📊 Wave Summary ({degree} Degree)")
            st.markdown(f"Confidence: <span class='confidence-{confidence}'>{confidence.capitalize()}</span>", unsafe_allow_html=True)
            st.info(f"Detected {len(historical_patterns)} complete historical patterns. Total pivots: {len(pivots)}.")

            if historical_patterns:
                st.subheader("Recent Historical Complete Pattern")
                recent_pattern = historical_patterns[-1]
                recent_impulse = recent_pattern['impulse']
                recent_abc = recent_pattern['abc']
                hist_all = {**recent_impulse, **recent_abc}
                wave_df_hist = pd.DataFrame.from_dict(hist_all, orient='index')
                st.dataframe(wave_df_hist[['price', 'type']], use_container_width=True)

            if partial_waves:
                st.subheader("Current Partial Pattern")
                wave_df_partial = pd.DataFrame.from_dict(partial_waves, orient='index')
                st.dataframe(wave_df_partial[['price', 'type']], use_container_width=True)

            if fib_historical:
                st.subheader("Historical Fibonacci Analysis (Recent Complete Pattern)")
                fib_hist_df = pd.DataFrame(fib_historical.items(), columns=['Metric', 'Value'])
                fib_hist_df['Value'] = fib_hist_df['Value'].apply(lambda x: f"{x:.3f}")
                st.table(fib_hist_df)

            if fib_partial:
                st.subheader("Partial Fibonacci Analysis (Incomplete Pattern)")
                fib_part_df = pd.DataFrame(fib_partial.items(), columns=['Metric', 'Value'])
                fib_part_df['Value'] = fib_part_df['Value'].apply(lambda x: f"{x:.3f}")
                st.table(fib_part_df)

            st.subheader("🎯 Trading Signals (Based on Current Partial)")
            is_consolidating = False
            if not partial_waves and historical_patterns:
                recent_candles = df.iloc[-10:]
                price_range = recent_candles['high'].max() - recent_candles['low'].min()
                atr = recent_candles['atr'].iloc[-1]
                if price_range < 2 * atr:
                    is_consolidating = True

            if signals:
                for sig in signals:
                    color = 'buy' if sig['action'] == 'Buy' else 'sell'
                    st.markdown(f'<div class="signal-{color}">{sig["action"]} Signal: {sig["reason"]}<br>Entry ~ {sig["entry_price"]:.2f}, SL: {sig["sl"]:.2f}, TP: {sig["tp"]:.2f}</div>', unsafe_allow_html=True)
            elif is_consolidating:
                st.markdown('<div class="signal-neutral">Market in consolidation. Await breakout.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-neutral">Monitor for pattern completion.</div>', unsafe_allow_html=True)

            indicators_to_show = []
            if show_volume:
                indicators_to_show.append("Volume")
            if show_rsi:
                indicators_to_show.append("RSI")
            if show_macd:
                indicators_to_show.append("MACD")

            st.plotly_chart(
                create_elliott_chart(raw_df, df, historical_patterns, partial_waves, trend, indicators_to_show, show_ema, show_bb),
                use_container_width=True,
                key=f"chart_{symbol}_{degree}_{interval}"
            )

            st.subheader("📈 Technical Indicators")
            latest = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                rsi_value = latest['rsi']
                st.metric("RSI", f"{rsi_value:.2f}")
                st.caption("30-70 range")
            with col2:
                st.metric(f"EMA (20)", f"{latest['ema']:.4f}")
                ema_rel = "Above" if df['close'].iloc[-1] > latest['ema'] else "Below"
                st.caption(f"Price {ema_rel} EMA")
            with col3:
                st.metric("ATR", f"{latest['atr']:.4f}")
                st.caption("Volatility")
            with col4:
                bb_value = latest['bb_percentB'] * 100
                st.metric("BB %B", f"{bb_value:.2f}%")
                st.caption("Bands position")