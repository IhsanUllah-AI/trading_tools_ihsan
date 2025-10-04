import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def calculate_gann_fan_angles(df, pivot_price, pivot_index, trend_direction='bullish', price_scale=None):
    angles = {}
    gann_ratios = {
        '1x1': (1, 1),    # 45°
        '1x2': (1, 2),    # 63.75°
        '2x1': (2, 1),    # 26.25°
        '1x4': (1, 4),    # 75°
        '4x1': (4, 1),    # 15°
        '1x8': (1, 8),    # 82.5°
        '8x1': (8, 1),    # 7.5°
        '3x1': (3, 1),    # 18.75°
        '1x3': (1, 3)     # 71.25°
    }
    
    if price_scale is None:
        if len(df) > 0 and 'atr' in df and pd.notnull(df['atr'].iloc[-1]):
            price_scale = df['atr'].iloc[-1] * 0.1
        else:
            price_scale = df['close'].iloc[-1] * 0.001 if len(df) > 0 else 0.01

    for angle_name, (price_ratio, time_ratio) in gann_ratios.items():
        slope = (price_scale * price_ratio) / time_ratio
        if trend_direction == 'bullish':
            angle_values = []
            for i in range(len(df)):
                if i >= pivot_index:
                    time_diff = i - pivot_index
                    angle_value = pivot_price + (time_diff * slope)
                    angle_values.append(angle_value)
                else:
                    angle_values.append(np.nan)
        else:
            angle_values = []
            for i in range(len(df)):
                if i >= pivot_index:
                    time_diff = i - pivot_index
                    angle_value = pivot_price - (time_diff * slope)
                    angle_values.append(angle_value)
                else:
                    angle_values.append(np.nan)
        angles[angle_name] = angle_values
    
    return angles, price_scale

def gann_square_of_9(price, levels=8):
    if price <= 0:
        return []
    sqrt_price = math.sqrt(price)
    base_levels = []
    for i in range(-levels, levels + 1):
        level = (sqrt_price + i * 0.125) ** 2
        if level > 0:
            base_levels.append(level)
    unique_levels = sorted(set(round(level, 4) for level in base_levels))
    price_range = price * 0.5, price * 1.5
    filtered_levels = [level for level in unique_levels if price_range[0] <= level <= price_range[1]]
    return filtered_levels[:16]

def gann_box(df, interval, lookback=50):
    if len(df) == 0:
        return [], []
    
    effective_lookback = min(lookback, len(df))
    price_min = df['low'].tail(effective_lookback).min()
    price_max = df['high'].tail(effective_lookback).max()
    price_range = price_max - price_min
    ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    price_levels = [price_min + r * price_range for r in ratios]
    
    # Fixed: Use proper pandas frequency-based time arithmetic
    time_current = df['closetime'].iloc[-1]
    
    # Calculate time levels based on interval
    time_levels = []
    
    for r in ratios:
        try:
            # Calculate future time points using proper frequency
            time_point = time_current + pd.Timedelta(hours=r * 24)  # Use hours as base
            time_levels.append(time_point)
        except:
            # Fallback: use simple time addition
            try:
                time_point = time_current + pd.Timedelta(days=r)
                time_levels.append(time_point)
            except:
                time_levels.append(time_current)
    
    return price_levels, time_levels

def gann_square_fixed(high, low, current_price):
    if high <= low:
        return {}
    price_range = high - low
    ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    levels = {}
    for ratio in ratios:
        level = low + ratio * price_range
        levels[f'{int(ratio*100)}%'] = level
    extension_ratios = [1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0]
    for ratio in extension_ratios:
        level = low + ratio * price_range
        levels[f'{int(ratio*100)}%'] = level
    return levels

def compute_gann_results(df, params, symbol, interval, current_price):
    if len(df) == 0:
        return {'info_df': pd.DataFrame(), 'trend': 'Neutral', 'messages': [], 'significant_low': 0, 'significant_high': 0}
    
    gann_subtools = params.get('gann_subtools', ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'])
    pivot_choice = params.get('pivot_choice', 'Auto (based on trend)')
    
    use_pivot_low = False
    trend_direction = 'bullish'
    if pivot_choice == 'Auto (based on trend)':
        if 'ema' in df and 'ema50' in df and len(df) > 0:
            trend_direction = 'bullish' if df['ema'].iloc[-1] > df['ema50'].iloc[-1] else 'bearish'
            use_pivot_low = (trend_direction == 'bullish')
    elif pivot_choice == 'Latest Pivot Low':
        use_pivot_low = True
        trend_direction = 'bullish'
    elif pivot_choice == 'Latest Pivot High':
        use_pivot_low = False
        trend_direction = 'bearish'
    
    # Calculate significant levels for Gann tools
    lookback = min(50, len(df))
    if len(df) > 0:
        significant_low = df['low'].tail(lookback).min()
        significant_high = df['high'].tail(lookback).max()
    else:
        significant_low = 0
        significant_high = 0
    
    # Gann Fan calculation
    if use_pivot_low:
        pivot_lows = df[df['pivot_low'] == True]
        if not pivot_lows.empty and len(pivot_lows) > 0:
            pivot_low_idx = pivot_lows.index[-1]
            if pivot_low_idx < len(df):
                pivot_low_price = df['low'].iloc[pivot_low_idx]
                recent_atr = df['atr'].iloc[pivot_low_idx:].mean() if pivot_low_idx < len(df) else df['atr'].mean()
                price_scale = recent_atr * 0.1 if pd.notnull(recent_atr) and recent_atr > 0 else pivot_low_price * 0.001
                angles, calculated_scale = calculate_gann_fan_angles(df, pivot_low_price, pivot_low_idx, 'bullish', price_scale)
                for angle_name, values in angles.items():
                    df[f'gann_{angle_name}'] = values
                df.loc[pivot_low_idx:, 'pivot_low_price'] = pivot_low_price
                df['gann_price_scale'] = calculated_scale
    else:
        pivot_highs = df[df['pivot_high'] == True]
        if not pivot_highs.empty and len(pivot_highs) > 0:
            pivot_high_idx = pivot_highs.index[-1]
            if pivot_high_idx < len(df):
                pivot_high_price = df['high'].iloc[pivot_high_idx]
                recent_atr = df['atr'].iloc[pivot_high_idx:].mean() if pivot_high_idx < len(df) else df['atr'].mean()
                price_scale = recent_atr * 0.1 if pd.notnull(recent_atr) and recent_atr > 0 else pivot_high_price * 0.001
                angles, calculated_scale = calculate_gann_fan_angles(df, pivot_high_price, pivot_high_idx, 'bearish', price_scale)
                for angle_name, values in angles.items():
                    df[f'gann_{angle_name}'] = values
                df.loc[pivot_high_idx:, 'pivot_high_price'] = pivot_high_price
                df['gann_price_scale'] = calculated_scale
    
    # Gann Square calculations
    df['gann_square_levels_low'] = [gann_square_of_9(significant_low)] * len(df)
    df['gann_square_levels_high'] = [gann_square_of_9(significant_high)] * len(df)
    df['gann_square_levels_current'] = [gann_square_of_9(current_price)] * len(df)
    
    # Gann Box calculations - with proper error handling
    try:
        box_price_levels, box_time_levels = gann_box(df, interval, lookback=50)
        df['gann_box_price_levels'] = [box_price_levels] * len(df)
        df['gann_box_time_levels'] = [box_time_levels] * len(df)
    except Exception as e:
        # If Gann Box fails, provide empty data but don't crash
        df['gann_box_price_levels'] = [[]] * len(df)
        df['gann_box_time_levels'] = [[]] * len(df)
    
    # Gann Square Fixed calculations
    fixed_levels = gann_square_fixed(significant_high, significant_low, current_price) if significant_high > significant_low else {}
    df['gann_square_fixed_levels'] = [fixed_levels] * len(df)
    
    last = df.iloc[-1]
    info_data = []
    messages = []
    
    # Determine trend
    trend = 'Bullish' if 'gann_1x1' in last and pd.notnull(last['gann_1x1']) and current_price > last['gann_1x1'] else \
            'Bearish' if 'gann_1x1' in last and pd.notnull(last['gann_1x1']) and current_price < last['gann_1x1'] else \
            'Bullish' if 'ema' in df and 'ema50' in df and df['ema'].iloc[-1] > df['ema50'].iloc[-1] else 'Bearish'
    
    # Gann Fan information
    if 'Gann Fan' in gann_subtools:
        if 'gann_1x1' in last and pd.notnull(last['gann_1x1']):
            fan_trend = 'Bullish' if current_price > last['gann_1x1'] else 'Bearish' if current_price < last['gann_1x1'] else 'Neutral'
            pivot_type = 'Low' if 'pivot_low_price' in last and pd.notnull(last['pivot_low_price']) else 'High' if 'pivot_high_price' in last and pd.notnull(last['pivot_high_price']) else 'N/A'
            price_scale = last['gann_price_scale'] if 'gann_price_scale' in last and pd.notnull(last['gann_price_scale']) else None
            
            # Properly format values before adding to dictionary
            gann_1x1_display = f"{last['gann_1x1']:.4f}" if pd.notnull(last['gann_1x1']) else 'N/A'
            gann_2x1_display = f"{last['gann_2x1']:.4f}" if 'gann_2x1' in last and pd.notnull(last['gann_2x1']) else 'N/A'
            gann_1x2_display = f"{last['gann_1x2']:.4f}" if 'gann_1x2' in last and pd.notnull(last['gann_1x2']) else 'N/A'
            price_scale_display = f"{price_scale:.6f}" if isinstance(price_scale, (int, float)) and pd.notnull(price_scale) else 'N/A'
            current_vs_1x1 = 'Above' if current_price > last['gann_1x1'] else 'Below' if pd.notnull(last['gann_1x1']) else 'N/A'
            
            info_data.append({
                'Tool': 'Gann Fan',
                'Trend Strength': fan_trend,
                '1x1 Level (45°)': gann_1x1_display,
                '2x1 Level (26.25°)': gann_2x1_display,
                '1x2 Level (63.75°)': gann_1x2_display,
                'Price Scale': price_scale_display,
                'Pivot Used': pivot_type,
                'Current vs 1x1': current_vs_1x1
            })
            messages.append(
                f"**Gann Fan (Trend Strength)**: The trend strength is {fan_trend} because the current price ({current_price:.4f}) is "
                f"{'above' if fan_trend == 'Bullish' else 'below' if fan_trend == 'Bearish' else 'near'} the 1x1 (45°) angle "
                f"({gann_1x1_display})."
            )
        else:
            # Add basic Gann Fan info even if no pivot found
            info_data.append({
                'Tool': 'Gann Fan',
                'Trend Strength': 'N/A',
                '1x1 Level (45°)': 'N/A',
                '2x1 Level (26.25°)': 'N/A',
                '1x2 Level (63.75°)': 'N/A',
                'Price Scale': 'N/A',
                'Pivot Used': 'No pivot found',
                'Current vs 1x1': 'N/A'
            })
            messages.append("**Gann Fan**: No suitable pivot point found for Gann Fan calculation.")
    
    # Gann Square information
    if 'Gann Square' in gann_subtools:
        levels_low = last['gann_square_levels_low'][:5] if 'gann_square_levels_low' in last and isinstance(last['gann_square_levels_low'], list) else []
        levels_high = last['gann_square_levels_high'][:5] if 'gann_square_levels_high' in last and isinstance(last['gann_square_levels_high'], list) else []
        levels_current = last['gann_square_levels_current'][:5] if 'gann_square_levels_current' in last and isinstance(last['gann_square_levels_current'], list) else []
        all_levels = levels_low + levels_high + levels_current
        nearest_level = min(all_levels, key=lambda x: abs(x - current_price)) if all_levels else None
        role = 'Support' if nearest_level and current_price > nearest_level else 'Resistance' if nearest_level else 'N/A'
        
        nearest_level_display = f"{nearest_level:.4f}" if nearest_level is not None else 'N/A'
        distance_pct = f"{abs((current_price - nearest_level) / current_price * 100):.2f}%" if nearest_level else 'N/A'
        low_targets = ', '.join([f"{x:.4f}" for x in levels_low[:3]]) if levels_low else 'N/A'
        high_targets = ', '.join([f"{x:.4f}" for x in levels_high[:3]]) if levels_high else 'N/A'
        
        info_data.append({
            'Tool': 'Gann Square',
            'Nearest Price Target': nearest_level_display,
            'Role': role,
            'Distance %': distance_pct,
            'Key Low Targets': low_targets,
            'Key High Targets': high_targets
        })
        messages.append(
            f"**Gann Square (Future Price Targets)**: The nearest price target is {nearest_level_display} "
            f"(acting as {role.lower()}), {distance_pct} away. "
            f"Derived from Square of 9 using recent low ({significant_low:.4f}) and high ({significant_high:.4f})."
        )
    
    # Gann Box information
    if 'Gann Box' in gann_subtools:
        price_levels = last['gann_box_price_levels'] if 'gann_box_price_levels' in last and isinstance(last['gann_box_price_levels'], list) else []
        nearest_price_level = min(price_levels, key=lambda x: abs(x - current_price)) if price_levels else None
        role = 'Support' if nearest_price_level and current_price > nearest_price_level else 'Resistance' if nearest_price_level else 'N/A'
        time_levels = last['gann_box_time_levels'] if 'gann_box_time_levels' in last and isinstance(last['gann_box_time_levels'], list) else []
        
        # Format time levels properly
        formatted_time_levels = []
        for t in time_levels:
            try:
                if isinstance(t, pd.Timestamp):
                    formatted_time_levels.append(t.strftime('%m/%d %H:%M'))
                else:
                    formatted_time_levels.append(str(t))
            except:
                formatted_time_levels.append('N/A')
        
        nearest_price_display = f"{nearest_price_level:.4f}" if nearest_price_level is not None else 'N/A'
        price_levels_display = ', '.join([f"{x:.4f}" for x in price_levels[:3]]) if price_levels else 'N/A'
        time_targets = ', '.join(formatted_time_levels[:2]) if formatted_time_levels else 'N/A'
        
        info_data.append({
            'Tool': 'Gann Box',
            'Nearest Price Level': nearest_price_display,
            'Role': role,
            'Key Price Levels': price_levels_display,
            'Next Time Targets': time_targets
        })
        messages.append(
            f"**Gann Box (Time & Price Balance)**: The nearest price level is {nearest_price_display} "
            f"(acting as {role.lower()}). Next time targets: {time_targets}."
        )
    
    # Gann Square Fixed information
    if 'Gann Square Fixed' in gann_subtools:
        levels_dict = last['gann_square_fixed_levels'] if 'gann_square_fixed_levels' in last and isinstance(last['gann_square_fixed_levels'], dict) else {}
        levels_list = list(levels_dict.values()) if levels_dict else []
        nearest_level = min(levels_list, key=lambda x: abs(x - current_price)) if levels_list else None
        role = 'Support' if nearest_level and current_price > nearest_level else 'Resistance' if nearest_level else 'N/A'
        
        nearest_ratio = None
        for ratio, level in levels_dict.items():
            if level == nearest_level:
                nearest_ratio = ratio
                break
        
        nearest_level_display = f"{nearest_level:.4f}" if nearest_level is not None else 'N/A'
        key_levels = ', '.join([f"{k}: {v:.4f}" for k, v in list(levels_dict.items())[:3]]) if levels_dict else 'N/A'
        
        info_data.append({
            'Tool': 'Gann Square Fixed',
            'Nearest Level': nearest_level_display,
            'Ratio': nearest_ratio or 'N/A',
            'Role': role,
            'Key Levels': key_levels
        })
        messages.append(
            f"**Gann Square Fixed (Retracement Levels)**: The nearest level is {nearest_level_display} "
            f"({nearest_ratio or 'N/A'}) acting as {role.lower()}. Based on range {significant_low:.4f} - {significant_high:.4f}."
        )
    
    return {
        'info_df': pd.DataFrame(info_data),
        'trend': trend,
        'messages': messages,
        'significant_low': significant_low,
        'significant_high': significant_high
    }

def create_gann_chart(raw_df, df_live, trend, indicators_to_show, show_ema, show_bb, interval, gann_tool):
    if len(raw_df) == 0 or len(df_live) == 0:
        st.warning("No data available for chart")
        return go.Figure()
    
    last = df_live.iloc[-1]
    num_subplots = len(indicators_to_show) + 1
    row_heights = [0.6] + [0.4 / len(indicators_to_show)] * len(indicators_to_show) if indicators_to_show else [1.0]
    subplot_titles = [f'{gann_tool} Analysis - {trend.capitalize()} Trend'] + indicators_to_show
    
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
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350',
            line=dict(width=0.5)
        ),
        row=1, col=1
    )
    
    if show_ema and 'ema' in df_live and 'ema50' in df_live:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['ema'],
                name='EMA (20)',
                line=dict(color='#FFA500', width=1.5)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['ema50'],
                name='EMA (50)',
                line=dict(color='#800080', width=1.5)
            ),
            row=1, col=1
        )
    
    if show_bb and 'bb_upper' in df_live and 'bb_lower' in df_live:
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
    
    if gann_tool == 'Gann Fan' and 'gann_1x1' in df_live and not df_live['gann_1x1'].isna().all():
        angle_styles = [
            ('gann_1x1', '#0000FF', 'dash', '1x1 (45°)'),
            ('gann_2x1', '#00FF00', 'dot', '2x1 (26.25°)'),
            ('gann_1x2', '#FF0000', 'dot', '1x2 (63.75°)'),
            ('gann_4x1', '#00FFFF', 'dashdot', '4x1 (15°)'),
            ('gann_1x4', '#FF00FF', 'dashdot', '1x4 (75°)'),
            ('gann_8x1', '#FFFF00', 'solid', '8x1 (7.5°)'),
            ('gann_3x1', '#FF4500', 'dash', '3x1 (18.75°)'),
            ('gann_1x3', '#4B0082', 'dot', '1x3 (71.25°)'),
            ('gann_1x8', '#808000', 'dashdot', '1x8 (82.5°)')
        ]
        for angle, color, dash, label in angle_styles:
            if angle in df_live and not df_live[angle].isna().all():
                valid_data = df_live[df_live[angle].notna()]
                if not valid_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data['closetime'],
                            y=valid_data[angle],
                            name=label,
                            line=dict(color=color, width=2 if angle == 'gann_1x1' else 1.5, dash=dash),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
    
    if gann_tool == 'Gann Fan':
        if 'pivot_low' in df_live:
            pivot_lows = df_live[df_live['pivot_low'] == True]
            if not pivot_lows.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_lows['closetime'],
                        y=pivot_lows['low'],
                        mode='markers+text',
                        name='Pivot Low',
                        marker=dict(symbol='triangle-up', color='#00FF00', size=12),
                        text=['PL' for _ in range(len(pivot_lows))],
                        textposition='bottom center',
                        textfont=dict(color='#00FF00', size=10)
                    ),
                    row=1, col=1
                )
        if 'pivot_high' in df_live:
            pivot_highs = df_live[df_live['pivot_high'] == True]
            if not pivot_highs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pivot_highs['closetime'],
                        y=pivot_highs['high'],
                        mode='markers+text',
                        name='Pivot High',
                        marker=dict(symbol='triangle-down', color='#FF0000', size=12),
                        text=['PH' for _ in range(len(pivot_highs))],
                        textposition='top center',
                        textfont=dict(color='#FF0000', size=10)
                    ),
                    row=1, col=1
                )
    
    if gann_tool == 'Gann Square':
        levels_low = last['gann_square_levels_low'][:8] if 'gann_square_levels_low' in last and isinstance(last['gann_square_levels_low'], list) else []
        levels_high = last['gann_square_levels_high'][:8] if 'gann_square_levels_high' in last and isinstance(last['gann_square_levels_high'], list) else []
        levels_current = last['gann_square_levels_current'][:8] if 'gann_square_levels_current' in last and isinstance(last['gann_square_levels_current'], list) else []
        for i, level in enumerate(levels_low):
            fig.add_hline(
                y=level, 
                line_dash="dot", 
                line_color="#00FFFF", 
                row=1, col=1, 
                annotation_text=f"SqL{i+1}",
                annotation_position="right",
                opacity=0.7
            )
        for i, level in enumerate(levels_high):
            fig.add_hline(
                y=level, 
                line_dash="dot", 
                line_color="#FF00FF", 
                row=1, col=1, 
                annotation_text=f"SqH{i+1}",
                annotation_position="right",
                opacity=0.7
            )
        for i, level in enumerate(levels_current[:4]):
            fig.add_hline(
                y=level, 
                line_dash="solid", 
                line_color="#FFFFFF", 
                line_width=2,
                row=1, col=1, 
                annotation_text=f"Curr{i+1}",
                annotation_position="left",
                opacity=0.9
            )
    
    if gann_tool == 'Gann Square Fixed':
        levels_dict = last['gann_square_fixed_levels'] if 'gann_square_fixed_levels' in last and isinstance(last['gann_square_fixed_levels'], dict) else {}
        if levels_dict:
            for ratio, level in list(levels_dict.items())[:10]:
                fig.add_hline(
                    y=level, 
                    line_dash="solid" if '100' in ratio else "dot", 
                    line_color="#008000", 
                    row=1, col=1, 
                    annotation_text=f"{ratio}",
                    annotation_position="right",
                    opacity=0.8
                )
    
    if gann_tool == 'Gann Box':
        price_levels = last['gann_box_price_levels'] if 'gann_box_price_levels' in last and isinstance(last['gann_box_price_levels'], list) else []
        time_levels = last['gann_box_time_levels'] if 'gann_box_time_levels' in last and isinstance(last['gann_box_time_levels'], list) else []
        
        # Add price levels as horizontal lines
        for i, price_level in enumerate(price_levels):
            fig.add_hline(
                y=price_level,
                line_dash="dash",
                line_color="#FFA500",
                row=1, col=1,
                annotation_text=f"BoxP{i+1}",
                annotation_position="left",
                opacity=0.7
            )
        
        # Add time levels as vertical lines (only if we have valid timestamps)
        for i, time_level in enumerate(time_levels):
            try:
                if isinstance(time_level, pd.Timestamp):
                    fig.add_vline(
                        x=time_level,
                        line_dash="dash",
                        line_color="#800080",
                        row=1, col=1,
                        annotation_text=f"BoxT{i+1}",
                        annotation_position="top",
                        opacity=0.7
                    )
            except Exception as e:
                # Skip invalid time levels
                continue
    
    current_row = 2
    if "Volume" in indicators_to_show and 'volume' in raw_df:
        colors_volume = ['#EF5350' if row['open'] > row['close'] else '#26A69A' for _, row in raw_df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=raw_df['closetime'],
                y=raw_df['volume'],
                name='Volume',
                marker_color=colors_volume,
                opacity=0.8
            ),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1, gridcolor='rgba(255,255,255,0.05)')
        current_row += 1
    
    if "RSI" in indicators_to_show and 'rsi' in df_live:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['rsi'],
                name='RSI',
                line=dict(color='#800080', width=2)
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", row=current_row, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#26A69A", row=current_row, col=1, annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="#808080", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100], gridcolor='rgba(255,255,255,0.05)')
        current_row += 1
    
    if "MACD" in indicators_to_show and 'macd' in df_live:
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd'],
                name='MACD Line',
                line=dict(color='#0000FF', width=2)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_live['closetime'],
                y=df_live['macd_signal'],
                name='Signal Line',
                line=dict(color='#FFA500', width=2)
            ),
            row=current_row, col=1
        )
        if 'macd_histogram' in df_live:
            histogram_colors = ['#26A69A' if val >= 0 else '#EF5350' for val in df_live['macd_histogram']]
            fig.add_trace(
                go.Bar(
                    x=df_live['closetime'],
                    y=df_live['macd_histogram'],
                    name='MACD Histogram',
                    marker_color=histogram_colors,
                    opacity=0.8
                ),
                row=current_row, col=1
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#808080", row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1, gridcolor='rgba(255,255,255,0.05)')
    
    fig.update_layout(
        height=800,
        title=f"{gann_tool} Analysis - {trend.capitalize()} Trend | Price: {df_live['close'].iloc[-1]:.4f}",
        yaxis_title='Price (USDT)',
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode='pan',
        hovermode='x unified',
        plot_bgcolor='#141414',
        paper_bgcolor='#141414',
        font=dict(color='#FFFFFF', family="Arial", size=12),
        margin=dict(l=60, r=60, t=100, b=60)
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
            ]),
            bgcolor='#2A2A2A',
            font=dict(color='#FFFFFF')
        ),
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False,
        tickformat='%Y-%m-%d %H:%M'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(255,255,255,0.05)',
        zeroline=False,
        tickformat='.4f'
    )
    
    if len(raw_df) > 0:
        initial_start_idx = max(0, len(raw_df) - 200)
        initial_start = raw_df['closetime'].iloc[initial_start_idx]
        initial_end = raw_df['closetime'].iloc[-1]
        for r in range(1, num_subplots + 1):
            fig.update_xaxes(range=[initial_start, initial_end], row=r, col=1, matches='x')
    
    return fig

def display_gann_results(results, df, symbol, interval, params):
    if len(df) == 0:
        st.warning(f"No data available for {symbol} ({interval})")
        return
    
    # Check if we have any Gann data to display
    if results['info_df'].empty:
        st.warning(f"No Gann analysis data available for {symbol} ({interval})")
        return
    
    show_ema = params.get('show_ema', True)
    show_bb = params.get('show_bb', True)
    show_volume = params.get('show_volume', True)
    show_rsi = params.get('show_rsi', True)
    show_macd = params.get('show_macd', True)
    gann_subtools = params.get('gann_subtools', ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'])
    
    indicators_to_show = []
    if show_volume:
        indicators_to_show.append("Volume")
    if show_rsi:
        indicators_to_show.append("RSI")
    if show_macd:
        indicators_to_show.append("MACD")
    
    st.markdown(f"### Gann Analysis for {symbol} ({interval})")
    st.markdown(f"**Trend**: {results['trend']}")
    
    # Get available tools from results
    available_tools = results['info_df']['Tool'].tolist()
    
    if available_tools:
        # Create tabs for each available tool
        tabs = st.tabs(available_tools)
        
        for idx, gann_tool in enumerate(available_tools):
            with tabs[idx]:
                # Display chart for this specific tool
                try:
                    fig = create_gann_chart(df, df, results['trend'], indicators_to_show, show_ema, show_bb, interval, gann_tool)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart for {gann_tool}: {e}")
                    st.info(f"Chart data might not be available for {gann_tool}")
                
                # Display data table for this specific tool only
                tool_data = results['info_df'][results['info_df']['Tool'] == gann_tool]
                
                if not tool_data.empty:
                    st.markdown(f"#### {gann_tool} Analysis Details")
                    
                    # Remove Tool column and display the data
                    display_data = tool_data.drop('Tool', axis=1).iloc[0].to_dict()
                    
                    # Create two columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for key, value in list(display_data.items())[:len(display_data)//2]:
                            if pd.notnull(value) and value != 'N/A':
                                st.write(f"**{key}**: {value}")
                    
                    with col2:
                        for key, value in list(display_data.items())[len(display_data)//2:]:
                            if pd.notnull(value) and value != 'N/A':
                                st.write(f"**{key}**: {value}")
                
                # Display message for this specific tool only
                tool_message = next((msg for msg in results['messages'] if gann_tool in msg), None)
                if tool_message:
                    st.markdown("#### Key Insight")
                    st.markdown(f"- {tool_message}")
    else:
        st.info("No Gann tools data available to display")