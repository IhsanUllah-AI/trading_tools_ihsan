import streamlit as st
import pandas as pd
from datetime import datetime
import pytz  # Add pytz for timezone handling
import requests
import time
import uuid

# Import common functions
from common import fetch_candles_from_binance, fetch_current_price, calculate_indicators

# Import tool modules
from tools import fib, elliott, ichimoku, wyckoff, gann

st.set_page_config(page_title="Master Trading Analysis App", layout="wide", page_icon="📊")

# Custom CSS (unchanged)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #0e1117;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .info-box {
        background-color: #262730;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .signal-buy {
        background-color: #06D6A0;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
    }
    .signal-sell {
        background-color: #EF476F;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
    }
    .signal-neutral {
        background-color: #FFD166;
        color: black;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem;
    }
    .confidence-high {
        color: #06D6A0;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFD166;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF476F;
        font-weight: bold;
    }
    .trade-signal {
        background-color: #1f77b4;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .fib-explanation {
        background-color: #2c2f36;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Master Trading Analysis App</h1>', unsafe_allow_html=True)

# Initialize session state for global trades per tool
tools_list = ["Fibonacci", "Elliott", "Ichimoku", "Wyckoff", "Gann"]
for tool in tools_list:
    if tool + '_active_trades' not in st.session_state:
        st.session_state[tool + '_active_trades'] = {}  # symbol: trade_dict
    if tool + '_trade_history' not in st.session_state:
        st.session_state[tool + '_trade_history'] = []  # list of closed trades

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Sidebar for global settings (unchanged)
st.sidebar.header("⚙️ Global Settings")
popular_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", 
                   "LTCUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT"]
symbols = st.sidebar.multiselect("Select Coins", popular_symbols, default=["BTCUSDT"])

# Custom symbols input
custom_symbols_input = st.sidebar.text_input("Enter Custom Symbols (comma-separated, e.g., FETUSDT,PEPEUSDT)", "").upper()
custom_symbols = [s.strip() for s in custom_symbols_input.split(',') if s.strip()]
for custom_symbol in custom_symbols:
    if custom_symbol and custom_symbol not in symbols:
        test_price = fetch_current_price(custom_symbol)
        if test_price is not None:
            symbols.append(custom_symbol)
            st.sidebar.success(f"Added custom symbol: {custom_symbol}")
        else:
            st.sidebar.error(f"Invalid symbol: {custom_symbol}. Please check the format (e.g., FETUSDT).")

# Interval selection per coin
intervals = ["1m", "3m", "5m", "15m",  "1h", "2h","3h", "4h","1d"]
symbol_intervals = {}
st.sidebar.subheader("Select Intervals for Each Coin")
for symbol in symbols:
    symbol_intervals[symbol] = st.sidebar.selectbox(
        f"Interval for {symbol}", 
        intervals, 
        index=2 if symbol == "BTCUSDT" else 4 if symbol == "ETHUSDT" else 2,
        key=f"interval_{symbol}"
    )

candle_limit = st.sidebar.number_input(
    "Number of candles to fetch", 
    min_value=100, 
    max_value=50000, 
    value=1000, 
    step=100,
    help="Select how many historical candles to analyze"
)

auto_refresh = st.sidebar.checkbox("Auto Refresh every 60 seconds")

# Tool selection
selected_tools = st.sidebar.multiselect("Select Analysis Tools", tools_list, default=["Fibonacci", "Elliott", "Ichimoku", "Wyckoff", "Gann"])

# Tool-specific settings (unchanged)
tool_params = {}
for tool in selected_tools:
    with st.sidebar.expander(f"{tool} Settings"):
        if tool == "Fibonacci":
            window = st.number_input(
                "Window size for Fibonacci calculation", 
                min_value=10, 
                max_value=candle_limit, 
                value=60, 
                step=10,
                help=f"Number of most recent candles to use for Fibonacci calculations (max: {candle_limit})",
                key=f"fib_window"
            )
            fib_threshold = st.number_input(
                "Fibonacci Threshold (%)", 
                min_value=0.1, 
                max_value=5.0, 
                value=0.3, 
                step=0.1,
                help="Percentage distance from Fibonacci level to trigger signal",
                key=f"fib_threshold"
            ) / 100.0
            tool_params[tool] = {"window": window, "fib_threshold": fib_threshold}
        elif tool == "Elliott":
            thresholds = {}
            thresholds['Minor'] = st.number_input(
                "Minor Degree Threshold (%)", 
                min_value=0.001, 
                max_value=10.0, 
                value=0.3, 
                step=0.01, 
                format="%.3f",
                key=f"elliott_minor_threshold"
            ) / 100
            thresholds['Intermediate'] = st.number_input(
                "Intermediate Degree Threshold (%)", 
                min_value=0.001, 
                max_value=10.0, 
                value=1.5, 
                step=0.01, 
                format="%.3f",
                key=f"elliott_intermediate_threshold"
            ) / 100
            thresholds['Major'] = st.number_input(
                "Major Degree Threshold (%)", 
                min_value=0.001, 
                max_value=10.0, 
                value=3.0, 
                step=0.01, 
                format="%.3f",
                key=f"elliott_major_threshold"
            ) / 100
            selected_degrees = st.multiselect(
                "Select Wave Degrees", 
                list(thresholds.keys()), 
                default=['Minor','Intermediate','Major'],
                key=f"elliott_selected_degrees"
            )
            use_smoothing = st.checkbox("Smooth Prices with EMA", value=False, key=f"elliott_use_smoothing")
            smooth_period = 3
            if use_smoothing:
                smooth_period = st.number_input("Smooth EMA Period", min_value=2, max_value=5, value=3, key=f"elliott_smooth_period")
            show_ema = st.checkbox("Show EMA", value=True, key=f"elliott_show_ema")
            show_bb = st.checkbox("Show Bollinger Bands", value=True, key=f"elliott_show_bb")
            show_volume = st.checkbox("Show Volume", value=True, key=f"elliott_show_volume")
            show_rsi = st.checkbox("Show RSI", value=True, key=f"elliott_show_rsi")
            show_macd = st.checkbox("Show MACD", value=True, key=f"elliott_show_macd")
            tool_params[tool] = {
                "thresholds": thresholds,
                "selected_degrees": selected_degrees,
                "use_smoothing": use_smoothing,
                "smooth_period": smooth_period,
                "show_ema": show_ema,
                "show_bb": show_bb,
                "show_volume": show_volume,
                "show_rsi": show_rsi,
                "show_macd": show_macd
            }
        elif tool == "Ichimoku":
            show_ema = st.checkbox("Show EMA", value=True, key=f"ichimoku_show_ema")
            show_bb = st.checkbox("Show Bollinger Bands", value=True, key=f"ichimoku_show_bb")
            show_volume = st.checkbox("Show Volume", value=True, key=f"ichimoku_show_volume")
            show_rsi = st.checkbox("Show RSI", value=True, key=f"ichimoku_show_rsi")
            show_macd = st.checkbox("Show MACD", value=True, key=f"ichimoku_show_macd")
            tool_params[tool] = {
                "tenkan_period": 9,
                "kijun_period": 26,
                "senkou_b_period": 52,
                "show_ema": show_ema,
                "show_bb": show_bb,
                "show_volume": show_volume,
                "show_rsi": show_rsi,
                "show_macd": show_macd
            }
        elif tool == "Wyckoff":
            show_ema = st.checkbox("Show EMA (20, 50)", value=True, key=f"wyckoff_show_ema")
            show_bb = st.checkbox("Show Bollinger Bands", value=True, key=f"wyckoff_show_bb")
            show_volume = st.checkbox("Show Volume", value=True, key=f"wyckoff_show_volume")
            show_rsi = st.checkbox("Show RSI", value=True, key=f"wyckoff_show_rsi")
            show_macd = st.checkbox("Show MACD", value=True, key=f"wyckoff_show_macd")
            tool_params[tool] = {
                "show_ema": show_ema,
                "show_bb": show_bb,
                "show_volume": show_volume,
                "show_rsi": show_rsi,
                "show_macd": show_macd
            }
        elif tool == "Gann":
            show_ema = st.checkbox("Show EMA (20, 50)", value=True, key=f"gann_show_ema")
            show_bb = st.checkbox("Show Bollinger Bands", value=True, key=f"gann_show_bb")
            show_volume = st.checkbox("Show Volume", value=True, key=f"gann_show_volume")
            show_rsi = st.checkbox("Show RSI", value=True, key=f"gann_show_rsi")
            show_macd = st.checkbox("Show MACD", value=True, key=f"gann_show_macd")
            gann_subtools = st.multiselect(
                "Select Gann Tools to Display",
                ["Gann Fan", "Gann Square", "Gann Box", "Gann Square Fixed"],
                default=["Gann Fan", "Gann Square", "Gann Box", "Gann Square Fixed"],
                key=f"gann_subtools"
            )
            pivot_choice = st.selectbox(
                "Gann Fan Pivot Selection", 
                ['Auto (based on trend)', 'Latest Pivot Low', 'Latest Pivot High'],
                key=f"gann_pivot_choice"
            )
            tool_params[tool] = {
                "show_ema": show_ema,
                "show_bb": show_bb,
                "show_volume": show_volume,
                "show_rsi": show_rsi,
                "show_macd": show_macd,
                "gann_subtools": gann_subtools,
                "pivot_choice": pivot_choice
            }

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ How to use"):
    st.info("""
    **How to use:**
    1. Select multiple coins and custom intervals for each.
    2. Add a custom symbol if needed (e.g., LUNABUSD).
    3. Select one or more analysis tools (e.g., Fibonacci, Elliott, Ichimoku, Gann).
    4. Configure tool-specific settings in the expanders.
    5. Click 'Run Analysis' to start.
    6. Results display per coin and per tool in expanders.
    7. Trades are simulated and tracked independently per tool.
    8. For Gann, select sub-tools (e.g., Gann Fan, Gann Square) to display in horizontal tabs.
    """)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    st.session_state.run_analysis = True

# Define PKT timezone
pkt_tz = pytz.timezone('Asia/Karachi')

# Main logic: Run selected tools
if st.session_state.run_analysis:
    try:
        for symbol in symbols:
            interval = symbol_intervals.get(symbol, "5m")
            with st.expander(f"Analysis for {symbol} ({interval})", expanded=True):
                with st.spinner(f'Fetching data for {symbol} ({interval})...'):
                    df_live = fetch_candles_from_binance(symbol, interval, candle_limit)
                    if df_live is None:
                        st.error(f"Skipping analysis for {symbol} due to data fetch failure.")
                        continue

                current_price = fetch_current_price(symbol) or df_live['close'].iloc[-1]

                tabs = st.tabs(selected_tools)
                for idx, tool in enumerate(selected_tools):
                    with tabs[idx]:
                        params = tool_params.get(tool, {})
                        if tool == "Fibonacci":
                            results = fib.compute_fib_results(df_live, params, symbol, interval)

                            # Fibonacci Trade Management
                            trade_action = results['trade_action']
                            entry_price = results['entry_price']
                            stop_loss = results['stop_loss']
                            take_profit = results['take_profit']
                            fib_signal_triggered = results['fib_signal_triggered']
                            confidence = results['confidence']
                            signals = results['signals']
                            signal_descriptions = results['signal_descriptions']

                            active_key = tool + '_active_trades'
                            history_key = tool + '_trade_history'

                            # Update existing trades
                            if symbol in st.session_state[active_key]:
                                trade = st.session_state[active_key][symbol]
                                hit_sl = (trade['action'] == "BUY" and current_price <= trade['stop_loss']) or (trade['action'] == "SELL" and current_price >= trade['stop_loss'])
                                hit_tp = (trade['action'] == "BUY" and current_price >= trade['take_profit']) or (trade['action'] == "SELL" and current_price <= trade['take_profit'])

                                if hit_sl or hit_tp:
                                    outcome = 'win' if hit_tp else 'loss'
                                    close_price = trade['take_profit'] if hit_tp else trade['stop_loss']
                                    profit_pct = ((close_price - trade['entry_price']) / trade['entry_price'] * 100) if trade['action'] == "BUY" else ((trade['entry_price'] - close_price) / trade['entry_price'] * 100)
                                    
                                    closed_trade = trade.copy()
                                    closed_trade.update({
                                        'outcome': outcome,
                                        'close_price': close_price,
                                        'profit_pct': profit_pct,
                                        'close_time': datetime.now(pkt_tz)  # Changed to PKT
                                    })
                                    st.session_state[history_key].append(closed_trade)
                                    del st.session_state[active_key][symbol]
                                    st.success(f"{tool} Trade for {symbol} ({interval}) closed: {outcome.upper()} with {profit_pct:.2f}% return")

                            # Open new trade
                            if symbol not in st.session_state[active_key] and trade_action is not None:
                                active_trade = {
                                    'symbol': symbol,
                                    'action': trade_action,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'entry_time': datetime.now(pkt_tz),  # Changed to PKT
                                    'signals': signals,
                                    'signal_descriptions': signal_descriptions,
                                    'confidence': confidence,
                                    'interval': interval,
                                    'degree': 'N/A'
                                }
                                st.session_state[active_key][symbol] = active_trade
                                st.info(f"{tool} New {trade_action} trade started for {symbol} ({interval}) at {entry_price:.4f}")

                            fib.display_fib_results(results)

                        elif tool == "Elliott":
                            results = elliott.compute_elliott_results(df_live, params, symbol, interval)

                            # Elliott Trade Management
                            signals = results['signals']
                            active_key = tool + '_active_trades'
                            history_key = tool + '_trade_history'

                            # Update existing trades
                            for symbol_key in list(st.session_state[active_key]):
                                trade = st.session_state[active_key][symbol_key]
                                if trade['symbol'] == symbol:
                                    hit_sl = (trade['action'] == "Buy" and current_price <= trade['stop_loss']) or (trade['action'] == "Sell" and current_price >= trade['stop_loss'])
                                    hit_tp = (trade['action'] == "Buy" and current_price >= trade['take_profit']) or (trade['action'] == "Sell" and current_price <= trade['take_profit'])

                                    if hit_sl or hit_tp:
                                        outcome = 'win' if hit_tp else 'loss'
                                        close_price = trade['take_profit'] if hit_tp else trade['stop_loss']
                                        profit_pct = ((close_price - trade['entry_price']) / trade['entry_price'] * 100) if trade['action'] == "Buy" else ((trade['entry_price'] - close_price) / trade['entry_price'] * 100)

                                        closed_trade = trade.copy()
                                        closed_trade.update({
                                            'outcome': outcome,
                                            'close_price': close_price,
                                            'profit_pct': profit_pct,
                                            'close_time': datetime.now(pkt_tz)  # Changed to PKT
                                        })
                                        st.session_state[history_key].append(closed_trade)
                                        del st.session_state[active_key][symbol_key]
                                        st.success(f"{tool} Trade for {symbol} ({interval}) closed: {outcome.upper()} with {profit_pct:.2f}% return")

                            # Open new trades
                            for sig in signals:
                                symbol_key = f"{symbol}_{sig.get('degree', 'Minor')}"
                                if symbol_key not in st.session_state[active_key]:
                                    trade = {
                                        'symbol': symbol,
                                        'action': sig['action'],
                                        'entry_price': sig['entry_price'],
                                        'stop_loss': sig['sl'],
                                        'take_profit': sig['tp'],
                                        'entry_time': datetime.now(pkt_tz),  # Changed to PKT
                                        'signals': [sig['reason']],
                                        'signal_descriptions': [sig['reason']],
                                        'confidence': results['wave_data_by_degree'][sig.get('degree', 'Minor')]['confidence'],
                                        'interval': interval,
                                        'degree': sig.get('degree', 'Minor')
                                    }
                                    st.session_state[active_key][symbol_key] = trade
                                    st.info(f"{tool} Opened {sig['action']} Trade for {symbol} ({interval}): {sig['reason']} at ~{sig['entry_price']:.2f}")

                            elliott.display_elliott_results(results)

                        elif tool == "Ichimoku":
                            results = ichimoku.compute_ichimoku_results(df_live, params, symbol, interval)

                            # Ichimoku Trade Management
                            signals = results['signals']
                            active_key = tool + '_active_trades'
                            history_key = tool + '_trade_history'

                            # Update existing trades
                            for symbol_key in list(st.session_state[active_key]):
                                trade = st.session_state[active_key][symbol_key]
                                if trade['symbol'] == symbol:
                                    hit_sl = (trade['action'] == "Buy" and current_price <= trade['stop_loss']) or (trade['action'] == "Sell" and current_price >= trade['stop_loss'])
                                    hit_tp = (trade['action'] == "Buy" and current_price >= trade['take_profit']) or (trade['action'] == "Sell" and current_price <= trade['take_profit'])

                                    if hit_sl or hit_tp:
                                        outcome = 'win' if hit_tp else 'loss'
                                        close_price = trade['take_profit'] if hit_tp else trade['stop_loss']
                                        profit_pct = ((close_price - trade['entry_price']) / trade['entry_price'] * 100) if trade['action'] == "Buy" else ((trade['entry_price'] - close_price) / trade['entry_price'] * 100)

                                        closed_trade = trade.copy()
                                        closed_trade.update({
                                            'outcome': outcome,
                                            'close_price': close_price,
                                            'profit_pct': profit_pct,
                                            'close_time': datetime.now(pkt_tz)  # Changed to PKT
                                        })
                                        st.session_state[history_key].append(closed_trade)
                                        del st.session_state[active_key][symbol_key]
                                        st.success(f"{tool} Trade for {symbol} ({interval}) closed: {outcome.upper()} with {profit_pct:.2f}% return")

                            # Open new trades
                            for sig in signals:
                                symbol_key = f"{symbol}_N/A"
                                if symbol_key not in st.session_state[active_key]:
                                    trade = {
                                        'symbol': symbol,
                                        'action': sig['type'],
                                        'entry_price': sig['entry_price'],
                                        'stop_loss': sig['sl'],
                                        'take_profit': sig['tp'],
                                        'entry_time': datetime.now(pkt_tz),  # Changed to PKT
                                        'signals': [sig['reason']],
                                        'signal_descriptions': [sig['reason']],
                                        'confidence': results['confidence'],
                                        'interval': interval,
                                        'degree': 'N/A'
                                    }
                                    st.session_state[active_key][symbol_key] = trade
                                    st.info(f"{tool} Opened {sig['type']} Trade for {symbol} ({interval}): {sig['reason']} at ~{sig['entry_price']:.2f}")

                            ichimoku.display_ichimoku_results(results)

                        elif tool == "Wyckoff":
                            results = wyckoff.compute_wyckoff_results(df_live, params, symbol, interval)

                            # Wyckoff Trade Management
                            signals = results['signals']
                            active_key = tool + '_active_trades'
                            history_key = tool + '_trade_history'

                            # Update existing trades
                            if symbol in st.session_state[active_key]:
                                trade = st.session_state[active_key][symbol]
                                hit_sl = (trade['action'] == "BUY" and current_price <= trade['stop_loss']) or (trade['action'] == "SELL" and current_price >= trade['stop_loss'])
                                hit_tp = (trade['action'] == "BUY" and current_price >= trade['take_profit']) or (trade['action'] == "SELL" and current_price <= trade['take_profit'])

                                if hit_sl or hit_tp:
                                    outcome = 'win' if hit_tp else 'loss'
                                    close_price = trade['take_profit'] if hit_tp else trade['stop_loss']
                                    profit_pct = ((close_price - trade['entry_price']) / trade['entry_price'] * 100) if trade['action'] == "BUY" else ((trade['entry_price'] - close_price) / trade['entry_price'] * 100)
                                    
                                    closed_trade = trade.copy()
                                    closed_trade.update({
                                        'outcome': outcome,
                                        'close_price': close_price,
                                        'profit_pct': profit_pct,
                                        'close_time': datetime.now(pkt_tz)  # Changed to PKT
                                    })
                                    st.session_state[history_key].append(closed_trade)
                                    del st.session_state[active_key][symbol]
                                    st.success(f"{tool} Trade for {symbol} ({interval}) closed: {outcome.upper()} with {profit_pct:.2f}% return")

                            # Open new trade
                            for sig in signals:
                                if symbol not in st.session_state[active_key]:
                                    active_trade = {
                                        'symbol': symbol,
                                        'action': sig['type'],
                                        'entry_price': sig['entry_price'],
                                        'stop_loss': sig['sl'],
                                        'take_profit': sig['tp'],
                                        'entry_time': datetime.now(pkt_tz),  # Changed to PKT
                                        'signals': [sig['reason']],
                                        'signal_descriptions': [sig['reason']],
                                        'confidence': results['confidence'],
                                        'interval': interval,
                                        'degree': 'N/A'
                                    }
                                    st.session_state[active_key][symbol] = active_trade
                                    st.info(f"{tool} New {sig['type']} trade started for {symbol} ({interval}) at {sig['entry_price']:.4f}")

                            wyckoff.display_wyckoff_results(results, df_live, symbol, interval, params)

                        elif tool == "Gann":
                            results = gann.compute_gann_results(df_live, params, symbol, interval, current_price)
                            gann.display_gann_results(results, df_live, symbol, interval, params)

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.info("Check symbol format (e.g., BTCUSDT) or try again.")

# Display Trades per Tool
for tool in selected_tools:
    if tool != "Gann":  # Gann has no trading system
        st.subheader(f"{tool} Active Trades")
        active_key = tool + '_active_trades'
        if st.session_state[active_key]:
            for symbol_key, trade in st.session_state[active_key].items():
                with st.expander(f"Active {trade['action']} Trade for {trade['symbol']} ({trade['interval']}) - Degree: {trade['degree']}"):
                    st.write(f"**Entry Price:** {trade['entry_price']:.4f}")
                    st.write(f"**Stop Loss:** {trade['stop_loss']:.4f}")
                    st.write(f"**Take Profit:** {trade['take_profit']:.4f}")
                    st.write(f"**Entry Time (PKT):** {trade['entry_time']}")  # Updated label to PKT
                    st.write(f"**Confidence:** {trade['confidence']}")
                    st.write("**Signals:**")
                    for desc in trade['signal_descriptions']:
                        st.markdown(f"- {desc}")
        else:
            st.info(f"No active trades for {tool}.")

        st.subheader(f"{tool} Trade History")
        history_key = tool + '_trade_history'
        if st.session_state[history_key]:
            history_df = pd.DataFrame(st.session_state[history_key])
            history_df = history_df[['symbol', 'action', 'entry_price', 'close_price', 'profit_pct', 'outcome', 'entry_time', 'close_time', 'interval', 'degree']]
            st.dataframe(history_df.style.format({
                'entry_price': '{:.4f}',
                'close_price': '{:.4f}',
                'profit_pct': '{:.2f}%'
            }), use_container_width=True)
            
            wins = sum(1 for t in st.session_state[history_key] if t['outcome'] == 'win')
            losses = len(st.session_state[history_key]) - wins
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wins", wins)
            with col2:
                st.metric("Losses", losses)

            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {tool} Trade History as CSV",
                data=csv,
                file_name=f'{tool}_trade_history.csv',
                mime='text/csv'
            )
        else:
            st.info(f"No trade history for {tool} yet.")

# Refresh Button
if st.button("🔄 Refresh Data", type="primary"):
    st.rerun()

# Auto Refresh
if auto_refresh:
    time.sleep(60)
    st.rerun()

st.markdown("---")
st.caption("ℹ️ Data provided by Binance API • Disclaimer: This tool is for educational purposes only. Trades are simulated.")
