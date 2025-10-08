import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import requests
import time
import uuid
import os

# Import common functions
from common import fetch_candles_from_binance, fetch_current_price, calculate_indicators

# Import tool modules
from tools import fib, elliott, ichimoku, wyckoff, gann

st.set_page_config(page_title="Master Trading Analysis App", layout="wide", page_icon="📊")

# Custom CSS
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
    .combined-signal-strong {
        background-color: #06D6A0;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .combined-signal-weak {
        background-color: #FFD166;
        color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .combined-signal-neutral {
        background-color: #6C757D;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .tool-agreement-high {
        background-color: #06D6A0;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
    }
    .tool-agreement-medium {
        background-color: #FFD166;
        color: black;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
    }
    .tool-agreement-low {
        background-color: #EF476F;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
    }
    .signal-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .signal-table th, .signal-table td {
        border: 1px solid #444;
        padding: 0.5rem;
        text-align: left;
    }
    .signal-table th {
        background-color: #1f77b4;
        color: white;
    }
    .signal-table tr:nth-child(even) {
        background-color: #2a2a2a;
    }
    .rr-ratio-good {
        color: #06D6A0;
        font-weight: bold;
    }
    .rr-ratio-poor {
        color: #EF476F;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Master Trading Analysis App</h1>', unsafe_allow_html=True)

# Initialize session state for trades
def initialize_session_state():
    # Gann is removed from tools list since it doesn't have trade management
    tools_list = ["Fibonacci", "Elliott", "Ichimoku", "Wyckoff", "Combined"]
    
    for tool in tools_list:
        if tool + '_active_trades' not in st.session_state:
            st.session_state[tool + '_active_trades'] = {}
        if tool + '_trade_history' not in st.session_state:
            st.session_state[tool + '_trade_history'] = []
    
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    if 'trade_counter' not in st.session_state:
        st.session_state.trade_counter = 0
    if 'combined_signals' not in st.session_state:
        st.session_state.combined_signals = {}

initialize_session_state()

# Sidebar for global settings
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
intervals = ["1m", "3m","5m", "15m","1h","2h","3h", "4h","1d"]
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

# Tool selection - Gann is included for analysis but not for trading
tools_list = ["Fibonacci", "Elliott", "Ichimoku", "Wyckoff", "Gann"]
selected_tools = st.sidebar.multiselect("Select Analysis Tools", tools_list, default=tools_list)

# Combined trading settings
st.sidebar.subheader("🔗 Combined Trading Settings")
combined_confidence_threshold = st.sidebar.slider(
    "Minimum Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.6,
    help="Minimum confidence level for combined signals"
)

min_tool_agreement = st.sidebar.slider(
    "Minimum Tool Agreement", 
    min_value=1, 
    max_value=len([t for t in tools_list if t != "Gann"]),  # Exclude Gann from agreement count
    value=2,
    help="Minimum number of tools that must agree on a signal (excluding Gann)"
)

# Risk-Reward Settings
st.sidebar.subheader("⚖️ Risk-Reward Settings")
risk_reward_ratio = st.sidebar.selectbox(
    "Risk-Reward Ratio",
    ["1:1", "1:1.5", "1:2", "1:2.5", "1:3"],
    index=2,
    help="One win covers multiple losses (1:2 means one win covers two losses)"
)

# Parse risk-reward ratio
rr_ratio_map = {"1:1": 1.0, "1:1.5": 1.5, "1:2": 2.0, "1:2.5": 2.5, "1:3": 3.0}
selected_rr_ratio = rr_ratio_map[risk_reward_ratio]

# Tool weight settings - Only for tools that generate direct signals
st.sidebar.subheader("⚖️ Tool Weights")
tool_weights = {}
for tool in selected_tools:
    if tool != "Gann":  # Gann doesn't get weight since it's indirect
        tool_weights[tool] = st.sidebar.slider(
            f"{tool} Weight", 
            min_value=0.1, 
            max_value=2.0, 
            value=1.0,
            help=f"Relative weight for {tool} in combined analysis"
        )

# Tool-specific settings
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
    3. Select one or more analysis tools.
    4. **Gann provides indirect information only** - no direct trades.
    5. Configure tool-specific settings in the expanders.
    6. Set tool weights for combined analysis (Gann excluded).
    7. Configure risk-reward ratio (1:2 means one win covers two losses).
    8. Click 'Run Analysis' to start.
    9. Results display per coin and per tool in expanders.
    10. Trades are simulated for Fibonacci, Elliott, Ichimoku, Wyckoff only.
    11. Combined trading system aggregates signals from direct tools + Gann insights.
    12. Trade history is saved and can be exported as CSV.
    """)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    st.session_state.run_analysis = True

# PKT timezone
pkt_tz = timezone(timedelta(hours=5))

# Enhanced combined trading logic
def convert_confidence_to_numeric(confidence):
    """Convert string confidence levels to numeric values"""
    if isinstance(confidence, (int, float)):
        return float(confidence)
    elif isinstance(confidence, str):
        confidence_lower = confidence.lower()
        if confidence_lower == 'high':
            return 0.8
        elif confidence_lower == 'medium':
            return 0.6
        elif confidence_lower == 'low':
            return 0.4
        else:
            return 0.3
    else:
        return 0.3

def convert_numeric_to_confidence(numeric_confidence):
    """Convert numeric confidence to string levels"""
    if numeric_confidence >= 0.7:
        return 'High'
    elif numeric_confidence >= 0.5:
        return 'Medium'
    else:
        return 'Low'

def analyze_gann_information(gann_results, current_price):
    """Analyze Gann's indirect information to derive trading bias"""
    if not gann_results or gann_results['info_df'].empty:
        return {'bias': 'neutral', 'confidence': 'Low', 'reasons': [], 'detailed_analysis': {}}
    
    bias_score = 0
    reasons = []
    detailed_analysis = {}
    
    # Analyze all Gann tools
    for tool_name in ["Gann Fan", "Gann Square", "Gann Box", "Gann Square Fixed"]:
        tool_data = gann_results['info_df'][gann_results['info_df']['Tool'] == tool_name]
        if not tool_data.empty:
            tool_row = tool_data.iloc[0]
            tool_analysis = {}
            
            if tool_name == "Gann Fan":
                trend_strength = tool_row.get('Trend Strength', 'N/A')
                current_vs_1x1 = tool_row.get('Current vs 1x1', 'N/A')
                
                if trend_strength == 'Bullish':
                    bias_score += 1
                    reasons.append("Gann Fan: Bullish trend")
                    tool_analysis['bias'] = 'bullish'
                elif trend_strength == 'Bearish':
                    bias_score -= 1
                    reasons.append("Gann Fan: Bearish trend")
                    tool_analysis['bias'] = 'bearish'
                    
                if current_vs_1x1 == 'Above':
                    bias_score += 0.5
                    tool_analysis['position'] = 'above_1x1'
                elif current_vs_1x1 == 'Below':
                    bias_score -= 0.5
                    tool_analysis['position'] = 'below_1x1'
                    
            elif tool_name == "Gann Square":
                role = tool_row.get('Role', 'N/A')
                nearest_target = tool_row.get('Nearest Price Target', 'N/A')
                
                if role == 'Support':
                    bias_score += 0.5
                    reasons.append(f"Gann Square: Support at {nearest_target}")
                    tool_analysis['role'] = 'support'
                elif role == 'Resistance':
                    bias_score -= 0.5
                    reasons.append(f"Gann Square: Resistance at {nearest_target}")
                    tool_analysis['role'] = 'resistance'
                    
            elif tool_name == "Gann Box":
                role = tool_row.get('Role', 'N/A')
                nearest_level = tool_row.get('Nearest Price Level', 'N/A')
                
                if role == 'Support':
                    bias_score += 0.5
                    reasons.append(f"Gann Box: Support at {nearest_level}")
                    tool_analysis['role'] = 'support'
                elif role == 'Resistance':
                    bias_score -= 0.5
                    reasons.append(f"Gann Box: Resistance at {nearest_level}")
                    tool_analysis['role'] = 'resistance'
                    
            elif tool_name == "Gann Square Fixed":
                role = tool_row.get('Role', 'N/A')
                nearest_level = tool_row.get('Nearest Level', 'N/A')
                
                if role == 'Support':
                    bias_score += 0.5
                    reasons.append(f"Gann Square Fixed: Support at {nearest_level}")
                    tool_analysis['role'] = 'support'
                elif role == 'Resistance':
                    bias_score -= 0.5
                    reasons.append(f"Gann Square Fixed: Resistance at {nearest_level}")
                    tool_analysis['role'] = 'resistance'
            
            detailed_analysis[tool_name] = tool_analysis
    
    # Determine overall bias and confidence
    if bias_score > 1.5:
        bias = 'bullish'
        confidence_level = 'High'
    elif bias_score > 0.5:
        bias = 'bullish'
        confidence_level = 'Medium'
    elif bias_score < -1.5:
        bias = 'bearish'
        confidence_level = 'High'
    elif bias_score < -0.5:
        bias = 'bearish'
        confidence_level = 'Medium'
    else:
        bias = 'neutral'
        confidence_level = 'Low'
    
    return {
        'bias': bias, 
        'confidence': confidence_level, 
        'reasons': reasons, 
        'detailed_analysis': detailed_analysis,
        'bias_score': bias_score
    }

def calculate_tp_sl_2_1_rr(entry_price, action, atr, current_price):
    """Calculate TP and SL with 2:1 risk-reward ratio"""
    if action == "BUY":
        # For BUY: SL below entry, TP above entry with 2:1 ratio
        sl_distance = atr * 1.5  # Stop loss distance
        sl_price = entry_price - sl_distance
        tp_distance = sl_distance * selected_rr_ratio  # Take profit distance (2x risk)
        tp_price = entry_price + tp_distance
    else:  # SELL
        # For SELL: SL above entry, TP below entry with 2:1 ratio
        sl_distance = atr * 1.5  # Stop loss distance
        sl_price = entry_price + sl_distance
        tp_distance = sl_distance * selected_rr_ratio  # Take profit distance (2x risk)
        tp_price = entry_price - tp_distance
    
    return sl_price, tp_price

def generate_combined_signal(tool_signals, current_price, symbol, interval, tool_weights):
    """Generate combined trading signal based on all tool signals with weighted scoring"""
    
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    # Collect signals from all tools
    for tool, signal_data in tool_signals.items():
        if tool == "Gann":
            # Special handling for Gann's indirect information
            gann_analysis = analyze_gann_information(signal_data.get('gann_results'), current_price)
            action = 'BUY' if gann_analysis['bias'] == 'bullish' else 'SELL' if gann_analysis['bias'] == 'bearish' else 'HOLD'
            confidence_numeric = convert_confidence_to_numeric(gann_analysis['confidence'])
            reason = " | ".join(gann_analysis['reasons']) if gann_analysis['reasons'] else "No clear bias - Neutral position"
            signal_info = {
                'tool': tool,
                'action': action,
                'confidence': confidence_numeric,
                'reason': reason,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'weight': 0.5,  # Fixed weight for Gann since it's indirect
                'is_direct': False,
                'confidence_original': gann_analysis['confidence'],
                'detailed_analysis': gann_analysis['detailed_analysis']
            }
            if action == 'BUY':
                buy_signals.append(signal_info)
            elif action == 'SELL':
                sell_signals.append(signal_info)
            else:
                hold_signals.append(signal_info)
            continue
            
        # For other tools
        trade_action = signal_data.get('trade_action')
        if trade_action:
            action = trade_action.upper()
            confidence_raw = signal_data.get('confidence', 'Low')
            confidence_numeric = convert_confidence_to_numeric(confidence_raw)
            reason = signal_data.get('signal_descriptions', ['No reason provided'])[0]
        else:
            # If no signal, treat as HOLD with low confidence
            action = 'HOLD'
            confidence_numeric = 0.3
            confidence_raw = 'Low'
            reason = 'No clear signal from tool'
        
        signal_info = {
            'tool': tool,
            'action': action,
            'confidence': confidence_numeric,
            'reason': reason,
            'entry_price': signal_data.get('entry_price', current_price) if action != 'HOLD' else None,
            'stop_loss': signal_data.get('stop_loss') if action != 'HOLD' else None,
            'take_profit': signal_data.get('take_profit') if action != 'HOLD' else None,
            'weight': tool_weights.get(tool, 1.0),
            'is_direct': True,
            'confidence_original': confidence_raw if isinstance(confidence_raw, str) else convert_numeric_to_confidence(confidence_numeric)
        }
        
        if action == "BUY":
            buy_signals.append(signal_info)
        elif action == "SELL":
            sell_signals.append(signal_info)
        else:
            hold_signals.append(signal_info)
    
    # Calculate weighted combined signal strength
    buy_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in buy_signals)
    sell_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in sell_signals)
    hold_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in hold_signals)
    
    # Only count direct tools for agreement (exclude Gann)
    direct_tools = [sig for sig in buy_signals + sell_signals + hold_signals if sig['is_direct']]
    total_direct_tools = len(set(sig['tool'] for sig in direct_tools))  # Number of tools that provided a signal
    buy_count = len([sig for sig in buy_signals if sig['is_direct']])
    sell_count = len([sig for sig in sell_signals if sig['is_direct']])
    hold_count = len([sig for sig in hold_signals if sig['is_direct']])
    
    # Calculate agreement percentage based on direct tools only
    total_signals = buy_count + sell_count + hold_count
    agreement_percentage = max(buy_count, sell_count, hold_count) / total_signals if total_signals > 0 else 0
    
    # Determine combined action with weighted scoring
    weighted_buy_score = buy_strength * agreement_percentage if buy_count > 0 else 0
    weighted_sell_score = sell_strength * agreement_percentage if sell_count > 0 else 0
    weighted_hold_score = hold_strength * agreement_percentage if hold_count > 0 else 0
    
    max_score = max(weighted_buy_score, weighted_sell_score, weighted_hold_score)
    
    reasons = []
    
    if max_score == weighted_buy_score and buy_count >= min_tool_agreement and weighted_buy_score >= combined_confidence_threshold:
        combined_action = "BUY"
        combined_confidence_numeric = weighted_buy_score
        reasons = [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in buy_signals]
        agreement_level = 'high' if agreement_percentage >= 0.75 else 'medium' if agreement_percentage >= 0.5 else 'low'
        
    elif max_score == weighted_sell_score and sell_count >= min_tool_agreement and weighted_sell_score >= combined_confidence_threshold:
        combined_action = "SELL"
        combined_confidence_numeric = weighted_sell_score
        reasons = [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in sell_signals]
        agreement_level = 'high' if agreement_percentage >= 0.75 else 'medium' if agreement_percentage >= 0.5 else 'low'
        
    else:
        # Default to HOLD if no clear buy/sell or if hold is strongest
        combined_action = "HOLD"
        combined_confidence_numeric = weighted_hold_score if weighted_hold_score > 0 else 0.5
        if total_signals == 0:
            reasons.append("No signals from any tools")
        if max(buy_count, sell_count) < min_tool_agreement:
            reasons.append(f"Insufficient tool agreement: {max(buy_count, sell_count)}/{min_tool_agreement} tools agree on BUY/SELL")
        if max(weighted_buy_score, weighted_sell_score) < combined_confidence_threshold:
            reasons.append(f"Confidence score {max(weighted_buy_score, weighted_sell_score):.2f} below threshold {combined_confidence_threshold}")
        if not reasons:
            reasons = ["Insufficient tool agreement or confidence"]
        reasons += [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in hold_signals]
        agreement_level = 'medium' if combined_confidence_numeric > 0.5 else 'low'
    
    # Convert numeric confidence back to string for display
    combined_confidence = convert_numeric_to_confidence(combined_confidence_numeric)
    
    # Calculate position sizing based on confidence and agreement
    if combined_action in ["BUY", "SELL"] and combined_confidence_numeric > 0:
        # Dynamic position sizing based on confidence and agreement
        base_size = min(combined_confidence_numeric * 100, 100)
        agreement_multiplier = 1.0 if agreement_level == 'high' else 0.7 if agreement_level == 'medium' else 0.5
        position_size = base_size * agreement_multiplier
        
        # Calculate stop loss and take profit with selected risk-reward ratio
        atr = tool_signals.get('Fibonacci', {}).get('latest', {}).get('atr', current_price * 0.02)
        stop_loss, take_profit = calculate_tp_sl_2_1_rr(current_price, combined_action, atr, current_price)
        
    else:
        position_size = 0
        stop_loss = None
        take_profit = None
    
    return {
        'action': combined_action,
        'confidence': combined_confidence,
        'confidence_numeric': combined_confidence_numeric,
        'position_size': position_size,
        'entry_price': current_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'reasons': reasons,
        'agreement_level': agreement_level,
        'risk_reward_ratio': selected_rr_ratio,
        'tool_breakdown': {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'total_direct_tools': total_direct_tools,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'agreement_percentage': agreement_percentage
        }
    }

# Trade management functions - Gann is excluded
def update_trades(tool, symbol, current_price, trade_data=None):
    """Update active trades for a tool and check for TP/SL"""
    # Skip trade management for Gann
    if tool == "Gann":
        return
        
    active_key = tool + '_active_trades'
    history_key = tool + '_trade_history'
    
    # Update existing trades
    trades_to_remove = []
    has_open_trade = False
    for trade_key, trade in list(st.session_state[active_key].items()):
        if trade['symbol'] == symbol:
            has_open_trade = True
            # Check for TP/SL hit
            hit_sl = False
            hit_tp = False
            
            if trade['action'] in ["BUY", "Buy"]:
                hit_sl = current_price <= trade['stop_loss']
                hit_tp = current_price >= trade['take_profit']
            else:  # SELL or Sell
                hit_sl = current_price >= trade['stop_loss']
                hit_tp = current_price <= trade['take_profit']
            
            if hit_sl or hit_tp:
                # Close trade
                outcome = 'win' if hit_tp else 'loss'
                status = 'hit_tp' if hit_tp else 'hit_sl'
                close_price = trade['take_profit'] if hit_tp else trade['stop_loss']
                
                # Calculate profit percentage
                if trade['action'] in ["BUY", "Buy"]:
                    profit_pct = ((close_price - trade['entry_price']) / trade['entry_price']) * 100
                else:
                    profit_pct = ((trade['entry_price'] - close_price) / trade['entry_price']) * 100
                
                # Create closed trade record
                closed_trade = trade.copy()
                closed_trade.update({
                    'outcome': outcome,
                    'status': status,
                    'close_price': close_price,
                    'profit_pct': profit_pct,
                    'close_time': datetime.now(pkt_tz),
                    'trade_id': trade.get('trade_id', str(uuid.uuid4())[:8])
                })
                
                st.session_state[history_key].append(closed_trade)
                trades_to_remove.append(trade_key)
                
                st.success(f"{tool} Trade for {symbol} closed: {outcome.upper()} ({status}) with {profit_pct:.2f}% return")
    
    # Remove closed trades
    for trade_key in trades_to_remove:
        del st.session_state[active_key][trade_key]
    
    # Open new trade if provided and action is BUY or SELL (not HOLD) and no open trade
    if trade_data and trade_data.get('action') in ["BUY", "SELL"]:
        if has_open_trade:
            st.warning(f"Already trade open for {symbol} with {tool}. New signal detected but not opening another trade.")
        else:
            trade_id = f"{symbol}_{tool}_{st.session_state.trade_counter}"
            st.session_state.trade_counter += 1
            
            new_trade = {
                'trade_id': trade_id,
                'symbol': symbol,
                'action': trade_data['action'],
                'entry_price': trade_data.get('entry_price', current_price),
                'stop_loss': trade_data.get('stop_loss'),
                'take_profit': trade_data.get('take_profit'),
                'entry_time': datetime.now(pkt_tz),
                'signals': trade_data.get('signals', []),
                'signal_descriptions': trade_data.get('signal_descriptions', []),
                'confidence': trade_data.get('confidence', 'Low'),
                'interval': trade_data.get('interval', 'N/A'),
                'degree': trade_data.get('degree', 'N/A'),
                'position_size': trade_data.get('position_size', 100),
                'reasons': trade_data.get('reasons', []),
                'risk_reward_ratio': trade_data.get('risk_reward_ratio', selected_rr_ratio)
            }
            
            st.session_state[active_key][trade_id] = new_trade
            st.info(f"{tool} New {trade_data['action']} trade started for {symbol} at {new_trade['entry_price']:.4f}")

def save_trade_history_to_csv(tool):
    """Save trade history to CSV file"""
    # Skip for Gann
    if tool == "Gann":
        return None
        
    history_key = tool + '_trade_history'
    
    if st.session_state[history_key]:
        df = pd.DataFrame(st.session_state[history_key])
        
        # Ensure directory exists
        os.makedirs('trade_history', exist_ok=True)
        
        filename = f'trade_history/{tool.lower()}_trades.csv'
        df.to_csv(filename, index=False)
        return filename
    return None

# Main analysis logic
if st.session_state.run_analysis:
    try:
        # Combined signals tab
        combined_tab, individual_tab = st.tabs(["🔗 Combined Signals", "📊 Individual Tools"])
        
        with combined_tab:
            st.header("🔗 Combined Trading Signals")
            
            for symbol in symbols:
                with st.expander(f"Combined Analysis for {symbol}", expanded=True):
                    interval = symbol_intervals.get(symbol, "5m")
                    
                    with st.spinner(f'Fetching data for {symbol} ({interval})...'):
                        df_live = fetch_candles_from_binance(symbol, interval, candle_limit)
                        if df_live is None:
                            st.error(f"Skipping analysis for {symbol} due to data fetch failure.")
                            continue

                    current_price = fetch_current_price(symbol) or df_live['close'].iloc[-1]
                    
                    # Collect signals from all tools
                    tool_signals = {}
                    for tool in selected_tools:
                        params = tool_params.get(tool, {})
                        
                        try:
                            if tool == "Fibonacci":
                                results = fib.compute_fib_results(df_live, params, symbol, interval)
                                tool_signals[tool] = {
                                    'trade_action': results['trade_action'].upper() if results['trade_action'] else None,
                                    'confidence': results['confidence'],
                                    'entry_price': results['entry_price'],
                                    'stop_loss': results['stop_loss'],
                                    'take_profit': results['take_profit'],
                                    'signal_descriptions': results['signal_descriptions'],
                                    'latest': results['latest']
                                }
                            elif tool == "Elliott":
                                results = elliott.compute_elliott_results(df_live, params, symbol, interval)
                                # Use the first signal for combined analysis
                                if results['signals']:
                                    signal = results['signals'][0]
                                    tool_signals[tool] = {
                                        'trade_action': signal['action'].upper(),
                                        'confidence': results['wave_data_by_degree'][signal.get('degree', 'Minor')].get('confidence', 'medium'),
                                        'entry_price': signal['entry_price'],
                                        'stop_loss': signal['sl'],
                                        'take_profit': signal['tp'],
                                        'signal_descriptions': [signal['reason']]
                                    }
                                else:
                                    tool_signals[tool] = {'trade_action': None}
                            elif tool == "Ichimoku":
                                results = ichimoku.compute_ichimoku_results(df_live, params, symbol, interval)
                                if results['signals']:
                                    signal = results['signals'][0]
                                    tool_signals[tool] = {
                                        'trade_action': signal['type'].upper(),
                                        'confidence': results['confidence'],
                                        'entry_price': signal['entry_price'],
                                        'stop_loss': signal['sl'],
                                        'take_profit': signal['tp'],
                                        'signal_descriptions': [signal['reason']]
                                    }
                                else:
                                    tool_signals[tool] = {'trade_action': None}
                            elif tool == "Wyckoff":
                                results = wyckoff.compute_wyckoff_results(df_live, params, symbol, interval)
                                if results['signals']:
                                    signal = results['signals'][0]
                                    tool_signals[tool] = {
                                        'trade_action': signal['type'].upper(),
                                        'confidence': results['confidence'],
                                        'entry_price': signal['entry_price'],
                                        'stop_loss': signal['sl'],
                                        'take_profit': signal['tp'],
                                        'signal_descriptions': [signal['reason']]
                                    }
                                else:
                                    tool_signals[tool] = {'trade_action': None}
                            elif tool == "Gann":
                                # Gann provides indirect information only
                                results = gann.compute_gann_results(df_live, params, symbol, interval, current_price)
                                tool_signals[tool] = {
                                    'trade_action': None,  # Gann doesn't generate direct signals
                                    'confidence': 'Low',
                                    'signal_descriptions': ["Gann provides trend analysis, not direct signals"],
                                    'gann_results': results
                                }
                        except Exception as e:
                            st.error(f"Error in {tool} analysis for {symbol}: {str(e)}")
                            tool_signals[tool] = {'trade_action': None}
                            continue
                    
                    # Generate combined signal
                    combined_signal = generate_combined_signal(tool_signals, current_price, symbol, interval, tool_weights)
                    
                    # Store for later display
                    st.session_state.combined_signals[symbol] = combined_signal
                    
                    # Display combined signal
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if combined_signal['action'] in ["BUY", "SELL"]:
                            signal_class = "combined-signal-strong" if combined_signal['confidence'] == 'High' else "combined-signal-weak"
                            st.markdown(f'<div class="{signal_class}">', unsafe_allow_html=True)
                            st.subheader(f"🎯 {combined_signal['action']} Signal")
                            confidence_class = f"confidence-{combined_signal['confidence'].lower()}"
                            st.markdown(f"**Confidence:** <span class='{confidence_class}'>{combined_signal['confidence']}</span>", unsafe_allow_html=True)
                            st.metric("Position Size", f"{combined_signal['position_size']:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:  # HOLD
                            st.markdown('<div class="combined-signal-neutral">', unsafe_allow_html=True)
                            st.subheader("🛑 HOLD Signal")
                            confidence_class = f"confidence-{combined_signal['confidence'].lower()}"
                            st.markdown(f"**Confidence:** <span class='{confidence_class}'>{combined_signal['confidence']}</span>", unsafe_allow_html=True)
                            st.metric("Position Size", "0% (Maintain)")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Trade Details")
                        if combined_signal['action'] in ["BUY", "SELL"]:
                            st.write(f"**Entry Price:** {combined_signal['entry_price']:.4f}")
                            st.write(f"**Stop Loss:** {combined_signal['stop_loss']:.4f}")
                            st.write(f"**Take Profit:** {combined_signal['take_profit']:.4f}")
                            st.write(f"**Entry Time (PKT):** {datetime.now(pkt_tz)}")
                            
                            # Risk/Reward Ratio with visual indicator
                            if combined_signal['action'] == "BUY":
                                risk = combined_signal['entry_price'] - combined_signal['stop_loss']
                                reward = combined_signal['take_profit'] - combined_signal['entry_price']
                            else:
                                risk = combined_signal['stop_loss'] - combined_signal['entry_price']
                                reward = combined_signal['entry_price'] - combined_signal['take_profit']
                            
                            if risk > 0:
                                rr_ratio = reward / risk
                                rr_class = "rr-ratio-good" if rr_ratio >= 2 else "rr-ratio-poor"
                                st.markdown(f"**Risk/Reward:** <span class='{rr_class}'>{rr_ratio:.2f}:1</span>", unsafe_allow_html=True)
                                st.write(f"**Strategy:** 1 win = {selected_rr_ratio} losses covered")
                        else:
                            st.write("**Recommendation:** Hold current position or stay out")
                            st.write("**Entry Price:** N/A")
                            st.write("**Stop Loss:** N/A")
                            st.write("**Take Profit:** N/A")
                    
                    with col3:
                        st.subheader("Tool Agreement")
                        breakdown = combined_signal['tool_breakdown']
                        agreement_class = f"tool-agreement-{combined_signal['agreement_level']}"
                        st.markdown(f'<div class="{agreement_class}">', unsafe_allow_html=True)
                        st.write(f"**Agreement Level:** {combined_signal['agreement_level'].upper()}")
                        st.write(f"**Agreement %:** {breakdown['agreement_percentage']:.1%}")
                        st.write(f"**Direct Tools:** {breakdown['total_direct_tools']}")
                        st.write(f"**Buy Signals:** {breakdown['buy_count']}")
                        st.write(f"**Sell Signals:** {breakdown['sell_count']}")
                        st.write(f"**Hold Signals:** {breakdown['hold_count']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.subheader("Configuration")
                        st.write(f"**Risk-Reward:** {risk_reward_ratio}")
                        st.write("**Tool Weights:**")
                        for tool in selected_tools:
                            if tool != "Gann":  # Gann doesn't have weight
                                weight = tool_weights.get(tool, 1.0)
                                st.write(f"- {tool}: {weight:.1f}x")
                        st.write("- Gann: 0.5x (indirect)")
                    
                    # Display detailed signal table
                    st.subheader("📋 Detailed Signal Breakdown")
                    
                    # Create signal table
                    signal_data = []
                    for tool in selected_tools:
                        if tool in tool_signals:
                            signal_info = tool_signals[tool]
                            if tool == "Gann":
                                # Special handling for Gann
                                gann_analysis = analyze_gann_information(signal_info.get('gann_results'), current_price)
                                confidence_display = gann_analysis['confidence']
                                signal = gann_analysis['bias'].upper() if gann_analysis['bias'] != 'neutral' else 'HOLD'
                                reason = gann_analysis['reasons'][0] if gann_analysis['reasons'] else 'No clear bias - Neutral position'
                                signal_data.append({
                                    'Tool': tool,
                                    'Signal': signal,
                                    'Confidence': confidence_display,
                                    'Weight': "0.5x",
                                    'Type': 'Indirect',
                                    'Reason': reason
                                })
                            else:
                                action = signal_info.get('trade_action', 'NONE')
                                confidence_raw = signal_info.get('confidence', 'Low')
                                
                                # Display original confidence for tools
                                if isinstance(confidence_raw, str):
                                    confidence_display = confidence_raw.capitalize()
                                else:
                                    confidence_display = convert_numeric_to_confidence(confidence_raw)
                                
                                signal_data.append({
                                    'Tool': tool,
                                    'Signal': action,
                                    'Confidence': confidence_display,
                                    'Weight': f"{tool_weights.get(tool, 1.0):.1f}x",
                                    'Type': 'Direct',
                                    'Reason': signal_info.get('signal_descriptions', ['No reason'])[0]
                                })
                    
                    if signal_data:
                        signal_df = pd.DataFrame(signal_data)
                        st.dataframe(signal_df.style.hide(axis="index"), use_container_width=True)
                    
                    # Display Gann subtools details table if available
                    if "Gann" in selected_tools:
                        gann_info_df = tool_signals.get("Gann", {}).get('gann_results', {}).get('info_df', pd.DataFrame())
                        if not gann_info_df.empty:
                            st.subheader("📊 Gann Subtools Details (Actual Values)")
                            st.dataframe(gann_info_df.style.hide(axis="index"), use_container_width=True)
                    
                    # Display reasons
                    st.subheader("🎯 Combined Signal Reasons")
                    for reason in combined_signal['reasons']:
                        st.write(f"• {reason}")
                    
                    # Update combined trades (Gann is automatically excluded)
                    if combined_signal['action'] in ["BUY", "SELL"]:
                        trade_data = {
                            'action': combined_signal['action'],
                            'entry_price': combined_signal['entry_price'],
                            'stop_loss': combined_signal['stop_loss'],
                            'take_profit': combined_signal['take_profit'],
                            'confidence': combined_signal['confidence'],
                            'interval': interval,
                            'position_size': combined_signal['position_size'],
                            'reasons': combined_signal['reasons'],
                            'risk_reward_ratio': combined_signal['risk_reward_ratio']
                        }
                        update_trades("Combined", symbol, current_price, trade_data)
                    else:
                        update_trades("Combined", symbol, current_price)
        
        with individual_tab:
            # Original individual tool analysis
            for symbol in symbols:
                interval = symbol_intervals.get(symbol, "5m")
                with st.expander(f"Individual Analysis for {symbol} ({interval})", expanded=True):
                    with st.spinner(f'Fetching data for {symbol} ({interval})...'):
                        df_live = fetch_candles_from_binance(symbol, interval, candle_limit)
                        if df_live is None:
                            st.error(f"Skipping analysis for {symbol} due to data fetch failure.")
                            continue

                    current_price = fetch_current_price(symbol) or df_live['close'].iloc[-1]

                    # Filter out Gann from trading tools for tabs
                    trading_tools = [tool for tool in selected_tools if tool != "Gann"]
                    if "Gann" in selected_tools:
                        # Add Gann as a separate tab
                        tabs = st.tabs(trading_tools + ["Gann (Info Only)"])
                    else:
                        tabs = st.tabs(trading_tools)
                    
                    tab_index = 0
                    for tool in selected_tools:
                        if tool != "Gann":
                            with tabs[tab_index]:
                                params = tool_params.get(tool, {})
                                if tool == "Fibonacci":
                                    results = fib.compute_fib_results(df_live, params, symbol, interval)
                                    # Update trades for Fibonacci
                                    if results['trade_action']:
                                        trade_data = {
                                            'action': results['trade_action'].upper(),
                                            'entry_price': results['entry_price'],
                                            'stop_loss': results['stop_loss'],
                                            'take_profit': results['take_profit'],
                                            'confidence': convert_numeric_to_confidence(results['confidence']),
                                            'interval': interval,
                                            'signal_descriptions': results['signal_descriptions']
                                        }
                                        update_trades(tool, symbol, current_price, trade_data)
                                    else:
                                        update_trades(tool, symbol, current_price)
                                    fib.display_fib_results(results)

                                elif tool == "Elliott":
                                    results = elliott.compute_elliott_results(df_live, params, symbol, interval)
                                    # Update trades for Elliott - use first signal only
                                    if results['signals']:
                                        signal = results['signals'][0]  # Take first signal
                                        trade_data = {
                                            'action': signal['action'].upper(),
                                            'entry_price': signal['entry_price'],
                                            'stop_loss': signal['sl'],
                                            'take_profit': signal['tp'],
                                            'confidence': results['wave_data_by_degree'][signal.get('degree', 'Minor')].get('confidence', 'medium'),
                                            'interval': interval,
                                            'degree': signal.get('degree', 'Minor'),
                                            'signal_descriptions': [signal['reason']]
                                        }
                                        update_trades(tool, symbol, current_price, trade_data)
                                    else:
                                        update_trades(tool, symbol, current_price)
                                    elliott.display_elliott_results(results)

                                elif tool == "Ichimoku":
                                    results = ichimoku.compute_ichimoku_results(df_live, params, symbol, interval)
                                    # Update trades for Ichimoku - use first signal only
                                    if results['signals']:
                                        signal = results['signals'][0]  # Take first signal
                                        trade_data = {
                                            'action': signal['type'].upper(),
                                            'entry_price': signal['entry_price'],
                                            'stop_loss': signal['sl'],
                                            'take_profit': signal['tp'],
                                            'confidence': results['confidence'],
                                            'interval': interval,
                                            'signal_descriptions': [signal['reason']]
                                        }
                                        update_trades(tool, symbol, current_price, trade_data)
                                    else:
                                        update_trades(tool, symbol, current_price)
                                    ichimoku.display_ichimoku_results(results)

                                elif tool == "Wyckoff":
                                    results = wyckoff.compute_wyckoff_results(df_live, params, symbol, interval)
                                    # Update trades for Wyckoff - use first signal only
                                    if results['signals']:
                                        signal = results['signals'][0]  # Take first signal
                                        trade_data = {
                                            'action': signal['type'].upper(),
                                            'entry_price': signal['entry_price'],
                                            'stop_loss': signal['sl'],
                                            'take_profit': signal['tp'],
                                            'confidence': results['confidence'],
                                            'interval': interval,
                                            'signal_descriptions': [signal['reason']]
                                        }
                                        update_trades(tool, symbol, current_price, trade_data)
                                    else:
                                        update_trades(tool, symbol, current_price)
                                    wyckoff.display_wyckoff_results(results, df_live, symbol, interval, params)
                            tab_index += 1
                        else:
                            # Gann tab (information only)
                            with tabs[-1]:
                                st.info("**Gann Analysis - Information Only**")
                                st.warning("Gann provides indirect trend analysis and does not generate direct trading signals")
                                params = tool_params.get("Gann", {})
                                results = gann.compute_gann_results(df_live, params, symbol, interval, current_price)
                                gann.display_gann_results(results, df_live, symbol, interval, params)
                                # No trade management for Gann

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.info("Check symbol format (e.g., BTCUSDT) or try again.")

# Display Trades and History Section
st.header("📊 Trade Management")

# Create tabs for different tools and combined - Exclude Gann
trade_tools = [tool for tool in selected_tools if tool != "Gann"]
trade_tabs = st.tabs(["Combined"] + trade_tools)

# Combined trades tab
with trade_tabs[0]:
    st.subheader("🔗 Combined Active Trades")
    active_key = "Combined_active_trades"
    active_trades_list = []
    for trade_key, trade in st.session_state[active_key].items():
        current_price = fetch_current_price(trade['symbol']) or trade['entry_price']
        pl_pct = ((current_price - trade['entry_price']) / trade['entry_price'] * 100) if trade['action'] == "BUY" else ((trade['entry_price'] - current_price) / trade['entry_price'] * 100)
        active_trade = trade.copy()
        active_trade['current_price'] = current_price
        active_trade['current_pl_pct'] = pl_pct
        active_trade['status'] = 'open'
        active_trades_list.append(active_trade)
    
    if active_trades_list:
        active_df = pd.DataFrame(active_trades_list)
        display_cols = ['symbol', 'action', 'entry_price', 'stop_loss', 'take_profit', 'current_price', 'current_pl_pct', 'status', 'entry_time', 'interval', 'position_size', 'confidence']
        display_cols = [col for col in display_cols if col in active_df.columns]
        st.dataframe(active_df[display_cols].style.format({
            'entry_price': '{:.4f}',
            'stop_loss': '{:.4f}',
            'take_profit': '{:.4f}',
            'current_price': '{:.4f}',
            'current_pl_pct': '{:.2f}%'
        }), use_container_width=True)
    else:
        st.info("No active combined trades.")

    st.subheader("🔗 Combined Trade History")
    history_key = "Combined_trade_history"
    if st.session_state[history_key]:
        history_df = pd.DataFrame(st.session_state[history_key])
        display_cols = ['symbol', 'action', 'entry_price', 'stop_loss', 'take_profit', 'close_price', 'profit_pct', 'outcome', 'status', 'entry_time', 'close_time', 'interval', 'position_size', 'confidence']
        display_cols = [col for col in display_cols if col in history_df.columns]
        
        st.dataframe(history_df[display_cols].style.format({
            'entry_price': '{:.4f}',
            'stop_loss': '{:.4f}',
            'take_profit': '{:.4f}',
            'close_price': '{:.4f}',
            'profit_pct': '{:.2f}%',
            'position_size': '{:.1f}%'
        }), use_container_width=True)
        
        # Statistics
        wins = sum(1 for t in st.session_state[history_key] if t['outcome'] == 'win')
        losses = len(st.session_state[history_key]) - wins
        total_return = sum(t['profit_pct'] for t in st.session_state[history_key])
        win_rate = (wins / len(st.session_state[history_key])) * 100 if st.session_state[history_key] else 0
        
        # Calculate required win rate for profitability with current RR ratio
        required_win_rate = (1 / (1 + selected_rr_ratio)) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Wins", wins)
        with col2:
            st.metric("Losses", losses)
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col5:
            st.metric("Req Win Rate", f"{required_win_rate:.1f}%", 
                     delta=f"{win_rate - required_win_rate:.1f}%" if win_rate >= required_win_rate else f"{win_rate - required_win_rate:.1f}%",
                     delta_color="normal" if win_rate >= required_win_rate else "inverse")
        
        # Save to CSV
        if st.button("💾 Save Combined Trade History to CSV"):
            filename = save_trade_history_to_csv("Combined")
            if filename:
                st.success(f"Combined trade history saved to {filename}")
                
                # Provide download link
                with open(filename, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download CSV",
                        data=file,
                        file_name=filename,
                        mime="text/csv"
                    )
            else:
                st.warning("No trade history to save")
    else:
        st.info("No combined trade history yet.")

# Individual tool trade tabs (Gann excluded)
for i, tool in enumerate(trade_tools, 1):
    with trade_tabs[i]:
        st.subheader(f"{tool} Active Trades")
        active_key = tool + '_active_trades'
        active_trades_list = []
        for trade_key, trade in st.session_state[active_key].items():
            current_price = fetch_current_price(trade['symbol']) or trade['entry_price']
            pl_pct = ((current_price - trade['entry_price']) / trade['entry_price'] * 100) if trade['action'] == "BUY" else ((trade['entry_price'] - current_price) / trade['entry_price'] * 100)
            active_trade = trade.copy()
            active_trade['current_price'] = current_price
            active_trade['current_pl_pct'] = pl_pct
            active_trade['status'] = 'open'
            active_trades_list.append(active_trade)
        
        if active_trades_list:
            active_df = pd.DataFrame(active_trades_list)
            display_cols = ['symbol', 'action', 'entry_price', 'stop_loss', 'take_profit', 'current_price', 'current_pl_pct', 'status', 'entry_time', 'interval', 'degree', 'confidence']
            display_cols = [col for col in display_cols if col in active_df.columns]
            st.dataframe(active_df[display_cols].style.format({
                'entry_price': '{:.4f}',
                'stop_loss': '{:.4f}',
                'take_profit': '{:.4f}',
                'current_price': '{:.4f}',
                'current_pl_pct': '{:.2f}%'
            }), use_container_width=True)
        else:
            st.info(f"No active trades for {tool}.")

        st.subheader(f"{tool} Trade History")
        history_key = tool + '_trade_history'
        if st.session_state[history_key]:
            history_df = pd.DataFrame(st.session_state[history_key])
            display_cols = ['symbol', 'action', 'entry_price', 'stop_loss', 'take_profit', 'close_price', 'profit_pct', 'outcome', 'status', 'entry_time', 'close_time', 'interval', 'degree', 'confidence']
            display_cols = [col for col in display_cols if col in history_df.columns]
            
            st.dataframe(history_df[display_cols].style.format({
                'entry_price': '{:.4f}',
                'stop_loss': '{:.4f}',
                'take_profit': '{:.4f}',
                'close_price': '{:.4f}',
                'profit_pct': '{:.2f}%'
            }), use_container_width=True)
            
            wins = sum(1 for t in st.session_state[history_key] if t['outcome'] == 'win')
            losses = len(st.session_state[history_key]) - wins
            total_return = sum(t['profit_pct'] for t in st.session_state[history_key])
            win_rate = (wins / len(st.session_state[history_key])) * 100 if st.session_state[history_key] else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Wins", wins)
            with col2:
                st.metric("Losses", losses)
            with col3:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col4:
                st.metric("Total Return", f"{total_return:.2f}%")
            
            # Save to CSV
            if st.button(f"💾 Save {tool} Trade History to CSV", key=f"save_{tool}"):
                filename = save_trade_history_to_csv(tool)
                if filename:
                    st.success(f"{tool} trade history saved to {filename}")
                    
                    # Provide download link
                    with open(filename, "rb") as file:
                        btn = st.download_button(
                            label="📥 Download CSV",
                            data=file,
                            file_name=filename,
                            mime="text/csv"
                        )
                else:
                    st.warning(f"No {tool} trade history to save")
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


