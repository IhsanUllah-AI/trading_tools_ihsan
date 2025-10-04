# Trading Analysis System

A **multi-tool trading analysis app** built with **Streamlit**.  
It integrates 5 major analysis methods into one master system with a unified control panel.

## Tools Included
- **Elliott Wave** (wave detection + Fibonacci retracements/extensions)  
- **Fibonacci Retracement & Extensions** (SL/TP, Risk/Reward, indicator confidence)  
- **Ichimoku Cloud** (trend signals with Tenkan, Kijun, Senkou, Chikou)  
- **Wyckoff Method** (phase detection, scalping signals, confidence scoring)  
- **Gann Theory** (Fan, Square, Box, Fixed Square with TradingView validation)  

## How It Works
- **Main Page**  
  - User selects **coin(s)**, **timeframe(s)**, and chooses which **tools** to run.  
  - Default settings are pre-optimized but can be adjusted.  
  - A single **"Run Analysis"** button executes all selected tools in parallel.  
  - Results from each tool are summarized on the main page.  

- **Pages**  
  - Each tool also has its own Streamlit page for deeper inspection.  
  - Trade results (Win/Loss, SL/TP) are logged into CSV per tool.  
  - Summary page compares statistics across all tools.  

## Features
- Parallel analysis on multiple coins and intervals.  
- Trade management system (prevents overlapping trades).  
- Auto-refresh with live Binance data (every 60s).  
- Confidence scoring with RSI, Bollinger Bands, EMA, MACD, Fib.  
- Supports both **individual tool review** and **combined workflow**.  


