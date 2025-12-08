#!/usr/bin/env python3
"""
web_state.py - Dual SMA Strategy Dashboard with III Dynamic Leverage
Displays strategy with SMA 1 (57), SMA 2 (124), III, and dynamic leverage
"""

from flask import Flask, render_template_string, send_file
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

app = Flask(__name__)

# Import the Kraken Futures API client
import kraken_futures as kf

# Configuration
STATE_FILE = Path("web_state.json")
BACKTEST_CHART_PATH = Path("/app/static/backtest.png")
UPDATE_INTERVAL = 300  # 5 minutes
SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_OHLC_KRAKEN = "XBTUSD"
INTERVAL_KRAKEN = 1440
SMA_PERIOD_1 = 40
SMA_PERIOD_2 = 120
STATIC_STOP_PCT = 2.0  # 2% static stop
TAKE_PROFIT_PCT = 16.0  # 16% take profit
III_WINDOW = 35
III_T_LOW = 0.13
III_T_HIGH = 0.18
LEV_LOW = 0.5
LEV_MID = 4.5
LEV_HIGH = 2.45

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("web_dashboard")

class DashboardMonitor:
    def __init__(self):
        self.api = None
        self.last_update = None
        self.state = self.load_state()
        # Ensure static directory exists
        static_dir = Path("/app/static")
        static_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_api(self):
        """Initialize Kraken API client"""
        api_key = os.getenv("KRAKEN_API_KEY")
        api_sec = os.getenv("KRAKEN_API_SECRET")
        if not api_key or not api_sec:
            raise ValueError("KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables required")
        
        self.api = kf.KrakenFuturesApi(api_key, api_sec)
        log.info("Kraken API client initialized")

    def load_state(self) -> Dict[str, Any]:
        """Load state from local file"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Error loading state file: {e}")
        
        # Default state
        return {
            "trades": [],
            "performance": {
                "current_value": 0,
                "starting_capital": 0,
                "total_return_pct": 0,
                "total_trades": 0
            },
            "current_position": None,
            "market_data": {},
            "strategy_info": {
                "sma_period_1": SMA_PERIOD_1,
                "sma_period_2": SMA_PERIOD_2,
                "stop_loss_pct": STATIC_STOP_PCT,
                "take_profit_pct": TAKE_PROFIT_PCT,
                "iii_window": III_WINDOW,
                "leverage_tiers": f"{LEV_LOW}x / {LEV_MID}x / {LEV_HIGH}x"
            },
            "last_updated": None
        }

    def save_state(self):
        """Save state to local file"""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
            log.info("State saved successfully")
        except Exception as e:
            log.error(f"Error saving state file: {e}")

    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            accounts = self.api.get_accounts()
            return float(accounts["accounts"]["flex"]["portfolioValue"])
        except Exception as e:
            log.error(f"Error getting portfolio value: {e}")
            return 0.0

    def get_mark_price(self) -> float:
        """Get current mark price"""
        try:
            tickers = self.api.get_tickers()
            for ticker in tickers["tickers"]:
                if ticker["symbol"] == SYMBOL_FUTS_UC:
                    return float(ticker["markPrice"])
            raise RuntimeError("Mark price not found")
        except Exception as e:
            log.error(f"Error getting mark price: {e}")
            return 0.0

    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """Get current open position"""
        try:
            positions = self.api.get_open_positions()
            for position in positions.get("openPositions", []):
                if position["symbol"] == SYMBOL_FUTS_UC:
                    return {
                        "signal": "LONG" if position["side"] == "long" else "SHORT",
                        "side": position["side"],
                        "size_btc": abs(float(position["size"])),
                        "unrealized_pnl": float(position.get("unrealizedFunding", 0))
                    }
            return None
        except Exception as e:
            log.error(f"Error getting current position: {e}")
            return None

    def get_ohlc_data(self) -> pd.DataFrame:
        """Get OHLC data for analysis"""
        try:
            params = {"pair": SYMBOL_OHLC_KRAKEN, "interval": INTERVAL_KRAKEN}
            response = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            
            if payload["error"]:
                raise RuntimeError("Kraken error: " + ", ".join(payload["error"]))
            
            key = list(payload["result"].keys())[0]
            raw = payload["result"][key]
            
            df = pd.DataFrame(
                raw,
                columns=["time", "open", "high", "low", "close", "vwap", "volume", "trades"],
            )
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            return df.astype(float)
        except Exception as e:
            log.error(f"Error getting OHLC data: {e}")
            return pd.DataFrame()

    def calculate_iii(self, df: pd.DataFrame) -> float:
        """Calculate Inefficiency Index (III) over the last 35 days"""
        if len(df) < III_WINDOW + 1:
            return 0.0
        
        recent_df = df.tail(III_WINDOW + 1).copy()
        recent_df['log_ret'] = np.log(recent_df['close'] / recent_df['close'].shift(1))
        log_rets = recent_df['log_ret'].dropna()
        
        if len(log_rets) < III_WINDOW:
            return 0.0
        
        net_direction = abs(log_rets.sum())
        path_length = log_rets.abs().sum()
        epsilon = 1e-8
        iii = net_direction / (path_length + epsilon)
        
        return iii
    
    def determine_leverage(self, iii: float) -> float:
        """Determine leverage based on III value"""
        if iii < III_T_LOW:
            return LEV_LOW
        elif iii < III_T_HIGH:
            return LEV_MID
        else:
            return LEV_HIGH

    def calculate_smas(self, df: pd.DataFrame) -> tuple:
        """Calculate SMA 1 (57) and SMA 2 (124)"""
        if len(df) < SMA_PERIOD_2:
            return 0, 0
        
        df = df.copy()
        sma_1 = df['close'].rolling(window=SMA_PERIOD_1).mean().iloc[-1]
        sma_2 = df['close'].rolling(window=SMA_PERIOD_2).mean().iloc[-1]
        
        return float(sma_1), float(sma_2)
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 1000.0) -> pd.DataFrame:
        """
        Run backtest on historical data (last 720 days)
        Returns DataFrame with equity, signal, and leverage for each day
        """
        # Limit to 720 days
        df = df.tail(720).copy()
        
        # Calculate SMAs
        df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
        df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
        
        # Calculate III
        iii_values = []
        for i in range(len(df)):
            if i < III_WINDOW:
                iii_values.append(0.0)
            else:
                window_df = df.iloc[max(0, i-III_WINDOW):i+1]
                iii = self.calculate_iii(window_df)
                iii_values.append(iii)
        df['iii'] = iii_values
        
        # Determine leverage for each day
        df['leverage'] = df['iii'].apply(self.determine_leverage)
        
        # Generate signals (using previous close vs SMAs)
        signals = []
        for i in range(len(df)):
            if i == 0 or pd.isna(df['sma_1'].iloc[i]) or pd.isna(df['sma_2'].iloc[i]):
                signals.append('FLAT')
            else:
                prev_close = df['close'].iloc[i-1]
                sma_1 = df['sma_1'].iloc[i]
                sma_2 = df['sma_2'].iloc[i]
                
                if prev_close > sma_1 and prev_close > sma_2:
                    signals.append('LONG')
                elif prev_close < sma_1 and prev_close < sma_2:
                    signals.append('SHORT')
                else:
                    signals.append('FLAT')
        df['signal'] = signals
        
        # Backtest with SL/TP
        equity = initial_capital
        equity_curve = []
        
        for i in range(len(df)):
            if i == 0:
                equity_curve.append(equity)
                continue
            
            signal = df['signal'].iloc[i]
            leverage = df['leverage'].iloc[i]
            open_price = df['open'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            close_price = df['close'].iloc[i]
            
            daily_return = 0.0
            
            if signal == 'LONG':
                entry = open_price
                sl = entry * (1 - STATIC_STOP_PCT / 100)
                tp = entry * (1 + TAKE_PROFIT_PCT / 100)
                
                if low_price <= sl:
                    daily_return = -(STATIC_STOP_PCT / 100)
                elif high_price >= tp:
                    daily_return = (TAKE_PROFIT_PCT / 100)
                else:
                    daily_return = (close_price - entry) / entry
                    
            elif signal == 'SHORT':
                entry = open_price
                sl = entry * (1 + STATIC_STOP_PCT / 100)
                tp = entry * (1 - TAKE_PROFIT_PCT / 100)
                
                if high_price >= sl:
                    daily_return = -(STATIC_STOP_PCT / 100)
                elif low_price <= tp:
                    daily_return = (TAKE_PROFIT_PCT / 100)
                else:
                    daily_return = (entry - close_price) / entry
            
            # Apply leverage
            leveraged_return = daily_return * leverage
            equity *= (1 + leveraged_return)
            equity_curve.append(equity)
        
        df['equity'] = equity_curve
        return df
    
    def generate_backtest_chart(self):
        """Generate backtest chart and save as PNG"""
        try:
            log.info("Generating backtest chart...")
            
            # Get OHLC data
            ohlc_data = self.get_ohlc_data()
            if ohlc_data.empty:
                log.error("No OHLC data available for backtest")
                return
            
            # Get portfolio value for initial capital
            portfolio_value = self.state["performance"].get("starting_capital", 1000)
            
            # Run backtest
            backtest_df = self.run_backtest(ohlc_data, portfolio_value)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(backtest_df.index, backtest_df['equity'], color='black', linewidth=1.5, label='Equity')
            
            # Color background by signal
            for i in range(1, len(backtest_df)):
                signal = backtest_df['signal'].iloc[i]
                color = 'lightgreen' if signal == 'LONG' else 'lightcoral' if signal == 'SHORT' else 'lightgray'
                ax1.axvspan(backtest_df.index[i-1], backtest_df.index[i], alpha=0.3, color=color)
            
            ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
            ax1.set_title(f'720-Day Backtest: SMA({SMA_PERIOD_1}/{SMA_PERIOD_2}) + III Dynamic Leverage', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
            # Plot leverage
            colors_lev = []
            for lev in backtest_df['leverage']:
                if lev == LEV_LOW:
                    colors_lev.append('orange')
                elif lev == LEV_MID:
                    colors_lev.append('green')
                else:
                    colors_lev.append('blue')
            
            ax2.bar(backtest_df.index, backtest_df['leverage'], color=colors_lev, alpha=0.6, width=1)
            ax2.set_ylabel('Leverage (x)', fontsize=11)
            ax2.set_xlabel('Date', fontsize=11)
            ax2.set_title('Dynamic Leverage (0.5x=Orange, 4.5x=Green, 2.45x=Blue)', fontsize=11)
            ax2.set_ylim(0, 5)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Calculate stats
            final_equity = backtest_df['equity'].iloc[-1]
            total_return = (final_equity / portfolio_value - 1) * 100
            
            # Add stats box
            stats_text = f"Initial: ${portfolio_value:.2f}\nFinal: ${final_equity:.2f}\nReturn: {total_return:.1f}%"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(BACKTEST_CHART_PATH, dpi=150, bbox_inches='tight')
            plt.close()
            
            log.info(f"Backtest chart saved to {BACKTEST_CHART_PATH}")
            
        except Exception as e:
            log.error(f"Error generating backtest chart: {e}")
            import traceback
            log.error(traceback.format_exc())

    def generate_signal(self, current_price: float, sma_1: float, sma_2: float, prev_close: float, leverage: float) -> str:
        """Generate trading signal based on dual SMA (simplified backtest logic)"""
        signal = "FLAT"
        
        # LONG: prev_close above both SMAs
        if prev_close > sma_1 and prev_close > sma_2:
            signal = "LONG"
        # SHORT: prev_close below both SMAs
        elif prev_close < sma_1 and prev_close < sma_2:
            signal = "SHORT"
        
        return signal

    def update_data(self):
        """Update all dashboard data"""
        if not self.api:
            self.initialize_api()

        log.info("Updating dashboard data...")
        
        try:
            # Get current market data
            portfolio_value = self.get_portfolio_value()
            mark_price = self.get_mark_price()
            current_position = self.get_current_position()
            ohlc_data = self.get_ohlc_data()
            
            # Calculate technical indicators
            sma_1, sma_2 = self.calculate_smas(ohlc_data)
            iii = self.calculate_iii(ohlc_data)
            leverage = self.determine_leverage(iii)
            
            # Get previous close for signal generation
            prev_close = ohlc_data['close'].iloc[-2] if len(ohlc_data) > 1 else mark_price
            signal = self.generate_signal(mark_price, sma_1, sma_2, prev_close, leverage)
            
            # Calculate bands for display (not used in signal)
            upper_band = sma_1 * 1.05
            lower_band = sma_1 * 0.95
            
            # Update performance metrics
            if self.state["performance"]["starting_capital"] == 0:
                self.state["performance"]["starting_capital"] = portfolio_value
            
            starting_capital = self.state["performance"]["starting_capital"]
            total_return_pct = 0
            if starting_capital > 0:
                total_return_pct = (portfolio_value - starting_capital) / starting_capital * 100
            
            # Update state
            self.state["performance"] = {
                "current_value": portfolio_value,
                "starting_capital": starting_capital,
                "total_return_pct": total_return_pct,
                "total_trades": len(self.state["trades"])
            }
            
            self.state["current_position"] = current_position
            self.state["market_data"] = {
                "current_price": mark_price,
                "sma_1": sma_1,
                "sma_2": sma_2,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "iii": iii,
                "leverage": leverage,
                "signal": signal,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Log new trade if position changed
            self._detect_new_trade(current_position)
            
            self.save_state()
            self.last_update = time.time()
            log.info("Dashboard data updated successfully")
            
            # Generate backtest chart
            self.generate_backtest_chart()
            
        except Exception as e:
            log.error(f"Error updating dashboard data: {e}")

    def _detect_new_trade(self, current_position: Optional[Dict]):
        """Detect and log new trades based on position changes"""
        old_position = self.state.get("current_position")
        
        # If position changed
        if (old_position is None and current_position is not None) or \
           (old_position is not None and current_position is None) or \
           (old_position and current_position and 
            (old_position["size_btc"] != current_position["size_btc"] or 
             old_position["side"] != current_position["side"])):
            
            # Use current market price
            current_price = self.state["market_data"].get("current_price", 0)
            
            trade_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": current_position["signal"] if current_position else "FLAT",
                "side": current_position["side"] if current_position else "none",
                "size_btc": current_position["size_btc"] if current_position else 0,
                "fill_price": current_price,
                "portfolio_value": self.state["performance"]["current_value"],
                "sma_1": self.state["market_data"].get("sma_1", 0),
                "sma_2": self.state["market_data"].get("sma_2", 0),
                "iii": self.state["market_data"].get("iii", 0),
                "leverage": self.state["market_data"].get("leverage", 0),
                "stop_distance": (current_price * (STATIC_STOP_PCT / 100)) if current_position else 0,
                "stop_loss_pct": STATIC_STOP_PCT,
                "tp_distance": (current_price * (TAKE_PROFIT_PCT / 100)) if current_position else 0,
                "take_profit_pct": TAKE_PROFIT_PCT
            }
            
            self.state["trades"].append(trade_record)
            log.info(f"New trade detected and logged: {trade_record['signal']}")

    def should_update(self) -> bool:
        """Check if it's time to update data"""
        if self.last_update is None:
            return True
        return (time.time() - self.last_update) >= UPDATE_INTERVAL

# Global monitor instance
monitor = DashboardMonitor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Dual SMA Trading Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: #ffffff;
            color: #000000;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 40px;
        }
        h1 {
            color: #000000;
            margin-bottom: 8px;
            font-size: 2em;
            text-align: center;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        .subtitle {
            text-align: center;
            color: #666666;
            margin-bottom: 40px;
            font-size: 0.95em;
            font-weight: 400;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1px;
            margin-bottom: 40px;
            border: 1px solid #000000;
        }
        .card {
            background: #ffffff;
            color: #000000;
            padding: 30px;
            border-right: 1px solid #000000;
            border-bottom: 1px solid #000000;
        }
        .card:last-child {
            border-right: none;
        }
        .card h2 {
            font-size: 0.75em;
            margin-bottom: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666666;
        }
        .card-value {
            font-size: 2.2em;
            font-weight: 300;
            margin-bottom: 8px;
            letter-spacing: -1px;
        }
        .card-label {
            font-size: 0.85em;
            color: #666666;
            font-weight: 400;
        }
        .section {
            background: #ffffff;
            padding: 0;
            margin-bottom: 40px;
            border-top: 2px solid #000000;
        }
        .section h2 {
            color: #000000;
            margin-bottom: 25px;
            margin-top: 25px;
            font-size: 1em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border: 1px solid #000000;
        }
        th {
            background: #000000;
            color: #ffffff;
            padding: 14px 12px;
            text-align: left;
            font-weight: 500;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            padding: 14px 12px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 0.9em;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover {
            background: #f9f9f9;
        }
        .long {
            color: #000000;
            font-weight: 600;
        }
        .short {
            color: #666666;
            font-weight: 600;
        }
        .flat {
            color: #999999;
            font-weight: 600;
        }
        .positive {
            color: #000000;
            font-weight: 600;
        }
        .negative {
            color: #666666;
            font-weight: 600;
        }
        .strategy-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1px;
            border: 1px solid #000000;
        }
        .strategy-stat {
            background: #ffffff;
            padding: 20px;
            border-right: 1px solid #000000;
            border-bottom: 1px solid #000000;
        }
        .strategy-stat:last-child {
            border-right: none;
        }
        .strategy-stat-label {
            font-size: 0.75em;
            color: #666666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        .strategy-stat-value {
            font-size: 1.4em;
            font-weight: 300;
            color: #000000;
        }
        .timestamp {
            text-align: center;
            color: #999999;
            margin-top: 40px;
            font-size: 0.8em;
            font-weight: 400;
        }
        .no-data {
            text-align: center;
            padding: 60px 20px;
            color: #999999;
            font-style: normal;
            font-size: 0.9em;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: 1px solid #000000;
        }
        .badge-long {
            background: #000000;
            color: #ffffff;
        }
        .badge-short {
            background: #ffffff;
            color: #000000;
        }
        .badge-flat {
            background: #f0f0f0;
            color: #666666;
            border-color: #999999;
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-live {
            background: #00ff00;
        }
        .status-offline {
            background: #ff0000;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dual SMA State Machine Strategy</h1>
        <div class="subtitle">
            <span class="status-indicator {% if data_fresh %}status-live{% else %}status-offline{% endif %}"></span>
            SMA1(57) + SMA2(124) + III Dynamic Leverage | 2% SL + 16% TP
        </div>
        
        {% if not api_configured %}
        <div style="background: #fff3cd; color: #856404; padding: 20px; border: 1px solid #ffeaa7; margin-bottom: 40px; text-align: center;">
            <strong>API Configuration Required</strong><br>
            Set KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables to enable live data fetching.
        </div>
        {% endif %}
        
        <!-- Performance Cards -->
        <div class="grid">
            <div class="card">
                <h2>Current Position</h2>
                <div class="card-value">{{ current_signal }}</div>
                <div class="card-label">{{ current_size }} BTC @ ${{ current_price }}</div>
            </div>
            <div class="card">
                <h2>Portfolio Value</h2>
                <div class="card-value">${{ current_value }}</div>
                <div class="card-label">Started: ${{ starting_capital }}</div>
            </div>
            <div class="card">
                <h2>Total Return</h2>
                <div class="card-value {{ 'positive' if total_return_raw >= 0 else 'negative' }}">{{ total_return }}%</div>
                <div class="card-label">{{ total_trades }} trades detected</div>
            </div>
        </div>

        <!-- Market Data -->
        <div class="grid">
            <div class="card">
                <h2>Current Price</h2>
                <div class="card-value">${{ market_price }}</div>
                <div class="card-label">BTC Mark Price</div>
            </div>
            <div class="card">
                <h2>SMA 1 (Logic)</h2>
                <div class="card-value">${{ sma_1 }}</div>
                <div class="card-label">57-day MA with Bands</div>
            </div>
            <div class="card">
                <h2>SMA 2 (Filter)</h2>
                <div class="card-value">${{ sma_2 }}</div>
                <div class="card-label">124-day MA Filter</div>
            </div>
            <div class="card">
                <h2>Signal</h2>
                <div class="card-value">{{ market_signal }}</div>
                <div class="card-label">Current Strategy Signal</div>
            </div>
        </div>

        <!-- III and Leverage -->
        <div class="grid">
            <div class="card">
                <h2>Upper Band</h2>
                <div class="card-value">${{ upper_band }}</div>
                <div class="card-label">SMA1 + 5%</div>
            </div>
            <div class="card">
                <h2>Lower Band</h2>
                <div class="card-value">${{ lower_band }}</div>
                <div class="card-label">SMA1 - 5%</div>
            </div>
            <div class="card">
                <h2>III (35-day)</h2>
                <div class="card-value">{{ iii }}</div>
                <div class="card-label">Inefficiency Index</div>
            </div>
            <div class="card">
                <h2>Dynamic Leverage</h2>
                <div class="card-value">{{ current_leverage }}x</div>
                <div class="card-label">Based on III</div>
            </div>
        </div>

        <!-- Strategy Information -->
        <div class="section">
            <h2>Strategy Configuration</h2>
            <div class="strategy-info">
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Strategy Type</div>
                    <div class="strategy-stat-value">Dual SMA + III</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">SMA 1 (Logic)</div>
                    <div class="strategy-stat-value">{{ sma_period_1 }} days</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">SMA 2 (Filter)</div>
                    <div class="strategy-stat-value">{{ sma_period_2 }} days</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Band Width</div>
                    <div class="strategy-stat-value">{{ band_width_pct }}%</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Stop Loss</div>
                    <div class="strategy-stat-value">{{ stop_loss_pct }}%</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Take Profit</div>
                    <div class="strategy-stat-value">{{ take_profit_pct }}%</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">III Window</div>
                    <div class="strategy-stat-value">{{ iii_window }} days</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Leverage Tiers</div>
                    <div class="strategy-stat-value">0.5x/4.5x/2.45x</div>
                </div>
            </div>
        </div>

        <!-- Backtest Chart -->
        <div class="section">
            <h2>720-Day Historical Performance</h2>
            <div style="text-align: center; padding: 20px; background: #f9f9f9; border: 1px solid #e0e0e0;">
                <img src="/backtest-chart" alt="Backtest Performance Chart" style="max-width: 100%; height: auto; border: 1px solid #ddd;">
            </div>
        </div>

        <!-- Trade History -->
        <div class="section">
            <h2>Trade Detection Log</h2>
            {% if trades %}
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Signal</th>
                        <th>Side</th>
                        <th>Size (BTC)</th>
                        <th>Fill Price</th>
                        <th>SMA 1</th>
                        <th>SMA 2</th>
                        <th>III</th>
                        <th>Leverage</th>
                        <th>SL %</th>
                        <th>TP %</th>
                        <th>Portfolio</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades|reverse %}
                    <tr>
                        <td>{{ trade.timestamp }}</td>
                        <td><span class="badge badge-{{ trade.signal.lower() }}">{{ trade.signal }}</span></td>
                        <td class="{{ trade.signal.lower() }}">{{ trade.side.upper() }}</td>
                        <td>{{ trade.size_btc }}</td>
                        <td>${{ trade.fill_price }}</td>
                        <td>${{ trade.sma_1 }}</td>
                        <td>${{ trade.sma_2 }}</td>
                        <td>{{ trade.iii }}</td>
                        <td>{{ trade.leverage }}x</td>
                        <td>{{ trade.stop_loss_pct }}%</td>
                        <td>{{ trade.take_profit_pct }}%</td>
                        <td>${{ trade.portfolio_value }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
                <div class="no-data">No trades detected yet. Monitoring for position changes...</div>
            {% endif %}
        </div>

        <div class="timestamp">
            Last updated: {{ last_updated }} UTC | Auto-refreshes every 30 seconds | Data updates every 5 minutes
            {% if not data_fresh %}
            <br><span style="color: #ff6b6b;">Data may be stale - check API configuration</span>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    # Update data if needed
    if monitor.should_update():
        try:
            monitor.update_data()
        except Exception as e:
            log.error(f"Failed to update data: {e}")
    
    state = monitor.state
    
    # Check if API is configured
    api_configured = bool(os.getenv("KRAKEN_API_KEY") and os.getenv("KRAKEN_API_SECRET"))
    
    # Check data freshness
    data_fresh = False
    if state.get("last_updated"):
        last_update = datetime.fromisoformat(state["last_updated"].replace('Z', '+00:00'))
        data_fresh = (datetime.now(timezone.utc) - last_update) < timedelta(minutes=10)
    
    # Current position
    current_position = state.get("current_position")
    current_signal = "FLAT"
    current_size = "0.0000"
    
    # Use current market price for display
    market_data = state.get("market_data", {})
    current_price = f"{market_data.get('current_price', 0):.2f}"
    
    if current_position:
        current_signal = current_position["signal"]
        current_size = f"{current_position['size_btc']:.4f}"
    
    # Performance metrics
    performance = state.get("performance", {})
    current_value = f"{performance.get('current_value', 0):.2f}"
    starting_capital = f"{performance.get('starting_capital', 0):.2f}"
    total_return_raw = performance.get('total_return_pct', 0)
    total_return = f"{total_return_raw:.2f}"
    total_trades = performance.get('total_trades', 0)
    
    # Market data
    market_price = f"{market_data.get('current_price', 0):.2f}"
    sma_1 = f"{market_data.get('sma_1', 0):.2f}"
    sma_2 = f"{market_data.get('sma_2', 0):.2f}"
    upper_band = f"{market_data.get('upper_band', 0):.2f}"
    lower_band = f"{market_data.get('lower_band', 0):.2f}"
    market_signal = market_data.get('signal', 'N/A')
    iii = f"{market_data.get('iii', 0):.4f}"
    current_leverage = market_data.get('leverage', 0)
    
    # Strategy info
    strategy_info = state.get("strategy_info", {})
    sma_period_1 = strategy_info.get('sma_period_1', 57)
    sma_period_2 = strategy_info.get('sma_period_2', 124)
    band_width_pct = strategy_info.get('band_width_pct', 5.0)
    stop_loss_pct = strategy_info.get('stop_loss_pct', 2.0)
    take_profit_pct = strategy_info.get('take_profit_pct', 16.0)
    iii_window = strategy_info.get('iii_window', 14)
    
    # Format trades
    trades = []
    for trade in state.get("trades", []):
        trade_copy = trade.copy()
        try:
            dt = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            trade_copy['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            trade_copy['timestamp'] = trade['timestamp']
        trade_copy['size_btc'] = f"{trade.get('size_btc', 0):.4f}"
        trade_copy['fill_price'] = f"{trade.get('fill_price', 0):.2f}"
        trade_copy['portfolio_value'] = f"{trade.get('portfolio_value', 0):.2f}"
        trade_copy['sma_1'] = f"{trade.get('sma_1', 0):.2f}"
        trade_copy['sma_2'] = f"{trade.get('sma_2', 0):.2f}"
        trade_copy['iii'] = f"{trade.get('iii', 0):.4f}"
        trade_copy['leverage'] = f"{trade.get('leverage', 0):.1f}"
        trade_copy['stop_loss_pct'] = f"{trade.get('stop_loss_pct', 2.0):.1f}"
        trade_copy['take_profit_pct'] = f"{trade.get('take_profit_pct', 16.0):.1f}"
        trades.append(trade_copy)
    
    # Reverse for display (newest first)
    trades = list(reversed(trades))
    
    return render_template_string(
        HTML_TEMPLATE,
        api_configured=api_configured,
        data_fresh=data_fresh,
        current_signal=current_signal,
        current_size=current_size,
        current_price=current_price,
        current_value=current_value,
        starting_capital=starting_capital,
        total_return=total_return,
        total_return_raw=total_return_raw,
        total_trades=total_trades,
        market_price=market_price,
        sma_1=sma_1,
        sma_2=sma_2,
        upper_band=upper_band,
        lower_band=lower_band,
        market_signal=market_signal,
        iii=iii,
        current_leverage=current_leverage,
        sma_period_1=sma_period_1,
        sma_period_2=sma_period_2,
        band_width_pct=band_width_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        iii_window=iii_window,
        trades=trades,
        last_updated=state.get("last_updated", "Never").replace('T', ' ').replace('Z', '')
    )

@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "last_update": monitor.last_update,
        "data_fresh": monitor.should_update()
    }

@app.route('/backtest-chart')
def backtest_chart():
    """Serve the backtest chart image"""
    if BACKTEST_CHART_PATH.exists():
        return send_file(str(BACKTEST_CHART_PATH), mimetype='image/png')
    else:
        return "Chart not generated yet", 404

@app.route('/force-update')
def force_update():
    """Force data update"""
    try:
        monitor.update_data()
        return {"status": "success", "message": "Data updated successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

if __name__ == '__main__':
    # Initialize the monitor
    try:
        monitor.initialize_api()
        monitor.update_data()  # Initial data fetch
    except Exception as e:
        log.error(f"Initialization failed: {e}")
        log.info("Dashboard will start in read-only mode")
    
    port = int(os.getenv('PORT', 8080))
    log.info(f"Starting web dashboard on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
