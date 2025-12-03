#!/usr/bin/env python3
"""
web_state.py - Independent SMA 365 + 120 Trading Dashboard
Fetches data directly from Kraken API and maintains own state file
"""

from flask import Flask, render_template_string
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

app = Flask(__name__)

# Import the Kraken Futures API client
import kraken_futures as kf

# Configuration
STATE_FILE = Path("web_state.json")
UPDATE_INTERVAL = 300  # 5 minutes
SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_OHLC_KRAKEN = "XBTUSD"
INTERVAL_KRAKEN = 1440
SMA_PERIOD_LONG = 365
SMA_PERIOD_SHORT = 120
ATR_PERIOD = 14
STATIC_STOP_PCT = 3.0  # 3% static stop
LEV = 2.5  # 2.5x leverage

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("web_dashboard")

class DashboardMonitor:
    def __init__(self):
        self.api = None
        self.last_update = None
        self.state = self.load_state()
        
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
                "sma_period_long": SMA_PERIOD_LONG,
                "sma_period_short": SMA_PERIOD_SHORT,
                "stop_loss_pct": STATIC_STOP_PCT,
                "leverage": LEV
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

    def calculate_sma_and_atr(self, df: pd.DataFrame) -> tuple:
        """Calculate SMA 365, SMA 120, and ATR values"""
        if len(df) < SMA_PERIOD_LONG:
            return 0, 0, 0
        
        df = df.copy()
        
        # Calculate SMAs
        sma_365 = df['close'].rolling(window=SMA_PERIOD_LONG).mean().iloc[-1]
        sma_120 = df['close'].rolling(window=SMA_PERIOD_SHORT).mean().iloc[-1]
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=ATR_PERIOD).mean().iloc[-1]
        
        return float(sma_365), float(sma_120), float(atr)

    def generate_signal(self, current_price: float, sma_365: float, sma_120: float) -> str:
        """Generate trading signal based on both SMAs"""
        if current_price > sma_365 and current_price > sma_120:
            return "LONG"
        elif current_price < sma_365 and current_price < sma_120:
            return "SHORT"
        else:
            return "FLAT"

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
            sma_365, sma_120, atr = self.calculate_sma_and_atr(ohlc_data)
            signal = self.generate_signal(mark_price, sma_365, sma_120)
            
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
                "sma_365": sma_365,
                "sma_120": sma_120,
                "atr": atr,
                "signal": signal,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Log new trade if position changed
            self._detect_new_trade(current_position)
            
            self.save_state()
            self.last_update = time.time()
            log.info("Dashboard data updated successfully")
            
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
                "sma_365": self.state["market_data"].get("sma_365", 0),
                "sma_120": self.state["market_data"].get("sma_120", 0),
                "atr": self.state["market_data"].get("atr", 0),
                "stop_distance": (current_price * (STATIC_STOP_PCT / 100)) if current_position else 0,
                "stop_loss_pct": STATIC_STOP_PCT
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
    <title>SMA 365 + 120 BTC Trading Dashboard</title>
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
        <h1>SMA 365 + 120 Trading Dashboard</h1>
        <div class="subtitle">
            <span class="status-indicator {% if data_fresh %}status-live{% else %}status-offline{% endif %}"></span>
            Dual SMA Strategy (365/120) with 3% Static Stop | 2.5x Leverage | Limit Orders
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
                <h2>SMA 365</h2>
                <div class="card-value">${{ sma_365 }}</div>
                <div class="card-label">365-day Moving Average</div>
            </div>
            <div class="card">
                <h2>SMA 120</h2>
                <div class="card-value">${{ sma_120 }}</div>
                <div class="card-label">120-day Moving Average (Filter)</div>
            </div>
            <div class="card">
                <h2>Signal</h2>
                <div class="card-value">{{ market_signal }}</div>
                <div class="card-label">Price vs Both SMAs</div>
            </div>
        </div>

        <!-- Strategy Information -->
        <div class="section">
            <h2>Strategy Configuration</h2>
            <div class="strategy-info">
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Strategy Type</div>
                    <div class="strategy-stat-value">Dual SMA</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Long SMA</div>
                    <div class="strategy-stat-value">{{ sma_period_long }} days</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Short SMA (Filter)</div>
                    <div class="strategy-stat-value">{{ sma_period_short }} days</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Stop Loss</div>
                    <div class="strategy-stat-value">{{ stop_loss_pct }}% Static</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Leverage</div>
                    <div class="strategy-stat-value">{{ leverage }}x</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Order Type</div>
                    <div class="strategy-stat-value">Limit (0.02%)</div>
                </div>
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
                        <th>SMA 365</th>
                        <th>SMA 120</th>
                        <th>Stop %</th>
                        <th>Stop Distance</th>
                        <th>Portfolio Value</th>
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
                        <td>${{ trade.sma_365 }}</td>
                        <td>${{ trade.sma_120 }}</td>
                        <td>{{ trade.stop_loss_pct }}%</td>
                        <td>${{ trade.stop_distance }}</td>
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
    sma_365 = f"{market_data.get('sma_365', 0):.2f}"
    sma_120 = f"{market_data.get('sma_120', 0):.2f}"
    market_signal = market_data.get('signal', 'N/A')
    
    # Strategy info
    strategy_info = state.get("strategy_info", {})
    sma_period_long = strategy_info.get('sma_period_long', 365)
    sma_period_short = strategy_info.get('sma_period_short', 120)
    stop_loss_pct = strategy_info.get('stop_loss_pct', 5.0)
    leverage = strategy_info.get('leverage', 4)
    
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
        trade_copy['sma_365'] = f"{trade.get('sma_365', 0):.2f}"
        trade_copy['sma_120'] = f"{trade.get('sma_120', 0):.2f}"
        trade_copy['stop_distance'] = f"{trade.get('stop_distance', 0):.2f}"
        trade_copy['stop_loss_pct'] = f"{trade.get('stop_loss_pct', 5.0):.1f}"
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
        sma_365=sma_365,
        sma_120=sma_120,
        market_signal=market_signal,
        sma_period_long=sma_period_long,
        sma_period_short=sma_period_short,
        stop_loss_pct=stop_loss_pct,
        leverage=leverage,
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
