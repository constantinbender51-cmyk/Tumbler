#!/usr/bin/env python3
"""
web_state.py - SMA 365 Trading Dashboard
Autonomously fetches data from Kraken API and updates state every 5 minutes
Displays execution logs and performance metrics
"""

from flask import Flask, render_template_string
import json
from pathlib import Path
from datetime import datetime, timezone
import os
import time
import threading
import logging
from typing import Dict, Any, Optional

# Import Kraken API client
import kraken_futures as kf

app = Flask(__name__)

STATE_FILE = Path("sma_state.json")
UPDATE_INTERVAL = 300  # 5 minutes in seconds

# Strategy configuration
SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_FUTS_LC = "pf_xbtusd"
SMA_PERIOD = 365
ATR_PERIOD = 14
ATR_MULTIPLIER = 3.2
LEVERAGE = 1.5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("web_state")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>SMA 365 BTC Trading Dashboard</title>
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
        .update-info {
            text-align: center;
            color: #666666;
            margin-bottom: 20px;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SMA 365 Trading Dashboard</h1>
        <div class="subtitle">365-Day Simple Moving Average Strategy with ATR Stop Loss</div>
        
        <div class="update-info">
            <span class="status-indicator status-{{ status_class }}"></span>
            Data updates every 5 minutes | Last update: {{ last_data_update }}
        </div>
        
        {% if total_trades == 0 %}
        <div style="background: #f5f5f5; color: #666666; padding: 20px; border: 1px solid #e0e0e0; margin-bottom: 40px; text-align: center; font-size: 0.9em;">
            <strong>Awaiting first trade execution at 00:01 UTC</strong><br>
            <span style="font-size: 0.85em; color: #999999;">Strategy initialized. Live trading data will appear after execution.</span>
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
                <div class="card-label">{{ total_trades }} trades executed</div>
            </div>
        </div>

        <!-- Strategy Information -->
        <div class="section">
            <h2>Strategy Configuration</h2>
            <div class="strategy-info">
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Strategy Type</div>
                    <div class="strategy-stat-value">SMA 365</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">SMA Period</div>
                    <div class="strategy-stat-value">{{ sma_period }} days</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">ATR Period</div>
                    <div class="strategy-stat-value">{{ atr_period }} days</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">ATR Multiplier</div>
                    <div class="strategy-stat-value">{{ atr_multiplier }}x</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Leverage</div>
                    <div class="strategy-stat-value">{{ leverage }}x</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Rebalancing</div>
                    <div class="strategy-stat-value">Daily 00:01 UTC</div>
                </div>
            </div>
        </div>

        <!-- Account Information -->
        <div class="section">
            <h2>Account Information</h2>
            <div class="strategy-info">
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Account Balance</div>
                    <div class="strategy-stat-value">${{ account_balance }}</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Available Margin</div>
                    <div class="strategy-stat-value">${{ available_margin }}</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Used Margin</div>
                    <div class="strategy-stat-value">${{ used_margin }}</div>
                </div>
                <div class="strategy-stat">
                    <div class="strategy-stat-label">Unrealized P&L</div>
                    <div class="strategy-stat-value {{ 'positive' if unrealized_pnl >= 0 else 'negative' }}">${{ unrealized_pnl }}</div>
                </div>
            </div>
        </div>

        <!-- Trade History -->
        <div class="section">
            <h2>Trade Execution Log</h2>
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
                        <th>ATR</th>
                        <th>Stop Distance</th>
                        <th>Portfolio Value</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades|reverse %}
                    <tr>
                        <td>{{ trade.timestamp }}</td>
                        <td><span class="badge badge-{{ trade.signal.lower() }}">{{ trade.signal }}</span></td>
                        <td class="{{ 'long' if trade.side == 'buy' else 'short' }}">{{ trade.side.upper() }}</td>
                        <td>{{ trade.size_btc }}</td>
                        <td>${{ trade.fill_price }}</td>
                        <td>${{ trade.sma_365 }}</td>
                        <td>${{ trade.atr }}</td>
                        <td>${{ trade.stop_distance }}</td>
                        <td>${{ trade.portfolio_value }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
                <div class="no-data">No trades yet. Waiting for first execution...</div>
            {% endif %}
        </div>

        <div class="timestamp">
            Last updated: {{ last_updated }} UTC | Auto-refreshes every 30 seconds
        </div>
    </div>
</body>
</html>
"""


class KrakenDataFetcher:
    def __init__(self):
        self.api_key = os.getenv("KRAKEN_API_KEY")
        self.api_secret = os.getenv("KRAKEN_API_SECRET")
        self.api = None
        self.initialize_api()
    
    def initialize_api(self):
        """Initialize Kraken API client"""
        if self.api_key and self.api_secret:
            try:
                self.api = kf.KrakenFuturesApi(self.api_key, self.api_secret)
                log.info("Kraken API client initialized successfully")
            except Exception as e:
                log.error(f"Failed to initialize Kraken API: {e}")
                self.api = None
        else:
            log.warning("Kraken API credentials not found in environment variables")
            self.api = None
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value in USD"""
        if not self.api:
            return 0.0
        
        try:
            accounts = self.api.get_accounts()
            return float(accounts["accounts"]["flex"]["portfolioValue"])
        except Exception as e:
            log.error(f"Failed to get portfolio value: {e}")
            return 0.0
    
    def get_mark_price(self) -> float:
        """Get current mark price for BTC"""
        if not self.api:
            return 0.0
        
        try:
            tickers = self.api.get_tickers()
            for ticker in tickers["tickers"]:
                if ticker["symbol"] == SYMBOL_FUTS_UC:
                    return float(ticker["markPrice"])
            return 0.0
        except Exception as e:
            log.error(f"Failed to get mark price: {e}")
            return 0.0
    
    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """Get current open position from Kraken"""
        if not self.api:
            return None
        
        try:
            positions = self.api.get_open_positions()
            for position in positions.get("openPositions", []):
                if position["symbol"] == SYMBOL_FUTS_UC:
                    return {
                        "signal": "LONG" if position["side"] == "long" else "SHORT",
                        "side": position["side"],
                        "size_btc": abs(float(position["size"])),
                        "fill_price": float(position.get("fillPrice", 0)),
                        "unrealized_pnl": float(position.get("unrealizedFunding", 0)),
                    }
            return None
        except Exception as e:
            log.error(f"Failed to get current position: {e}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance information"""
        if not self.api:
            return {"balance": 0.0, "available_margin": 0.0, "used_margin": 0.0}
        
        try:
            accounts = self.api.get_accounts()
            flex_account = accounts["accounts"]["flex"]
            return {
                "balance": float(flex_account.get("balance", 0)),
                "available_margin": float(flex_account.get("availableMargin", 0)),
                "used_margin": float(flex_account.get("usedMargin", 0)),
            }
        except Exception as e:
            log.error(f"Failed to get account balance: {e}")
            return {"balance": 0.0, "available_margin": 0.0, "used_margin": 0.0}
    
    def get_trade_history(self) -> list:
        """Get recent trade history"""
        if not self.api:
            return []
        
        try:
            # Get recent fills (trades)
            fills = self.api.get_fills({"limit": 50})
            return fills.get("fills", [])
        except Exception as e:
            log.error(f"Failed to get trade history: {e}")
            return []


def load_state():
    """Load state from file"""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception as e:
            log.error(f"Failed to load state file: {e}")
    
    # Return default state
    return {
        "trades": [],
        "starting_capital": None,
        "performance": {},
        "strategy_info": {},
        "current_position": None,
        "current_portfolio_value": 0,
        "account_balance": {},
        "last_update": None
    }


def save_state(state: Dict[str, Any]):
    """Save state to file"""
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
        log.info(f"State file updated at {datetime.now(timezone.utc).isoformat()}")
    except Exception as e:
        log.error(f"Failed to save state file: {e}")


def update_state_from_kraken():
    """Fetch data from Kraken API and update state file"""
    log.info("Updating state from Kraken API...")
    
    fetcher = KrakenDataFetcher()
    state = load_state()
    
    # Get current data from Kraken
    portfolio_value = fetcher.get_portfolio_value()
    current_position = fetcher.get_current_position()
    account_balance = fetcher.get_account_balance()
    mark_price = fetcher.get_mark_price()
    
    # Update state with current data
    state["current_portfolio_value"] = portfolio_value
    state["current_position"] = current_position
    state["account_balance"] = account_balance
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    
    # Initialize starting capital if not set
    if state["starting_capital"] is None and portfolio_value > 0:
        state["starting_capital"] = portfolio_value
        log.info(f"Initialized starting capital: ${portfolio_value:.2f}")
    
    # Calculate performance metrics
    if state["starting_capital"] and portfolio_value > 0:
        total_return_pct = (portfolio_value - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": portfolio_value,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return_pct,
            "total_trades": len(state.get("trades", [])),
        }
    
    # Update strategy info
    state["strategy_info"] = {
        "sma_period": SMA_PERIOD,
        "atr_period": ATR_PERIOD,
        "atr_multiplier": ATR_MULTIPLIER,
        "leverage": LEVERAGE,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    # Save updated state
    save_state(state)
    
    # Log update
    if current_position:
        log.info(f"Position: {current_position['signal']} {current_position['size_btc']:.4f} BTC @ ${current_position['fill_price']:.2f}")
    else:
        log.info("No current position")
    
    log.info(f"Portfolio: ${portfolio_value:.2f} | Mark: ${mark_price:.2f}")


def background_updater():
    """Background thread to update state every 5 minutes"""
    while True:
        try:
            update_state_from_kraken()
        except Exception as e:
            log.error(f"Background update failed: {e}")
        
        # Wait 5 minutes
        time.sleep(UPDATE_INTERVAL)


@app.route('/')
def dashboard():
    state = load_state()
    
    # Determine status
    last_update = state.get("last_update")
    if last_update:
        try:
            last_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - last_dt).total_seconds()
            status_class = "live" if age < UPDATE_INTERVAL * 2 else "offline"
            last_data_update = last_dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            status_class = "offline"
            last_data_update = "Never"
    else:
        status_class = "offline"
        last_data_update = "Never"
    
    # Current position
    current_position = state.get("current_position")
    current_signal = "FLAT"
    current_size = "0.0000"
    current_price = "0.00"
    unrealized_pnl = 0.0
    
    if current_position:
        current_signal = current_position["signal"]
        current_size = f"{current_position['size_btc']:.4f}"
        current_price = f"{current_position['fill_price']:.2f}"
        unrealized_pnl = current_position.get("unrealized_pnl", 0)
    
    # Performance metrics
    performance = state.get("performance", {})
    current_value_raw = state.get("current_portfolio_value", performance.get('current_value', 0))
    current_value = f"{current_value_raw:.2f}" if current_value_raw else "0.00"
    
    starting_capital_raw = performance.get('starting_capital') or state.get('starting_capital') or 0
    starting_capital = f"{starting_capital_raw:.2f}" if starting_capital_raw else "0.00"
    
    total_return_raw = performance.get('total_return_pct', 0)
    total_return = f"{total_return_raw:.2f}" if total_return_raw else "0.00"
    total_trades = performance.get('total_trades', len(state.get("trades", [])))
    
    # Strategy info
    strategy_info = state.get("strategy_info", {})
    sma_period = strategy_info.get('sma_period', SMA_PERIOD)
    atr_period = strategy_info.get('atr_period', ATR_PERIOD)
    atr_multiplier = strategy_info.get('atr_multiplier', ATR_MULTIPLIER)
    leverage = strategy_info.get('leverage', LEVERAGE)
    
    # Account balance
    account_balance_info = state.get("account_balance", {})
    account_balance = f"{account_balance_info.get('balance', 0):.2f}"
    available_margin = f"{account_balance_info.get('available_margin', 0):.2f}"
    used_margin = f"{account_balance_info.get('used_margin', 0):.2f}"
    
    # Format trades
    trades = []
    for trade in state.get("trades", []):
        trade_copy = trade.copy()
        try:
            dt = datetime.fromisoformat(trade['timestamp'])
            trade_copy['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        trade_copy['size_btc'] = f"{trade.get('size_btc', 0):.4f}"
        trade_copy['fill_price'] = f"{trade.get('fill_price', 0):.2f}"
        trade_copy['portfolio_value'] = f"{trade.get('portfolio_value', 0):.2f}"
        trade_copy['sma_365'] = f"{trade.get('sma_365', 0):.2f}"
        trade_copy['atr'] = f"{trade.get('atr', 0):.2f}"
        trade_copy['stop_distance'] = f"{trade.get('stop_distance', 0):.2f}"
        trades.append(trade_copy)
    
    return render_template_string(
        HTML_TEMPLATE,
        current_signal=current_signal,
        current_size=current_size,
        current_price=current_price,
        current_value=current_value,
        starting_capital=starting_capital,
        total_return=total_return,
        total_return_raw=total_return_raw,
        total_trades=total_trades,
        sma_period=sma_period,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        leverage=leverage,
        account_balance=account_balance,
        available_margin=available_margin,
        used_margin=used_margin,
        unrealized_pnl=unrealized_pnl,
        trades=trades,
        last_updated=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        last_data_update=last_data_update,
        status_class=status_class
    )


def start_background_updater():
    """Start the background update thread"""
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    log.info("Background state updater started")


if __name__ == '__main__':
    # Start background updater
    start_background_updater()
    
    # Run Flask app
    port = int(os.getenv('PORT', 8080))
    log.info(f"Starting web dashboard on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
