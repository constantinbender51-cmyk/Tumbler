#!/usr/bin/env python3
"""
web_state.py - SMA 365 Trading Dashboard
Displays execution logs and performance metrics
"""

from flask import Flask, render_template_string
import json
from pathlib import Path
from datetime import datetime, timezone
import os

app = Flask(__name__)

STATE_FILE = Path("sma_state.json")

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
    </style>
</head>
<body>
    <div class="container">
        <h1>SMA 365 Trading Dashboard</h1>
        <div class="subtitle">365-Day Simple Moving Average Strategy with ATR Stop Loss</div>
        
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


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "trades": [],
        "starting_capital": None,
        "performance": {},
        "strategy_info": {},
        "current_position": None,
        "current_portfolio_value": 0
    }


@app.route('/')
def dashboard():
    state = load_state()
    
    # Debug logging
    app.logger.info(f"State loaded: {len(state)} keys")
    app.logger.info(f"Current portfolio value: {state.get('current_portfolio_value')}")
    app.logger.info(f"Starting capital: {state.get('starting_capital')}")
    app.logger.info(f"Performance: {state.get('performance')}")
    app.logger.info(f"Number of trades: {len(state.get('trades', []))}")
    
    # Current position - prioritize live position from Kraken
    current_position = state.get("current_position")
    current_signal = "N/A"
    current_size = "0.0000"
    current_price = "0.00"
    
    if current_position:
        # Use live position from Kraken
        current_signal = current_position["signal"]
        current_size = f"{current_position['size_btc']:.4f}"
        current_price = f"{current_position['fill_price']:.2f}"
    elif state["trades"]:
        # Fall back to last trade if no live position
        last_trade = state["trades"][-1]
        current_signal = last_trade["signal"]
        current_size = f"{last_trade['size_btc']:.4f}"
        current_price = f"{last_trade['fill_price']:.2f}"
    
    # Performance metrics - use live portfolio value if available
    performance = state.get("performance", {})
    current_value_raw = state.get("current_portfolio_value", performance.get('current_value', 0))
    
    # If still zero, try to get from last trade
    if current_value_raw == 0 and state.get("trades"):
        current_value_raw = state["trades"][-1].get("portfolio_value", 0)
    
    current_value = f"{current_value_raw:.2f}"
    
    starting_capital_raw = performance.get('starting_capital', state.get('starting_capital', 0))
    starting_capital = f"{starting_capital_raw:.2f}"
    
    total_return_raw = performance.get('total_return_pct', 0)
    
    # Calculate return if we have values but no calculated return
    if total_return_raw == 0 and starting_capital_raw > 0 and current_value_raw > 0:
        total_return_raw = (current_value_raw - starting_capital_raw) / starting_capital_raw * 100
    
    total_return = f"{total_return_raw:.2f}"
    total_trades = performance.get('total_trades', len(state.get("trades", [])))
    
    # Strategy info
    strategy_info = state.get("strategy_info", {})
    sma_period = strategy_info.get('sma_period', 365)
    atr_period = strategy_info.get('atr_period', 14)
    atr_multiplier = strategy_info.get('atr_multiplier', 3.2)
    leverage = strategy_info.get('leverage', 1.5)
    
    # Format trades
    trades = []
    for trade in state.get("trades", []):
        trade_copy = trade.copy()
        try:
            dt = datetime.fromisoformat(trade['timestamp'])
            trade_copy['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        trade_copy['size_btc'] = f"{trade['size_btc']:.4f}"
        trade_copy['fill_price'] = f"{trade['fill_price']:.2f}"
        trade_copy['portfolio_value'] = f"{trade['portfolio_value']:.2f}"
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
        trades=trades,
        last_updated=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    )


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
