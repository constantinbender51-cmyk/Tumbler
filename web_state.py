#!/usr/bin/env python3
"""
web_state.py - Comprehensive Trading Dashboard
Displays execution logs, predictions, and performance metrics
"""

from flask import Flask, render_template_string
import json
from pathlib import Path
from datetime import datetime, timezone
import os
import sys

# Import Kraken API to fetch live data
try:
    import kraken_futures as kf
except ImportError:
    kf = None
    print("WARNING: kraken_futures not available, live data will not be fetched")

app = Flask(__name__)

STATE_FILE = Path("tumbler_state.json")

# Kraken API credentials
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")
SYMBOL_FUTS_UC = "PF_XBTUSD"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>LSTM BTC Trading Dashboard</title>
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
        .model-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1px;
            border: 1px solid #000000;
        }
        .model-stat {
            background: #ffffff;
            padding: 20px;
            border-right: 1px solid #000000;
            border-bottom: 1px solid #000000;
        }
        .model-stat:last-child {
            border-right: none;
        }
        .model-stat-label {
            font-size: 0.75em;
            color: #666666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }
        .model-stat-value {
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
        .prediction-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 12px;
            background: white;
            margin-bottom: 1px;
            border: 1px solid #e0e0e0;
        }
        .prediction-date {
            font-weight: 600;
            color: #000000;
            font-size: 0.9em;
        }
        .prediction-values {
            display: flex;
            gap: 30px;
        }
        .prediction-item {
            text-align: center;
        }
        .prediction-item-label {
            font-size: 0.75em;
            color: #666666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .prediction-item-value {
            font-size: 1em;
            font-weight: 600;
            color: #000000;
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
        <h1>Tumbler Trading Dashboard</h1>
        <div class="subtitle">LSTM Neural Network Strategy with 20-Day Lookback and On-Chain Metrics</div>
        
        {% if total_trades == 0 %}
        <div style="background: #f5f5f5; color: #666666; padding: 20px; border: 1px solid #e0e0e0; margin-bottom: 40px; text-align: center; font-size: 0.9em;">
            <strong>Awaiting first trade execution at 00:01 UTC</strong><br>
            <span style="font-size: 0.85em; color: #999999;">Model trained. Live trading data will appear after execution.</span>
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
            <div class="card">
                <h2>Today's Prediction</h2>
                <div class="card-value">${{ today_prediction }}</div>
                <div class="card-label">Next close estimate</div>
            </div>
        </div>

        <!-- Model Information -->
        <div class="section">
            <h2>Model Configuration</h2>
            <div class="model-info">
                <div class="model-stat">
                    <div class="model-stat-label">Model Type</div>
                    <div class="model-stat-value">LSTM Neural Network</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">Training MSE</div>
                    <div class="model-stat-value">{{ train_mse }}</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">Lookback Period</div>
                    <div class="model-stat-value">{{ lookback }} days</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">Leverage</div>
                    <div class="model-stat-value">{{ leverage }}x</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">Last Trained</div>
                    <div class="model-stat-value">{{ last_trained }}</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">Rebalancing</div>
                    <div class="model-stat-value">Daily 00:01 UTC</div>
                </div>
            </div>
        </div>

        <!-- Recent Predictions vs Actuals -->
        <div class="section">
            <h2>Recent Predictions vs Actuals</h2>
            {% if predictions %}
                <div style="border: 1px solid #000000;">
                {% for pred in predictions[-10:]|reverse %}
                <div class="prediction-row">
                    <div class="prediction-date">{{ pred.date }}</div>
                    <div class="prediction-values">
                        <div class="prediction-item">
                            <div class="prediction-item-label">Predicted</div>
                            <div class="prediction-item-value">${{ pred.predicted }}</div>
                        </div>
                        <div class="prediction-item">
                            <div class="prediction-item-label">Actual</div>
                            <div class="prediction-item-value">
                                {% if pred.actual %}
                                    ${{ pred.actual }}
                                {% else %}
                                    Pending
                                {% endif %}
                            </div>
                        </div>
                        {% if pred.actual %}
                        <div class="prediction-item">
                            <div class="prediction-item-label">Error</div>
                            <div class="prediction-item-value {{ 'positive' if ((pred.predicted_raw - pred.actual_raw)|abs < 100) else 'negative' }}">
                                {{ ((pred.predicted_raw - pred.actual_raw) / pred.actual_raw * 100)|round(2) }}%
                            </div>
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                </div>
            {% else %}
                <div class="no-data">No predictions yet. Waiting for first trade...</div>
            {% endif %}
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
                        <th>Portfolio Value</th>
                        <th>Yesterday Pred</th>
                        <th>Yesterday Actual</th>
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
                        <td>${{ trade.portfolio_value }}</td>
                        <td>${{ trade.yesterday_prediction }}</td>
                        <td>${{ trade.yesterday_actual }}</td>
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


def get_live_kraken_data():
    """Fetch live data directly from Kraken API"""
    if not kf or not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        print("DEBUG: Cannot fetch live Kraken data - API not configured")
        return None
    
    try:
        api = kf.KrakenFuturesApi(KRAKEN_API_KEY, KRAKEN_API_SECRET)
        
        # Get portfolio value
        accounts = api.get_accounts()
        portfolio_value = float(accounts["accounts"]["flex"]["portfolioValue"])
        
        # Get mark price
        tickers = api.get_tickers()
        mark_price = 0
        for t in tickers["tickers"]:
            if t["symbol"] == SYMBOL_FUTS_UC:
                mark_price = float(t["markPrice"])
                break
        
        # Get open position
        positions = api.get_open_positions()
        current_position = None
        for p in positions.get("openPositions", []):
            if p["symbol"] == SYMBOL_FUTS_UC:
                current_position = {
                    "signal": "LONG" if p["side"] == "long" else "SHORT",
                    "side": p["side"],
                    "size_btc": abs(float(p["size"])),
                    "fill_price": float(p.get("fillPrice", 0)),
                }
                break
        
        print(f"DEBUG: Fetched live Kraken data - portfolio=${portfolio_value:.2f}, mark=${mark_price:.2f}, position={current_position}")
        
        return {
            "portfolio_value": portfolio_value,
            "mark_price": mark_price,
            "current_position": current_position
        }
    except Exception as e:
        print(f"ERROR: Failed to fetch live Kraken data: {e}")
        return None


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "trades": [],
        "predictions": [],
        "starting_capital": None,
        "performance": {},
        "model_info": {},
        "current_position": None,
        "current_portfolio_value": 0
    }


@app.route('/')
def dashboard():
    state = load_state()
    
    # DEBUG: Print entire state
    print("="*80)
    print("DEBUG: Full state loaded:")
    print(json.dumps(state, indent=2))
    print("="*80)
    
    # Fetch live Kraken data
    live_data = get_live_kraken_data()
    if live_data:
        print(f"DEBUG: Live Kraken data available")
        # Override state with live data
        if live_data["current_position"]:
            state["current_position"] = live_data["current_position"]
        state["current_portfolio_value"] = live_data["portfolio_value"]
        
        # Update performance with live data
        if state.get("starting_capital") is None and live_data["portfolio_value"] > 0:
            state["starting_capital"] = live_data["portfolio_value"]
        
        if state.get("starting_capital"):
            total_return = (live_data["portfolio_value"] - state["starting_capital"]) / state["starting_capital"] * 100
            if "performance" not in state:
                state["performance"] = {}
            state["performance"]["current_value"] = live_data["portfolio_value"]
            state["performance"]["starting_capital"] = state["starting_capital"]
            state["performance"]["total_return_pct"] = total_return
    
    # Current position - prioritize live position from Kraken
    current_position = state.get("current_position")
    print(f"DEBUG: current_position = {current_position}")
    
    current_signal = "N/A"
    current_size = "0.0000"
    current_price = "0.00"
    
    if current_position:
        # Use live position from Kraken
        current_signal = current_position["signal"]
        current_size = f"{current_position['size_btc']:.4f}"
        current_price = f"{current_position['fill_price']:.2f}"
        print(f"DEBUG: Using current_position - signal={current_signal}, size={current_size}, price={current_price}")
    elif state["trades"]:
        # Fall back to last trade if no live position
        last_trade = state["trades"][-1]
        current_signal = last_trade["signal"]
        current_size = f"{last_trade['size_btc']:.4f}"
        current_price = f"{last_trade['fill_price']:.2f}"
        print(f"DEBUG: Using last trade - signal={current_signal}, size={current_size}, price={current_price}")
    else:
        print("DEBUG: No position or trades found")
    
    # Today's prediction
    today_prediction = "N/A"
    if state["trades"]:
        last_trade = state["trades"][-1]
        today_prediction = f"{last_trade['today_prediction']:.2f}"
        print(f"DEBUG: today_prediction = {today_prediction}")
    else:
        print("DEBUG: No trades for prediction")
    
    # Performance metrics - use live portfolio value if available
    performance = state.get("performance", {})
    print(f"DEBUG: performance = {performance}")
    
    current_value_raw = state.get("current_portfolio_value", performance.get('current_value', 0))
    print(f"DEBUG: current_value_raw = {current_value_raw}")
    
    current_value = f"{current_value_raw:.2f}"
    starting_capital = f"{performance.get('starting_capital', 0):.2f}"
    total_return_raw = performance.get('total_return_pct', 0)
    total_return = f"{total_return_raw:.2f}"
    total_trades = performance.get('total_trades', 0)
    
    print(f"DEBUG: Final values - current_value={current_value}, starting_capital={starting_capital}, total_return={total_return}, total_trades={total_trades}")
    print("="*80)
    
    # Model info
    model_info = state.get("model_info", {})
    train_mse = f"{model_info.get('train_mse', 0):.2f}"
    lookback = model_info.get('lookback', 20)
    leverage = model_info.get('leverage', 5.0)
    last_trained = model_info.get('last_trained', 'N/A')
    if last_trained != 'N/A':
        try:
            dt = datetime.fromisoformat(last_trained)
            last_trained = dt.strftime('%Y-%m-%d %H:%M')
        except:
            pass
    
    # Format predictions
    predictions = []
    for pred in state.get("predictions", []):
        pred_copy = pred.copy()
        try:
            dt = datetime.fromisoformat(pred['date'])
            pred_copy['date'] = dt.strftime('%Y-%m-%d')
        except:
            pred_copy['date'] = pred['date']
        
        # Keep raw values for calculations
        pred_copy['predicted_raw'] = pred['predicted'] if pred['predicted'] else 0
        pred_copy['actual_raw'] = pred.get('actual') if pred.get('actual') else None
        
        # Format for display
        pred_copy['predicted'] = f"{pred['predicted']:.2f}" if pred['predicted'] else 'N/A'
        pred_copy['actual'] = f"{pred['actual']:.2f}" if pred.get('actual') else None
        predictions.append(pred_copy)
    
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
        trade_copy['yesterday_prediction'] = f"{trade['yesterday_prediction']:.2f}"
        trade_copy['yesterday_actual'] = f"{trade['yesterday_actual']:.2f}"
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
        today_prediction=today_prediction,
        train_mse=train_mse,
        lookback=lookback,
        leverage=leverage,
        last_trained=last_trained,
        predictions=predictions,
        trades=trades,
        last_updated=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    )


@app.route('/debug')
def debug():
    """Debug endpoint to check state file and live data"""
    debug_info = {
        "state_file_path": str(STATE_FILE.absolute()),
        "state_file_exists": STATE_FILE.exists(),
        "state_file_size": STATE_FILE.stat().st_size if STATE_FILE.exists() else 0,
        "kraken_api_configured": bool(KRAKEN_API_KEY and KRAKEN_API_SECRET and kf),
        "state_content": load_state(),
        "live_kraken_data": get_live_kraken_data()
    }
    return f"<pre>{json.dumps(debug_info, indent=2)}</pre>"


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
