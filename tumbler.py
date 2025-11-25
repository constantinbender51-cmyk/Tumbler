#!/usr/bin/env python3
"""
tumbler.py - SMA 365 BTC Trading Strategy
Uses 365-day SMA with ATR-based stop loss
Trades daily at 00:01 UTC: LONG if price > SMA 365, otherwise SHORT
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple
import subprocess
import numpy as np
import pandas as pd

import kraken_futures as kf
import kraken_ohlc
import binance_ohlc

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
RUN_TRADE_NOW = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_FUTS_LC = "pf_xbtusd"
SYMBOL_OHLC_KRAKEN = "XBTUSD"
SYMBOL_OHLC_BINANCE = "BTCUSDT"
INTERVAL_KRAKEN = 1440
INTERVAL_BINANCE = "1d"
SMA_PERIOD = 365
ATR_PERIOD = 14
ATR_MULTIPLIER = 3.2
LEV = 1.5
STATE_FILE = Path("sma_state.json")
TEST_SIZE_BTC = 0.0001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("sma_strategy")


def calculate_sma_and_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 365-day SMA and 14-day ATR"""
    df = df.copy()
    
    # Calculate 365-day SMA
    df['sma_365'] = df['close'].rolling(window=SMA_PERIOD).mean()
    
    # Calculate True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Calculate 14-day ATR
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    
    return df


def generate_signal(df: pd.DataFrame, current_price: float) -> Tuple[str, float, float]:
    """
    Generate trading signal based on SMA 365
    Returns: (signal, sma_365, atr)
    """
    df_calc = calculate_sma_and_atr(df)
    
    # Get latest values
    sma_365 = df_calc['sma_365'].iloc[-1]
    atr = df_calc['atr'].iloc[-1]
    
    # Check if we have valid values
    if pd.isna(sma_365) or pd.isna(atr):
        raise ValueError("Not enough historical data for SMA 365 or ATR calculation")
    
    # Generate signal
    if current_price > sma_365:
        signal = "LONG"
    else:
        signal = "SHORT"
    
    log.info(f"Current price: ${current_price:.2f}")
    log.info(f"SMA 365: ${sma_365:.2f}")
    log.info(f"ATR (14-day): ${atr:.2f}")
    log.info(f"Signal: {signal}")
    
    return signal, sma_365, atr


def portfolio_usd(api: kf.KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])


def mark_price(api: kf.KrakenFuturesApi) -> float:
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC:
            return float(t["markPrice"])
    raise RuntimeError("Mark-price for PF_XBTUSD not found")


def cancel_all(api: kf.KrakenFuturesApi):
    log.info("Cancelling all orders")
    try:
        api.cancel_all_orders()
    except Exception as e:
        log.warning("cancel_all_orders failed: %s", e)


def flatten_position(api: kf.KrakenFuturesApi):
    pos = api.get_open_positions()
    for p in pos.get("openPositions", []):
        if p["symbol"] != SYMBOL_FUTS_UC:
            continue
        side = "sell" if p["side"] == "long" else "buy"
        size = abs(float(p["size"]))
        log.info("Flatten %s position %.4f BTC", p["side"], size)
        api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size, 4),
        })


def place_stop(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float, atr: float):
    """Place ATR-based stop loss"""
    stop_distance = ATR_MULTIPLIER * atr
    
    if side == "buy":
        stop_price = fill_price - stop_distance
        stop_side = "sell"
        limit_price = stop_price * 0.9999
    else:
        stop_price = fill_price + stop_distance
        stop_side = "buy"
        limit_price = stop_price * 1.0001
    
    log.info(f"Placing ATR stop: {stop_side} at ${stop_price:.2f} (distance: ${stop_distance:.2f})")
    
    api.send_order({
        "orderType": "stp",
        "symbol": SYMBOL_FUTS_LC,
        "side": stop_side,
        "size": round(size_btc, 4),
        "stopPrice": int(round(stop_price)),
        "limitPrice": int(round(limit_price)),
    })


def get_current_position(api: kf.KrakenFuturesApi) -> Dict:
    """Get current open position from Kraken"""
    try:
        pos = api.get_open_positions()
        for p in pos.get("openPositions", []):
            if p["symbol"] == SYMBOL_FUTS_UC:
                return {
                    "signal": "LONG" if p["side"] == "long" else "SHORT",
                    "side": p["side"],
                    "size_btc": abs(float(p["size"])),
                    "fill_price": float(p.get("fillPrice", 0)),
                }
        return None
    except Exception as e:
        log.warning(f"Failed to get position: {e}")
        return None


def smoke_test(api: kf.KrakenFuturesApi):
    """Run smoke test to verify API connectivity"""
    log.info("=== Smoke-test start ===")
    
    try:
        # Test portfolio access
        usd = portfolio_usd(api)
        log.info(f"Portfolio value: ${usd:.2f} USD")
        
        # Test market data
        mp = mark_price(api)
        log.info(f"BTC mark price: ${mp:.2f}")
        
        # Check open positions
        current_pos = get_current_position(api)
        if current_pos:
            log.info(f"Open position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC @ ${current_pos['fill_price']:.2f}")
        else:
            log.info("No open positions")
        
        # Test historical data
        df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
        log.info(f"Historical data: {len(df)} days available")
        
        if len(df) < SMA_PERIOD:
            log.warning(f"Only {len(df)} days available, need {SMA_PERIOD} for SMA calculation")
        
        log.info("=== Smoke-test complete ===")
        return True
    except Exception as e:
        log.error(f"Smoke test failed: {e}")
        return False


def load_state() -> Dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {
        "trades": [],
        "starting_capital": None,
        "performance": {},
        "current_position": None,
        "current_portfolio_value": 0,
        "strategy_info": {}
    }


def save_state(st: Dict):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def update_state_with_current_position(api: kf.KrakenFuturesApi):
    """Update state file with current position from Kraken"""
    state = load_state()
    
    # Get current position
    current_pos = get_current_position(api)
    portfolio_value = portfolio_usd(api)
    
    # Update state with current info
    state["current_position"] = current_pos
    state["current_portfolio_value"] = portfolio_value
    
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
        log.info(f"Initialized starting capital: ${portfolio_value:.2f}")
    
    # Calculate performance if we have starting capital
    if state["starting_capital"]:
        total_return = (portfolio_value - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": portfolio_value,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return,
            "total_trades": len(state.get("trades", [])),
        }
    
    # Ensure strategy_info exists
    if "strategy_info" not in state:
        state["strategy_info"] = {
            "sma_period": SMA_PERIOD,
            "atr_period": ATR_PERIOD,
            "atr_multiplier": ATR_MULTIPLIER,
            "leverage": LEV,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    save_state(state)
    log.info(f"Updated state with current position and portfolio value: ${portfolio_value:.2f}")
    
    if current_pos:
        log.info(f"Current position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC @ ${current_pos['fill_price']:.2f}")
    else:
        log.info("No current position")


def daily_trade(api: kf.KrakenFuturesApi):
    """Execute daily trading strategy"""
    state = load_state()
    
    # Get current market data
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    current_price = mark_price(api)
    portfolio_value = portfolio_usd(api)
    
    # Set starting capital on first run
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    
    # Generate signal
    signal, sma_365, atr = generate_signal(df, current_price)
    
    # Close existing position
    log.info("Closing any existing positions...")
    cancel_all(api)
    flatten_position(api)
    time.sleep(2)
    
    # Calculate position size
    collateral = portfolio_usd(api)  # Get fresh portfolio value after flatten
    notional = collateral * LEV
    size_btc = round(notional / current_price, 4)
    
    side = "buy" if signal == "LONG" else "sell"
    
    log.info(f"Opening {signal} position: {side} {size_btc} BTC at ~${current_price:.2f}")
    
    if dry:
        log.info(f"DRY-RUN: {signal} {size_btc} BTC at ${current_price:.2f}")
        fill_price = current_price
    else:
        ord = api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": size_btc,
        })
        fill_price = float(ord.get("price", current_price))
        
        # Place ATR-based stop loss
        place_stop(api, side, size_btc, fill_price, atr)
    
    # Record trade
    trade_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "side": side,
        "size_btc": size_btc,
        "fill_price": fill_price,
        "portfolio_value": collateral,
        "sma_365": sma_365,
        "atr": atr,
        "stop_distance": ATR_MULTIPLIER * atr,
    }
    
    state["trades"].append(trade_record)
    
    # Calculate performance
    if state["starting_capital"]:
        total_return = (collateral - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": collateral,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return,
            "total_trades": len(state["trades"]),
        }
    
    # Update strategy info
    state["strategy_info"] = {
        "sma_period": SMA_PERIOD,
        "atr_period": ATR_PERIOD,
        "atr_multiplier": ATR_MULTIPLIER,
        "leverage": LEV,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    save_state(state)
    log.info(f"Trade executed and logged. Portfolio: ${collateral:.2f}")


def wait_until_00_01_utc():
    """Wait until 00:01 UTC for daily execution"""
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    wait_sec = (next_run - now).total_seconds()
    log.info("Next run at 00:01 UTC (%s), sleeping %.0f s", next_run.strftime("%Y-%m-%d"), wait_sec)
    time.sleep(wait_sec)


def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key or not api_sec:
        log.error("Env vars KRAKEN_API_KEY / KRAKEN_API_SECRET missing")
        sys.exit(1)

    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    log.info("Initializing strategy and state...")
    
    # Run smoke test first
    if not smoke_test(api):
        log.error("Smoke test failed, exiting")
        sys.exit(1)
    
    # Update state with current position - this creates the state file
    update_state_with_current_position(api)
    
    log.info("State file initialized with current portfolio data")

    if RUN_TRADE_NOW:
        log.info("RUN_TRADE_NOW=true – executing trade now")
        try:
            daily_trade(api)
        except Exception as exc:
            log.exception("Immediate trade failed: %s", exc)

    log.info("Starting web dashboard on port %s", os.getenv("PORT", 8080))
    subprocess.Popen([sys.executable, "web_state.py"])

    while True:
        wait_until_00_01_utc()
        try:
            daily_trade(api)
        except KeyboardInterrupt:
            log.info("Interrupted")
            break
        except Exception as exc:
            log.exception("Daily trade failed: %s", exc)


if __name__ == "__main__":
    main()
