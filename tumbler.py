#!/usr/bin/env python3
"""
tumbler.py - Dual SMA Strategy with State Machine
SMA 1 (57 days): Primary logic with proximity bands and cross detection
SMA 2 (124 days): Hard trend filter
Trades daily at 00:01 UTC with 2% stop loss and 3x leverage
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional
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

# Strategy Parameters (from optimization)
SMA_PERIOD_1 = 57   # Primary logic SMA
SMA_PERIOD_2 = 124  # Filter SMA
BAND_WIDTH = 0.05   # 5% proximity bands around SMA 1
STATIC_STOP_PCT = 0.02  # 2% static stop loss
LEV = 3  # 3x leverage
LIMIT_OFFSET_PCT = 0.0002  # 0.02% offset for limit orders
STOP_WAIT_TIME = 600  # Wait 10 minutes

STATE_FILE = Path("sma_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("dual_sma_strategy")


def calculate_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate SMA 1 (57) and SMA 2 (124)"""
    df = df.copy()
    df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    return df


def generate_signal(df: pd.DataFrame, current_price: float, prev_cross_flag: int) -> Tuple[str, float, float, int]:
    """
    Generate trading signal using dual SMA strategy with state machine
    
    Returns: (signal, sma_1, sma_2, new_cross_flag)
    
    State Machine Logic:
    - cross_flag = 0: No recent cross
    - cross_flag = 1: Just crossed UP through SMA 1
    - cross_flag = -1: Just crossed DOWN through SMA 1
    
    Signal Logic:
    - LONG: (price > upper_band) OR (price > SMA1 AND cross_flag=1)
    - SHORT: (price < lower_band) OR (price < SMA1 AND cross_flag=-1)
    - Filter: LONG requires price > SMA2, SHORT requires price < SMA2
    """
    df_calc = calculate_smas(df)
    
    # Get latest values
    sma_1 = df_calc['sma_1'].iloc[-1]
    sma_2 = df_calc['sma_2'].iloc[-1]
    
    # Get previous close for cross detection
    prev_close = df_calc['close'].iloc[-2]
    
    # Check if we have valid values
    if pd.isna(sma_1) or pd.isna(sma_2):
        raise ValueError(f"Not enough historical data for SMA {SMA_PERIOD_1} or SMA {SMA_PERIOD_2}")
    
    # Calculate proximity bands around SMA 1
    upper_band = sma_1 * (1 + BAND_WIDTH)
    lower_band = sma_1 * (1 - BAND_WIDTH)
    
    # Update cross flag based on previous close to current price
    cross_flag = prev_cross_flag
    
    # Detect crosses
    if prev_close < sma_1 and current_price > sma_1:
        cross_flag = 1  # Just crossed UP
        log.info("CROSS UP detected through SMA 1")
    elif prev_close > sma_1 and current_price < sma_1:
        cross_flag = -1  # Just crossed DOWN
        log.info("CROSS DOWN detected through SMA 1")
    
    # Reset flag if price exits bands
    if current_price > upper_band or current_price < lower_band:
        if cross_flag != 0:
            log.info("Price exited bands - resetting cross flag")
        cross_flag = 0
    
    # Generate base signal from SMA 1 logic
    signal = "FLAT"
    
    # LONG conditions
    if current_price > upper_band:
        signal = "LONG"
        log.info("LONG: Price above upper band")
    elif current_price > sma_1 and cross_flag == 1:
        signal = "LONG"
        log.info("LONG: Price above SMA1 with recent cross UP")
    # SHORT conditions
    elif current_price < lower_band:
        signal = "SHORT"
        log.info("SHORT: Price below lower band")
    elif current_price < sma_1 and cross_flag == -1:
        signal = "SHORT"
        log.info("SHORT: Price below SMA1 with recent cross DOWN")
    
    # Apply SMA 2 filter
    if signal == "LONG" and current_price < sma_2:
        log.info("LONG filtered out: price below SMA 2")
        signal = "FLAT"
    elif signal == "SHORT" and current_price > sma_2:
        log.info("SHORT filtered out: price above SMA 2")
        signal = "FLAT"
    
    log.info(f"Current price: ${current_price:.2f}")
    log.info(f"Previous close: ${prev_close:.2f}")
    log.info(f"SMA 1 (57): ${sma_1:.2f}")
    log.info(f"SMA 2 (124): ${sma_2:.2f}")
    log.info(f"Upper band: ${upper_band:.2f}")
    log.info(f"Lower band: ${lower_band:.2f}")
    log.info(f"Cross flag: {cross_flag}")
    log.info(f"Final signal: {signal}")
    
    return signal, sma_1, sma_2, cross_flag


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


def get_current_position(api: kf.KrakenFuturesApi) -> Optional[Dict]:
    """Get current open position from Kraken"""
    try:
        pos = api.get_open_positions()
        for p in pos.get("openPositions", []):
            if p["symbol"] == SYMBOL_FUTS_UC:
                return {
                    "signal": "LONG" if p["side"] == "long" else "SHORT",
                    "side": p["side"],
                    "size_btc": abs(float(p["size"])),
                }
        return None
    except Exception as e:
        log.warning(f"Failed to get position: {e}")
        return None


def flatten_position_limit(api: kf.KrakenFuturesApi, current_price: float):
    """Flatten position with limit order (0.02% in favorable direction)"""
    pos = get_current_position(api)
    if not pos:
        log.info("No position to flatten")
        return
    
    side = "sell" if pos["side"] == "long" else "buy"
    size = pos["size_btc"]
    
    # Calculate limit price: favorable direction + 0.02%
    if side == "sell":
        # Selling: place limit above market
        limit_price = current_price * (1 + LIMIT_OFFSET_PCT)
    else:
        # Buying: place limit below market
        limit_price = current_price * (1 - LIMIT_OFFSET_PCT)
    
    log.info(f"Flatten with limit: {side} {size:.4f} BTC at ${limit_price:.2f} (market: ${current_price:.2f})")
    
    try:
        api.send_order({
            "orderType": "lmt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size, 4),
            "limitPrice": int(round(limit_price)),
        })
    except Exception as e:
        log.warning(f"Flatten limit order failed: {e}")


def flatten_position_market(api: kf.KrakenFuturesApi):
    """Flatten any remaining position with market order"""
    pos = get_current_position(api)
    if not pos:
        log.info("No remaining position to flatten")
        return
    
    side = "sell" if pos["side"] == "long" else "buy"
    size = pos["size_btc"]
    
    log.info(f"Flatten remaining with market: {side} {size:.4f} BTC")
    
    try:
        api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size, 4),
        })
    except Exception as e:
        log.warning(f"Flatten market order failed: {e}")


def place_entry_limit(api: kf.KrakenFuturesApi, side: str, size_btc: float, current_price: float) -> float:
    """Place entry limit order (0.02% in favorable direction)"""
    # Calculate limit price: favorable direction + 0.02%
    if side == "buy":
        # Buying: place limit below market
        limit_price = current_price * (1 - LIMIT_OFFSET_PCT)
    else:
        # Selling: place limit above market
        limit_price = current_price * (1 + LIMIT_OFFSET_PCT)
    
    log.info(f"Entry limit: {side} {size_btc:.4f} BTC at ${limit_price:.2f} (market: ${current_price:.2f})")
    
    try:
        ord = api.send_order({
            "orderType": "lmt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size_btc, 4),
            "limitPrice": int(round(limit_price)),
        })
        return limit_price
    except Exception as e:
        log.error(f"Entry limit order failed: {e}")
        return current_price


def place_entry_market_remaining(api: kf.KrakenFuturesApi, side: str, intended_size: float, current_price: float) -> float:
    """Place market order for any remaining unfilled amount
    Returns: final_size"""
    pos = get_current_position(api)
    
    if pos and pos["side"] == ("long" if side == "buy" else "short"):
        filled_size = pos["size_btc"]
        log.info(f"Limit order filled {filled_size:.4f} BTC of {intended_size:.4f} BTC")
        
        remaining = intended_size - filled_size
        if remaining > 0.0001:  # Only if significant remaining
            log.info(f"Entry market for remaining: {side} {remaining:.4f} BTC")
            try:
                api.send_order({
                    "orderType": "mkt",
                    "symbol": SYMBOL_FUTS_LC,
                    "side": side,
                    "size": round(remaining, 4),
                })
                return intended_size
            except Exception as e:
                log.warning(f"Entry market order failed: {e}")
                return filled_size
        else:
            log.info("Limit order fully filled, no market order needed")
            return filled_size
    else:
        log.warning("No position found after limit order, placing full market order")
        try:
            api.send_order({
                "orderType": "mkt",
                "symbol": SYMBOL_FUTS_LC,
                "side": side,
                "size": round(intended_size, 4),
            })
            return intended_size
        except Exception as e:
            log.error(f"Full market order failed: {e}")
            return 0
    
    return 0


def place_stop(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float):
    """Place 2% static stop loss"""
    stop_distance = fill_price * STATIC_STOP_PCT
    
    if side == "buy":
        stop_price = fill_price - stop_distance
        stop_side = "sell"
        limit_price = stop_price * 0.9999
    else:
        stop_price = fill_price + stop_distance
        stop_side = "buy"
        limit_price = stop_price * 1.0001
    
    log.info(f"Placing 2% static stop: {stop_side} at ${stop_price:.2f} (distance: ${stop_distance:.2f})")
    
    try:
        api.send_order({
            "orderType": "stp",
            "symbol": SYMBOL_FUTS_LC,
            "side": stop_side,
            "size": round(size_btc, 4),
            "stopPrice": int(round(stop_price)),
            "limitPrice": int(round(limit_price)),
        })
    except Exception as e:
        log.error(f"Stop loss order failed: {e}")


def smoke_test(api: kf.KrakenFuturesApi):
    """Run smoke test to verify API connectivity and order placement"""
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
            log.info(f"Open position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC")
        else:
            log.info("No open positions")
        
        # Test historical data
        df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
        log.info(f"Historical data: {len(df)} days available")
        
        if len(df) < SMA_PERIOD_2:
            log.warning(f"Only {len(df)} days available, need {SMA_PERIOD_2} for SMA calculation")
        
        # === TEST ORDER PLACEMENT ===
        log.info("=== Testing Order Placement ===")
        
        # Calculate test order size (very small to minimize risk)
        test_collateral = usd
        test_notional = test_collateral * LEV
        test_size = test_notional / mp
        test_size_rounded = round(test_size, 4)
        
        log.info(f"Test order calculation:")
        log.info(f"  Collateral: ${test_collateral:.2f}")
        log.info(f"  Leverage: {LEV}x")
        log.info(f"  Notional: ${test_notional:.2f}")
        log.info(f"  Price: ${mp:.2f}")
        log.info(f"  Raw size: {test_size:.8f} BTC")
        log.info(f"  Rounded size: {test_size_rounded:.4f} BTC")
        
        # Test BUY limit order
        buy_limit_price = mp * (1 - LIMIT_OFFSET_PCT)
        buy_limit_price_int = int(round(buy_limit_price))
        
        log.info(f"Test BUY limit order:")
        log.info(f"  Side: buy")
        log.info(f"  Size (raw): {test_size_rounded}")
        log.info(f"  Size (rounded): {round(test_size_rounded, 4)}")
        log.info(f"  Limit price (raw): ${buy_limit_price:.2f}")
        log.info(f"  Limit price (int): ${buy_limit_price_int}")
        log.info(f"  Offset: {LIMIT_OFFSET_PCT * 100}% below market")
        
        try:
            test_order_params = {
                "orderType": "lmt",
                "symbol": SYMBOL_FUTS_LC,
                "side": "buy",
                "size": round(test_size_rounded, 4),
                "limitPrice": buy_limit_price_int,
            }
            log.info(f"Sending order with params: {test_order_params}")
            
            test_order = api.send_order(test_order_params)
            log.info(f"Order response: {test_order}")
            
            if test_order.get("sendStatus", {}).get("status") == "placed":
                order_id = test_order.get("sendStatus", {}).get("order_id")
                log.info(f"✓ Test BUY limit order placed successfully! Order ID: {order_id}")
                
                # Wait a moment then cancel it
                time.sleep(2)
                log.info(f"Cancelling test order {order_id}...")
                cancel_result = api.cancel_order({"order_id": order_id})
                log.info(f"Cancel result: {cancel_result}")
            else:
                log.error(f"✗ Test order failed: {test_order}")
                
        except Exception as e:
            log.error(f"✗ Test order exception: {e}")
        
        # Test SELL limit order
        sell_limit_price = mp * (1 + LIMIT_OFFSET_PCT)
        sell_limit_price_int = int(round(sell_limit_price))
        
        log.info(f"Test SELL limit order:")
        log.info(f"  Side: sell")
        log.info(f"  Size (raw): {test_size_rounded}")
        log.info(f"  Size (rounded): {round(test_size_rounded, 4)}")
        log.info(f"  Limit price (raw): ${sell_limit_price:.2f}")
        log.info(f"  Limit price (int): ${sell_limit_price_int}")
        log.info(f"  Offset: {LIMIT_OFFSET_PCT * 100}% above market")
        
        try:
            test_order_params = {
                "orderType": "lmt",
                "symbol": SYMBOL_FUTS_LC,
                "side": "sell",
                "size": round(test_size_rounded, 4),
                "limitPrice": sell_limit_price_int,
            }
            log.info(f"Sending order with params: {test_order_params}")
            
            test_order = api.send_order(test_order_params)
            log.info(f"Order response: {test_order}")
            
            if test_order.get("sendStatus", {}).get("status") == "placed":
                order_id = test_order.get("sendStatus", {}).get("order_id")
                log.info(f"✓ Test SELL limit order placed successfully! Order ID: {order_id}")
                
                # Wait a moment then cancel it
                time.sleep(2)
                log.info(f"Cancelling test order {order_id}...")
                cancel_result = api.cancel_order({"order_id": order_id})
                log.info(f"Cancel result: {cancel_result}")
            else:
                log.error(f"✗ Test order failed: {test_order}")
                
        except Exception as e:
            log.error(f"✗ Test order exception: {e}")
        
        # Test stop loss order format
        log.info(f"Test STOP LOSS order (not placing, just checking format):")
        stop_distance = mp * STATIC_STOP_PCT
        stop_price_buy = mp + stop_distance
        stop_limit_buy = stop_price_buy * 1.0001
        
        log.info(f"  For a SHORT position (buy stop):")
        log.info(f"    Entry price: ${mp:.2f}")
        log.info(f"    Stop distance: ${stop_distance:.2f} ({STATIC_STOP_PCT * 100}%)")
        log.info(f"    Stop price: ${stop_price_buy:.2f}")
        log.info(f"    Stop price (int): {int(round(stop_price_buy))}")
        log.info(f"    Limit price: ${stop_limit_buy:.2f}")
        log.info(f"    Limit price (int): {int(round(stop_limit_buy))}")
        
        log.info("=== Smoke-test complete ===")
        return True
    except Exception as e:
        log.error(f"Smoke test failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


def load_state() -> Dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {
        "trades": [],
        "starting_capital": None,
        "performance": {},
        "current_position": None,
        "current_portfolio_value": 0,
        "strategy_info": {},
        "cross_flag": 0  # State machine flag
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
            "sma_period_1": SMA_PERIOD_1,
            "sma_period_2": SMA_PERIOD_2,
            "band_width_pct": BAND_WIDTH * 100,
            "stop_loss_pct": STATIC_STOP_PCT * 100,
            "leverage": LEV,
            "order_type": "limit",
            "limit_offset_pct": LIMIT_OFFSET_PCT * 100,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    # Initialize cross_flag if not present
    if "cross_flag" not in state:
        state["cross_flag"] = 0
    
    save_state(state)
    log.info(f"Updated state with current position and portfolio value: ${portfolio_value:.2f}")
    
    if current_pos:
        log.info(f"Current position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC")
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
    
    # Get previous cross flag from state
    prev_cross_flag = state.get("cross_flag", 0)
    log.info(f"Previous cross flag: {prev_cross_flag}")
    
    # Generate signal with state machine
    signal, sma_1, sma_2, new_cross_flag = generate_signal(df, current_price, prev_cross_flag)
    
    # Update cross flag in state
    state["cross_flag"] = new_cross_flag
    
    # === STEP 1: Flatten with limit order ===
    log.info("=== STEP 1: Flatten with limit order ===")
    flatten_position_limit(api, current_price)
    
    # === STEP 2: Sleep 600 seconds ===
    log.info("=== STEP 2: Sleeping 600 seconds ===")
    time.sleep(600)
    
    # === STEP 3: Flatten remaining with market order ===
    log.info("=== STEP 3: Flatten remaining with market order ===")
    flatten_position_market(api)
    
    # === STEP 4: Cancel all orders ===
    log.info("=== STEP 4: Cancel all orders ===")
    cancel_all(api)
    time.sleep(2)
    
    # Get fresh portfolio value after flatten
    collateral = portfolio_usd(api)
    
    # Handle FLAT signal - stay out of market
    if signal == "FLAT":
        log.info("Signal is FLAT - staying out of market (no position)")
        
        # Record the decision to stay flat
        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": "FLAT",
            "side": "none",
            "size_btc": 0,
            "fill_price": current_price,
            "portfolio_value": collateral,
            "sma_1": sma_1,
            "sma_2": sma_2,
            "cross_flag": new_cross_flag,
            "iii": iii,
            "leverage": leverage,
            "stop_distance": 0,
            "tp_distance": 0,
            "note": "Stayed flat - no trade" if leverage == 0 else "Stayed flat due to signal logic"
        }
        
        state["trades"].append(trade_record)
        state["current_position"] = None
        
    else:
        # Calculate position size for LONG or SHORT with dynamic leverage
        notional = collateral * leverage
        size_btc = round(notional / current_price, 4)
        
        side = "buy" if signal == "LONG" else "sell"
        
        log.info(f"Opening {signal} position with {leverage}x leverage: {size_btc} BTC")
        
        if dry:
            log.info(f"DRY-RUN: {signal} {size_btc} BTC at ${current_price:.2f}")
            fill_price = current_price
            final_size = size_btc
        else:
            # === STEP 5: Place entry limit order ===
            log.info("=== STEP 5: Place entry limit order ===")
            limit_price = place_entry_limit(api, side, size_btc, current_price)
            
            # === STEP 6: Sleep 600 seconds ===
            log.info("=== STEP 6: Sleeping 600 seconds ===")
            time.sleep(600)
            
            # === STEP 7: Place entry market for remaining ===
            log.info("=== STEP 7: Place entry market for remaining ===")
            # Get fresh current price after 10 minutes
            current_price = mark_price(api)
            final_size = place_entry_market_remaining(api, side, size_btc, current_price)
            
            # Use current price as fill price for simplicity
            fill_price = current_price
            
            # === STEP 8: Cancel all orders ===
            log.info("=== STEP 8: Cancel all orders ===")
            cancel_all(api)
            time.sleep(2)
            
            log.info(f"Final position: {final_size:.4f} BTC @ ${fill_price:.2f} (current market price)")
            
            # === STEP 9: Place stop loss ===
            log.info("=== STEP 9: Place stop loss ===")
            place_stop(api, side, final_size, fill_price)
            
            # === STEP 10: Place take profit ===
            log.info("=== STEP 10: Place take profit ===")
            place_take_profit(api, side, final_size, fill_price)
        
        # Record trade
        stop_distance = fill_price * STATIC_STOP_PCT
        tp_distance = fill_price * TAKE_PROFIT_PCT
        
        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "side": side,
            "size_btc": final_size if not dry else size_btc,
            "fill_price": fill_price,
            "portfolio_value": collateral,
            "sma_1": sma_1,
            "sma_2": sma_2,
            "cross_flag": new_cross_flag,
            "iii": iii,
            "leverage": leverage,
            "stop_distance": stop_distance,
            "stop_loss_pct": STATIC_STOP_PCT * 100,
            "tp_distance": tp_distance,
            "take_profit_pct": TAKE_PROFIT_PCT * 100,
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
        "sma_period_1": SMA_PERIOD_1,
        "sma_period_2": SMA_PERIOD_2,
        "band_width_pct": BAND_WIDTH * 100,
        "stop_loss_pct": STATIC_STOP_PCT * 100,
        "take_profit_pct": TAKE_PROFIT_PCT * 100,
        "iii_window": III_WINDOW,
        "leverage_tiers": f"{LEV_LOW}x / {LEV_MID}x / {LEV_HIGH}x",
        "current_iii": iii,
        "current_leverage": leverage,
        "order_type": "limit",
        "limit_offset_pct": LIMIT_OFFSET_PCT * 100,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    save_state(state)
    log.info(f"Trade executed and logged. Portfolio: ${collateral:.2f}")
    log.info(f"New cross flag saved: {new_cross_flag}")
    log.info(f"III: {iii:.4f}, Leverage: {leverage}x")


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
    
    log.info("Initializing Dual SMA strategy with III dynamic leverage...")
    
    # Run smoke test first
    if not smoke_test(api):
        log.error("Smoke test failed, exiting")
        sys.exit(1)
    
    # Update state with current position - this creates the state file
    update_state_with_current_position(api)
    
    log.info("State file initialized with current portfolio data")
    
    # Ensure state file exists and is written
    if STATE_FILE.exists():
        log.info(f"State file confirmed at: {STATE_FILE.absolute()}")
    else:
        log.error("State file was not created!")

    if RUN_TRADE_NOW:
        log.info("RUN_TRADE_NOW=true – executing trade now")
        try:
            daily_trade(api)
        except Exception as exc:
            log.exception("Immediate trade failed: %s", exc)

    log.info("Starting web dashboard on port %s", os.getenv("PORT", 8080))
    time.sleep(1)  # Give state file time to be fully written
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

            log.info("Interrupted")
            break
        except Exception as exc:
            log.exception("Daily trade failed: %s", exc)


if __name__ == "__main__":
    main()
 with market order ===
    log.info("=== STEP 3: Flatten remaining with market order ===")
    flatten_position_market(api)
    
    # === STEP 4: Cancel all orders ===
    log.info("=== STEP 4: Cancel all orders ===")
    cancel_all(api)
    time.sleep(2)
    
    # Get fresh portfolio value after flatten
    collateral = portfolio_usd(api)
    
    # Handle FLAT signal - stay out of market
    if signal == "FLAT":
        log.info("Signal is FLAT - staying out of market (no position)")
        
        # Record the decision to stay flat
        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": "FLAT",
            "side": "none",
            "size_btc": 0,
            "fill_price": current_price,
            "portfolio_value": collateral,
            "sma_1": sma_1,
            "sma_2": sma_2,
            "cross_flag": new_cross_flag,
            "stop_distance": 0,
            "note": "Stayed flat due to signal logic"
        }
        
        state["trades"].append(trade_record)
        state["current_position"] = None
        
    else:
        # Calculate position size for LONG or SHORT
        notional = collateral * LEV
        size_btc = round(notional / current_price, 4)
        
        side = "buy" if signal == "LONG" else "sell"
        
        log.info(f"Opening {signal} position: {size_btc} BTC")
        
        if dry:
            log.info(f"DRY-RUN: {signal} {size_btc} BTC at ${current_price:.2f}")
            fill_price = current_price
            final_size = size_btc
        else:
            # === STEP 5: Place entry limit order ===
            log.info("=== STEP 5: Place entry limit order ===")
            limit_price = place_entry_limit(api, side, size_btc, current_price)
            
            # === STEP 6: Sleep 600 seconds ===
            log.info("=== STEP 6: Sleeping 600 seconds ===")
            time.sleep(600)
            
            # === STEP 7: Place entry market for remaining ===
            log.info("=== STEP 7: Place entry market for remaining ===")
            # Get fresh current price after 10 minutes
            current_price = mark_price(api)
            final_size = place_entry_market_remaining(api, side, size_btc, current_price)
            
            # Use current price as fill price for simplicity
            fill_price = current_price
            
            # === STEP 8: Cancel all orders ===
            log.info("=== STEP 8: Cancel all orders ===")
            cancel_all(api)
            time.sleep(2)
            
            log.info(f"Final position: {final_size:.4f} BTC @ ${fill_price:.2f} (current market price)")
            
            # === STEP 9: Place stop loss ===
            log.info("=== STEP 9: Place stop loss ===")
            place_stop(api, side, final_size, fill_price)
        
        # Record trade
        stop_distance = fill_price * STATIC_STOP_PCT
        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "side": side,
            "size_btc": final_size if not dry else size_btc,
            "fill_price": fill_price,
            "portfolio_value": collateral,
            "sma_1": sma_1,
            "sma_2": sma_2,
            "cross_flag": new_cross_flag,
            "stop_distance": stop_distance,
            "stop_loss_pct": STATIC_STOP_PCT * 100,
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
        "sma_period_1": SMA_PERIOD_1,
        "sma_period_2": SMA_PERIOD_2,
        "band_width_pct": BAND_WIDTH * 100,
        "stop_loss_pct": STATIC_STOP_PCT * 100,
        "leverage": LEV,
        "order_type": "limit",
        "limit_offset_pct": LIMIT_OFFSET_PCT * 100,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    save_state(state)
    log.info(f"Trade executed and logged. Portfolio: ${collateral:.2f}")
    log.info(f"New cross flag saved: {new_cross_flag}")


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
    
    log.info("Initializing Dual SMA strategy with state machine...")
    
    # Run smoke test first
    if not smoke_test(api):
        log.error("Smoke test failed, exiting")
        sys.exit(1)
    
    # Update state with current position - this creates the state file
    update_state_with_current_position(api)
    
    log.info("State file initialized with current portfolio data")
    
    # Ensure state file exists and is written
    if STATE_FILE.exists():
        log.info(f"State file confirmed at: {STATE_FILE.absolute()}")
    else:
        log.error("State file was not created!")

    if RUN_TRADE_NOW:
        log.info("RUN_TRADE_NOW=true – executing trade now")
        try:
            daily_trade(api)
        except Exception as exc:
            log.exception("Immediate trade failed: %s", exc)

    log.info("Starting web dashboard on port %s", os.getenv("PORT", 8080))
    time.sleep(1)  # Give state file time to be fully written
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
