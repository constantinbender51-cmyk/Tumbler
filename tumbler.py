#!/usr/bin/env python3
"""
tumbler.py - Dual SMA Strategy with Flat Regime Detection + III Dynamic Leverage
SMA 1 (40 days): Primary logic
SMA 2 (120 days): Hard trend filter
III-Based Leverage: 0.5x (choppy) / 4.5x (trending) / 2.45x (overextended)
Flat Regime: Pauses trading when III < 0.16, resumes when price enters 4.5% bands
Trades daily at 00:01 UTC with 2% SL + 16% TP
Uses CURRENT data for all live trading decisions
Initializes flat regime state on startup from historical data
FIXED: III calculation now matches app.py exactly (no window slicing)
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

# Strategy Parameters - OPTIMIZED FROM GENETIC ALGORITHM
SMA_PERIOD_1 = 32   # Primary logic SMA (optimized from 40)
SMA_PERIOD_2 = 114  # Filter SMA (optimized from 120)
STATIC_STOP_PCT = 0.043  # 4.3% static stop loss (optimized from 2%)
TAKE_PROFIT_PCT = 0.126  # 12.6% take profit (optimized from 16%)
LIMIT_OFFSET_PCT = 0.0002  # 0.02% offset for limit orders
STOP_WAIT_TIME = 600  # Wait 10 minutes

# III Parameters - OPTIMIZED
III_WINDOW = 27  # 27-day window for III calculation (optimized from 35)
III_T_LOW = 0.058  # Below this: 0.079x leverage (choppy)
III_T_HIGH = 0.259  # Above this: 3.868x leverage (overextended)
LEV_LOW = 0.079   # Choppy market - essentially flat
LEV_MID = 4.327   # Sweet spot trending (0.058-0.259) - MAXIMUM CONVICTION
LEV_HIGH = 3.868  # Overextended market (>0.259) - reduce slightly

# Flat Regime Parameters - OPTIMIZED
FLAT_REGIME_THRESHOLD = 0.356  # III below this triggers flat regime (optimized from 0.16)
BAND_WIDTH_PCT = 0.077  # 7.7% bandwidth for regime release (optimized from 4.5%)

STATE_FILE = Path("sma_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("dual_sma_strategy")


def calculate_iii(df: pd.DataFrame) -> float:
    """
    Calculate Inefficiency Index (III) over the last 35 days
    Uses EXACT same method as app.py - rolling operations on full dataframe
    III = net_direction / path_length
    Higher III = more efficient/trending
    Lower III = more choppy/inefficient
    """
    if len(df) < III_WINDOW + 1:
        return 0.0
    
    # Create a copy to avoid modifying original
    df_calc = df.copy()
    
    # Calculate log returns
    df_calc['log_ret'] = np.log(df_calc['close'] / df_calc['close'].shift(1))
    
    # Calculate III using rolling operations (EXACT match to app.py)
    w = III_WINDOW
    iii_series = (df_calc['log_ret'].rolling(w).sum().abs() / 
                  df_calc['log_ret'].abs().rolling(w).sum()).fillna(0)
    
    # Return the most recent III value
    return float(iii_series.iloc[-1])


def determine_leverage(iii: float) -> float:
    """
    Determine leverage based on III value - OPTIMIZED STRUCTURE
    - III < 0.058: 0.079x (choppy - stay out)
    - 0.058 ≤ III < 0.259: 4.327x (sweet spot - MAXIMUM CONVICTION)
    - III ≥ 0.259: 3.868x (overextended - reduce slightly)
    """
    if iii < III_T_LOW:
        return LEV_LOW  # 0.079x
    elif iii < III_T_HIGH:
        return LEV_MID  # 4.327x - THE MONEY ZONE
    else:
        return LEV_HIGH  # 3.868x


def calculate_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate SMA 1 (40) and SMA 2 (120)"""
    df = df.copy()
    df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    return df


def check_flat_regime_trigger(iii: float, current_flat_regime: bool) -> bool:
    """
    Check if we should enter flat regime
    Trigger: III < 0.16
    """
    if iii < FLAT_REGIME_THRESHOLD:
        if not current_flat_regime:
            log.info(f"ENTERING FLAT REGIME: III={iii:.4f} < {FLAT_REGIME_THRESHOLD}")
        return True
    return current_flat_regime


def check_flat_regime_release(df: pd.DataFrame, current_flat_regime: bool) -> bool:
    """
    Check if we should exit flat regime
    Release: Price enters 4.5% band around EITHER SMA (using CURRENT data)
    """
    if not current_flat_regime:
        return False
    
    df_calc = calculate_smas(df)
    
    # Use CURRENT close and CURRENT SMAs to check bands
    current_close = df_calc['close'].iloc[-1]
    current_sma_1 = df_calc['sma_1'].iloc[-1]
    current_sma_2 = df_calc['sma_2'].iloc[-1]
    
    if pd.isna(current_sma_1) or pd.isna(current_sma_2):
        return True  # Release if SMAs not ready
    
    # Check if price is within BAND_WIDTH_PCT of EITHER SMA
    diff_sma1 = abs(current_close - current_sma_1)
    diff_sma2 = abs(current_close - current_sma_2)
    
    thresh_sma1 = current_sma_1 * BAND_WIDTH_PCT
    thresh_sma2 = current_sma_2 * BAND_WIDTH_PCT
    
    in_band_1 = diff_sma1 <= thresh_sma1
    in_band_2 = diff_sma2 <= thresh_sma2
    
    if in_band_1 or in_band_2:
        log.info(f"RELEASING FLAT REGIME: Price ${current_close:.2f} entered band")
        log.info(f"  Current SMA1: ${current_sma_1:.2f} (band: ±${thresh_sma1:.2f}), diff: ${diff_sma1:.2f}")
        log.info(f"  Current SMA2: ${current_sma_2:.2f} (band: ±${thresh_sma2:.2f}), diff: ${diff_sma2:.2f}")
        return False
    
    return True


def generate_signal(df: pd.DataFrame, current_price: float, is_flat_regime: bool) -> Tuple[str, float, float]:
    """
    Generate trading signal using dual SMA strategy with flat regime override
    Uses CURRENT data for live trading
    
    Returns: (signal, sma_1, sma_2)
    
    Logic:
    - If in flat regime: FLAT (no position)
    - Otherwise:
      - LONG: current_price > SMA1 AND current_price > SMA2
      - SHORT: current_price < SMA1 AND current_price < SMA2
      - FLAT: contradictory signals
    """
    df_calc = calculate_smas(df)
    
    # Get CURRENT values for live trading
    current_sma_1 = df_calc['sma_1'].iloc[-1]
    current_sma_2 = df_calc['sma_2'].iloc[-1]
    
    # Check if we have valid values
    if pd.isna(current_sma_1) or pd.isna(current_sma_2):
        raise ValueError(f"Not enough historical data for SMA {SMA_PERIOD_1} or SMA {SMA_PERIOD_2}")
    
    # FLAT REGIME OVERRIDE
    if is_flat_regime:
        log.info("FLAT REGIME ACTIVE: Forcing FLAT signal (no position)")
        return "FLAT", current_sma_1, current_sma_2
    
    # Generate signal based on CURRENT price vs CURRENT SMAs
    signal = "FLAT"
    
    if current_price > current_sma_1 and current_price > current_sma_2:
        signal = "LONG"
        log.info("LONG: current_price above both SMAs")
    elif current_price < current_sma_1 and current_price < current_sma_2:
        signal = "SHORT"
        log.info("SHORT: current_price below both SMAs")
    else:
        log.info("FLAT: contradictory SMA signals")
    
    log.info(f"Current price: ${current_price:.2f}")
    log.info(f"Current SMA 1 (40): ${current_sma_1:.2f}")
    log.info(f"Current SMA 2 (120): ${current_sma_2:.2f}")
    log.info(f"Final signal: {signal}")
    
    return signal, current_sma_1, current_sma_2


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
        limit_price = current_price * (1 + LIMIT_OFFSET_PCT)
    else:
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
    if side == "buy":
        limit_price = current_price * (1 - LIMIT_OFFSET_PCT)
    else:
        limit_price = current_price * (1 + LIMIT_OFFSET_PCT)
    
    log.info(f"Entry limit: {side} {size_btc:.4f} BTC at ${limit_price:.2f} (market: ${current_price:.2f})")
    
    try:
        api.send_order({
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
    """Place market order for any remaining unfilled amount"""
    pos = get_current_position(api)
    
    if pos and pos["side"] == ("long" if side == "buy" else "short"):
        filled_size = pos["size_btc"]
        log.info(f"Limit order filled {filled_size:.4f} BTC of {intended_size:.4f} BTC")
        
        remaining = intended_size - filled_size
        if remaining > 0.0001:
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


def place_take_profit(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float):
    """Place 16% take profit limit order"""
    if side == "buy":
        tp_price = fill_price * (1 + TAKE_PROFIT_PCT)
        tp_side = "sell"
    else:
        tp_price = fill_price * (1 - TAKE_PROFIT_PCT)
        tp_side = "buy"
    
    log.info(f"Placing 16% take profit: {tp_side} at ${tp_price:.2f}")
    
    try:
        api.send_order({
            "orderType": "lmt",
            "symbol": SYMBOL_FUTS_LC,
            "side": tp_side,
            "size": round(size_btc, 4),
            "limitPrice": int(round(tp_price)),
        })
    except Exception as e:
        log.error(f"Take profit order failed: {e}")


def initialize_flat_regime_state(api: kf.KrakenFuturesApi) -> bool:
    """
    Initialize flat regime state based on historical data
    Uses EXACT same III calculation as app.py
    Returns True if currently in flat regime, False otherwise
    """
    try:
        df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
        
        # Calculate III using rolling operations (matching app.py)
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        w = III_WINDOW
        df['iii'] = (df['log_ret'].rolling(w).sum().abs() / 
                     df['log_ret'].abs().rolling(w).sum()).fillna(0)
        
        # Get current III (most recent value)
        current_iii = df['iii'].iloc[-1]
        
        log.info(f"=== INITIALIZING FLAT REGIME STATE ===")
        log.info(f"Current III: {current_iii:.4f}")
        log.info(f"Flat regime threshold: {FLAT_REGIME_THRESHOLD}")
        
        # Check if we should be in flat regime
        if current_iii < FLAT_REGIME_THRESHOLD:
            log.info(f"III {current_iii:.4f} < {FLAT_REGIME_THRESHOLD} - Checking historical band breaches...")
            
            # Find when III last dropped below threshold
            below_threshold = df['iii'] < FLAT_REGIME_THRESHOLD
            last_trigger_idx = None
            
            # Search backwards to find the most recent trigger
            for i in range(len(df) - 1, -1, -1):
                if below_threshold.iloc[i] and (i == 0 or df['iii'].iloc[i-1] >= FLAT_REGIME_THRESHOLD):
                    last_trigger_idx = i
                    break
            
            if last_trigger_idx is None:
                log.info("Could not find trigger point - assuming FLAT REGIME")
                return True
            
            log.info(f"Last flat regime trigger at index {last_trigger_idx} ({df.index[last_trigger_idx]})")
            
            # Check if bands were breached since trigger
            df_calc = calculate_smas(df)
            
            for i in range(last_trigger_idx, len(df)):
                close = df_calc['close'].iloc[i]
                sma_1 = df_calc['sma_1'].iloc[i]
                sma_2 = df_calc['sma_2'].iloc[i]
                
                if pd.isna(sma_1) or pd.isna(sma_2):
                    continue
                
                diff_sma1 = abs(close - sma_1)
                diff_sma2 = abs(close - sma_2)
                thresh_sma1 = sma_1 * BAND_WIDTH_PCT
                thresh_sma2 = sma_2 * BAND_WIDTH_PCT
                
                if diff_sma1 <= thresh_sma1 or diff_sma2 <= thresh_sma2:
                    log.info(f"Band breach found at index {i} ({df.index[i]}) - RELEASED")
                    return False
            
            log.info("No band breach found since trigger - FLAT REGIME ACTIVE")
            return True
        else:
            log.info(f"III {current_iii:.4f} >= {FLAT_REGIME_THRESHOLD} - Normal trading active")
            return False
            
    except Exception as e:
        log.error(f"Error initializing flat regime state: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


def wait_for_flat_regime_release(api: kf.KrakenFuturesApi):
    """
    Wait until flat regime release condition is met
    Checks every hour until price enters bands around SMAs
    III recovery does NOT release - only band breach releases
    """
    log.info("=== WAITING FOR FLAT REGIME RELEASE ===")
    log.info("Checking every hour until price enters 4.5% bands around SMAs")
    log.info("NOTE: III recovery does NOT release flat regime - only band breach")
    
    while True:
        try:
            df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
            
            # Calculate current III for logging only
            df_calc_iii = df.copy()
            df_calc_iii['log_ret'] = np.log(df_calc_iii['close'] / df_calc_iii['close'].shift(1))
            w = III_WINDOW
            df_calc_iii['iii'] = (df_calc_iii['log_ret'].rolling(w).sum().abs() / 
                                  df_calc_iii['log_ret'].abs().rolling(w).sum()).fillna(0)
            iii = df_calc_iii['iii'].iloc[-1]
            
            # Check if price entered bands
            df_calc = calculate_smas(df)
            current_close = df_calc['close'].iloc[-1]
            current_sma_1 = df_calc['sma_1'].iloc[-1]
            current_sma_2 = df_calc['sma_2'].iloc[-1]
            current_price = mark_price(api)
            
            diff_sma1 = abs(current_close - current_sma_1)
            diff_sma2 = abs(current_close - current_sma_2)
            thresh_sma1 = current_sma_1 * BAND_WIDTH_PCT
            thresh_sma2 = current_sma_2 * BAND_WIDTH_PCT
            
            in_band_1 = diff_sma1 <= thresh_sma1
            in_band_2 = diff_sma2 <= thresh_sma2
            
            if in_band_1 or in_band_2:
                log.info(f"Price ${current_close:.2f} entered bands!")
                log.info("FLAT REGIME RELEASED - resuming normal trading")
                return
            
            # Still waiting
            log.info(f"Still in flat regime - III: {iii:.4f} (for info only), Price: ${current_price:.2f}")
            log.info(f"  SMA1: ${current_sma_1:.2f}, diff: ${diff_sma1:.2f}, need: ${thresh_sma1:.2f}")
            log.info(f"  SMA2: ${current_sma_2:.2f}, diff: ${diff_sma2:.2f}, need: ${thresh_sma2:.2f}")
            log.info("Next check in 1 hour...")
            
            time.sleep(3600)  # Wait 1 hour
            
        except Exception as e:
            log.error(f"Error checking flat regime release: {e}")
            time.sleep(3600)  # Wait 1 hour and retry


def smoke_test(api: kf.KrakenFuturesApi):
    """Run smoke test to verify API connectivity"""
    log.info("=== Smoke-test start ===")
    
    try:
        usd = portfolio_usd(api)
        log.info(f"Portfolio value: ${usd:.2f} USD")
        
        mp = mark_price(api)
        log.info(f"BTC mark price: ${mp:.2f}")
        
        current_pos = get_current_position(api)
        if current_pos:
            log.info(f"Open position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC")
        else:
            log.info("No open positions")
        
        df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
        log.info(f"Historical data: {len(df)} days available")
        
        if len(df) < SMA_PERIOD_2:
            log.warning(f"Only {len(df)} days available, need {SMA_PERIOD_2} for SMA calculation")
        
        iii = calculate_iii(df)
        leverage = determine_leverage(iii)
        log.info(f"III (35-day): {iii:.4f}")
        log.info(f"Determined leverage: {leverage}x")
        log.info(f"Flat regime threshold: {FLAT_REGIME_THRESHOLD}")
        log.info(f"Band width for release: {BAND_WIDTH_PCT:.1%}")
        
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
        "flat_regime_active": False
    }


def save_state(st: Dict):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def update_state_with_current_position(api: kf.KrakenFuturesApi):
    """Update state file with current position"""
    state = load_state()
    
    current_pos = get_current_position(api)
    portfolio_value = portfolio_usd(api)
    
    state["current_position"] = current_pos
    state["current_portfolio_value"] = portfolio_value
    
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
        log.info(f"Initialized starting capital: ${portfolio_value:.2f}")
    
    if state["starting_capital"]:
        total_return = (portfolio_value - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": portfolio_value,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return,
            "total_trades": len(state.get("trades", [])),
        }
    
    if "strategy_info" not in state:
        state["strategy_info"] = {
            "sma_period_1": SMA_PERIOD_1,
            "sma_period_2": SMA_PERIOD_2,
            "stop_loss_pct": STATIC_STOP_PCT * 100,
            "take_profit_pct": TAKE_PROFIT_PCT * 100,
            "iii_window": III_WINDOW,
            "leverage_tiers": f"{LEV_LOW}x / {LEV_MID}x / {LEV_HIGH}x",
            "flat_regime_threshold": FLAT_REGIME_THRESHOLD,
            "band_width_pct": BAND_WIDTH_PCT * 100,
            "order_type": "limit",
            "limit_offset_pct": LIMIT_OFFSET_PCT * 100,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    save_state(state)
    log.info(f"Updated state with current position and portfolio value: ${portfolio_value:.2f}")
    
    if current_pos:
        log.info(f"Current position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC")
    else:
        log.info("No current position")


def daily_trade(api: kf.KrakenFuturesApi):
    """Execute daily trading strategy with flat regime detection"""
    state = load_state()
    
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    current_price = mark_price(api)
    portfolio_value = portfolio_usd(api)
    
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    
    # Calculate III and leverage using CURRENT data
    iii = calculate_iii(df)
    leverage = determine_leverage(iii)
    
    log.info(f"=== III CALCULATION ===")
    log.info(f"III (35-day): {iii:.4f}")
    log.info(f"Thresholds: Low={III_T_LOW}, High={III_T_HIGH
