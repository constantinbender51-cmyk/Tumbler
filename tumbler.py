#!/usr/bin/env python3
"""
tumbler.py - Dual SMA Strategy with State Machine + III Dynamic Leverage
SMA 1 (57 days): Primary logic with proximity bands and cross detection
SMA 2 (124 days): Hard trend filter
III-Based Leverage: 0x (choppy) / 3x (trending) / 1.5x (overextended)
Trades daily at 00:01 UTC with 2% SL + 16% TP
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

# Strategy Parameters
SMA_PERIOD_1 = 57   # Primary logic SMA
SMA_PERIOD_2 = 124  # Filter SMA
BAND_WIDTH = 0.05   # 5% proximity bands around SMA 1
STATIC_STOP_PCT = 0.02  # 2% static stop loss
TAKE_PROFIT_PCT = 0.16  # 16% take profit
LIMIT_OFFSET_PCT = 0.0002  # 0.02% offset for limit orders
STOP_WAIT_TIME = 600  # Wait 10 minutes

# III Parameters
III_WINDOW = 35  # 35-day window for III calculation
III_T_LOW = 0.13  # Below this: 0.5x leverage (choppy)
III_T_HIGH = 0.18  # Above this: 2.45x leverage (overextended)
LEV_LOW = 0.5   # Choppy market
LEV_MID = 4.5   # Trending market
LEV_HIGH = 2.45  # Overextended market

STATE_FILE = Path("sma_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("dual_sma_strategy")


def calculate_iii(df: pd.DataFrame) -> float:
    """
    Calculate Inefficiency Index (III) over the last 35 days
    III = net_direction / path_length
    Higher III = more efficient/trending
    Lower III = more choppy/inefficient
    """
    if len(df) < III_WINDOW + 1:
        return 0.0
    
    # Get last 35 days + 1 for calculating returns
    recent_df = df.tail(III_WINDOW + 1).copy()
    
    # Calculate log returns
    recent_df['log_ret'] = np.log(recent_df['close'] / recent_df['close'].shift(1))
    
    # Remove NaN from first row
    log_rets = recent_df['log_ret'].dropna()
    
    if len(log_rets) < III_WINDOW:
        return 0.0
    
    # Net direction = sum of log returns (total movement)
    net_direction = abs(log_rets.sum())
    
    # Path length = sum of absolute log returns (total distance traveled)
    path_length = log_rets.abs().sum()
    
    # III calculation
    epsilon = 1e-8
    iii = net_direction / (path_length + epsilon)
    
    return iii


def determine_leverage(iii: float) -> float:
    """
    Determine leverage based on III value
    - III < 0.13: 0.5x (choppy)
    - 0.13 <= III < 0.18: 4.5x (trending nicely)
    - III >= 0.18: 2.45x (overextended, reduce risk)
    """
    if iii < III_T_LOW:
        return LEV_LOW  # 0.5x
    elif iii < III_T_HIGH:
        return LEV_MID  # 4.5x
    else:
        return LEV_HIGH  # 2.45x


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
        "cross_flag": 0
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
            "band_width_pct": BAND_WIDTH * 100,
            "stop_loss_pct": STATIC_STOP_PCT * 100,
            "take_profit_pct": TAKE_PROFIT_PCT * 100,
            "iii_window": III_WINDOW,
            "leverage_tiers": f"{LEV_LOW}x / {LEV_MID}x / {LEV_HIGH}x",
            "order_type": "limit",
            "limit_offset_pct": LIMIT_OFFSET_PCT * 100,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
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
    
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    current_price = mark_price(api)
    portfolio_value = portfolio_usd(api)
    
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    
    # Calculate III and leverage
    iii = calculate_iii(df)
    leverage = determine_leverage(iii)
    
    log.info(f"=== III CALCULATION ===")
    log.info(f"III (35-day): {iii:.4f}")
    log.info(f"Thresholds: Low={III_T_LOW}, High={III_T_HIGH}")
    log.info(f"Determined leverage: {leverage}x")
    
    if leverage < 1.0:
        log.info(f"Leverage is {leverage}x - choppy market, reduced position sizing")
    
    prev_cross_flag = state.get("cross_flag", 0)
    log.info(f"Previous cross flag: {prev_cross_flag}")
    
    signal, sma_1, sma_2, new_cross_flag = generate_signal(df, current_price, prev_cross_flag)
    
    # Note: We no longer override signal to FLAT based on leverage
    # Even with 0.5x leverage, we still take the signal
    
    state["cross_flag"] = new_cross_flag
    
    # Flatten
    log.info("=== STEP 1: Flatten with limit order ===")
    flatten_position_limit(api, current_price)
    
    log.info("=== STEP 2: Sleeping 600 seconds ===")
    time.sleep(600)
    
    log.info("=== STEP 3: Flatten remaining with market order ===")
    flatten_position_market(api)
    
    log.info("=== STEP 4: Cancel all orders ===")
    cancel_all(api)
    time.sleep(2)
    
    collateral = portfolio_usd(api)
    
    if signal == "FLAT":
        log.info("Signal is FLAT - staying out of market (no position)")
        
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
            "note": "Stayed flat due to signal logic"
        }
        
        state["trades"].append(trade_record)
        state["current_position"] = None
        
    else:
        notional = collateral * leverage
        size_btc = round(notional / current_price, 4)
        side = "buy" if signal == "LONG" else "sell"
        
        log.info(f"Opening {signal} position with {leverage}x leverage: {size_btc} BTC")
        
        if dry:
            log.info(f"DRY-RUN: {signal} {size_btc} BTC at ${current_price:.2f}")
            fill_price = current_price
            final_size = size_btc
        else:
            log.info("=== STEP 5: Place entry limit order ===")
            limit_price = place_entry_limit(api, side, size_btc, current_price)
            
            log.info("=== STEP 6: Sleeping 600 seconds ===")
            time.sleep(600)
            
            log.info("=== STEP 7: Place entry market for remaining ===")
            current_price = mark_price(api)
            final_size = place_entry_market_remaining(api, side, size_btc, current_price)
            fill_price = current_price
            
            log.info("=== STEP 8: Cancel all orders ===")
            cancel_all(api)
            time.sleep(2)
            
            log.info(f"Final position: {final_size:.4f} BTC @ ${fill_price:.2f}")
            
            log.info("=== STEP 9: Place stop loss ===")
            place_stop(api, side, final_size, fill_price)
            
            log.info("=== STEP 10: Place take profit ===")
            place_take_profit(api, side, final_size, fill_price)
        
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
    
    if state["starting_capital"]:
        total_return = (collateral - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": collateral,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return,
            "total_trades": len(state["trades"]),
        }
    
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
    
    if not smoke_test(api):
        log.error("Smoke test failed, exiting")
        sys.exit(1)
    
    update_state_with_current_position(api)
    
    log.info("State file initialized with current portfolio data")
    
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
    time.sleep(1)
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
