#!/usr/bin/env python3
"""
state_manager.py - Shared state management using Redis
Works across Railway services by using Redis as shared storage
"""

import json
import os
import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    log.warning("Redis not available, falling back to file storage")

# Configuration
REDIS_URL = os.getenv("REDIS_URL")
STATE_KEY = "tumbler:state"
USE_REDIS = REDIS_AVAILABLE and REDIS_URL

# Redis client (if available)
redis_client = None
if USE_REDIS:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        log.info("Redis connection established")
    except Exception as e:
        log.error(f"Failed to connect to Redis: {e}")
        USE_REDIS = False


def get_default_state() -> Dict:
    """Return default empty state"""
    return {
        "trades": [],
        "predictions": [],
        "starting_capital": None,
        "performance": {},
        "model_info": {},
        "current_position": None,
        "current_portfolio_value": 0
    }


def load_state() -> Dict:
    """Load state from Redis or fallback to default"""
    if USE_REDIS and redis_client:
        try:
            data = redis_client.get(STATE_KEY)
            if data:
                state = json.loads(data)
                log.info(f"State loaded from Redis: {len(state.get('trades', []))} trades")
                return state
            else:
                log.info("No state found in Redis, using default")
        except Exception as e:
            log.error(f"Error loading state from Redis: {e}")
    
    return get_default_state()


def save_state(state: Dict) -> bool:
    """Save state to Redis"""
    if USE_REDIS and redis_client:
        try:
            redis_client.set(STATE_KEY, json.dumps(state, indent=2))
            log.info(f"State saved to Redis: {len(state.get('trades', []))} trades")
            return True
        except Exception as e:
            log.error(f"Error saving state to Redis: {e}")
            return False
    else:
        log.warning("Redis not available, state not saved")
        return False


def get_state_info() -> Dict:
    """Get information about state storage for debugging"""
    info = {
        "storage_type": "redis" if USE_REDIS else "none",
        "redis_available": REDIS_AVAILABLE,
        "redis_url_set": bool(REDIS_URL),
        "redis_connected": False,
    }
    
    if USE_REDIS and redis_client:
        try:
            redis_client.ping()
            info["redis_connected"] = True
            
            # Get state size
            data = redis_client.get(STATE_KEY)
            if data:
                info["state_exists"] = True
                info["state_size_bytes"] = len(data)
                state = json.loads(data)
                info["trade_count"] = len(state.get("trades", []))
            else:
                info["state_exists"] = False
        except Exception as e:
            info["error"] = str(e)
    
    return info


# Backward compatibility functions for existing code
def load_state_compat() -> Dict:
    """Compatibility wrapper for load_state()"""
    return load_state()


def save_state_compat(st: Dict):
    """Compatibility wrapper for save_state()"""
    save_state(st)
