#!/usr/bin/env python3
"""
Simple Configuration for XAU/USD Tick Trading Bot
"""

# Alpaca API Configuration
ALPACA_CONFIG = {
    'paper_trading': True,
    'api_key': 'PK43BTKX4DJCXAVB5BIS',
    'secret_key': 'LOulDBLtPY9H3z6TfXMMCzDPTtjBXjI59pxD2So5',
    'symbol': 'XAUUSD',
    'quantity': 0.1
}

# Trading Strategy Settings
STRATEGY_CONFIG = {
    'profit_target_ticks': 4,    # 4 ticks = $0.40 profit target
    'stop_loss_ticks': 2,        # 2 ticks = $0.20 stop loss
    'tick_size': 0.10,           # XAU/USD tick size
    'min_confidence': 0.65       # Minimum signal confidence
}

# Bot Settings
BOT_CONFIG = {
    'log_file': 'xauusd_trades.csv',
    'max_position_time': 60,     # Max 60 seconds in position
    'status_update_interval': 10, # Status updates every 10 seconds
}