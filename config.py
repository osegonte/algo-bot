#!/usr/bin/env python3
"""
Configuration file for Alpaca Trading Bot
"""

# =================================================================
# ALPACA API CONFIGURATION
# =================================================================
# To enable real Alpaca trading, get your API keys from:
# https://app.alpaca.markets/
# Then update the values below:

ALPACA_CONFIG = {
    # Set to False for live trading with real money
    'paper_trading': True,
    
    # Your Alpaca API credentials (keep these secure!)
    'api_key': 'PK43BTKX4DJCXAVB5BIS',  
    'secret_key': 'LOulDBLtPY9H3z6TfXMMCzDPTtjBXjI59pxD2So5',  # Add your Alpaca secret key here
    
    # Trading settings
    'symbol': 'XAUUSD',  # Stock symbol to trade
    'quantity': 0.1      # Number of shares per trade
}

# =================================================================
# TRADING STRATEGY CONFIGURATION  
# =================================================================
STRATEGY_CONFIG = {
    'tick_threshold': 0.01,  # 0.01% price movement to trigger entry
    'profit_target': 0.02,   # 0.02% profit target
    'stop_loss': 0.01,       # 0.01% stop loss
}

# =================================================================
# BOT SETTINGS
# =================================================================
BOT_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'trades.csv',
    'status_update_interval': 30,  # seconds
}
