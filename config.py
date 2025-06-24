#!/usr/bin/env python3
"""
Aggressive Configuration for XAU/USD Tick Trading Bot
Enhanced for higher returns with increased risk parameters
"""

# Alpaca API Configuration
ALPACA_CONFIG = {
    'paper_trading': True,
    'api_key': 'PK43BTKX4DJCXAVB5BIS',
    'secret_key': 'LOulDBLtPY9H3z6TfXMMCzDPTtjBXjI59pxD2So5',
    'symbol': 'XAUUSD',
    'quantity': 0.3                  # INCREASED: From 0.1 to 0.3 oz (3x position size)
}

# Aggressive Trading Strategy Settings
STRATEGY_CONFIG = {
    'profit_target_ticks': 8,        # INCREASED: From 4 to 8 ticks = $0.80 profit target
    'stop_loss_ticks': 4,            # INCREASED: From 2 to 4 ticks = $0.40 stop loss
    'tick_size': 0.10,               # XAU/USD tick size
    'min_confidence': 0.70,          # INCREASED: From 0.65 to 0.75 (higher confidence required)
    
    # NEW: Advanced signal parameters
    'momentum_threshold': 0.0006,    # INCREASED: From 0.0005 to 0.0008 (stronger momentum required)
    'price_change_threshold': 0.035, # INCREASED: From 0.02 to 0.035 (larger price moves)
    'volatility_filter': 0.25,      # TIGHTENED: From 0.3 to 0.25 (less volatile conditions)
    'spread_filter': 0.12,          # TIGHTENED: From 0.15 to 0.12 (tighter spreads)
    
    # NEW: Aggressive signal enhancement
    'use_enhanced_signals': True,     # Enable enhanced signal logic
    'multi_timeframe_check': True,    # Check multiple timeframes for confirmation
    'volume_confirmation': True,      # Require volume confirmation
    'trend_alignment': True,          # Require trend alignment
}

# Enhanced Bot Settings
BOT_CONFIG = {
    'log_file': 'xauusd_aggressive_trades.csv',  # Separate log for aggressive strategy
    'max_position_time': 90,         # INCREASED: From 60 to 90 seconds (more time for larger moves)
    'status_update_interval': 8,     # DECREASED: More frequent updates for monitoring
    'max_daily_trades': 15,          # DECREASED: From 20 to 15 (quality over quantity)
    'max_daily_loss': 5.0,           # NEW: Maximum daily loss limit ($5.00)
    'max_consecutive_losses': 3,     # NEW: Stop after 3 consecutive losses
    'daily_profit_target': 15.0,     # NEW: Daily profit target ($15.00)
}

# NEW: Enhanced Risk Management
RISK_CONFIG = {
    'position_sizing': {
        'base_quantity': 0.3,         # Base position size
        'max_quantity': 0.5,          # Maximum position size
        'scale_on_confidence': True,   # Scale position based on signal confidence
        'confidence_multiplier': 1.5,  # Multiply base size by this when confidence > 0.85
    },
    
    'dynamic_stops': {
        'use_trailing_stop': True,     # Enable trailing stop loss
        'trailing_distance': 3,       # Trailing stop distance in ticks
        'breakeven_threshold': 4,     # Move to breakeven after 4 ticks profit
    },
    
    'session_limits': {
        'max_drawdown': 8.0,          # Maximum session drawdown ($8.00)
        'profit_protection': 0.7,     # Protect 70% of profits when daily target hit
        'aggressive_hours': [         # Most aggressive during these UTC hours
            (13, 17),  # London afternoon / NY morning overlap
            (20, 22),  # NY afternoon
        ]
    }
}

# NEW: Enhanced ML Configuration  
ML_CONFIG_AGGRESSIVE = {
    'lookback_ticks': 30,            # INCREASED: More data for better predictions
    'model_file': 'xauusd_aggressive_ml.pkl',
    'min_confidence': 0.78,          # INCREASED: Higher ML confidence threshold
    'retrain_interval': 25,          # More frequent retraining
    'feature_enhancement': True,     # Enable advanced feature engineering
    'ensemble_models': True,         # Use multiple models for predictions
}

# NEW: Performance Tracking Configuration
PERFORMANCE_CONFIG = {
    'detailed_analytics': True,      # Enable detailed performance analytics
    'real_time_metrics': [
        'sharpe_ratio',
        'max_drawdown', 
        'profit_factor',
        'avg_trade_duration',
        'win_streak',
        'loss_streak'
    ],
    'benchmark_tracking': True,      # Track against buy-and-hold benchmark
    'export_frequency': 'hourly',    # Export performance data hourly
}

# NEW: Alert Configuration
ALERT_CONFIG = {
    'enable_alerts': True,
    'alert_triggers': [
        'large_profit',     # Alert on profits > $2.00
        'large_loss',       # Alert on losses > $1.50  
        'daily_target',     # Alert when daily target reached
        'max_drawdown',     # Alert when approaching max drawdown
        'consecutive_losses' # Alert on consecutive loss streak
    ],
    'console_alerts': True,    # Print alerts to console
    'log_alerts': True,        # Log alerts to file
}

# Validation function for aggressive settings
def validate_aggressive_config():
    """Validate that aggressive configuration is properly set"""
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check position size increase
    if ALPACA_CONFIG['quantity'] <= 0.1:
        validation_results['warnings'].append("Position size not increased for aggressive strategy")
    
    # Check profit targets
    if STRATEGY_CONFIG['profit_target_ticks'] <= 4:
        validation_results['warnings'].append("Profit target not increased for aggressive strategy")
    
    # Check confidence threshold
    if STRATEGY_CONFIG['min_confidence'] < 0.7:
        validation_results['warnings'].append("Confidence threshold may be too low for aggressive strategy")
    
    # Risk ratio validation
    risk_ratio = STRATEGY_CONFIG['profit_target_ticks'] / STRATEGY_CONFIG['stop_loss_ticks']
    if risk_ratio < 1.8:
        validation_results['warnings'].append(f"Risk-reward ratio ({risk_ratio:.1f}) may be insufficient")
    
    # Daily limits validation
    max_loss = BOT_CONFIG['max_daily_loss']
    profit_target = BOT_CONFIG['daily_profit_target']
    if profit_target / max_loss < 2.0:
        validation_results['warnings'].append("Daily profit target vs max loss ratio is aggressive")
    
    return validation_results

# Strategy comparison for monitoring
STRATEGY_COMPARISON = {
    'conservative': {
        'quantity': 0.1,
        'profit_target': 4,
        'stop_loss': 2,
        'min_confidence': 0.65
    },
    'aggressive': {
        'quantity': ALPACA_CONFIG['quantity'],
        'profit_target': STRATEGY_CONFIG['profit_target_ticks'],
        'stop_loss': STRATEGY_CONFIG['stop_loss_ticks'], 
        'min_confidence': STRATEGY_CONFIG['min_confidence']
    }
}

if __name__ == "__main__":
    # Validate configuration when run directly
    print("🔧 Validating Aggressive Trading Configuration...")
    
    validation = validate_aggressive_config()
    
    print("\n📊 Configuration Summary:")
    print(f"   Position Size: {ALPACA_CONFIG['quantity']} oz (3x increase)")
    print(f"   Profit Target: {STRATEGY_CONFIG['profit_target_ticks']} ticks (${STRATEGY_CONFIG['profit_target_ticks'] * 0.1:.2f})")
    print(f"   Stop Loss: {STRATEGY_CONFIG['stop_loss_ticks']} ticks (${STRATEGY_CONFIG['stop_loss_ticks'] * 0.1:.2f})")
    print(f"   Risk-Reward: {STRATEGY_CONFIG['profit_target_ticks']/STRATEGY_CONFIG['stop_loss_ticks']:.1f}:1")
    print(f"   Confidence Threshold: {STRATEGY_CONFIG['min_confidence']:.2f}")
    print(f"   Daily Profit Target: ${BOT_CONFIG['daily_profit_target']:.2f}")
    print(f"   Daily Loss Limit: ${BOT_CONFIG['max_daily_loss']:.2f}")
    
    if validation['warnings']:
        print("\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    if validation['errors']:
        print("\n❌ Errors:")
        for error in validation['errors']:
            print(f"   - {error}")
    else:
        print("\n✅ Configuration validated successfully!")
        print("\n🚀 Ready for aggressive trading strategy!")
        print("\n⚠️  IMPORTANT: Test thoroughly in paper trading before going live!")