# Distributed AI-Driven Forex & Multi-Asset Trading System

## Quick Start

1. **Initialize the project:**
   ```bash
   bash init_project.sh
   ```

2. **Configure your API keys:**
   Edit `config/base_config.yaml` and add your Alpaca API credentials.

3. **Run the trading bot:**
   ```bash
   # Activate virtual environment
   source .venv/bin/activate
   
   # Run child bot (trading)
   python run_trading_bot.py --mode child
   
   # Run parent bot (analysis)
   python run_trading_bot.py --mode parent
   ```

## Project Structure

- `core/` - Main trading engine and parent controller
- `modules/strategies/` - Trading strategies (Martingale, Breakout, Mean Reversion)
- `modules/risk_management/` - Risk management components
- `modules/reporting/` - Performance and risk reporting
- `modules/sync/` - Synchronization between parent and child bots
- `modules/ai/` - AI components and strategy recommendation
- `modules/intelligence/` - News sentiment and market intelligence
- `config/` - Configuration files
- `logs/` - Trade logs and data

## Configuration

Before running, update `config/base_config.yaml` with your actual Alpaca API credentials:

```yaml
alpaca:
  api_key: "YOUR_ACTUAL_ALPACA_KEY"
  api_secret: "YOUR_ACTUAL_ALPACA_SECRET"
  base_url: "https://paper-api.alpaca.markets"  # Use paper trading for testing
```

## Safety Notice

This is a demo/development system. Always test with paper trading first and implement proper risk management before using real funds.
