# Trading Bot Project Structure (Post-Cleanup)

## 📁 Core Directories

```
trading-bot/
├── core/                     # Core trading logic
│   ├── trading_engine.py
│   ├── enhanced_trading_engine.py
│   ├── parent_controller.py
│   ├── enhanced_parent_controller.py
│   └── config_loader.py
├── modules/                  # Modular components
│   ├── ai/                   # Strategy AI & recommendations
│   ├── alerts/               # Alert system
│   ├── intelligence/         # Market intelligence
│   ├── market_data/          # Data gateways & streaming
│   ├── ml/                   # Machine learning pipeline
│   ├── monitoring/           # System monitoring
│   └── sync/                 # Parent-child sync
├── config/                   # Configuration files
├── data/                     # Data storage
│   ├── features/            # ML features
│   ├── historical/          # Historical market data
│   └── cache/               # Cached data
├── logs/                     # System logs
├── models/                   # ML models
├── scripts/                  # Utility scripts
├── web/                      # Dashboard & UI
└── tests/                    # Test suites
```

## 🎯 Key Files

- `run_trading_bot.py` - Main entry point
- `stage3_complete.json` - Completion status
- `test_suite_consolidated.py` - Unified test runner
- `requirements.txt` - Dependencies
- `init_project.sh` - Project setup

## 🧪 Testing

Run tests with:
```bash
python test_suite_consolidated.py all     # All tests
python test_suite_consolidated.py level8  # Specific level
```

## 🏃‍♂️ Quick Start

```bash
# Setup
bash init_project.sh

# Configure
edit config/base_config.yaml

# Run child bot
python run_trading_bot.py --mode child

# Run parent analysis  
python run_trading_bot.py --mode parent
```

## 📈 Status

- ✅ Stage 2 Complete (Levels 0-4)
- ✅ Stage 3 Complete (Levels 5-8) 
- 🎯 Ready for production optimization
