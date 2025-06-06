# Trading Bot Project Structure

## Core Components
```
core/
├── trading_engine.py          # Basic trading execution
├── enhanced_trading_engine.py # Level 5 live data integration
├── parent_controller.py       # Basic parent analysis
├── enhanced_parent_controller.py # Level 5 enhanced P&L
└── config_loader.py          # Configuration management

modules/
├── ai/                       # Strategy scoring & ML
│   ├── deepseek_start.py     # Rule-based scoring (Level 1)
│   └── strategy_recommender.py # Config generation (Level 2)
├── intelligence/             # Market intelligence (Level 6)
│   ├── news_sentiment_scraper.py
│   ├── economic_calendar_ingest.py
│   ├── regime_detector.py
│   ├── intel_bundle_packager.py
│   └── intel_fetcher.py
├── market_data/              # Live data (Level 5)
│   ├── gateway.py
│   ├── stream_manager.py
│   └── failover_manager.py
├── ml/                       # Machine learning (Level 7)
│   ├── level7a_feature_engineering.py
│   ├── level7b_baseline_model.py
│   ├── level7c_live_inference.py
│   └── level7d_hybrid_ranking.py
├── alerts/                   # Level 8 alerts
│   └── unified_alert_hub.py
├── monitoring/               # Level 8 monitoring
│   ├── kpi_endpoint.py
│   └── stability_watch.py
└── sync/                     # Parent-child sync (Level 3)
    ├── report_uploader.py
    └── update_fetcher.py

config/                       # Configuration files
logs/                        # All log outputs
intel/                       # Intelligence data
data/                        # ML features & models
web/                         # Dashboard & UI
scripts/                     # Utility scripts
```

## Level 8 Components
- **8-A**: Unified Alert Hub (`modules/alerts/unified_alert_hub.py`)
- **8-B**: Live KPI Endpoint (`modules/monitoring/kpi_endpoint.py`)
- **8-C**: Mini Dashboard (`web/dashboard.html`)
- **8-D**: Stability Watch (`modules/monitoring/stability_watch.py`)
- **8-E**: 24h Burn-In Test (`scripts/burn_in_test.py`)

## Quick Start
```bash
# Test Level 8 components
python scripts/run_level8.py test_complete

# Run individual components
python scripts/run_level8.py alert_hub --args --test
python scripts/run_level8.py kpi_endpoint --args --port 8000
python scripts/run_level8.py stability_watch --args --check

# Start dashboard
python -m http.server 8080
# Then open: http://localhost:8080/web/dashboard.html
```
