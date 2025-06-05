import argparse
from core.trading_engine import TradingEngine
import yaml

parser = argparse.ArgumentParser(description="Run Child or Parent Bot")
parser.add_argument("--mode", choices=["child", "parent"], required=True)
args = parser.parse_args()

with open("config/base_config.yaml") as f:
    cfg = yaml.safe_load(f)

if args.mode == "child":
    engine = TradingEngine(**cfg["alpaca"])
    # example trade — remove once strategies are wired
    engine.submit_order("AAPL", 1, "buy")

elif args.mode == "parent":
    from core.parent_controller import ParentController
    pc = ParentController()
    pc.ingest_logs()
    pc.basic_stats()
