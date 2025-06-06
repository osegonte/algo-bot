import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from core.trading_engine import TradingEngine
from core.config_loader import ConfigLoader

# Import alerts (create __init__.py files first)
try:
    from modules.alerts.telegram_alerts import notify, notify_trade
except ImportError:
    # Fallback if alerts not set up yet
    def notify(msg, level="INFO"):
        print(f"[{level}] {msg}")
    def notify_trade(symbol, side, qty, status="filled"):
        print(f"Trade: {side} {qty} {symbol}")

def main():
    parser = argparse.ArgumentParser(description="Run Child or Parent Bot")
    parser.add_argument("--mode", choices=["child", "parent"], required=True)
    parser.add_argument("--child-id", help="Child bot identifier (env: CHILD_ID)")
    args = parser.parse_args()

    # Set child ID from argument or environment
    child_id = args.child_id or os.getenv("CHILD_ID", "trader_001")
    if args.mode == "child":
        os.environ["CHILD_ID"] = child_id

    if args.mode == "child":
        print(f"🚀 Starting Child Bot: {child_id}")
        print("=" * 50)
        
        # Load configurations
        config_loader = ConfigLoader(child_id)
        base_config, ai_config = config_loader.load_configs()
        
        # Initialize trading engine
        engine = TradingEngine(**base_config["alpaca"])
        
        # Get strategy settings
        strategy_config = config_loader.get_strategy_config()
        print(f"📈 Strategy: {strategy_config.get('default', 'martingale')}")
        print(f"🎯 Risk Level: {config_loader.get_risk_config().get('risk_level', 'medium')}")
        
        # Send startup alert
        notify(f"Child bot {child_id} started with strategy: {strategy_config.get('default')}")
        
        # Execute example trade
        try:
            order = engine.submit_order("AAPL", 1, "buy")
            notify_trade("AAPL", "buy", 1, order.status)
            print(f"✅ Trade executed: {order.symbol} {order.side} {order.qty}")
        except Exception as e:
            notify(f"Trade failed for {child_id}: {str(e)}", "ERROR")
            print(f"❌ Trade failed: {e}")

    elif args.mode == "parent":
        print("🧠 Starting Parent Controller...")
        print("=" * 50)
        
        from core.parent_controller import ParentController
        
        try:
            pc = ParentController()
            pc.ingest_logs()
            pc.basic_stats()
            
            # Send parent analysis alert
            notify("Parent analysis completed", "INFO")
            print("📊 Parent analysis complete!")
            
        except Exception as e:
            notify(f"Parent controller failed: {str(e)}", "ERROR")
            print(f"❌ Parent analysis failed: {e}")

if __name__ == "__main__":
    main()