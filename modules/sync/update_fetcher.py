from pathlib import Path
import yaml

PARENT_CFG_DIR = Path("../parent_bot/config")
LOCAL_CFG = Path("config/ai_trading_config_trader_001.yaml")

def pull_updates():
    updates = PARENT_CFG_DIR / LOCAL_CFG.name
    if updates.exists():
        LOCAL_CFG.write_bytes(updates.read_bytes())
        print("[SYNC] Local AI config updated from parent.")

if __name__ == "__main__":
    pull_updates()
