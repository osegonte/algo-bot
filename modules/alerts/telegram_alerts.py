import os
import requests
import json
from datetime import datetime
from pathlib import Path

class AlertManager:
    def __init__(self, config_path="config/base_config.yaml"):
        self.config = self._load_config(config_path)
        self.alert_level = os.getenv("ALERT_LEVEL", "INFO")
        
    def _load_config(self, config_path):
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("alerts", {})
        except Exception as e:
            print(f"Warning: Could not load alert config: {e}")
            return {}
    
    def notify(self, message, level="INFO"):
        """Send alert via configured method"""
        formatted_msg = f"🤖 [{level}] {datetime.now().strftime('%H:%M:%S')}\n{message}"
        
        # For now, just print to console (Telegram setup is optional)
        print(f"ALERT: {formatted_msg}")
        return True

# Global instance for easy importing
alerts = AlertManager()

def notify(message, level="INFO"):
    return alerts.notify(message, level)

def notify_trade(symbol, side, qty, status="filled"):
    message = f"Trade executed: {side.upper()} {qty} {symbol} ({status})"
    return notify(message, "INFO")

def notify_error(error_msg):
    return notify(f"System error: {error_msg}", "ERROR")
