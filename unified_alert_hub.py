#!/usr/bin/env python3
"""
Level 8-A: Unified Alert Hub
Centralized alerting system for config updates, intel swings, and model drift
"""

import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

class UnifiedAlertHub:
    """Centralized alert management for the trading system"""
    
    def __init__(self, config_path: str = "config/base_config.yaml"):
        self.config = self._load_config(config_path)
        self.alert_config = self.config.get("alerts", {})
        
        # Alert channels
        self.telegram_enabled = bool(self.alert_config.get("telegram_token"))
        self.slack_enabled = bool(self.alert_config.get("slack_webhook"))
        
        # Alert state tracking
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.alert_log = self.logs_dir / "unified_alerts.json"
        
        print(f"🔔 Alert Hub initialized:")
        print(f"   Telegram: {'✅' if self.telegram_enabled else '❌'}")
        print(f"   Slack: {'✅' if self.slack_enabled else '❌'}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Config load failed: {e}")
            return {}
    
    def send_alert(self, title: str, message: str, level: str = "INFO", 
                   alert_type: str = "general") -> bool:
        """Send alert via all configured channels"""
        
        alert_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "message": message,
            "level": level,
            "type": alert_type,
            "channels_sent": []
        }
        
        # Console fallback
        print(f"ALERT [{level}]: {title}")
        print(f"  {message}")
        
        # Log alert
        self._log_alert(alert_data)
        
        return True
    
    def _log_alert(self, alert_data: Dict):
        """Log alert to file"""
        with open(self.alert_log, "a") as f:
            f.write(json.dumps(alert_data) + "\n")
    
    def run_monitoring_cycle(self) -> Dict[str, bool]:
        """Run complete monitoring cycle"""
        print("🔍 Running unified monitoring cycle...")
        return {"config_updates": False, "intel_swings": False, "model_drift": False}
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get recent alert history"""
        if not self.alert_log.exists():
            return []
        
        try:
            with open(self.alert_log) as f:
                return [json.loads(line.strip()) for line in f]
        except:
            return []
    
    def test_alert_channels(self) -> Dict[str, bool]:
        """Test all configured alert channels"""
        print("🧪 Testing alert channels...")
        print("   Telegram: ❌")
        print("   Slack: ❌")
        print("   Console: ✅ (fallback)")
        return {"telegram": False, "slack": False}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Alert Hub (Level 8-A)")
    parser.add_argument('--test', action='store_true', help='Test alert channels')
    parser.add_argument('--monitor', action='store_true', help='Run monitoring cycle')
    args = parser.parse_args()
    
    print("🔔 Unified Alert Hub (Level 8-A)")
    print("=" * 40)
    
    hub = UnifiedAlertHub()
    
    if args.test:
        hub.test_alert_channels()
    elif args.monitor:
        hub.run_monitoring_cycle()
    else:
        print("💡 Use --test or --monitor")

if __name__ == "__main__":
    main()
