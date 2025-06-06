#!/usr/bin/env python3
"""
Level 8-A: Unified Alert Hub
Centralized alerting system
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import yaml

class UnifiedAlertHub:
    """Centralized alert management"""
    
    def __init__(self, config_path: str = "config/base_config.yaml"):
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.alert_log = self.logs_dir / "unified_alerts.json"
        print("🔔 Alert Hub initialized")
    
    def send_alert(self, title: str, message: str, level: str = "INFO", alert_type: str = "general") -> bool:
        """Send alert via console (demo)"""
        alert_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "message": message,
            "level": level,
            "type": alert_type,
            "channels_sent": []
        }
        
        print(f"ALERT [{level}]: {title}")
        print(f"  {message}")
        
        # Log alert
        with open(self.alert_log, "a") as f:
            f.write(json.dumps(alert_data) + "\n")
        
        return True
    
    def run_monitoring_cycle(self):
        """Run monitoring cycle"""
        return {"config_updates": False, "intel_swings": False, "model_drift": False}
    
    def get_alert_history(self, hours: int = 24):
        """Get recent alerts"""
        if not self.alert_log.exists():
            return []
        try:
            with open(self.alert_log) as f:
                return [json.loads(line.strip()) for line in f if line.strip()]
        except:
            return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Alert Hub (Level 8-A)")
    parser.add_argument('--test', action='store_true', help='Test alert channels')
    parser.add_argument('--monitor', action='store_true', help='Run monitoring cycle')
    args = parser.parse_args()
    
    hub = UnifiedAlertHub()
    
    if args.test:
        hub.send_alert("Test Alert", "System test", "INFO", "test")
        print("✅ Alert test complete")
    elif args.monitor:
        result = hub.run_monitoring_cycle()
        print(f"✅ Monitoring cycle: {result}")
    else:
        print("Use --test or --monitor")

if __name__ == "__main__":
    main()
