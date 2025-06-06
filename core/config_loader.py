import os
import yaml
import json
from datetime import datetime
from pathlib import Path

class ConfigLoader:
    def __init__(self, child_id=None):
        self.child_id = child_id or os.getenv("CHILD_ID", "default")
        self.base_config = None
        self.ai_config = None
        self.config_loaded = False
        
    def load_configs(self):
        """Load both base and AI-specific configurations"""
        try:
            # Load base config
            with open("config/base_config.yaml") as f:
                self.base_config = yaml.safe_load(f)
            
            # Load child-specific AI config
            ai_config_path = f"config/ai_trading_config_{self.child_id}.yaml"
            if Path(ai_config_path).exists():
                with open(ai_config_path) as f:
                    self.ai_config = yaml.safe_load(f)
                print(f"✅ Loaded AI config for child: {self.child_id}")
            else:
                print(f"⚠️  No AI config found for {self.child_id}, using defaults")
                self.ai_config = self._get_default_ai_config()
                
            self.config_loaded = True
            self._log_config_load()
            
            return self.base_config, self.ai_config
            
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            self.config_loaded = False
            raise
    
    def _get_default_ai_config(self):
        """Default AI configuration when child-specific config doesn't exist"""
        return {
            "ai_settings": {
                "model": "deepseek",
                "confidence_threshold": 0.7,
                "max_trades_per_day": 10
            },
            "strategy": {
                "default": "martingale",
                "risk_multiplier": 1.5
            },
            "risk": {
                "max_position_size": 1000,
                "max_daily_loss": 500,
                "risk_level": "medium"
            }
        }
    
    def _log_config_load(self):
        """Log successful config loading"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "child_id": self.child_id,
            "config_loaded": self.config_loaded,
            "ai_config_exists": self.ai_config is not None,
            "strategy": self.ai_config.get("strategy", {}).get("default", "unknown") if self.ai_config else "unknown"
        }
        
        os.makedirs("logs", exist_ok=True)
        with open("logs/config_loads.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_strategy_config(self):
        """Get current strategy configuration"""
        if not self.ai_config:
            return self._get_default_ai_config()["strategy"]
        return self.ai_config.get("strategy", {})
    
    def get_risk_config(self):
        """Get current risk management configuration"""
        if not self.ai_config:
            return self._get_default_ai_config()["risk"]
        return self.ai_config.get("risk", {})
    
    def get_ai_settings(self):
        """Get AI model settings"""
        if not self.ai_config:
            return self._get_default_ai_config()["ai_settings"]
        return self.ai_config.get("ai_settings", {})
