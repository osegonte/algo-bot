import random
import yaml
import json
from datetime import datetime
from pathlib import Path

class StrategyRecommender:
    def __init__(self):
        self.strategies = ["martingale", "breakout", "mean_reversion"]
        self.risk_levels = ["low", "medium", "high"]
        
    def recommend_strategy(self, market_conditions=None):
        """Original placeholder logic - now enhanced"""
        
        # Try to get recommendation from strategy scoring
        try:
            from modules.ai.deepseek_start import get_top_strategy
            recommended = get_top_strategy()
            if recommended in self.strategies:
                return recommended
        except ImportError:
            pass
        
        # Fallback to rule-based logic
        if market_conditions and market_conditions.get("volatility", 0) > 0.5:
            return "breakout"
        elif market_conditions and market_conditions.get("trend", 0) < -0.3:
            return "mean_reversion"
        else:
            return random.choice(self.strategies)
            
    def get_confidence(self):
        return random.uniform(0.6, 0.95)  # Placeholder confidence score
    
    def generate_config_for_child(self, child_id, strategy=None, market_conditions=None):
        """Level 2: Generate complete config file for a child bot"""
        
        # Get strategy recommendation
        if not strategy:
            strategy = self.recommend_strategy(market_conditions)
        
        confidence = self.get_confidence()
        
        # Strategy-specific configuration
        strategy_config = self._get_strategy_config(strategy, confidence)
        risk_config = self._get_risk_config(strategy, confidence)
        ai_config = self._get_ai_config(strategy, confidence)
        
        # Create complete configuration
        config = {
            "ai_settings": ai_config,
            "strategy": strategy_config,
            "risk": risk_config,
            "updated": datetime.utcnow().isoformat(),
            "generated_by": "strategy_recommender",
            "confidence": round(confidence, 3),
            "market_conditions": market_conditions or {}
        }
        
        # Write config file
        config_path = Path(f"config/ai_trading_config_{child_id}.yaml")
        self._write_config_file(config_path, config)
        
        print(f"📝 Generated config for {child_id}:")
        print(f"   Strategy: {strategy}")
        print(f"   Risk Level: {risk_config['risk_level']}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   File: {config_path}")
        
        return config
    
    def _get_strategy_config(self, strategy, confidence):
        """Get strategy-specific configuration"""
        base_config = {
            "default": strategy,
            "risk_multiplier": 1.5
        }
        
        if strategy == "breakout":
            base_config.update({
                "lookback_period": 20,
                "breakout_threshold": 0.02,
                "risk_multiplier": 2.0  # Higher risk for breakout
            })
        elif strategy == "mean_reversion":
            base_config.update({
                "window": 20,
                "std_threshold": 2,
                "risk_multiplier": 1.2  # Lower risk for mean reversion
            })
        elif strategy == "martingale":
            base_config.update({
                "initial_bet": 100,
                "multiplier": 2,
                "risk_multiplier": 0.8  # Much lower risk for martingale
            })
        
        # Adjust risk multiplier based on confidence
        if confidence < 0.7:
            base_config["risk_multiplier"] *= 0.8
        elif confidence > 0.9:
            base_config["risk_multiplier"] *= 1.2
            
        return base_config
    
    def _get_risk_config(self, strategy, confidence):
        """Get risk management configuration"""
        
        # Base risk settings
        if strategy == "martingale":
            # More conservative for martingale
            risk_config = {
                "max_position_size": 500,
                "max_daily_loss": 200,
                "risk_level": "low"
            }
        elif strategy == "breakout":
            # Moderate risk for breakout
            risk_config = {
                "max_position_size": 1500,
                "max_daily_loss": 750,
                "risk_level": "high"
            }
        else:  # mean_reversion
            # Balanced risk
            risk_config = {
                "max_position_size": 1000,
                "max_daily_loss": 500,
                "risk_level": "medium"
            }
        
        # Adjust based on confidence
        if confidence < 0.7:
            risk_config["max_position_size"] = int(risk_config["max_position_size"] * 0.7)
            risk_config["max_daily_loss"] = int(risk_config["max_daily_loss"] * 0.7)
            if risk_config["risk_level"] == "high":
                risk_config["risk_level"] = "medium"
            elif risk_config["risk_level"] == "medium":
                risk_config["risk_level"] = "low"
        
        return risk_config
    
    def _get_ai_config(self, strategy, confidence):
        """Get AI-specific configuration"""
        return {
            "model": "deepseek",
            "confidence_threshold": max(0.6, confidence - 0.1),
            "max_trades_per_day": 10 if confidence > 0.8 else 5,
            "strategy_confidence": confidence
        }
    
    def _write_config_file(self, config_path, config):
        """Write configuration to YAML file"""
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def update_all_child_configs(self, child_ids=None):
        """Level 2: Update configurations for multiple child bots"""
        
        if not child_ids:
            # Auto-discover existing child configs
            child_ids = self._discover_child_ids()
        
        if not child_ids:
            child_ids = ["trader_001"]  # Default
        
        print(f"\n🔄 Updating configs for {len(child_ids)} child bots...")
        
        updated_configs = {}
        for child_id in child_ids:
            try:
                # Could add child-specific market conditions here
                config = self.generate_config_for_child(child_id)
                updated_configs[child_id] = config
            except Exception as e:
                print(f"❌ Failed to update config for {child_id}: {e}")
        
        # Log the update
        self._log_config_updates(updated_configs)
        
        return updated_configs
    
    def _discover_child_ids(self):
        """Discover existing child bot IDs from config files"""
        config_dir = Path("config")
        if not config_dir.exists():
            return []
        
        child_ids = []
        for config_file in config_dir.glob("ai_trading_config_*.yaml"):
            # Extract child ID from filename
            name = config_file.stem
            if name.startswith("ai_trading_config_"):
                child_id = name.replace("ai_trading_config_", "")
                child_ids.append(child_id)
        
        return child_ids
    
    def _log_config_updates(self, updated_configs):
        """Log configuration updates"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "config_update",
            "updated_children": list(updated_configs.keys()),
            "strategies": {child_id: config["strategy"]["default"] 
                          for child_id, config in updated_configs.items()}
        }
        
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        
        with open(log_path / "config_updates.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# Convenience functions for Level 2
def generate_config_for_child(child_id, strategy=None):
    """Generate config for a specific child"""
    recommender = StrategyRecommender()
    return recommender.generate_config_for_child(child_id, strategy)

def update_all_configs():
    """Update all child configurations"""
    recommender = StrategyRecommender()
    return recommender.update_all_child_configs()

if __name__ == "__main__":
    # Level 2 demonstration
    recommender = StrategyRecommender()
    
    # Generate config for default child
    config = recommender.generate_config_for_child("trader_001")
    
    # Update all child configs
    updated = recommender.update_all_child_configs()
    
    print(f"\n✅ Level 2 complete: Generated {len(updated)} child configurations")