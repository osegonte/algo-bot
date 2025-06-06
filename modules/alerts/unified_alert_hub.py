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
import hashlib

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
        
        # Thresholds
        self.sentiment_sigma_threshold = 2.0  # 2 standard deviations
        self.drift_mae_threshold = 15.0  # MAE threshold for drift
        
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
        
        success = False
        
        # Format message with emoji based on level
        level_emojis = {
            "INFO": "ℹ️",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "SUCCESS": "✅",
            "DRIFT": "📊",
            "REGIME": "📈",
            "CONFIG": "⚙️"
        }
        
        emoji = level_emojis.get(level, "🔔")
        formatted_msg = f"{emoji} **{title}**\n{message}"
        
        # Send to Telegram
        if self.telegram_enabled:
            if self._send_telegram(formatted_msg):
                alert_data["channels_sent"].append("telegram")
                success = True
        
        # Send to Slack
        if self.slack_enabled:
            if self._send_slack(formatted_msg):
                alert_data["channels_sent"].append("slack")
                success = True
        
        # Log alert
        self._log_alert(alert_data)
        
        # Console fallback
        if not success:
            print(f"ALERT [{level}]: {title}")
            print(f"  {message}")
            success = True
        
        return success
    
    def _send_telegram(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            token = self.alert_config.get("telegram_token")
            chat_id = self.alert_config.get("telegram_chat_id")
            
            if not token or not chat_id:
                return False
            
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
            return False
    
    def _send_slack(self, message: str) -> bool:
        """Send message to Slack"""
        try:
            webhook_url = self.alert_config.get("slack_webhook")
            
            if not webhook_url or webhook_url == "YOUR_SLACK_WEBHOOK_URL":
                return False
            
            payload = {
                "text": message,
                "username": "Trading Bot",
                "icon_emoji": ":robot_face:"
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Slack send failed: {e}")
            return False
    
    def _log_alert(self, alert_data: Dict):
        """Log alert to file"""
        with open(self.alert_log, "a") as f:
            f.write(json.dumps(alert_data) + "\n")
    
    def check_config_updates(self) -> bool:
        """Check for recent config updates and alert"""
        try:
            config_log = self.logs_dir / "config_updates.json"
            
            if not config_log.exists():
                return False
            
            # Check last 5 minutes for updates
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
            
            with open(config_log) as f:
                lines = f.readlines()
            
            recent_updates = []
            for line in lines[-10:]:  # Check last 10 entries
                try:
                    entry = json.loads(line.strip())
                    entry_time = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                    
                    if entry_time > cutoff:
                        recent_updates.append(entry)
                except:
                    continue
            
            for update in recent_updates:
                updated_children = update.get("updated_children", [])
                strategies = update.get("strategies", {})
                
                if updated_children:
                    strategy_list = ", ".join([f"{child}: {strat}" for child, strat in strategies.items()])
                    
                    self.send_alert(
                        title="Config Update Detected",
                        message=f"Updated {len(updated_children)} child configs:\n{strategy_list}",
                        level="CONFIG",
                        alert_type="config_update"
                    )
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Config update check failed: {e}")
            return False
    
    def check_intel_swings(self) -> bool:
        """Check for significant intel swings and alert"""
        alert_sent = False
        
        # Check sentiment swings
        if self._check_sentiment_swing():
            alert_sent = True
        
        # Check regime flips
        if self._check_regime_flip():
            alert_sent = True
        
        return alert_sent
    
    def _check_sentiment_swing(self) -> bool:
        """Check for sentiment ≥ ±2 standard deviations"""
        try:
            intel_dir = Path("intel")
            sentiment_file = intel_dir / "news_sentiment.csv"
            
            if not sentiment_file.exists():
                return False
            
            # Load sentiment data
            df = pd.read_csv(sentiment_file)
            
            if len(df) < 10:  # Need minimum data
                return False
            
            sentiments = df['sentiment_compound'].astype(float)
            
            # Calculate statistics
            mean_sentiment = sentiments.mean()
            std_sentiment = sentiments.std()
            latest_sentiment = sentiments.iloc[-1]
            
            # Check for swing
            z_score = (latest_sentiment - mean_sentiment) / std_sentiment
            
            if abs(z_score) >= self.sentiment_sigma_threshold:
                direction = "📈 Positive" if z_score > 0 else "📉 Negative"
                
                self.send_alert(
                    title=f"Sentiment Swing Alert",
                    message=f"{direction} sentiment spike detected!\n"
                           f"Z-score: {z_score:.2f} (threshold: ±{self.sentiment_sigma_threshold})\n"
                           f"Current: {latest_sentiment:.3f} | Mean: {mean_sentiment:.3f}",
                    level="WARNING", 
                    alert_type="sentiment_swing"
                )
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Sentiment swing check failed: {e}")
            return False
    
    def _check_regime_flip(self) -> bool:
        """Check for recent regime changes"""
        try:
            intel_dir = Path("intel")
            regime_file = intel_dir / "market_regime.json"
            
            if not regime_file.exists():
                return False
            
            # Load current regime data
            with open(regime_file) as f:
                current_data = json.load(f)
            
            symbols = current_data.get("symbols", {})
            
            # Check if we have historical regime data to compare
            regime_history_file = self.logs_dir / "regime_history.json"
            
            if regime_history_file.exists():
                with open(regime_history_file) as f:
                    lines = f.readlines()
                    if lines:
                        previous_entry = json.loads(lines[-1].strip())
                        previous_symbols = previous_entry.get("symbols", {})
                        
                        # Check for regime flips
                        flips = []
                        for symbol, current_info in symbols.items():
                            current_regime = current_info.get("regime")
                            previous_regime = previous_symbols.get(symbol, {}).get("regime")
                            
                            if previous_regime and current_regime != previous_regime:
                                flips.append({
                                    "symbol": symbol,
                                    "from": previous_regime,
                                    "to": current_regime,
                                    "confidence": current_info.get("confidence", 0)
                                })
                        
                        if flips:
                            flip_descriptions = []
                            for flip in flips:
                                confidence_pct = flip["confidence"] * 100
                                flip_descriptions.append(f"{flip['symbol']}: {flip['from']} → {flip['to']} ({confidence_pct:.0f}%)")
                            
                            self.send_alert(
                                title="Market Regime Flip Detected",
                                message=f"Regime changes detected:\n" + "\n".join(flip_descriptions),
                                level="REGIME",
                                alert_type="regime_flip"
                            )
                            
                            # Log current state to history
                            self._log_regime_history(current_data)
                            return True
            
            # Always log current state for next comparison
            self._log_regime_history(current_data)
            return False
            
        except Exception as e:
            print(f"❌ Regime flip check failed: {e}")
            return False
    
    def _log_regime_history(self, regime_data: Dict):
        """Log regime data for historical comparison"""
        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": regime_data.get("symbols", {}),
            "summary": regime_data.get("summary", {})
        }
        
        regime_history_file = self.logs_dir / "regime_history.json"
        with open(regime_history_file, "a") as f:
            f.write(json.dumps(history_entry) + "\n")
    
    def check_model_drift(self) -> bool:
        """Check for model drift warnings"""
        try:
            metrics_file = self.logs_dir / "model_metrics.json"
            
            if not metrics_file.exists():
                return False
            
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            mae = metrics.get("mae_last_7_predictions", 0)
            drift_detected = metrics.get("model_drift", False)
            
            if drift_detected or mae > self.drift_mae_threshold:
                self.send_alert(
                    title="Model Drift Warning",
                    message=f"ML model performance degraded!\n"
                           f"7-day MAE: {mae:.2f} (threshold: {self.drift_mae_threshold})\n"
                           f"Recommendation: Schedule model retraining",
                    level="DRIFT",
                    alert_type="model_drift"
                )
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Model drift check failed: {e}")
            return False
    
    def run_monitoring_cycle(self) -> Dict[str, bool]:
        """Run complete monitoring cycle"""
        print("🔍 Running unified monitoring cycle...")
        
        results = {
            "config_updates": self.check_config_updates(),
            "intel_swings": self.check_intel_swings(), 
            "model_drift": self.check_model_drift()
        }
        
        alerts_sent = sum(results.values())
        print(f"📊 Monitoring complete: {alerts_sent} alerts sent")
        
        return results
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get recent alert history"""
        if not self.alert_log.exists():
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_alerts = []
        
        try:
            with open(self.alert_log) as f:
                for line in f:
                    alert = json.loads(line.strip())
                    alert_time = datetime.fromisoformat(alert["timestamp"].replace("Z", "+00:00"))
                    
                    if alert_time > cutoff:
                        recent_alerts.append(alert)
            
            return sorted(recent_alerts, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            print(f"❌ Error reading alert history: {e}")
            return []
    
    def test_alert_channels(self) -> Dict[str, bool]:
        """Test all configured alert channels"""
        print("🧪 Testing alert channels...")
        
        test_results = {}
        
        # Test message
        test_msg = f"🧪 Alert Hub Test\nTimestamp: {datetime.now().strftime('%H:%M:%S')}\nAll systems operational!"
        
        if self.telegram_enabled:
            test_results["telegram"] = self._send_telegram(test_msg)
            print(f"   Telegram: {'✅' if test_results['telegram'] else '❌'}")
        
        if self.slack_enabled:
            test_results["slack"] = self._send_slack(test_msg)
            print(f"   Slack: {'✅' if test_results['slack'] else '❌'}")
        
        # Always test console
        print(f"   Console: ✅ (fallback)")
        
        return test_results

def main():
    """Command line interface for alert hub"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Alert Hub (Level 8-A)")
    parser.add_argument('--test', action='store_true', help='Test alert channels')
    parser.add_argument('--monitor', action='store_true', help='Run monitoring cycle')
    parser.add_argument('--history', type=int, default=24, help='Show alert history (hours)')
    parser.add_argument('--daemon', action='store_true', help='Run as monitoring daemon')
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval (seconds)')
    
    args = parser.parse_args()
    
    print("🔔 Unified Alert Hub (Level 8-A)")
    print("=" * 40)
    
    hub = UnifiedAlertHub()
    
    if args.test:
        # Test alert channels
        results = hub.test_alert_channels()
        working_channels = sum(results.values())
        print(f"✅ {working_channels}/{len(results)} alert channels working")
    
    elif args.history > 0:
        # Show alert history
        alerts = hub.get_alert_history(args.history)
        print(f"\n📜 Alert History ({args.history}h):")
        
        if alerts:
            for alert in alerts[:10]:  # Show last 10
                timestamp = alert["timestamp"][:19].replace("T", " ")
                level = alert["level"]
                title = alert["title"]
                channels = ", ".join(alert.get("channels_sent", ["console"]))
                print(f"   {timestamp} [{level}] {title} → {channels}")
        else:
            print("   No recent alerts")
    
    elif args.daemon:
        # Run as monitoring daemon
        import time
        
        print(f"🔄 Starting monitoring daemon (interval: {args.interval}s)")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                results = hub.run_monitoring_cycle()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring daemon stopped")
    
    elif args.monitor:
        # Single monitoring cycle
        results = hub.run_monitoring_cycle()
        
        print(f"\n📊 Monitoring Results:")
        for check, alert_sent in results.items():
            status = "🔔 Alert sent" if alert_sent else "✅ OK"
            print(f"   {check}: {status}")
    
    else:
        print("💡 Use --test, --monitor, --history, or --daemon")
        print("   Examples:")
        print("   python level8a_unified_alerts.py --test")
        print("   python level8a_unified_alerts.py --monitor")
        print("   python level8a_unified_alerts.py --daemon --interval 300")

if __name__ == "__main__":
    main()