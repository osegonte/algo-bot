#!/usr/bin/env python3
"""
Level 8-D: Stability Watch
Monitors system latency and triggers alerts when performance degrades
"""

import json
import time
import statistics
import asyncio
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

class StabilityWatcher:
    """Monitor system performance and detect latency anomalies"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Directories
        self.logs_dir = Path("logs")
        self.intel_dir = Path("intel")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.latency_history = deque(maxlen=50)  # Keep last 50 measurements
        self.performance_log = self.logs_dir / "stability_watch.json"
        
        # Alert thresholds
        self.latency_multiplier = 2.0  # Alert when > 2x rolling median
        self.min_samples = 10  # Need minimum samples for analysis
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        
        # Last alert timestamps
        self.last_alerts = {}
        
        # API endpoints to monitor
        self.api_base = "http://127.0.0.1:8000"
        self.endpoints = ["/status", "/kpis", "/intel", "/health"]
        
        print(f"👁️ Stability Watch initialized")
        print(f"   Alert threshold: {self.latency_multiplier}x rolling median")
        print(f"   Monitoring {len(self.endpoints)} endpoints")
    
    async def measure_api_latency(self) -> Dict[str, float]:
        """Measure latency for all API endpoints"""
        latencies = {}
        
        for endpoint in self.endpoints:
            url = f"{self.api_base}{endpoint}"
            
            try:
                start_time = time.time()
                
                # Make async request
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: requests.get(url, timeout=10)
                )
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                latencies[endpoint] = latency_ms
                
            except Exception as e:
                # Record failed requests as very high latency
                latencies[endpoint] = 9999.0
                print(f"⚠️ Request failed {endpoint}: {e}")
        
        return latencies
    
    def measure_data_fetch_latency(self) -> Dict[str, float]:
        """Measure time to read local data files"""
        latencies = {}
        
        # Test files to measure
        test_files = {
            "strategy_scores": self.logs_dir / "strategy_scores.json",
            "news_sentiment": self.intel_dir / "news_sentiment.csv", 
            "market_regime": self.intel_dir / "market_regime.json",
            "parent_summary": self._find_latest_summary()
        }
        
        for name, file_path in test_files.items():
            if not file_path or not file_path.exists():
                latencies[name] = 0.0
                continue
            
            try:
                start_time = time.time()
                
                # Read file
                if file_path.suffix == '.json':
                    with open(file_path) as f:
                        json.load(f)
                elif file_path.suffix == '.csv':
                    import pandas as pd
                    pd.read_csv(file_path)
                else:
                    with open(file_path) as f:
                        f.read()
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                latencies[name] = latency_ms
                
            except Exception as e:
                latencies[name] = 999.0  # Failed read
                print(f"⚠️ Data read failed {name}: {e}")
        
        return latencies
    
    def measure_intel_processing_latency(self) -> float:
        """Measure time to process intelligence data"""
        try:
            start_time = time.time()
            
            # Simulate typical intel processing
            self._process_sample_intel()
            
            end_time = time.time()
            return (end_time - start_time) * 1000
            
        except Exception as e:
            print(f"⚠️ Intel processing failed: {e}")
            return 999.0
    
    def _process_sample_intel(self):
        """Sample intelligence processing operations"""
        # Load and analyze sentiment data
        sentiment_file = self.intel_dir / "news_sentiment.csv"
        if sentiment_file.exists():
            import pandas as pd
            df = pd.read_csv(sentiment_file)
            
            if len(df) > 0:
                # Calculate sentiment statistics
                sentiments = df['sentiment_compound'].astype(float)
                _ = sentiments.mean()
                _ = sentiments.std()
                _ = (sentiments > 0.1).sum()
        
        # Load regime data
        regime_file = self.intel_dir / "market_regime.json"
        if regime_file.exists():
            with open(regime_file) as f:
                regime_data = json.load(f)
                
                # Count regimes
                symbols = regime_data.get("symbols", {})
                for symbol_info in symbols.values():
                    _ = symbol_info.get("regime", "unknown")
    
    def _find_latest_summary(self) -> Optional[Path]:
        """Find the latest parent summary file"""
        summary_files = list(self.logs_dir.glob("parent_summary_*.json"))
        
        if not summary_files:
            return None
        
        return max(summary_files, key=lambda f: f.stat().st_mtime)
    
    async def run_stability_check(self) -> Dict[str, any]:
        """Run complete stability check and return results"""
        check_start = time.time()
        
        # Measure different components
        api_latencies = await self.measure_api_latency()
        data_latencies = self.measure_data_fetch_latency()
        intel_latency = self.measure_intel_processing_latency()
        
        # Combine all latencies
        all_latencies = {
            **{f"api{k}": v for k, v in api_latencies.items()},
            **{f"data_{k}": v for k, v in data_latencies.items()},
            "intel_processing": intel_latency
        }
        
        # Calculate aggregate metrics
        total_latency = sum(all_latencies.values())
        max_latency = max(all_latencies.values())
        avg_latency = total_latency / len(all_latencies)
        
        # Create performance record
        performance_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latencies": all_latencies,
            "total_latency_ms": total_latency,
            "max_latency_ms": max_latency,
            "avg_latency_ms": avg_latency,
            "check_duration_ms": (time.time() - check_start) * 1000
        }
        
        # Add to history
        self.latency_history.append({
            "timestamp": performance_record["timestamp"],
            "total_latency": total_latency,
            "max_latency": max_latency,
            "avg_latency": avg_latency
        })
        
        # Log performance
        self._log_performance(performance_record)
        
        # Check for anomalies
        alerts_triggered = self._check_latency_alerts(performance_record)
        
        performance_record["alerts_triggered"] = alerts_triggered
        
        return performance_record
    
    def _check_latency_alerts(self, record: Dict) -> List[Dict]:
        """Check for latency anomalies and trigger alerts"""
        alerts = []
        
        if len(self.latency_history) < self.min_samples:
            return alerts  # Not enough data yet
        
        # Calculate rolling statistics
        recent_totals = [h["total_latency"] for h in list(self.latency_history)[-20:]]
        recent_maxes = [h["max_latency"] for h in list(self.latency_history)[-20:]]
        
        median_total = statistics.median(recent_totals)
        median_max = statistics.median(recent_maxes)
        
        current_total = record["total_latency_ms"]
        current_max = record["max_latency_ms"]
        
        # Check total latency alert
        if current_total > median_total * self.latency_multiplier:
            alert = {
                "type": "total_latency_spike",
                "current_ms": current_total,
                "median_ms": median_total,
                "multiplier": current_total / median_total,
                "threshold": self.latency_multiplier,
                "timestamp": record["timestamp"]
            }
            
            if self._should_send_alert("total_latency"):
                alerts.append(alert)
                self._send_latency_alert(alert)
                self.last_alerts["total_latency"] = time.time()
        
        # Check max latency alert
        if current_max > median_max * self.latency_multiplier:
            alert = {
                "type": "max_latency_spike",
                "current_ms": current_max,
                "median_ms": median_max,
                "multiplier": current_max / median_max,
                "threshold": self.latency_multiplier,
                "timestamp": record["timestamp"]
            }
            
            if self._should_send_alert("max_latency"):
                alerts.append(alert)
                self._send_latency_alert(alert)
                self.last_alerts["max_latency"] = time.time()
        
        # Check for individual component alerts
        component_alerts = self._check_component_alerts(record)
        alerts.extend(component_alerts)
        
        return alerts
    
    def _check_component_alerts(self, record: Dict) -> List[Dict]:
        """Check individual components for latency spikes"""
        alerts = []
        latencies = record["latencies"]
        
        # Define thresholds for different components
        thresholds = {
            "api/status": 2000,  # 2 seconds
            "api/kpis": 1000,    # 1 second
            "data_news_sentiment": 500,  # 500ms
            "intel_processing": 1000     # 1 second
        }
        
        for component, latency in latencies.items():
            threshold = thresholds.get(component, 2000)  # Default 2s
            
            if latency > threshold:
                alert = {
                    "type": "component_latency_spike",
                    "component": component,
                    "current_ms": latency,
                    "threshold_ms": threshold,
                    "timestamp": record["timestamp"]
                }
                
                if self._should_send_alert(f"component_{component}"):
                    alerts.append(alert)
                    self._send_latency_alert(alert)
                    self.last_alerts[f"component_{component}"] = time.time()
        
        return alerts
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        last_alert_time = self.last_alerts.get(alert_type, 0)
        return time.time() - last_alert_time > self.alert_cooldown
    
    def _send_latency_alert(self, alert: Dict):
        """Send latency alert via unified alert hub"""
        try:
            # Import here to avoid circular dependency
            from level8a_unified_alerts import UnifiedAlertHub
            
            hub = UnifiedAlertHub()
            
            alert_type = alert["type"]
            
            if alert_type == "total_latency_spike":
                title = "System Latency Spike"
                message = (f"Total system latency spike detected!\n"
                          f"Current: {alert['current_ms']:.0f}ms\n"
                          f"Median: {alert['median_ms']:.0f}ms\n"
                          f"Multiplier: {alert['multiplier']:.1f}x (threshold: {alert['threshold']}x)")
                
            elif alert_type == "max_latency_spike":
                title = "Maximum Latency Spike"
                message = (f"Maximum component latency spike detected!\n"
                          f"Current: {alert['current_ms']:.0f}ms\n"
                          f"Median: {alert['median_ms']:.0f}ms\n"
                          f"Multiplier: {alert['multiplier']:.1f}x")
                
            elif alert_type == "component_latency_spike":
                title = f"Component Latency Alert"
                component = alert['component'].replace('_', ' ').title()
                message = (f"High latency detected in {component}\n"
                          f"Current: {alert['current_ms']:.0f}ms\n"
                          f"Threshold: {alert['threshold_ms']:.0f}ms")
            
            else:
                title = "Latency Alert"
                message = f"Latency anomaly detected: {alert}"
            
            hub.send_alert(
                title=title,
                message=message,
                level="WARNING",
                alert_type="latency_spike"
            )
            
            print(f"📨 Latency alert sent: {title}")
            
        except Exception as e:
            print(f"❌ Failed to send latency alert: {e}")
    
    def _log_performance(self, record: Dict):
        """Log performance record to file"""
        with open(self.performance_log, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, any]:
        """Get performance summary for the last N hours"""
        if not self.performance_log.exists():
            return {"error": "No performance data available"}
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        records = []
        
        try:
            with open(self.performance_log) as f:
                for line in f:
                    record = json.loads(line.strip())
                    record_time = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                    
                    if record_time > cutoff:
                        records.append(record)
            
            if not records:
                return {"error": "No recent performance data"}
            
            # Calculate summary statistics
            total_latencies = [r["total_latency_ms"] for r in records]
            max_latencies = [r["max_latency_ms"] for r in records]
            avg_latencies = [r["avg_latency_ms"] for r in records]
            
            alerts_count = sum(len(r.get("alerts_triggered", [])) for r in records)
            
            summary = {
                "timeframe_hours": hours,
                "total_checks": len(records),
                "latency_stats": {
                    "total": {
                        "min": min(total_latencies),
                        "max": max(total_latencies),
                        "avg": statistics.mean(total_latencies),
                        "median": statistics.median(total_latencies)
                    },
                    "max_component": {
                        "min": min(max_latencies),
                        "max": max(max_latencies),
                        "avg": statistics.mean(max_latencies),
                        "median": statistics.median(max_latencies)
                    }
                },
                "alerts_triggered": alerts_count,
                "latest_check": records[-1]["timestamp"],
                "performance_trend": self._calculate_trend(total_latencies)
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to analyze performance data: {e}"}
    
    def _calculate_trend(self, latencies: List[float]) -> str:
        """Calculate performance trend from latency data"""
        if len(latencies) < 5:
            return "insufficient_data"
        
        # Compare first and last halves
        first_half = latencies[:len(latencies)//2]
        second_half = latencies[len(latencies)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change_pct = ((second_avg - first_avg) / first_avg) * 100
        
        if change_pct > 20:
            return "degrading"
        elif change_pct < -20:
            return "improving"
        else:
            return "stable"
    
    async def run_monitoring_daemon(self, interval: int = 60):
        """Run continuous monitoring daemon"""
        print(f"🔄 Starting stability monitoring daemon (interval: {interval}s)")
        
        try:
            while True:
                print(f"⏱️ {datetime.now().strftime('%H:%M:%S')} - Running stability check...")
                
                result = await self.run_stability_check()
                
                alerts_count = len(result.get("alerts_triggered", []))
                total_latency = result["total_latency_ms"]
                
                if alerts_count > 0:
                    print(f"🚨 {alerts_count} latency alerts triggered")
                else:
                    print(f"✅ Stability check OK (total: {total_latency:.0f}ms)")
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stability monitoring stopped")
        except Exception as e:
            print(f"❌ Monitoring daemon error: {e}")

def main():
    """Command line interface for stability watch"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stability Watch (Level 8-D)")
    parser.add_argument('--check', action='store_true', help='Run single stability check')
    parser.add_argument('--daemon', action='store_true', help='Run monitoring daemon')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval (seconds)')
    parser.add_argument('--summary', type=int, default=24, help='Show performance summary (hours)')
    parser.add_argument('--threshold', type=float, default=2.0, help='Alert threshold multiplier')
    
    args = parser.parse_args()
    
    print("👁️ Stability Watch (Level 8-D)")
    print("=" * 40)
    
    watcher = StabilityWatcher()
    watcher.latency_multiplier = args.threshold
    
    if args.summary > 0:
        # Show performance summary
        summary = watcher.get_performance_summary(args.summary)
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
        else:
            print(f"\n📊 Performance Summary ({args.summary}h):")
            print(f"   Total checks: {summary['total_checks']}")
            print(f"   Alerts triggered: {summary['alerts_triggered']}")
            print(f"   Performance trend: {summary['performance_trend']}")
            
            total_stats = summary['latency_stats']['total']
            print(f"\n⏱️ Total Latency Stats:")
            print(f"   Min: {total_stats['min']:.0f}ms")
            print(f"   Max: {total_stats['max']:.0f}ms")
            print(f"   Avg: {total_stats['avg']:.0f}ms")
            print(f"   Median: {total_stats['median']:.0f}ms")
    
    elif args.check:
        # Run single check
        async def run_check():
            result = await watcher.run_stability_check()
            
            print(f"\n⏱️ Stability Check Results:")
            print(f"   Total latency: {result['total_latency_ms']:.0f}ms")
            print(f"   Max component: {result['max_latency_ms']:.0f}ms")
            print(f"   Average: {result['avg_latency_ms']:.0f}ms")
            
            alerts = result.get("alerts_triggered", [])
            if alerts:
                print(f"\n🚨 Alerts Triggered ({len(alerts)}):")
                for alert in alerts:
                    print(f"   • {alert['type']}: {alert.get('current_ms', 'N/A'):.0f}ms")
            else:
                print(f"\n✅ No latency alerts")
        
        asyncio.run(run_check())
    
    elif args.daemon:
        # Run monitoring daemon
        async def run_daemon():
            await watcher.run_monitoring_daemon(args.interval)
        
        asyncio.run(run_daemon())
    
    else:
        print("💡 Use --check, --daemon, or --summary")
        print("   Examples:")
        print("   python level8d_stability_watch.py --check")
        print("   python level8d_stability_watch.py --daemon --interval 60")
        print("   python level8d_stability_watch.py --summary 24")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Level 8-D: Stability Watch
Monitors system latency and triggers alerts when performance degrades
"""

import json
import time
import statistics
import asyncio
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

class StabilityWatcher:
    """Monitor system performance and detect latency anomalies"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Directories
        self.logs_dir = Path("logs")
        self.intel_dir = Path("intel")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.latency_history = deque(maxlen=50)  # Keep last 50 measurements
        self.performance_log = self.logs_dir / "stability_watch.json"
        
        # Alert thresholds
        self.latency_multiplier = 2.0  # Alert when > 2x rolling median
        self.min_samples = 10  # Need minimum samples for analysis
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        
        # Last alert timestamps
        self.last_alerts = {}
        
        # API endpoints to monitor
        self.api_base = "http://127.0.0.1:8000"
        self.endpoints = ["/status", "/kpis", "/intel", "/health"]
        
        print(f"👁️ Stability Watch initialized")
        print(f"   Alert threshold: {self.latency_multiplier}x rolling median")
        print(f"   Monitoring {len(self.endpoints)} endpoints")
    
    async def measure_api_latency(self) -> Dict[str, float]:
        """Measure latency for all API endpoints"""
        latencies = {}
        
        for endpoint in self.endpoints:
            url = f"{self.api_base}{endpoint}"
            
            try:
                start_time = time.time()
                
                # Make async request
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: requests.get(url, timeout=10)
                )
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                latencies[endpoint] = latency_ms
                
            except Exception as e:
                # Record failed requests as very high latency
                latencies[endpoint] = 9999.0
                print(f"⚠️ Request failed {endpoint}: {e}")
        
        return latencies
    
    def measure_data_fetch_latency(self) -> Dict[str, float]:
        """Measure time to read local data files"""
        latencies = {}
        
        # Test files to measure
        test_files = {
            "strategy_scores": self.logs_dir / "strategy_scores.json",
            "news_sentiment": self.intel_dir / "news_sentiment.csv", 
            "market_regime": self.intel_dir / "market_regime.json",
            "parent_summary": self._find_latest_summary()
        }
        
        for name, file_path in test_files.items():
            if not file_path or not file_path.exists():
                latencies[name] = 0.0
                continue
            
            try:
                start_time = time.time()
                
                # Read file
                if file_path.suffix == '.json':
                    with open(file_path) as f:
                        json.load(f)
                elif file_path.suffix == '.csv':
                    import pandas as pd
                    pd.read_csv(file_path)
                else:
                    with open(file_path) as f:
                        f.read()
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                latencies[name] = latency_ms
                
            except Exception as e:
                latencies[name] = 999.0  # Failed read
                print(f"⚠️ Data read failed {name}: {e}")
        
        return latencies
    
    def measure_intel_processing_latency(self) -> float:
        """Measure time to process intelligence data"""
        try:
            start_time = time.time()
            
            # Simulate typical intel processing
            self._process_sample_intel()
            
            end_time = time.time()
            return (end_time - start_time) * 1000
            
        except Exception as e:
            print(f"⚠️ Intel processing failed: {e}")
            return 999.0
    
    def _process_sample_intel(self):
        """Sample intelligence processing operations"""
        # Load and analyze sentiment data
        sentiment_file = self.intel_dir / "news_sentiment.csv"
        if sentiment_file.exists():
            import pandas as pd
            df = pd.read_csv(sentiment_file)
            
            if len(df) > 0:
                # Calculate sentiment statistics
                sentiments = df['sentiment_compound'].astype(float)
                _ = sentiments.mean()
                _ = sentiments.std()
                _ = (sentiments > 0.1).sum()
        
        # Load regime data
        regime_file = self.intel_dir / "market_regime.json"
        if regime_file.exists():
            with open(regime_file) as f:
                regime_data = json.load(f)
                
                # Count regimes
                symbols = regime_data.get("symbols", {})
                for symbol_info in symbols.values():
                    _ = symbol_info.get("regime", "unknown")
    
    def _find_latest_summary(self) -> Optional[Path]:
        """Find the latest parent summary file"""
        summary_files = list(self.logs_dir.glob("parent_summary_*.json"))
        
        if not summary_files:
            return None
        
        return max(summary_files, key=lambda f: f.stat().st_mtime)
    
    async def run_stability_check(self) -> Dict[str, any]:
        """Run complete stability check and return results"""
        check_start = time.time()
        
        # Measure different components
        api_latencies = await self.measure_api_latency()
        data_latencies = self.measure_data_fetch_latency()
        intel_latency = self.measure_intel_processing_latency()
        
        # Combine all latencies
        all_latencies = {
            **{f"api{k}": v for k, v in api_latencies.items()},
            **{f"data_{k}": v for k, v in data_latencies.items()},
            "intel_processing": intel_latency
        }
        
        # Calculate aggregate metrics
        total_latency = sum(all_latencies.values())
        max_latency = max(all_latencies.values())
        avg_latency = total_latency / len(all_latencies)
        
        # Create performance record
        performance_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latencies": all_latencies,
            "total_latency_ms": total_latency,
            "max_latency_ms": max_latency,
            "avg_latency_ms": avg_latency,
            "check_duration_ms": (time.time() - check_start) * 1000
        }
        
        # Add to history
        self.latency_history.append({
            "timestamp": performance_record["timestamp"],
            "total_latency": total_latency,
            "max_latency": max_latency,
            "avg_latency": avg_latency
        })
        
        # Log performance
        self._log_performance(performance_record)
        
        # Check for anomalies
        alerts_triggered = self._check_latency_alerts(performance_record)
        
        performance_record["alerts_triggered"] = alerts_triggered
        
        return performance_record
    
    def _check_latency_alerts(self, record: Dict) -> List[Dict]:
        """Check for latency anomalies and trigger alerts"""
        alerts = []
        
        if len(self.latency_history) < self.min_samples:
            return alerts  # Not enough data yet
        
        # Calculate rolling statistics
        recent_totals = [h["total_latency"] for h in list(self.latency_history)[-20:]]
        recent_maxes = [h["max_latency"] for h in list(self.latency_history)[-20:]]
        
        median_total = statistics.median(recent_totals)
        median_max = statistics.median(recent_maxes)
        
        current_total = record["total_latency_ms"]
        current_max = record["max_latency_ms"]
        
        # Check total latency alert
        if current_total > median_total * self.latency_multiplier:
            alert = {
                "type": "total_latency_spike",
                "current_ms": current_total,
                "median_ms": median_total,
                "multiplier": current_total / median_total,
                "threshold": self.latency_multiplier,
                "timestamp": record["timestamp"]
            }
            
            if self._should_send_alert("total_latency"):
                alerts.append(alert)
                self._send_latency_alert(alert)
                self.last_alerts["total_latency"] = time.time()
        
        # Check max latency alert
        if current_max > median_max * self.latency_multiplier:
            alert = {
                "type": "max_latency_spike",
                "current_ms": current_max,
                "median_ms": median_max,
                "multiplier": current_max / median_max,
                "threshold": self.latency_multiplier,
                "timestamp": record["timestamp"]
            }
            
            if self._should_send_alert("max_latency"):
                alerts.append(alert)
                self._send_latency_alert(alert)
                self.last_alerts["max_latency"] = time.time()
        
        # Check for individual component alerts
        component_alerts = self._check_component_alerts(record)
        alerts.extend(component_alerts)
        
        return alerts
    
    def _check_component_alerts(self, record: Dict) -> List[Dict]:
        """Check individual components for latency spikes"""
        alerts = []
        latencies = record["latencies"]
        
        # Define thresholds for different components
        thresholds = {
            "api/status": 2000,  # 2 seconds
            "api/kpis": 1000,    # 1 second
            "data_news_sentiment": 500,  # 500ms
            "intel_processing": 1000     # 1 second
        }
        
        for component, latency in latencies.items():
            threshold = thresholds.get(component, 2000)  # Default 2s
            
            if latency > threshold:
                alert = {
                    "type": "component_latency_spike",
                    "component": component,
                    "current_ms": latency,
                    "threshold_ms": threshold,
                    "timestamp": record["timestamp"]
                }
                
                if self._should_send_alert(f"component_{component}"):
                    alerts.append(alert)
                    self._send_latency_alert(alert)
                    self.last_alerts[f"component_{component}"] = time.time()
        
        return alerts
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        last_alert_time = self.last_alerts.get(alert_type, 0)
        return time.time() - last_alert_time > self.alert_cooldown
    
    def _send_latency_alert(self, alert: Dict):
        """Send latency alert via unified alert hub"""
        try:
            # Import here to avoid circular dependency
            from level8a_unified_alerts import UnifiedAlertHub
            
            hub = UnifiedAlertHub()
            
            alert_type = alert["type"]
            
            if alert_type == "total_latency_spike":
                title = "System Latency Spike"
                message = (f"Total system latency spike detected!\n"
                          f"Current: {alert['current_ms']:.0f}ms\n"
                          f"Median: {alert['median_ms']:.0f}ms\n"
                          f"Multiplier: {alert['multiplier']:.1f}x (threshold: {alert['threshold']}x)")
                
            elif alert_type == "max_latency_spike":
                title = "Maximum Latency Spike"
                message = (f"Maximum component latency spike detected!\n"
                          f"Current: {alert['current_ms']:.0f}ms\n"
                          f"Median: {alert['median_ms']:.0f}ms\n"
                          f"Multiplier: {alert['multiplier']:.1f}x")
                
            elif alert_type == "component_latency_spike":
                title = f"Component Latency Alert"
                component = alert['component'].replace('_', ' ').title()
                message = (f"High latency detected in {component}\n"
                          f"Current: {alert['current_ms']:.0f}ms\n"
                          f"Threshold: {alert['threshold_ms']:.0f}ms")
            
            else:
                title = "Latency Alert"
                message = f"Latency anomaly detected: {alert}"
            
            hub.send_alert(
                title=title,
                message=message,
                level="WARNING",
                alert_type="latency_spike"
            )
            
            print(f"📨 Latency alert sent: {title}")
            
        except Exception as e:
            print(f"❌ Failed to send latency alert: {e}")
    
    def _log_performance(self, record: Dict):
        """Log performance record to file"""
        with open(self.performance_log, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, any]:
        """Get performance summary for the last N hours"""
        if not self.performance_log.exists():
            return {"error": "No performance data available"}
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        records = []
        
        try:
            with open(self.performance_log) as f:
                for line in f:
                    record = json.loads(line.strip())
                    record_time = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                    
                    if record_time > cutoff:
                        records.append(record)
            
            if not records:
                return {"error": "No recent performance data"}
            
            # Calculate summary statistics
            total_latencies = [r["total_latency_ms"] for r in records]
            max_latencies = [r["max_latency_ms"] for r in records]
            avg_latencies = [r["avg_latency_ms"] for r in records]
            
            alerts_count = sum(len(r.get("alerts_triggered", [])) for r in records)
            
            summary = {
                "timeframe_hours": hours,
                "total_checks": len(records),
                "latency_stats": {
                    "total": {
                        "min": min(total_latencies),
                        "max": max(total_latencies),
                        "avg": statistics.mean(total_latencies),
                        "median": statistics.median(total_latencies)
                    },
                    "max_component": {
                        "min": min(max_latencies),
                        "max": max(max_latencies),
                        "avg": statistics.mean(max_latencies),
                        "median": statistics.median(max_latencies)
                    }
                },
                "alerts_triggered": alerts_count,
                "latest_check": records[-1]["timestamp"],
                "performance_trend": self._calculate_trend(total_latencies)
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to analyze performance data: {e}"}
    
    def _calculate_trend(self, latencies: List[float]) -> str:
        """Calculate performance trend from latency data"""
        if len(latencies) < 5:
            return "insufficient_data"
        
        # Compare first and last halves
        first_half = latencies[:len(latencies)//2]
        second_half = latencies[len(latencies)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change_pct = ((second_avg - first_avg) / first_avg) * 100
        
        if change_pct > 20:
            return "degrading"
        elif change_pct < -20:
            return "improving"
        else:
            return "stable"
    
    async def run_monitoring_daemon(self, interval: int = 60):
        """Run continuous monitoring daemon"""
        print(f"🔄 Starting stability monitoring daemon (interval: {interval}s)")
        
        try:
            while True:
                print(f"⏱️ {datetime.now().strftime('%H:%M:%S')} - Running stability check...")
                
                result = await self.run_stability_check()
                
                alerts_count = len(result.get("alerts_triggered", []))
                total_latency = result["total_latency_ms"]
                
                if alerts_count > 0:
                    print(f"🚨 {alerts_count} latency alerts triggered")
                else:
                    print(f"✅ Stability check OK (total: {total_latency:.0f}ms)")
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stability monitoring stopped")
        except Exception as e:
            print(f"❌ Monitoring daemon error: {e}")

def main():
    """Command line interface for stability watch"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stability Watch (Level 8-D)")
    parser.add_argument('--check', action='store_true', help='Run single stability check')
    parser.add_argument('--daemon', action='store_true', help='Run monitoring daemon')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval (seconds)')
    parser.add_argument('--summary', type=int, default=24, help='Show performance summary (hours)')
    parser.add_argument('--threshold', type=float, default=2.0, help='Alert threshold multiplier')
    
    args = parser.parse_args()
    
    print("👁️ Stability Watch (Level 8-D)")
    print("=" * 40)
    
    watcher = StabilityWatcher()
    watcher.latency_multiplier = args.threshold
    
    if args.summary > 0:
        # Show performance summary
        summary = watcher.get_performance_summary(args.summary)
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
        else:
            print(f"\n📊 Performance Summary ({args.summary}h):")
            print(f"   Total checks: {summary['total_checks']}")
            print(f"   Alerts triggered: {summary['alerts_triggered']}")
            print(f"   Performance trend: {summary['performance_trend']}")
            
            total_stats = summary['latency_stats']['total']
            print(f"\n⏱️ Total Latency Stats:")
            print(f"   Min: {total_stats['min']:.0f}ms")
            print(f"   Max: {total_stats['max']:.0f}ms")
            print(f"   Avg: {total_stats['avg']:.0f}ms")
            print(f"   Median: {total_stats['median']:.0f}ms")
    
    elif args.check:
        # Run single check
        async def run_check():
            result = await watcher.run_stability_check()
            
            print(f"\n⏱️ Stability Check Results:")
            print(f"   Total latency: {result['total_latency_ms']:.0f}ms")
            print(f"   Max component: {result['max_latency_ms']:.0f}ms")
            print(f"   Average: {result['avg_latency_ms']:.0f}ms")
            
            alerts = result.get("alerts_triggered", [])
            if alerts:
                print(f"\n🚨 Alerts Triggered ({len(alerts)}):")
                for alert in alerts:
                    print(f"   • {alert['type']}: {alert.get('current_ms', 'N/A'):.0f}ms")
            else:
                print(f"\n✅ No latency alerts")
        
        asyncio.run(run_check())
    
    elif args.daemon:
        # Run monitoring daemon
        async def run_daemon():
            await watcher.run_monitoring_daemon(args.interval)
        
        asyncio.run(run_daemon())
    
    else:
        print("💡 Use --check, --daemon, or --summary")
        print("   Examples:")
        print("   python level8d_stability_watch.py --check")
        print("   python level8d_stability_watch.py --daemon --interval 60")
        print("   python level8d_stability_watch.py --summary 24")

if __name__ == "__main__":
    main()