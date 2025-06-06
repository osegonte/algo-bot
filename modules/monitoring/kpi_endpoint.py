#!/usr/bin/env python3
"""
Level 8-B: Live KPI Endpoint
FastAPI service providing real-time KPIs, intel snapshot, and top strategies
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uvicorn

app = FastAPI(
    title="Trading Bot Live KPI API",
    description="Real-time KPIs, intelligence data, and strategy rankings",
    version="1.0.0"
)

# Enable CORS for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LiveKPIService:
    """Service for providing live trading system status"""
    
    def __init__(self):
        self.logs_dir = Path("logs")
        self.intel_dir = Path("intel")
        self.models_dir = Path("models")
        
    def get_latest_kpis(self) -> Dict[str, Any]:
        """Get latest KPI data from parent summaries"""
        try:
            # Find most recent parent summary
            summary_files = list(self.logs_dir.glob("parent_summary_*.json"))
            
            if not summary_files:
                return {"error": "No KPI data found", "kpis": {}}
            
            latest_file = max(summary_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file) as f:
                data = json.load(f)
            
            kpis = data.get("kpis", {})
            
            # Add metadata
            kpis["last_updated"] = data.get("generated_at", "unknown")
            kpis["data_source"] = "parent_summary"
            kpis["trades_analyzed"] = data.get("metadata", {}).get("completed_trades", 0)
            
            return kpis
            
        except Exception as e:
            return {"error": f"KPI load failed: {str(e)}", "kpis": {}}
    
    def get_intel_snapshot(self) -> Dict[str, Any]:
        """Get current intelligence data snapshot"""
        intel_snapshot = {
            "news_sentiment": self._get_sentiment_summary(),
            "market_regime": self._get_regime_summary(),
            "economic_calendar": self._get_econ_summary(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        return intel_snapshot
    
    def _get_sentiment_summary(self) -> Dict[str, Any]:
        """Get news sentiment summary"""
        try:
            sentiment_file = self.intel_dir / "news_sentiment.csv"
            
            if not sentiment_file.exists():
                return {"available": False, "reason": "no_data"}
            
            df = pd.read_csv(sentiment_file)
            
            if len(df) == 0:
                return {"available": False, "reason": "empty_data"}
            
            sentiments = df['sentiment_compound'].astype(float)
            
            # Get most recent headlines (top 5)
            recent_headlines = df.tail(5)[['title', 'sentiment_compound', 'source']].to_dict('records')
            
            return {
                "available": True,
                "total_headlines": len(df),
                "avg_sentiment": float(sentiments.mean()),
                "sentiment_std": float(sentiments.std()),
                "positive_count": int((sentiments > 0.1).sum()),
                "negative_count": int((sentiments < -0.1).sum()),
                "neutral_count": int(len(sentiments) - (sentiments > 0.1).sum() - (sentiments < -0.1).sum()),
                "latest_sentiment": float(sentiments.iloc[-1]),
                "recent_headlines": recent_headlines
            }
            
        except Exception as e:
            return {"available": False, "reason": f"error: {str(e)}"}
    
    def _get_regime_summary(self) -> Dict[str, Any]:
        """Get market regime summary"""
        try:
            regime_file = self.intel_dir / "market_regime.json"
            
            if not regime_file.exists():
                return {"available": False, "reason": "no_data"}
            
            with open(regime_file) as f:
                data = json.load(f)
            
            symbols = data.get("symbols", {})
            summary = data.get("summary", {})
            
            # Extract regime info for each symbol
            regime_details = {}
            for symbol, info in symbols.items():
                regime_details[symbol] = {
                    "regime": info.get("regime", "unknown"),
                    "confidence": info.get("confidence", 0),
                    "price": info.get("price", 0)
                }
            
            return {
                "available": True,
                "total_symbols": summary.get("total_symbols", 0),
                "bull_count": summary.get("bull_count", 0),
                "bear_count": summary.get("bear_count", 0),
                "range_count": summary.get("range_count", 0),
                "avg_confidence": summary.get("avg_confidence", 0),
                "analysis_date": data.get("detection_date", "unknown"),
                "regime_details": regime_details
            }
            
        except Exception as e:
            return {"available": False, "reason": f"error: {str(e)}"}
    
    def _get_econ_summary(self) -> Dict[str, Any]:
        """Get economic calendar summary"""
        try:
            econ_file = self.intel_dir / "econ_calendar.json"
            
            if not econ_file.exists():
                return {"available": False, "reason": "no_data"}
            
            with open(econ_file) as f:
                data = json.load(f)
            
            events = data.get("events", [])
            now = datetime.now(timezone.utc)
            
            # Get upcoming events (next 24 hours)
            upcoming_events = []
            for event in events:
                event_time_str = event.get("time_utc", "")
                if event_time_str:
                    try:
                        event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                        hours_until = (event_time - now).total_seconds() / 3600
                        
                        if 0 <= hours_until <= 24:
                            upcoming_events.append({
                                "name": event.get("name", "Unknown"),
                                "impact": event.get("impact", "low"),
                                "hours_until": round(hours_until, 1),
                                "currency": event.get("currency", "Unknown")
                            })
                    except ValueError:
                        continue
            
            # Sort by time
            upcoming_events.sort(key=lambda e: e["hours_until"])
            
            return {
                "available": True,
                "total_events": len(events),
                "upcoming_24h": len(upcoming_events),
                "next_events": upcoming_events[:5],  # Next 5 events
                "impact_breakdown": data.get("metadata", {}).get("impact_breakdown", {})
            }
            
        except Exception as e:
            return {"available": False, "reason": f"error: {str(e)}"}
    
    def get_top_strategies(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get top N strategies from latest rankings"""
        try:
            # Try hybrid rankings first
            hybrid_file = self.logs_dir / "latest_hybrid_rankings.json"
            
            if hybrid_file.exists():
                with open(hybrid_file) as f:
                    data = json.load(f)
                
                rankings = data.get("rankings", [])
                top_strategies = []
                
                for i, ranking in enumerate(rankings[:limit]):
                    strategy_info = {
                        "rank": i + 1,
                        "strategy": ranking.get("strategy", "unknown"),
                        "score": ranking.get("hybrid_score", 0),
                        "score_type": "hybrid",
                        "rule_score": ranking.get("rule_score", 0),
                        "ml_score": ranking.get("ml_score", 0),
                        "ml_confidence": ranking.get("ml_confidence", 0),
                        "scoring_method": ranking.get("scoring_method", "unknown")
                    }
                    top_strategies.append(strategy_info)
                
                return top_strategies
            
            # Fallback to rule-based rankings
            rule_file = self.logs_dir / "strategy_scores.json"
            
            if rule_file.exists():
                with open(rule_file) as f:
                    data = json.load(f)
                
                scores = data.get("scores", [])
                top_strategies = []
                
                for i, score in enumerate(scores[:limit]):
                    strategy_info = {
                        "rank": i + 1,
                        "strategy": score.get("strategy", "unknown"),
                        "score": score.get("score", 0),
                        "score_type": "rule_based",
                        "win_rate": score.get("win_rate", 0),
                        "profit_factor": score.get("profit_factor", 0),
                        "net_pnl": score.get("net_pnl", 0)
                    }
                    top_strategies.append(strategy_info)
                
                return top_strategies
            
            return []
            
        except Exception as e:
            return [{"error": f"Strategy ranking load failed: {str(e)}"}]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        
        health = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "unknown",
            "components": {},
            "last_activity": {},
            "alerts": 0
        }
        
        # Check component health
        components = {
            "kpis": self._check_file_freshness("parent_summary_*.json", hours=24),
            "intel": self._check_file_freshness("intel/news_sentiment.csv", hours=24),
            "strategies": self._check_file_freshness("strategy_scores.json", hours=24),
            "ml_model": self._check_file_freshness("models/latest_model.pkl", hours=168)  # 1 week
        }
        
        health["components"] = components
        
        # Overall status
        healthy_components = sum(1 for status in components.values() if status == "healthy")
        total_components = len(components)
        
        if healthy_components == total_components:
            health["status"] = "healthy"
        elif healthy_components >= total_components * 0.75:
            health["status"] = "degraded"
        else:
            health["status"] = "unhealthy"
        
        # Check for recent alerts
        try:
            alert_file = self.logs_dir / "unified_alerts.json"
            if alert_file.exists():
                cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
                alert_count = 0
                
                with open(alert_file) as f:
                    for line in f:
                        alert = json.loads(line.strip())
                        alert_time = datetime.fromisoformat(alert["timestamp"].replace("Z", "+00:00"))
                        
                        if alert_time > cutoff:
                            alert_count += 1
                
                health["alerts"] = alert_count
        except:
            pass
        
        return health
    
    def _check_file_freshness(self, pattern: str, hours: int = 24) -> str:
        """Check if file exists and is recent"""
        try:
            if "*" in pattern:
                # Glob pattern
                if "/" in pattern:
                    dir_path, file_pattern = pattern.rsplit("/", 1)
                    files = list(Path(dir_path).glob(file_pattern))
                else:
                    files = list(self.logs_dir.glob(file_pattern))
            else:
                # Direct file path
                files = [Path(pattern)]
            
            if not files:
                return "missing"
            
            # Check most recent file
            latest_file = max(files, key=lambda f: f.stat().st_mtime if f.exists() else 0)
            
            if not latest_file.exists():
                return "missing"
            
            # Check freshness
            file_time = datetime.fromtimestamp(latest_file.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - file_time).total_seconds() / 3600
            
            if age_hours <= hours:
                return "healthy"
            else:
                return "stale"
                
        except Exception:
            return "error"

# Create service instance
kpi_service = LiveKPIService()

# API Routes

@app.get("/", summary="API Info")
async def root():
    """API information and available endpoints"""
    return {
        "name": "Trading Bot Live KPI API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "/status": "Complete system status with KPIs, intel, and top strategies",
            "/kpis": "Latest KPI data only",
            "/intel": "Intelligence snapshot only", 
            "/strategies": "Top strategy rankings only",
            "/health": "System health check"
        }
    }

@app.get("/status", summary="Complete System Status")
async def get_status():
    """
    Main endpoint: Returns complete system status including:
    - Latest KPIs (P&L, win rate, etc.)
    - Intelligence snapshot (sentiment, regime, economic events)
    - Top 3 strategy rankings
    """
    try:
        # Gather all status data
        kpis = kpi_service.get_latest_kpis()
        intel = kpi_service.get_intel_snapshot()
        strategies = kpi_service.get_top_strategies(limit=3)
        health = kpi_service.get_system_health()
        
        status_response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": health["status"],
            "kpis": kpis,
            "intelligence": intel,
            "top_strategies": strategies,
            "health": health
        }
        
        return JSONResponse(content=status_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status endpoint failed: {str(e)}")

@app.get("/kpis", summary="Latest KPIs")
async def get_kpis():
    """Get latest KPI data from parent controller"""
    kpis = kpi_service.get_latest_kpis()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kpis": kpis
    }

@app.get("/intel", summary="Intelligence Snapshot")
async def get_intel():
    """Get current intelligence data (sentiment, regime, economic events)"""
    intel = kpi_service.get_intel_snapshot()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intelligence": intel
    }

@app.get("/strategies", summary="Strategy Rankings")
async def get_strategies(limit: int = 3):
    """Get top N strategy rankings"""
    if limit < 1 or limit > 10:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 10")
    
    strategies = kpi_service.get_top_strategies(limit=limit)
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_strategies": strategies,
        "limit": limit
    }

@app.get("/health", summary="System Health Check")
async def get_health():
    """Get system health and component status"""
    health = kpi_service.get_system_health()
    
    return health

@app.get("/alerts", summary="Recent Alerts")
async def get_alerts(hours: int = 24):
    """Get recent alerts from the alert hub"""
    try:
        alert_file = Path("logs/unified_alerts.json")
        
        if not alert_file.exists():
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alerts": [],
                "count": 0
            }
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_alerts = []
        
        with open(alert_file) as f:
            for line in f:
                alert = json.loads(line.strip())
                alert_time = datetime.fromisoformat(alert["timestamp"].replace("Z", "+00:00"))
                
                if alert_time > cutoff:
                    recent_alerts.append(alert)
        
        # Sort by timestamp, most recent first
        recent_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alerts": recent_alerts,
            "count": len(recent_alerts),
            "timeframe_hours": hours
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alerts endpoint failed: {str(e)}")

@app.get("/metrics", summary="Performance Metrics")
async def get_metrics():
    """Get system performance metrics"""
    try:
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_uptime": "unknown",
            "total_requests": "unknown", 
            "avg_response_time": "unknown"
        }
        
        # Add file system metrics
        metrics["disk_usage"] = {}
        
        for dir_name in ["logs", "intel", "models", "data"]:
            dir_path = Path(dir_name)
            if dir_path.exists():
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                file_count = len(list(dir_path.rglob('*')))
                
                metrics["disk_usage"][dir_name] = {
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": file_count
                }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics endpoint failed: {str(e)}")

def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    print(f"🚀 Starting Live KPI API server...")
    print(f"📡 Server: http://{host}:{port}")
    print(f"📊 Status endpoint: http://{host}:{port}/status")
    print(f"📖 API docs: http://{host}:{port}/docs")
    
    uvicorn.run(
        "level8b_kpi_endpoint:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live KPI Endpoint (Level 8-B)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--test", action="store_true", help="Test endpoints")
    
    args = parser.parse_args()
    
    if args.test:
        # Test the service
        print("🧪 Testing KPI Service...")
        
        service = LiveKPIService()
        
        # Test KPIs
        kpis = service.get_latest_kpis()
        print(f"✅ KPIs: {len(kpis)} fields loaded")
        
        # Test intel
        intel = service.get_intel_snapshot()
        print(f"✅ Intel: {len(intel)} categories loaded")
        
        # Test strategies
        strategies = service.get_top_strategies()
        print(f"✅ Strategies: {len(strategies)} rankings loaded")
        
        # Test health
        health = service.get_system_health()
        print(f"✅ Health: {health['status']} status")
        
        print("🎯 Level 8-B service ready!")
        
    else:
        start_server(host=args.host, port=args.port, reload=args.reload)