#!/usr/bin/env python3
"""
Level 5-F: Failure & Fallback System
Robust market data system with automatic failover between providers
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Import our gateway components
from market_data_gateway import MarketDataGateway, Quote, AlpacaGateway, IEXGateway, ForexGateway

@dataclass
class DataSourceStatus:
    """Track data source health and performance"""
    name: str
    is_healthy: bool
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    failure_count: int
    success_count: int
    avg_response_time: float
    priority: int

class RobustMarketDataManager:
    """Market data manager with automatic failover capabilities"""
    
    def __init__(self, config: dict, log_path: str = "logs"):
        self.config = config
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)
        
        # Initialize gateways in priority order
        self.gateways: List[MarketDataGateway] = []
        self.gateway_status: Dict[str, DataSourceStatus] = {}
        
        self.setup_gateways()
        
        # Failover settings
        self.max_retries = 3
        self.failover_threshold = 3  # failures before switching
        self.health_check_interval = 300  # seconds
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def setup_gateways(self):
        """Initialize all available gateways in priority order"""
        
        # Priority 1: Alpaca (primary)
        alpaca_config = self.config.get("alpaca", {})
        if alpaca_config.get("api_key"):
            alpaca = AlpacaGateway(
                alpaca_config["api_key"],
                alpaca_config["api_secret"],
                alpaca_config.get("base_url", "https://paper-api.alpaca.markets")
            )
            self.gateways.append(alpaca)
            self.gateway_status["alpaca"] = DataSourceStatus(
                name="alpaca",
                is_healthy=True,
                last_success=None,
                last_failure=None,
                failure_count=0,
                success_count=0,
                avg_response_time=0.0,
                priority=1
            )
        
        # Priority 2: IEX Cloud (backup)
        iex_config = self.config.get("iex", {})
        if iex_config.get("api_key"):
            iex = IEXGateway(iex_config["api_key"])
            self.gateways.append(iex)
            self.gateway_status["iex"] = DataSourceStatus(
                name="iex",
                is_healthy=True,
                last_success=None,
                last_failure=None,
                failure_count=0,
                success_count=0,
                avg_response_time=0.0,
                priority=2
            )
        
        # Priority 3: Forex simulation (always available)
        forex = ForexGateway()
        self.gateways.append(forex)
        self.gateway_status["forex"] = DataSourceStatus(
            name="forex",
            is_healthy=True,
            last_success=None,
            last_failure=None,
            failure_count=0,
            success_count=0,
            avg_response_time=0.0,
            priority=3
        )
        
        self.logger.info(f"🔌 Initialized {len(self.gateways)} data gateways")
    
    async def get_quote_with_failover(self, symbol: str) -> Quote:
        """Get quote with automatic failover between data sources"""
        
        start_time = datetime.now()
        
        # Get suitable gateways for this symbol
        suitable_gateways = self._get_suitable_gateways(symbol)
        
        if not suitable_gateways:
            raise Exception(f"No suitable gateways found for {symbol}")
        
        # Try gateways in priority order
        last_exception = None
        
        for gateway in suitable_gateways:
            gateway_name = gateway.__class__.__name__.lower().replace('gateway', '')
            
            # Skip if gateway is marked unhealthy
            status = self.gateway_status.get(gateway_name)
            if status and not status.is_healthy:
                self.logger.warning(f"⚠️ Skipping unhealthy gateway: {gateway_name}")
                continue
            
            try:
                self.logger.debug(f"🔄 Trying {gateway_name} for {symbol}...")
                
                # Attempt to get quote
                quote = await gateway.get_live_quote(symbol)
                
                # Success! Update status and return
                response_time = (datetime.now() - start_time).total_seconds()
                self._record_success(gateway_name, response_time)
                
                # Log successful failover if we're not using primary
                if gateway != suitable_gateways[0]:
                    self._log_failover_event(symbol, gateway_name, "success")
                
                return quote
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"❌ {gateway_name} failed for {symbol}: {e}")
                
                # Record failure
                self._record_failure(gateway_name, str(e))
                
                # Log failover attempt
                self._log_failover_event(symbol, gateway_name, "failed", str(e))
                
                continue
        
        # All gateways failed
        self.logger.error(f"💥 All gateways failed for {symbol}")
        self._log_complete_failure(symbol)
        
        # Return a fallback quote to prevent system crash
        return self._create_fallback_quote(symbol, last_exception)
    
    def _get_suitable_gateways(self, symbol: str) -> List[MarketDataGateway]:
        """Get gateways that support the given symbol, sorted by priority"""
        suitable = []
        
        for gateway in self.gateways:
            if symbol in gateway.get_supported_symbols():
                suitable.append(gateway)
        
        # Sort by priority (lower number = higher priority)
        gateway_priorities = {
            'AlpacaGateway': 1,
            'IEXGateway': 2, 
            'ForexGateway': 3
        }
        
        suitable.sort(key=lambda g: gateway_priorities.get(g.__class__.__name__, 999))
        
        return suitable
    
    def _record_success(self, gateway_name: str, response_time: float):
        """Record successful gateway operation"""
        status = self.gateway_status.get(gateway_name)
        if status:
            status.last_success = datetime.now(timezone.utc)
            status.success_count += 1
            status.is_healthy = True
            
            # Update average response time
            if status.avg_response_time == 0:
                status.avg_response_time = response_time
            else:
                # Simple moving average
                status.avg_response_time = (status.avg_response_time + response_time) / 2
    
    def _record_failure(self, gateway_name: str, error: str):
        """Record gateway failure and update health status"""
        status = self.gateway_status.get(gateway_name)
        if status:
            status.last_failure = datetime.now(timezone.utc)
            status.failure_count += 1
            
            # Mark as unhealthy if too many consecutive failures
            if status.failure_count >= self.failover_threshold:
                status.is_healthy = False
                self.logger.warning(f"🚨 Marking {gateway_name} as unhealthy after {status.failure_count} failures")
    
    def _log_failover_event(self, symbol: str, gateway_name: str, result: str, error: str = None):
        """Log failover events for analysis"""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "failover_attempt",
            "symbol": symbol,
            "gateway": gateway_name,
            "result": result,
            "error": error,
            "data_source_fallback": result == "success" and gateway_name != "alpaca"
        }
        
        failover_log = self.log_path / "data_failover.json"
        with open(failover_log, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def _log_complete_failure(self, symbol: str):
        """Log when all gateways fail"""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "complete_failure",
            "symbol": symbol,
            "all_gateways_failed": True,
            "using_fallback_quote": True
        }
        
        failover_log = self.log_path / "data_failover.json"
        with open(failover_log, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def _create_fallback_quote(self, symbol: str, last_exception: Exception) -> Quote:
        """Create a fallback quote when all sources fail"""
        
        # Use reasonable defaults based on symbol type
        if symbol == "EURUSD":
            price = 1.0500
        elif symbol in ["AAPL", "TSLA", "MSFT"]:
            price = 150.0
        else:
            price = 100.0
        
        fallback_quote = Quote(
            symbol=symbol,
            bid=price - 0.05,
            ask=price + 0.05,
            last=price,
            timestamp=datetime.now(timezone.utc),
            source="fallback_emergency"
        )
        
        self.logger.warning(f"🆘 Using emergency fallback quote for {symbol}: ${price}")
        
        return fallback_quote
    
    async def health_check_all_gateways(self) -> Dict[str, bool]:
        """Perform health check on all gateways"""
        health_results = {}
        
        self.logger.info("🔍 Performing gateway health checks...")
        
        for gateway in self.gateways:
            gateway_name = gateway.__class__.__name__.lower().replace('gateway', '')
            
            try:
                # Test connection
                is_healthy = await gateway.connect()
                health_results[gateway_name] = is_healthy
                
                # Update status
                if gateway_name in self.gateway_status:
                    self.gateway_status[gateway_name].is_healthy = is_healthy
                    
                    if is_healthy:
                        # Reset failure count on successful health check
                        self.gateway_status[gateway_name].failure_count = 0
                
                status_emoji = "✅" if is_healthy else "❌"
                self.logger.info(f"{status_emoji} {gateway_name}: {'healthy' if is_healthy else 'unhealthy'}")
                
            except Exception as e:
                health_results[gateway_name] = False
                self.logger.error(f"❌ {gateway_name} health check failed: {e}")
        
        return health_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_gateways": len(self.gateways),
            "healthy_gateways": sum(1 for status in self.gateway_status.values() if status.is_healthy),
            "gateway_details": {
                name: {
                    "healthy": status.is_healthy,
                    "success_count": status.success_count,
                    "failure_count": status.failure_count,
                    "avg_response_time": round(status.avg_response_time, 3),
                    "last_success": status.last_success.isoformat() if status.last_success else None,
                    "last_failure": status.last_failure.isoformat() if status.last_failure else None,
                    "priority": status.priority
                }
                for name, status in self.gateway_status.items()
            }
        }

# Demo and testing
async def test_failover_system():
    """Test the failover system with various scenarios"""
    
    config = {
        "alpaca": {
            "api_key": "PKKJRF8U4QTYWMAVLYUI",
            "api_secret": "4kGChcW8cqkxasJfvyVMYfGRKbPqwNcx8MzM26ws",
            "base_url": "https://paper-api.alpaca.markets"
        },
        "iex": {
            "api_key": "demo_key"  # This will fail, demonstrating failover
        }
    }
    
    print("🚀 Testing Market Data Failover System (Level 5-F)")
    print("=" * 60)
    
    manager = RobustMarketDataManager(config)
    
    # Health check
    print("🔍 Initial health check...")
    health = await manager.health_check_all_gateways()
    
    # Test quotes with failover
    test_symbols = ["AAPL", "EURUSD"]
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol} with failover...")
        
        try:
            quote = await manager.get_quote_with_failover(symbol)
            print(f"✅ Got quote: ${quote.last:.4f} from {quote.source}")
            
            if "fallback" in quote.source:
                print("🆘 Used emergency fallback!")
            
        except Exception as e:
            print(f"❌ Complete failure for {symbol}: {e}")
    
    # Show system status
    print(f"\n📊 System Status:")
    status = manager.get_system_status()
    print(f"Healthy gateways: {status['healthy_gateways']}/{status['total_gateways']}")
    
    for name, details in status['gateway_details'].items():
        health_emoji = "✅" if details['healthy'] else "❌"
        print(f"{health_emoji} {name}: {details['success_count']} successes, {details['failure_count']} failures")
    
    print("\n✅ Failover system test complete!")

if __name__ == "__main__":
    asyncio.run(test_failover_system())