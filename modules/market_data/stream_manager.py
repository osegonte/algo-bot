#!/usr/bin/env python3
"""
Fixed Stream Manager that can run directly or be imported
File: modules/market_data/stream_manager.py
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Callable, Optional
from queue import Queue
import threading
import time
from pathlib import Path

# Handle both direct execution and import scenarios
if __name__ == "__main__":
    # Direct execution - add project root to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from modules.market_data.gateway import MarketDataGateway, Quote, DataGatewayFactory
else:
    # Normal import - use relative import
    try:
        from .gateway import MarketDataGateway, Quote, DataGatewayFactory
    except ImportError:
        # Fallback for edge cases
        from gateway import MarketDataGateway, Quote, DataGatewayFactory

class PriceStreamManager:
    """Manages real-time price feeds with streaming and polling options"""
    
    def __init__(self, config: dict, log_path: str = "logs"):
        self.config = config
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)
        
        self.gateways: Dict[str, MarketDataGateway] = {}
        self.price_queue = Queue()
        self.latest_quotes: Dict[str, Quote] = {}
        
        self.is_streaming = False
        self.stream_thread = None
        self.poll_interval = 5  # seconds
        
        self.subscribers: List[Callable[[Quote], None]] = []
        
        # Symbols to track
        self.symbols = ["AAPL", "EURUSD"]
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def start_streaming(self, use_websocket: bool = True) -> bool:
        """Start price streaming - websocket preferred, polling fallback"""
        
        self.logger.info(f"🚀 Starting price stream (websocket: {use_websocket})")
        
        # Initialize gateways for each symbol
        for symbol in self.symbols:
            gateway = DataGatewayFactory.create_gateway(symbol, self.config)
            self.gateways[symbol] = gateway
            connected = await gateway.connect()
            self.logger.info(f"🔗 {symbol} gateway connected: {connected}")
        
        if use_websocket:
            success = await self._try_websocket_stream()
            if success:
                self._log_stream_status(True, "websocket")
                return True
            else:
                self.logger.warning("📡 Websocket failed, falling back to polling")
        
        # Fallback to polling
        success = await self._start_polling_stream()
        if success:
            self._log_stream_status(True, "polling")
            return True
        else:
            self.logger.error("❌ Both streaming methods failed")
            self._log_stream_status(False, "failed")
            return False
    
    async def _try_websocket_stream(self) -> bool:
        """Attempt websocket streaming (simplified simulation)"""
        try:
            # For demo, we'll simulate websocket with rapid polling
            self.logger.info("📡 Simulating websocket stream...")
            
            self.is_streaming = True
            self.stream_thread = threading.Thread(
                target=self._websocket_simulation_loop, 
                daemon=True
            )
            self.stream_thread.start()
            
            # Give it a moment to start
            await asyncio.sleep(1)
            return self.is_streaming
            
        except Exception as e:
            self.logger.error(f"Websocket setup failed: {e}")
            return False
    
    def _websocket_simulation_loop(self):
        """Simulate websocket updates every 2-3 seconds"""
        while self.is_streaming:
            try:
                # Get quotes for all symbols
                for symbol in self.symbols:
                    gateway = self.gateways.get(symbol)
                    if gateway:
                        # Run async code in thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        quote = loop.run_until_complete(gateway.get_live_quote(symbol))
                        loop.close()
                        
                        self._process_quote(quote)
                
                # Wait 2-3 seconds for next update
                time.sleep(2.5)
                
            except Exception as e:
                self.logger.error(f"Websocket loop error: {e}")
                time.sleep(5)
    
    async def _start_polling_stream(self) -> bool:
        """Start polling-based price updates"""
        try:
            self.logger.info(f"📊 Starting polling stream (interval: {self.poll_interval}s)")
            
            self.is_streaming = True
            self.stream_thread = threading.Thread(
                target=self._polling_loop,
                daemon=True
            )
            self.stream_thread.start()
            
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            self.logger.error(f"Polling setup failed: {e}")
            return False
    
    def _polling_loop(self):
        """Main polling loop"""
        while self.is_streaming:
            try:
                # Poll each symbol
                for symbol in self.symbols:
                    gateway = self.gateways.get(symbol)
                    if gateway:
                        # Run async in thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        quote = loop.run_until_complete(gateway.get_live_quote(symbol))
                        loop.close()
                        
                        self._process_quote(quote)
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                self.logger.error(f"Polling error: {e}")
                time.sleep(self.poll_interval)
    
    def _process_quote(self, quote: Quote):
        """Process incoming quote"""
        # Update latest quotes
        self.latest_quotes[quote.symbol] = quote
        
        # Add to queue
        self.price_queue.put(quote)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(quote)
            except Exception as e:
                self.logger.error(f"Subscriber error: {e}")
        
        # Log quote update (less verbose)
        if quote.symbol == "AAPL":  # Only log AAPL to reduce noise
            self._log_quote_update(quote)
    
    def subscribe(self, callback: Callable[[Quote], None]):
        """Subscribe to price updates"""
        self.subscribers.append(callback)
        self.logger.info(f"📬 Added price subscriber")
    
    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for symbol"""
        return self.latest_quotes.get(symbol)
    
    def get_price_queue(self) -> Queue:
        """Get the price update queue"""
        return self.price_queue
    
    def stop_streaming(self):
        """Stop price streaming"""
        self.logger.info("⏹️ Stopping price stream...")
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        
        self._log_stream_status(False, "stopped")
    
    def _log_stream_status(self, started: bool, method: str):
        """Log streaming status"""
        status_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "price_stream_status",
            "price_stream_started": started,
            "method": method,
            "symbols": self.symbols
        }
        
        status_file = self.log_path / "price_stream.json"
        with open(status_file, "a") as f:
            f.write(json.dumps(status_entry) + "\n")
        
        if started:
            self.logger.info(f"✅ Price stream started via {method}")
        else:
            self.logger.info(f"⏹️ Price stream stopped")
    
    def _log_quote_update(self, quote: Quote):
        """Log individual quote updates"""
        quote_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": quote.symbol,
            "bid": quote.bid,
            "ask": quote.ask,
            "last": quote.last,
            "source": quote.source,
            "quote_timestamp": quote.timestamp.isoformat()
        }
        
        # Log to symbol-specific file
        quote_file = self.log_path / f"quotes_{quote.symbol.lower()}.json"
        with open(quote_file, "a") as f:
            f.write(json.dumps(quote_entry) + "\n")

class PriceStreamMonitor:
    """Monitor price stream health and provide status"""
    
    def __init__(self, stream_manager: PriceStreamManager):
        self.stream_manager = stream_manager
        self.last_update_times: Dict[str, datetime] = {}
        
    def check_stream_health(self) -> Dict[str, any]:
        """Check if price stream is healthy"""
        now = datetime.now(timezone.utc)
        health_status = {
            "is_streaming": self.stream_manager.is_streaming,
            "symbols_tracked": len(self.stream_manager.symbols),
            "latest_quotes": {},
            "stale_symbols": [],
            "overall_health": "unknown"
        }
        
        for symbol in self.stream_manager.symbols:
            quote = self.stream_manager.get_latest_quote(symbol)
            if quote:
                age_seconds = (now - quote.timestamp).total_seconds()
                health_status["latest_quotes"][symbol] = {
                    "last_price": quote.last,
                    "age_seconds": age_seconds,
                    "source": quote.source
                }
                
                # Flag stale quotes (>30 seconds old)
                if age_seconds > 30:
                    health_status["stale_symbols"].append(symbol)
            else:
                health_status["stale_symbols"].append(symbol)
        
        # Overall health assessment
        if not self.stream_manager.is_streaming:
            health_status["overall_health"] = "stopped"
        elif len(health_status["stale_symbols"]) == 0:
            health_status["overall_health"] = "healthy"
        elif len(health_status["stale_symbols"]) < len(self.stream_manager.symbols):
            health_status["overall_health"] = "degraded"
        else:
            health_status["overall_health"] = "unhealthy"
        
        return health_status

# Demonstration and testing
async def demo_price_streaming():
    """Demo the price streaming system"""
    
    # Load config from actual file if available
    config_path = Path("../../config/base_config.yaml")  # From modules/market_data/
    if not config_path.exists():
        config_path = Path("config/base_config.yaml")  # From project root
    
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "alpaca": {
                "api_key": "PKKJRF8U4QTYWMAVLYUI",
                "api_secret": "4kGChcW8cqkxasJfvyVMYfGRKbPqwNcx8MzM26ws",
                "base_url": "https://paper-api.alpaca.markets"
            }
        }
    
    print("🚀 Starting Fixed Price Stream Demo")
    print("=" * 40)
    
    # Create stream manager
    stream_manager = PriceStreamManager(config)
    monitor = PriceStreamMonitor(stream_manager)
    
    # Add subscriber to show updates
    def quote_subscriber(quote: Quote):
        print(f"📊 {quote.symbol}: ${quote.last:.4f} [{quote.source}] @ {quote.timestamp.strftime('%H:%M:%S')}")
    
    stream_manager.subscribe(quote_subscriber)
    
    # Start streaming
    success = await stream_manager.start_streaming(use_websocket=True)
    
    if success:
        print("✅ Price stream started successfully!")
        
        # Let it run for 10 seconds
        print("⏱️ Running for 10 seconds...")
        await asyncio.sleep(10)
        
        # Check health
        health = monitor.check_stream_health()
        print(f"\n📋 Stream Health: {health['overall_health']}")
        print(f"   Symbols: {list(health['latest_quotes'].keys())}")
        
        # Show latest prices
        for symbol, info in health['latest_quotes'].items():
            print(f"   {symbol}: ${info['last_price']:.4f} [{info['source']}]")
        
        # Stop streaming
        stream_manager.stop_streaming()
        print("✅ Demo complete!")
        
    else:
        print("❌ Failed to start price stream")

if __name__ == "__main__":
    asyncio.run(demo_price_streaming())