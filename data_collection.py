#!/usr/bin/env python3
"""
Enhanced Data Collection Module for XAU/USD Tick Trading
Optimized for tick-level scalping with advanced market microstructure analysis
"""

import asyncio
import json
import logging
import queue
import threading
import time
import websocket
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass

@dataclass
class TickData:
    """Enhanced tick data structure"""
    timestamp: datetime
    price: float
    size: int
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: int = 0
    conditions: List[str] = None
    exchange: str = "ALPACA"

class AdvancedTickCollector:
    """Enhanced tick data collector with microstructure analysis"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.is_running = False
        
        # Tick storage with enhanced capacity
        self.tick_buffer = deque(maxlen=2000)  # Increased capacity
        self.raw_tick_queue = queue.Queue(maxsize=1000)
        
        # Real-time analytics
        self.tick_velocity = 0.0
        self.price_acceleration = 0.0
        self.order_flow_imbalance = 0.0
        self.market_impact = 0.0
        
        # Market microstructure metrics
        self.spread_history = deque(maxlen=100)
        self.volume_profile = deque(maxlen=200)
        self.price_levels = {}  # Support/resistance tracking
        
        # Alpaca connection
        self.alpaca_api = None
        self.alpaca_stream = None
        
        # Callbacks for tick processing
        self.tick_callbacks = []
        self.analytics_callbacks = []
        
        # Performance tracking
        self.ticks_processed = 0
        self.connection_start_time = None
        self.last_tick_time = None
        
        logging.info(f"✅ Enhanced tick collector initialized for {symbol}")
    
    def add_tick_callback(self, callback: Callable):
        """Add callback function to be called on each tick"""
        self.tick_callbacks.append(callback)
        logging.info(f"Added tick callback: {callback.__name__}")
    
    def add_analytics_callback(self, callback: Callable):
        """Add callback for analytics updates"""
        self.analytics_callbacks.append(callback)
    
    def connect_to_alpaca(self, api_key: str, secret_key: str, paper_trading: bool = True) -> bool:
        """Establish connection to Alpaca"""
        try:
            import alpaca_trade_api as tradeapi
            
            # Initialize Alpaca API
            base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
            self.alpaca_api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Test connection
            account = self.alpaca_api.get_account()
            logging.info(f"✅ Alpaca connection established - Account: ${float(account.portfolio_value):,.2f}")
            
            return True
            
        except ImportError:
            logging.error("❌ alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            return False
        except Exception as e:
            logging.error(f"❌ Failed to connect to Alpaca: {e}")
            return False
    
    def stream_tick_data(self, api_key: str = "", secret_key: str = "", paper_trading: bool = True):
        """Start streaming tick data from Alpaca or fallback to simulation"""
        
        self.connection_start_time = datetime.now()
        
        if api_key and secret_key:
            success = self._start_alpaca_stream(api_key, secret_key, paper_trading)
            if success:
                logging.info(f"🌐 Alpaca tick stream started for {self.symbol}")
            else:
                logging.warning("⚠️  Alpaca stream failed, falling back to simulation")
                self._start_simulation_stream()
        else:
            logging.info("🎮 Starting simulation stream (no API keys provided)")
            self._start_simulation_stream()
        
        # Start tick processing thread
        self._start_tick_processor()
        
        self.is_running = True
    
    def _start_alpaca_stream(self, api_key: str, secret_key: str, paper_trading: bool) -> bool:
        """Start real Alpaca tick stream"""
        try:
            import alpaca_trade_api as tradeapi
            
            base_url = 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
            
            # Initialize stream connection
            self.alpaca_stream = tradeapi.StreamConn(
                api_key, secret_key,
                base_url=base_url,
                data_feed='sip'  # Use SIP feed for comprehensive data
            )
            
            # Set up tick data handler
            @self.alpaca_stream.on(r'^T\.' + self.symbol)
            async def on_tick(conn, channel, data):
                await self._process_alpaca_tick(data)
            
            # Set up quote data handler
            @self.alpaca_stream.on(r'^Q\.' + self.symbol)
            async def on_quote(conn, channel, data):
                await self._process_alpaca_quote(data)
            
            # Subscribe to data streams
            self.alpaca_stream.subscribe_trades(self.symbol)
            self.alpaca_stream.subscribe_quotes(self.symbol)
            
            # Start stream in separate thread
            def run_stream():
                try:
                    self.alpaca_stream.run()
                except Exception as e:
                    logging.error(f"Alpaca stream error: {e}")
                    self._start_simulation_stream()
            
            stream_thread = threading.Thread(target=run_stream, daemon=True)
            stream_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start Alpaca stream: {e}")
            return False
    
    async def _process_alpaca_tick(self, tick_data):
        """Process incoming Alpaca tick data"""
        try:
            tick = TickData(
                timestamp=datetime.now(),
                price=float(tick_data.price),
                size=int(tick_data.size),
                conditions=tick_data.conditions,
                exchange=tick_data.exchange
            )
            
            # Add to processing queue
            if not self.raw_tick_queue.full():
                self.raw_tick_queue.put(tick)
            
        except Exception as e:
            logging.error(f"Error processing Alpaca tick: {e}")
    
    async def _process_alpaca_quote(self, quote_data):
        """Process Alpaca quote data to enhance ticks"""
        try:
            # Update the most recent tick with bid/ask data
            if self.tick_buffer:
                latest_tick = self.tick_buffer[-1]
                latest_tick.bid = float(quote_data.bid_price)
                latest_tick.ask = float(quote_data.ask_price)
                latest_tick.spread = latest_tick.ask - latest_tick.bid
                
        except Exception as e:
            logging.error(f"Error processing Alpaca quote: {e}")
    
    def _start_simulation_stream(self):
        """Start simulated tick data for testing"""
        
        def simulate_ticks():
            base_price = 2000.0  # XAU/USD starting price
            price_trend = 0.0
            volatility = 0.1
            
            while self.is_running:
                try:
                    # Realistic price movement simulation
                    price_change = np.random.normal(price_trend, volatility)
                    base_price += price_change
                    
                    # Add some trend persistence
                    price_trend = price_trend * 0.95 + price_change * 0.05
                    
                    # Simulate bid/ask spread
                    spread = np.random.uniform(0.05, 0.15)
                    bid = base_price - spread/2
                    ask = base_price + spread/2
                    
                    # Create realistic tick
                    tick = TickData(
                        timestamp=datetime.now(),
                        price=round(base_price, 2),
                        size=np.random.randint(1, 100),
                        bid=round(bid, 2),
                        ask=round(ask, 2),
                        spread=round(spread, 3),
                        volume=np.random.randint(10, 200),
                        conditions=['@', 'F'],
                        exchange='SIM'
                    )
                    
                    # Add to processing queue
                    if not self.raw_tick_queue.full():
                        self.raw_tick_queue.put(tick)
                    
                    # Variable tick frequency (3-15 ticks per second)
                    time.sleep(np.random.uniform(0.067, 0.333))
                    
                except Exception as e:
                    logging.error(f"Simulation error: {e}")
                    time.sleep(1)
        
        sim_thread = threading.Thread(target=simulate_ticks, daemon=True)
        sim_thread.start()
        logging.info("🎮 Simulation tick stream started")
    
    def _start_tick_processor(self):
        """Start tick processing thread"""
        
        def process_ticks():
            while self.is_running:
                try:
                    # Get tick from queue with timeout
                    tick = self.raw_tick_queue.get(timeout=1)
                    
                    # Process the tick
                    self._process_tick(tick)
                    
                    # Mark task as done
                    self.raw_tick_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Tick processing error: {e}")
        
        processor_thread = threading.Thread(target=process_ticks, daemon=True)
        processor_thread.start()
        logging.info("⚙️  Tick processor started")
    
    def _process_tick(self, tick: TickData):
        """Process individual tick and update analytics"""
        
        # Add to buffer
        self.tick_buffer.append(tick)
        self.ticks_processed += 1
        self.last_tick_time = tick.timestamp
        
        # Update real-time analytics
        self._update_analytics()
        
        # Call registered callbacks
        for callback in self.tick_callbacks:
            try:
                callback(self._tick_to_dict(tick))
            except Exception as e:
                logging.error(f"Tick callback error in {callback.__name__}: {e}")
        
        # Periodic analytics callbacks
        if self.ticks_processed % 10 == 0:
            analytics = self.get_tick_analysis()
            for callback in self.analytics_callbacks:
                try:
                    callback(analytics)
                except Exception as e:
                    logging.error(f"Analytics callback error: {e}")
    
    def _tick_to_dict(self, tick: TickData) -> Dict:
        """Convert tick data to dictionary for callbacks"""
        return {
            'timestamp': tick.timestamp,
            'price': tick.price,
            'size': tick.size,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.spread,
            'volume': tick.volume,
            'conditions': tick.conditions or [],
            'exchange': tick.exchange
        }
    
    def _update_analytics(self):
        """Update real-time analytics metrics"""
        
        if len(self.tick_buffer) < 10:
            return
        
        recent_ticks = list(self.tick_buffer)[-20:]
        
        # Calculate tick velocity (ticks per second)
        if len(recent_ticks) >= 2:
            time_span = (recent_ticks[-1].timestamp - recent_ticks[0].timestamp).total_seconds()
            self.tick_velocity = (len(recent_ticks) - 1) / time_span if time_span > 0 else 0
        
        # Calculate price acceleration
        prices = [tick.price for tick in recent_ticks]
        if len(prices) >= 3:
            price_changes = np.diff(prices)
            if len(price_changes) >= 2:
                accelerations = np.diff(price_changes)
                self.price_acceleration = accelerations[-1] if len(accelerations) > 0 else 0
        
        # Calculate order flow imbalance (simplified Lee-Ready)
        self.order_flow_imbalance = self._calculate_order_flow_imbalance(recent_ticks)
        
        # Calculate market impact
        self.market_impact = self._calculate_market_impact(recent_ticks)
        
        # Update spread and volume tracking
        if recent_ticks[-1].spread > 0:
            self.spread_history.append(recent_ticks[-1].spread)
        
        self.volume_profile.append(recent_ticks[-1].size)
    
    def _calculate_order_flow_imbalance(self, ticks: List[TickData]) -> float:
        """Calculate order flow imbalance using Lee-Ready algorithm"""
        
        buy_volume = 0
        sell_volume = 0
        
        for i, tick in enumerate(ticks):
            if tick.bid > 0 and tick.ask > 0:
                mid_price = (tick.bid + tick.ask) / 2
                
                if tick.price > mid_price:
                    buy_volume += tick.size
                elif tick.price < mid_price:
                    sell_volume += tick.size
                else:
                    # Use tick rule for trades at mid
                    if i > 0:
                        prev_price = ticks[i-1].price
                        if tick.price > prev_price:
                            buy_volume += tick.size
                        elif tick.price < prev_price:
                            sell_volume += tick.size
        
        total_volume = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
    
    def _calculate_market_impact(self, ticks: List[TickData]) -> float:
        """Calculate market impact of large trades"""
        
        if len(ticks) < 5:
            return 0
        
        sizes = [tick.size for tick in ticks]
        prices = [tick.price for tick in ticks]
        
        avg_size = np.mean(sizes)
        price_changes = np.abs(np.diff(prices))
        
        # Find large trades and their price impact
        large_trade_impact = 0
        count = 0
        
        for i, (size, price_change) in enumerate(zip(sizes[1:], price_changes)):
            if size > avg_size * 1.5:  # Large trade threshold
                large_trade_impact += price_change
                count += 1
        
        return large_trade_impact / count if count > 0 else 0
    
    def get_tick_analysis(self, lookback_ticks: int = 50) -> Dict:
        """Get comprehensive tick analysis"""
        
        if len(self.tick_buffer) < lookback_ticks:
            lookback_ticks = len(self.tick_buffer)
        
        if lookback_ticks == 0:
            return {}
        
        recent_ticks = list(self.tick_buffer)[-lookback_ticks:]
        prices = [tick.price for tick in recent_ticks]
        sizes = [tick.size for tick in recent_ticks]
        spreads = [tick.spread for tick in recent_ticks if tick.spread > 0]
        
        analysis = {
            # Basic metrics
            'tick_count': len(recent_ticks),
            'current_price': prices[-1] if prices else 0,
            'price_range': max(prices) - min(prices) if prices else 0,
            'avg_tick_size': np.mean(sizes) if sizes else 0,
            'price_volatility': np.std(prices) if len(prices) > 1 else 0,
            
            # Advanced metrics
            'tick_velocity': self.tick_velocity,
            'price_acceleration': self.price_acceleration,
            'order_flow_imbalance': self.order_flow_imbalance,
            'market_impact': self.market_impact,
            
            # Spread and liquidity
            'avg_spread': np.mean(spreads) if spreads else 0,
            'spread_volatility': np.std(spreads) if len(spreads) > 1 else 0,
            'current_spread': spreads[-1] if spreads else 0,
            
            # Trend analysis
            'trend_direction': 1 if prices[-1] > prices[0] else -1 if prices else 0,
            'momentum_5': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0,
            'momentum_10': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0,
            
            # Volume analysis
            'volume_trend': np.mean(sizes[-5:]) - np.mean(sizes[-10:-5]) if len(sizes) >= 10 else 0,
            'large_tick_ratio': sum(1 for s in sizes[-10:] if s > np.mean(sizes)) / min(10, len(sizes)) if sizes else 0,
            
            # Support/resistance
            'support_level': self._find_support_level(prices),
            'resistance_level': self._find_resistance_level(prices),
            
            # Market regime
            'volatility_regime': self._classify_volatility_regime(prices),
            'liquidity_regime': self._classify_liquidity_regime(),
        }
        
        return analysis
    
    def _find_support_level(self, prices: List[float]) -> float:
        """Find approximate support level"""
        if len(prices) < 10:
            return min(prices) if prices else 0
        
        # Simple support: average of recent lows
        lows = []
        for i in range(2, len(prices) - 2):
            if prices[i] <= min(prices[i-2:i+3]):
                lows.append(prices[i])
        
        return np.mean(lows) if lows else min(prices)
    
    def _find_resistance_level(self, prices: List[float]) -> float:
        """Find approximate resistance level"""
        if len(prices) < 10:
            return max(prices) if prices else 0
        
        # Simple resistance: average of recent highs
        highs = []
        for i in range(2, len(prices) - 2):
            if prices[i] >= max(prices[i-2:i+3]):
                highs.append(prices[i])
        
        return np.mean(highs) if highs else max(prices)
    
    def _classify_volatility_regime(self, prices: List[float]) -> str:
        """Classify current volatility regime"""
        if len(prices) < 20:
            return 'unknown'
        
        volatility = np.std(prices) / np.mean(prices) * 100
        
        if volatility < 0.05:
            return 'low'
        elif volatility < 0.15:
            return 'normal'
        elif volatility < 0.30:
            return 'high'
        else:
            return 'extreme'
    
    def _classify_liquidity_regime(self) -> str:
        """Classify current liquidity regime"""
        if self.tick_velocity > 8:
            return 'high'
        elif self.tick_velocity > 4:
            return 'normal'
        else:
            return 'low'
    
    def get_connection_stats(self) -> Dict:
        """Get connection and performance statistics"""
        
        uptime = (datetime.now() - self.connection_start_time).total_seconds() if self.connection_start_time else 0
        
        return {
            'is_connected': self.is_running,
            'uptime_seconds': uptime,
            'ticks_processed': self.ticks_processed,
            'ticks_per_second': self.ticks_processed / uptime if uptime > 0 else 0,
            'buffer_size': len(self.tick_buffer),
            'queue_size': self.raw_tick_queue.qsize(),
            'last_tick_time': self.last_tick_time.isoformat() if self.last_tick_time else None,
            'data_source': 'ALPACA' if self.alpaca_stream else 'SIMULATION'
        }
    
    def stop_stream(self):
        """Stop the tick data stream"""
        
        logging.info("🛑 Stopping tick data stream...")
        
        self.is_running = False
        
        # Close Alpaca stream
        if self.alpaca_stream:
            try:
                self.alpaca_stream.close()
            except Exception as e:
                logging.error(f"Error closing Alpaca stream: {e}")
        
        # Wait for queue to empty
        try:
            self.raw_tick_queue.join()
        except Exception as e:
            logging.error(f"Error waiting for queue: {e}")
        
        stats = self.get_connection_stats()
        logging.info(f"✅ Stream stopped. Processed {stats['ticks_processed']} ticks in {stats['uptime_seconds']:.1f}s")


# Legacy compatibility class
class DataCollector(AdvancedTickCollector):
    """Legacy compatibility wrapper"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        super().__init__(symbol)
        logging.info("Using legacy DataCollector interface")
    
    def start_data_feed(self):
        """Legacy method - start with simulation"""
        self.stream_tick_data()
    
    def get_latest_price(self) -> Optional[float]:
        """Legacy method - get current price"""
        if self.tick_buffer:
            return self.tick_buffer[-1].price
        return None
    
    def get_price_change(self, periods: int = 5) -> float:
        """Legacy method - get price change percentage"""
        if len(self.tick_buffer) < periods:
            return 0.0
        
        current_price = self.tick_buffer[-1].price
        past_price = self.tick_buffer[-periods].price
        
        return ((current_price - past_price) / past_price) * 100 if past_price != 0 else 0.0
    
    def get_price_history(self, count: int = 10) -> List[Dict]:
        """Legacy method - get price history"""
        recent_ticks = list(self.tick_buffer)[-count:] if self.tick_buffer else []
        
        return [{
            'timestamp': tick.timestamp.isoformat(),
            'price': tick.price
        } for tick in recent_ticks]
    
    def is_data_available(self) -> bool:
        """Legacy method - check if data is available"""
        return self.is_running and len(self.tick_buffer) > 0
    
    def stop_data_feed(self):
        """Legacy method - stop data feed"""
        self.stop_stream()


# Factory function for easy initialization
def create_data_collector(symbol: str = "XAUUSD", enhanced: bool = True):
    """Factory function to create data collector"""
    
    if enhanced:
        return AdvancedTickCollector(symbol)
    else:
        return DataCollector(symbol)


if __name__ == "__main__":
    # Test the enhanced data collector
    collector = AdvancedTickCollector("XAUUSD")
    
    def on_tick(tick_data):
        print(f"Tick: {tick_data['price']:.4f} @ {tick_data['timestamp']}")
    
    def on_analytics(analytics):
        print(f"Analytics: Velocity={analytics.get('tick_velocity', 0):.1f}, "
              f"Price={analytics.get('current_price', 0):.4f}, "
              f"Spread={analytics.get('current_spread', 0):.3f}")
    
    collector.add_tick_callback(on_tick)
    collector.add_analytics_callback(on_analytics)
    
    try:
        collector.stream_tick_data()
        time.sleep(30)  # Run for 30 seconds
        
        # Print final stats
        stats = collector.get_connection_stats()
        print(f"\nFinal Stats: {stats}")
        
        analysis = collector.get_tick_analysis()
        print(f"Final Analysis: {analysis}")
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        collector.stop_stream()