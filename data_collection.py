#!/usr/bin/env python3
"""
Simple Data Collection for XAU/USD Tick Trading
"""

import time
import logging
import threading
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Callable


class SimpleTickData:
    """Simple tick data structure"""
    def __init__(self, price: float, size: int, timestamp: datetime):
        self.price = price
        self.size = size
        self.timestamp = timestamp
        self.bid = price - 0.05  # Simulated bid
        self.ask = price + 0.05  # Simulated ask
        self.spread = self.ask - self.bid


class SimpleDataCollector:
    """Simple tick data collector for XAU/USD"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.is_running = False
        self.tick_buffer = deque(maxlen=100)
        self.tick_callbacks = []
        
        # Basic analytics
        self.current_price = 2000.0
        self.tick_count = 0
        
        logging.info(f"✅ Data collector initialized for {symbol}")
    
    def add_tick_callback(self, callback: Callable):
        """Add callback function for tick updates"""
        self.tick_callbacks.append(callback)
    
    def start_data_feed(self, api_key: str = "", secret_key: str = ""):
        """Start data collection (simulation or real)"""
        self.is_running = True
        
        if api_key and secret_key:
            # Try to connect to real Alpaca data
            try:
                self._start_alpaca_feed(api_key, secret_key)
            except Exception as e:
                logging.warning(f"Alpaca connection failed: {e}")
                self._start_simulation()
        else:
            logging.info("Starting simulation mode")
            self._start_simulation()
    
    def _start_alpaca_feed(self, api_key: str, secret_key: str):
        """Start real Alpaca data feed"""
        try:
            import alpaca_trade_api as tradeapi
            
            # Initialize Alpaca stream
            self.alpaca_stream = tradeapi.StreamConn(
                api_key, secret_key,
                base_url='https://paper-api.alpaca.markets',
                data_feed='sip'
            )
            
            @self.alpaca_stream.on(f'^T\.{self.symbol}')
            async def on_tick(conn, channel, data):
                tick = SimpleTickData(
                    price=float(data.price),
                    size=int(data.size),
                    timestamp=datetime.now()
                )
                self._process_tick(tick)
            
            # Start stream in thread
            def run_stream():
                self.alpaca_stream.run()
            
            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()
            
            logging.info("✅ Alpaca data feed started")
            
        except ImportError:
            logging.error("alpaca-trade-api not installed")
            raise
    
    def _start_simulation(self):
        """Start simulated tick data"""
        def simulate_ticks():
            base_price = 2000.0
            
            while self.is_running:
                try:
                    # Generate realistic price movement
                    change = np.random.normal(0, 0.1)
                    base_price += change
                    
                    # Create tick
                    tick = SimpleTickData(
                        price=round(base_price, 2),
                        size=np.random.randint(1, 50),
                        timestamp=datetime.now()
                    )
                    
                    self._process_tick(tick)
                    
                    # Variable tick rate (3-10 per second)
                    time.sleep(np.random.uniform(0.1, 0.33))
                    
                except Exception as e:
                    logging.error(f"Simulation error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=simulate_ticks, daemon=True)
        thread.start()
        logging.info("🎮 Simulation started")
    
    def _process_tick(self, tick: SimpleTickData):
        """Process incoming tick data"""
        self.tick_buffer.append(tick)
        self.current_price = tick.price
        self.tick_count += 1
        
        # Call registered callbacks
        tick_data = {
            'price': tick.price,
            'size': tick.size,
            'timestamp': tick.timestamp,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.spread
        }
        
        for callback in self.tick_callbacks:
            try:
                callback(tick_data)
            except Exception as e:
                logging.error(f"Callback error: {e}")
    
    def get_current_price(self) -> float:
        """Get current price"""
        return self.current_price
    
    def get_price_change(self, periods: int = 5) -> float:
        """Get price change over periods"""
        if len(self.tick_buffer) < periods:
            return 0.0
        
        current = self.tick_buffer[-1].price
        past = self.tick_buffer[-periods].price
        
        return ((current - past) / past) * 100 if past != 0 else 0.0
    
    def get_market_analysis(self) -> Dict:
        """Get basic market analysis"""
        if len(self.tick_buffer) < 10:
            return {}
        
        prices = [tick.price for tick in self.tick_buffer]
        
        return {
            'current_price': prices[-1],
            'price_change_5': self.get_price_change(5),
            'price_volatility': np.std(prices) / np.mean(prices),
            'tick_count': len(self.tick_buffer),
            'price_range': max(prices) - min(prices),
            'momentum': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        }
    
    def stop_data_feed(self):
        """Stop data collection"""
        self.is_running = False
        logging.info("📡 Data feed stopped")


if __name__ == "__main__":
    # Test data collector
    collector = SimpleDataCollector()
    
    def on_tick(tick_data):
        print(f"Price: ${tick_data['price']:.2f}")
    
    collector.add_tick_callback(on_tick)
    collector.start_data_feed()
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop_data_feed()