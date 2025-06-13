#!/usr/bin/env python3
"""Data Collection Module - Simplified for quick start"""

import json
import logging
import queue
import websocket
from datetime import datetime
from threading import Thread
from typing import Optional, List, Dict

class DataCollector:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.current_price = None
        self.price_history = []
        self.data_queue = queue.Queue()
        self.is_running = False
        self.ws = None
        
    def start_data_feed(self):
        self.is_running = True
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'p' in data:
                    price = float(data['p'])
                    self.current_price = price
                    price_data = {
                        'timestamp': datetime.now().isoformat(),
                        'price': price
                    }
                    self.price_history.append(price_data)
                    if len(self.price_history) > 100:
                        self.price_history.pop(0)
                    self.data_queue.put(price)
            except Exception as e:
                logging.error(f"Error processing market data: {e}")
        
        def on_error(ws, error):
            logging.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logging.info("WebSocket connection closed")
            self.is_running = False
        
        def on_open(ws):
            logging.info(f"WebSocket connection opened for {self.symbol}")
        
        # Use crypto symbol for demo data
        crypto_symbol = "btcusdt" if "BTC" in self.symbol.upper() else "ethusdt"
        ws_url = f"wss://stream.binance.com:9443/ws/{crypto_symbol}@ticker"
        
        def run_websocket():
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            self.ws.run_forever()
        
        ws_thread = Thread(target=run_websocket)
        ws_thread.daemon = True
        ws_thread.start()
        
        logging.info(f"Data feed started for {self.symbol}")
    
    def get_latest_price(self) -> Optional[float]:
        return self.current_price
    
    def get_price_change(self, periods: int = 5) -> float:
        if len(self.price_history) < periods:
            return 0.0
        current_price = self.price_history[-1]['price']
        past_price = self.price_history[-periods]['price']
        return ((current_price - past_price) / past_price) * 100
    
    def get_price_history(self, count: int = 10) -> List[Dict]:
        return self.price_history[-count:] if self.price_history else []
    
    def is_data_available(self) -> bool:
        return self.current_price is not None and self.is_running
    
    def stop_data_feed(self):
        self.is_running = False
        if self.ws:
            self.ws.close()
        logging.info("Data feed stopped")
