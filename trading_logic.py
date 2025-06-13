#!/usr/bin/env python3
"""Trading Logic Module - Simplified"""

import logging
from typing import Optional, Dict

class TradingLogic:
    def __init__(self, tick_threshold: float = 0.01, profit_target: float = 0.02, 
                 stop_loss: float = 0.01, quantity: float = 0.001):
        self.tick_threshold = tick_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.quantity = quantity
        self.position = None
        self.entry_price = None
        self.entry_time = None
        
    def analyze_market(self, current_price: float, price_change: float) -> str:
        if not current_price:
            return 'hold'
        
        if self.position and self.entry_price:
            if self._check_exit_conditions(current_price):
                return 'close'
        
        if not self.position:
            entry_signal = self._check_entry_conditions(price_change)
            if entry_signal:
                return entry_signal
        
        return 'hold'
    
    def _check_exit_conditions(self, current_price: float) -> bool:
        if not self.entry_price:
            return False
            
        pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
        
        if self.position == 'long':
            if pnl_percent >= self.profit_target:
                logging.info(f"Long position profit target hit: {pnl_percent:.3f}%")
                return True
            elif pnl_percent <= -self.stop_loss:
                logging.info(f"Long position stop loss triggered: {pnl_percent:.3f}%")
                return True
        elif self.position == 'short':
            if pnl_percent <= -self.profit_target:
                logging.info(f"Short position profit target hit: {pnl_percent:.3f}%")
                return True
            elif pnl_percent >= self.stop_loss:
                logging.info(f"Short position stop loss triggered: {pnl_percent:.3f}%")
                return True
        
        return False
    
    def _check_entry_conditions(self, price_change: float) -> Optional[str]:
        if price_change > self.tick_threshold:
            logging.info(f"Long entry signal: price change {price_change:.3f}%")
            return 'buy'
        elif price_change < -self.tick_threshold:
            logging.info(f"Short entry signal: price change {price_change:.3f}%")
            return 'sell'
        return None
    
    def update_position(self, action: str, price: float, timestamp: str = None):
        if action == 'buy':
            self.position = 'long'
            self.entry_price = price
            self.entry_time = timestamp
            logging.info(f"Position updated: LONG at {price}")
        elif action == 'sell':
            self.position = 'short'
            self.entry_price = price
            self.entry_time = timestamp
            logging.info(f"Position updated: SHORT at {price}")
        elif action == 'close':
            logging.info(f"Position closed: {self.position} from {self.entry_price} to {price}")
            self.position = None
            self.entry_price = None
            self.entry_time = None
    
    def get_position_info(self) -> Dict:
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'quantity': self.quantity
        }
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        if not self.position or not self.entry_price:
            return 0.0
        
        if self.position == 'long':
            return (current_price - self.entry_price) * self.quantity
        elif self.position == 'short':
            return (self.entry_price - current_price) * self.quantity
        return 0.0
    
    def reset_position(self):
        self.position = None
        self.entry_price = None
        self.entry_time = None
        logging.warning("Position manually reset")
