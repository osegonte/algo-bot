import pandas as pd

class BreakoutStrategy:
    def __init__(self, lookback_period=20, breakout_threshold=0.02):
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        
    def should_trade(self, price_data):
        if len(price_data) < self.lookback_period:
            return False, None
            
        recent_high = price_data[-self.lookback_period:].max()
        recent_low = price_data[-self.lookback_period:].min()
        current_price = price_data[-1]
        
        if current_price > recent_high * (1 + self.breakout_threshold):
            return True, "buy"
        elif current_price < recent_low * (1 - self.breakout_threshold):
            return True, "sell"
            
        return False, None
