import numpy as np

class MeanReversionStrategy:
    def __init__(self, window=20, std_threshold=2):
        self.window = window
        self.std_threshold = std_threshold
        
    def generate_signal(self, prices):
        if len(prices) < self.window:
            return None
            
        mean = np.mean(prices[-self.window:])
        std = np.std(prices[-self.window:])
        current_price = prices[-1]
        
        z_score = (current_price - mean) / std
        
        if z_score > self.std_threshold:
            return "sell"  # Price is too high, expect reversion
        elif z_score < -self.std_threshold:
            return "buy"   # Price is too low, expect reversion
            
        return None
