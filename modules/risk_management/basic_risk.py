class BasicRiskManager:
    def __init__(self, max_position_size=1000, max_daily_loss=500):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0
        
    def validate_trade(self, symbol, qty, side, current_price):
        position_value = qty * current_price
        
        if position_value > self.max_position_size:
            return False, "Position size exceeds maximum allowed"
            
        if self.daily_pnl < -self.max_daily_loss:
            return False, "Daily loss limit reached"
            
        return True, "Trade approved"
        
    def update_pnl(self, pnl):
        self.daily_pnl += pnl
