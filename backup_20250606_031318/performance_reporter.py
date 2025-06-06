import json
from pathlib import Path
from datetime import datetime

class PerformanceReporter:
    def __init__(self, log_path="logs/trades.json"):
        self.log_path = Path(log_path)
        
    def generate_report(self):
        if not self.log_path.exists():
            return {"error": "No trade logs found"}
            
        trades = []
        with open(self.log_path) as f:
            for line in f:
                trades.append(json.loads(line))
                
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t["side"] == "buy"])
        sell_trades = len([t for t in trades if t["side"] == "sell"])
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "trade_balance": buy_trades - sell_trades
        }
        
        return report
