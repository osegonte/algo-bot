import json
from pathlib import Path

class RiskReporter:
    def __init__(self, log_path="logs/trades.json"):
        self.log_path = Path(log_path)
        
    def risk_metrics(self):
        if not self.log_path.exists():
            return {"error": "No trade logs found"}
            
        trades = []
        with open(self.log_path) as f:
            for line in f:
                trades.append(json.loads(line))
                
        symbols = list(set([t["symbol"] for t in trades]))
        max_position = max([int(t["qty"]) for t in trades]) if trades else 0
        
        return {
            "unique_symbols": len(symbols),
            "symbols_traded": symbols,
            "max_position_size": max_position,
            "total_exposure": sum([int(t["qty"]) for t in trades])
        }
