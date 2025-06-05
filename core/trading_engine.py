from datetime import datetime
import json
import os
import alpaca_trade_api as tradeapi

class TradingEngine:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.account = self.api.get_account()

    def submit_order(self, symbol: str, qty: int, side: str, order_type: str = "market", time_in_force: str = "gtc"):
        order = self.api.submit_order(symbol=symbol, qty=qty, side=side, type=order_type, time_in_force=time_in_force)
        self._log_trade(order)
        return order

    def _log_trade(self, order):
        trade_log = {
            "symbol": order.symbol,
            "qty": order.qty,
            "side": order.side,
            "status": order.status,
            "filled_at": order.filled_at,
            "timestamp": datetime.utcnow().isoformat()
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/trades.json", "a") as f:
            f.write(json.dumps(trade_log) + "\n")
