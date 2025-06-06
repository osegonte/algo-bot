import json
import glob
from pathlib import Path

LOG_PATH = Path("logs")

class ParentController:
    def __init__(self):
        self.logs = []

    def ingest_logs(self):
        """Ingest all trade logs from JSON files"""
        trade_files = list(LOG_PATH.glob("trades*.json"))
        
        if not trade_files:
            print("⚠️  No trade log files found")
            return
            
        for logfile in trade_files:
            try:
                with open(logfile) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                log_entry = json.loads(line)
                                self.logs.append(log_entry)
                            except json.JSONDecodeError as e:
                                print(f"⚠️  Skipping invalid JSON on line {line_num} in {logfile}: {e}")
            except Exception as e:
                print(f"❌ Error reading {logfile}: {e}")
        
        print(f"📊 Ingested {len(self.logs)} trade records")

    def basic_stats(self):
        """Calculate and display basic trading statistics"""
        if not self.logs:
            print("No trade data to analyze")
            return
            
        # Handle different log formats
        buys = []
        sells = []
        
        for log in self.logs:
            # Handle both old and new log formats
            side = log.get("side", "").lower()
            
            if side == "buy":
                buys.append(log)
            elif side == "sell":
                sells.append(log)
            elif side:  # Unknown side value
                print(f"⚠️  Unknown trade side: {side}")
        
        total_trades = len(self.logs)
        buy_count = len(buys)
        sell_count = len(sells)
        
        print(f"📈 Total trades: {total_trades}")
        print(f"🟢 Buys: {buy_count}")
        print(f"🔴 Sells: {sell_count}")
        print(f"⚖️  Trade balance: {buy_count - sell_count}")
        
        # Show recent trades
        if self.logs:
            print(f"\n📋 Recent trades:")
            for log in self.logs[-3:]:  # Last 3 trades
                symbol = log.get("symbol", "?")
                side = log.get("side", "?")
                qty = log.get("qty", "?")
                timestamp = log.get("timestamp", "?")
                print(f"  • {timestamp[:19]} - {side.upper()} {qty} {symbol}")

    def detailed_analysis(self):
        """More detailed analysis for Stage 2"""
        # This will be expanded in Stage 2
        symbols = set(log.get("symbol") for log in self.logs if log.get("symbol"))
        
        print(f"\n📊 Symbols traded: {', '.join(sorted(symbols))}")
        
        # Group by symbol
        by_symbol = {}
        for log in self.logs:
            symbol = log.get("symbol")
            if symbol:
                if symbol not in by_symbol:
                    by_symbol[symbol] = {"buys": 0, "sells": 0}
                
                side = log.get("side", "").lower()
                if side == "buy":
                    by_symbol[symbol]["buys"] += 1
                elif side == "sell":
                    by_symbol[symbol]["sells"] += 1
        
        for symbol, counts in by_symbol.items():
            print(f"  {symbol}: {counts['buys']} buys, {counts['sells']} sells")

if __name__ == "__main__":
    pc = ParentController()
    pc.ingest_logs()
    pc.basic_stats()
    pc.detailed_analysis()