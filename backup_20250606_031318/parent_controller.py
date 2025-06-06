import json
import glob
from pathlib import Path
from datetime import datetime
import numpy as np

LOG_PATH = Path("logs")

class ParentController:
    def __init__(self):
        self.logs = []
        self.trades_with_prices = []

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
        
        # Enhance trades with simulated price data for KPI calculation
        # In production, this would come from actual price feeds
        self._enhance_with_price_data()

    def _enhance_with_price_data(self):
        """Add simulated price and P&L data for KPI calculations"""
        # Simulate realistic AAPL prices and some winning/losing trades
        base_price = 150.0
        
        for i, trade in enumerate(self.logs):
            # Simulate price movement
            price_change = np.random.normal(0, 2)  # Random price movement
            fill_price = base_price + price_change
            
            enhanced_trade = trade.copy()
            enhanced_trade['fill_price'] = round(fill_price, 2)
            
            # For sells, calculate P&L (simplified - assumes previous buy at different price)
            if trade['side'] == 'sell':
                # Simulate entry price (could be higher or lower)
                entry_price = base_price + np.random.normal(0, 3)
                pnl = (fill_price - entry_price) * int(trade['qty'])
                enhanced_trade['pnl'] = round(pnl, 2)
                enhanced_trade['entry_price'] = round(entry_price, 2)
            else:
                enhanced_trade['pnl'] = 0  # No P&L until sold
                
            self.trades_with_prices.append(enhanced_trade)
            base_price = fill_price  # Update base for next trade

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

    def kpis(self):
        """Level 0: Calculate and display key performance indicators"""
        if not self.trades_with_prices:
            print("❌ No enhanced trade data available for KPI calculation")
            return {}
            
        # Extract completed trades (sells with P&L)
        completed_trades = [t for t in self.trades_with_prices if t.get('pnl', 0) != 0]
        
        if not completed_trades:
            # Simulate some completed trades for demo
            print("ℹ️  No completed trades found, simulating data for KPI demo...")
            completed_trades = self._simulate_completed_trades()
        
        # Calculate KPIs
        pnls = [trade['pnl'] for trade in completed_trades]
        
        # 1. Gross P&L
        gross_pnl = sum(pnls)
        
        # 2. Win Rate
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0
        
        # 3. Profit Factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 4. Simple Sharpe Ratio (using daily returns)
        if len(pnls) > 1:
            avg_return = np.mean(pnls)
            std_return = np.std(pnls)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        kpi_data = {
            "gross_pnl": round(gross_pnl, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "sharpe_ratio": round(sharpe_ratio, 3),
            "total_trades": len(completed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(pnls) - len(winning_trades),
            "avg_win": round(np.mean(winning_trades), 2) if winning_trades else 0,
            "avg_loss": round(np.mean([pnl for pnl in pnls if pnl < 0]), 2) if any(pnl < 0 for pnl in pnls) else 0
        }
        
        # Display KPIs
        print("\n" + "="*50)
        print("📊 KEY PERFORMANCE INDICATORS")
        print("="*50)
        print(f"💰 Gross P&L: ${kpi_data['gross_pnl']}")
        print(f"🎯 Win Rate: {kpi_data['win_rate']}%")
        print(f"📈 Profit Factor: {kpi_data['profit_factor']}")
        print(f"⚡ Sharpe Ratio: {kpi_data['sharpe_ratio']}")
        print(f"📋 Total Trades: {kpi_data['total_trades']}")
        print(f"🟢 Winners: {kpi_data['winning_trades']} (Avg: ${kpi_data['avg_win']})")
        print(f"🔴 Losers: {kpi_data['losing_trades']} (Avg: ${kpi_data['avg_loss']})")
        
        # Save summary to file
        self._save_summary(kpi_data)
        
        return kpi_data

    def _simulate_completed_trades(self):
        """Simulate some completed trades for demo purposes"""
        simulated_trades = []
        for i in range(8):  # Create 8 demo trades
            pnl = np.random.normal(5, 15)  # Mean profit $5, std $15
            trade = {
                'symbol': 'AAPL',
                'side': 'sell',
                'qty': '1',
                'pnl': round(pnl, 2),
                'fill_price': round(150 + np.random.normal(0, 2), 2),
                'timestamp': datetime.utcnow().isoformat()
            }
            simulated_trades.append(trade)
        return simulated_trades

    def _save_summary(self, kpi_data):
        """Save KPI summary to JSON file"""
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "kpis": kpi_data,
            "metadata": {
                "total_raw_trades": len(self.logs),
                "completed_trades": kpi_data["total_trades"]
            }
        }
        
        # Create filename with date
        date_str = datetime.utcnow().strftime("%Y%m%d")
        summary_file = LOG_PATH / f"parent_summary_{date_str}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved to: {summary_file}")

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
    pc.kpis()  # Level 0 implementation
    pc.detailed_analysis()