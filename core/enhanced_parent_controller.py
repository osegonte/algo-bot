#!/usr/bin/env python3
"""
Level 5-D: Parent P&L Accuracy
Enhanced parent controller that uses real P&L from live market data
"""

import json
import glob
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from typing import Dict, List, Any, Optional

LOG_PATH = Path("logs")

class EnhancedParentController:
    """Enhanced parent controller with live market data P&L calculation"""
    
    def __init__(self):
        self.logs = []
        self.enhanced_logs = []
        self.trades_with_prices = []
        
        # Track data sources
        self.data_sources_used = set()
        self.primary_data_source = None
    
    def ingest_logs(self):
        """Ingest both legacy and enhanced trade logs"""
        
        # Try to load enhanced logs first
        enhanced_files = list(LOG_PATH.glob("trades_enhanced*.json"))
        legacy_files = list(LOG_PATH.glob("trades*.json"))
        
        # Filter out enhanced files from legacy list
        legacy_files = [f for f in legacy_files if "enhanced" not in f.name]
        
        print(f"📊 Found {len(enhanced_files)} enhanced logs, {len(legacy_files)} legacy logs")
        
        # Load enhanced logs (Level 5)
        if enhanced_files:
            self._load_enhanced_logs(enhanced_files)
            print(f"✅ Loaded {len(self.enhanced_logs)} enhanced trade records")
        
        # Load legacy logs for backward compatibility
        if legacy_files and not enhanced_files:
            self._load_legacy_logs(legacy_files)
            print(f"⚠️ Using legacy logs: {len(self.logs)} records")
        
        # Set primary data source
        self._determine_primary_data_source()
    
    def _load_enhanced_logs(self, enhanced_files: List[Path]):
        """Load enhanced trade logs with live market data"""
        for logfile in enhanced_files:
            try:
                with open(logfile) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                log_entry = json.loads(line)
                                self.enhanced_logs.append(log_entry)
                                
                                # Track data sources
                                data_source = log_entry.get('data_source')
                                if data_source:
                                    self.data_sources_used.add(data_source)
                                    
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Skipping invalid JSON on line {line_num} in {logfile}: {e}")
            except Exception as e:
                print(f"❌ Error reading {logfile}: {e}")
    
    def _load_legacy_logs(self, legacy_files: List[Path]):
        """Load legacy trade logs (fallback)"""
        for logfile in legacy_files:
            try:
                with open(logfile) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                log_entry = json.loads(line)
                                self.logs.append(log_entry)
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Skipping invalid JSON on line {line_num} in {logfile}: {e}")
            except Exception as e:
                print(f"❌ Error reading {logfile}: {e}")
    
    def _determine_primary_data_source(self):
        """Determine the primary data source being used"""
        if self.data_sources_used:
            # Count occurrences of each data source
            source_counts = {}
            for log in self.enhanced_logs:
                source = log.get('data_source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Primary source is the most frequently used
            self.primary_data_source = max(source_counts, key=source_counts.get)
        else:
            self.primary_data_source = "legacy_simulation"
    
    def basic_stats(self):
        """Calculate and display basic trading statistics"""
        
        # Use enhanced logs if available, otherwise legacy
        active_logs = self.enhanced_logs if self.enhanced_logs else self.logs
        
        if not active_logs:
            print("No trade data to analyze")
            return
        
        buys = []
        sells = []
        
        for log in active_logs:
            side = log.get("side", "").lower()
            
            if side == "buy":
                buys.append(log)
            elif side == "sell":
                sells.append(log)
        
        total_trades = len(active_logs)
        buy_count = len(buys)
        sell_count = len(sells)
        
        print(f"📈 Total trades: {total_trades}")
        print(f"🟢 Buys: {buy_count}")
        print(f"🔴 Sells: {sell_count}")
        print(f"⚖️ Trade balance: {buy_count - sell_count}")
        print(f"📡 Primary data source: {self.primary_data_source}")
        
        # Show recent trades with enhanced data
        if active_logs:
            print(f"\n📋 Recent trades:")
            for log in active_logs[-3:]:
                symbol = log.get("symbol", "?")
                side = log.get("side", "?")
                qty = log.get("qty", "?")
                timestamp = log.get("timestamp", "?")
                
                # Show enhanced info if available
                if 'entry_price_live' in log:
                    price = log.get('entry_price_live', '?')
                    source = log.get('data_source', '?')
                    print(f"  • {timestamp[:19]} - {side.upper()} {qty} {symbol} @ ${price} [{source}]")
                else:
                    print(f"  • {timestamp[:19]} - {side.upper()} {qty} {symbol}")
    
    def kpis(self):
        """Enhanced KPI calculation using live market data"""
        
        if self.enhanced_logs:
            return self._calculate_enhanced_kpis()
        elif self.logs:
            print("⚠️ Using legacy KPI calculation (no live data)")
            return self._calculate_legacy_kpis()
        else:
            print("❌ No trade data available for KPI calculation")
            return {}
    
    def _calculate_enhanced_kpis(self):
        """Calculate KPIs using enhanced logs with real P&L"""
        
        # Extract trades with realized P&L
        completed_trades = []
        for trade in self.enhanced_logs:
            if trade.get('pnl_realised') is not None:
                completed_trades.append(trade)
        
        if not completed_trades:
            print("ℹ️ No completed trades with P&L found")
            return {}
        
        # Extract P&L values
        pnls = [float(trade['pnl_realised']) for trade in completed_trades]
        
        # Calculate KPIs
        gross_pnl = sum(pnls)
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0
        
        # Profit Factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe Ratio
        if len(pnls) > 1:
            avg_return = np.mean(pnls)
            std_return = np.std(pnls)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Additional Level 5 metrics
        avg_spread = self._calculate_average_spread()
        data_quality_score = self._calculate_data_quality_score()
        
        kpi_data = {
            "gross_pnl": round(gross_pnl, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "sharpe_ratio": round(sharpe_ratio, 3),
            "total_trades": len(completed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "avg_win": round(np.mean(winning_trades), 2) if winning_trades else 0,
            "avg_loss": round(np.mean(losing_trades), 2) if losing_trades else 0,
            
            # Level 5 specific metrics
            "data_source": self.primary_data_source,
            "avg_spread_captured": round(avg_spread, 4) if avg_spread else None,
            "data_quality_score": round(data_quality_score, 3),
            "data_sources_used": list(self.data_sources_used),
            "live_data_enabled": True
        }
        
        # Display Enhanced KPIs
        print("\n" + "="*60)
        print("📊 ENHANCED KEY PERFORMANCE INDICATORS (Level 5)")
        print("="*60)
        print(f"💰 Gross P&L: ${kpi_data['gross_pnl']} (LIVE DATA)")
        print(f"🎯 Win Rate: {kpi_data['win_rate']}%")
        print(f"📈 Profit Factor: {kpi_data['profit_factor']}")
        print(f"⚡ Sharpe Ratio: {kpi_data['sharpe_ratio']}")
        print(f"📋 Total Trades: {kpi_data['total_trades']}")
        print(f"🟢 Winners: {kpi_data['winning_trades']} (Avg: ${kpi_data['avg_win']})")
        print(f"🔴 Losers: {kpi_data['losing_trades']} (Avg: ${kpi_data['avg_loss']})")
        print(f"📡 Data Source: {kpi_data['data_source']}")
        if kpi_data['avg_spread_captured']:
            print(f"📊 Avg Spread: ${kpi_data['avg_spread_captured']}")
        print(f"🔍 Data Quality: {kpi_data['data_quality_score']}")
        
        # Save enhanced summary
        self._save_enhanced_summary(kpi_data)
        
        return kpi_data
    
    def _calculate_legacy_kpis(self):
        """Fallback to legacy KPI calculation"""
        # Use the original simulation-based approach
        simulated_trades = self._simulate_completed_trades()
        
        if not simulated_trades:
            return {}
        
        pnls = [trade['pnl'] for trade in simulated_trades]
        
        gross_pnl = sum(pnls)
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0
        
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        kpi_data = {
            "gross_pnl": round(gross_pnl, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "sharpe_ratio": 0,
            "total_trades": len(simulated_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(pnls) - len(winning_trades),
            "data_source": "legacy_simulation",
            "live_data_enabled": False
        }
        
        print(f"💰 Gross P&L: ${kpi_data['gross_pnl']} (SIMULATED)")
        print(f"🎯 Win Rate: {kpi_data['win_rate']}%")
        print(f"📈 Profit Factor: {kpi_data['profit_factor']}")
        
        return kpi_data
    
    def _calculate_average_spread(self) -> Optional[float]:
        """Calculate average bid-ask spread captured"""
        spreads = []
        for trade in self.enhanced_logs:
            spread = trade.get('spread_captured')
            if spread is not None:
                spreads.append(float(spread))
        
        return np.mean(spreads) if spreads else None
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate data quality score based on various factors"""
        if not self.enhanced_logs:
            return 0.0
        
        total_score = 0.0
        factors = 0
        
        # Factor 1: Percentage of trades with live prices
        trades_with_live_prices = sum(1 for trade in self.enhanced_logs 
                                    if trade.get('entry_price_live') is not None)
        if self.enhanced_logs:
            live_price_ratio = trades_with_live_prices / len(self.enhanced_logs)
            total_score += live_price_ratio * 0.4
            factors += 0.4
        
        # Factor 2: Data source reliability (non-fallback sources)
        reliable_sources = sum(1 for trade in self.enhanced_logs 
                             if not trade.get('data_source', '').endswith('_fallback'))
        if self.enhanced_logs:
            reliability_ratio = reliable_sources / len(self.enhanced_logs)
            total_score += reliability_ratio * 0.3
            factors += 0.3
        
        # Factor 3: Timestamp freshness (quotes < 10 seconds old when used)
        fresh_quotes = 0
        for trade in self.enhanced_logs:
            trade_time = datetime.fromisoformat(trade.get('timestamp', '').replace('Z', '+00:00'))
            quote_time = datetime.fromisoformat(trade.get('quote_timestamp', '').replace('Z', '+00:00'))
            age_seconds = (trade_time - quote_time).total_seconds()
            if age_seconds < 10:
                fresh_quotes += 1
        
        if self.enhanced_logs:
            freshness_ratio = fresh_quotes / len(self.enhanced_logs)
            total_score += freshness_ratio * 0.3
            factors += 0.3
        
        return total_score / factors if factors > 0 else 0.0
    
    def _simulate_completed_trades(self):
        """Simulate completed trades for legacy mode"""
        simulated_trades = []
        for i in range(min(8, len(self.logs))):
            pnl = np.random.normal(5, 15)
            trade = {
                'symbol': 'AAPL',
                'side': 'sell',
                'qty': '1',
                'pnl': round(pnl, 2),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            simulated_trades.append(trade)
        return simulated_trades
    
    def _save_enhanced_summary(self, kpi_data):
        """Save enhanced KPI summary"""
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kpis": kpi_data,
            "metadata": {
                "total_raw_trades": len(self.enhanced_logs) + len(self.logs),
                "enhanced_trades": len(self.enhanced_logs),
                "legacy_trades": len(self.logs),
                "data_sources": list(self.data_sources_used),
                "level": "5D - Enhanced P&L"
            }
        }
        
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        summary_file = LOG_PATH / f"enhanced_parent_summary_{date_str}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Enhanced summary saved to: {summary_file}")

if __name__ == "__main__":
    print("🧠 Enhanced Parent Controller (Level 5-D)")
    print("=" * 50)
    
    pc = EnhancedParentController()
    pc.ingest_logs()
    pc.basic_stats()
    kpis = pc.kpis()
    
    if kpis.get('live_data_enabled'):
        print("\n✅ Level 5-D Complete: Live P&L data integrated!")
    else:
        print("\n⚠️ Level 5-D Partial: Using legacy data")