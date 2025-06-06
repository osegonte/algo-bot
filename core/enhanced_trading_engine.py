#!/usr/bin/env python3
"""
Level 5-C: Trading Engine Hook-up
Enhanced trading engine that captures live quotes for P&L calculation
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import alpaca_trade_api as tradeapi

# Import our new market data components
from market_data_gateway import Quote, DataGatewayFactory
from price_stream_manager import PriceStreamManager

@dataclass
class EnhancedTradeLog:
    """Enhanced trade log with live market data"""
    symbol: str
    qty: str
    side: str
    status: str
    filled_at: Optional[str]
    timestamp: str
    
    # New Level 5 fields
    entry_price_live: float
    quote_timestamp: str
    data_source: str
    pnl_realised: Optional[float] = None
    exit_price_live: Optional[float] = None
    spread_captured: Optional[float] = None

class EnhancedTradingEngine:
    """Enhanced trading engine with live market data integration"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str, config: dict = None):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.account = self.api.get_account()
        
        # Market data integration
        self.config = config or {}
        self.price_stream = None
        
        # Position tracking for P&L calculation
        self.open_positions: Dict[str, Dict] = {}
        
        # Initialize market data
        self._setup_market_data()
    
    def _setup_market_data(self):
        """Setup market data streaming"""
        try:
            self.price_stream = PriceStreamManager(self.config)
            print("📡 Market data system initialized")
        except Exception as e:
            print(f"⚠️ Market data setup failed: {e}")
            self.price_stream = None
    
    async def start_market_data(self):
        """Start market data streaming"""
        if self.price_stream:
            success = await self.price_stream.start_streaming()
            if success:
                print("✅ Market data stream started")
                return True
        return False
    
    def stop_market_data(self):
        """Stop market data streaming"""
        if self.price_stream:
            self.price_stream.stop_streaming()
            print("⏹️ Market data stream stopped")
    
    async def submit_order_with_live_data(self, symbol: str, qty: int, side: str, 
                                        order_type: str = "market", time_in_force: str = "gtc"):
        """Submit order and capture live market data"""
        
        # Get live quote before submitting order
        live_quote = await self._get_live_quote(symbol)
        
        # Submit the order
        order = self.api.submit_order(
            symbol=symbol, 
            qty=qty, 
            side=side, 
            type=order_type, 
            time_in_force=time_in_force
        )
        
        # Calculate entry price and P&L if this is a closing trade
        entry_price = self._get_entry_price_from_quote(live_quote, side)
        pnl_realised = None
        exit_price = None
        spread_captured = None
        
        if side == "sell" and symbol in self.open_positions:
            # This is a closing trade - calculate P&L
            position = self.open_positions[symbol]
            exit_price = entry_price
            
            # Calculate P&L: (exit_price - entry_price) * qty
            pnl_realised = (exit_price - position['entry_price']) * int(qty)
            
            # Calculate spread captured
            spread_captured = live_quote.ask - live_quote.bid
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            print(f"💰 P&L Realised: ${pnl_realised:.2f} on {qty} {symbol}")
            
        elif side == "buy":
            # This is an opening trade - track position
            self.open_positions[symbol] = {
                'entry_price': entry_price,
                'qty': int(qty),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Create enhanced trade log
        enhanced_log = EnhancedTradeLog(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            status=order.status,
            filled_at=order.filled_at,
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry_price_live=entry_price,
            quote_timestamp=live_quote.timestamp.isoformat(),
            data_source=live_quote.source,
            pnl_realised=pnl_realised,
            exit_price_live=exit_price,
            spread_captured=spread_captured
        )
        
        # Log the enhanced trade
        self._log_enhanced_trade(enhanced_log)
        
        return order, enhanced_log
    
    async def _get_live_quote(self, symbol: str) -> Quote:
        """Get live quote for symbol"""
        if self.price_stream:
            # Try to get from stream first
            quote = self.price_stream.get_latest_quote(symbol)
            if quote:
                return quote
        
        # Fallback: create gateway and get quote directly
        gateway = DataGatewayFactory.create_gateway(symbol, self.config)
        await gateway.connect()
        quote = await gateway.get_live_quote(symbol)
        await gateway.disconnect()
        
        return quote
    
    def _get_entry_price_from_quote(self, quote: Quote, side: str) -> float:
        """Get appropriate entry price based on trade side"""
        if side == "buy":
            # For buys, we pay the ask price
            return quote.ask
        else:
            # For sells, we receive the bid price
            return quote.bid
    
    def _log_enhanced_trade(self, trade_log: EnhancedTradeLog):
        """Log enhanced trade with live market data"""
        os.makedirs("logs", exist_ok=True)
        
        # Convert to dict for JSON serialization
        log_dict = asdict(trade_log)
        
        # Write to enhanced trades log
        with open("logs/trades_enhanced.json", "a") as f:
            f.write(json.dumps(log_dict) + "\n")
        
        # Also write to original trades.json for backward compatibility
        legacy_log = {
            "symbol": trade_log.symbol,
            "qty": trade_log.qty,
            "side": trade_log.side,
            "status": trade_log.status,
            "filled_at": trade_log.filled_at,
            "timestamp": trade_log.timestamp
        }
        
        with open("logs/trades.json", "a") as f:
            f.write(json.dumps(legacy_log) + "\n")
    
    def get_open_positions(self) -> Dict[str, Dict]:
        """Get current open positions"""
        return self.open_positions.copy()
    
    def get_unrealized_pnl(self) -> Dict[str, float]:
        """Calculate unrealized P&L for open positions"""
        unrealized_pnl = {}
        
        for symbol, position in self.open_positions.items():
            if self.price_stream:
                current_quote = self.price_stream.get_latest_quote(symbol)
                if current_quote:
                    # Calculate unrealized P&L
                    current_price = current_quote.bid  # Use bid for conservative estimate
                    pnl = (current_price - position['entry_price']) * position['qty']
                    unrealized_pnl[symbol] = pnl
        
        return unrealized_pnl
    
    # Backward compatibility method
    def submit_order(self, symbol: str, qty: int, side: str, order_type: str = "market", time_in_force: str = "gtc"):
        """Legacy submit_order method for backward compatibility"""
        import asyncio
        
        # Run the async version
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            order, enhanced_log = loop.run_until_complete(
                self.submit_order_with_live_data(symbol, qty, side, order_type, time_in_force)
            )
            return order
        finally:
            loop.close()

class TradingPerformanceTracker:
    """Track trading performance with live data"""
    
    def __init__(self, log_file: str = "logs/trades_enhanced.json"):
        self.log_file = log_file
        
    def get_realized_pnl_summary(self) -> Dict[str, Any]:
        """Get summary of realized P&L from enhanced logs"""
        if not os.path.exists(self.log_file):
            return {"error": "No enhanced trade logs found"}
        
        total_pnl = 0
        winning_trades = 0
        losing_trades = 0
        total_trades = 0
        trades_by_symbol = {}
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    
                    if trade.get('pnl_realised') is not None:
                        pnl = float(trade['pnl_realised'])
                        total_pnl += pnl
                        total_trades += 1
                        
                        if pnl > 0:
                            winning_trades += 1
                        elif pnl < 0:
                            losing_trades += 1
                        
                        # Track by symbol
                        symbol = trade['symbol']
                        if symbol not in trades_by_symbol:
                            trades_by_symbol[symbol] = {
                                'total_pnl': 0,
                                'trades': 0,
                                'data_sources': set()
                            }
                        
                        trades_by_symbol[symbol]['total_pnl'] += pnl
                        trades_by_symbol[symbol]['trades'] += 1
                        trades_by_symbol[symbol]['data_sources'].add(trade.get('data_source', 'unknown'))
                
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Convert sets to lists for JSON serialization
        for symbol_data in trades_by_symbol.values():
            symbol_data['data_sources'] = list(symbol_data['data_sources'])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "total_realized_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 1),
            "avg_trade": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
            "trades_by_symbol": trades_by_symbol,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

# Demo function
async def demo_enhanced_trading():
    """Demo the enhanced trading engine"""
    
    config = {
        "alpaca": {
            "api_key": "PKKJRF8U4QTYWMAVLYUI",
            "api_secret": "4kGChcW8cqkxasJfvyVMYfGRKbPqwNcx8MzM26ws",
            "base_url": "https://paper-api.alpaca.markets"
        }
    }
    
    print("🚀 Enhanced Trading Engine Demo")
    print("=" * 40)
    
    # Create enhanced trading engine
    engine = EnhancedTradingEngine(
        config["alpaca"]["api_key"],
        config["alpaca"]["api_secret"],
        config["alpaca"]["base_url"],
        config
    )
    
    # Start market data
    await engine.start_market_data()
    
    try:
        # Simulate a buy order
        print("📈 Executing BUY order...")
        buy_order, buy_log = await engine.submit_order_with_live_data("AAPL", 1, "buy")
        print(f"   Entry Price: ${buy_log.entry_price_live:.2f}")
        print(f"   Data Source: {buy_log.data_source}")
        
        # Wait a moment for price movement
        import asyncio
        await asyncio.sleep(3)
        
        # Simulate a sell order
        print("📉 Executing SELL order...")
        sell_order, sell_log = await engine.submit_order_with_live_data("AAPL", 1, "sell")
        print(f"   Exit Price: ${sell_log.exit_price_live:.2f}")
        print(f"   P&L Realized: ${sell_log.pnl_realised:.2f}")
        print(f"   Spread: ${sell_log.spread_captured:.4f}")
        
        # Show performance summary
        tracker = TradingPerformanceTracker()
        summary = tracker.get_realized_pnl_summary()
        print(f"\n📊 Performance Summary:")
        print(f"   Total P&L: ${summary.get('total_realized_pnl', 0):.2f}")
        print(f"   Win Rate: {summary.get('win_rate', 0):.1f}%")
        
    finally:
        engine.stop_market_data()
    
    print("✅ Enhanced trading demo complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_enhanced_trading())