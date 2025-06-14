#!/usr/bin/env python3
"""
Simple Trade Execution for XAU/USD
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Optional


class SimpleTrade:
    """Simple trade data structure"""
    def __init__(self, symbol: str, side: str, quantity: float, price: float, status: str = "filled"):
        self.timestamp = datetime.now().isoformat()
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.trade_id = f"{symbol}_{side}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.status = status


class SimpleTradeExecutor:
    """Simple trade executor for XAU/USD"""
    
    def __init__(self, paper_trading: bool = True, api_key: str = "", secret_key: str = ""):
        self.paper_trading = paper_trading
        self.api_key = api_key
        self.secret_key = secret_key
        self.api = None
        
        # Simulated account
        self.account_balance = 100000.0
        self.cash = 100000.0
        self.positions = {}
        
        # Trading stats
        self.total_trades = 0
        self.successful_trades = 0
        
        # Initialize API if credentials provided
        if api_key and secret_key:
            self._init_alpaca_api()
        else:
            logging.info("🎮 Running in simulation mode")
    
    def _init_alpaca_api(self):
        """Initialize Alpaca API connection"""
        try:
            import alpaca_trade_api as tradeapi
            
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            
            self.api = tradeapi.REST(self.api_key, self.secret_key, base_url, api_version='v2')
            
            # Test connection
            account = self.api.get_account()
            self.account_balance = float(account.portfolio_value)
            self.cash = float(account.cash)
            
            mode = "📄 PAPER" if self.paper_trading else "💰 LIVE"
            logging.info(f"✅ Alpaca connected - {mode} mode")
            logging.info(f"💰 Account balance: ${self.account_balance:,.2f}")
            
        except ImportError:
            logging.warning("alpaca-trade-api not installed - using simulation")
            self.api = None
        except Exception as e:
            logging.error(f"Alpaca connection failed: {e}")
            self.api = None
    
    def place_order(self, symbol: str, side: str, quantity: float, current_price: float) -> SimpleTrade:
        """Place a market order"""
        
        self.total_trades += 1
        
        try:
            if self.api:
                # Real Alpaca order
                return self._place_alpaca_order(symbol, side, quantity, current_price)
            else:
                # Simulated order
                return self._place_simulated_order(symbol, side, quantity, current_price)
                
        except Exception as e:
            logging.error(f"Order failed: {e}")
            return SimpleTrade(symbol, side, quantity, current_price, "failed")
    
    def _place_alpaca_order(self, symbol: str, side: str, quantity: float, current_price: float) -> SimpleTrade:
        """Place order via Alpaca API"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            # Wait briefly for fill
            for _ in range(10):  # 1 second max
                updated_order = self.api.get_order(order.id)
                if updated_order.status == 'filled':
                    fill_price = float(updated_order.filled_avg_price)
                    self.successful_trades += 1
                    self._update_positions(symbol, side, quantity, fill_price)
                    
                    logging.info(f"✅ Alpaca order filled: {side.upper()} {quantity} {symbol} @ ${fill_price:.2f}")
                    return SimpleTrade(symbol, side, quantity, fill_price, "filled")
                
                time.sleep(0.1)
            
            # Order not filled immediately
            logging.warning(f"⚠️ Order not filled immediately: {order.id}")
            return SimpleTrade(symbol, side, quantity, current_price, "pending")
            
        except Exception as e:
            logging.error(f"Alpaca order error: {e}")
            return SimpleTrade(symbol, side, quantity, current_price, "failed")
    
    def _place_simulated_order(self, symbol: str, side: str, quantity: float, current_price: float) -> SimpleTrade:
        """Place simulated order"""
        
        # Add small random slippage
        import random
        slippage = random.uniform(-0.05, 0.05)  # ±$0.05 slippage
        fill_price = current_price + slippage
        
        # Update simulated account
        trade_value = quantity * fill_price
        
        if side == 'buy':
            self.cash -= trade_value
        else:  # sell
            self.cash += trade_value
        
        self._update_positions(symbol, side, quantity, fill_price)
        self.successful_trades += 1
        
        logging.info(f"✅ Simulated order: {side.upper()} {quantity} {symbol} @ ${fill_price:.2f}")
        return SimpleTrade(symbol, side, quantity, fill_price, "filled")
    
    def _update_positions(self, symbol: str, side: str, quantity: float, price: float):
        """Update position tracking"""
        
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        
        pos = self.positions[symbol]
        
        if side == 'buy':
            # Add to long position
            total_value = (pos['quantity'] * pos['avg_price']) + (quantity * price)
            pos['quantity'] += quantity
            pos['avg_price'] = total_value / pos['quantity'] if pos['quantity'] > 0 else 0
            
        else:  # sell
            # Reduce position
            pos['quantity'] -= quantity
            if pos['quantity'] <= 0:
                pos['quantity'] = 0
                pos['avg_price'] = 0
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        
        if self.api:
            try:
                account = self.api.get_account()
                return {
                    'balance': float(account.portfolio_value),
                    'cash': float(account.cash),
                    'buying_power': float(account.buying_power)
                }
            except:
                pass
        
        # Return simulated account
        return {
            'balance': self.account_balance,
            'cash': self.cash,
            'buying_power': self.cash
        }
    
    def get_position(self, symbol: str) -> Dict:
        """Get position for symbol"""
        
        if self.api:
            try:
                positions = self.api.list_positions()
                for pos in positions:
                    if pos.symbol == symbol:
                        return {
                            'quantity': float(pos.qty),
                            'avg_price': float(pos.avg_cost),
                            'market_value': float(pos.market_value),
                            'unrealized_pnl': float(pos.unrealized_pnl)
                        }
            except:
                pass
        
        # Return simulated position
        pos = self.positions.get(symbol, {'quantity': 0, 'avg_price': 0})
        return {
            'quantity': pos['quantity'],
            'avg_price': pos['avg_price'],
            'market_value': 0,  # Would need current price to calculate
            'unrealized_pnl': 0
        }
    
    def get_trading_stats(self) -> Dict:
        """Get trading statistics"""
        
        success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': success_rate,
            'mode': 'Paper' if self.paper_trading else 'Live',
            'api_connected': bool(self.api)
        }
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        
        if self.api:
            try:
                clock = self.api.get_clock()
                return clock.is_open
            except:
                pass
        
        # Assume always open for simulation (XAU/USD trades 24/5)
        return True


if __name__ == "__main__":
    # Test trade executor
    executor = SimpleTradeExecutor()
    
    # Test simulated trade
    trade = executor.place_order("XAUUSD", "buy", 0.1, 2000.50)
    print(f"Trade: {trade.side} {trade.quantity} {trade.symbol} @ ${trade.price:.2f}")
    print(f"Status: {trade.status}")
    print(f"Trade ID: {trade.trade_id}")
    
    # Check account
    account = executor.get_account_info()
    print(f"Account: ${account['balance']:,.2f}")
    
    # Check position
    position = executor.get_position("XAUUSD")
    print(f"Position: {position}")
    
    # Trading stats
    stats = executor.get_trading_stats()
    print(f"Stats: {stats}")