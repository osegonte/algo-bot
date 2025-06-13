#!/usr/bin/env python3
"""
Enhanced Trade Execution Module for Alpaca Integration
Supports both paper and live trading with proper error handling
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class Trade:
    """Trade data structure"""
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    trade_id: str
    status: str
    profit_loss: float = 0.0
    commission: float = 0.0
    exchange: str = "ALPACA"

class TradeExecutor:
    """Enhanced trade executor with Alpaca integration"""
    
    def __init__(self, paper_trading: bool = True, alpaca_api_key: str = "", 
                 alpaca_secret_key: str = ""):
        self.paper_trading = paper_trading
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.api = None
        self.account_info = {"balance": 100000.0, "buying_power": 100000.0}  # Default sim values
        
        # Position tracking
        self.positions = {}
        self.open_orders = {}
        
        # Trading statistics
        self.total_trades = 0
        self.successful_trades = 0
        
        # Initialize API connection
        self._initialize_alpaca_api()
        
    def _initialize_alpaca_api(self):
        """Initialize Alpaca API connection"""
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            logging.warning("⚠️  No Alpaca credentials provided - using simulation mode")
            self.paper_trading = True
            return
        
        try:
            import alpaca_trade_api as tradeapi
            
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            
            self.api = tradeapi.REST(
                self.alpaca_api_key, 
                self.alpaca_secret_key, 
                base_url, 
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.account_info = {
                "balance": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "cash": float(account.cash)
            }
            
            mode = "📄 PAPER" if self.paper_trading else "💰 LIVE"
            logging.info(f"✅ Alpaca API connected - {mode} trading mode")
            logging.info(f"💰 Account balance: ${self.account_info['balance']:,.2f}")
            
        except ImportError:
            logging.error("❌ alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            self.api = None
        except Exception as e:
            logging.error(f"❌ Failed to connect to Alpaca API: {e}")
            self.api = None
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   price: Optional[float] = None, order_type: str = "market") -> Trade:
        """Place a buy or sell order"""
        
        trade_id = f"{symbol}_{side}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        try:
            if self.api and not self._is_simulation_mode():
                # Real Alpaca order
                alpaca_order = self._place_alpaca_order(symbol, side, quantity, price, order_type)
                
                if alpaca_order:
                    trade = Trade(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=float(alpaca_order.filled_avg_price or price or 0),
                        trade_id=alpaca_order.id,
                        status="filled" if alpaca_order.status == "filled" else "pending"
                    )
                    
                    if trade.status == "filled":
                        self.successful_trades += 1
                        self._update_positions(trade)
                    
                    self.total_trades += 1
                    return trade
                else:
                    # Failed order
                    return Trade(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price or 0,
                        trade_id=trade_id,
                        status="failed"
                    )
            else:
                # Simulation mode
                simulated_price = price or self._get_simulated_price(symbol)
                
                trade = Trade(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=simulated_price,
                    trade_id=trade_id,
                    status="filled"
                )
                
                # Update simulated account
                self._update_simulated_account(trade)
                self._update_positions(trade)
                
                self.successful_trades += 1
                self.total_trades += 1
                
                logging.info(f"✅ Simulated order filled: {side.upper()} {quantity} {symbol} @ ${simulated_price:.4f}")
                return trade
                
        except Exception as e:
            logging.error(f"❌ Order execution failed: {e}")
            return Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price or 0,
                trade_id=trade_id,
                status="failed"
            )
    
    def _place_alpaca_order(self, symbol: str, side: str, quantity: float, 
                           price: Optional[float], order_type: str):
        """Place order via Alpaca API"""
        try:
            # Convert crypto symbols to Alpaca format
            alpaca_symbol = self._convert_symbol_format(symbol)
            
            if order_type == "market":
                order = self.api.submit_order(
                    symbol=alpaca_symbol,
                    qty=quantity,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
            else:  # limit order
                order = self.api.submit_order(
                    symbol=alpaca_symbol,
                    qty=quantity,
                    side=side,
                    type='limit',
                    time_in_force='day',
                    limit_price=price
                )
            
            # Wait for order to fill (up to 5 seconds for market orders)
            if order_type == "market":
                for _ in range(50):  # 5 seconds max
                    updated_order = self.api.get_order(order.id)
                    if updated_order.status == 'filled':
                        return updated_order
                    time.sleep(0.1)
            
            return order
            
        except Exception as e:
            logging.error(f"Alpaca order error: {e}")
            return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol to Alpaca format"""
        symbol_map = {
            'XAUUSD': 'XAUUSD',  # XAU/USD (Gold spot)
            'BTCUSD': 'BTCUSD',
            'ETHUSD': 'ETHUSD',
            'AAPL': 'AAPL',
            'TSLA': 'TSLA'
        }
        return symbol_map.get(symbol, symbol)
    
    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for testing"""
        import random
        
        base_prices = {
            'XAUUSD': 2000.0,
            'BTCUSD': 45000.0,
            'ETHUSD': 3000.0,
            'AAPL': 180.0,
            'TSLA': 250.0
        }
        
        base = base_prices.get(symbol, 100.0)
        # Add small random variation
        variation = random.uniform(-0.001, 0.001)
        return round(base * (1 + variation), 4)
    
    def _update_simulated_account(self, trade: Trade):
        """Update simulated account balance"""
        trade_value = trade.quantity * trade.price
        
        if trade.side == 'buy':
            self.account_info['cash'] -= trade_value
            self.account_info['buying_power'] -= trade_value
        else:  # sell
            self.account_info['cash'] += trade_value
            self.account_info['buying_power'] += trade_value
        
        # Update portfolio value
        self.account_info['balance'] = self.account_info['cash'] + sum(
            pos['quantity'] * pos['avg_price'] 
            for pos in self.positions.values() 
            if pos['quantity'] > 0
        )
    
    def _update_positions(self, trade: Trade):
        """Update position tracking"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'side': None
            }
        
        current_pos = self.positions[symbol]
        
        if trade.side == 'buy':
            if current_pos['quantity'] <= 0:  # New long position or covering short
                current_pos['quantity'] += trade.quantity
                current_pos['avg_price'] = trade.price
                current_pos['side'] = 'long'
            else:  # Adding to long position
                total_value = (current_pos['quantity'] * current_pos['avg_price'] + 
                              trade.quantity * trade.price)
                current_pos['quantity'] += trade.quantity
                current_pos['avg_price'] = total_value / current_pos['quantity']
        
        else:  # sell
            if current_pos['quantity'] >= 0:  # New short position or closing long
                current_pos['quantity'] -= trade.quantity
                if current_pos['quantity'] < 0:
                    current_pos['side'] = 'short'
                    current_pos['avg_price'] = trade.price
                elif current_pos['quantity'] == 0:
                    current_pos['side'] = None
    
    def close_position(self, symbol: str, original_side: str, quantity: float, 
                      current_price: float) -> Trade:
        """Close an existing position"""
        
        # Determine opposite side
        close_side = 'sell' if original_side == 'buy' else 'buy'
        
        # Create closing trade
        close_trade_id = f"{symbol}_{close_side}_close_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        return self.place_order(
            symbol=symbol,
            side=close_side,
            quantity=quantity,
            price=current_price,
            order_type="market"
        )
    
    def get_position_info(self) -> Dict:
        """Get current account and position information"""
        try:
            if self.api and not self._is_simulation_mode():
                # Get real account info
                account = self.api.get_account()
                positions = self.api.list_positions()
                
                return {
                    'balance': float(account.portfolio_value),
                    'buying_power': float(account.buying_power),
                    'cash': float(account.cash),
                    'equity': float(account.equity),
                    'positions': {pos.symbol: {
                        'quantity': float(pos.qty),
                        'avg_price': float(pos.avg_cost),
                        'market_value': float(pos.market_value),
                        'unrealized_pnl': float(pos.unrealized_pnl)
                    } for pos in positions}
                }
            else:
                # Return simulated account info
                return {
                    'balance': self.account_info['balance'],
                    'buying_power': self.account_info['buying_power'],
                    'cash': self.account_info.get('cash', self.account_info['balance']),
                    'equity': self.account_info['balance'],
                    'positions': self.positions.copy()
                }
                
        except Exception as e:
            logging.error(f"Error getting position info: {e}")
            return self.account_info.copy()
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            if self.api and not self._is_simulation_mode():
                self.api.cancel_order(order_id)
                logging.info(f"✅ Order {order_id} cancelled")
                return True
            else:
                # Simulated cancellation
                if order_id in self.open_orders:
                    del self.open_orders[order_id]
                    logging.info(f"✅ Simulated order {order_id} cancelled")
                    return True
                return False
                
        except Exception as e:
            logging.error(f"❌ Failed to cancel order {order_id}: {e}")
            return False
    
    def get_open_orders(self) -> List[Dict]:
        """Get list of open orders"""
        try:
            if self.api and not self._is_simulation_mode():
                orders = self.api.list_orders(status='open')
                return [{
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': float(order.qty),
                    'price': float(order.limit_price) if order.limit_price else None,
                    'type': order.order_type,
                    'status': order.status,
                    'created_at': order.created_at
                } for order in orders]
            else:
                # Return simulated open orders
                return list(self.open_orders.values())
                
        except Exception as e:
            logging.error(f"Error getting open orders: {e}")
            return []
    
    def _is_simulation_mode(self) -> bool:
        """Check if running in simulation mode"""
        return not self.api or self.paper_trading
    
    def get_trading_stats(self) -> Dict:
        """Get trading execution statistics"""
        success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.total_trades - self.successful_trades,
            'success_rate': round(success_rate, 2),
            'mode': 'Paper Trading' if self.paper_trading else 'Live Trading',
            'api_connected': bool(self.api)
        }
    
    def check_market_hours(self, symbol: str) -> Dict:
        """Check if market is open for trading"""
        try:
            if self.api:
                clock = self.api.get_clock()
                return {
                    'is_open': clock.is_open,
                    'next_open': clock.next_open.isoformat() if clock.next_open else None,
                    'next_close': clock.next_close.isoformat() if clock.next_close else None,
                    'current_time': clock.timestamp.isoformat()
                }
            else:
                # Assume always open for simulation
                return {
                    'is_open': True,
                    'next_open': None,
                    'next_close': None,
                    'current_time': datetime.now().isoformat()
                }
                
        except Exception as e:
            logging.error(f"Error checking market hours: {e}")
            return {'is_open': True}  # Default to open
    
    def get_historical_trades(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get historical trades from Alpaca"""
        try:
            if self.api and not self._is_simulation_mode():
                # Get portfolio history
                portfolio_history = self.api.get_portfolio_history(
                    period='1D',
                    timeframe='1Min'
                )
                
                # This is a simplified version - you might want to enhance this
                return [{
                    'timestamp': timestamp.isoformat(),
                    'portfolio_value': value
                } for timestamp, value in zip(portfolio_history.timestamp, portfolio_history.equity)]
            else:
                return []  # No historical data in simulation mode
                
        except Exception as e:
            logging.error(f"Error getting historical trades: {e}")
            return []
    
    def calculate_position_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L for a position"""
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        if pos['quantity'] == 0:
            return 0.0
        
        if pos['side'] == 'long':
            return (current_price - pos['avg_price']) * pos['quantity']
        else:  # short
            return (pos['avg_price'] - current_price) * abs(pos['quantity'])
    
    def set_stop_loss(self, symbol: str, stop_price: float, quantity: float) -> bool:
        """Set a stop-loss order"""
        try:
            if self.api and not self._is_simulation_mode():
                # Determine side based on current position
                current_pos = self.positions.get(symbol, {})
                if current_pos.get('quantity', 0) > 0:
                    side = 'sell'  # Stop loss for long position
                elif current_pos.get('quantity', 0) < 0:
                    side = 'buy'   # Stop loss for short position
                else:
                    logging.warning("No position to set stop loss for")
                    return False
                
                alpaca_symbol = self._convert_symbol_format(symbol)
                
                order = self.api.submit_order(
                    symbol=alpaca_symbol,
                    qty=quantity,
                    side=side,
                    type='stop',
                    time_in_force='day',
                    stop_price=stop_price
                )
                
                logging.info(f"✅ Stop loss set for {symbol}: ${stop_price}")
                return True
            else:
                # Simulated stop loss
                logging.info(f"✅ Simulated stop loss set for {symbol}: ${stop_price}")
                return True
                
        except Exception as e:
            logging.error(f"❌ Failed to set stop loss: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources and close connections"""
        try:
            # Cancel any open orders
            open_orders = self.get_open_orders()
            for order in open_orders:
                self.cancel_order(order['id'])
            
            logging.info("✅ Trade executor cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


# Additional utility functions for enhanced trading

def calculate_position_size(account_balance: float, risk_percent: float, 
                          entry_price: float, stop_loss_price: float) -> float:
    """Calculate position size based on risk management"""
    risk_amount = account_balance * (risk_percent / 100)
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0
    
    position_size = risk_amount / price_risk
    return round(position_size, 4)


def validate_order_parameters(symbol: str, side: str, quantity: float, 
                             price: Optional[float] = None) -> bool:
    """Validate order parameters before execution"""
    if not symbol or len(symbol) < 2:
        logging.error("Invalid symbol")
        return False
    
    if side not in ['buy', 'sell']:
        logging.error("Invalid side - must be 'buy' or 'sell'")
        return False
    
    if quantity <= 0:
        logging.error("Invalid quantity - must be positive")
        return False
    
    if price is not None and price <= 0:
        logging.error("Invalid price - must be positive")
        return False
    
    return True