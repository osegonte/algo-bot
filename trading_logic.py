#!/usr/bin/env python3
"""
Simple Trading Logic for XAU/USD Scalping
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional


class TradingSignal:
    """Simple trading signal"""
    def __init__(self, signal_type: str, confidence: float, reasoning: str):
        self.signal_type = signal_type  # 'buy', 'sell', 'hold', 'close'
        self.confidence = confidence    # 0.0 to 1.0
        self.reasoning = reasoning


class SimpleTradingLogic:
    """Simple trading logic for XAU/USD"""
    
    def __init__(self, profit_target_ticks: int = 4, stop_loss_ticks: int = 2, 
                 tick_size: float = 0.10, min_confidence: float = 0.65):
        
        # Strategy parameters
        self.profit_target_ticks = profit_target_ticks
        self.stop_loss_ticks = stop_loss_ticks
        self.tick_size = tick_size
        self.min_confidence = min_confidence
        
        # Position tracking
        self.position = None      # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.max_position_time = 60  # seconds
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        
        logging.info(f"✅ Trading logic initialized - Target: {profit_target_ticks} ticks, Stop: {stop_loss_ticks} ticks")
    
    def evaluate_tick(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Main method to evaluate tick and generate signals"""
        
        # Check exit conditions first if in position
        if self.position:
            exit_signal = self._check_exit_conditions(tick_data, market_analysis)
            if exit_signal.signal_type == 'close':
                return exit_signal
        
        # Check entry conditions if no position
        if not self.position:
            return self._check_entry_conditions(tick_data, market_analysis)
        
        return TradingSignal('hold', 0.0, 'No action needed')
    
    def _check_entry_conditions(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Check for entry signals"""
        
        current_price = tick_data.get('price', 0)
        spread = tick_data.get('spread', 0)
        momentum = market_analysis.get('momentum', 0)
        volatility = market_analysis.get('price_volatility', 0)
        price_change = market_analysis.get('price_change_5', 0)
        
        # Basic filters
        if spread > 0.15:  # Spread too wide
            return TradingSignal('hold', 0.0, 'Spread too wide')
        
        if volatility > 0.3:  # Too volatile
            return TradingSignal('hold', 0.0, 'Volatility too high')
        
        # Simple momentum strategy
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Bullish momentum
        if momentum > 0.0005 and price_change > 0.02:  # 0.05% momentum, 0.02% recent change
            confidence = min(0.85, 0.5 + momentum * 1000 + abs(price_change) * 10)
            signal_type = 'buy'
            reasoning = f"Bullish momentum: {momentum:.4f}, change: {price_change:.2f}%"
        
        # Bearish momentum
        elif momentum < -0.0005 and price_change < -0.02:
            confidence = min(0.85, 0.5 + abs(momentum) * 1000 + abs(price_change) * 10)
            signal_type = 'sell'
            reasoning = f"Bearish momentum: {momentum:.4f}, change: {price_change:.2f}%"
        
        # Apply confidence filter
        if confidence < self.min_confidence:
            return TradingSignal('hold', confidence, f"Confidence too low: {confidence:.2f}")
        
        return TradingSignal(signal_type, confidence, reasoning)
    
    def _check_exit_conditions(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Check for exit conditions"""
        
        if not self.position or not self.entry_price:
            return TradingSignal('hold', 0.0, 'No position')
        
        current_price = tick_data.get('price', 0)
        current_time = tick_data.get('timestamp', datetime.now())
        
        # Calculate P&L in ticks
        if self.position == 'long':
            pnl_ticks = (current_price - self.entry_price) / self.tick_size
        else:  # short
            pnl_ticks = (self.entry_price - current_price) / self.tick_size
        
        # Time-based exit
        if self.entry_time:
            time_in_position = (current_time - self.entry_time).total_seconds()
            if time_in_position > self.max_position_time:
                return TradingSignal('close', 1.0, f'Time exit: {time_in_position:.0f}s')
        
        # Profit target
        if pnl_ticks >= self.profit_target_ticks:
            return TradingSignal('close', 1.0, f'Profit target: +{pnl_ticks:.1f} ticks')
        
        # Stop loss
        if pnl_ticks <= -self.stop_loss_ticks:
            return TradingSignal('close', 1.0, f'Stop loss: {pnl_ticks:.1f} ticks')
        
        return TradingSignal('hold', 0.0, f'Hold: {pnl_ticks:+.1f} ticks')
    
    def update_position(self, action: str, price: float, timestamp: str = None):
        """Update position state"""
        
        if action in ['buy', 'sell']:
            self.position = 'long' if action == 'buy' else 'short'
            self.entry_price = price
            self.entry_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            
            logging.info(f"Position opened: {self.position.upper()} @ ${price:.2f}")
            
        elif action == 'close':
            if self.position and self.entry_price:
                # Calculate final P&L
                if self.position == 'long':
                    pnl_ticks = (price - self.entry_price) / self.tick_size
                else:
                    pnl_ticks = (self.entry_price - price) / self.tick_size
                
                # Update performance
                self.total_signals += 1
                if pnl_ticks > 0:
                    self.successful_signals += 1
                
                logging.info(f"Position closed: {pnl_ticks:+.1f} ticks (${pnl_ticks * self.tick_size:+.2f})")
            
            # Reset position
            self.position = None
            self.entry_price = None
            self.entry_time = None
    
    def get_position_info(self) -> Dict:
        """Get current position information"""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
        }
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L in USD"""
        
        if not self.position or not self.entry_price:
            return 0.0
        
        if self.position == 'long':
            return current_price - self.entry_price
        else:  # short
            return self.entry_price - current_price
    
    def calculate_unrealized_pnl_ticks(self, current_price: float) -> float:
        """Calculate unrealized P&L in ticks"""
        pnl_usd = self.calculate_unrealized_pnl(current_price)
        return pnl_usd / self.tick_size
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        win_rate = (self.successful_signals / self.total_signals * 100) if self.total_signals > 0 else 0
        
        return {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'win_rate': win_rate,
            'current_position': self.position
        }


if __name__ == "__main__":
    # Test trading logic
    logic = SimpleTradingLogic()
    
    # Sample tick data
    tick_data = {
        'price': 2000.50,
        'spread': 0.10,
        'timestamp': datetime.now()
    }
    
    # Sample market analysis
    market_analysis = {
        'momentum': 0.0008,
        'price_volatility': 0.12,
        'price_change_5': 0.05
    }
    
    # Test signal generation
    signal = logic.evaluate_tick(tick_data, market_analysis)
    print(f"Signal: {signal.signal_type}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reasoning: {signal.reasoning}")
    
    # Test position update
    if signal.signal_type in ['buy', 'sell']:
        logic.update_position(signal.signal_type, tick_data['price'])
        
        # Check exit
        exit_signal = logic._check_exit_conditions(tick_data, market_analysis)
        print(f"Exit signal: {exit_signal.signal_type} - {exit_signal.reasoning}")
    
    print(f"Performance: {logic.get_performance_stats()}")