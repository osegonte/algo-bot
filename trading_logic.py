#!/usr/bin/env python3
"""
Enhanced Trading Logic Module for XAU/USD Tick Scalping
Implements advanced signal generation and market analysis
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TradingSignal:
    """Enhanced trading signal structure"""
    signal_type: str  # 'buy', 'sell', 'hold', 'close'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    expected_profit_ticks: float
    max_risk_ticks: float
    time_horizon_seconds: int
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0

class TradingLogic:
    """Enhanced trading logic with advanced signal generation"""
    
    def __init__(self, tick_threshold: float = 0.01, profit_target: float = 0.02, 
                 stop_loss: float = 0.01, quantity: float = 1.0):
        
        # Basic parameters
        self.tick_threshold = tick_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.quantity = quantity
        
        # Position tracking
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.entry_signal = None
        
        # Enhanced parameters for tick trading
        self.tick_size = 0.10  # XAU/USD tick size
        self.profit_target_ticks = 3.0  # Default profit target in ticks
        self.stop_loss_ticks = 2.0  # Default stop loss in ticks
        self.max_position_time = 60  # Maximum time in position (seconds)
        
        # Market analysis
        self.price_history = deque(maxlen=100)
        self.signal_history = deque(maxlen=50)
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.5
        }
        
        # Dynamic thresholds
        self.adaptive_thresholds = True
        self.min_tick_velocity = 4.0
        self.max_spread_threshold = 0.20
        self.volatility_threshold = 0.15
        
        # Signal weights
        self.signal_weights = {
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'order_flow': 0.25,
            'breakout': 0.25
        }
        
        logging.info("✅ Enhanced trading logic initialized")
    
    def evaluate_tick_signals(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Main method to evaluate tick data and generate trading signals"""
        
        # Add to price history
        current_price = tick_data.get('price', 0)
        if current_price > 0:
            self.price_history.append({
                'price': current_price,
                'timestamp': tick_data.get('timestamp', datetime.now()),
                'volume': tick_data.get('size', 0),
                'spread': tick_data.get('spread', 0)
            })
        
        # Check exit conditions first if in position
        if self.position:
            exit_signal = self._check_exit_conditions(tick_data, market_analysis)
            if exit_signal.signal_type == 'close':
                return exit_signal
        
        # Check entry conditions if no position
        if not self.position:
            return self._evaluate_entry_signals(tick_data, market_analysis)
        
        # Hold if no conditions met
        return TradingSignal('hold', 0.0, 'No conditions met', 0, 0, 0)
    
    def _evaluate_entry_signals(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Evaluate entry conditions using multiple strategies"""
        
        # Pre-flight checks
        if not self._market_suitable_for_trading(market_analysis):
            return TradingSignal('hold', 0.0, 'Market conditions not suitable', 0, 0, 0)
        
        # Generate signals from different strategies
        signals = []
        
        # 1. Momentum-based signals
        momentum_signal = self._generate_momentum_signal(tick_data, market_analysis)
        if momentum_signal.confidence > 0:
            signals.append(momentum_signal)
        
        # 2. Mean reversion signals
        reversion_signal = self._generate_mean_reversion_signal(tick_data, market_analysis)
        if reversion_signal.confidence > 0:
            signals.append(reversion_signal)
        
        # 3. Order flow signals
        order_flow_signal = self._generate_order_flow_signal(tick_data, market_analysis)
        if order_flow_signal.confidence > 0:
            signals.append(order_flow_signal)
        
        # 4. Breakout signals
        breakout_signal = self._generate_breakout_signal(tick_data, market_analysis)
        if breakout_signal.confidence > 0:
            signals.append(breakout_signal)
        
        # Select best signal
        if signals:
            best_signal = max(signals, key=lambda s: s.confidence)
            
            # Apply final filters
            if self._validate_signal(best_signal, tick_data, market_analysis):
                return self._enhance_signal_with_levels(best_signal, tick_data)
        
        return TradingSignal('hold', 0.0, 'No strong signals detected', 0, 0, 0)
    
    def _market_suitable_for_trading(self, market_analysis: Dict) -> bool:
        """Check if market conditions are suitable for scalping"""
        
        # Check spread
        current_spread = market_analysis.get('current_spread', 0)
        if current_spread > self.max_spread_threshold:
            return False
        
        # Check tick velocity (liquidity)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        if tick_velocity < self.min_tick_velocity:
            return False
        
        # Check volatility regime
        volatility_regime = market_analysis.get('volatility_regime', 'unknown')
        if volatility_regime == 'extreme':
            return False
        
        # Check if market is moving (not dead)
        price_range = market_analysis.get('price_range', 0)
        if price_range < self.tick_size * 0.5:  # Less than half tick movement
            return False
        
        return True
    
    def _generate_momentum_signal(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Generate momentum-based trading signal"""
        
        current_price = tick_data.get('price', 0)
        price_acceleration = market_analysis.get('price_acceleration', 0)
        order_flow = market_analysis.get('order_flow_imbalance', 0)
        momentum_5 = market_analysis.get('momentum_5', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Strong bullish momentum
        if (momentum_5 > 0.001 and  # 0.1% momentum
            price_acceleration > 0.01 and
            order_flow > 0.3 and
            tick_velocity > 6):
            
            confidence = min(0.85, 0.4 + momentum_5 * 100 + abs(price_acceleration) * 10)
            signal_type = 'buy'
            reasoning = f"Strong bullish momentum: {momentum_5:.3f}% with order flow {order_flow:.2f}"
        
        # Strong bearish momentum
        elif (momentum_5 < -0.001 and
              price_acceleration < -0.01 and
              order_flow < -0.3 and
              tick_velocity > 6):
            
            confidence = min(0.85, 0.4 + abs(momentum_5) * 100 + abs(price_acceleration) * 10)
            signal_type = 'sell'
            reasoning = f"Strong bearish momentum: {momentum_5:.3f}% with order flow {order_flow:.2f}"
        
        # Moderate momentum signals
        elif (momentum_5 > 0.0005 and order_flow > 0.2):
            confidence = min(0.65, 0.3 + momentum_5 * 100)
            signal_type = 'buy'
            reasoning = f"Moderate bullish momentum: {momentum_5:.3f}%"
        
        elif (momentum_5 < -0.0005 and order_flow < -0.2):
            confidence = min(0.65, 0.3 + abs(momentum_5) * 100)
            signal_type = 'sell'
            reasoning = f"Moderate bearish momentum: {momentum_5:.3f}%"
        
        return TradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=3.5,
            max_risk_ticks=2.0,
            time_horizon_seconds=30
        )
    
    def _generate_mean_reversion_signal(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Generate mean reversion trading signal"""
        
        current_price = tick_data.get('price', 0)
        support_level = market_analysis.get('support_level', 0)
        resistance_level = market_analysis.get('resistance_level', 0)
        volatility = market_analysis.get('price_volatility', 0)
        price_range = market_analysis.get('price_range', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Calculate distance to support/resistance as percentage
        if support_level > 0 and resistance_level > 0:
            range_size = resistance_level - support_level
            
            if range_size > 0:
                support_distance = (current_price - support_level) / range_size
                resistance_distance = (resistance_level - current_price) / range_size
                
                # Bounce from support
                if (support_distance < 0.05 and  # Within 5% of support
                    volatility > 0.08 and  # Sufficient volatility for reversion
                    price_range > self.tick_size):
                    
                    confidence = min(0.75, 0.4 + (0.05 - support_distance) * 10)
                    signal_type = 'buy'
                    reasoning = f"Support bounce at {support_level:.2f} (distance: {support_distance:.3f})"
                
                # Rejection from resistance
                elif (resistance_distance < 0.05 and  # Within 5% of resistance
                      volatility > 0.08 and
                      price_range > self.tick_size):
                    
                    confidence = min(0.75, 0.4 + (0.05 - resistance_distance) * 10)
                    signal_type = 'sell'
                    reasoning = f"Resistance rejection at {resistance_level:.2f} (distance: {resistance_distance:.3f})"
        
        return TradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=2.5,
            max_risk_ticks=1.5,
            time_horizon_seconds=20
        )
    
    def _generate_order_flow_signal(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Generate order flow imbalance signal"""
        
        order_flow = market_analysis.get('order_flow_imbalance', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        market_impact = market_analysis.get('market_impact', 0)
        volume_trend = market_analysis.get('volume_trend', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Strong buy order flow
        if (order_flow > 0.4 and
            tick_velocity > 5 and
            market_impact > 0.001):
            
            confidence = min(0.80, 0.3 + order_flow * 0.8 + market_impact * 100)
            signal_type = 'buy'
            reasoning = f"Strong buy order flow: {order_flow:.2f} with market impact {market_impact:.3f}"
        
        # Strong sell order flow
        elif (order_flow < -0.4 and
              tick_velocity > 5 and
              market_impact > 0.001):
            
            confidence = min(0.80, 0.3 + abs(order_flow) * 0.8 + market_impact * 100)
            signal_type = 'sell'
            reasoning = f"Strong sell order flow: {order_flow:.2f} with market impact {market_impact:.3f}"
        
        # Moderate order flow with volume confirmation
        elif (order_flow > 0.25 and volume_trend > 0):
            confidence = min(0.65, 0.25 + order_flow * 0.6)
            signal_type = 'buy'
            reasoning = f"Moderate buy flow with volume: {order_flow:.2f}"
        
        elif (order_flow < -0.25 and volume_trend > 0):
            confidence = min(0.65, 0.25 + abs(order_flow) * 0.6)
            signal_type = 'sell'
            reasoning = f"Moderate sell flow with volume: {order_flow:.2f}"
        
        return TradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=2.0,
            max_risk_ticks=1.5,
            time_horizon_seconds=15
        )
    
    def _generate_breakout_signal(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Generate breakout trading signal"""
        
        current_price = tick_data.get('price', 0)
        resistance_level = market_analysis.get('resistance_level', 0)
        support_level = market_analysis.get('support_level', 0)
        volume_trend = market_analysis.get('volume_trend', 0)
        price_range = market_analysis.get('price_range', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Bullish breakout
        if (current_price > resistance_level and
            volume_trend > 0 and
            price_range > self.tick_size * 2 and
            tick_velocity > 6):
            
            breakout_strength = (current_price - resistance_level) / resistance_level
            confidence = min(0.85, 0.4 + breakout_strength * 1000 + volume_trend * 0.1)
            signal_type = 'buy'
            reasoning = f"Bullish breakout above {resistance_level:.2f} with volume"
        
        # Bearish breakdown
        elif (current_price < support_level and
              volume_trend > 0 and
              price_range > self.tick_size * 2 and
              tick_velocity > 6):
            
            breakdown_strength = (support_level - current_price) / support_level
            confidence = min(0.85, 0.4 + breakdown_strength * 1000 + volume_trend * 0.1)
            signal_type = 'sell'
            reasoning = f"Bearish breakdown below {support_level:.2f} with volume"
        
        return TradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=4.0,
            max_risk_ticks=2.5,
            time_horizon_seconds=45
        )
    
    def _validate_signal(self, signal: TradingSignal, tick_data: Dict, market_analysis: Dict) -> bool:
        """Apply final validation filters to trading signal"""
        
        # Minimum confidence threshold
        min_confidence = 0.6
        if signal.confidence < min_confidence:
            return False
        
        # Spread filter
        current_spread = tick_data.get('spread', 0)
        if current_spread > self.max_spread_threshold:
            return False
        
        # Volatility filter
        volatility = market_analysis.get('price_volatility', 0)
        if volatility > self.volatility_threshold:
            return False
        
        # Recent performance filter
        if self.adaptive_thresholds:
            required_confidence = self._get_adaptive_confidence_threshold()
            if signal.confidence < required_confidence:
                return False
        
        return True
    
    def _get_adaptive_confidence_threshold(self) -> float:
        """Get adaptive confidence threshold based on recent performance"""
        
        base_threshold = 0.6
        
        # Adjust based on recent win rate
        if self.performance_metrics['win_rate'] < 0.4:
            return base_threshold + 0.2  # Be more selective
        elif self.performance_metrics['win_rate'] > 0.7:
            return base_threshold - 0.1  # Be more aggressive
        
        return base_threshold
    
    def _enhance_signal_with_levels(self, signal: TradingSignal, tick_data: Dict) -> TradingSignal:
        """Enhance signal with specific price levels"""
        
        current_price = tick_data.get('price', 0)
        
        if signal.signal_type == 'buy':
            signal.entry_price = current_price
            signal.take_profit_price = current_price + (signal.expected_profit_ticks * self.tick_size)
            signal.stop_loss_price = current_price - (signal.max_risk_ticks * self.tick_size)
        
        elif signal.signal_type == 'sell':
            signal.entry_price = current_price
            signal.take_profit_price = current_price - (signal.expected_profit_ticks * self.tick_size)
            signal.stop_loss_price = current_price + (signal.max_risk_ticks * self.tick_size)
        
        return signal
    
    def _check_exit_conditions(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Check exit conditions for current position"""
        
        if not self.position or not self.entry_price:
            return TradingSignal('hold', 0.0, 'No position to exit', 0, 0, 0)
        
        current_price = tick_data.get('price', 0)
        current_time = tick_data.get('timestamp', datetime.now())
        
        # Calculate P&L in ticks
        if self.position == 'long':
            ticks_pnl = (current_price - self.entry_price) / self.tick_size
        else:  # short
            ticks_pnl = (self.entry_price - current_price) / self.tick_size
        
        # Time-based exit
        if self.entry_time:
            time_in_position = (current_time - self.entry_time).total_seconds()
            if time_in_position > self.max_position_time:
                return TradingSignal('close', 1.0, f'Time exit: {time_in_position:.1f}s', ticks_pnl, 0, 0)
        
        # Profit target
        if ticks_pnl >= self.profit_target_ticks:
            return TradingSignal('close', 1.0, f'Profit target: +{ticks_pnl:.1f} ticks', ticks_pnl, 0, 0)
        
        # Stop loss
        if ticks_pnl <= -self.stop_loss_ticks:
            return TradingSignal('close', 1.0, f'Stop loss: {ticks_pnl:.1f} ticks', ticks_pnl, 0, 0)
        
        # Dynamic exit based on signal reversal
        reversal_signal = self._check_signal_reversal(tick_data, market_analysis)
        if reversal_signal:
            return TradingSignal('close', 0.8, f'Signal reversal: {reversal_signal}', ticks_pnl, 0, 0)
        
        # Trailing stop (optional)
        trailing_exit = self._check_trailing_stop(current_price, ticks_pnl)
        if trailing_exit:
            return TradingSignal('close', 0.9, f'Trailing stop: {trailing_exit}', ticks_pnl, 0, 0)
        
        return TradingSignal('hold', 0.0, f'Hold position: {ticks_pnl:+.1f} ticks', ticks_pnl, 0, 0)
    
    def _check_signal_reversal(self, tick_data: Dict, market_analysis: Dict) -> Optional[str]:
        """Check for signal reversal conditions"""
        
        order_flow = market_analysis.get('order_flow_imbalance', 0)
        momentum_5 = market_analysis.get('momentum_5', 0)
        
        # Long position reversal signals
        if self.position == 'long':
            if order_flow < -0.4 and momentum_5 < -0.001:
                return "Strong bearish reversal"
            elif order_flow < -0.25:
                return "Moderate bearish pressure"
        
        # Short position reversal signals
        elif self.position == 'short':
            if order_flow > 0.4 and momentum_5 > 0.001:
                return "Strong bullish reversal"
            elif order_flow > 0.25:
                return "Moderate bullish pressure"
        
        return None
    
    def _check_trailing_stop(self, current_price: float, current_pnl_ticks: float) -> Optional[str]:
        """Check trailing stop conditions"""
        
        # Only apply trailing stop if in profit
        if current_pnl_ticks <= 1.0:
            return None
        
        # Simple trailing stop: give back 1 tick from peak
        if not hasattr(self, 'peak_pnl'):
            self.peak_pnl = current_pnl_ticks
        
        if current_pnl_ticks > self.peak_pnl:
            self.peak_pnl = current_pnl_ticks
        
        # Trail by 1 tick
        if current_pnl_ticks < self.peak_pnl - 1.0:
            return f"Trailing from peak {self.peak_pnl:.1f}"
        
        return None
    
    def generate_trade_signal(self, current_price: float, price_change: float) -> str:
        """Legacy method for backward compatibility"""
        
        # Create simple tick data for legacy support
        tick_data = {
            'price': current_price,
            'timestamp': datetime.now(),
            'size': 1,
            'spread': 0.1
        }
        
        # Create basic market analysis
        market_analysis = {
            'price_volatility': abs(price_change) / 100,
            'momentum_5': price_change / 100,
            'current_spread': 0.1,
            'tick_velocity': 5.0,
            'volatility_regime': 'normal',
            'order_flow_imbalance': 0.0,
            'price_acceleration': 0.0,
            'market_impact': 0.0,
            'volume_trend': 0.0,
            'support_level': current_price * 0.999,
            'resistance_level': current_price * 1.001,
            'price_range': current_price * 0.001
        }
        
        signal = self.evaluate_tick_signals(tick_data, market_analysis)
        return signal.signal_type
    
    def analyze_market(self, current_price: float, price_change: float) -> str:
        """Legacy method for backward compatibility"""
        return self.generate_trade_signal(current_price, price_change)
    
    def update_position(self, action: str, price: float, timestamp: str = None):
        """Update position state"""
        
        if action in ['buy', 'sell']:
            self.position = 'long' if action == 'buy' else 'short'
            self.entry_price = price
            self.entry_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            
            # Reset peak P&L for trailing stop
            if hasattr(self, 'peak_pnl'):
                delattr(self, 'peak_pnl')
            
            logging.info(f"Position updated: {self.position.upper()} @ ${price:.4f}")
            
        elif action == 'close':
            if self.position and self.entry_price:
                # Calculate final P&L
                if self.position == 'long':
                    ticks_pnl = (price - self.entry_price) / self.tick_size
                else:
                    ticks_pnl = (self.entry_price - price) / self.tick_size
                
                # Update performance metrics
                self._update_performance_metrics(ticks_pnl > 0)
                
                logging.info(f"Position closed: {self.position.upper()} | P&L: {ticks_pnl:+.1f} ticks")
            
            self.position = None
            self.entry_price = None
            self.entry_time = None
            self.entry_signal = None
            
            if hasattr(self, 'peak_pnl'):
                delattr(self, 'peak_pnl')
    
    def _update_performance_metrics(self, is_winning_trade: bool):
        """Update performance tracking metrics"""
        
        self.performance_metrics['total_signals'] += 1
        
        if is_winning_trade:
            self.performance_metrics['successful_signals'] += 1
        
        # Calculate win rate
        total = self.performance_metrics['total_signals']
        successful = self.performance_metrics['successful_signals']
        self.performance_metrics['win_rate'] = successful / total if total > 0 else 0.5
    
    def get_position_info(self) -> Dict:
        """Get current position information"""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'quantity': self.quantity,
            'peak_pnl': getattr(self, 'peak_pnl', 0.0)
        }
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L in dollars"""
        
        if not self.position or not self.entry_price:
            return 0.0
        
        if self.position == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - current_price) * self.quantity
    
    def calculate_unrealized_pnl_ticks(self, current_price: float) -> float:
        """Calculate unrealized P&L in ticks"""
        
        if not self.position or not self.entry_price:
            return 0.0
        
        if self.position == 'long':
            return (current_price - self.entry_price) / self.tick_size
        else:  # short
            return (self.entry_price - current_price) / self.tick_size
    
    def adjust_parameters(self, win_rate: float, avg_pnl: float):
        """Dynamically adjust trading parameters based on performance"""
        
        if not self.adaptive_thresholds:
            return
        
        # Adjust profit target and stop loss based on performance
        if win_rate < 0.4:  # Poor performance
            self.profit_target_ticks = min(self.profit_target_ticks * 1.1, 5.0)
            self.stop_loss_ticks = max(self.stop_loss_ticks * 0.9, 1.0)
            logging.info(f"Poor performance: PT={self.profit_target_ticks:.1f}, SL={self.stop_loss_ticks:.1f}")
            
        elif win_rate > 0.7:  # Good performance
            self.profit_target_ticks = max(self.profit_target_ticks * 0.95, 2.0)
            self.stop_loss_ticks = min(self.stop_loss_ticks * 1.05, 3.0)
            logging.info(f"Good performance: PT={self.profit_target_ticks:.1f}, SL={self.stop_loss_ticks:.1f}")
    
    def reset_position(self):
        """Manually reset position (emergency)"""
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.entry_signal = None
        
        if hasattr(self, 'peak_pnl'):
            delattr(self, 'peak_pnl')
        
        logging.warning("⚠️  Position manually reset")
    
    def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        return {
            'total_signals': self.performance_metrics['total_signals'],
            'successful_signals': self.performance_metrics['successful_signals'],
            'win_rate': self.performance_metrics['win_rate'],
            'current_position': self.position,
            'profit_target_ticks': self.profit_target_ticks,
            'stop_loss_ticks': self.stop_loss_ticks,
            'max_position_time': self.max_position_time,
            'adaptive_thresholds': self.adaptive_thresholds
        }


# Factory function for easy initialization
def create_trading_logic(config: Dict = None) -> TradingLogic:
    """Factory function to create trading logic with configuration"""
    
    if config is None:
        config = {}
    
    return TradingLogic(
        tick_threshold=config.get('tick_threshold', 0.01),
        profit_target=config.get('profit_target', 0.02),
        stop_loss=config.get('stop_loss', 0.01),
        quantity=config.get('quantity', 1.0)
    )


if __name__ == "__main__":
    # Test the enhanced trading logic
    logic = TradingLogic()
    
    # Sample tick data
    sample_tick = {
        'price': 2000.50,
        'timestamp': datetime.now(),
        'size': 25,
        'spread': 0.10
    }
    
    # Sample market analysis
    sample_analysis = {
        'price_volatility': 0.12,
        'momentum_5': 0.002,  # 0.2% momentum
        'current_spread': 0.10,
        'tick_velocity': 6.5,
        'volatility_regime': 'normal',
        'order_flow_imbalance': 0.35,
        'price_acceleration': 0.015,
        'market_impact': 0.002,
        'volume_trend': 1.2,
        'support_level': 1999.80,
        'resistance_level': 2001.20,
        'price_range': 1.40
    }
    
    # Test signal generation
    signal = logic.evaluate_tick_signals(sample_tick, sample_analysis)
    
    print(f"Signal: {signal.signal_type}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reasoning: {signal.reasoning}")
    print(f"Expected Profit: {signal.expected_profit_ticks:.1f} ticks")
    print(f"Max Risk: {signal.max_risk_ticks:.1f} ticks")
    
    # Test position management
    if signal.signal_type in ['buy', 'sell']:
        logic.update_position(signal.signal_type, sample_tick['price'])
        print(f"Position Info: {logic.get_position_info()}")
        
        # Test exit conditions
        exit_signal = logic._check_exit_conditions(sample_tick, sample_analysis)
        print(f"Exit Signal: {exit_signal.signal_type} - {exit_signal.reasoning}")
    
    # Print strategy stats
    stats = logic.get_strategy_stats()
    print(f"Strategy Stats: {stats}")