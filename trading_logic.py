#!/usr/bin/env python3
"""
Enhanced Trading Logic Module for XAU/USD Tick Scalping
Specialized algorithms for gold spot trading with tick-level precision
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class XAUUSDTradingSignal:
    """Enhanced trading signal structure for XAU/USD"""
    signal_type: str  # 'buy', 'sell', 'hold', 'close'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    expected_profit_ticks: float
    max_risk_ticks: float
    time_horizon_seconds: int
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    
    # XAU/USD specific fields
    gold_volatility_regime: str = "normal"  # low, normal, high, extreme
    market_session: str = "london"          # asian, london, ny, overlap
    news_risk_level: str = "low"            # low, medium, high


class XAUUSDTradingLogic:
    """Enhanced trading logic specialized for XAU/USD tick scalping"""
    
    def __init__(self, tick_threshold: float = 0.50, profit_target: float = 2.0, 
                 stop_loss: float = 1.0, quantity: float = 0.1):
        
        # XAU/USD specific parameters
        self.symbol = "XAUUSD"
        self.tick_size = 0.10  # Standard XAU/USD tick size
        self.pip_value = 0.01  # For precision calculations
        self.quantity = quantity
        
        # Trading thresholds (in USD)
        self.tick_threshold = tick_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        
        # Convert to ticks for internal calculations
        self.profit_target_ticks = profit_target / self.tick_size
        self.stop_loss_ticks = stop_loss / self.tick_size
        self.entry_threshold_ticks = tick_threshold / self.tick_size
        
        # Position tracking
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.peak_profit_ticks = 0.0
        self.max_position_time = 60  # seconds
        
        # Market analysis buffers
        self.price_history = deque(maxlen=200)  # Larger buffer for gold analysis
        self.volume_history = deque(maxlen=100)
        self.spread_history = deque(maxlen=50)
        self.tick_velocity_history = deque(maxlen=30)
        
        # XAU/USD specific analysis
        self.support_resistance_levels = {}
        self.volatility_regime = "normal"
        self.market_session = "unknown"
        self.trend_direction = 0  # -1 bearish, 0 neutral, 1 bullish
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.5,
            'avg_profit_ticks': 0.0,
            'avg_loss_ticks': 0.0,
            'best_trade_ticks': 0.0,
            'worst_trade_ticks': 0.0
        }
        
        # Adaptive parameters
        self.adaptive_enabled = True
        self.confidence_threshold = 0.65  # Dynamic threshold
        self.recent_performance = deque(maxlen=20)
        
        # Market microstructure for gold
        self.gold_specific_params = {
            'london_fix_hours': [(10, 30), (15, 00)],  # London fix times UTC
            'high_volatility_sessions': ['london_open', 'ny_open', 'overlap'],
            'news_events_impact': ['fomc', 'nfp', 'cpi', 'gdp', 'pce'],
            'safe_haven_correlation': 0.8,  # USD/Gold inverse correlation
            'seasonal_patterns': self._initialize_seasonal_patterns()
        }
        
        logging.info(f"✅ XAU/USD Trading Logic initialized - Profit Target: {self.profit_target_ticks:.1f} ticks, Stop Loss: {self.stop_loss_ticks:.1f} ticks")
    
    def _initialize_seasonal_patterns(self) -> Dict:
        """Initialize gold seasonal trading patterns"""
        return {
            'asian_session': {'volatility': 'low', 'trend_strength': 'weak'},
            'london_session': {'volatility': 'high', 'trend_strength': 'strong'},
            'ny_session': {'volatility': 'medium', 'trend_strength': 'medium'},
            'london_ny_overlap': {'volatility': 'highest', 'trend_strength': 'strongest'},
            'friday_close': {'volatility': 'decreasing', 'trend_strength': 'weak'},
            'monday_open': {'volatility': 'gap_risk', 'trend_strength': 'uncertain'}
        }
    
    def evaluate_xauusd_signals(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Main method to evaluate XAU/USD tick data and generate trading signals"""
        
        # Update internal analysis
        self._update_market_analysis(tick_data, market_analysis)
        
        # Check exit conditions first if in position
        if self.position:
            exit_signal = self._check_xauusd_exit_conditions(tick_data, market_analysis)
            if exit_signal.signal_type == 'close':
                return exit_signal
        
        # Check entry conditions if no position
        if not self.position:
            return self._evaluate_xauusd_entry_signals(tick_data, market_analysis)
        
        # Hold if no conditions met
        return XAUUSDTradingSignal('hold', 0.0, 'No trading conditions met', 0, 0, 0)
    
    def _update_market_analysis(self, tick_data: Dict, market_analysis: Dict):
        """Update internal market analysis with XAU/USD specific metrics"""
        
        current_price = tick_data.get('price', 0)
        current_time = tick_data.get('timestamp', datetime.now())
        
        # Add to price history
        if current_price > 0:
            self.price_history.append({
                'price': current_price,
                'time': current_time,
                'volume': tick_data.get('size', 1),
                'spread': tick_data.get('spread', 0.1)
            })
        
        # Update market session
        self.market_session = self._determine_market_session(current_time)
        
        # Update volatility regime
        self.volatility_regime = self._classify_volatility_regime(market_analysis)
        
        # Update trend direction
        self.trend_direction = self._calculate_trend_direction()
        
        # Update support/resistance levels
        self._update_support_resistance_levels()
        
        # Track performance metrics
        self._update_performance_tracking()
    
    def _determine_market_session(self, current_time: datetime) -> str:
        """Determine current market session for XAU/USD"""
        
        # Convert to UTC hour
        utc_hour = current_time.hour
        
        # Define session times (UTC)
        if 22 <= utc_hour or utc_hour < 8:
            return "asian"
        elif 8 <= utc_hour < 13:
            return "london"
        elif 13 <= utc_hour < 17:
            return "london_ny_overlap"
        elif 17 <= utc_hour < 22:
            return "ny"
        else:
            return "unknown"
    
    def _classify_volatility_regime(self, market_analysis: Dict) -> str:
        """Classify current volatility regime for gold"""
        
        volatility = market_analysis.get('price_volatility', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        
        # XAU/USD specific volatility thresholds
        if volatility > 0.25 or tick_velocity > 15:
            return "extreme"
        elif volatility > 0.15 or tick_velocity > 10:
            return "high"
        elif volatility > 0.08 or tick_velocity > 5:
            return "normal"
        else:
            return "low"
    
    def _calculate_trend_direction(self) -> int:
        """Calculate trend direction using multiple timeframes"""
        
        if len(self.price_history) < 20:
            return 0
        
        recent_prices = [p['price'] for p in list(self.price_history)[-20:]]
        
        # Short-term trend (5 ticks)
        short_trend = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
        
        # Medium-term trend (10 ticks)
        medium_trend = (recent_prices[-1] - recent_prices[-10]) / recent_prices[-10] if len(recent_prices) >= 10 else 0
        
        # Long-term trend (20 ticks)
        long_trend = (recent_prices[-1] - recent_prices[-20]) / recent_prices[-20] if len(recent_prices) >= 20 else 0
        
        # Weighted trend calculation
        trend_score = (short_trend * 0.5) + (medium_trend * 0.3) + (long_trend * 0.2)
        
        if trend_score > 0.0005:  # 0.05% threshold
            return 1  # Bullish
        elif trend_score < -0.0005:
            return -1  # Bearish
        else:
            return 0  # Neutral
    
    def _update_support_resistance_levels(self):
        """Update dynamic support and resistance levels"""
        
        if len(self.price_history) < 50:
            return
        
        recent_prices = [p['price'] for p in list(self.price_history)[-50:]]
        
        # Find pivot points
        highs = []
        lows = []
        
        for i in range(2, len(recent_prices) - 2):
            # Local high
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i+1] and
                recent_prices[i] > recent_prices[i-2] and 
                recent_prices[i] > recent_prices[i+2]):
                highs.append(recent_prices[i])
            
            # Local low
            if (recent_prices[i] < recent_prices[i-1] and 
                recent_prices[i] < recent_prices[i+1] and
                recent_prices[i] < recent_prices[i-2] and 
                recent_prices[i] < recent_prices[i+2]):
                lows.append(recent_prices[i])
        
        # Update levels
        if highs:
            self.support_resistance_levels['resistance'] = np.mean(highs[-3:]) if len(highs) >= 3 else highs[-1]
        if lows:
            self.support_resistance_levels['support'] = np.mean(lows[-3:]) if len(lows) >= 3 else lows[-1]
    
    def _evaluate_xauusd_entry_signals(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Evaluate entry conditions specific to XAU/USD characteristics"""
        
        # Pre-flight checks for gold trading
        if not self._is_suitable_for_gold_scalping(tick_data, market_analysis):
            return XAUUSDTradingSignal('hold', 0.0, 'Market conditions unsuitable for gold scalping', 0, 0, 0)
        
        # Generate signals from multiple XAU/USD strategies
        signals = []
        
        # 1. Gold momentum strategy
        momentum_signal = self._generate_gold_momentum_signal(tick_data, market_analysis)
        if momentum_signal.confidence > 0:
            signals.append(momentum_signal)
        
        # 2. London fix strategy
        london_fix_signal = self._generate_london_fix_signal(tick_data, market_analysis)
        if london_fix_signal.confidence > 0:
            signals.append(london_fix_signal)
        
        # 3. Safe haven flow strategy
        safe_haven_signal = self._generate_safe_haven_signal(tick_data, market_analysis)
        if safe_haven_signal.confidence > 0:
            signals.append(safe_haven_signal)
        
        # 4. Technical breakout strategy
        breakout_signal = self._generate_gold_breakout_signal(tick_data, market_analysis)
        if breakout_signal.confidence > 0:
            signals.append(breakout_signal)
        
        # 5. Mean reversion strategy
        reversion_signal = self._generate_gold_reversion_signal(tick_data, market_analysis)
        if reversion_signal.confidence > 0:
            signals.append(reversion_signal)
        
        # Select best signal
        if signals:
            best_signal = max(signals, key=lambda s: s.confidence)
            
            # Apply XAU/USD specific filters
            if self._validate_gold_signal(best_signal, tick_data, market_analysis):
                return self._enhance_signal_with_gold_levels(best_signal, tick_data)
        
        return XAUUSDTradingSignal('hold', 0.0, 'No strong gold signals detected', 0, 0, 0)
    
    def _is_suitable_for_gold_scalping(self, tick_data: Dict, market_analysis: Dict) -> bool:
        """Check if conditions are suitable for gold scalping"""
        
        # Check spread
        current_spread = tick_data.get('spread', 0)
        if current_spread > 0.20:  # $0.20 max spread for gold
            return False
        
        # Check liquidity
        tick_velocity = market_analysis.get('tick_velocity', 0)
        if tick_velocity < 3.0:  # Minimum liquidity for gold
            return False
        
        # Check volatility regime
        if self.volatility_regime == 'extreme':
            return False
        
        # Check market session suitability
        if self.market_session == 'asian' and self.volatility_regime == 'low':
            return False  # Avoid low volatility Asian session
        
        # Check for news events (simplified)
        current_time = tick_data.get('timestamp', datetime.now())
        if self._is_news_time(current_time):
            return False
        
        return True
    
    def _is_news_time(self, current_time: datetime) -> bool:
        """Check if current time is close to major news events"""
        
        # Simplified news avoidance - avoid round hours during NY session
        if self.market_session in ['ny', 'london_ny_overlap']:
            if current_time.minute < 5 or current_time.minute > 55:
                return True
        
        return False
    
    def _generate_gold_momentum_signal(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Generate momentum signal specific to gold characteristics"""
        
        current_price = tick_data.get('price', 0)
        momentum_5 = market_analysis.get('momentum_5', 0)
        momentum_10 = market_analysis.get('momentum_10', 0)
        order_flow = market_analysis.get('order_flow_imbalance', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Strong bullish momentum for gold
        if (momentum_5 > 0.0008 and  # 0.08% momentum (adjusted for gold)
            momentum_10 > 0.0005 and
            order_flow > 0.35 and
            tick_velocity > 6 and
            self.trend_direction >= 0):
            
            confidence = min(0.88, 0.5 + momentum_5 * 500 + abs(order_flow) * 0.5)
            signal_type = 'buy'
            reasoning = f"Strong gold bullish momentum: {momentum_5:.4f} with flow {order_flow:.2f}"
        
        # Strong bearish momentum for gold
        elif (momentum_5 < -0.0008 and
              momentum_10 < -0.0005 and
              order_flow < -0.35 and
              tick_velocity > 6 and
              self.trend_direction <= 0):
            
            confidence = min(0.88, 0.5 + abs(momentum_5) * 500 + abs(order_flow) * 0.5)
            signal_type = 'sell'
            reasoning = f"Strong gold bearish momentum: {momentum_5:.4f} with flow {order_flow:.2f}"
        
        # Session-specific momentum adjustments
        if signal_type != 'hold':
            confidence = self._adjust_confidence_for_session(confidence, signal_type)
        
        return XAUUSDTradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=4.0,
            max_risk_ticks=2.0,
            time_horizon_seconds=45,
            gold_volatility_regime=self.volatility_regime,
            market_session=self.market_session
        )
    
    def _generate_london_fix_signal(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Generate signals around London gold fix times"""
        
        current_time = tick_data.get('timestamp', datetime.now())
        
        # Check if we're near London fix times (10:30 AM and 3:00 PM UTC)
        utc_time = current_time.replace(tzinfo=None)
        
        fix_times = [
            utc_time.replace(hour=10, minute=30, second=0, microsecond=0),
            utc_time.replace(hour=15, minute=0, second=0, microsecond=0)
        ]
        
        near_fix = False
        for fix_time in fix_times:
            time_diff = abs((utc_time - fix_time).total_seconds())
            if time_diff <= 300:  # Within 5 minutes of fix
                near_fix = True
                break
        
        if not near_fix:
            return XAUUSDTradingSignal('hold', 0.0, '', 0, 0, 0)
        
        # Analyze pre-fix momentum
        momentum_5 = market_analysis.get('momentum_5', 0)
        volume_trend = market_analysis.get('volume_trend', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Strong pre-fix momentum with volume
        if abs(momentum_5) > 0.001 and volume_trend > 0 and tick_velocity > 8:
            confidence = min(0.75, 0.4 + abs(momentum_5) * 300 + volume_trend * 0.1)
            signal_type = 'buy' if momentum_5 > 0 else 'sell'
            reasoning = f"London fix momentum: {momentum_5:.4f} with volume increase"
        
        return XAUUSDTradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=3.0,
            max_risk_ticks=2.0,
            time_horizon_seconds=30,
            gold_volatility_regime=self.volatility_regime,
            market_session=self.market_session
        )
    
    def _generate_safe_haven_signal(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Generate signals based on safe haven flows into gold"""
        
        # This would typically require external market data (USD index, equity futures, etc.)
        # For now, use volatility spikes as proxy for risk-off sentiment
        
        volatility = market_analysis.get('price_volatility', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        order_flow = market_analysis.get('order_flow_imbalance', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Volatility spike with buying pressure (safe haven demand)
        if (volatility > 0.15 and
            tick_velocity > 10 and
            order_flow > 0.4 and
            self.market_session in ['london', 'ny', 'london_ny_overlap']):
            
            confidence = min(0.80, 0.3 + volatility * 2 + order_flow * 0.8)
            signal_type = 'buy'
            reasoning = f"Safe haven demand: vol={volatility:.3f}, flow={order_flow:.2f}"
        
        return XAUUSDTradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=5.0,
            max_risk_ticks=2.5,
            time_horizon_seconds=60,
            gold_volatility_regime=self.volatility_regime,
            market_session=self.market_session
        )
    
    def _generate_gold_breakout_signal(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Generate breakout signals for gold"""
        
        current_price = tick_data.get('price', 0)
        resistance = self.support_resistance_levels.get('resistance', 0)
        support = self.support_resistance_levels.get('support', 0)
        volume_trend = market_analysis.get('volume_trend', 0)
        tick_velocity = market_analysis.get('tick_velocity', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Bullish breakout
        if (resistance > 0 and
            current_price > resistance and
            volume_trend > 0.5 and
            tick_velocity > 7):
            
            breakout_strength = (current_price - resistance) / resistance
            confidence = min(0.85, 0.4 + breakout_strength * 2000 + volume_trend * 0.2)
            signal_type = 'buy'
            reasoning = f"Bullish breakout above {resistance:.2f} with volume"
        
        # Bearish breakdown
        elif (support > 0 and
              current_price < support and
              volume_trend > 0.5 and
              tick_velocity > 7):
            
            breakdown_strength = (support - current_price) / support
            confidence = min(0.85, 0.4 + breakdown_strength * 2000 + volume_trend * 0.2)
            signal_type = 'sell'
            reasoning = f"Bearish breakdown below {support:.2f} with volume"
        
        return XAUUSDTradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=6.0,
            max_risk_ticks=3.0,
            time_horizon_seconds=90,
            gold_volatility_regime=self.volatility_regime,
            market_session=self.market_session
        )
    
    def _generate_gold_reversion_signal(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Generate mean reversion signals for gold"""
        
        current_price = tick_data.get('price', 0)
        resistance = self.support_resistance_levels.get('resistance', 0)
        support = self.support_resistance_levels.get('support', 0)
        volatility = market_analysis.get('price_volatility', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        if resistance > 0 and support > 0:
            range_size = resistance - support
            
            if range_size > 0:
                # Distance from support/resistance as percentage of range
                support_distance = (current_price - support) / range_size
                resistance_distance = (resistance - current_price) / range_size
                
                # Bounce from support
                if (support_distance < 0.08 and  # Within 8% of support
                    volatility > 0.10 and
                    self.trend_direction >= 0):
                    
                    confidence = min(0.78, 0.4 + (0.08 - support_distance) * 5)
                    signal_type = 'buy'
                    reasoning = f"Gold support bounce at {support:.2f}"
                
                # Rejection from resistance
                elif (resistance_distance < 0.08 and  # Within 8% of resistance
                      volatility > 0.10 and
                      self.trend_direction <= 0):
                    
                    confidence = min(0.78, 0.4 + (0.08 - resistance_distance) * 5)
                    signal_type = 'sell'
                    reasoning = f"Gold resistance rejection at {resistance:.2f}"
        
        return XAUUSDTradingSignal(
            signal_type, confidence, reasoning,
            expected_profit_ticks=3.0,
            max_risk_ticks=1.5,
            time_horizon_seconds=30,
            gold_volatility_regime=self.volatility_regime,
            market_session=self.market_session
        )
    
    def _adjust_confidence_for_session(self, confidence: float, signal_type: str) -> float:
        """Adjust signal confidence based on market session"""
        
        session_multipliers = {
            'asian': 0.8,           # Lower confidence in Asian session
            'london': 1.2,          # Higher confidence in London session
            'ny': 1.0,              # Normal confidence in NY session
            'london_ny_overlap': 1.3, # Highest confidence during overlap
            'unknown': 0.7
        }
        
        multiplier = session_multipliers.get(self.market_session, 1.0)
        adjusted_confidence = confidence * multiplier
        
        return min(adjusted_confidence, 1.0)
    
    def _validate_gold_signal(self, signal: XAUUSDTradingSignal, tick_data: Dict, market_analysis: Dict) -> bool:
        """Apply XAU/USD specific validation filters"""
        
        # Minimum confidence threshold (adaptive)
        if signal.confidence < self.confidence_threshold:
            return False
        
        # Spread filter for gold
        current_spread = tick_data.get('spread', 0)
        if current_spread > 0.15:  # Tighter spread requirement
            return False
        
        # Volatility filter
        if self.volatility_regime == 'extreme':
            return False
        
        # Session filter
        if self.market_session == 'asian' and signal.confidence < 0.75:
            return False  # Higher threshold for Asian session
        
        # Trend alignment filter
        if signal.signal_type == 'buy' and self.trend_direction < -1:
            return False  # Don't buy against strong bearish trend
        if signal.signal_type == 'sell' and self.trend_direction > 1:
            return False  # Don't sell against strong bullish trend
        
        return True
    
    def _enhance_signal_with_gold_levels(self, signal: XAUUSDTradingSignal, tick_data: Dict) -> XAUUSDTradingSignal:
        """Enhance signal with XAU/USD specific price levels"""
        
        current_price = tick_data.get('price', 0)
        
        if signal.signal_type == 'buy':
            signal.entry_price = current_price
            signal.take_profit_price = current_price + (signal.expected_profit_ticks * self.tick_size)
            signal.stop_loss_price = current_price - (signal.max_risk_ticks * self.tick_size)
            
            # Adjust for resistance levels
            resistance = self.support_resistance_levels.get('resistance', 0)
            if resistance > 0 and signal.take_profit_price > resistance:
                signal.take_profit_price = resistance - self.tick_size  # Take profit before resistance
        
        elif signal.signal_type == 'sell':
            signal.entry_price = current_price
            signal.take_profit_price = current_price - (signal.expected_profit_ticks * self.tick_size)
            signal.stop_loss_price = current_price + (signal.max_risk_ticks * self.tick_size)
            
            # Adjust for support levels
            support = self.support_resistance_levels.get('support', 0)
            if support > 0 and signal.take_profit_price < support:
                signal.take_profit_price = support + self.tick_size  # Take profit before support
        
        return signal
    
    def _check_xauusd_exit_conditions(self, tick_data: Dict, market_analysis: Dict) -> XAUUSDTradingSignal:
        """Check exit conditions specific to XAU/USD trading"""
        
        if not self.position or not self.entry_price:
            return XAUUSDTradingSignal('hold', 0.0, 'No position to exit', 0, 0, 0)
        
        current_price = tick_data.get('price', 0)
        current_time = tick_data.get('timestamp', datetime.now())
        
        # Calculate P&L in ticks
        if self.position == 'long':
            ticks_pnl = (current_price - self.entry_price) / self.tick_size
        else:  # short
            ticks_pnl = (self.entry_price - current_price) / self.tick_size
        
        # Update peak profit for trailing stop
        if ticks_pnl > self.peak_profit_ticks:
            self.peak_profit_ticks = ticks_pnl
        
        # Time-based exit
        if self.entry_time:
            time_in_position = (current_time - self.entry_time).total_seconds()
            if time_in_position > self.max_position_time:
                return XAUUSDTradingSignal('close', 1.0, f'Time exit: {time_in_position:.1f}s', ticks_pnl, 0, 0)
        
        # Profit target
        if ticks_pnl >= self.profit_target_ticks:
            return XAUUSDTradingSignal('close', 1.0, f'Profit target: +{ticks_pnl:.1f} ticks', ticks_pnl, 0, 0)
        
        # Stop loss
        if ticks_pnl <= -self.stop_loss_ticks:
            return XAUUSDTradingSignal('close', 1.0, f'Stop loss: {ticks_pnl:.1f} ticks', ticks_pnl, 0, 0)
        
        # Gold-specific exit conditions
        gold_exit = self._check_gold_specific_exits(current_price, ticks_pnl, market_analysis)
        if gold_exit:
            return XAUUSDTradingSignal('close', 0.9, gold_exit, ticks_pnl, 0, 0)
        
        # Trailing stop for gold
        trailing_exit = self._check_gold_trailing_stop(ticks_pnl)
        if trailing_exit:
            return XAUUSDTradingSignal('close', 0.95, trailing_exit, ticks_pnl, 0, 0)
        
        return XAUUSDTradingSignal('hold', 0.0, f'Hold position: {ticks_pnl:+.1f} ticks', ticks_pnl, 0, 0)
    
    def _check_gold_specific_exits(self, current_price: float, ticks_pnl: float, market_analysis: Dict) -> Optional[str]:
        """Check XAU/USD specific exit conditions"""
        
        # Volatility spike exit
        if self.volatility_regime == 'extreme' and ticks_pnl > 1.0:
            return "Volatility spike - securing profit"
        
        # Session change exit
        if self.market_session == 'asian' and ticks_pnl > 0.5:
            return "Asian session - reduced liquidity"
        
        # Support/resistance level exit
        resistance = self.support_resistance_levels.get('resistance', 0)
        support = self.support_resistance_levels.get('support', 0)
        
        if self.position == 'long' and resistance > 0:
            if current_price >= resistance - self.tick_size:
                return f"Approaching resistance at {resistance:.2f}"
        
        if self.position == 'short' and support > 0:
            if current_price <= support + self.tick_size:
                return f"Approaching support at {support:.2f}"
        
        # Trend reversal exit
        if self.position == 'long' and self.trend_direction < 0 and ticks_pnl > 0:
            return "Trend reversal detected"
        
        if self.position == 'short' and self.trend_direction > 0 and ticks_pnl > 0:
            return "Trend reversal detected"
        
        return None
    
    def _check_gold_trailing_stop(self, current_pnl_ticks: float) -> Optional[str]:
        """Check gold-specific trailing stop"""
        
        # Only apply trailing stop if in profit
        if current_pnl_ticks <= 1.5:  # Activate after 1.5 ticks profit
            return None
        
        # Trail by 1 tick from peak
        if current_pnl_ticks < self.peak_profit_ticks - 1.0:
            return f"Trailing stop: {current_pnl_ticks:.1f} from peak {self.peak_profit_ticks:.1f}"
        
        return None
    
    def _update_performance_tracking(self):
        """Update performance metrics and adaptive parameters"""
        
        # Update adaptive confidence threshold
        if len(self.recent_performance) >= 10:
            recent_win_rate = sum(self.recent_performance) / len(self.recent_performance)
            
            if recent_win_rate < 0.4:
                self.confidence_threshold = min(0.80, self.confidence_threshold + 0.05)
            elif recent_win_rate > 0.7:
                self.confidence_threshold = max(0.60, self.confidence_threshold - 0.02)
    
    # Legacy compatibility methods
    def evaluate_tick_signals(self, tick_data: Dict, market_analysis: Dict):
        """Legacy compatibility method"""
        return self.evaluate_xauusd_signals(tick_data, market_analysis)
    
    def generate_trade_signal(self, current_price: float, price_change: float) -> str:
        """Legacy compatibility method"""
        # Create simplified tick data
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
            'momentum_10': price_change / 100,
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
        
        signal = self.evaluate_xauusd_signals(tick_data, market_analysis)
        return signal.signal_type
    
    def analyze_market(self, current_price: float, price_change: float) -> str:
        """Legacy compatibility method"""
        return self.generate_trade_signal(current_price, price_change)
    
    def update_position(self, action: str, price: float, timestamp: str = None):
        """Update position state for XAU/USD"""
        
        if action in ['buy', 'sell']:
            self.position = 'long' if action == 'buy' else 'short'
            self.entry_price = price
            self.entry_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            self.peak_profit_ticks = 0.0  # Reset peak profit tracking
            
            logging.info(f"XAU/USD position updated: {self.position.upper()} @ ${price:.4f}")
            
        elif action == 'close':
            if self.position and self.entry_price:
                # Calculate final P&L in ticks
                if self.position == 'long':
                    ticks_pnl = (price - self.entry_price) / self.tick_size
                else:
                    ticks_pnl = (self.entry_price - price) / self.tick_size
                
                # Update performance metrics
                self._update_trade_performance(ticks_pnl > 0, ticks_pnl)
                
                logging.info(f"XAU/USD position closed: {self.position.upper()} | P&L: {ticks_pnl:+.1f} ticks (${ticks_pnl * self.tick_size:+.2f})")
            
            # Reset position state
            self.position = None
            self.entry_price = None
            self.entry_time = None
            self.peak_profit_ticks = 0.0
    
    def _update_trade_performance(self, is_winner: bool, ticks_pnl: float):
        """Update performance tracking with trade result"""
        
        self.performance_metrics['total_signals'] += 1
        
        if is_winner:
            self.performance_metrics['successful_signals'] += 1
            
        # Update win rate
        total = self.performance_metrics['total_signals']
        successful = self.performance_metrics['successful_signals']
        self.performance_metrics['win_rate'] = successful / total if total > 0 else 0.5
        
        # Update profit/loss tracking
        if is_winner:
            self.performance_metrics['avg_profit_ticks'] = (
                (self.performance_metrics['avg_profit_ticks'] * (successful - 1) + ticks_pnl) / successful
                if successful > 0 else ticks_pnl
            )
            if ticks_pnl > self.performance_metrics['best_trade_ticks']:
                self.performance_metrics['best_trade_ticks'] = ticks_pnl
        else:
            losing_trades = total - successful
            self.performance_metrics['avg_loss_ticks'] = (
                (self.performance_metrics['avg_loss_ticks'] * (losing_trades - 1) + abs(ticks_pnl)) / losing_trades
                if losing_trades > 0 else abs(ticks_pnl)
            )
            if ticks_pnl < self.performance_metrics['worst_trade_ticks']:
                self.performance_metrics['worst_trade_ticks'] = ticks_pnl
        
        # Add to recent performance for adaptive adjustments
        self.recent_performance.append(1.0 if is_winner else 0.0)
    
    def get_position_info(self) -> Dict:
        """Get current position information for XAU/USD"""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'quantity': self.quantity,
            'peak_profit_ticks': self.peak_profit_ticks,
            'symbol': self.symbol,
            'market_session': self.market_session,
            'volatility_regime': self.volatility_regime,
            'trend_direction': self.trend_direction
        }
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L in USD"""
        
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
    
    def adjust_parameters(self, win_rate: float, avg_pnl_ticks: float):
        """Dynamically adjust trading parameters based on XAU/USD performance"""
        
        if not self.adaptive_enabled:
            return
        
        # Adjust profit target and stop loss based on performance
        if win_rate < 0.35:  # Poor performance
            self.profit_target_ticks = min(self.profit_target_ticks * 1.15, 8.0)  # Increase target
            self.stop_loss_ticks = max(self.stop_loss_ticks * 0.9, 1.5)  # Tighten stop
            self.confidence_threshold = min(0.85, self.confidence_threshold + 0.05)  # Be more selective
            
            logging.info(f"XAU/USD poor performance adjustment: PT={self.profit_target_ticks:.1f}, SL={self.stop_loss_ticks:.1f}, Conf={self.confidence_threshold:.2f}")
            
        elif win_rate > 0.75:  # Excellent performance
            self.profit_target_ticks = max(self.profit_target_ticks * 0.95, 3.0)  # Reduce target slightly
            self.stop_loss_ticks = min(self.stop_loss_ticks * 1.05, 3.0)  # Loosen stop slightly
            self.confidence_threshold = max(0.55, self.confidence_threshold - 0.02)  # Be more aggressive
            
            logging.info(f"XAU/USD excellent performance adjustment: PT={self.profit_target_ticks:.1f}, SL={self.stop_loss_ticks:.1f}, Conf={self.confidence_threshold:.2f}")
        
        # Adjust for average P&L
        if avg_pnl_ticks > 0 and avg_pnl_ticks < 1.0:  # Small wins
            self.profit_target_ticks = max(self.profit_target_ticks * 0.9, 2.5)  # Take profits quicker
        elif avg_pnl_ticks > 3.0:  # Large wins available
            self.profit_target_ticks = min(self.profit_target_ticks * 1.1, 6.0)  # Let winners run more
        
        # Session-specific adjustments
        if self.market_session == 'asian':
            self.confidence_threshold = max(self.confidence_threshold, 0.70)  # Higher threshold for Asian session
        elif self.market_session == 'london_ny_overlap':
            self.confidence_threshold = max(0.55, self.confidence_threshold - 0.05)  # Lower threshold for high-liquidity overlap
    
    def get_strategy_stats(self) -> Dict:
        """Get comprehensive XAU/USD strategy statistics"""
        
        # Calculate additional XAU/USD specific metrics
        total_signals = self.performance_metrics['total_signals']
        
        stats = {
            'total_signals': total_signals,
            'successful_signals': self.performance_metrics['successful_signals'],
            'win_rate': self.performance_metrics['win_rate'],
            'avg_profit_ticks': self.performance_metrics['avg_profit_ticks'],
            'avg_loss_ticks': self.performance_metrics['avg_loss_ticks'],
            'best_trade_ticks': self.performance_metrics['best_trade_ticks'],
            'worst_trade_ticks': self.performance_metrics['worst_trade_ticks'],
            
            # Position info
            'current_position': self.position,
            'entry_price': self.entry_price,
            'peak_profit_ticks': self.peak_profit_ticks,
            
            # Strategy parameters
            'profit_target_ticks': self.profit_target_ticks,
            'stop_loss_ticks': self.stop_loss_ticks,
            'confidence_threshold': self.confidence_threshold,
            'max_position_time': self.max_position_time,
            
            # Market analysis
            'market_session': self.market_session,
            'volatility_regime': self.volatility_regime,
            'trend_direction': self.trend_direction,
            'support_level': self.support_resistance_levels.get('support', 0),
            'resistance_level': self.support_resistance_levels.get('resistance', 0),
            
            # XAU/USD specific
            'symbol': self.symbol,
            'tick_size': self.tick_size,
            'adaptive_enabled': self.adaptive_enabled,
            'price_history_size': len(self.price_history)
        }
        
        # Calculate profit factor if we have both wins and losses
        if self.performance_metrics['avg_loss_ticks'] > 0:
            profit_factor = (
                self.performance_metrics['avg_profit_ticks'] * self.performance_metrics['successful_signals']
            ) / (
                self.performance_metrics['avg_loss_ticks'] * (total_signals - self.performance_metrics['successful_signals'])
            )
            stats['profit_factor'] = profit_factor
        
        return stats
    
    def reset_position(self):
        """Manually reset position (emergency use)"""
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.peak_profit_ticks = 0.0
        
        logging.warning("⚠️  XAU/USD position manually reset")
    
    def get_market_conditions_summary(self) -> Dict:
        """Get current market conditions summary for XAU/USD"""
        
        return {
            'symbol': self.symbol,
            'market_session': self.market_session,
            'volatility_regime': self.volatility_regime,
            'trend_direction': self.trend_direction,
            'trend_description': {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}.get(self.trend_direction, 'Unknown'),
            'support_level': self.support_resistance_levels.get('support', 0),
            'resistance_level': self.support_resistance_levels.get('resistance', 0),
            'confidence_threshold': self.confidence_threshold,
            'adaptive_enabled': self.adaptive_enabled,
            'tick_size': self.tick_size,
            'current_position': self.position,
            'performance_summary': {
                'win_rate': f"{self.performance_metrics['win_rate']:.1%}",
                'total_signals': self.performance_metrics['total_signals'],
                'avg_profit_ticks': f"{self.performance_metrics['avg_profit_ticks']:.1f}",
                'avg_loss_ticks': f"{self.performance_metrics['avg_loss_ticks']:.1f}"
            }
        }


# Factory function for easy initialization
def create_xauusd_trading_logic(config: Dict = None) -> XAUUSDTradingLogic:
    """Factory function to create XAU/USD trading logic with configuration"""
    
    if config is None:
        config = {}
    
    return XAUUSDTradingLogic(
        tick_threshold=config.get('tick_threshold', 0.50),
        profit_target=config.get('profit_target', 2.0),
        stop_loss=config.get('stop_loss', 1.0),
        quantity=config.get('quantity', 0.1)
    )


# Backward compatibility
TradingLogic = XAUUSDTradingLogic


if __name__ == "__main__":
    # Test the XAU/USD trading logic
    logic = XAUUSDTradingLogic()
    
    # Sample XAU/USD tick data
    sample_tick = {
        'price': 2045.50,
        'timestamp': datetime.now(),
        'size': 25,
        'spread': 0.12
    }
    
    # Sample market analysis for gold
    sample_analysis = {
        'price_volatility': 0.11,
        'momentum_5': 0.0012,  # 0.12% momentum
        'momentum_10': 0.0008,
        'current_spread': 0.12,
        'tick_velocity': 7.2,
        'volatility_regime': 'normal',
        'order_flow_imbalance': 0.42,
        'price_acceleration': 0.018,
        'market_impact': 0.0015,
        'volume_trend': 1.4,
        'support_level': 2044.20,
        'resistance_level': 2047.80,
        'price_range': 3.60
    }
    
    # Test signal generation
    signal = logic.evaluate_xauusd_signals(sample_tick, sample_analysis)
    
    print(f"🥇 XAU/USD Signal: {signal.signal_type}")
    print(f"🎯 Confidence: {signal.confidence:.2f}")
    print(f"💡 Reasoning: {signal.reasoning}")
    print(f"💰 Expected Profit: {signal.expected_profit_ticks:.1f} ticks (${signal.expected_profit_ticks * 0.10:.2f})")
    print(f"🛡️  Max Risk: {signal.max_risk_ticks:.1f} ticks (${signal.max_risk_ticks * 0.10:.2f})")
    print(f"⏰ Time Horizon: {signal.time_horizon_seconds}s")
    print(f"📊 Market Session: {signal.market_session}")
    print(f"📈 Volatility Regime: {signal.gold_volatility_regime}")
    
    # Test position management
    if signal.signal_type in ['buy', 'sell']:
        logic.update_position(signal.signal_type, sample_tick['price'])
        print(f"\n📍 Position Info: {logic.get_position_info()}")
        
        # Test exit conditions
        exit_signal = logic._check_xauusd_exit_conditions(sample_tick, sample_analysis)
        print(f"🚪 Exit Signal: {exit_signal.signal_type} - {exit_signal.reasoning}")
    
    # Print strategy stats
    stats = logic.get_strategy_stats()
    print(f"\n📊 Strategy Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Print market conditions
    conditions = logic.get_market_conditions_summary()
    print(f"\n🌍 Market Conditions:")
    for key, value in conditions.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")