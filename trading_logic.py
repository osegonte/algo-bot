#!/usr/bin/env python3
"""
Enhanced Aggressive Trading Logic for XAU/USD Scalping
Implements multiple signal confirmations and dynamic risk management
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from collections import deque


class TradingSignal:
    """Enhanced trading signal with additional metadata"""
    def __init__(self, signal_type: str, confidence: float, reasoning: str, 
                 strength: float = 0.0, indicators: Dict = None):
        self.signal_type = signal_type  # 'buy', 'sell', 'hold', 'close'
        self.confidence = confidence    # 0.0 to 1.0
        self.reasoning = reasoning
        self.strength = strength        # NEW: Signal strength (0.0 to 1.0)
        self.indicators = indicators or {}  # NEW: Contributing indicators
        self.timestamp = datetime.now()


class EnhancedTechnicalIndicators:
    """Enhanced technical indicators for aggressive trading"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history = deque(maxlen=lookback_period)
        self.volume_history = deque(maxlen=lookback_period)
        self.timestamp_history = deque(maxlen=lookback_period)
    
    def add_tick(self, price: float, volume: int, timestamp: datetime = None):
        """Add new tick data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.timestamp_history.append(timestamp or datetime.now())
    
    def calculate_ema(self, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(self.price_history) < period:
            return 0.0
        
        prices = list(self.price_history)[-period:]
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(self.price_history) < period + 1:
            return 50.0
        
        prices = np.array(list(self.price_history)[-period-1:])
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        if len(self.price_history) < period:
            current_price = self.price_history[-1] if self.price_history else 0
            return current_price, current_price, current_price
        
        prices = np.array(list(self.price_history)[-period:])
        middle = np.mean(prices)
        std = np.std(prices)
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD (macd_line, signal_line, histogram)"""
        if len(self.price_history) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        macd_line = ema_fast - ema_slow
        
        # Simplified signal line calculation
        signal_line = macd_line * 0.2  # Simplified for real-time calculation
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_momentum(self, period: int = 10) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < period + 1:
            return 0.0
        
        current_price = self.price_history[-1]
        past_price = self.price_history[-period-1]
        
        return (current_price - past_price) / past_price if past_price > 0 else 0.0
    
    def calculate_volume_rate(self, period: int = 10) -> float:
        """Calculate volume rate of change"""
        if len(self.volume_history) < period + 1:
            return 1.0
        
        current_vol = self.volume_history[-1]
        avg_vol = np.mean(list(self.volume_history)[-period:])
        
        return current_vol / avg_vol if avg_vol > 0 else 1.0


class AggressiveTradingLogic:
    """Enhanced aggressive trading logic with multiple confirmations"""
    
    def __init__(self, profit_target_ticks: int = 8, stop_loss_ticks: int = 4, 
                 tick_size: float = 0.10, min_confidence: float = 0.75,
                 momentum_threshold: float = 0.0008, price_change_threshold: float = 0.035,
                 volatility_filter: float = 0.25, spread_filter: float = 0.12,
                 use_enhanced_signals: bool = True, multi_timeframe_check: bool = True,
                 volume_confirmation: bool = True, trend_alignment: bool = True):
        
        # Enhanced strategy parameters
        self.profit_target_ticks = profit_target_ticks
        self.stop_loss_ticks = stop_loss_ticks
        self.tick_size = tick_size
        self.min_confidence = min_confidence
        self.momentum_threshold = momentum_threshold
        self.price_change_threshold = price_change_threshold
        self.volatility_filter = volatility_filter
        self.spread_filter = spread_filter
        
        # Enhanced features
        self.use_enhanced_signals = use_enhanced_signals
        self.multi_timeframe_check = multi_timeframe_check
        self.volume_confirmation = volume_confirmation
        self.trend_alignment = trend_alignment
        
        # Position tracking
        self.position = None      # 'long', 'short', or None
        self.entry_price = None
        self.entry_time = None
        self.max_position_time = 90  # seconds
        self.trailing_stop_price = None
        self.highest_profit = 0.0
        
        # Technical indicators
        self.indicators = EnhancedTechnicalIndicators(50)
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.signal_history = deque(maxlen=100)
        
        # Risk management
        self.daily_pnl = 0.0
        self.session_start = datetime.now()
        self.trade_count_today = 0
        
        logging.info(f"✅ Enhanced aggressive trading logic initialized")
        logging.info(f"   🎯 Target: {profit_target_ticks} ticks, Stop: {stop_loss_ticks} ticks")
        logging.info(f"   🔥 Momentum threshold: {momentum_threshold:.4f}")
        logging.info(f"   📊 Confidence threshold: {min_confidence:.2f}")
    
    def evaluate_tick(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Enhanced tick evaluation with multiple confirmations"""
        
        # Add tick to indicators
        current_price = tick_data.get('price', 0)
        volume = tick_data.get('size', 1)
        self.indicators.add_tick(current_price, volume)
        
        # Check exit conditions first if in position
        if self.position:
            exit_signal = self._check_enhanced_exit_conditions(tick_data, market_analysis)
            if exit_signal.signal_type == 'close':
                return exit_signal
        
        # Check entry conditions if no position
        if not self.position:
            return self._check_enhanced_entry_conditions(tick_data, market_analysis)
        
        return TradingSignal('hold', 0.0, 'No action needed')
    
    def _check_enhanced_entry_conditions(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Enhanced entry conditions with multiple confirmations"""
        
        current_price = tick_data.get('price', 0)
        spread = tick_data.get('spread', 0)
        momentum = market_analysis.get('momentum', 0)
        volatility = market_analysis.get('price_volatility', 0)
        price_change = market_analysis.get('price_change_5', 0)
        
        # Enhanced basic filters
        if spread > self.spread_filter:
            return TradingSignal('hold', 0.0, f'Spread too wide: {spread:.3f}')
        
        if volatility > self.volatility_filter:
            return TradingSignal('hold', 0.0, f'Volatility too high: {volatility:.3f}')
        
        # Calculate technical indicators
        indicators = self._calculate_all_indicators()
        
        # Enhanced signal logic
        if self.use_enhanced_signals:
            return self._generate_enhanced_signal(tick_data, market_analysis, indicators)
        else:
            return self._generate_basic_signal(tick_data, market_analysis, indicators)
    
    def _calculate_all_indicators(self) -> Dict:
        """Calculate all technical indicators"""
        
        if len(self.indicators.price_history) < 10:
            return {}
        
        indicators = {}
        
        # Moving averages
        indicators['ema_8'] = self.indicators.calculate_ema(8)
        indicators['ema_21'] = self.indicators.calculate_ema(21)
        indicators['ema_50'] = self.indicators.calculate_ema(50)
        
        # Oscillators
        indicators['rsi'] = self.indicators.calculate_rsi(14)
        indicators['momentum_10'] = self.indicators.calculate_momentum(10)
        indicators['momentum_20'] = self.indicators.calculate_momentum(20)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(20, 2.0)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        current_price = self.indicators.price_history[-1]
        indicators['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # MACD
        macd_line, signal_line, histogram = self.indicators.calculate_macd()
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # Volume
        indicators['volume_rate'] = self.indicators.calculate_volume_rate(10)
        
        return indicators
    
    def _generate_enhanced_signal(self, tick_data: Dict, market_analysis: Dict, indicators: Dict) -> TradingSignal:
        """Generate enhanced signal with multiple confirmations"""
        
        if not indicators:
            return TradingSignal('hold', 0.0, 'Insufficient data for indicators')
        
        current_price = tick_data.get('price', 0)
        momentum = market_analysis.get('momentum', 0)
        price_change = market_analysis.get('price_change_5', 0)
        
        # Initialize scoring system
        bullish_score = 0
        bearish_score = 0
        max_score = 0
        confirmations = []
        
        # 1. Enhanced Momentum Analysis (Weight: 3)
        max_score += 3
        if abs(momentum) > self.momentum_threshold:
            if momentum > self.momentum_threshold:
                bullish_score += 3
                confirmations.append(f"Strong bullish momentum: {momentum:.4f}")
            elif momentum < -self.momentum_threshold:
                bearish_score += 3
                confirmations.append(f"Strong bearish momentum: {momentum:.4f}")
        
        # 2. Price Change Confirmation (Weight: 2)
        max_score += 2
        if abs(price_change) > self.price_change_threshold:
            if price_change > self.price_change_threshold:
                bullish_score += 2
                confirmations.append(f"Bullish price change: {price_change:.3f}%")
            elif price_change < -self.price_change_threshold:
                bearish_score += 2
                confirmations.append(f"Bearish price change: {price_change:.3f}%")
        
        # 3. EMA Trend Alignment (Weight: 2)
        if self.trend_alignment:
            max_score += 2
            ema_8 = indicators.get('ema_8', 0)
            ema_21 = indicators.get('ema_21', 0)
            
            if ema_8 > ema_21 and current_price > ema_8:
                bullish_score += 2
                confirmations.append("EMA trend alignment: Bullish")
            elif ema_8 < ema_21 and current_price < ema_8:
                bearish_score += 2
                confirmations.append("EMA trend alignment: Bearish")
        
        # 4. RSI Confirmation (Weight: 1)
        max_score += 1
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 70:  # Not oversold/overbought
            if rsi > 55:
                bullish_score += 1
                confirmations.append(f"RSI bullish: {rsi:.1f}")
            elif rsi < 45:
                bearish_score += 1
                confirmations.append(f"RSI bearish: {rsi:.1f}")
        
        # 5. MACD Confirmation (Weight: 2)
        max_score += 2
        macd_line = indicators.get('macd_line', 0)
        macd_histogram = indicators.get('macd_histogram', 0)
        
        if macd_line > 0 and macd_histogram > 0:
            bullish_score += 2
            confirmations.append("MACD bullish crossover")
        elif macd_line < 0 and macd_histogram < 0:
            bearish_score += 2
            confirmations.append("MACD bearish crossover")
        
        # 6. Volume Confirmation (Weight: 1)
        if self.volume_confirmation:
            max_score += 1
            volume_rate = indicators.get('volume_rate', 1.0)
            if volume_rate > 1.2:  # Above average volume
                if bullish_score > bearish_score:
                    bullish_score += 1
                    confirmations.append(f"Volume confirmation: {volume_rate:.2f}x")
                elif bearish_score > bullish_score:
                    bearish_score += 1
                    confirmations.append(f"Volume confirmation: {volume_rate:.2f}x")
        
        # 7. Bollinger Band Position (Weight: 1)
        max_score += 1
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.3:  # Near lower band
            bullish_score += 1
            confirmations.append("Near BB lower band")
        elif bb_position > 0.7:  # Near upper band
            bearish_score += 1
            confirmations.append("Near BB upper band")
        
        # Calculate final signal
        if bullish_score > bearish_score and bullish_score >= max_score * 0.6:
            confidence = min(0.95, bullish_score / max_score)
            signal_strength = bullish_score / max_score
            
            if confidence >= self.min_confidence:
                reasoning = f"Enhanced bullish signal: {bullish_score}/{max_score} points"
                reasoning += f" | Confirmations: {', '.join(confirmations[:3])}"
                
                return TradingSignal('buy', confidence, reasoning, signal_strength, indicators)
        
        elif bearish_score > bullish_score and bearish_score >= max_score * 0.6:
            confidence = min(0.95, bearish_score / max_score)
            signal_strength = bearish_score / max_score
            
            if confidence >= self.min_confidence:
                reasoning = f"Enhanced bearish signal: {bearish_score}/{max_score} points"
                reasoning += f" | Confirmations: {', '.join(confirmations[:3])}"
                
                return TradingSignal('sell', confidence, reasoning, signal_strength, indicators)
        
        # No strong signal
        total_score = max(bullish_score, bearish_score)
        confidence = total_score / max_score if max_score > 0 else 0
        
        return TradingSignal('hold', confidence, f'Insufficient signal strength: {total_score}/{max_score}')
    
    def _generate_basic_signal(self, tick_data: Dict, market_analysis: Dict, indicators: Dict) -> TradingSignal:
        """Generate basic signal (fallback)"""
        
        momentum = market_analysis.get('momentum', 0)
        price_change = market_analysis.get('price_change_5', 0)
        
        confidence = 0.0
        signal_type = 'hold'
        reasoning = ""
        
        # Enhanced momentum strategy
        if momentum > self.momentum_threshold and price_change > self.price_change_threshold:
            confidence = min(0.85, 0.5 + momentum * 1000 + abs(price_change) * 10)
            signal_type = 'buy'
            reasoning = f"Basic bullish: momentum {momentum:.4f}, change {price_change:.2f}%"
        
        elif momentum < -self.momentum_threshold and price_change < -self.price_change_threshold:
            confidence = min(0.85, 0.5 + abs(momentum) * 1000 + abs(price_change) * 10)
            signal_type = 'sell'
            reasoning = f"Basic bearish: momentum {momentum:.4f}, change {price_change:.2f}%"
        
        if confidence < self.min_confidence:
            return TradingSignal('hold', confidence, f"Basic confidence too low: {confidence:.2f}")
        
        return TradingSignal(signal_type, confidence, reasoning)
    
    def _check_enhanced_exit_conditions(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Enhanced exit conditions with trailing stops"""
        
        if not self.position or not self.entry_price:
            return TradingSignal('hold', 0.0, 'No position')
        
        current_price = tick_data.get('price', 0)
        current_time = tick_data.get('timestamp', datetime.now())
        
        # Calculate P&L in ticks and USD
        if self.position == 'long':
            pnl_ticks = (current_price - self.entry_price) / self.tick_size
            pnl_usd = current_price - self.entry_price
        else:  # short
            pnl_ticks = (self.entry_price - current_price) / self.tick_size
            pnl_usd = self.entry_price - current_price
        
        # Update highest profit for trailing stop
        if pnl_usd > self.highest_profit:
            self.highest_profit = pnl_usd
        
        # 1. Profit target (enhanced)
        if pnl_ticks >= self.profit_target_ticks:
            return TradingSignal('close', 1.0, f'Profit target reached: +{pnl_ticks:.1f} ticks')
        
        # 2. Stop loss (enhanced)
        if pnl_ticks <= -self.stop_loss_ticks:
            return TradingSignal('close', 1.0, f'Stop loss hit: {pnl_ticks:.1f} ticks')
        
        # 3. Trailing stop loss
        if self.trailing_stop_price and self.highest_profit > 0:
            trailing_distance_usd = 3 * self.tick_size  # 3 ticks trailing distance
            
            if self.position == 'long':
                new_stop = current_price - trailing_distance_usd
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                
                if current_price <= self.trailing_stop_price:
                    return TradingSignal('close', 1.0, f'Trailing stop triggered: {pnl_ticks:.1f} ticks')
            
            else:  # short position
                new_stop = current_price + trailing_distance_usd
                if new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                
                if current_price >= self.trailing_stop_price:
                    return TradingSignal('close', 1.0, f'Trailing stop triggered: {pnl_ticks:.1f} ticks')
        
        # 4. Breakeven stop
        breakeven_threshold = 4 * self.tick_size  # Move to breakeven after 4 ticks profit
        if pnl_usd >= breakeven_threshold and not self.trailing_stop_price:
            if self.position == 'long':
                self.trailing_stop_price = self.entry_price
            else:
                self.trailing_stop_price = self.entry_price
            logging.info(f"Breakeven stop activated at ${self.entry_price:.2f}")
        
        # 5. Time-based exit (enhanced)
        if self.entry_time:
            time_in_position = (current_time - self.entry_time).total_seconds()
            if time_in_position > self.max_position_time:
                return TradingSignal('close', 1.0, f'Time exit: {time_in_position:.0f}s')
        
        # 6. Reversal signal detection
        indicators = self._calculate_all_indicators()
        if indicators:
            reversal_signal = self._detect_reversal_signal(current_price, indicators)
            if reversal_signal:
                return TradingSignal('close', 0.8, f'Reversal detected: {reversal_signal}')
        
        return TradingSignal('hold', 0.0, f'Hold: {pnl_ticks:+.1f} ticks (${pnl_usd:+.2f})')
    
    def _detect_reversal_signal(self, current_price: float, indicators: Dict) -> Optional[str]:
        """Detect potential reversal signals"""
        
        # RSI divergence
        rsi = indicators.get('rsi', 50)
        if self.position == 'long' and rsi > 75:
            return "RSI overbought"
        elif self.position == 'short' and rsi < 25:
            return "RSI oversold"
        
        # MACD divergence
        macd_histogram = indicators.get('macd_histogram', 0)
        if self.position == 'long' and macd_histogram < -0.1:
            return "MACD bearish divergence"
        elif self.position == 'short' and macd_histogram > 0.1:
            return "MACD bullish divergence"
        
        # Bollinger Band extremes
        bb_position = indicators.get('bb_position', 0.5)
        if self.position == 'long' and bb_position > 0.95:
            return "BB upper extreme"
        elif self.position == 'short' and bb_position < 0.05:
            return "BB lower extreme"
        
        return None
    
    def update_position(self, action: str, price: float, timestamp: str = None):
        """Enhanced position update with risk tracking"""
        
        if action in ['buy', 'sell']:
            self.position = 'long' if action == 'buy' else 'short'
            self.entry_price = price
            self.entry_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            self.trailing_stop_price = None
            self.highest_profit = 0.0
            self.trade_count_today += 1
            
            logging.info(f"🚀 Enhanced position opened: {self.position.upper()} @ ${price:.2f}")
            logging.info(f"   🎯 Target: +{self.profit_target_ticks} ticks (${self.profit_target_ticks * self.tick_size:.2f})")
            logging.info(f"   🛡️ Stop: -{self.stop_loss_ticks} ticks (${self.stop_loss_ticks * self.tick_size:.2f})")
            
        elif action == 'close':
            if self.position and self.entry_price:
                # Calculate final P&L
                if self.position == 'long':
                    pnl_ticks = (price - self.entry_price) / self.tick_size
                    pnl_usd = price - self.entry_price
                else:
                    pnl_ticks = (self.entry_price - price) / self.tick_size
                    pnl_usd = self.entry_price - price
                
                # Update daily P&L
                self.daily_pnl += pnl_usd
                
                # Update performance tracking
                self.total_signals += 1
                if pnl_ticks > 0:
                    self.successful_signals += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                
                # Log enhanced closure
                duration = (datetime.now() - self.entry_time).total_seconds() if self.entry_time else 0
                logging.info(f"📊 Enhanced position closed:")
                logging.info(f"   💰 P&L: {pnl_ticks:+.1f} ticks (${pnl_usd:+.2f})")
                logging.info(f"   ⏱️ Duration: {duration:.1f}s")
                logging.info(f"   📈 Daily P&L: ${self.daily_pnl:+.2f}")
                logging.info(f"   🎯 Win Rate: {self.get_win_rate():.1f}%")
            
            # Reset position
            self.position = None
            self.entry_price = None
            self.entry_time = None
            self.trailing_stop_price = None
            self.highest_profit = 0.0
    
    def get_position_info(self) -> Dict:
        """Enhanced position information"""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'trailing_stop_price': self.trailing_stop_price,
            'highest_profit': self.highest_profit,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'trade_count_today': self.trade_count_today
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
        """Enhanced performance statistics"""
        win_rate = self.get_win_rate()
        
        # Calculate additional metrics
        avg_win = 0.0
        avg_loss = 0.0
        if len(self.signal_history) > 0:
            wins = [s for s in self.signal_history if s.get('pnl', 0) > 0]
            losses = [s for s in self.signal_history if s.get('pnl', 0) < 0]
            
            avg_win = np.mean([s['pnl'] for s in wins]) if wins else 0.0
            avg_loss = np.mean([s['pnl'] for s in losses]) if losses else 0.0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'win_rate': win_rate,
            'current_position': self.position,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'trade_count_today': self.trade_count_today,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'enhanced_features_active': self.use_enhanced_signals
        }
    
    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        return (self.successful_signals / self.total_signals * 100) if self.total_signals > 0 else 0
    
    def should_continue_trading(self) -> Tuple[bool, str]:
        """Check if trading should continue based on risk limits"""
        
        # Daily loss limit
        if hasattr(self, 'max_daily_loss') and self.daily_pnl <= -getattr(self, 'max_daily_loss', 5.0):
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        # Consecutive loss limit
        if hasattr(self, 'max_consecutive_losses_limit') and self.consecutive_losses >= getattr(self, 'max_consecutive_losses_limit', 3):
            return False, f"Consecutive loss limit: {self.consecutive_losses}"
        
        # Daily profit target
        if hasattr(self, 'daily_profit_target') and self.daily_pnl >= getattr(self, 'daily_profit_target', 15.0):
            return False, f"Daily profit target reached: ${self.daily_pnl:.2f}"
        
        return True, "Continue trading"
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.trade_count_today = 0
        self.consecutive_losses = 0
        self.session_start = datetime.now()
        logging.info("📊 Daily statistics reset")
    
    def get_signal_strength_analysis(self) -> Dict:
        """Analyze recent signal strengths"""
        if len(self.signal_history) < 5:
            return {'status': 'insufficient_data'}
        
        recent_signals = list(self.signal_history)[-10:]
        strengths = [s.get('strength', 0) for s in recent_signals if s.get('strength')]
        
        if not strengths:
            return {'status': 'no_strength_data'}
        
        return {
            'status': 'analyzed',
            'avg_strength': np.mean(strengths),
            'min_strength': min(strengths),
            'max_strength': max(strengths),
            'strength_trend': 'improving' if strengths[-3:] > strengths[-6:-3] else 'declining'
        }


# Enhanced configuration validation
def validate_aggressive_logic_config(config: Dict) -> Dict:
    """Validate aggressive trading logic configuration"""
    
    results = {'valid': True, 'warnings': [], 'errors': []}
    
    # Risk-reward ratio
    profit_target = config.get('profit_target_ticks', 8)
    stop_loss = config.get('stop_loss_ticks', 4)
    risk_reward = profit_target / stop_loss
    
    if risk_reward < 1.5:
        results['warnings'].append(f"Low risk-reward ratio: {risk_reward:.1f}")
    
    # Confidence threshold
    min_confidence = config.get('min_confidence', 0.75)
    if min_confidence < 0.7:
        results['warnings'].append(f"Low confidence threshold: {min_confidence}")
    
    # Momentum threshold
    momentum_threshold = config.get('momentum_threshold', 0.0008)
    if momentum_threshold < 0.0005:
        results['warnings'].append("Momentum threshold may be too sensitive")
    
    return results


if __name__ == "__main__":
    # Test enhanced aggressive trading logic
    print("🧪 Testing Enhanced Aggressive Trading Logic...")
    
    # Create enhanced logic instance
    logic = AggressiveTradingLogic(
        profit_target_ticks=8,
        stop_loss_ticks=4,
        min_confidence=0.75,
        momentum_threshold=0.0008,
        use_enhanced_signals=True,
        multi_timeframe_check=True,
        volume_confirmation=True,
        trend_alignment=True
    )
    
    # Test with sample data
    tick_data = {
        'price': 2000.50,
        'spread': 0.10,
        'size': 25,
        'timestamp': datetime.now()
    }
    
    market_analysis = {
        'momentum': 0.0012,  # Strong momentum
        'price_volatility': 0.20,
        'price_change_5': 0.04  # 4% price change
    }
    
    # Add some historical data
    for i in range(30):
        price = 2000.00 + i * 0.02 + np.random.normal(0, 0.05)
        logic.indicators.add_tick(price, np.random.randint(10, 50))
    
    # Test signal generation
    signal = logic.evaluate_tick(tick_data, market_analysis)
    
    print(f"\n📈 Enhanced Signal Results:")
    print(f"   Signal: {signal.signal_type.upper()}")
    print(f"   Confidence: {signal.confidence:.3f}")
    print(f"   Strength: {signal.strength:.3f}")
    print(f"   Reasoning: {signal.reasoning}")
    
    # Test position management
    if signal.signal_type in ['buy', 'sell']:
        print(f"\n🚀 Testing position management...")
        logic.update_position(signal.signal_type, tick_data['price'])
        
        # Test exit conditions
        exit_tick = tick_data.copy()
        exit_tick['price'] += 0.40 if signal.signal_type == 'buy' else -0.40
        
        exit_signal = logic._check_enhanced_exit_conditions(exit_tick, market_analysis)
        print(f"   Exit Signal: {exit_signal.signal_type}")
        print(f"   Exit Reasoning: {exit_signal.reasoning}")
        
        if exit_signal.signal_type == 'close':
            logic.update_position('close', exit_tick['price'])
    
    # Performance stats
    stats = logic.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Enhanced aggressive trading logic test completed!")