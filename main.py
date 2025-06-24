#!/usr/bin/env python3
"""
Aggressive XAU/USD Tick Trading Bot - Enhanced Main Entry Point
Implements aggressive trading strategy with enhanced risk management
"""

import asyncio
import logging
import time
import sys
from datetime import datetime, timedelta

# Import aggressive configuration
try:
    from config import (ALPACA_CONFIG, STRATEGY_CONFIG, BOT_CONFIG, 
                       RISK_CONFIG, ML_CONFIG_AGGRESSIVE, PERFORMANCE_CONFIG,
                       ALERT_CONFIG, validate_aggressive_config)
    AGGRESSIVE_CONFIG_AVAILABLE = True
except ImportError:
    # Fallback to original config
    from config import ALPACA_CONFIG, STRATEGY_CONFIG, BOT_CONFIG
    AGGRESSIVE_CONFIG_AVAILABLE = False
    print("⚠️ Aggressive config not found, using standard configuration")

# Import core modules
from data_collection import SimpleDataCollector

# Try to import aggressive trading logic
try:
    from trading_logic import AggressiveTradingLogic
    AGGRESSIVE_LOGIC_AVAILABLE = True
except ImportError:
    from trading_logic import SimpleTradingLogic
    AGGRESSIVE_LOGIC_AVAILABLE = False
    print("⚠️ Aggressive trading logic not found, using standard logic")

from trade_execution import SimpleTradeExecutor

# Try to import enhanced performance monitor
try:
    from logger import AggressivePerformanceMonitor
    ENHANCED_MONITOR_AVAILABLE = True
except ImportError:
    from logger import SimpleTradeLogger
    ENHANCED_MONITOR_AVAILABLE = False
    print("⚠️ Enhanced performance monitor not found, using standard logger")

# Import ML interface
try:
    from ml_interface import SimpleMLInterface
    if AGGRESSIVE_CONFIG_AVAILABLE:
        ML_CONFIG = ML_CONFIG_AGGRESSIVE
    else:
        from ml_interface import ML_CONFIG
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False


class AggressiveXAUUSDBot:
    """Enhanced Aggressive XAU/USD Trading Bot"""
    
    def __init__(self):
        # Validate aggressive configuration if available
        if AGGRESSIVE_CONFIG_AVAILABLE:
            validation = validate_aggressive_config()
            if validation['warnings']:
                print("\n⚠️ Configuration Warnings:")
                for warning in validation['warnings']:
                    print(f"   - {warning}")
            if validation['errors']:
                print("\n❌ Configuration Errors:")
                for error in validation['errors']:
                    print(f"   - {error}")
                sys.exit(1)
        
        # Bot configuration
        self.symbol = ALPACA_CONFIG['symbol']
        self.quantity = ALPACA_CONFIG['quantity']
        self.api_key = ALPACA_CONFIG['api_key']
        self.secret_key = ALPACA_CONFIG['secret_key']
        
        # Enhanced bot state
        self.is_running = False
        self.trades_today = 0
        self.max_daily_trades = BOT_CONFIG.get('max_daily_trades', 15)
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        
        # Risk management limits
        if AGGRESSIVE_CONFIG_AVAILABLE:
            self.max_daily_loss = BOT_CONFIG.get('max_daily_loss', 5.0)
            self.max_consecutive_losses = BOT_CONFIG.get('max_consecutive_losses', 3)
            self.daily_profit_target = BOT_CONFIG.get('daily_profit_target', 15.0)
        else:
            self.max_daily_loss = 3.0
            self.max_consecutive_losses = 3
            self.daily_profit_target = 8.0
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_trade_time = None
        self.performance_start_balance = 100000.0  # Starting balance for tracking
        
        # Initialize components
        self.setup_logging()
        self.initialize_components()
        
        print(f"\n🚀 Aggressive XAU/USD Trading Bot Initialized")
        self.display_config()
    
    def setup_logging(self):
        """Setup enhanced logging"""
        log_filename = f'aggressive_bot_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        logging.info("🚀 Aggressive XAU/USD Bot Starting...")
        if AGGRESSIVE_CONFIG_AVAILABLE:
            logging.info("✅ Aggressive configuration loaded")
        if AGGRESSIVE_LOGIC_AVAILABLE:
            logging.info("✅ Aggressive trading logic loaded")
        if ENHANCED_MONITOR_AVAILABLE:
            logging.info("✅ Enhanced performance monitor loaded")
    
    def initialize_components(self):
        """Initialize enhanced trading components"""
        
        print("🔧 Initializing aggressive components...")
        
        # 1. Data collector (same as before)
        self.data_collector = SimpleDataCollector(self.symbol)
        
        # 2. Enhanced Trading logic
        if AGGRESSIVE_LOGIC_AVAILABLE:
            self.trading_logic = AggressiveTradingLogic(
                profit_target_ticks=STRATEGY_CONFIG['profit_target_ticks'],
                stop_loss_ticks=STRATEGY_CONFIG['stop_loss_ticks'],
                tick_size=STRATEGY_CONFIG['tick_size'],
                min_confidence=STRATEGY_CONFIG['min_confidence'],
                momentum_threshold=STRATEGY_CONFIG.get('momentum_threshold', 0.0008),
                price_change_threshold=STRATEGY_CONFIG.get('price_change_threshold', 0.035),
                volatility_filter=STRATEGY_CONFIG.get('volatility_filter', 0.25),
                spread_filter=STRATEGY_CONFIG.get('spread_filter', 0.12),
                use_enhanced_signals=STRATEGY_CONFIG.get('use_enhanced_signals', True),
                multi_timeframe_check=STRATEGY_CONFIG.get('multi_timeframe_check', True),
                volume_confirmation=STRATEGY_CONFIG.get('volume_confirmation', True),
                trend_alignment=STRATEGY_CONFIG.get('trend_alignment', True)
            )
            print("✅ Aggressive trading logic initialized")
        else:
            self.trading_logic = SimpleTradingLogic(
                profit_target_ticks=STRATEGY_CONFIG['profit_target_ticks'],
                stop_loss_ticks=STRATEGY_CONFIG['stop_loss_ticks'],
                tick_size=STRATEGY_CONFIG['tick_size'],
                min_confidence=STRATEGY_CONFIG['min_confidence']
            )
            print("⚠️ Using standard trading logic")
        
        # 3. Trade executor (enhanced position sizing)
        self.trade_executor = SimpleTradeExecutor(
            paper_trading=ALPACA_CONFIG['paper_trading'],
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        # 4. Enhanced performance monitor
        if ENHANCED_MONITOR_AVAILABLE:
            self.performance_monitor = AggressivePerformanceMonitor(
                BOT_CONFIG.get('log_file', 'xauusd_aggressive_trades.csv')
            )
            print("✅ Enhanced performance monitor initialized")
        else:
            self.performance_monitor = SimpleTradeLogger(
                BOT_CONFIG.get('log_file', 'xauusd_trades.csv')
            )
            print("⚠️ Using standard trade logger")
        
        # 5. ML interface (enhanced if available)
        if ML_ENABLED:
            self.ml_interface = SimpleMLInterface(ML_CONFIG)
            print("🤖 Enhanced ML interface enabled")
        else:
            self.ml_interface = None
            print("⚠️ ML interface not available")
        
        # Setup data callback
        self.data_collector.add_tick_callback(self.on_tick_received)
        
        print("✅ All aggressive components initialized")
    
    def display_config(self):
        """Display enhanced bot configuration"""
        
        mode = "📄 PAPER" if ALPACA_CONFIG['paper_trading'] else "💰 LIVE"
        api_status = "🔑 API" if self.api_key else "🎮 SIM"
        ml_status = "🤖 ML" if self.ml_interface else "🔓 NO ML"
        strategy_type = "🚀 AGGRESSIVE" if AGGRESSIVE_LOGIC_AVAILABLE else "📊 STANDARD"
        monitor_type = "📈 ENHANCED" if ENHANCED_MONITOR_AVAILABLE else "📋 BASIC"
        
        print(f"   📊 Symbol: {self.symbol}")
        print(f"   💹 Mode: {mode}")
        print(f"   🔗 Connection: {api_status}")
        print(f"   🎯 Strategy: {strategy_type}")
        print(f"   📈 Monitor: {monitor_type}")
        print(f"   🤖 AI: {ml_status}")
        print(f"   📦 Position Size: {self.quantity} oz")
        print(f"   🎯 Profit Target: {STRATEGY_CONFIG['profit_target_ticks']} ticks (${STRATEGY_CONFIG['profit_target_ticks'] * 0.1:.2f})")
        print(f"   🛡️ Stop Loss: {STRATEGY_CONFIG['stop_loss_ticks']} ticks (${STRATEGY_CONFIG['stop_loss_ticks'] * 0.1:.2f})")
        print(f"   📊 Daily Limits: {self.max_daily_trades} trades, ${self.max_daily_loss:.2f} loss, ${self.daily_profit_target:.2f} target")
        
        # Risk-reward ratio
        risk_reward = STRATEGY_CONFIG['profit_target_ticks'] / STRATEGY_CONFIG['stop_loss_ticks']
        print(f"   ⚖️ Risk-Reward: {risk_reward:.1f}:1")
    
    async def start_trading(self):
        """Start the aggressive trading bot"""
        
        print("\n" + "="*70)
        print("           🚀 STARTING AGGRESSIVE XAUUSD BOT")
        print("="*70)
        
        try:
            # Pre-flight checks
            if not self._pre_flight_checks():
                return
            
            # Start data feed
            print("📡 Starting data feed...")
            self.data_collector.start_data_feed(self.api_key, self.secret_key)
            
            # Wait for data
            await self.wait_for_data()
            
            # Display account info
            self.display_account_info()
            
            # Display strategy info
            self._display_strategy_info()
            
            # Main trading loop
            print("🔄 Starting aggressive trading loop...")
            print("⏹️ Press Ctrl+C to stop")
            print("-" * 70)
            
            self.is_running = True
            await self.main_trading_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            await self.shutdown()
        except Exception as e:
            logging.error(f"Aggressive bot error: {e}")
            await self.shutdown()
    
    def _pre_flight_checks(self) -> bool:
        """Perform pre-flight safety checks"""
        
        print("🔍 Performing pre-flight checks...")
        
        # Check if aggressive settings are properly configured
        if self.quantity <= 0.1:
            print("⚠️ Warning: Position size not increased for aggressive strategy")
        
        if STRATEGY_CONFIG['profit_target_ticks'] <= 4:
            print("⚠️ Warning: Profit target not increased for aggressive strategy")
        
        # Check risk parameters
        risk_reward = STRATEGY_CONFIG['profit_target_ticks'] / STRATEGY_CONFIG['stop_loss_ticks']
        if risk_reward < 1.5:
            print(f"⚠️ Warning: Risk-reward ratio ({risk_reward:.1f}) may be insufficient")
        
        # Validate daily limits
        if self.daily_profit_target / self.max_daily_loss < 2.0:
            print("⚠️ Warning: Aggressive daily limits - monitor carefully")
        
        print("✅ Pre-flight checks completed")
        return True
    
    def _display_strategy_info(self):
        """Display strategy-specific information"""
        
        if AGGRESSIVE_LOGIC_AVAILABLE:
            print(f"\n🚀 AGGRESSIVE STRATEGY ACTIVE:")
            print(f"   🎯 Enhanced Signals: Multiple confirmations required")
            print(f"   📊 Momentum Threshold: {STRATEGY_CONFIG.get('momentum_threshold', 0.0008):.4f}")
            print(f"   📈 Price Change Threshold: {STRATEGY_CONFIG.get('price_change_threshold', 0.035):.1%}")
            print(f"   🔄 Multi-timeframe: {STRATEGY_CONFIG.get('multi_timeframe_check', True)}")
            print(f"   📊 Volume Confirmation: {STRATEGY_CONFIG.get('volume_confirmation', True)}")
            print(f"   📈 Trend Alignment: {STRATEGY_CONFIG.get('trend_alignment', True)}")
        else:
            print(f"\n📊 STANDARD STRATEGY:")
            print(f"   Basic momentum-based signals")
    
    async def wait_for_data(self, max_wait: int = 10):
        """Wait for data connection"""
        
        for i in range(max_wait):
            if self.data_collector.tick_count > 0:
                price = self.data_collector.get_current_price()
                print(f"✅ Data connected - Current {self.symbol}: ${price:.2f}")
                return True
            
            await asyncio.sleep(1)
            if i % 2 == 0:
                print(f"   Waiting for data... ({i+1}/{max_wait})")
        
        print("⚠️ Data connection timeout - continuing anyway")
        return False
    
    def display_account_info(self):
        """Display account information"""
        
        account = self.trade_executor.get_account_info()
        print(f"💰 Account Balance: ${account['balance']:,.2f}")
        print(f"💵 Available Cash: ${account['cash']:,.2f}")
        
        # Set performance tracking baseline
        if hasattr(self.performance_monitor, 'starting_balance'):
            self.performance_monitor.starting_balance = account['balance']
            self.performance_monitor.current_balance = account['balance']
            self.performance_monitor.peak_balance = account['balance']
    
    async def main_trading_loop(self):
        """Enhanced main trading loop with risk management"""
        
        last_status_time = time.time()
        status_interval = BOT_CONFIG.get('status_update_interval', 8)
        last_risk_check = time.time()
        risk_check_interval = 30  # Check risk limits every 30 seconds
        
        while self.is_running:
            try:
                # Enhanced risk checks
                current_time = time.time()
                if current_time - last_risk_check > risk_check_interval:
                    if not self._check_risk_limits():
                        break
                    last_risk_check = current_time
                
                # Periodic status update (more frequent for aggressive strategy)
                if current_time - last_status_time > status_interval:
                    self.log_enhanced_status_update()
                    last_status_time = current_time
                
                await asyncio.sleep(0.05)  # Faster loop for aggressive strategy
                
            except Exception as e:
                logging.error(f"Trading loop error: {e}")
                await asyncio.sleep(1)
        
        await self.shutdown()
    
    def _check_risk_limits(self) -> bool:
        """Enhanced risk limit checks"""
        
        # Daily trade limit
        if self.trades_today >= self.max_daily_trades:
            print(f"🛑 Daily trade limit reached: {self.trades_today}")
            self._send_alert("DAILY_LIMIT", f"Trade limit reached: {self.trades_today}")
            return False
        
        # Daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"🛑 Daily loss limit reached: ${self.daily_pnl:.2f}")
            self._send_alert("LOSS_LIMIT", f"Daily loss limit: ${self.daily_pnl:.2f}")
            return False
        
        # Daily profit target
        if self.daily_pnl >= self.daily_profit_target:
            print(f"🎯 Daily profit target reached: ${self.daily_pnl:.2f}")
            self._send_alert("PROFIT_TARGET", f"Daily target reached: ${self.daily_pnl:.2f}")
            return False
        
        # Consecutive loss limit
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"🛑 Consecutive loss limit: {self.consecutive_losses}")
            self._send_alert("CONSECUTIVE_LOSSES", f"Loss streak: {self.consecutive_losses}")
            return False
        
        # Additional checks for aggressive strategy
        if hasattr(self.trading_logic, 'should_continue_trading'):
            should_continue, reason = self.trading_logic.should_continue_trading()
            if not should_continue:
                print(f"🛑 Trading logic stop: {reason}")
                return False
        
        return True
    
    def _send_alert(self, alert_type: str, message: str):
        """Send risk alerts"""
        
        if AGGRESSIVE_CONFIG_AVAILABLE and ALERT_CONFIG.get('enable_alerts', True):
            timestamp = datetime.now().strftime('%H:%M:%S')
            alert_msg = f"🚨 {alert_type}: {message}"
            
            if ALERT_CONFIG.get('console_alerts', True):
                print(f"\n{alert_msg}")
            
            if ALERT_CONFIG.get('log_alerts', True):
                logging.warning(f"ALERT - {alert_type}: {message}")
    
    def on_tick_received(self, tick_data):
        """Enhanced tick processing with aggressive strategy"""
        
        try:
            # Get market analysis
            market_analysis = self.data_collector.get_market_analysis()
            
            if not market_analysis:
                return
            
            # Get ML signal if available
            ml_signal = None
            if self.ml_interface:
                ml_signal = self.ml_interface.process_tick(tick_data)
            
            # Generate trading signal
            signal = self.trading_logic.evaluate_tick(tick_data, market_analysis)
            
            # Enhance signal with ML if available and aggressive config
            if (ml_signal and ml_signal.signal != 'hold' and 
                ml_signal.confidence > 0.75 and AGGRESSIVE_LOGIC_AVAILABLE):
                
                if signal.signal_type == 'hold':  # Use ML signal if logic says hold
                    signal.signal_type = ml_signal.signal
                    signal.confidence = min(signal.confidence + ml_signal.confidence * 0.3, 0.95)
                    signal.reasoning = f"ML Enhanced: {ml_signal.reasoning}"
            
            # Execute signals
            if signal.signal_type in ['buy', 'sell']:
                self.execute_enhanced_signal(signal, tick_data, ml_signal)
            elif signal.signal_type == 'close':
                self.close_enhanced_position(tick_data, signal.reasoning)
            
        except Exception as e:
            logging.error(f"Enhanced tick processing error: {e}")
            logging.basicConfig(level=logging.DEBUG)
    
    def execute_enhanced_signal(self, signal, tick_data, ml_signal=None):
        """Execute trading signal with enhanced logging"""
        
        current_price = tick_data['price']
        
        # Enhanced signal display
        print(f"\n🚀 AGGRESSIVE SIGNAL: {signal.signal_type.upper()} @ ${current_price:.2f}")
        print(f"   🎯 Confidence: {signal.confidence:.3f}")
        if hasattr(signal, 'strength') and signal.strength > 0:
            print(f"   💪 Strength: {signal.strength:.3f}")
        print(f"   💡 Reasoning: {signal.reasoning}")
        
        if ml_signal and ml_signal.signal != 'hold':
            print(f"   🤖 ML Enhancement: {ml_signal.signal} ({ml_signal.confidence:.2f})")
        
        # Dynamic position sizing for aggressive strategy
        quantity = self._calculate_position_size(signal)
        
        # Place order
        trade = self.trade_executor.place_order(
            self.symbol, signal.signal_type, quantity, current_price
        )
        
        if trade.status == "filled":
            # Update position in trading logic
            self.trading_logic.update_position(signal.signal_type, current_price, trade.timestamp)
            
            # Enhanced logging
            signal_data = {
                'confidence': signal.confidence,
                'strength': getattr(signal, 'strength', 0),
                'reasoning': signal.reasoning
            }
            
            if hasattr(self.performance_monitor, 'log_enhanced_trade'):
                self.performance_monitor.log_enhanced_trade(
                    trade, trade_type="entry", **signal_data
                )
            else:
                self.performance_monitor.log_trade(trade, trade_type="entry")
            
            # Update counters
            self.trades_today += 1
            self.last_trade_time = datetime.now()
            
            print(f"✅ AGGRESSIVE ENTRY: {signal.signal_type.upper()} position opened")
            print(f"   📦 Quantity: {quantity} oz")
            print(f"   🆔 Trade ID: {trade.trade_id}")
            print(f"   📊 Trade #{self.trades_today} today")
        else:
            print(f"❌ ENTRY FAILED: {trade.status}")
    
    def _calculate_position_size(self, signal) -> float:
        """Calculate dynamic position size for aggressive strategy"""
        
        base_quantity = self.quantity
        
        # Scale based on signal confidence if aggressive config available
        if (AGGRESSIVE_CONFIG_AVAILABLE and 
            RISK_CONFIG.get('position_sizing', {}).get('scale_on_confidence', False)):
            
            if signal.confidence > 0.85:
                multiplier = RISK_CONFIG['position_sizing'].get('confidence_multiplier', 1.5)
                max_quantity = RISK_CONFIG['position_sizing'].get('max_quantity', 0.5)
                scaled_quantity = base_quantity * multiplier
                return min(scaled_quantity, max_quantity)
        
        return base_quantity
    
    def close_enhanced_position(self, tick_data, reasoning):
        """Close position with enhanced tracking"""
        
        current_price = tick_data['price']
        position_info = self.trading_logic.get_position_info()
        
        if not position_info['position']:
            return
        
        print(f"\n📉 AGGRESSIVE EXIT: {reasoning} @ ${current_price:.2f}")
        
        # Determine close side
        close_side = 'sell' if position_info['position'] == 'long' else 'buy'
        
        # Calculate quantity (may be different from entry if dynamic sizing)
        quantity = self.quantity  # For simplicity, using base quantity
        
        # Place closing order
        exit_trade = self.trade_executor.place_order(
            self.symbol, close_side, quantity, current_price
        )
        
        if exit_trade.status == "filled":
            # Calculate P&L
            pnl_usd = self.trading_logic.calculate_unrealized_pnl(current_price)
            pnl_ticks = self.trading_logic.calculate_unrealized_pnl_ticks(current_price)
            
            # Calculate trade duration
            duration = 0
            if position_info.get('entry_time'):
                entry_time = datetime.fromisoformat(position_info['entry_time'])
                duration = (datetime.now() - entry_time).total_seconds()
            
            # Update daily P&L and streaks
            self.daily_pnl += pnl_usd
            
            if pnl_usd > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            # Update position in trading logic
            self.trading_logic.update_position('close', current_price, exit_trade.timestamp)
            
            # Enhanced logging
            exit_data = {
                'duration': duration,
                'exit_reason': reasoning
            }
            
            if hasattr(self.performance_monitor, 'log_enhanced_trade'):
                self.performance_monitor.log_enhanced_trade(
                    exit_trade, trade_type="exit", 
                    profit_loss=pnl_usd, **exit_data
                )
            else:
                self.performance_monitor.log_trade(
                    exit_trade, trade_type="exit", profit_loss=pnl_usd
                )
            
            # Record ML outcome if available
            if self.ml_interface:
                market_analysis = self.data_collector.get_market_analysis()
                if market_analysis:
                    features = self.ml_interface.feature_extractor.extract_features()
                    self.ml_interface.record_trade_outcome(features, close_side, pnl_usd)
            
            # Enhanced results display
            pnl_symbol = "🟢" if pnl_usd > 0 else "🔴"
            print(f"✅ AGGRESSIVE EXIT: Position closed")
            print(f"   💰 P&L: ${pnl_usd:+.2f} ({pnl_ticks:+.1f} ticks) {pnl_symbol}")
            print(f"   ⏱️ Duration: {duration:.1f}s")
            print(f"   📊 Daily P&L: ${self.daily_pnl:+.2f}")
            print(f"   🆔 Trade ID: {exit_trade.trade_id}")
            
            # Check for alerts
            if pnl_usd > 2.0:
                self._send_alert("LARGE_WIN", f"${pnl_usd:.2f}")
            elif pnl_usd < -1.5:
                self._send_alert("LARGE_LOSS", f"${pnl_usd:.2f}")
            
        else:
            print(f"❌ EXIT FAILED: {exit_trade.status}")
    
    def log_enhanced_status_update(self):
        """Enhanced periodic status update"""
        
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = self.data_collector.get_current_price()
        position_info = self.trading_logic.get_position_info()
        
        print(f"\n📊 AGGRESSIVE STATUS - {current_time}")
        print(f"   💹 {self.symbol}: ${current_price:.2f}")
        
        if position_info['position']:
            unrealized_pnl = self.trading_logic.calculate_unrealized_pnl(current_price)
            unrealized_ticks = self.trading_logic.calculate_unrealized_pnl_ticks(current_price)
            
            # Enhanced position info
            entry_price = position_info.get('entry_price', 0)
            print(f"   📍 Position: {position_info['position'].upper()} @ ${entry_price:.2f}")
            print(f"   💰 Unrealized: ${unrealized_pnl:+.2f} ({unrealized_ticks:+.1f} ticks)")
            
            # Show trailing stop if available
            if position_info.get('trailing_stop_price'):
                print(f"   🛡️ Trailing Stop: ${position_info['trailing_stop_price']:.2f}")
        else:
            print(f"   📍 Position: NONE")
        
        print(f"   📊 Trades Today: {self.trades_today}/{self.max_daily_trades}")
        print(f"   💰 Daily P&L: ${self.daily_pnl:+.2f} (Target: ${self.daily_profit_target:.2f})")
        
        if self.consecutive_losses > 0:
            print(f"   🔴 Loss Streak: {self.consecutive_losses}")
        
        # Enhanced performance summary
        if hasattr(self.performance_monitor, 'get_enhanced_performance_summary'):
            perf = self.performance_monitor.get_enhanced_performance_summary()
            if perf and perf.get('total_trades', 0) > 0:
                print(f"   🎯 Win Rate: {perf['win_rate']:.1f}%")
                print(f"   📈 Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"   💪 Avg Strength: {perf.get('avg_signal_strength', 0):.2f}")
        
        # ML stats if available
        if self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            if ml_stats['model_trained']:
                print(f"   🤖 ML Accuracy: {ml_stats['accuracy']:.1f}%")
    
    async def shutdown(self):
        """Enhanced graceful shutdown"""
        
        print("\n🛑 Shutting down aggressive bot...")
        self.is_running = False
        
        try:
            # Close any open position
            position_info = self.trading_logic.get_position_info()
            if position_info['position']:
                print("🔄 Closing open position...")
                current_price = self.data_collector.get_current_price()
                self.close_enhanced_position({'price': current_price}, "Bot shutdown")
            
            # Stop data feed
            self.data_collector.stop_data_feed()
            
            # Retrain ML model if available
            if self.ml_interface:
                print("🤖 Retraining ML model...")
                self.ml_interface.force_retrain()
            
            # Generate enhanced final report
            print("\n📊 FINAL AGGRESSIVE STRATEGY REPORT:")
            if hasattr(self.performance_monitor, 'print_enhanced_dashboard'):
                self.performance_monitor.print_enhanced_dashboard()
            else:
                self.performance_monitor.print_performance_summary()
            
            # Export analytics if available
            if hasattr(self.performance_monitor, 'export_enhanced_analytics'):
                export_file = self.performance_monitor.export_enhanced_analytics()
                if export_file:
                    print(f"\n📁 Analytics exported to: {export_file}")
            
            # Final session summary
            session_duration = datetime.now() - self.session_start
            print(f"\n⏱️ Session Duration: {session_duration}")
            print(f"📊 Total Trades: {self.trades_today}")
            print(f"💰 Session P&L: ${self.daily_pnl:+.2f}")
            
            if self.daily_pnl > 0:
                print(f"🎉 Profitable session! Target: ${self.daily_profit_target:.2f}")
            
            # Cleanup
            if hasattr(self.performance_monitor, 'cleanup'):
                self.performance_monitor.cleanup()
            
            print("✅ Aggressive bot shutdown completed")
            
        except Exception as e:
            logging.error(f"Shutdown error: {e}")


async def main():
    """Main entry point for aggressive bot"""
    
    print("🚀 Aggressive XAU/USD Tick Trading Bot")
    print("=" * 50)
    
    # Display configuration status
    if AGGRESSIVE_CONFIG_AVAILABLE:
        print("✅ Aggressive configuration loaded")
    else:
        print("⚠️ Using standard configuration")
    
    if AGGRESSIVE_LOGIC_AVAILABLE:
        print("✅ Aggressive trading logic loaded")
    else:
        print("⚠️ Using standard trading logic")
    
    if ENHANCED_MONITOR_AVAILABLE:
        print("✅ Enhanced performance monitor loaded")
    else:
        print("⚠️ Using standard performance monitor")
    
    # Check configuration
    if not ALPACA_CONFIG['api_key']:
        print("⚠️ No API keys - running in simulation mode")
        print("   Edit config.py to add Alpaca credentials")
        print()
    
    # Create and start aggressive bot
    bot = AggressiveXAUUSDBot()
    await bot.start_trading()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Aggressive bot terminated by user")
    except Exception as e:
        print(f"\n❌ Aggressive bot crashed: {e}")
        logging.error(f"Fatal error: {e}")
        sys.exit(1)