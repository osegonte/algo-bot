#!/usr/bin/env python3
"""
XAU/USD Tick Trading Bot - Enhanced Main Entry Point
Focused scalping system for gold trading with tick-level precision
"""

import sys
import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import configuration
try:
    from config import ALPACA_CONFIG, STRATEGY_CONFIG, BOT_CONFIG
    print("✅ Configuration loaded successfully")
except ImportError:
    print("⚠️  config.py not found, using default settings")
    ALPACA_CONFIG = {
        'paper_trading': True,
        'api_key': '',
        'secret_key': '',
        'symbol': 'XAUUSD',  # Changed to focus on gold
        'quantity': 0.1      # Smaller position size for XAU/USD
    }
    STRATEGY_CONFIG = {
        'tick_threshold': 0.5,   # $0.50 movement threshold for gold
        'profit_target': 2.0,    # $2.00 profit target
        'stop_loss': 1.0,        # $1.00 stop loss
        'profit_target_ticks': 4, # 4 ticks profit target
        'stop_loss_ticks': 2     # 2 ticks stop loss
    }
    BOT_CONFIG = {
        'log_level': 'INFO',
        'log_file': 'xauusd_trades.csv',
        'status_update_interval': 10,  # More frequent updates for tick trading
        'max_position_time': 60        # Maximum 60 seconds in position
    }

# Import core modules with error handling
try:
    from data_collection import AdvancedTickCollector
    from trading_logic import TradingLogic
    from trade_execution import TradeExecutor
    from logger import EnhancedTradeLogger
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Please ensure all required modules are in the same directory")
    sys.exit(1)

# Optional ML interface
try:
    from ml_interface import MLInterface, ML_CONFIG
    ML_AVAILABLE = True
    print("✅ ML interface available")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML interface not available - continuing without ML features")


class XAUUSDTickTradingBot:
    """Enhanced XAU/USD Tick Trading Bot with ML Integration"""
    
    def __init__(self):
        """Initialize the XAU/USD tick trading bot"""
        
        # Core configuration
        self.symbol = ALPACA_CONFIG['symbol']
        self.quantity = ALPACA_CONFIG['quantity']
        self.paper_trading = ALPACA_CONFIG['paper_trading']
        self.api_key = ALPACA_CONFIG['api_key']
        self.secret_key = ALPACA_CONFIG['secret_key']
        
        # Bot state
        self.is_running = False
        self.trades_today = 0
        self.max_trades_per_day = 50  # Risk management
        self.daily_pnl = 0.0
        self.max_daily_loss = -50.0  # $50 daily loss limit
        
        # Initialize core components
        self.setup_logging()
        self.initialize_components()
        
        # Performance tracking
        self.session_start_time = datetime.now()
        self.last_signal_time = None
        self.signal_cooldown = 2.0  # 2 seconds between signals
        
        print(f"\n🤖 XAU/USD Tick Trading Bot Initialized")
        self.display_configuration()
    
    def setup_logging(self):
        """Setup enhanced logging for tick trading"""
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Main log file
        logging.basicConfig(
            level=getattr(logging, BOT_CONFIG['log_level']),
            format=log_format,
            handlers=[
                logging.FileHandler(f'logs/xauusd_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Tick-specific logger
        self.tick_logger = logging.getLogger('tick_data')
        tick_handler = logging.FileHandler(f'logs/tick_data_{datetime.now().strftime("%Y%m%d")}.log')
        tick_handler.setFormatter(logging.Formatter(log_format))
        self.tick_logger.addHandler(tick_handler)
        self.tick_logger.setLevel(logging.DEBUG)
    
    def initialize_components(self):
        """Initialize all trading components"""
        
        print("🔧 Initializing trading components...")
        
        # 1. Enhanced data collector for XAU/USD
        self.data_collector = AdvancedTickCollector(self.symbol)
        
        # 2. Enhanced trading logic with tick-specific parameters
        self.trading_logic = TradingLogic(
            tick_threshold=STRATEGY_CONFIG['tick_threshold'],
            profit_target=STRATEGY_CONFIG['profit_target'],
            stop_loss=STRATEGY_CONFIG['stop_loss'],
            quantity=self.quantity
        )
        
        # Set XAU/USD specific parameters
        self.trading_logic.tick_size = 0.10  # XAU/USD tick size
        self.trading_logic.profit_target_ticks = STRATEGY_CONFIG.get('profit_target_ticks', 4)
        self.trading_logic.stop_loss_ticks = STRATEGY_CONFIG.get('stop_loss_ticks', 2)
        self.trading_logic.max_position_time = BOT_CONFIG.get('max_position_time', 60)
        
        # 3. Trade executor with Alpaca integration
        self.trade_executor = TradeExecutor(
            paper_trading=self.paper_trading,
            alpaca_api_key=self.api_key,
            alpaca_secret_key=self.secret_key
        )
        
        # 4. Enhanced trade logger
        self.trade_logger = EnhancedTradeLogger(
            log_file=BOT_CONFIG['log_file'],
            log_level=getattr(logging, BOT_CONFIG['log_level']),
            use_database=True,
            database_file='xauusd_trades.db'
        )
        
        # 5. ML interface (if available)
        if ML_AVAILABLE:
            self.ml_interface = MLInterface(ML_CONFIG)
            print("🤖 ML interface initialized")
        else:
            self.ml_interface = None
        
        # Setup data collector callbacks
        self.setup_data_callbacks()
        
        print("✅ All components initialized successfully")
    
    def setup_data_callbacks(self):
        """Setup callbacks for real-time data processing"""
        
        # Tick callback for real-time processing
        def on_tick_received(tick_data):
            """Process each incoming tick"""
            try:
                self.tick_logger.debug(f"Tick: {tick_data['price']:.4f} @ {tick_data['timestamp']}")
                
                # Process with ML if available
                if self.ml_interface:
                    ml_insights = self.ml_interface.process_tick(tick_data)
                    self.process_ml_signal(tick_data, ml_insights)
                else:
                    self.process_traditional_signal(tick_data)
                    
            except Exception as e:
                logging.error(f"Error processing tick: {e}")
        
        # Analytics callback for market analysis
        def on_analytics_update(analytics):
            """Process market analytics updates"""
            try:
                # Update trading logic with market conditions
                self.update_market_conditions(analytics)
                
                # Log key metrics
                if analytics.get('tick_count', 0) % 100 == 0:  # Every 100 ticks
                    self.log_market_status(analytics)
                    
            except Exception as e:
                logging.error(f"Error processing analytics: {e}")
        
        # Register callbacks
        self.data_collector.add_tick_callback(on_tick_received)
        self.data_collector.add_analytics_callback(on_analytics_update)
    
    def display_configuration(self):
        """Display current bot configuration"""
        
        trading_mode = "📄 PAPER" if self.paper_trading else "💰 LIVE"
        api_status = "🔑 CONNECTED" if (self.api_key and self.secret_key) else "🔓 LOCAL SIM"
        ml_status = "🤖 ENABLED" if self.ml_interface else "🔓 DISABLED"
        
        print(f"   📊 Symbol: {self.symbol}")
        print(f"   💹 Mode: {trading_mode}")
        print(f"   🔗 API: {api_status}")
        print(f"   🤖 ML: {ml_status}")
        print(f"   📈 Position Size: {self.quantity} oz")
        print(f"   🎯 Profit Target: {STRATEGY_CONFIG['profit_target_ticks']} ticks (${STRATEGY_CONFIG['profit_target']})")
        print(f"   🛡️  Stop Loss: {STRATEGY_CONFIG['stop_loss_ticks']} ticks (${STRATEGY_CONFIG['stop_loss']})")
        print(f"   ⏱️  Max Position Time: {BOT_CONFIG['max_position_time']}s")
        print(f"   📊 Max Daily Trades: {self.max_trades_per_day}")
        print(f"   💸 Max Daily Loss: ${self.max_daily_loss}")
    
    async def start_trading(self):
        """Start the tick trading bot"""
        
        print("\n" + "="*80)
        print("           🚀 STARTING XAU/USD TICK TRADING BOT")
        print("="*80)
        
        try:
            # Pre-flight checks
            if not await self.run_preflight_checks():
                print("❌ Pre-flight checks failed. Aborting.")
                return
            
            # Start data collection
            print("🌐 Connecting to market data...")
            self.data_collector.stream_tick_data(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper_trading=self.paper_trading
            )
            
            # Wait for data connection
            print("⏳ Waiting for market data connection...")
            await self.wait_for_data_connection()
            
            # Display account information
            await self.display_account_info()
            
            # Start main trading loop
            print("🔄 Starting tick trading loop...")
            print("⏹️  Press Ctrl+C to stop the bot")
            print("-" * 80)
            
            self.is_running = True
            await self.run_main_trading_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
            await self.shutdown()
        except Exception as e:
            logging.error(f"Critical error in trading bot: {e}")
            await self.emergency_shutdown()
    
    async def run_preflight_checks(self) -> bool:
        """Run pre-flight checks before starting trading"""
        
        print("🔍 Running pre-flight checks...")
        
        # Check market hours
        market_status = self.trade_executor.check_market_hours(self.symbol)
        if not market_status.get('is_open', True):
            print(f"⚠️  Market closed. Next open: {market_status.get('next_open', 'Unknown')}")
            # For XAU/USD, we can trade 24/5, so this is informational
        
        # Check account connectivity
        if self.api_key and self.secret_key:
            account_info = self.trade_executor.get_position_info()
            if account_info.get('balance', 0) < 100:
                print("⚠️  Low account balance detected")
                return False
        
        # Check for existing positions
        positions = self.trade_executor.get_position_info().get('positions', {})
        if self.symbol in positions and positions[self.symbol].get('quantity', 0) != 0:
            print(f"⚠️  Existing {self.symbol} position detected: {positions[self.symbol]}")
            # Ask user what to do or auto-close
        
        print("✅ Pre-flight checks completed")
        return True
    
    async def wait_for_data_connection(self, max_wait: int = 30):
        """Wait for data connection to be established"""
        
        wait_time = 0
        while wait_time < max_wait:
            stats = self.data_collector.get_connection_stats()
            
            if stats['is_connected'] and stats['ticks_processed'] > 0:
                current_price = None
                if self.data_collector.tick_buffer:
                    current_price = self.data_collector.tick_buffer[-1].price
                
                print(f"✅ Market data connected")
                print(f"   📊 Current {self.symbol}: ${current_price:.2f}" if current_price else "")
                print(f"   📈 Data source: {stats['data_source']}")
                print(f"   ⚡ Tick rate: {stats['ticks_per_second']:.1f}/sec")
                return True
            
            await asyncio.sleep(1)
            wait_time += 1
            
            if wait_time % 5 == 0:
                print(f"   Still waiting... ({wait_time}/{max_wait})")
        
        print("❌ Failed to establish data connection")
        return False
    
    async def display_account_info(self):
        """Display account information"""
        
        if self.api_key and self.secret_key:
            account_info = self.trade_executor.get_position_info()
            print(f"💰 Account Information:")
            print(f"   Balance: ${account_info.get('balance', 0):,.2f}")
            print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            print(f"   Cash: ${account_info.get('cash', 0):,.2f}")
        else:
            print("💰 Running in simulation mode with virtual account")
    
    async def run_main_trading_loop(self):
        """Main trading loop with enhanced tick processing"""
        
        logging.info("🔄 Main trading loop started")
        
        last_status_time = time.time()
        last_performance_time = time.time()
        status_interval = BOT_CONFIG['status_update_interval']
        performance_interval = 60  # 1 minute
        
        while self.is_running:
            try:
                # Check daily limits
                if not self.check_daily_limits():
                    print("🛑 Daily limits reached. Stopping trading.")
                    break
                
                # Get latest market analysis
                analysis = self.data_collector.get_tick_analysis()
                
                if analysis and analysis.get('tick_count', 0) > 0:
                    current_price = analysis.get('current_price', 0)
                    
                    # Check for position exit first
                    if self.trading_logic.position:
                        await self.check_position_exit(current_price, analysis)
                    
                    # Check for new entry signals
                    if not self.trading_logic.position:
                        await self.check_entry_signals(current_price, analysis)
                
                # Periodic status updates
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    self.log_status_update(analysis)
                    last_status_time = current_time
                
                # Performance updates
                if current_time - last_performance_time > performance_interval:
                    self.log_performance_update()
                    last_performance_time = current_time
                
                # Brief pause to prevent excessive CPU usage
                await asyncio.sleep(0.05)  # 50ms for high-frequency trading
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(1)
        
        await self.shutdown()
    
    def process_ml_signal(self, tick_data: Dict, ml_insights: Dict):
        """Process ML-generated trading insights"""
        
        if not ml_insights or ml_insights.get('confidence', 0) < 0.6:
            return
        
        signal = ml_insights.get('signal', 'hold')
        confidence = ml_insights.get('confidence', 0)
        
        # Apply signal cooldown
        if self.last_signal_time and (time.time() - self.last_signal_time) < self.signal_cooldown:
            return
        
        if signal in ['buy', 'sell'] and not self.trading_logic.position:
            # Create enhanced signal with ML insights
            self.execute_ml_signal(signal, tick_data, ml_insights)
    
    def process_traditional_signal(self, tick_data: Dict):
        """Process traditional signal generation"""
        
        # Get current market analysis
        analysis = self.data_collector.get_tick_analysis()
        
        if not analysis:
            return
        
        # Generate signal using traditional logic
        signal = self.trading_logic.evaluate_tick_signals(tick_data, analysis)
        
        if signal.signal_type in ['buy', 'sell'] and signal.confidence > 0.6:
            # Apply signal cooldown
            if self.last_signal_time and (time.time() - self.last_signal_time) < self.signal_cooldown:
                return
            
            self.execute_traditional_signal(signal, tick_data)
    
    async def execute_ml_signal(self, signal: str, tick_data: Dict, ml_insights: Dict):
        """Execute ML-generated trading signal"""
        
        current_price = tick_data.get('price', 0)
        confidence = ml_insights.get('confidence', 0)
        
        print(f"\n🤖 ML SIGNAL: {signal.upper()} @ ${current_price:.4f}")
        print(f"   🎯 Confidence: {confidence:.2f}")
        print(f"   📊 Features: {ml_insights.get('feature_count', 0)}")
        
        # Execute the trade
        await self.execute_trade_signal(signal, current_price, 
                                       confidence=confidence,
                                       reasoning=f"ML signal with {confidence:.2f} confidence",
                                       ml_insights=ml_insights)
    
    async def execute_traditional_signal(self, signal, tick_data: Dict):
        """Execute traditional trading signal"""
        
        current_price = tick_data.get('price', 0)
        
        print(f"\n📈 SIGNAL: {signal.signal_type.upper()} @ ${current_price:.4f}")
        print(f"   🎯 Confidence: {signal.confidence:.2f}")
        print(f"   💡 Reasoning: {signal.reasoning}")
        print(f"   💰 Expected Profit: {signal.expected_profit_ticks} ticks")
        print(f"   🛡️  Max Risk: {signal.max_risk_ticks} ticks")
        
        # Execute the trade
        await self.execute_trade_signal(signal.signal_type, current_price,
                                       confidence=signal.confidence,
                                       reasoning=signal.reasoning,
                                       expected_profit=signal.expected_profit_ticks,
                                       max_risk=signal.max_risk_ticks)
    
    async def execute_trade_signal(self, signal_type: str, current_price: float,
                                  confidence: float = 0.0, reasoning: str = "",
                                  expected_profit: float = 0, max_risk: float = 0,
                                  ml_insights: Dict = None):
        """Execute a trading signal with enhanced error handling"""
        
        try:
            # Place the order
            trade = self.trade_executor.place_order(
                symbol=self.symbol,
                side=signal_type,
                quantity=self.quantity,
                price=current_price
            )
            
            if trade.status == "filled":
                # Update trading logic
                self.trading_logic.update_position(signal_type, current_price, trade.timestamp)
                
                # Prepare market conditions for logging
                analysis = self.data_collector.get_tick_analysis()
                market_conditions = {
                    'current_price': current_price,
                    'tick_velocity': analysis.get('tick_velocity', 0),
                    'spread': analysis.get('current_spread', 0),
                    'volatility': analysis.get('price_volatility', 0),
                    'order_flow': analysis.get('order_flow_imbalance', 0),
                    'expected_profit_ticks': expected_profit,
                    'max_risk_ticks': max_risk
                }
                
                # Add ML insights if available
                if ml_insights:
                    market_conditions.update({
                        'ml_confidence': ml_insights.get('confidence', 0),
                        'ml_features': ml_insights.get('feature_count', 0),
                        'ml_patterns': ml_insights.get('patterns', {})
                    })
                
                # Log the trade
                self.trade_logger.log_trade(
                    trade, 
                    trade_type="entry",
                    signal_confidence=confidence,
                    signal_reasoning=reasoning,
                    market_conditions=market_conditions
                )
                
                # Update counters
                self.trades_today += 1
                self.last_signal_time = time.time()
                
                print(f"✅ ENTRY EXECUTED: {signal_type.upper()} position opened")
                print(f"   📊 Price: ${current_price:.4f}")
                print(f"   📦 Quantity: {self.quantity} oz")
                print(f"   🆔 Trade ID: {trade.trade_id}")
                print(f"   📊 Trade #{self.trades_today} today")
                
            else:
                print(f"❌ ENTRY FAILED: {trade.status}")
                logging.error(f"Trade execution failed: {trade.status}")
                
        except Exception as e:
            logging.error(f"Error executing trade signal: {e}")
            print(f"❌ Trade execution error: {e}")
    
    async def check_position_exit(self, current_price: float, analysis: Dict):
        """Check for position exit conditions"""
        
        if not self.trading_logic.position:
            return
        
        # Create tick data for exit evaluation
        tick_data = {
            'price': current_price,
            'timestamp': datetime.now(),
            'size': analysis.get('avg_tick_size', 1),
            'spread': analysis.get('current_spread', 0)
        }
        
        # Check exit signal
        exit_signal = self.trading_logic._check_exit_conditions(tick_data, analysis)
        
        if exit_signal.signal_type == 'close':
            await self.execute_position_exit(current_price, exit_signal.reasoning)
    
    async def execute_position_exit(self, current_price: float, reasoning: str):
        """Execute position exit"""
        
        try:
            if not self.trading_logic.position:
                return
            
            print(f"\n📉 EXIT SIGNAL: {reasoning} @ ${current_price:.4f}")
            
            # Determine opposite side
            original_side = 'buy' if self.trading_logic.position == 'long' else 'sell'
            close_side = 'sell' if original_side == 'buy' else 'buy'
            
            # Execute closing trade
            exit_trade = self.trade_executor.place_order(
                symbol=self.symbol,
                side=close_side,
                quantity=self.quantity,
                price=current_price
            )
            
            if exit_trade.status == "filled":
                # Calculate P&L
                pnl = self.trading_logic.calculate_unrealized_pnl(current_price)
                pnl_ticks = self.trading_logic.calculate_unrealized_pnl_ticks(current_price)
                
                # Update daily P&L
                self.daily_pnl += pnl
                
                # Get market conditions for logging
                analysis = self.data_collector.get_tick_analysis()
                market_conditions = {
                    'exit_price': current_price,
                    'pnl_dollars': pnl,
                    'pnl_ticks': pnl_ticks,
                    'time_in_position': (datetime.now() - self.trading_logic.entry_time).total_seconds() if self.trading_logic.entry_time else 0
                }
                
                # Log the exit trade
                self.trade_logger.log_trade(
                    exit_trade,
                    trade_type="exit",
                    signal_confidence=1.0,
                    signal_reasoning=reasoning,
                    market_conditions=market_conditions
                )
                
                # Record outcome for ML if available
                if self.ml_interface:
                    features = analysis if analysis else {}
                    self.ml_interface.record_trade_outcome(features, close_side, pnl)
                
                # Update trading logic
                self.trading_logic.update_position('close', current_price, exit_trade.timestamp)
                
                # Display results
                pnl_symbol = "🟢" if pnl > 0 else "🔴"
                print(f"✅ EXIT EXECUTED: Position closed")
                print(f"   📊 Price: ${current_price:.4f}")
                print(f"   💰 P&L: ${pnl:+.2f} ({pnl_ticks:+.1f} ticks) {pnl_symbol}")
                print(f"   🆔 Trade ID: {exit_trade.trade_id}")
                print(f"   💰 Daily P&L: ${self.daily_pnl:+.2f}")
                
                # Update performance
                self.log_trade_performance()
                
            else:
                print(f"❌ EXIT FAILED: {exit_trade.status}")
                logging.error(f"Exit trade failed: {exit_trade.status}")
                
        except Exception as e:
            logging.error(f"Error executing position exit: {e}")
            print(f"❌ Exit execution error: {e}")
    
    async def check_entry_signals(self, current_price: float, analysis: Dict):
        """Check for new entry signals"""
        
        # Create tick data for signal evaluation
        tick_data = {
            'price': current_price,
            'timestamp': datetime.now(),
            'size': analysis.get('avg_tick_size', 1),
            'spread': analysis.get('current_spread', 0),
            'bid': current_price - analysis.get('current_spread', 0.1) / 2,
            'ask': current_price + analysis.get('current_spread', 0.1) / 2
        }
        
        # Process with ML if available
        if self.ml_interface:
            ml_insights = self.ml_interface.process_tick(tick_data)
            if ml_insights.get('signal') in ['buy', 'sell'] and ml_insights.get('confidence', 0) > 0.6:
                await self.execute_ml_signal(ml_insights['signal'], tick_data, ml_insights)
        else:
            # Use traditional signal generation
            signal = self.trading_logic.evaluate_tick_signals(tick_data, analysis)
            if signal.signal_type in ['buy', 'sell'] and signal.confidence > 0.6:
                await self.execute_traditional_signal(signal, tick_data)
    
    def check_daily_limits(self) -> bool:
        """Check if daily trading limits are exceeded"""
        
        # Check trade count limit
        if self.trades_today >= self.max_trades_per_day:
            logging.warning(f"Daily trade limit reached: {self.trades_today}/{self.max_trades_per_day}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            logging.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    def update_market_conditions(self, analytics: Dict):
        """Update trading logic with current market conditions"""
        
        # Update adaptive parameters if enabled
        if hasattr(self.trading_logic, 'adaptive_thresholds') and self.trading_logic.adaptive_thresholds:
            # Get performance metrics
            perf = self.trade_logger.get_performance_summary()
            win_rate = perf.get('win_rate', 50) / 100
            avg_pnl = perf.get('total_pnl', 0) / max(perf.get('total_trades', 1), 1)
            
            # Adjust parameters
            self.trading_logic.adjust_parameters(win_rate, avg_pnl)
    
    def log_status_update(self, analysis: Optional[Dict]):
        """Log periodic status updates"""
        
        current_time = datetime.now().strftime('%H:%M:%S')
        
        if analysis:
            current_price = analysis.get('current_price', 0)
            tick_velocity = analysis.get('tick_velocity', 0)
            spread = analysis.get('current_spread', 0)
            volatility = analysis.get('price_volatility', 0)
            
            print(f"\n📊 STATUS UPDATE - {current_time}")
            print(f"   💹 {self.symbol}: ${current_price:.4f}")
            print(f"   ⚡ Tick Rate: {tick_velocity:.1f}/sec")
            print(f"   📏 Spread: ${spread:.3f}")
            print(f"   📈 Volatility: {volatility:.3f}")
            
            # Position info
            position_info = self.trading_logic.get_position_info()
            if position_info['position']:
                unrealized_pnl = self.trading_logic.calculate_unrealized_pnl(current_price)
                unrealized_ticks = self.trading_logic.calculate_unrealized_pnl_ticks(current_price)
                print(f"   📍 Position: {position_info['position'].upper()} @ ${position_info['entry_price']:.4f}")
                print(f"   💰 Unrealized: ${unrealized_pnl:+.2f} ({unrealized_ticks:+.1f} ticks)")
            else:
                print(f"   📍 Position: NONE - Scanning for signals...")
            
            print(f"   📊 Trades Today: {self.trades_today}/{self.max_trades_per_day}")
            print(f"   💰 Daily P&L: ${self.daily_pnl:+.2f}")
        
        # ML insights if available
        if self.ml_interface:
            ml_insights = self.ml_interface.get_ml_insights()
            if ml_insights.get('model_trained'):
                print(f"   🤖 ML Accuracy: {ml_insights.get('recent_accuracy', 0):.1%}")
    
    def log_performance_update(self):
        """Log detailed performance metrics"""
        
        try:
            # Get comprehensive performance summary
            perf = self.trade_logger.get_performance_summary()
            
            if perf.get('total_trades', 0) > 0:
                print(f"\n📈 PERFORMANCE UPDATE:")
                print(f"   🔢 Total Trades: {perf['total_trades']}")
                print(f"   🎯 Win Rate: {perf['win_rate']:.1f}%")
                print(f"   💰 Total P&L: ${perf['total_pnl']:+.2f}")
                print(f"   📊 Avg P&L: ${perf['total_pnl']/perf['total_trades']:+.2f}")
                print(f"   📉 Max Drawdown: ${perf['max_drawdown']:.2f}")
                
                if perf.get('recent_win_rate') is not None:
                    print(f"   🔥 Recent Win Rate: {perf['recent_win_rate']:.1f}%")
                
                # ML performance if available
                if self.ml_interface:
                    ml_stats = self.ml_interface.get_ml_insights()
                    print(f"   🤖 ML Training Samples: {ml_stats.get('training_samples', 0)}")
        
        except Exception as e:
            logging.error(f"Error logging performance update: {e}")
    
    def log_trade_performance(self):
        """Log individual trade performance"""
        
        try:
            # Quick performance summary after each trade
            perf = self.trade_logger.get_performance_summary()
            if perf.get('total_trades', 0) > 0:
                print(f"\n📊 QUICK STATS:")
                print(f"   📈 Session Trades: {perf['total_trades']}")
                print(f"   🎯 Win Rate: {perf['win_rate']:.1f}%")
                print(f"   💰 Session P&L: ${perf['total_pnl']:+.2f}")
        
        except Exception as e:
            logging.error(f"Error logging trade performance: {e}")
    
    def log_market_status(self, analytics: Dict):
        """Log detailed market analysis"""
        
        try:
            self.tick_logger.info(f"Market Analysis: {analytics}")
            
            # Log key market metrics
            if analytics.get('tick_count', 0) % 500 == 0:  # Every 500 ticks
                logging.info(f"Market Status - Velocity: {analytics.get('tick_velocity', 0):.1f}, "
                           f"Spread: {analytics.get('current_spread', 0):.3f}, "
                           f"Volatility: {analytics.get('price_volatility', 0):.3f}")
        
        except Exception as e:
            logging.error(f"Error logging market status: {e}")
    
    async def shutdown(self):
        """Graceful shutdown of the trading bot"""
        
        print("\n🛑 Initiating graceful shutdown...")
        
        self.is_running = False
        
        try:
            # Close any open positions
            if self.trading_logic.position:
                print("🔄 Closing open position...")
                current_price = 0
                if self.data_collector.tick_buffer:
                    current_price = self.data_collector.tick_buffer[-1].price
                
                if current_price > 0:
                    await self.execute_position_exit(current_price, "Bot shutdown")
            
            # Stop data collection
            print("📡 Stopping data collection...")
            self.data_collector.stop_stream()
            
            # Force ML model training if enough samples
            if self.ml_interface:
                print("🤖 Saving ML model...")
                self.ml_interface.force_retrain()
            
            # Generate final reports
            print("📊 Generating final reports...")
            self.generate_final_reports()
            
            # Cleanup resources
            self.trade_logger.cleanup()
            
            print("✅ Shutdown completed successfully")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            print(f"⚠️  Error during shutdown: {e}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown in case of critical errors"""
        
        print("\n🚨 EMERGENCY SHUTDOWN INITIATED")
        
        self.is_running = False
        
        try:
            # Try to close positions immediately
            if self.trading_logic.position:
                try:
                    # Get last known price
                    current_price = 0
                    if self.data_collector.tick_buffer:
                        current_price = self.data_collector.tick_buffer[-1].price
                    
                    if current_price > 0:
                        # Emergency position close
                        close_side = 'sell' if self.trading_logic.position == 'long' else 'buy'
                        emergency_trade = self.trade_executor.place_order(
                            symbol=self.symbol,
                            side=close_side,
                            quantity=self.quantity,
                            price=current_price
                        )
                        
                        if emergency_trade.status == "filled":
                            print(f"✅ Emergency position closed @ ${current_price:.4f}")
                        else:
                            print(f"❌ Failed to close position: {emergency_trade.status}")
                
                except Exception as e:
                    logging.error(f"Failed to close position during emergency: {e}")
            
            # Stop data collection
            self.data_collector.stop_stream()
            
            # Save what we can
            try:
                self.trade_logger.cleanup()
            except:
                pass
            
            print("🚨 Emergency shutdown completed")
            
        except Exception as e:
            logging.critical(f"Critical error during emergency shutdown: {e}")
    
    def generate_final_reports(self):
        """Generate comprehensive final reports"""
        
        try:
            # Session summary
            session_duration = datetime.now() - self.session_start_time
            
            print("\n" + "="*80)
            print("           📊 FINAL SESSION REPORT")
            print("="*80)
            print(f"Session Duration: {session_duration}")
            print(f"Trades Executed: {self.trades_today}")
            print(f"Daily P&L: ${self.daily_pnl:+.2f}")
            
            # Performance report
            self.trade_logger.print_performance_report()
            
            # Data collection stats
            data_stats = self.data_collector.get_connection_stats()
            print(f"\n📡 DATA COLLECTION STATS:")
            print(f"   Ticks Processed: {data_stats['ticks_processed']:,}")
            print(f"   Average Tick Rate: {data_stats['ticks_per_second']:.1f}/sec")
            print(f"   Data Source: {data_stats['data_source']}")
            
            # Trading logic stats
            strategy_stats = self.trading_logic.get_strategy_stats()
            print(f"\n🎯 STRATEGY STATS:")
            print(f"   Total Signals: {strategy_stats['total_signals']}")
            print(f"   Successful Signals: {strategy_stats['successful_signals']}")
            print(f"   Signal Win Rate: {strategy_stats['win_rate']:.1%}")
            
            # ML stats if available
            if self.ml_interface:
                ml_stats = self.ml_interface.get_ml_insights()
                print(f"\n🤖 ML STATS:")
                print(f"   Model Trained: {ml_stats['model_trained']}")
                print(f"   Training Samples: {ml_stats['training_samples']}")
                print(f"   Recent Accuracy: {ml_stats['recent_accuracy']:.1%}")
            
            # Account final status
            if self.api_key and self.secret_key:
                account_info = self.trade_executor.get_position_info()
                print(f"\n💰 FINAL ACCOUNT STATUS:")
                print(f"   Balance: ${account_info.get('balance', 0):,.2f}")
                print(f"   Cash: ${account_info.get('cash', 0):,.2f}")
            
            print(f"\n📁 Check {BOT_CONFIG['log_file']} for detailed trade log")
            print("="*80)
            
        except Exception as e:
            logging.error(f"Error generating final reports: {e}")


async def main():
    """Main entry point"""
    
    # Display startup banner
    print("\n" + "="*80)
    print("           🥇 XAU/USD TICK TRADING BOT v3.0")
    print("           Advanced Scalping System for Gold")
    print("="*80)
    
    # Check configuration
    if not ALPACA_CONFIG['api_key'] or not ALPACA_CONFIG['secret_key']:
        print("⚠️  NO ALPACA API KEYS DETECTED")
        print("   📝 Edit config.py to add your Alpaca credentials")
        print("   🌐 Get free keys at: https://app.alpaca.markets/")
        print("   🔒 Running in SIMULATION mode")
        print("")
    
    # Display ML status
    if ML_AVAILABLE:
        print("🤖 Machine Learning features ENABLED")
    else:
        print("⚠️  Machine Learning features DISABLED")
        print("   📦 Install ML dependencies: pip install scikit-learn pandas")
    
    print()
    
    # Create and start the bot
    bot = XAUUSDTickTradingBot()
    
    try:
        await bot.start_trading()
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received...")
        await bot.shutdown()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.error(f"Unexpected error in main: {e}")
        await bot.emergency_shutdown()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())