#!/usr/bin/env python3
"""
Simple XAU/USD Tick Trading Bot - Main Entry Point
"""

import asyncio
import logging
import time
from datetime import datetime

# Import configuration
from config import ALPACA_CONFIG, STRATEGY_CONFIG, BOT_CONFIG

# Import core modules
from data_collection import SimpleDataCollector
from trading_logic import SimpleTradingLogic
from trade_execution import SimpleTradeExecutor
from logger import SimpleTradeLogger

# Import ML interface
try:
    from ml_interface import SimpleMLInterface, ML_CONFIG, ML_AVAILABLE
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False


class SimpleXAUUSDBot:
    """Simple XAU/USD Trading Bot"""
    
    def __init__(self):
        # Bot configuration
        self.symbol = ALPACA_CONFIG['symbol']
        self.quantity = ALPACA_CONFIG['quantity']
        self.api_key = ALPACA_CONFIG['api_key']
        self.secret_key = ALPACA_CONFIG['secret_key']
        
        # Bot state
        self.is_running = False
        self.trades_today = 0
        self.max_daily_trades = 20
        
        # Initialize components
        self.setup_logging()
        self.initialize_components()
        
        print(f"\n🥇 Simple XAU/USD Trading Bot Initialized")
        self.display_config()
    
    def setup_logging(self):
        """Setup simple logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_components(self):
        """Initialize trading components"""
        
        print("🔧 Initializing components...")
        
        # 1. Data collector
        self.data_collector = SimpleDataCollector(self.symbol)
        
        # 2. Trading logic
        self.trading_logic = SimpleTradingLogic(
            profit_target_ticks=STRATEGY_CONFIG['profit_target_ticks'],
            stop_loss_ticks=STRATEGY_CONFIG['stop_loss_ticks'],
            tick_size=STRATEGY_CONFIG['tick_size'],
            min_confidence=STRATEGY_CONFIG['min_confidence']
        )
        
        # 3. Trade executor
        self.trade_executor = SimpleTradeExecutor(
            paper_trading=ALPACA_CONFIG['paper_trading'],
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        # 4. Trade logger
        self.trade_logger = SimpleTradeLogger(BOT_CONFIG['log_file'])
        
        # 5. ML interface (optional)
        if ML_ENABLED and ML_AVAILABLE:
            self.ml_interface = SimpleMLInterface(ML_CONFIG)
            print("🤖 ML interface enabled")
        else:
            self.ml_interface = None
            if ML_ENABLED:
                print("⚠️ ML libraries not available - install scikit-learn")
            else:
                print("⚠️ ML interface not imported")
        
        # Setup data callback
        self.data_collector.add_tick_callback(self.on_tick_received)
        
        print("✅ All components initialized")
    
    def display_config(self):
        """Display bot configuration"""
        
        mode = "📄 PAPER" if ALPACA_CONFIG['paper_trading'] else "💰 LIVE"
        api_status = "🔑 API" if self.api_key else "🎮 SIM"
        ml_status = "🤖 ML" if self.ml_interface else "🔓 NO ML"
        
        print(f"   📊 Symbol: {self.symbol}")
        print(f"   💹 Mode: {mode}")
        print(f"   🔗 Connection: {api_status}")
        print(f"   🤖 AI: {ml_status}")
        print(f"   📦 Position Size: {self.quantity} oz")
        print(f"   🎯 Profit Target: {STRATEGY_CONFIG['profit_target_ticks']} ticks")
        print(f"   🛡️  Stop Loss: {STRATEGY_CONFIG['stop_loss_ticks']} ticks")
        print(f"   📊 Max Daily Trades: {self.max_daily_trades}")
    
    async def start_trading(self):
        """Start the trading bot"""
        
        print("\n" + "="*60)
        print("           🚀 STARTING SIMPLE XAUUSD BOT")
        print("="*60)
        
        try:
            # Start data feed
            print("📡 Starting data feed...")
            self.data_collector.start_data_feed(self.api_key, self.secret_key)
            
            # Wait for data
            await self.wait_for_data()
            
            # Display account info
            self.display_account_info()
            
            # Main trading loop
            print("🔄 Starting trading loop...")
            print("⏹️  Press Ctrl+C to stop")
            print("-" * 60)
            
            self.is_running = True
            await self.main_trading_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            await self.shutdown()
        except Exception as e:
            logging.error(f"Bot error: {e}")
            await self.shutdown()
    
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
        
        print("⚠️  Data connection timeout - continuing anyway")
        return False
    
    def display_account_info(self):
        """Display account information"""
        
        account = self.trade_executor.get_account_info()
        print(f"💰 Account Balance: ${account['balance']:,.2f}")
        print(f"💵 Available Cash: ${account['cash']:,.2f}")
    
    async def main_trading_loop(self):
        """Main trading loop"""
        
        last_status_time = time.time()
        status_interval = BOT_CONFIG['status_update_interval']
        
        while self.is_running:
            try:
                # Check daily limits
                if self.trades_today >= self.max_daily_trades:
                    print(f"🛑 Daily trade limit reached: {self.trades_today}")
                    break
                
                # Periodic status update
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    self.log_status_update()
                    last_status_time = current_time
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                logging.error(f"Trading loop error: {e}")
                await asyncio.sleep(1)
        
        await self.shutdown()
    
    def on_tick_received(self, tick_data):
        """Process incoming tick data"""
        
        try:
            # Get market analysis
            market_analysis = self.data_collector.get_market_analysis()
            
            if not market_analysis:
                return
            
            # Get ML signal if available
            ml_signal = None
            if self.ml_interface:
                ml_signal = self.ml_interface.process_tick(tick_data)
            
            # Generate trading signal (can incorporate ML)
            signal = self.trading_logic.evaluate_tick(tick_data, market_analysis)
            
            # Enhance signal with ML if available
            if ml_signal and ml_signal.signal != 'hold' and ml_signal.confidence > 0.7:
                if signal.signal_type == 'hold':  # Use ML signal if logic says hold
                    signal.signal_type = ml_signal.signal
                    signal.confidence = ml_signal.confidence
                    signal.reasoning = f"ML: {ml_signal.reasoning}"
            
            if signal.signal_type in ['buy', 'sell']:
                self.execute_signal(signal, tick_data, ml_signal)
            elif signal.signal_type == 'close':
                self.close_position(tick_data, signal.reasoning)
            
        except Exception as e:
            logging.error(f"Tick processing error: {e}")
    
    def execute_signal(self, signal, tick_data, ml_signal=None):
        """Execute a trading signal"""
        
        current_price = tick_data['price']
        
        print(f"\n📈 SIGNAL: {signal.signal_type.upper()} @ ${current_price:.2f}")
        print(f"   🎯 Confidence: {signal.confidence:.2f}")
        print(f"   💡 Reasoning: {signal.reasoning}")
        
        if ml_signal and ml_signal.signal != 'hold':
            print(f"   🤖 ML: {ml_signal.signal} ({ml_signal.confidence:.2f})")
        
        # Place order
        trade = self.trade_executor.place_order(
            self.symbol, signal.signal_type, self.quantity, current_price
        )
        
        if trade.status == "filled":
            # Update position
            self.trading_logic.update_position(signal.signal_type, current_price, trade.timestamp)
            
            # Log trade
            self.trade_logger.log_trade(trade, trade_type="entry")
            
            # Update counters
            self.trades_today += 1
            
            print(f"✅ ENTRY: {signal.signal_type.upper()} position opened")
            print(f"   📦 Quantity: {self.quantity} oz")
            print(f"   🆔 Trade ID: {trade.trade_id}")
            print(f"   📊 Trade #{self.trades_today} today")
        else:
            print(f"❌ ENTRY FAILED: {trade.status}")
    
    def close_position(self, tick_data, reasoning):
        """Close current position"""
        
        current_price = tick_data['price']
        position_info = self.trading_logic.get_position_info()
        
        if not position_info['position']:
            return
        
        print(f"\n📉 EXIT: {reasoning} @ ${current_price:.2f}")
        
        # Determine close side
        close_side = 'sell' if position_info['position'] == 'long' else 'buy'
        
        # Place closing order
        exit_trade = self.trade_executor.place_order(
            self.symbol, close_side, self.quantity, current_price
        )
        
        if exit_trade.status == "filled":
            # Calculate P&L
            pnl_usd = self.trading_logic.calculate_unrealized_pnl(current_price)
            pnl_ticks = self.trading_logic.calculate_unrealized_pnl_ticks(current_price)
            
            # Update position
            self.trading_logic.update_position('close', current_price, exit_trade.timestamp)
            
            # Log exit trade
            self.trade_logger.log_trade(exit_trade, trade_type="exit", profit_loss=pnl_usd)
            
            # Record ML outcome if available
            if self.ml_interface:
                market_analysis = self.data_collector.get_market_analysis()
                if market_analysis:
                    features = self.ml_interface.feature_extractor.extract_features()
                    self.ml_interface.record_trade_outcome(features, close_side, pnl_usd)
            
            # Display results
            pnl_symbol = "🟢" if pnl_usd > 0 else "🔴"
            print(f"✅ EXIT: Position closed")
            print(f"   💰 P&L: ${pnl_usd:+.2f} ({pnl_ticks:+.1f} ticks) {pnl_symbol}")
            print(f"   🆔 Trade ID: {exit_trade.trade_id}")
            
        else:
            print(f"❌ EXIT FAILED: {exit_trade.status}")
    
    def log_status_update(self):
        """Log periodic status update"""
        
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = self.data_collector.get_current_price()
        position_info = self.trading_logic.get_position_info()
        
        print(f"\n📊 STATUS - {current_time}")
        print(f"   💹 {self.symbol}: ${current_price:.2f}")
        
        if position_info['position']:
            unrealized_pnl = self.trading_logic.calculate_unrealized_pnl(current_price)
            unrealized_ticks = self.trading_logic.calculate_unrealized_pnl_ticks(current_price)
            print(f"   📍 Position: {position_info['position'].upper()} @ ${position_info['entry_price']:.2f}")
            print(f"   💰 Unrealized: ${unrealized_pnl:+.2f} ({unrealized_ticks:+.1f} ticks)")
        else:
            print(f"   📍 Position: NONE")
        
        print(f"   📊 Trades Today: {self.trades_today}/{self.max_daily_trades}")
        
        # Performance summary
        perf = self.trade_logger.get_performance_summary()
        if perf['total_trades'] > 0:
            print(f"   🎯 Win Rate: {perf['win_rate']:.1f}%")
            print(f"   💰 Total P&L: ${perf['total_pnl']:+.2f}")
        
        # ML stats if available
        if self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            if ml_stats['model_trained']:
                print(f"   🤖 ML Accuracy: {ml_stats['accuracy']:.1f}%")
                print(f"   📊 ML Samples: {ml_stats['training_samples']}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        print("\n🛑 Shutting down...")
        self.is_running = False
        
        try:
            # Close any open position
            position_info = self.trading_logic.get_position_info()
            if position_info['position']:
                print("🔄 Closing open position...")
                current_price = self.data_collector.get_current_price()
                self.close_position({'price': current_price}, "Bot shutdown")
            
            # Stop data feed
            self.data_collector.stop_data_feed()
            
            # Retrain ML model if available
            if self.ml_interface:
                print("🤖 Retraining ML model...")
                self.ml_interface.force_retrain()
            
            # Generate final report
            print("\n📊 FINAL SUMMARY:")
            self.trade_logger.print_performance_summary()
            
            # Cleanup
            self.trade_logger.cleanup()
            
            print("✅ Shutdown completed")
            
        except Exception as e:
            logging.error(f"Shutdown error: {e}")


async def main():
    """Main entry point"""
    
    print("🥇 Simple XAU/USD Tick Trading Bot")
    print("=" * 40)
    
    # Check configuration
    if not ALPACA_CONFIG['api_key']:
        print("⚠️  No API keys - running in simulation mode")
        print("   Edit config.py to add Alpaca credentials")
        print()
    
    # Create and start bot
    bot = SimpleXAUUSDBot()
    await bot.start_trading()


if __name__ == "__main__":
    asyncio.run(main())