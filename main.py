#!/usr/bin/env python3
"""
Scalping Trading Bot - Final Version
Integrates with Alpaca for real/paper trading with proper error handling
"""

import sys
import os
import time
import logging
from datetime import datetime

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
        'symbol': 'AAPL',
        'quantity': 1
    }
    STRATEGY_CONFIG = {
        'tick_threshold': 0.01,
        'profit_target': 0.02,
        'stop_loss': 0.01
    }
    BOT_CONFIG = {
        'log_level': 'INFO',
        'log_file': 'trades.csv',
        'status_update_interval': 30
    }

# Import core modules
try:
    from data_collection import DataCollector
    from trading_logic import TradingLogic
    from trade_execution import TradeExecutor
    from logger import TradeLogger
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Run the setup script first: chmod +x setup_fix.sh && ./setup_fix.sh")
    sys.exit(1)


class ScalpingBot:
    """Main scalping bot with Alpaca integration"""
    
    def __init__(self):
        """Initialize bot with configuration from config.py"""
        
        # Load configuration
        self.symbol = ALPACA_CONFIG['symbol']
        self.quantity = ALPACA_CONFIG['quantity']
        self.paper_trading = ALPACA_CONFIG['paper_trading']
        self.api_key = ALPACA_CONFIG['api_key']
        self.secret_key = ALPACA_CONFIG['secret_key']
        
        self.is_running = False
        
        # Initialize core modules
        print(f"🤖 Initializing bot for {self.symbol}...")
        
        self.data_collector = DataCollector(self.symbol)
        self.trading_logic = TradingLogic(
            tick_threshold=STRATEGY_CONFIG['tick_threshold'],
            profit_target=STRATEGY_CONFIG['profit_target'],
            stop_loss=STRATEGY_CONFIG['stop_loss'],
            quantity=self.quantity
        )
        
        # Initialize trade executor with Alpaca credentials
        self.trade_executor = TradeExecutor(
            paper_trading=self.paper_trading,
            alpaca_api_key=self.api_key,
            alpaca_secret_key=self.secret_key
        )
        
        self.trade_logger = TradeLogger(
            log_file=BOT_CONFIG['log_file']
        )
        
        # Track current trade
        self.current_entry_trade = None
        
        # Display configuration
        trading_mode = "📄 PAPER" if self.paper_trading else "💰 LIVE"
        api_status = "🔑 CONNECTED" if (self.api_key and self.secret_key) else "🔓 LOCAL SIM"
        
        print(f"✅ Bot initialized:")
        print(f"   📊 Symbol: {self.symbol}")
        print(f"   💹 Mode: {trading_mode}")
        print(f"   🔗 API: {api_status}")
        print(f"   📈 Quantity: {self.quantity}")
        print(f"   🎯 Strategy: {STRATEGY_CONFIG['tick_threshold']}% tick threshold")
    
    def start(self):
        """Start the trading bot"""
        print("\n" + "="*60)
        print("           🚀 STARTING SCALPING BOT")
        print("="*60)
        
        try:
            # Start data collection
            print("🌐 Connecting to market data...")
            self.data_collector.start_data_feed()
            
            # Wait for initial data
            print("⏳ Waiting for market data...")
            retry_count = 0
            while not self.data_collector.is_data_available() and retry_count < 30:
                time.sleep(1)
                retry_count += 1
                if retry_count % 5 == 0:
                    print(f"   Still waiting... ({retry_count}/30)")
            
            if not self.data_collector.is_data_available():
                print("❌ Failed to establish data connection")
                print("💡 Check your internet connection and try again")
                return
            
            current_price = self.data_collector.get_latest_price()
            print(f"✅ Market data connected - Current {self.symbol}: ${current_price:.2f}")
            
            # Display account info if connected to Alpaca
            if self.api_key and self.secret_key:
                account_info = self.trade_executor.get_position_info()
                print(f"💰 Account Balance: ${account_info['balance']:,.2f}")
                print(f"💵 Buying Power: ${account_info['buying_power']:,.2f}")
            
            # Start main trading loop
            print("🔄 Starting trading loop...")
            print("⏹️  Press Ctrl+C to stop the bot")
            print("-" * 60)
            
            self.is_running = True
            self.run_trading_loop()
            
        except Exception as e:
            logging.error(f"Error starting bot: {e}")
            self.stop()
    
    def run_trading_loop(self):
        """Main trading loop"""
        logging.info("🔄 Trading loop active - monitoring market...")
        
        last_status_time = time.time()
        status_interval = BOT_CONFIG['status_update_interval']
        
        while self.is_running:
            try:
                # Get current market data
                current_price = self.data_collector.get_latest_price()
                price_change = self.data_collector.get_price_change(periods=5)
                
                if current_price is None:
                    time.sleep(0.5)
                    continue
                
                # Analyze market and get trading signal
                signal = self.trading_logic.analyze_market(current_price, price_change)
                
                # Execute trades based on signal
                if signal in ['buy', 'sell']:
                    self.execute_entry_trade(signal, current_price)
                    
                elif signal == 'close' and self.current_entry_trade:
                    self.execute_exit_trade(current_price)
                
                # Periodic status updates
                if time.time() - last_status_time > status_interval:
                    self.log_status_update(current_price, price_change)
                    last_status_time = time.time()
                
                # Brief pause
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                break
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                time.sleep(1)
        
        self.stop()
    
    def execute_entry_trade(self, signal: str, current_price: float):
        """Execute entry trade"""
        try:
            print(f"\n📈 ENTRY SIGNAL: {signal.upper()} at ${current_price:.4f}")
            
            # Place the order
            trade = self.trade_executor.place_order(
                symbol=self.symbol,
                side=signal,
                quantity=self.quantity,
                price=current_price
            )
            
            if trade.status == "filled":
                # Log the trade
                self.trade_logger.log_trade(trade, "entry")
                
                # Update trading logic
                self.trading_logic.update_position(signal, current_price, trade.timestamp)
                
                # Store for potential closing
                self.current_entry_trade = trade
                
                print(f"✅ ENTRY EXECUTED: {signal.upper()} position opened")
                print(f"   📊 Price: ${current_price:.4f}")
                print(f"   📦 Quantity: {self.quantity}")
                print(f"   🆔 Trade ID: {trade.trade_id}")
                
            else:
                print(f"❌ ENTRY FAILED: {trade.status}")
                
        except Exception as e:
            logging.error(f"Error executing entry trade: {e}")
    
    def execute_exit_trade(self, current_price: float):
        """Execute exit trade"""
        try:
            if not self.current_entry_trade:
                logging.warning("No entry trade found for exit")
                return
            
            print(f"\n📉 EXIT SIGNAL at ${current_price:.4f}")
            
            # Place closing order
            exit_trade = self.trade_executor.close_position(
                symbol=self.symbol,
                original_side=self.current_entry_trade.side,
                quantity=self.quantity,
                current_price=current_price
            )
            
            if exit_trade.status == "filled":
                # Calculate P&L
                pnl = self.trade_logger.calculate_trade_pnl(self.current_entry_trade, exit_trade)
                exit_trade.profit_loss = pnl
                
                # Log the exit trade
                self.trade_logger.log_trade(exit_trade, "exit", self.current_entry_trade)
                
                # Update trading logic
                self.trading_logic.update_position('close', current_price, exit_trade.timestamp)
                
                # Clear current trade
                self.current_entry_trade = None
                
                # Display results
                pnl_symbol = "🟢" if pnl > 0 else "🔴"
                print(f"✅ EXIT EXECUTED: Position closed")
                print(f"   📊 Price: ${current_price:.4f}")
                print(f"   💰 P&L: ${pnl:.4f} {pnl_symbol}")
                print(f"   🆔 Trade ID: {exit_trade.trade_id}")
                
                # Print performance update
                self.print_quick_performance()
            else:
                print(f"❌ EXIT FAILED: {exit_trade.status}")
                
        except Exception as e:
            logging.error(f"Error executing exit trade: {e}")
    
    def log_status_update(self, current_price: float, price_change: float):
        """Log periodic status updates"""
        position = self.trading_logic.get_position_info()
        unrealized_pnl = self.trading_logic.calculate_unrealized_pnl(current_price)
        
        print(f"\n📊 STATUS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   💹 {self.symbol}: ${current_price:.4f} ({price_change:+.3f}%)")
        
        if position['position']:
            print(f"   📍 Position: {position['position'].upper()} @ ${position['entry_price']:.4f}")
            print(f"   💰 Unrealized P&L: ${unrealized_pnl:+.4f}")
        else:
            print(f"   📍 Position: NONE - Scanning for entries...")
        
        # Show account info occasionally
        if hasattr(self, 'trade_executor') and self.api_key:
            account_info = self.trade_executor.get_position_info()
            print(f"   💵 Balance: ${account_info['balance']:,.2f}")
    
    def print_quick_performance(self):
        """Print quick performance summary"""
        perf = self.trade_logger.get_performance_summary()
        if perf.get('total_trades', 0) > 0:
            print(f"\n📈 PERFORMANCE UPDATE:")
            print(f"   🔢 Total Trades: {perf['total_trades']}")
            print(f"   🎯 Win Rate: {perf['win_rate']:.1f}%")
            print(f"   💰 Total P&L: ${perf['total_pnl']:+.2f}")
            if perf['total_trades'] > 1:
                print(f"   📊 Avg P&L: ${perf['average_pnl']:+.4f}")
    
    def stop(self):
        """Stop the trading bot"""
        print("\n🛑 Stopping bot...")
        
        self.is_running = False
        
        # Stop data collection
        self.data_collector.stop_data_feed()
        
        # Print final performance report
        print("\n" + "="*60)
        print("           📊 FINAL PERFORMANCE REPORT")
        print("="*60)
        self.trade_logger.print_performance_report()
        
        # Show final account status
        if hasattr(self, 'trade_executor'):
            account_info = self.trade_executor.get_position_info()
            print(f"\n💰 Final Account Status:")
            print(f"   Balance: ${account_info['balance']:,.2f}")
            if self.api_key:
                print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
        
        print("\n✅ Bot stopped successfully")
        print(f"📁 Check {BOT_CONFIG['log_file']} for detailed trade log")


def main():
    """Main function"""
    
    # Display startup banner
    print("\n" + "="*60)
    print("           🤖 ALPACA SCALPING BOT v2.0")
    print("="*60)
    
    # Check configuration
    if not ALPACA_CONFIG['api_key'] or not ALPACA_CONFIG['secret_key']:
        print("⚠️  NO ALPACA API KEYS DETECTED")
        print("   📝 Edit config.py to add your Alpaca credentials")
        print("   🌐 Get free keys at: https://app.alpaca.markets/")
        print("   🔒 Running in LOCAL SIMULATION mode")
        print("")
    
    # Create and start the bot
    bot = ScalpingBot()
    
    try:
        bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received...")
        bot.stop()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
        bot.stop()


if __name__ == "__main__":
    main()