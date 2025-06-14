#!/usr/bin/env python3
"""
Simple Trade Logger for XAU/USD
"""

import csv
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional


class SimpleTradeRecord:
    """Simple trade record"""
    def __init__(self, trade, trade_type: str = "entry", profit_loss: float = 0.0):
        self.timestamp = trade.timestamp
        self.symbol = trade.symbol
        self.side = trade.side
        self.quantity = trade.quantity
        self.price = trade.price
        self.trade_id = trade.trade_id
        self.status = trade.status
        self.trade_type = trade_type  # 'entry' or 'exit'
        self.profit_loss = profit_loss


class SimpleTradeLogger:
    """Simple trade logger for XAU/USD"""
    
    def __init__(self, log_file: str = "xauusd_trades.csv"):
        self.log_file = log_file
        self.trades = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.session_start = datetime.now()
        
        # Setup CSV file
        self._setup_csv_file()
        
        logging.info(f"✅ Trade logger initialized - File: {log_file}")
    
    def _setup_csv_file(self):
        """Setup CSV file with headers"""
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price', 
                    'trade_id', 'status', 'trade_type', 'profit_loss'
                ])
            logging.info(f"Created new CSV file: {self.log_file}")
        else:
            # Load existing trades
            self._load_existing_trades()
    
    def _load_existing_trades(self):
        """Load existing trades from CSV"""
        
        try:
            with open(self.log_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    record = SimpleTradeRecord(
                        trade=type('Trade', (), {
                            'timestamp': row['timestamp'],
                            'symbol': row['symbol'],
                            'side': row['side'],
                            'quantity': float(row['quantity']),
                            'price': float(row['price']),
                            'trade_id': row['trade_id'],
                            'status': row['status']
                        })(),
                        trade_type=row.get('trade_type', 'entry'),
                        profit_loss=float(row.get('profit_loss', 0))
                    )
                    self.trades.append(record)
                    
                    # Update performance metrics
                    if record.trade_type == 'exit' and record.profit_loss != 0:
                        self.total_trades += 1
                        self.total_pnl += record.profit_loss
                        if record.profit_loss > 0:
                            self.winning_trades += 1
            
            logging.info(f"Loaded {len(self.trades)} existing trades")
            
        except Exception as e:
            logging.error(f"Error loading trades: {e}")
    
    def log_trade(self, trade, trade_type: str = "entry", profit_loss: float = 0.0):
        """Log a trade to CSV and memory"""
        
        # Create trade record
        record = SimpleTradeRecord(trade, trade_type, profit_loss)
        self.trades.append(record)
        
        # Write to CSV
        self._write_to_csv(record)
        
        # Update performance metrics for exit trades
        if trade_type == "exit" and profit_loss != 0:
            self.total_trades += 1
            self.total_pnl += profit_loss
            
            if profit_loss > 0:
                self.winning_trades += 1
        
        # Log the trade
        pnl_str = f" | P&L: ${profit_loss:+.2f}" if profit_loss != 0 else ""
        logging.info(f"TRADE [{trade_type.upper()}]: {trade.side.upper()} {trade.quantity} {trade.symbol} @ ${trade.price:.2f}{pnl_str}")
    
    def _write_to_csv(self, record: SimpleTradeRecord):
        """Write trade record to CSV file"""
        
        try:
            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    record.timestamp,
                    record.symbol,
                    record.side,
                    record.quantity,
                    record.price,
                    record.trade_id,
                    record.status,
                    record.trade_type,
                    record.profit_loss
                ])
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
    
    def calculate_trade_pnl(self, entry_trade, exit_trade) -> float:
        """Calculate P&L between entry and exit trades"""
        
        try:
            entry_price = entry_trade.price
            exit_price = exit_trade.price
            quantity = entry_trade.quantity
            side = entry_trade.side
            
            if side == 'buy':
                pnl = (exit_price - entry_price) * quantity
            else:  # sell
                pnl = (entry_price - exit_price) * quantity
            
            return round(pnl, 2)
            
        except Exception as e:
            logging.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        session_duration = datetime.now() - self.session_start
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_pnl': avg_pnl,
            'session_duration': str(session_duration).split('.')[0],  # Remove microseconds
            'trades_logged': len(self.trades)
        }
    
    def get_recent_trades(self, count: int = 10) -> List[SimpleTradeRecord]:
        """Get recent trades"""
        return self.trades[-count:] if self.trades else []
    
    def get_trades_by_type(self, trade_type: str) -> List[SimpleTradeRecord]:
        """Get trades by type (entry/exit)"""
        return [trade for trade in self.trades if trade.trade_type == trade_type]
    
    def get_daily_summary(self) -> Dict:
        """Get today's trading summary"""
        
        today = datetime.now().date()
        today_trades = []
        
        for trade in self.trades:
            trade_date = datetime.fromisoformat(trade.timestamp).date()
            if trade_date == today:
                today_trades.append(trade)
        
        # Calculate today's stats
        today_exits = [t for t in today_trades if t.trade_type == 'exit']
        today_pnl = sum(t.profit_loss for t in today_exits)
        today_wins = sum(1 for t in today_exits if t.profit_loss > 0)
        today_total = len(today_exits)
        
        return {
            'date': today.isoformat(),
            'total_trades': today_total,
            'winning_trades': today_wins,
            'total_pnl': today_pnl,
            'win_rate': (today_wins / today_total * 100) if today_total > 0 else 0,
            'trades_logged': len(today_trades)
        }
    
    def print_performance_summary(self):
        """Print performance summary to console"""
        
        summary = self.get_performance_summary()
        
        print("\n" + "="*50)
        print("           TRADING PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Session Duration: {summary['session_duration']}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Winning Trades: {summary['winning_trades']}")
        print(f"Losing Trades: {summary['losing_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Total P&L: ${summary['total_pnl']:+.2f}")
        print(f"Average P&L: ${summary['average_pnl']:+.2f}")
        print(f"Trades Logged: {summary['trades_logged']}")
        print("="*50)
    
    def print_daily_summary(self):
        """Print today's summary"""
        
        daily = self.get_daily_summary()
        
        print(f"\n📊 TODAY'S SUMMARY ({daily['date']}):")
        print(f"   Trades: {daily['total_trades']}")
        print(f"   Wins: {daily['winning_trades']}")
        print(f"   Win Rate: {daily['win_rate']:.1f}%")
        print(f"   P&L: ${daily['total_pnl']:+.2f}")
    
    def export_trades(self, filename: Optional[str] = None) -> str:
        """Export trades to a new CSV file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_export_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'trade_id', 'status', 'trade_type', 'profit_loss'
                ])
                
                for record in self.trades:
                    writer.writerow([
                        record.timestamp,
                        record.symbol,
                        record.side,
                        record.quantity,
                        record.price,
                        record.trade_id,
                        record.status,
                        record.trade_type,
                        record.profit_loss
                    ])
            
            logging.info(f"✅ Trades exported to: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting trades: {e}")
            return ""
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get trade history as list of dictionaries"""
        
        trades = self.trades
        
        # Filter by symbol if specified
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Get recent trades
        recent_trades = trades[-limit:] if trades else []
        
        # Convert to dictionaries
        history = []
        for trade in recent_trades:
            history.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'trade_id': trade.trade_id,
                'status': trade.status,
                'trade_type': trade.trade_type,
                'profit_loss': trade.profit_loss
            })
        
        return history
    
    def calculate_statistics(self) -> Dict:
        """Calculate detailed trading statistics"""
        
        if self.total_trades == 0:
            return {'error': 'No completed trades'}
        
        # Get all exit trades (completed trades)
        exit_trades = [t for t in self.trades if t.trade_type == 'exit' and t.profit_loss != 0]
        
        if not exit_trades:
            return {'error': 'No completed trades with P&L'}
        
        profits = [t.profit_loss for t in exit_trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in exit_trades if t.profit_loss < 0]
        
        stats = {
            'total_trades': len(exit_trades),
            'winning_trades': len(profits),
            'losing_trades': len(losses),
            'win_rate': (len(profits) / len(exit_trades)) * 100,
            'total_pnl': sum(t.profit_loss for t in exit_trades),
            'average_win': sum(profits) / len(profits) if profits else 0,
            'average_loss': sum(losses) / len(losses) if losses else 0,
            'largest_win': max(profits) if profits else 0,
            'largest_loss': min(losses) if losses else 0,
            'profit_factor': abs(sum(profits) / sum(losses)) if losses else float('inf')
        }
        
        return stats
    
    def cleanup(self):
        """Clean up logger resources"""
        
        try:
            # Print final summaries
            self.print_performance_summary()
            self.print_daily_summary()
            
            # Print detailed statistics
            stats = self.calculate_statistics()
            if 'error' not in stats:
                print(f"\n📈 DETAILED STATISTICS:")
                print(f"   Average Win: ${stats['average_win']:+.2f}")
                print(f"   Average Loss: ${stats['average_loss']:+.2f}")
                print(f"   Largest Win: ${stats['largest_win']:+.2f}")
                print(f"   Largest Loss: ${stats['largest_loss']:+.2f}")
                print(f"   Profit Factor: {stats['profit_factor']:.2f}")
            
            logging.info("✅ Trade logger cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Test trade logger
    logger = SimpleTradeLogger()
    
    # Mock trade object for testing
    class MockTrade:
        def __init__(self, side: str, price: float):
            self.timestamp = datetime.now().isoformat()
            self.symbol = 'XAUUSD'
            self.side = side
            self.quantity = 0.1
            self.price = price
            self.trade_id = f'test_{side}_{int(datetime.now().timestamp())}'
            self.status = 'filled'
    
    # Test multiple trades
    print("🧪 Testing trade logger...")
    
    # Trade 1: Profitable
    entry1 = MockTrade('buy', 2000.50)
    logger.log_trade(entry1, trade_type="entry")
    
    exit1 = MockTrade('sell', 2002.00)
    pnl1 = logger.calculate_trade_pnl(entry1, exit1)
    logger.log_trade(exit1, trade_type="exit", profit_loss=pnl1)
    
    # Trade 2: Loss
    entry2 = MockTrade('buy', 2001.00)
    logger.log_trade(entry2, trade_type="entry")
    
    exit2 = MockTrade('sell', 2000.50)
    pnl2 = logger.calculate_trade_pnl(entry2, exit2)
    logger.log_trade(exit2, trade_type="exit", profit_loss=pnl2)
    
    # Print summaries
    logger.print_performance_summary()
    logger.print_daily_summary()
    
    # Test export
    export_file = logger.export_trades()
    print(f"Exported to: {export_file}")
    
    # Test statistics
    stats = logger.calculate_statistics()
    print(f"Statistics: {stats}")
    
    # Cleanup
    logger.cleanup()