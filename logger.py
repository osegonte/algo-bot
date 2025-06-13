#!/usr/bin/env python3
"""Logger Module - Simplified"""

import csv
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
from trade_execution import Trade

class TradeLogger:
    def __init__(self, log_file: str = "trades.csv", log_level: int = logging.INFO):
        self.log_file = log_file
        self.trades = []
        self.open_positions = {}
        
        self.setup_logging(log_level)
        self.setup_csv_file()
        
        logging.info(f"TradeLogger initialized - Log file: {log_file}")
    
    def setup_logging(self, log_level: int):
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_csv_file(self):
        try:
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                logging.info(f"Using existing log file: {self.log_file}")
                self.load_existing_trades()
            else:
                with open(self.log_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        'timestamp', 'symbol', 'side', 'quantity', 
                        'price', 'trade_id', 'status', 'profit_loss',
                        'trade_type', 'cumulative_pnl'
                    ])
                logging.info(f"Created new log file: {self.log_file}")
        except Exception as e:
            logging.error(f"Error setting up CSV file: {e}")
    
    def load_existing_trades(self):
        try:
            with open(self.log_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    trade = Trade(
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        side=row['side'],
                        quantity=float(row['quantity']),
                        price=float(row['price']),
                        trade_id=row['trade_id'],
                        status=row['status'],
                        profit_loss=float(row['profit_loss'])
                    )
                    self.trades.append(trade)
            
            logging.info(f"Loaded {len(self.trades)} existing trades from log file")
        except Exception as e:
            logging.error(f"Error loading existing trades: {e}")
    
    def log_trade(self, trade, trade_type: str = "entry", exit_trade: Optional = None):
        try:
            if trade_type == "exit" and exit_trade:
                trade.profit_loss = self.calculate_trade_pnl(exit_trade, trade)
            
            self.trades.append(trade)
            cumulative_pnl = sum(t.profit_loss for t in self.trades)
            
            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    trade.timestamp, trade.symbol, trade.side, trade.quantity,
                    trade.price, trade.trade_id, trade.status, trade.profit_loss,
                    trade_type, cumulative_pnl
                ])
            
            self.update_position_tracking(trade, trade_type)
            
            pnl_str = f" | P&L: ${trade.profit_loss:.2f}" if trade.profit_loss != 0 else ""
            logging.info(f"TRADE LOGGED [{trade_type.upper()}]: {trade.side.upper()} {trade.quantity} "
                        f"{trade.symbol} @ ${trade.price:.4f} | Status: {trade.status}{pnl_str}")
        except Exception as e:
            logging.error(f"Error logging trade: {e}")
    
    def update_position_tracking(self, trade, trade_type: str):
        if trade_type == "entry":
            self.open_positions[trade.trade_id] = trade
        elif trade_type == "exit":
            entry_id = trade.trade_id.replace("_close", "")
            if entry_id in self.open_positions:
                del self.open_positions[entry_id]
    
    def calculate_trade_pnl(self, entry_trade, exit_trade) -> float:
        try:
            if entry_trade.side == 'buy':
                pnl = (exit_trade.price - entry_trade.price) * entry_trade.quantity
            else:
                pnl = (entry_trade.price - exit_trade.price) * entry_trade.quantity
            return round(pnl, 4)
        except Exception as e:
            logging.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def get_performance_summary(self) -> Dict:
        if not self.trades:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "average_pnl": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0
            }
        
        completed_trades = [t for t in self.trades if t.profit_loss != 0]
        if not completed_trades:
            return {"message": "No completed trades yet"}
        
        total_pnl = sum(t.profit_loss for t in completed_trades)
        winning_trades = [t for t in completed_trades if t.profit_loss > 0]
        losing_trades = [t for t in completed_trades if t.profit_loss < 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        average_pnl = total_pnl / len(completed_trades) if completed_trades else 0
        
        largest_win = max([t.profit_loss for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.profit_loss for t in losing_trades]) if losing_trades else 0
        
        return {
            "total_trades": len(completed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate * 100, 2),
            "total_pnl": round(total_pnl, 2),
            "average_pnl": round(average_pnl, 4),
            "largest_win": round(largest_win, 4),
            "largest_loss": round(largest_loss, 4),
            "open_positions": len(self.open_positions)
        }
    
    def print_performance_report(self):
        stats = self.get_performance_summary()
        
        if "message" in stats:
            print(f"\n=== PERFORMANCE REPORT ===\n{stats['message']}")
            return
        
        print("\n" + "="*50)
        print("           PERFORMANCE REPORT")
        print("="*50)
        print(f"Total Completed Trades: {stats['total_trades']}")
        print(f"Winning Trades: {stats['winning_trades']}")
        print(f"Losing Trades: {stats['losing_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Total P&L: ${stats['total_pnl']:.2f}")
        print(f"Average P&L per Trade: ${stats['average_pnl']:.4f}")
        print(f"Largest Win: ${stats['largest_win']:.4f}")
        print(f"Largest Loss: ${stats['largest_loss']:.4f}")
        print(f"Open Positions: {stats['open_positions']}")
        print("="*50)
