#!/usr/bin/env python3
"""
Enhanced Logger Module for XAU/USD Tick Trading
Comprehensive logging with performance analytics and ML data collection
"""

import csv
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import deque
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class TradeRecord:
    """Enhanced trade record structure"""
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    trade_id: str
    status: str
    profit_loss: float
    trade_type: str  # 'entry' or 'exit'
    
    # Enhanced fields
    execution_time_ms: float = 0.0
    slippage_ticks: float = 0.0
    commission: float = 0.0
    signal_confidence: float = 0.0
    signal_reasoning: str = ""
    market_conditions: Dict = None
    
    # Performance metrics
    cumulative_pnl: float = 0.0
    running_win_rate: float = 0.0
    session_trade_number: int = 0

class EnhancedTradeLogger:
    """Enhanced trade logger with comprehensive analytics and ML integration"""
    
    def __init__(self, log_file: str = "enhanced_trades.csv", 
                 log_level: int = logging.INFO,
                 use_database: bool = False,
                 database_file: str = "trades.db"):
        
        self.log_file = log_file
        self.database_file = database_file
        self.use_database = use_database
        
        # Trade storage
        self.trades = deque(maxlen=10000)  # Keep last 10k trades in memory
        self.open_positions = {}
        self.session_stats = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': 0.0,
            'session_start': datetime.now()
        }
        
        # Real-time analytics
        self.trade_analytics = {
            'hourly_performance': {},
            'signal_performance': {},
            'market_condition_performance': {},
            'execution_quality': deque(maxlen=1000)
        }
        
        # ML data collection
        self.ml_features = deque(maxlen=5000)
        self.feature_columns = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize logging and storage
        self.setup_logging(log_level)
        self.setup_storage()
        
        logging.info(f"✅ Enhanced trade logger initialized - File: {log_file}")
    
    def setup_logging(self, log_level: int):
        """Setup enhanced logging configuration"""
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging with multiple handlers
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_level)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Trade-specific handler
        trade_handler = logging.FileHandler(
            f'logs/trades_{datetime.now().strftime("%Y%m%d")}.log'
        )
        trade_handler.setFormatter(log_formatter)
        trade_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Create trade logger
        self.trade_logger = logging.getLogger('trades')
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)
    
    def setup_storage(self):
        """Setup CSV and database storage"""
        
        # Setup CSV file
        self.setup_csv_file()
        
        # Setup database if enabled
        if self.use_database:
            self.setup_database()
    
    def setup_csv_file(self):
        """Setup CSV file with enhanced headers"""
        
        try:
            # Check if file exists and has content
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                logging.info(f"Using existing CSV file: {self.log_file}")
                self.load_existing_trades()
            else:
                # Create new CSV with enhanced headers
                with open(self.log_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        'timestamp', 'symbol', 'side', 'quantity', 'price', 
                        'trade_id', 'status', 'profit_loss', 'trade_type',
                        'execution_time_ms', 'slippage_ticks', 'commission',
                        'signal_confidence', 'signal_reasoning',
                        'market_conditions', 'cumulative_pnl', 'running_win_rate',
                        'session_trade_number'
                    ])
                logging.info(f"Created new CSV file: {self.log_file}")
                
        except Exception as e:
            logging.error(f"Error setting up CSV file: {e}")
    
    def setup_database(self):
        """Setup SQLite database for advanced querying"""
        
        try:
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    trade_id TEXT UNIQUE,
                    status TEXT,
                    profit_loss REAL,
                    trade_type TEXT,
                    execution_time_ms REAL,
                    slippage_ticks REAL,
                    commission REAL,
                    signal_confidence REAL,
                    signal_reasoning TEXT,
                    market_conditions TEXT,
                    cumulative_pnl REAL,
                    running_win_rate REAL,
                    session_trade_number INTEGER
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    additional_data TEXT
                )
            ''')
            
            # Create ML features table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    trade_id TEXT,
                    features TEXT,
                    outcome REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logging.info(f"✅ Database initialized: {self.database_file}")
            
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            self.use_database = False
    
    def load_existing_trades(self):
        """Load existing trades from CSV file"""
        
        try:
            with open(self.log_file, 'r') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Parse market conditions JSON
                    market_conditions = {}
                    if row.get('market_conditions'):
                        try:
                            market_conditions = json.loads(row['market_conditions'])
                        except:
                            pass
                    
                    trade_record = TradeRecord(
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        side=row['side'],
                        quantity=float(row['quantity']),
                        price=float(row['price']),
                        trade_id=row['trade_id'],
                        status=row['status'],
                        profit_loss=float(row.get('profit_loss', 0)),
                        trade_type=row.get('trade_type', 'unknown'),
                        execution_time_ms=float(row.get('execution_time_ms', 0)),
                        slippage_ticks=float(row.get('slippage_ticks', 0)),
                        commission=float(row.get('commission', 0)),
                        signal_confidence=float(row.get('signal_confidence', 0)),
                        signal_reasoning=row.get('signal_reasoning', ''),
                        market_conditions=market_conditions,
                        cumulative_pnl=float(row.get('cumulative_pnl', 0)),
                        running_win_rate=float(row.get('running_win_rate', 0)),
                        session_trade_number=int(row.get('session_trade_number', 0))
                    )
                    
                    self.trades.append(trade_record)
                
                # Update performance metrics
                self._recalculate_performance_metrics()
                
                logging.info(f"Loaded {len(self.trades)} existing trades")
                
        except Exception as e:
            logging.error(f"Error loading existing trades: {e}")
    
    def log_trade(self, trade, trade_type: str = "entry", 
                  signal_confidence: float = 0.0, signal_reasoning: str = "",
                  market_conditions: Dict = None, exit_trade = None):
        """Log trade with enhanced information"""
        
        with self.lock:
            try:
                # Calculate P&L for exit trades
                profit_loss = 0.0
                if trade_type == "exit" and exit_trade:
                    profit_loss = self.calculate_trade_pnl(exit_trade, trade)
                elif hasattr(trade, 'profit_loss'):
                    profit_loss = trade.profit_loss
                
                # Update cumulative metrics
                self.performance_metrics['total_trades'] += 1
                self.performance_metrics['total_pnl'] += profit_loss
                
                if profit_loss > 0:
                    self.performance_metrics['winning_trades'] += 1
                elif profit_loss < 0:
                    self.performance_metrics['losing_trades'] += 1
                
                # Calculate running metrics
                win_rate = self._calculate_current_win_rate()
                cumulative_pnl = self.performance_metrics['total_pnl']
                
                # Update drawdown tracking
                self._update_drawdown_tracking(cumulative_pnl)
                
                # Create enhanced trade record
                trade_record = TradeRecord(
                    timestamp=getattr(trade, 'timestamp', datetime.now().isoformat()),
                    symbol=getattr(trade, 'symbol', ''),
                    side=getattr(trade, 'side', ''),
                    quantity=getattr(trade, 'quantity', 0),
                    price=getattr(trade, 'price', 0),
                    trade_id=getattr(trade, 'trade_id', ''),
                    status=getattr(trade, 'status', ''),
                    profit_loss=profit_loss,
                    trade_type=trade_type,
                    execution_time_ms=getattr(trade, 'execution_time_ms', 0),
                    slippage_ticks=getattr(trade, 'slippage_ticks', 0),
                    commission=getattr(trade, 'commission', 0),
                    signal_confidence=signal_confidence,
                    signal_reasoning=signal_reasoning,
                    market_conditions=market_conditions or {},
                    cumulative_pnl=cumulative_pnl,
                    running_win_rate=win_rate,
                    session_trade_number=self.performance_metrics['total_trades']
                )
                
                # Store trade
                self.trades.append(trade_record)
                
                # Write to CSV
                self._write_to_csv(trade_record)
                
                # Write to database if enabled
                if self.use_database:
                    self._write_to_database(trade_record)
                
                # Update analytics
                self._update_trade_analytics(trade_record)
                
                # Update position tracking
                self._update_position_tracking(trade_record, trade_type)
                
                # Log the trade
                pnl_str = f" | P&L: ${profit_loss:.4f}" if profit_loss != 0 else ""
                confidence_str = f" | Confidence: {signal_confidence:.2f}" if signal_confidence > 0 else ""
                
                self.trade_logger.info(
                    f"TRADE [{trade_type.upper()}]: {trade.side.upper()} {trade.quantity} "
                    f"{trade.symbol} @ ${trade.price:.4f}{pnl_str}{confidence_str}"
                )
                
                logging.info(
                    f"Trade logged: {trade_type.upper()} | "
                    f"Total: {self.performance_metrics['total_trades']} | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"Cumulative P&L: ${cumulative_pnl:.2f}"
                )
                
            except Exception as e:
                logging.error(f"Error logging trade: {e}")
    
    def _write_to_csv(self, trade_record: TradeRecord):
        """Write trade record to CSV file"""
        
        try:
            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                
                # Convert market conditions to JSON string
                market_conditions_str = json.dumps(trade_record.market_conditions) if trade_record.market_conditions else ""
                
                writer.writerow([
                    trade_record.timestamp,
                    trade_record.symbol,
                    trade_record.side,
                    trade_record.quantity,
                    trade_record.price,
                    trade_record.trade_id,
                    trade_record.status,
                    trade_record.profit_loss,
                    trade_record.trade_type,
                    trade_record.execution_time_ms,
                    trade_record.slippage_ticks,
                    trade_record.commission,
                    trade_record.signal_confidence,
                    trade_record.signal_reasoning,
                    market_conditions_str,
                    trade_record.cumulative_pnl,
                    trade_record.running_win_rate,
                    trade_record.session_trade_number
                ])
                
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
    
    def _write_to_database(self, trade_record: TradeRecord):
        """Write trade record to database"""
        
        try:
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()
            
            market_conditions_str = json.dumps(trade_record.market_conditions) if trade_record.market_conditions else ""
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades (
                    timestamp, symbol, side, quantity, price, trade_id, status,
                    profit_loss, trade_type, execution_time_ms, slippage_ticks,
                    commission, signal_confidence, signal_reasoning,
                    market_conditions, cumulative_pnl, running_win_rate,
                    session_trade_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_record.timestamp, trade_record.symbol, trade_record.side,
                trade_record.quantity, trade_record.price, trade_record.trade_id,
                trade_record.status, trade_record.profit_loss, trade_record.trade_type,
                trade_record.execution_time_ms, trade_record.slippage_ticks,
                trade_record.commission, trade_record.signal_confidence,
                trade_record.signal_reasoning, market_conditions_str,
                trade_record.cumulative_pnl, trade_record.running_win_rate,
                trade_record.session_trade_number
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error writing to database: {e}")
    
    def _update_trade_analytics(self, trade_record: TradeRecord):
        """Update real-time trade analytics"""
        
        try:
            # Hourly performance tracking
            hour = datetime.fromisoformat(trade_record.timestamp).hour
            if hour not in self.trade_analytics['hourly_performance']:
                self.trade_analytics['hourly_performance'][hour] = {
                    'trades': 0, 'pnl': 0.0, 'wins': 0
                }
            
            hourly = self.trade_analytics['hourly_performance'][hour]
            hourly['trades'] += 1
            hourly['pnl'] += trade_record.profit_loss
            if trade_record.profit_loss > 0:
                hourly['wins'] += 1
            
            # Signal performance tracking
            if trade_record.signal_reasoning:
                signal_key = trade_record.signal_reasoning[:50]  # Truncate for grouping
                if signal_key not in self.trade_analytics['signal_performance']:
                    self.trade_analytics['signal_performance'][signal_key] = {
                        'trades': 0, 'pnl': 0.0, 'avg_confidence': 0.0
                    }
                
                signal_perf = self.trade_analytics['signal_performance'][signal_key]
                signal_perf['trades'] += 1
                signal_perf['pnl'] += trade_record.profit_loss
                signal_perf['avg_confidence'] = (
                    (signal_perf['avg_confidence'] * (signal_perf['trades'] - 1) + 
                     trade_record.signal_confidence) / signal_perf['trades']
                )
            
            # Execution quality tracking
            if trade_record.execution_time_ms > 0:
                self.trade_analytics['execution_quality'].append({
                    'timestamp': trade_record.timestamp,
                    'execution_time': trade_record.execution_time_ms,
                    'slippage': trade_record.slippage_ticks,
                    'symbol': trade_record.symbol
                })
            
        except Exception as e:
            logging.error(f"Error updating analytics: {e}")
    
    def _update_position_tracking(self, trade_record: TradeRecord, trade_type: str):
        """Update position tracking"""
        
        if trade_type == "entry":
            self.open_positions[trade_record.trade_id] = trade_record
        elif trade_type == "exit":
            # Find and remove corresponding entry
            entry_id = trade_record.trade_id.replace("_close", "")
            if entry_id in self.open_positions:
                del self.open_positions[entry_id]
    
    def _calculate_current_win_rate(self) -> float:
        """Calculate current win rate"""
        
        if self.performance_metrics['total_trades'] == 0:
            return 0.0
        
        return (self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades'] * 100)
    
    def _update_drawdown_tracking(self, current_pnl: float):
        """Update maximum drawdown tracking"""
        
        if current_pnl > self.performance_metrics['peak_equity']:
            self.performance_metrics['peak_equity'] = current_pnl
        
        current_drawdown = self.performance_metrics['peak_equity'] - current_pnl
        if current_drawdown > self.performance_metrics['max_drawdown']:
            self.performance_metrics['max_drawdown'] = current_drawdown
    
    def _recalculate_performance_metrics(self):
        """Recalculate performance metrics from loaded trades"""
        
        total_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        peak_equity = 0.0
        max_drawdown = 0.0
        
        for trade in self.trades:
            if trade.profit_loss != 0:  # Only count completed trades
                total_pnl += trade.profit_loss
                
                if trade.profit_loss > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                # Track drawdown
                if total_pnl > peak_equity:
                    peak_equity = total_pnl
                
                current_drawdown = peak_equity - total_pnl
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
        
        self.performance_metrics.update({
            'total_trades': len([t for t in self.trades if t.profit_loss != 0]),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'peak_equity': peak_equity,
            'max_drawdown': max_drawdown
        })
    
    def calculate_trade_pnl(self, entry_trade, exit_trade) -> float:
        """Calculate P&L between entry and exit trades"""
        
        try:
            entry_price = getattr(entry_trade, 'price', 0)
            exit_price = getattr(exit_trade, 'price', 0)
            quantity = getattr(entry_trade, 'quantity', 0)
            side = getattr(entry_trade, 'side', '')
            
            if side == 'buy':
                pnl = (exit_price - entry_price) * quantity
            else:  # sell
                pnl = (entry_price - exit_price) * quantity
                
            return round(pnl, 4)
            
        except Exception as e:
            logging.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def log_performance_metrics(self, additional_metrics: Dict = None):
        """Log current performance metrics"""
        
        try:
            # Calculate additional metrics
            total_trades = self.performance_metrics['total_trades']
            win_rate = self._calculate_current_win_rate()
            
            if total_trades > 0:
                avg_pnl = self.performance_metrics['total_pnl'] / total_trades
                profit_factor = self._calculate_profit_factor()
                sharpe_ratio = self._calculate_sharpe_ratio()
            else:
                avg_pnl = 0.0
                profit_factor = 0.0
                sharpe_ratio = 0.0
            
            # Combine all metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'total_trades': total_trades,
                'winning_trades': self.performance_metrics['winning_trades'],
                'losing_trades': self.performance_metrics['losing_trades'],
                'win_rate': win_rate,
                'total_pnl': self.performance_metrics['total_pnl'],
                'average_pnl': avg_pnl,
                'max_drawdown': self.performance_metrics['max_drawdown'],
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'open_positions': len(self.open_positions)
            }
            
            # Add additional metrics if provided
            if additional_metrics:
                metrics.update(additional_metrics)
            
            # Log to database if enabled
            if self.use_database:
                self._log_metrics_to_database(metrics)
            
            # Log summary
            logging.info(f"📊 Performance Metrics: {json.dumps(metrics, indent=2)}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error logging performance metrics: {e}")
            return {}
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t.profit_loss for t in self.trades if t.profit_loss > 0)
        gross_loss = abs(sum(t.profit_loss for t in self.trades if t.profit_loss < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate approximate Sharpe ratio"""
        
        if len(self.trades) < 10:
            return 0.0
        
        returns = [t.profit_loss for t in self.trades if t.profit_loss != 0]
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _log_metrics_to_database(self, metrics: Dict):
        """Log performance metrics to database"""
        
        try:
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    cursor.execute('''
                        INSERT INTO performance_metrics 
                        (timestamp, metric_name, metric_value, additional_data)
                        VALUES (?, ?, ?, ?)
                    ''', (metrics['timestamp'], metric_name, metric_value, ''))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error logging metrics to database: {e}")
    
    def collect_ml_features(self, trade_id: str, features: Dict, outcome: float):
        """Collect features and outcomes for ML training"""
        
        try:
            feature_record = {
                'timestamp': datetime.now().isoformat(),
                'trade_id': trade_id,
                'features': features,
                'outcome': outcome
            }
            
            self.ml_features.append(feature_record)
            
            # Update feature columns
            if features and not self.feature_columns:
                self.feature_columns = list(features.keys())
            
            # Log to database if enabled
            if self.use_database:
                conn = sqlite3.connect(self.database_file)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO ml_features (timestamp, trade_id, features, outcome)
                    VALUES (?, ?, ?, ?)
                ''', (
                    feature_record['timestamp'],
                    trade_id,
                    json.dumps(features),
                    outcome
                ))
                
                conn.commit()
                conn.close()
            
            logging.debug(f"ML features collected: {trade_id}")
            
        except Exception as e:
            logging.error(f"Error collecting ML features: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        with self.lock:
            summary = {
                'session_duration': str(datetime.now() - self.performance_metrics['session_start']),
                'total_trades': self.performance_metrics['total_trades'],
                'winning_trades': self.performance_metrics['winning_trades'],
                'losing_trades': self.performance_metrics['losing_trades'],
                'win_rate': self._calculate_current_win_rate(),
                'total_pnl': self.performance_metrics['total_pnl'],
                'max_drawdown': self.performance_metrics['max_drawdown'],
                'open_positions': len(self.open_positions),
                'profit_factor': self._calculate_profit_factor(),
                'sharpe_ratio': self._calculate_sharpe_ratio()
            }
            
            # Add recent performance (last 20 trades)
            recent_trades = [t for t in list(self.trades)[-20:] if t.profit_loss != 0]
            if recent_trades:
                recent_wins = sum(1 for t in recent_trades if t.profit_loss > 0)
                summary['recent_win_rate'] = (recent_wins / len(recent_trades)) * 100
                summary['recent_pnl'] = sum(t.profit_loss for t in recent_trades)
            else:
                summary['recent_win_rate'] = 0.0
                summary['recent_pnl'] = 0.0
            
            # Add execution quality metrics
            if self.trade_analytics['execution_quality']:
                exec_data = list(self.trade_analytics['execution_quality'])
                summary['avg_execution_time_ms'] = np.mean([e['execution_time'] for e in exec_data])
                summary['avg_slippage_ticks'] = np.mean([e['slippage'] for e in exec_data])
            
            return summary
    
    def get_trade_analytics(self) -> Dict:
        """Get detailed trade analytics"""
        
        return {
            'hourly_performance': dict(self.trade_analytics['hourly_performance']),
            'signal_performance': dict(self.trade_analytics['signal_performance']),
            'execution_quality': {
                'sample_size': len(self.trade_analytics['execution_quality']),
                'avg_execution_time': np.mean([e['execution_time'] 
                                             for e in self.trade_analytics['execution_quality']]) 
                                    if self.trade_analytics['execution_quality'] else 0,
                'avg_slippage': np.mean([e['slippage'] 
                                       for e in self.trade_analytics['execution_quality']])
                              if self.trade_analytics['execution_quality'] else 0
            }
        }
    
    def get_ml_training_data(self) -> Dict:
        """Get collected ML training data"""
        
        if not self.ml_features:
            return {'features': [], 'outcomes': [], 'feature_columns': []}
        
        features_list = []
        outcomes_list = []
        
        for record in self.ml_features:
            if record['features']:
                features_list.append(record['features'])
                outcomes_list.append(record['outcome'])
        
        return {
            'features': features_list,
            'outcomes': outcomes_list,
            'feature_columns': self.feature_columns,
            'sample_count': len(features_list)
        }
    
    def export_data(self, export_format: str = 'csv', 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> str:
        """Export trade data in various formats"""
        
        try:
            # Filter trades by date if specified
            filtered_trades = self.trades
            
            if start_date or end_date:
                filtered_trades = []
                start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
                end_dt = datetime.fromisoformat(end_date) if end_date else datetime.max
                
                for trade in self.trades:
                    trade_dt = datetime.fromisoformat(trade.timestamp)
                    if start_dt <= trade_dt <= end_dt:
                        filtered_trades.append(trade)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format.lower() == 'csv':
                filename = f"trades_export_{timestamp}.csv"
                self._export_to_csv(filtered_trades, filename)
            
            elif export_format.lower() == 'json':
                filename = f"trades_export_{timestamp}.json"
                self._export_to_json(filtered_trades, filename)
            
            elif export_format.lower() == 'excel':
                filename = f"trades_export_{timestamp}.xlsx"
                self._export_to_excel(filtered_trades, filename)
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logging.info(f"✅ Data exported: {filename} ({len(filtered_trades)} records)")
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting data: {e}")
            return ""
    
    def _export_to_csv(self, trades: List[TradeRecord], filename: str):
        """Export trades to CSV format"""
        
        with open(filename, 'w', newline='') as file:
            if not trades:
                return
            
            # Use the first trade to get field names
            fieldnames = list(asdict(trades[0]).keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            for trade in trades:
                row = asdict(trade)
                # Convert dict fields to JSON strings
                if isinstance(row['market_conditions'], dict):
                    row['market_conditions'] = json.dumps(row['market_conditions'])
                writer.writerow(row)
    
    def _export_to_json(self, trades: List[TradeRecord], filename: str):
        """Export trades to JSON format"""
        
        trade_dicts = [asdict(trade) for trade in trades]
        
        with open(filename, 'w') as file:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'trade_count': len(trades),
                'trades': trade_dicts
            }, file, indent=2)
    
    def _export_to_excel(self, trades: List[TradeRecord], filename: str):
        """Export trades to Excel format (requires openpyxl)"""
        
        try:
            import pandas as pd
            
            # Convert to DataFrame
            trade_dicts = []
            for trade in trades:
                trade_dict = asdict(trade)
                # Flatten market_conditions
                if isinstance(trade_dict['market_conditions'], dict):
                    for key, value in trade_dict['market_conditions'].items():
                        trade_dict[f'market_{key}'] = value
                    del trade_dict['market_conditions']
                trade_dicts.append(trade_dict)
            
            df = pd.DataFrame(trade_dicts)
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Add summary sheet
                summary = self.get_performance_summary()
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add analytics sheet if available
                analytics = self.get_trade_analytics()
                if analytics['hourly_performance']:
                    hourly_df = pd.DataFrame.from_dict(
                        analytics['hourly_performance'], orient='index'
                    )
                    hourly_df.to_excel(writer, sheet_name='Hourly_Performance')
                
        except ImportError:
            logging.warning("pandas not available for Excel export, falling back to CSV")
            csv_filename = filename.replace('.xlsx', '.csv')
            self._export_to_csv(trades, csv_filename)
        except Exception as e:
            logging.error(f"Error exporting to Excel: {e}")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        try:
            summary = self.get_performance_summary()
            analytics = self.get_trade_analytics()
            
            report_lines = [
                "="*80,
                "           COMPREHENSIVE TRADING PERFORMANCE REPORT",
                "="*80,
                f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Session Duration: {summary['session_duration']}",
                "",
                "📊 OVERALL PERFORMANCE:",
                f"   Total Trades: {summary['total_trades']}",
                f"   Winning Trades: {summary['winning_trades']}",
                f"   Losing Trades: {summary['losing_trades']}",
                f"   Win Rate: {summary['win_rate']:.1f}%",
                f"   Total P&L: ${summary['total_pnl']:+,.2f}",
                f"   Maximum Drawdown: ${summary['max_drawdown']:,.2f}",
                f"   Profit Factor: {summary['profit_factor']:.2f}",
                f"   Sharpe Ratio: {summary['sharpe_ratio']:.3f}",
                "",
                "📈 RECENT PERFORMANCE (Last 20 Trades):",
                f"   Recent Win Rate: {summary['recent_win_rate']:.1f}%",
                f"   Recent P&L: ${summary['recent_pnl']:+,.2f}",
                "",
                "⚡ EXECUTION QUALITY:",
                f"   Average Execution Time: {summary.get('avg_execution_time_ms', 0):.1f} ms",
                f"   Average Slippage: {summary.get('avg_slippage_ticks', 0):.2f} ticks",
                "",
                "🕐 HOURLY PERFORMANCE:"
            ]
            
            # Add hourly breakdown
            for hour, data in sorted(analytics['hourly_performance'].items()):
                win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
                report_lines.append(
                    f"   {hour:02d}:00 - Trades: {data['trades']}, "
                    f"P&L: ${data['pnl']:+.2f}, Win Rate: {win_rate:.1f}%"
                )
            
            # Add signal performance
            if analytics['signal_performance']:
                report_lines.extend([
                    "",
                    "🎯 TOP SIGNAL PERFORMANCE:"
                ])
                
                # Sort by P&L
                sorted_signals = sorted(
                    analytics['signal_performance'].items(),
                    key=lambda x: x[1]['pnl'],
                    reverse=True
                )[:5]
                
                for signal, data in sorted_signals:
                    win_rate = 0  # Would need to calculate from individual trades
                    report_lines.append(
                        f"   {signal[:40]}... - Trades: {data['trades']}, "
                        f"P&L: ${data['pnl']:+.2f}, Avg Conf: {data['avg_confidence']:.2f}"
                    )
            
            # Add risk metrics
            report_lines.extend([
                "",
                "🛡️  RISK METRICS:",
                f"   Open Positions: {summary['open_positions']}",
                f"   Max Drawdown: ${summary['max_drawdown']:,.2f}",
                f"   Peak Equity: ${self.performance_metrics['peak_equity']:,.2f}",
                "",
                "="*80
            ])
            
            report = "\n".join(report_lines)
            
            # Save report to file
            report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            
            logging.info(f"📊 Performance report generated: {report_filename}")
            return report
            
        except Exception as e:
            logging.error(f"Error generating performance report: {e}")
            return "Error generating report"
    
    def print_performance_report(self):
        """Print performance report to console"""
        
        report = self.generate_performance_report()
        print(report)
    
    def cleanup(self):
        """Clean up logger resources"""
        
        try:
            # Final performance log
            self.log_performance_metrics()
            
            # Close any open file handles
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)
            
            logging.info("✅ Trade logger cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Error during logger cleanup: {e}")
    
    def get_trade_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        """Get specific trade by ID"""
        
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """Get all trades for a specific symbol"""
        
        return [trade for trade in self.trades if trade.symbol == symbol]
    
    def get_trades_by_date_range(self, start_date: str, end_date: str) -> List[TradeRecord]:
        """Get trades within a date range"""
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        filtered_trades = []
        for trade in self.trades:
            trade_dt = datetime.fromisoformat(trade.timestamp)
            if start_dt <= trade_dt <= end_dt:
                filtered_trades.append(trade)
        
        return filtered_trades
    
    def calculate_metrics_by_period(self, period: str = 'daily') -> Dict:
        """Calculate performance metrics by time period"""
        
        try:
            period_metrics = {}
            
            for trade in self.trades:
                if trade.profit_loss == 0:  # Skip entry-only trades
                    continue
                
                trade_dt = datetime.fromisoformat(trade.timestamp)
                
                if period == 'daily':
                    key = trade_dt.strftime('%Y-%m-%d')
                elif period == 'hourly':
                    key = trade_dt.strftime('%Y-%m-%d %H:00')
                elif period == 'weekly':
                    key = f"{trade_dt.year}-W{trade_dt.isocalendar()[1]}"
                else:
                    key = trade_dt.strftime('%Y-%m')
                
                if key not in period_metrics:
                    period_metrics[key] = {
                        'trades': 0, 'wins': 0, 'losses': 0,
                        'pnl': 0.0, 'volume': 0.0
                    }
                
                metrics = period_metrics[key]
                metrics['trades'] += 1
                metrics['pnl'] += trade.profit_loss
                metrics['volume'] += trade.quantity * trade.price
                
                if trade.profit_loss > 0:
                    metrics['wins'] += 1
                else:
                    metrics['losses'] += 1
            
            # Calculate derived metrics
            for metrics in period_metrics.values():
                metrics['win_rate'] = (metrics['wins'] / metrics['trades'] * 100) if metrics['trades'] > 0 else 0
                metrics['avg_pnl'] = metrics['pnl'] / metrics['trades'] if metrics['trades'] > 0 else 0
            
            return period_metrics
            
        except Exception as e:
            logging.error(f"Error calculating period metrics: {e}")
            return {}


# Legacy compatibility class
class TradeLogger(EnhancedTradeLogger):
    """Legacy compatibility wrapper"""
    
    def __init__(self, log_file: str = "trades.csv", log_level: int = logging.INFO):
        super().__init__(log_file, log_level)
        logging.info("Using legacy TradeLogger interface")


# Factory function for easy initialization
def create_trade_logger(config: Dict = None) -> EnhancedTradeLogger:
    """Factory function to create trade logger with configuration"""
    
    if config is None:
        config = {}
    
    return EnhancedTradeLogger(
        log_file=config.get('log_file', 'enhanced_trades.csv'),
        log_level=getattr(logging, config.get('log_level', 'INFO')),
        use_database=config.get('use_database', False),
        database_file=config.get('database_file', 'trades.db')
    )


if __name__ == "__main__":
    # Test the enhanced trade logger
    logger = EnhancedTradeLogger()
    
    # Mock trade object for testing
    class MockTrade:
        def __init__(self):
            self.timestamp = datetime.now().isoformat()
            self.symbol = 'XAUUSD'
            self.side = 'buy'
            self.quantity = 1.0
            self.price = 2000.50
            self.trade_id = 'test_trade_001'
            self.status = 'filled'
            self.execution_time_ms = 150.5
            self.slippage_ticks = 0.2
            self.commission = 0.0
    
    # Test logging trades
    entry_trade = MockTrade()
    logger.log_trade(
        entry_trade, 
        trade_type="entry",
        signal_confidence=0.75,
        signal_reasoning="Strong momentum signal",
        market_conditions={'volatility': 0.12, 'spread': 0.10}
    )
    
    # Test exit trade
    exit_trade = MockTrade()
    exit_trade.side = 'sell'
    exit_trade.price = 2002.00
    exit_trade.trade_id = 'test_trade_001_close'
    
    logger.log_trade(
        exit_trade,
        trade_type="exit",
        signal_confidence=0.80,
        signal_reasoning="Profit target reached",
        exit_trade=entry_trade
    )
    
    # Test performance metrics
    metrics = logger.log_performance_metrics()
    print(f"Performance Metrics: {json.dumps(metrics, indent=2)}")
    
    # Test analytics
    analytics = logger.get_trade_analytics()
    print(f"Trade Analytics: {json.dumps(analytics, indent=2)}")
    
    # Test ML data collection
    logger.collect_ml_features(
        'test_trade_001',
        {'momentum': 0.002, 'volatility': 0.12, 'order_flow': 0.35},
        1.50  # profit
    )
    
    ml_data = logger.get_ml_training_data()
    print(f"ML Training Data: {ml_data}")
    
    # Test report generation
    logger.print_performance_report()
    
    # Test data export
    export_file = logger.export_data('json')
    print(f"Exported to: {export_file}")
    
    # Cleanup
    logger.cleanup()