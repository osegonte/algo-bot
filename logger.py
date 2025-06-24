#!/usr/bin/env python3
"""
Enhanced Performance Monitor for Aggressive XAU/USD Trading Strategy
Provides real-time analytics, risk monitoring, and performance tracking
"""

import logging
import csv
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np


class AggressivePerformanceMonitor:
    """Enhanced performance monitoring for aggressive trading strategy"""
    
    def __init__(self, log_file: str = "xauusd_aggressive_trades.csv", 
                 analytics_file: str = "aggressive_analytics.json"):
        
        self.log_file = log_file
        self.analytics_file = analytics_file
        
        # Enhanced tracking
        self.trades = []
        self.real_time_metrics = {}
        self.session_metrics = {}
        self.daily_metrics = {}
        
        # Risk monitoring
        self.risk_alerts = []
        self.drawdown_tracking = deque(maxlen=1000)
        self.equity_curve = deque(maxlen=1000)
        
        # Performance analytics
        self.sharpe_buffer = deque(maxlen=100)
        self.trade_durations = deque(maxlen=100)
        self.signal_strengths = deque(maxlen=100)
        
        # Session tracking
        self.session_start = datetime.now()
        self.starting_balance = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.max_drawdown = 0.0
        
        # Trade streaks
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
        # Strategy-specific metrics
        self.aggressive_metrics = {
            'large_wins': 0,      # Wins > $1.00
            'large_losses': 0,    # Losses > $0.50
            'quick_wins': 0,      # Wins in < 30 seconds
            'slow_trades': 0,     # Trades > 60 seconds
            'high_confidence_trades': 0,  # Confidence > 0.85
            'signal_accuracy': 0.0,
            'avg_signal_strength': 0.0
        }
        
        self._setup_enhanced_logging()
        
        logging.info("✅ Aggressive performance monitor initialized")
        logging.info(f"   📊 Trade log: {log_file}")
        logging.info(f"   📈 Analytics: {analytics_file}")
    
    def _setup_enhanced_logging(self):
        """Setup enhanced CSV logging with additional fields"""
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price', 
                    'trade_id', 'status', 'trade_type', 'profit_loss',
                    'profit_loss_ticks', 'trade_duration', 'signal_confidence',
                    'signal_strength', 'signal_reasoning', 'exit_reason',
                    'session_pnl', 'drawdown', 'win_streak', 'loss_streak'
                ])
            logging.info(f"Created enhanced CSV: {self.log_file}")
    
    def log_enhanced_trade(self, trade, trade_type: str = "entry", 
                          profit_loss: float = 0.0, **kwargs):
        """Log trade with enhanced metrics"""
        
        # Calculate enhanced metrics
        profit_loss_ticks = profit_loss / 0.10 if profit_loss != 0 else 0
        trade_duration = kwargs.get('duration', 0)
        signal_confidence = kwargs.get('confidence', 0)
        signal_strength = kwargs.get('strength', 0)
        signal_reasoning = kwargs.get('reasoning', '')
        exit_reason = kwargs.get('exit_reason', '')
        
        # Update session metrics
        if trade_type == "exit":
            self._update_session_metrics(profit_loss, trade_duration, 
                                       signal_confidence, signal_strength)
        
        # Create enhanced trade record
        enhanced_record = {
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'side': trade.side,
            'quantity': trade.quantity,
            'price': trade.price,
            'trade_id': trade.trade_id,
            'status': trade.status,
            'trade_type': trade_type,
            'profit_loss': profit_loss,
            'profit_loss_ticks': profit_loss_ticks,
            'trade_duration': trade_duration,
            'signal_confidence': signal_confidence,
            'signal_strength': signal_strength,
            'signal_reasoning': signal_reasoning,
            'exit_reason': exit_reason,
            'session_pnl': self.session_metrics.get('total_pnl', 0),
            'drawdown': self.max_drawdown,
            'win_streak': self.current_win_streak,
            'loss_streak': self.current_loss_streak
        }
        
        # Store trade
        self.trades.append(enhanced_record)
        
        # Write to CSV
        self._write_enhanced_csv(enhanced_record)
        
        # Update real-time analytics
        self._update_real_time_analytics()
        
        # Check for alerts
        self._check_risk_alerts(enhanced_record)
        
        # Log to console with enhanced info
        pnl_str = f" | P&L: ${profit_loss:+.2f} ({profit_loss_ticks:+.1f}t)" if profit_loss != 0 else ""
        conf_str = f" | Conf: {signal_confidence:.2f}" if signal_confidence > 0 else ""
        duration_str = f" | {trade_duration:.1f}s" if trade_duration > 0 else ""
        
        logging.info(f"ENHANCED TRADE [{trade_type.upper()}]: {trade.side.upper()} "
                    f"{trade.quantity} {trade.symbol} @ ${trade.price:.2f}"
                    f"{pnl_str}{conf_str}{duration_str}")
    
    def _write_enhanced_csv(self, record: Dict):
        """Write enhanced record to CSV"""
        
        try:
            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    record['timestamp'], record['symbol'], record['side'],
                    record['quantity'], record['price'], record['trade_id'],
                    record['status'], record['trade_type'], record['profit_loss'],
                    record['profit_loss_ticks'], record['trade_duration'],
                    record['signal_confidence'], record['signal_strength'],
                    record['signal_reasoning'], record['exit_reason'],
                    record['session_pnl'], record['drawdown'],
                    record['win_streak'], record['loss_streak']
                ])
        except Exception as e:
            logging.error(f"Error writing enhanced CSV: {e}")
    
    def _update_session_metrics(self, pnl: float, duration: float, 
                               confidence: float, strength: float):
        """Update session-level metrics"""
        
        # Initialize if first trade
        if not self.session_metrics:
            self.session_metrics = {
                'total_pnl': 0, 'total_trades': 0, 'winning_trades': 0,
                'total_duration': 0, 'avg_confidence': 0, 'avg_strength': 0
            }
        
        # Update basic metrics
        self.session_metrics['total_pnl'] += pnl
        self.session_metrics['total_trades'] += 1
        self.session_metrics['total_duration'] += duration
        
        if pnl > 0:
            self.session_metrics['winning_trades'] += 1
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
        else:
            self.current_loss_streak += 1
            self.current_win_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.current_loss_streak)
        
        # Update confidence and strength tracking
        if confidence > 0:
            self.signal_strengths.append(strength)
            # Running average
            total_conf = self.session_metrics['avg_confidence'] * (self.session_metrics['total_trades'] - 1)
            self.session_metrics['avg_confidence'] = (total_conf + confidence) / self.session_metrics['total_trades']
        
        # Update aggressive strategy metrics
        self._update_aggressive_metrics(pnl, duration, confidence, strength)
        
        # Update drawdown tracking
        self.current_balance = self.starting_balance + self.session_metrics['total_pnl']
        self.peak_balance = max(self.peak_balance, self.current_balance)
        current_drawdown = self.peak_balance - self.current_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Store for equity curve
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'balance': self.current_balance,
            'drawdown': current_drawdown
        })
    
    def _update_aggressive_metrics(self, pnl: float, duration: float, 
                                  confidence: float, strength: float):
        """Update aggressive strategy-specific metrics"""
        
        # Large wins/losses
        if pnl > 1.00:
            self.aggressive_metrics['large_wins'] += 1
        elif pnl < -0.50:
            self.aggressive_metrics['large_losses'] += 1
        
        # Trade speed
        if pnl > 0 and duration < 30:
            self.aggressive_metrics['quick_wins'] += 1
        elif duration > 60:
            self.aggressive_metrics['slow_trades'] += 1
        
        # High confidence trades
        if confidence > 0.85:
            self.aggressive_metrics['high_confidence_trades'] += 1
        
        # Update averages
        if self.signal_strengths:
            self.aggressive_metrics['avg_signal_strength'] = np.mean(list(self.signal_strengths))
    
    def _update_real_time_analytics(self):
        """Update real-time analytics"""
        
        if not self.trades:
            return
        
        # Get recent completed trades
        completed_trades = [t for t in self.trades if t['trade_type'] == 'exit']
        
        if not completed_trades:
            return
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['profit_loss'] for t in completed_trades[-20:]]  # Last 20 trades
        if len(returns) >= 5:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return) if std_return > 0 else 0
            self.real_time_metrics['sharpe_ratio'] = sharpe
        
        # Calculate other metrics
        self.real_time_metrics.update({
            'win_rate': len([t for t in completed_trades if t['profit_loss'] > 0]) / len(completed_trades) * 100,
            'avg_win': np.mean([t['profit_loss'] for t in completed_trades if t['profit_loss'] > 0]) if any(t['profit_loss'] > 0 for t in completed_trades) else 0,
            'avg_loss': np.mean([t['profit_loss'] for t in completed_trades if t['profit_loss'] < 0]) if any(t['profit_loss'] < 0 for t in completed_trades) else 0,
            'profit_factor': self._calculate_profit_factor(completed_trades),
            'max_consecutive_wins': self.max_win_streak,
            'max_consecutive_losses': self.max_loss_streak,
            'current_drawdown': self.max_drawdown,
            'total_trades': len(completed_trades),
            'session_duration': (datetime.now() - self.session_start).total_seconds() / 3600
        })
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor"""
        
        gross_profit = sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0)
        gross_loss = abs(sum(t['profit_loss'] for t in trades if t['profit_loss'] < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _check_risk_alerts(self, trade_record: Dict):
        """Check for risk-based alerts"""
        
        alerts = []
        
        # Large loss alert
        if trade_record['profit_loss'] < -1.50:
            alerts.append(f"LARGE LOSS: ${trade_record['profit_loss']:.2f}")
        
        # Large win alert
        if trade_record['profit_loss'] > 2.00:
            alerts.append(f"LARGE WIN: ${trade_record['profit_loss']:.2f}")
        
        # Consecutive loss alert
        if self.current_loss_streak >= 3:
            alerts.append(f"CONSECUTIVE LOSSES: {self.current_loss_streak}")
        
        # Drawdown alert
        if self.max_drawdown > 5.0:
            alerts.append(f"HIGH DRAWDOWN: ${self.max_drawdown:.2f}")
        
        # Low confidence alert
        if trade_record['signal_confidence'] > 0 and trade_record['signal_confidence'] < 0.70:
            alerts.append(f"LOW CONFIDENCE TRADE: {trade_record['signal_confidence']:.2f}")
        
        # Store and log alerts
        for alert in alerts:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'alert': alert,
                'trade_id': trade_record['trade_id']
            }
            self.risk_alerts.append(alert_record)
            
            print(f"\n🚨 RISK ALERT: {alert}")
            logging.warning(f"RISK ALERT: {alert} | Trade: {trade_record['trade_id']}")
    
    def get_enhanced_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        completed_trades = [t for t in self.trades if t['trade_type'] == 'exit']
        
        if not completed_trades:
            return {'status': 'no_completed_trades'}
        
        # Basic metrics
        total_pnl = sum(t['profit_loss'] for t in completed_trades)
        winning_trades = [t for t in completed_trades if t['profit_loss'] > 0]
        losing_trades = [t for t in completed_trades if t['profit_loss'] < 0]
        
        win_rate = len(winning_trades) / len(completed_trades) * 100
        avg_win = np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_loss'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        returns = [t['profit_loss'] for t in completed_trades]
        sharpe_ratio = self.real_time_metrics.get('sharpe_ratio', 0)
        profit_factor = self._calculate_profit_factor(completed_trades)
        
        # Aggressive strategy metrics
        large_wins_pct = (self.aggressive_metrics['large_wins'] / len(completed_trades)) * 100
        quick_wins_pct = (self.aggressive_metrics['quick_wins'] / len(winning_trades)) * 100 if winning_trades else 0
        high_conf_pct = (self.aggressive_metrics['high_confidence_trades'] / len(completed_trades)) * 100
        
        # Time-based metrics
        session_hours = (datetime.now() - self.session_start).total_seconds() / 3600
        trades_per_hour = len(completed_trades) / session_hours if session_hours > 0 else 0
        avg_duration = np.mean([t['trade_duration'] for t in completed_trades if t['trade_duration'] > 0])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'session_duration_hours': session_hours,
            
            # Core Performance
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': total_pnl / len(completed_trades),
            
            # Risk Metrics
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.peak_balance - self.current_balance,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'largest_win': max(returns) if returns else 0,
            'largest_loss': min(returns) if returns else 0,
            
            # Streak Analysis
            'current_win_streak': self.current_win_streak,
            'current_loss_streak': self.current_loss_streak,
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak,
            
            # Aggressive Strategy Metrics
            'large_wins_count': self.aggressive_metrics['large_wins'],
            'large_wins_percentage': large_wins_pct,
            'quick_wins_percentage': quick_wins_pct,
            'high_confidence_percentage': high_conf_pct,
            'avg_signal_strength': self.aggressive_metrics['avg_signal_strength'],
            
            # Efficiency Metrics
            'trades_per_hour': trades_per_hour,
            'avg_trade_duration': avg_duration,
            'pnl_per_hour': total_pnl / session_hours if session_hours > 0 else 0,
            
            # Alert Summary
            'total_alerts': len(self.risk_alerts),
            'recent_alerts': len([a for a in self.risk_alerts if 
                                datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)])
        }
    
    def print_enhanced_dashboard(self):
        """Print comprehensive trading dashboard"""
        
        summary = self.get_enhanced_performance_summary()
        
        if summary.get('status') == 'no_completed_trades':
            print("\n📊 AGGRESSIVE STRATEGY DASHBOARD")
            print("="*60)
            print("   No completed trades yet")
            return
        
        print("\n" + "="*80)
        print("           🚀 AGGRESSIVE XAU/USD STRATEGY DASHBOARD")
        print("="*80)
        
        # Session Info
        print(f"📅 Session Duration: {summary['session_duration_hours']:.2f} hours")
        print(f"⚡ Trading Rate: {summary['trades_per_hour']:.1f} trades/hour")
        print(f"⏱️  Avg Trade Duration: {summary['avg_trade_duration']:.1f} seconds")
        
        print("\n📈 CORE PERFORMANCE:")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.1f}% ({summary['winning_trades']}/{summary['total_trades']})")
        print(f"   Total P&L: ${summary['total_pnl']:+.2f}")
        print(f"   Avg Trade: ${summary['avg_trade']:+.2f}")
        print(f"   P&L/Hour: ${summary['pnl_per_hour']:+.2f}")
        
        print(f"\n💰 WIN/LOSS ANALYSIS:")
        print(f"   Average Win: ${summary['avg_win']:+.2f}")
        print(f"   Average Loss: ${summary['avg_loss']:+.2f}")
        print(f"   Profit Factor: {summary['profit_factor']:.2f}")
        print(f"   Largest Win: ${summary['largest_win']:+.2f}")
        print(f"   Largest Loss: ${summary['largest_loss']:+.2f}")
        
        print(f"\n🎯 RISK METRICS:")
        print(f"   Max Drawdown: ${summary['max_drawdown']:.2f}")
        print(f"   Current Drawdown: ${summary['current_drawdown']:.2f}")
        print(f"   Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        
        print(f"\n🔥 AGGRESSIVE STRATEGY METRICS:")
        print(f"   Large Wins (>$1.00): {summary['large_wins_count']} ({summary['large_wins_percentage']:.1f}%)")
        print(f"   Quick Wins (<30s): {summary['quick_wins_percentage']:.1f}%")
        print(f"   High Confidence (>85%): {summary['high_confidence_percentage']:.1f}%")
        print(f"   Avg Signal Strength: {summary['avg_signal_strength']:.2f}")
        
        print(f"\n📊 STREAK ANALYSIS:")
        if summary['current_win_streak'] > 0:
            print(f"   Current Streak: 🟢 {summary['current_win_streak']} wins")
        elif summary['current_loss_streak'] > 0:
            print(f"   Current Streak: 🔴 {summary['current_loss_streak']} losses")
        else:
            print(f"   Current Streak: ⚪ None")
        
        print(f"   Best Win Streak: {summary['max_win_streak']}")
        print(f"   Worst Loss Streak: {summary['max_loss_streak']}")
        
        # Alert summary
        if summary['total_alerts'] > 0:
            print(f"\n🚨 ALERTS:")
            print(f"   Total Alerts: {summary['total_alerts']}")
            print(f"   Recent Alerts (1h): {summary['recent_alerts']}")
            
            # Show recent alerts
            recent_alerts = [a for a in self.risk_alerts[-5:]]
            for alert in recent_alerts:
                time_str = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                print(f"   {time_str}: {alert['alert']}")
        
        print("="*80)
    
    def export_enhanced_analytics(self, filename: Optional[str] = None) -> str:
        """Export detailed analytics to JSON"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aggressive_analytics_{timestamp}.json"
        
        analytics_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'duration_hours': (datetime.now() - self.session_start).total_seconds() / 3600
            },
            'performance_summary': self.get_enhanced_performance_summary(),
            'aggressive_metrics': self.aggressive_metrics,
            'real_time_metrics': self.real_time_metrics,
            'risk_alerts': self.risk_alerts[-50:],  # Last 50 alerts
            'equity_curve': [
                {
                    'timestamp': point['timestamp'].isoformat(),
                    'balance': point['balance'],
                    'drawdown': point['drawdown']
                }
                for point in list(self.equity_curve)[-100:]  # Last 100 points
            ],
            'recent_trades': [
                trade for trade in self.trades[-20:]  # Last 20 trades
                if trade['trade_type'] == 'exit'
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)
            
            logging.info(f"✅ Analytics exported to: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting analytics: {e}")
            return ""
    
    def get_trade_analysis(self, lookback_trades: int = 20) -> Dict:
        """Analyze recent trade patterns"""
        
        completed_trades = [t for t in self.trades if t['trade_type'] == 'exit']
        recent_trades = completed_trades[-lookback_trades:] if completed_trades else []
        
        if not recent_trades:
            return {'status': 'insufficient_data'}
        
        # Time-based analysis
        morning_trades = [t for t in recent_trades if 
                         datetime.fromisoformat(t['timestamp']).hour < 12]
        afternoon_trades = [t for t in recent_trades if 
                          12 <= datetime.fromisoformat(t['timestamp']).hour < 18]
        evening_trades = [t for t in recent_trades if 
                         datetime.fromisoformat(t['timestamp']).hour >= 18]
        
        # Signal strength analysis
        high_strength_trades = [t for t in recent_trades if t['signal_strength'] > 0.8]
        medium_strength_trades = [t for t in recent_trades if 0.6 <= t['signal_strength'] <= 0.8]
        low_strength_trades = [t for t in recent_trades if t['signal_strength'] < 0.6]
        
        # Duration analysis
        quick_trades = [t for t in recent_trades if t['trade_duration'] < 30]
        normal_trades = [t for t in recent_trades if 30 <= t['trade_duration'] <= 60]
        slow_trades = [t for t in recent_trades if t['trade_duration'] > 60]
        
        return {
            'lookback_trades': len(recent_trades),
            'time_analysis': {
                'morning_win_rate': self._calculate_win_rate(morning_trades),
                'afternoon_win_rate': self._calculate_win_rate(afternoon_trades),
                'evening_win_rate': self._calculate_win_rate(evening_trades)
            },
            'strength_analysis': {
                'high_strength_win_rate': self._calculate_win_rate(high_strength_trades),
                'medium_strength_win_rate': self._calculate_win_rate(medium_strength_trades),
                'low_strength_win_rate': self._calculate_win_rate(low_strength_trades)
            },
            'duration_analysis': {
                'quick_trades_win_rate': self._calculate_win_rate(quick_trades),
                'normal_trades_win_rate': self._calculate_win_rate(normal_trades),
                'slow_trades_win_rate': self._calculate_win_rate(slow_trades)
            },
            'recommendations': self._generate_recommendations(recent_trades)
        }
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate for a subset of trades"""
        if not trades:
            return 0.0
        return len([t for t in trades if t['profit_loss'] > 0]) / len(trades) * 100
    
    def _generate_recommendations(self, recent_trades: List[Dict]) -> List[str]:
        """Generate strategy recommendations based on recent performance"""
        
        recommendations = []
        
        if not recent_trades:
            return ["Insufficient data for recommendations"]
        
        # Win rate analysis
        win_rate = self._calculate_win_rate(recent_trades)
        if win_rate < 50:
            recommendations.append("Consider tightening entry criteria - win rate below 50%")
        elif win_rate > 75:
            recommendations.append("Consider loosening entry criteria - may be missing opportunities")
        
        # Duration analysis
        avg_duration = np.mean([t['trade_duration'] for t in recent_trades if t['trade_duration'] > 0])
        if avg_duration > 70:
            recommendations.append("Trades running long - consider tighter profit targets")
        elif avg_duration < 20:
            recommendations.append("Very quick trades - consider wider profit targets")
        
        # Signal strength analysis
        avg_strength = np.mean([t['signal_strength'] for t in recent_trades if t['signal_strength'] > 0])
        if avg_strength < 0.6:
            recommendations.append("Low average signal strength - increase confidence threshold")
        
        # Loss streak analysis
        if self.current_loss_streak >= 2:
            recommendations.append("Consider taking a break - current loss streak detected")
        
        # Drawdown analysis
        if self.max_drawdown > 3.0:
            recommendations.append("High drawdown - consider reducing position size")
        
        return recommendations or ["Strategy performing within normal parameters"]
    
    def reset_session_metrics(self):
        """Reset session-level metrics"""
        
        self.session_start = datetime.now()
        self.session_metrics = {}
        self.starting_balance = self.current_balance
        self.peak_balance = self.current_balance
        self.max_drawdown = 0.0
        self.current_win_streak = 0
        self.current_loss_streak = 0
        
        # Reset aggressive metrics
        self.aggressive_metrics = {
            'large_wins': 0, 'large_losses': 0, 'quick_wins': 0,
            'slow_trades': 0, 'high_confidence_trades': 0,
            'signal_accuracy': 0.0, 'avg_signal_strength': 0.0
        }
        
        logging.info("📊 Session metrics reset")
    
    def cleanup(self):
        """Enhanced cleanup with final analytics"""
        
        try:
            # Print final dashboard
            self.print_enhanced_dashboard()
            
            # Generate trade analysis
            analysis = self.get_trade_analysis()
            if analysis.get('status') != 'insufficient_data':
                print(f"\n🔍 RECENT TRADE ANALYSIS:")
                print(f"   High Strength Win Rate: {analysis['strength_analysis']['high_strength_win_rate']:.1f}%")
                print(f"   Quick Trades Win Rate: {analysis['duration_analysis']['quick_trades_win_rate']:.1f}%")
                
                print(f"\n💡 RECOMMENDATIONS:")
                for rec in analysis['recommendations']:
                    print(f"   • {rec}")
            
            # Export final analytics
            export_file = self.export_enhanced_analytics()
            if export_file:
                print(f"\n📁 Final analytics exported to: {export_file}")
            
            logging.info("✅ Enhanced performance monitor cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during enhanced cleanup: {e}")


if __name__ == "__main__":
    # Test the enhanced performance monitor
    print("🧪 Testing Enhanced Performance Monitor...")
    
    monitor = AggressivePerformanceMonitor()
    
    # Simulate some trades
    class MockTrade:
        def __init__(self, side: str, price: float):
            self.timestamp = datetime.now().isoformat()
            self.symbol = 'XAUUSD'
            self.side = side
            self.quantity = 0.3
            self.price = price
            self.trade_id = f'test_{side}_{int(time.time())}'
            self.status = 'filled'
    
    # Simulate trading session
    print("\n🎮 Simulating aggressive trading session...")
    
    base_price = 2000.0
    monitor.starting_balance = 100000.0
    monitor.current_balance = 100000.0
    monitor.peak_balance = 100000.0
    
    # Simulate 10 trades
    for i in range(10):
        # Entry
        side = 'buy' if i % 2 == 0 else 'sell'
        entry_price = base_price + np.random.uniform(-1, 1)
        entry_trade = MockTrade(side, entry_price)
        
        monitor.log_enhanced_trade(
            entry_trade, 
            trade_type="entry",
            confidence=np.random.uniform(0.65, 0.95),
            strength=np.random.uniform(0.5, 0.9),
            reasoning="Simulated trade"
        )
        
        # Exit
        pnl = np.random.uniform(-0.8, 1.2)  # Random P&L
        duration = np.random.uniform(15, 90)  # Random duration
        exit_price = entry_price + (pnl if side == 'buy' else -pnl)
        exit_trade = MockTrade('sell' if side == 'buy' else 'buy', exit_price)
        
        monitor.log_enhanced_trade(
            exit_trade,
            trade_type="exit", 
            profit_loss=pnl,
            duration=duration,
            confidence=np.random.uniform(0.65, 0.95),
            strength=np.random.uniform(0.5, 0.9),
            exit_reason="Target/Stop"
        )
        
        time.sleep(0.1)  # Small delay
    
    # Display results
    monitor.print_enhanced_dashboard()
    
    # Test trade analysis
    analysis = monitor.get_trade_analysis()
    print(f"\n🔍 Trade Analysis: {analysis}")
    
    # Test export
    export_file = monitor.export_enhanced_analytics()
    print(f"📁 Exported to: {export_file}")
    
    print("\n✅ Enhanced performance monitor test completed!")