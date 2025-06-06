#!/usr/bin/env python3
"""
Level 7-A: Feature Engineering System
Merges KPIs + latest intelligence into flat feature rows for ML training
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering for ML-enhanced strategy scoring"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Data directories
        self.logs_dir = Path("logs")
        self.intel_dir = Path("intel")
        self.features_dir = Path("data/features")
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature sets
        self.kpi_features = ['gross_pnl', 'win_rate', 'profit_factor', 'sharpe_ratio', 
                           'total_trades', 'avg_win', 'avg_loss']
        self.intel_features = ['sentiment_avg', 'sentiment_positive_pct', 'regime_bull_pct', 
                             'regime_bear_pct', 'econ_high_impact_count', 'econ_upcoming_hours']
        
    def build_feature_dataset(self, lookback_days: int = 30) -> pd.DataFrame:
        """Build complete feature dataset from historical data"""
        
        print(f"🔧 Building ML feature dataset ({lookback_days} days lookback)")
        print("=" * 60)
        
        # Get date range
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=lookback_days)
        
        feature_rows = []
        
        # Generate features for each day in range
        for day_offset in range(lookback_days):
            feature_date = start_date + timedelta(days=day_offset)
            
            print(f"📅 Processing {feature_date}...")
            
            # Get KPI features for this date
            kpi_features = self._extract_kpi_features(feature_date)
            
            # Get intelligence features for this date
            intel_features = self._extract_intel_features(feature_date)
            
            # Get strategy performance for this date
            strategy_perfs = self._extract_strategy_performance(feature_date)
            
            # Create feature rows for each (strategy, date) combination
            for strategy in ['martingale', 'breakout', 'mean_reversion']:
                
                # Base features
                feature_row = {
                    'date': feature_date.isoformat(),
                    'strategy': strategy,
                    'child_id': 'trader_001',  # Could expand to multiple children
                    'timestamp': datetime.combine(feature_date, datetime.min.time()).isoformat()
                }
                
                # Add KPI features
                feature_row.update(kpi_features)
                
                # Add intelligence features
                feature_row.update(intel_features)
                
                # Add strategy-specific features
                strategy_perf = strategy_perfs.get(strategy, {})
                feature_row.update({
                    'strategy_pnl': strategy_perf.get('pnl', 0),
                    'strategy_trades': strategy_perf.get('trades', 0),
                    'strategy_win_rate': strategy_perf.get('win_rate', 0.5),
                    'strategy_score': strategy_perf.get('score', 50)
                })
                
                # Add target variable (next day performance)
                next_day = feature_date + timedelta(days=1)
                target = self._calculate_target_variable(strategy, next_day)
                feature_row.update(target)
                
                feature_rows.append(feature_row)
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_rows)
        
        # Feature engineering transformations
        df = self._engineer_derived_features(df)
        
        # Save feature dataset
        feature_file = self.features_dir / f"ml_features_{end_date.strftime('%Y%m%d')}.csv"
        df.to_csv(feature_file, index=False)
        
        print(f"✅ Feature dataset created: {len(df)} rows, {len(df.columns)} features")
        print(f"💾 Saved to: {feature_file}")
        
        # Display feature summary
        self._display_feature_summary(df)
        
        return df
    
    def _extract_kpi_features(self, date: datetime.date) -> Dict:
        """Extract KPI features for a given date"""
        
        kpi_features = {}
        
        # Look for parent summary files around this date
        date_str = date.strftime("%Y%m%d")
        summary_files = list(self.logs_dir.glob(f"parent_summary_{date_str}.json"))
        
        if not summary_files:
            # Try adjacent dates or latest
            for offset in [-1, 0, 1, -2, 2]:
                check_date = date + timedelta(days=offset)
                check_str = check_date.strftime("%Y%m%d")
                summary_files = list(self.logs_dir.glob(f"parent_summary_{check_str}.json"))
                if summary_files:
                    break
        
        if summary_files:
            # Load most recent summary
            with open(summary_files[0]) as f:
                summary_data = json.load(f)
            
            kpis = summary_data.get('kpis', {})
            
            # Extract standard KPI features
            for feature in self.kpi_features:
                kpi_features[f'kpi_{feature}'] = kpis.get(feature, 0)
                
        else:
            # Generate simulated KPIs for demo
            kpi_features = self._generate_simulated_kpis(date)
        
        return kpi_features
    
    def _extract_intel_features(self, date: datetime.date) -> Dict:
        """Extract intelligence features for a given date"""
        
        intel_features = {}
        
        # News sentiment features
        sentiment_features = self._extract_sentiment_features(date)
        intel_features.update(sentiment_features)
        
        # Market regime features
        regime_features = self._extract_regime_features(date)
        intel_features.update(regime_features)
        
        # Economic calendar features
        econ_features = self._extract_economic_features(date)
        intel_features.update(econ_features)
        
        return intel_features
    
    def _extract_sentiment_features(self, date: datetime.date) -> Dict:
        """Extract news sentiment features"""
        
        sentiment_file = self.intel_dir / "news_sentiment.csv"
        
        if sentiment_file.exists():
            try:
                df = pd.read_csv(sentiment_file)
                
                # Filter for relevant date (if timestamped)
                if 'published_at' in df.columns:
                    df['pub_date'] = pd.to_datetime(df['published_at']).dt.date
                    df = df[df['pub_date'] == date]
                
                if len(df) > 0:
                    sentiment_scores = df['sentiment_compound'].astype(float)
                    
                    return {
                        'sentiment_avg': float(sentiment_scores.mean()),
                        'sentiment_std': float(sentiment_scores.std()),
                        'sentiment_positive_pct': float((sentiment_scores > 0.1).mean()),
                        'sentiment_negative_pct': float((sentiment_scores < -0.1).mean()),
                        'sentiment_headline_count': len(df),
                        'sentiment_max': float(sentiment_scores.max()),
                        'sentiment_min': float(sentiment_scores.min())
                    }
            except Exception as e:
                print(f"⚠️ Error loading sentiment data: {e}")
        
        # Default sentiment features
        return {
            'sentiment_avg': 0.0,
            'sentiment_std': 0.2,
            'sentiment_positive_pct': 0.4,
            'sentiment_negative_pct': 0.3,
            'sentiment_headline_count': 0,
            'sentiment_max': 0.0,
            'sentiment_min': 0.0
        }
    
    def _extract_regime_features(self, date: datetime.date) -> Dict:
        """Extract market regime features"""
        
        regime_file = self.intel_dir / "market_regime.json"
        
        if regime_file.exists():
            try:
                with open(regime_file) as f:
                    regime_data = json.load(f)
                
                symbols = regime_data.get('symbols', {})
                summary = regime_data.get('summary', {})
                
                total_symbols = len(symbols)
                
                if total_symbols > 0:
                    return {
                        'regime_bull_pct': summary.get('bull_count', 0) / total_symbols,
                        'regime_bear_pct': summary.get('bear_count', 0) / total_symbols,
                        'regime_range_pct': summary.get('range_count', 0) / total_symbols,
                        'regime_avg_confidence': summary.get('avg_confidence', 0.5),
                        'regime_symbols_analyzed': total_symbols,
                        'regime_bull_count': summary.get('bull_count', 0),
                        'regime_bear_count': summary.get('bear_count', 0)
                    }
            except Exception as e:
                print(f"⚠️ Error loading regime data: {e}")
        
        # Default regime features
        return {
            'regime_bull_pct': 0.33,
            'regime_bear_pct': 0.33,
            'regime_range_pct': 0.33,
            'regime_avg_confidence': 0.5,
            'regime_symbols_analyzed': 0,
            'regime_bull_count': 0,
            'regime_bear_count': 0
        }
    
    def _extract_economic_features(self, date: datetime.date) -> Dict:
        """Extract economic calendar features"""
        
        econ_file = self.intel_dir / "econ_calendar.json"
        
        if econ_file.exists():
            try:
                with open(econ_file) as f:
                    econ_data = json.load(f)
                
                events = econ_data.get('events', [])
                
                # Count events by impact level
                high_impact = sum(1 for e in events if e.get('impact') == 'high')
                medium_impact = sum(1 for e in events if e.get('impact') == 'medium')
                total_events = len(events)
                
                # Calculate upcoming events (within 24 hours of date)
                target_datetime = datetime.combine(date, datetime.min.time())
                upcoming_count = 0
                
                for event in events:
                    try:
                        event_time = datetime.fromisoformat(event.get('time_utc', '').replace('Z', '+00:00'))
                        hours_diff = abs((event_time - target_datetime).total_seconds() / 3600)
                        if hours_diff <= 24:
                            upcoming_count += 1
                    except:
                        continue
                
                return {
                    'econ_total_events': total_events,
                    'econ_high_impact_count': high_impact,
                    'econ_medium_impact_count': medium_impact,
                    'econ_upcoming_24h': upcoming_count,
                    'econ_high_impact_pct': high_impact / total_events if total_events > 0 else 0,
                    'econ_impact_score': (high_impact * 3 + medium_impact * 2) / max(total_events, 1)
                }
            except Exception as e:
                print(f"⚠️ Error loading economic data: {e}")
        
        # Default economic features
        return {
            'econ_total_events': 0,
            'econ_high_impact_count': 0,
            'econ_medium_impact_count': 0,
            'econ_upcoming_24h': 0,
            'econ_high_impact_pct': 0.0,
            'econ_impact_score': 0.0
        }
    
    def _extract_strategy_performance(self, date: datetime.date) -> Dict[str, Dict]:
        """Extract strategy-specific performance for date"""
        
        strategy_perfs = {
            'martingale': {'pnl': 0, 'trades': 0, 'win_rate': 0.5, 'score': 50},
            'breakout': {'pnl': 0, 'trades': 0, 'win_rate': 0.5, 'score': 50},
            'mean_reversion': {'pnl': 0, 'trades': 0, 'win_rate': 0.5, 'score': 50}
        }
        
        # Try to load strategy scores from logs
        scores_file = self.logs_dir / "strategy_scores.json"
        if scores_file.exists():
            try:
                with open(scores_file) as f:
                    scores_data = json.load(f)
                
                for strategy_data in scores_data.get('scores', []):
                    strategy = strategy_data.get('strategy')
                    if strategy in strategy_perfs:
                        strategy_perfs[strategy] = {
                            'pnl': strategy_data.get('net_pnl', 0),
                            'trades': strategy_data.get('num_trades', 0),
                            'win_rate': strategy_data.get('win_rate', 50) / 100,
                            'score': strategy_data.get('score', 50)
                        }
            except Exception as e:
                print(f"⚠️ Error loading strategy scores: {e}")
        
        return strategy_perfs
    
    def _calculate_target_variable(self, strategy: str, target_date: datetime.date) -> Dict:
        """Calculate target variable (next day performance)"""
        
        # For now, simulate target based on strategy and date
        # In production, this would be actual next-day P&L
        
        np.random.seed(hash(f"{strategy}_{target_date}") % 10000)
        
        # Different strategies have different expected performance patterns
        if strategy == 'breakout':
            # Higher volatility, can have big wins or losses
            target_pnl = np.random.normal(5, 20)
        elif strategy == 'mean_reversion':
            # More consistent, smaller moves
            target_pnl = np.random.normal(2, 8)
        else:  # martingale
            # High win rate but occasional large losses
            if np.random.random() < 0.8:
                target_pnl = np.random.normal(3, 5)
            else:
                target_pnl = np.random.normal(-25, 10)
        
        return {
            'target_pnl': round(target_pnl, 2),
            'target_profitable': 1 if target_pnl > 0 else 0,
            'target_win_rate': 1.0 if target_pnl > 5 else 0.5 if target_pnl > -5 else 0.0
        }
    
    def _generate_simulated_kpis(self, date: datetime.date) -> Dict:
        """Generate simulated KPI features when no data available"""
        
        # Seed with date for consistency
        np.random.seed(date.toordinal() % 10000)
        
        return {
            'kpi_gross_pnl': round(np.random.normal(10, 50), 2),
            'kpi_win_rate': round(np.random.uniform(0.4, 0.8), 3),
            'kpi_profit_factor': round(np.random.uniform(0.8, 2.5), 2),
            'kpi_sharpe_ratio': round(np.random.normal(0.5, 0.3), 3),
            'kpi_total_trades': int(np.random.poisson(20)),
            'kpi_avg_win': round(np.random.uniform(5, 25), 2),
            'kpi_avg_loss': round(-np.random.uniform(5, 20), 2)
        }
    
    def _engineer_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from base features"""
        
        # Date-based features
        df['date_parsed'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date_parsed'].dt.dayofweek
        df['month'] = df['date_parsed'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['date_parsed'].dt.day >= 28).astype(int)
        
        # Strategy encoding
        strategy_dummies = pd.get_dummies(df['strategy'], prefix='strategy')
        df = pd.concat([df, strategy_dummies], axis=1)
        
        # Interaction features
        df['sentiment_regime_interaction'] = df['sentiment_avg'] * df['regime_bull_pct']
        df['pnl_trades_ratio'] = df['kpi_gross_pnl'] / (df['kpi_total_trades'] + 1)
        df['win_loss_ratio'] = df['kpi_avg_win'] / (abs(df['kpi_avg_loss']) + 1)
        df['econ_sentiment_interaction'] = df['econ_high_impact_count'] * df['sentiment_negative_pct']
        
        # Risk-adjusted metrics
        df['risk_adjusted_pnl'] = df['kpi_gross_pnl'] / (df['kpi_total_trades'] * abs(df['kpi_avg_loss']) + 1)
        df['volatility_proxy'] = df['sentiment_std'] + df['regime_avg_confidence']
        
        # Moving averages (if enough data)
        if len(df) > 7:
            for feature in ['sentiment_avg', 'kpi_gross_pnl', 'regime_bull_pct']:
                if feature in df.columns:
                    df[f'{feature}_ma7'] = df.groupby('strategy')[feature].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
        
        # Lag features
        for lag in [1, 3]:
            for feature in ['target_pnl', 'sentiment_avg']:
                if feature in df.columns:
                    df[f'{feature}_lag{lag}'] = df.groupby('strategy')[feature].shift(lag)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def _display_feature_summary(self, df: pd.DataFrame):
        """Display summary of created features"""
        
        print(f"\n📊 Feature Summary:")
        print(f"   Rows: {len(df)}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Strategies: {df['strategy'].nunique()}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Feature categories
        feature_categories = {
            'KPI': [col for col in df.columns if col.startswith('kpi_')],
            'Intelligence': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['sentiment_', 'regime_', 'econ_'])],
            'Strategy': [col for col in df.columns if col.startswith('strategy_')],
            'Target': [col for col in df.columns if col.startswith('target_')],
            'Derived': [col for col in df.columns if any(suffix in col for suffix in ['_ma7', '_lag', '_interaction', '_ratio'])]
        }
        
        print(f"\n📋 Feature Categories:")
        for category, features in feature_categories.items():
            print(f"   {category}: {len(features)} features")
        
        # Sample correlation with target
        if 'target_pnl' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlations = df[numeric_cols].corrwith(df['target_pnl']).abs().sort_values(ascending=False)
            
            print(f"\n🔗 Top 5 Features Correlated with Target P&L:")
            for feature, corr in correlations.head(5).items():
                if feature != 'target_pnl':
                    print(f"   {feature}: {corr:.3f}")
    
    def get_latest_features(self, strategy: str = None) -> Dict:
        """Get latest feature row for live inference"""
        
        today = datetime.now(timezone.utc).date()
        
        # Extract features for today
        kpi_features = self._extract_kpi_features(today)
        intel_features = self._extract_intel_features(today)
        
        if strategy:
            strategy_perfs = self._extract_strategy_performance(today)
            strategy_perf = strategy_perfs.get(strategy, {})
        else:
            strategy_perf = {}
        
        # Combine all features
        latest_features = {
            'date': today.isoformat(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kpi_features,
            **intel_features,
            'strategy_pnl': strategy_perf.get('pnl', 0),
            'strategy_trades': strategy_perf.get('trades', 0),
            'strategy_win_rate': strategy_perf.get('win_rate', 0.5),
            'strategy_score': strategy_perf.get('score', 50)
        }
        
        return latest_features
    
    def save_features_for_strategy(self, strategy: str, features: Dict) -> str:
        """Save feature row for a specific strategy"""
        
        # Add strategy to features
        features['strategy'] = strategy
        features['child_id'] = 'trader_001'
        
        # Save to strategy-specific file
        strategy_file = self.features_dir / f"latest_features_{strategy}.json"
        with open(strategy_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        return str(strategy_file)

def main():
    """Command line interface for feature engineering"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Feature Engineering (Level 7-A)")
    parser.add_argument('--lookback', type=int, default=30, help='Days to look back for features')
    parser.add_argument('--strategy', help='Generate features for specific strategy')
    parser.add_argument('--live', action='store_true', help='Generate live features only')
    
    args = parser.parse_args()
    
    print("🔧 ML Feature Engineering System (Level 7-A)")
    print("=" * 50)
    
    engineer = FeatureEngineer()
    
    if args.live:
        # Generate live features for inference
        print("📊 Generating live features for inference...")
        
        strategies = [args.strategy] if args.strategy else ['martingale', 'breakout', 'mean_reversion']
        
        for strategy in strategies:
            features = engineer.get_latest_features(strategy)
            saved_file = engineer.save_features_for_strategy(strategy, features)
            print(f"✅ {strategy}: {len(features)} features saved to {Path(saved_file).name}")
        
    else:
        # Build historical feature dataset
        df = engineer.build_feature_dataset(args.lookback)
        
        print(f"\n✅ Level 7-A Complete!")
        print(f"📊 Feature dataset: {len(df)} rows × {len(df.columns)} features")
        print(f"🎯 Ready for Level 7-B: Baseline Model Training")

if __name__ == "__main__":
    main()