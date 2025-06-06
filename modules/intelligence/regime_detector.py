#!/usr/bin/env python3
"""
Level 6-C: Market Regime Detector
Classifies each asset as bull, bear, or range using SMA slopes and ATR percentiles
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class RegimeDetector:
    """Detect market regimes using technical indicators"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.intel_dir = Path("intel")
        self.intel_dir.mkdir(exist_ok=True)
        
        # Historical data directory
        self.data_dir = Path("data/historical")
        
        # Regime detection parameters
        self.sma_short = 50
        self.sma_long = 200
        self.atr_period = 14
        self.lookback_days = 30  # Days to look back for regime analysis
        
        # Default symbols to analyze
        self.symbols = self.config.get("symbols", ["AAPL", "TSLA", "MSFT", "EURUSD"])
    
    def detect_regimes(self) -> Dict:
        """Main regime detection workflow"""
        
        print(f"📊 Starting regime detection for {len(self.symbols)} symbols")
        print(f"🔍 Parameters: SMA({self.sma_short}/{self.sma_long}), ATR({self.atr_period})")
        
        regime_results = {}
        detection_timestamp = datetime.now(timezone.utc)
        
        for symbol in self.symbols:
            print(f"\n📈 Analyzing {symbol}...")
            
            try:
                # Load historical data
                price_data = self._load_price_data(symbol)
                
                if price_data is None or len(price_data) < self.sma_long:
                    print(f"⚠️ Insufficient data for {symbol}")
                    regime_results[symbol] = self._create_insufficient_data_result(symbol)
                    continue
                
                # Calculate regime
                regime_info = self._calculate_regime(symbol, price_data)
                regime_results[symbol] = regime_info
                
                print(f"✅ {symbol}: {regime_info['regime'].upper()} regime (confidence: {regime_info['confidence']:.1%})")
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                regime_results[symbol] = self._create_error_result(symbol, str(e))
        
        # Create comprehensive results
        market_regime_data = {
            'timestamp': detection_timestamp.isoformat(),
            'detection_date': detection_timestamp.date().isoformat(),
            'parameters': {
                'sma_short': self.sma_short,
                'sma_long': self.sma_long,
                'atr_period': self.atr_period,
                'lookback_days': self.lookback_days
            },
            'symbols': regime_results,
            'summary': self._generate_summary(regime_results)
        }
        
        # Save results
        output_file = self.intel_dir / "market_regime.json"
        with open(output_file, 'w') as f:
            json.dump(market_regime_data, f, indent=2)
        
        print(f"\n💾 Regime analysis saved to: {output_file}")
        
        # Print summary
        summary = market_regime_data['summary']
        print(f"\n📊 Market Regime Summary:")
        print(f"   🐂 Bull markets: {summary['bull_count']}")
        print(f"   🐻 Bear markets: {summary['bear_count']}")
        print(f"   📦 Range markets: {summary['range_count']}")
        print(f"   ❓ Insufficient data: {summary['insufficient_data_count']}")
        
        return {
            'success': True,
            'symbols_analyzed': len(self.symbols),
            'output_file': str(output_file),
            'summary': summary
        }
    
    def _load_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical price data for symbol"""
        
        # Look for various file patterns
        patterns = [
            f"{symbol}_1min_*_*.json",
            f"{symbol}_1min_*_*.csv",
            f"{symbol}_daily_*.json",
            f"{symbol}_daily_*.csv"
        ]
        
        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                # Use the most recent file
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                data_file = files[0]
                
                print(f"📂 Loading data from: {data_file.name}")
                
                if data_file.suffix == '.json':
                    return self._load_json_data(data_file)
                elif data_file.suffix == '.csv':
                    return self._load_csv_data(data_file)
        
        # If no historical data, generate sample data for demo
        print(f"⚠️ No historical data found for {symbol}, generating sample data")
        return self._generate_sample_data(symbol)
    
    def _load_json_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load price data from JSON file"""
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            bars = data.get('bars', [])
            if not bars:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Convert timestamp and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 't' in df.columns:
                df['timestamp'] = pd.to_datetime(df['t'])
                df.set_index('timestamp', inplace=True)
            
            # Resample to daily if we have intraday data
            if len(df) > 1000:  # Likely intraday data
                df = self._resample_to_daily(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading JSON data: {e}")
            return None
    
    def _load_csv_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load price data from CSV file"""
        
        try:
            df = pd.read_csv(file_path)
            
            # Standardize columns
            df = self._standardize_columns(df)
            
            # Set timestamp index
            timestamp_cols = ['timestamp', 't', 'date', 'Date']
            timestamp_col = None
            
            for col in timestamp_cols:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df.set_index(timestamp_col, inplace=True)
            
            # Resample if intraday
            if len(df) > 1000:
                df = self._resample_to_daily(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources"""
        
        # Common column mappings
        column_mapping = {
            'o': 'open',
            'h': 'high', 
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                if 'close' in df.columns:
                    df[col] = df['close']  # Use close as fallback
                else:
                    df[col] = 100.0  # Default value
        
        return df
    
    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample intraday data to daily OHLCV"""
        
        try:
            # Resample to daily
            daily_df = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in df.columns else lambda x: 0
            }).dropna()
            
            return daily_df
            
        except Exception as e:
            print(f"⚠️ Resampling failed, using original data: {e}")
            return df
    
    def _generate_sample_data(self, symbol: str) -> pd.DataFrame:
        """Generate sample price data for demonstration"""
        
        # Generate 300 days of sample data
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
        
        # Different patterns for different symbols
        if symbol == "AAPL":
            base_price = 190.0
            trend = 0.0005  # Slight uptrend
            volatility = 0.02
        elif symbol == "TSLA":
            base_price = 250.0
            trend = -0.0002  # Slight downtrend
            volatility = 0.04  # More volatile
        elif symbol == "EURUSD":
            base_price = 1.0500
            trend = 0.0001
            volatility = 0.005  # Lower volatility for forex
        else:
            base_price = 150.0
            trend = 0.0003
            volatility = 0.025
        
        # Generate price series with trend and noise
        np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
        
        returns = np.random.normal(trend, volatility, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            daily_range = close * np.random.uniform(0.005, 0.03)
            
            open_price = close + np.random.normal(0, daily_range * 0.3)
            high_price = max(open_price, close) + np.random.uniform(0, daily_range * 0.5)
            low_price = min(open_price, close) - np.random.uniform(0, daily_range * 0.5)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })
        
        df = pd.DataFrame(data, index=dates)
        
        print(f"📊 Generated {len(df)} days of sample data for {symbol}")
        return df
    
    def _calculate_regime(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Calculate market regime for a symbol"""
        
        # Calculate moving averages
        df[f'sma_{self.sma_short}'] = df['close'].rolling(window=self.sma_short).mean()
        df[f'sma_{self.sma_long}'] = df['close'].rolling(window=self.sma_long).mean()
        
        # Calculate ATR
        df['atr'] = self._calculate_atr(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # SMA slope analysis (using last 10 days)
        sma_short_slope = self._calculate_slope(df[f'sma_{self.sma_short}'].tail(10))
        sma_long_slope = self._calculate_slope(df[f'sma_{self.sma_long}'].tail(10))
        
        # ATR percentile (14-day ATR vs 100-day history)
        atr_percentile = self._calculate_atr_percentile(df['atr'], lookback=100)
        
        # Current price vs SMAs
        price = latest['close']
        sma_short_val = latest[f'sma_{self.sma_short}']
        sma_long_val = latest[f'sma_{self.sma_long}']
        
        # Regime detection logic
        regime, confidence = self._determine_regime(
            price, sma_short_val, sma_long_val,
            sma_short_slope, sma_long_slope, atr_percentile
        )
        
        return {
            'symbol': symbol,
            'regime': regime,
            'confidence': confidence,
            'price': round(price, 4),
            'sma_50': round(sma_short_val, 4),
            'sma_200': round(sma_long_val, 4),
            'sma_50_slope': round(sma_short_slope, 6),
            'sma_200_slope': round(sma_long_slope, 6),
            'atr_percentile': round(atr_percentile, 2),
            'analysis_date': datetime.now(timezone.utc).date().isoformat(),
            'data_points': len(df),
            'indicators': {
                'price_above_sma50': price > sma_short_val,
                'price_above_sma200': price > sma_long_val,
                'sma50_above_sma200': sma_short_val > sma_long_val,
                'sma50_rising': sma_short_slope > 0,
                'sma200_rising': sma_long_slope > 0,
                'high_volatility': atr_percentile > 70
            }
        }
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate slope of a price series"""
        
        if len(series) < 2:
            return 0.0
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 2:
            return 0.0
        
        x = np.arange(len(clean_series))
        y = clean_series.values
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def _calculate_atr_percentile(self, atr_series: pd.Series, lookback: int = 100) -> float:
        """Calculate ATR percentile vs historical values"""
        
        if len(atr_series) < lookback:
            lookback = len(atr_series)
        
        recent_atr = atr_series.iloc[-1]
        historical_atr = atr_series.tail(lookback)
        
        if pd.isna(recent_atr) or len(historical_atr) == 0:
            return 50.0  # Default to median
        
        percentile = (historical_atr < recent_atr).sum() / len(historical_atr) * 100
        
        return percentile
    
    def _determine_regime(self, price: float, sma50: float, sma200: float,
                         sma50_slope: float, sma200_slope: float, atr_percentile: float) -> Tuple[str, float]:
        """Determine market regime based on indicators"""
        
        # Initialize scores
        bull_score = 0
        bear_score = 0
        range_score = 0
        
        # Price position relative to SMAs
        if price > sma50:
            bull_score += 2
        else:
            bear_score += 2
        
        if price > sma200:
            bull_score += 3
        else:
            bear_score += 3
        
        # SMA alignment
        if sma50 > sma200:
            bull_score += 2
        else:
            bear_score += 2
        
        # SMA slopes
        if sma50_slope > 0.001:  # Rising
            bull_score += 2
        elif sma50_slope < -0.001:  # Falling
            bear_score += 2
        else:  # Flat
            range_score += 1
        
        if sma200_slope > 0.0005:  # Rising (slower threshold)
            bull_score += 1
        elif sma200_slope < -0.0005:  # Falling
            bear_score += 1
        else:  # Flat
            range_score += 1
        
        # Volatility consideration
        if atr_percentile > 80:  # High volatility
            # High volatility can indicate strong trends or uncertainty
            max_score = max(bull_score, bear_score)
            if max_score > 5:  # Strong trend
                if bull_score > bear_score:
                    bull_score += 1
                else:
                    bear_score += 1
            else:  # Uncertain direction
                range_score += 2
        elif atr_percentile < 20:  # Low volatility
            range_score += 2
        
        # Determine regime
        max_score = max(bull_score, bear_score, range_score)
        total_score = bull_score + bear_score + range_score
        
        if max_score == bull_score:
            regime = "bull"
            confidence = bull_score / total_score if total_score > 0 else 0.5
        elif max_score == bear_score:
            regime = "bear"
            confidence = bear_score / total_score if total_score > 0 else 0.5
        else:
            regime = "range"
            confidence = range_score / total_score if total_score > 0 else 0.5
        
        # Adjust confidence based on score dominance
        if max_score >= 8:  # Strong signal
            confidence = min(confidence * 1.2, 0.95)
        elif max_score <= 3:  # Weak signal
            confidence = max(confidence * 0.8, 0.4)
        
        return regime, confidence
    
    def _create_insufficient_data_result(self, symbol: str) -> Dict:
        """Create result for symbols with insufficient data"""
        
        return {
            'symbol': symbol,
            'regime': 'unknown',
            'confidence': 0.0,
            'error': 'insufficient_data',
            'analysis_date': datetime.now(timezone.utc).date().isoformat(),
            'data_points': 0
        }
    
    def _create_error_result(self, symbol: str, error_msg: str) -> Dict:
        """Create result for symbols with analysis errors"""
        
        return {
            'symbol': symbol,
            'regime': 'error',
            'confidence': 0.0,
            'error': error_msg,
            'analysis_date': datetime.now(timezone.utc).date().isoformat(),
            'data_points': 0
        }
    
    def _generate_summary(self, regime_results: Dict) -> Dict:
        """Generate summary of regime analysis"""
        
        summary = {
            'total_symbols': len(regime_results),
            'bull_count': 0,
            'bear_count': 0,
            'range_count': 0,
            'error_count': 0,
            'insufficient_data_count': 0,
            'avg_confidence': 0.0,
            'bull_symbols': [],
            'bear_symbols': [],
            'range_symbols': []
        }
        
        confidences = []
        
        for symbol, result in regime_results.items():
            regime = result.get('regime', 'error')
            confidence = result.get('confidence', 0.0)
            
            if regime == 'bull':
                summary['bull_count'] += 1
                summary['bull_symbols'].append(symbol)
                confidences.append(confidence)
            elif regime == 'bear':
                summary['bear_count'] += 1
                summary['bear_symbols'].append(symbol)
                confidences.append(confidence)
            elif regime == 'range':
                summary['range_count'] += 1
                summary['range_symbols'].append(symbol)
                confidences.append(confidence)
            elif regime == 'unknown':
                summary['insufficient_data_count'] += 1
            else:
                summary['error_count'] += 1
        
        # Calculate average confidence
        if confidences:
            summary['avg_confidence'] = sum(confidences) / len(confidences)
        
        return summary

def load_config() -> Dict:
    """Load configuration from file"""
    config_file = Path("config/base_config.yaml")
    
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
            return config.get('regime_detection', {
                'symbols': ['AAPL', 'TSLA', 'MSFT', 'EURUSD']
            })
    
    return {
        'symbols': ['AAPL', 'TSLA', 'MSFT', 'EURUSD']
    }

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Regime Detector (Level 6-C)")
    parser.add_argument('--symbols', nargs='+', help='Symbols to analyze (overrides config)')
    parser.add_argument('--sma-short', type=int, default=50, help='Short SMA period')
    parser.add_argument('--sma-long', type=int, default=200, help='Long SMA period')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR calculation period')
    parser.add_argument('--show-details', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    print("📊 Market Regime Detector (Level 6-C)")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    if args.symbols:
        config['symbols'] = args.symbols
    
    # Initialize detector with custom parameters
    detector = RegimeDetector(config)
    detector.sma_short = args.sma_short
    detector.sma_long = args.sma_long
    detector.atr_period = args.atr_period
    
    # Run regime detection
    result = detector.detect_regimes()
    
    if result['success']:
        print(f"\n✅ Level 6-C Complete!")
        print(f"📊 Symbols analyzed: {result['symbols_analyzed']}")
        print(f"💾 Output: {result['output_file']}")
        
        # Show detailed results if requested
        if args.show_details:
            # Load and display detailed results
            try:
                with open(result['output_file']) as f:
                    regime_data = json.load(f)
                
                print(f"\n📋 Detailed Results:")
                print("-" * 60)
                
                for symbol, info in regime_data['symbols'].items():
                    if info.get('regime') not in ['error', 'unknown']:
                        regime_emoji = {"bull": "🐂", "bear": "🐻", "range": "📦"}.get(info['regime'], "❓")
                        
                        print(f"\n{regime_emoji} {symbol} - {info['regime'].upper()} (confidence: {info['confidence']:.1%})")
                        print(f"   Price: ${info.get('price', 0):.4f}")
                        print(f"   SMA 50: ${info.get('sma_50', 0):.4f} (slope: {info.get('sma_50_slope', 0):.6f})")
                        print(f"   SMA 200: ${info.get('sma_200', 0):.4f} (slope: {info.get('sma_200_slope', 0):.6f})")
                        print(f"   ATR Percentile: {info.get('atr_percentile', 0):.1f}")
                        
                        indicators = info.get('indicators', {})
                        print(f"   Indicators:")
                        print(f"     • Price > SMA50: {'✅' if indicators.get('price_above_sma50') else '❌'}")
                        print(f"     • Price > SMA200: {'✅' if indicators.get('price_above_sma200') else '❌'}")
                        print(f"     • SMA50 > SMA200: {'✅' if indicators.get('sma50_above_sma200') else '❌'}")
                        print(f"     • SMA50 Rising: {'✅' if indicators.get('sma50_rising') else '❌'}")
                        print(f"     • SMA200 Rising: {'✅' if indicators.get('sma200_rising') else '❌'}")
                        print(f"     • High Volatility: {'⚡' if indicators.get('high_volatility') else '🔇'}")
                
            except Exception as e:
                print(f"❌ Error loading detailed results: {e}")
        
        # Show regime distribution
        summary = result['summary']
        total_analyzed = summary['bull_count'] + summary['bear_count'] + summary['range_count']
        
        if total_analyzed > 0:
            print(f"\n📊 Market Regime Distribution:")
            print(f"   🐂 Bull: {summary['bull_count']}/{total_analyzed} ({summary['bull_count']/total_analyzed:.1%})")
            print(f"   🐻 Bear: {summary['bear_count']}/{total_analyzed} ({summary['bear_count']/total_analyzed:.1%})")
            print(f"   📦 Range: {summary['range_count']}/{total_analyzed} ({summary['range_count']/total_analyzed:.1%})")
            print(f"   📈 Avg Confidence: {summary['avg_confidence']:.1%}")
            
            # Show symbols by regime
            if summary['bull_symbols']:
                print(f"\n🐂 Bull Markets: {', '.join(summary['bull_symbols'])}")
            if summary['bear_symbols']:
                print(f"🐻 Bear Markets: {', '.join(summary['bear_symbols'])}")
            if summary['range_symbols']:
                print(f"📦 Range Markets: {', '.join(summary['range_symbols'])}")
        
        if summary['insufficient_data_count'] > 0:
            print(f"\n⚠️ {summary['insufficient_data_count']} symbols had insufficient data")
        if summary['error_count'] > 0:
            print(f"❌ {summary['error_count']} symbols had analysis errors")
    else:
        print("❌ Regime detection failed")
        exit(1)

if __name__ == "__main__":
    main()