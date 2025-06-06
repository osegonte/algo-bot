#!/usr/bin/env python3
"""
Level 5-E: Historical Backfill Tool
Downloads historical market data for backtesting and analysis
"""

import os
import json
import csv
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import time
from typing import Dict, List, Optional
import pandas as pd

class HistoricalDataDownloader:
    """Download and store historical market data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path("data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints and keys
        self.alpaca_config = config.get("alpaca", {})
        self.iex_key = config.get("iex", {}).get("api_key")
        self.av_key = config.get("alphavantage", {}).get("api_key")
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _rate_limit(self, requests_per_minute: int = 5):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 60 / requests_per_minute:
            sleep_time = (60 / requests_per_minute) - time_since_last
            print(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def download_alpaca_stock_data(self, symbol: str, days: int = 30) -> bool:
        """Download stock data from Alpaca"""
        try:
            print(f"📈 Downloading {symbol} data from Alpaca ({days} days)...")
            
            # Calculate date range
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=days)
            
            # Alpaca API headers
            headers = {
                'APCA-API-KEY-ID': self.alpaca_config.get('api_key', ''),
                'APCA-API-SECRET-KEY': self.alpaca_config.get('api_secret', ''),
            }
            
            # Get 1-minute bars
            base_url = self.alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
            url = f"{base_url}/v2/stocks/{symbol}/bars"
            
            params = {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'timeframe': '1Min',
                'adjustment': 'all',
                'feed': 'iex',
                'limit': 10000
            }
            
            self._rate_limit()
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            bars = data.get('bars', [])
            
            if not bars:
                print(f"⚠️ No data received for {symbol}")
                return False
            
            # Save raw JSON
            output_file = self.data_dir / f"{symbol}_1min_{days}d_alpaca.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'timeframe': '1Min',
                    'days': days,
                    'source': 'alpaca',
                    'downloaded_at': datetime.now(timezone.utc).isoformat(),
                    'bars': bars
                }, f, indent=2)
            
            # Also save as CSV for easy analysis
            csv_file = self.data_dir / f"{symbol}_1min_{days}d_alpaca.csv"
            df = pd.DataFrame(bars)
            df.to_csv(csv_file, index=False)
            
            print(f"✅ Saved {len(bars)} bars to {output_file}")
            print(f"📊 CSV version: {csv_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {symbol} from Alpaca: {e}")
            return False
    
    def download_forex_data_simulation(self, pair: str = "EURUSD", days: int = 30) -> bool:
        """Simulate forex data download (placeholder for real forex API)"""
        try:
            print(f"💱 Generating simulated {pair} data ({days} days)...")
            
            # Generate realistic EURUSD data
            import random
            import numpy as np
            
            # Start from current time, go back
            current_time = datetime.now(timezone.utc)
            start_time = current_time - timedelta(days=days)
            
            bars = []
            base_rate = 1.0500 if pair == "EURUSD" else 1.2000
            current_rate = base_rate
            
            # Generate 1-minute bars
            minutes_in_period = days * 24 * 60
            
            for i in range(minutes_in_period):
                timestamp = start_time + timedelta(minutes=i)
                
                # Skip weekends (forex markets closed)
                if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    continue
                
                # Add some realistic price movement
                change = np.random.normal(0, 0.0002)  # Small random walk
                current_rate += change
                
                # Keep within reasonable bounds
                current_rate = max(current_rate, base_rate * 0.95)
                current_rate = min(current_rate, base_rate * 1.05)
                
                # Create OHLC bar
                spread = 0.0002
                open_price = current_rate
                high_price = current_rate + random.uniform(0, 0.0005)
                low_price = current_rate - random.uniform(0, 0.0005)
                close_price = current_rate + np.random.normal(0, 0.0001)
                
                bar = {
                    'timestamp': timestamp.isoformat(),
                    'open': round(open_price, 5),
                    'high': round(high_price, 5),
                    'low': round(low_price, 5),
                    'close': round(close_price, 5),
                    'volume': random.randint(100, 1000),
                    'spread': spread
                }
                
                bars.append(bar)
                current_rate = close_price
            
            # Save data
            output_file = self.data_dir / f"{pair}_1min_{days}d_forex_sim.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'symbol': pair,
                    'timeframe': '1Min',
                    'days': days,
                    'source': 'forex_simulation',
                    'downloaded_at': datetime.now(timezone.utc).isoformat(),
                    'bars': bars
                }, f, indent=2)
            
            # CSV version
            csv_file = self.data_dir / f"{pair}_1min_{days}d_forex_sim.csv"
            df = pd.DataFrame(bars)
            df.to_csv(csv_file, index=False)
            
            print(f"✅ Generated {len(bars)} {pair} bars")
            print(f"📊 Saved to {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate {pair} data: {e}")
            return False
    
    def download_iex_stock_data(self, symbol: str, days: int = 30) -> bool:
        """Download stock data from IEX Cloud"""
        if not self.iex_key:
            print("⚠️ No IEX API key configured")
            return False
        
        try:
            print(f"📈 Downloading {symbol} from IEX ({days} days)...")
            
            # IEX historical data endpoint
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/{days}d"
            params = {
                'token': self.iex_key,
                'chartInterval': 1,  # 1-minute intervals
                'includeToday': True
            }
            
            self._rate_limit()
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                print(f"⚠️ No data received for {symbol}")
                return False
            
            # Save raw data
            output_file = self.data_dir / f"{symbol}_1min_{days}d_iex.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'timeframe': '1Min',
                    'days': days,
                    'source': 'iex',
                    'downloaded_at': datetime.now(timezone.utc).isoformat(),
                    'bars': data
                }, f, indent=2)
            
            # CSV version
            csv_file = self.data_dir / f"{symbol}_1min_{days}d_iex.csv"
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            
            print(f"✅ Saved {len(data)} bars from IEX")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {symbol} from IEX: {e}")
            return False
    
    def download_all_symbols(self, symbols: List[str], days: int = 30) -> Dict[str, bool]:
        """Download data for multiple symbols"""
        results = {}
        
        print(f"🚀 Starting bulk download for {len(symbols)} symbols")
        print("=" * 50)
        
        for symbol in symbols:
            print(f"\n📊 Processing {symbol}...")
            
            if symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
                # Forex pairs
                success = self.download_forex_data_simulation(symbol, days)
            else:
                # Stock symbols - try Alpaca first
                success = self.download_alpaca_stock_data(symbol, days)
                
                # Fallback to IEX if Alpaca fails
                if not success and self.iex_key:
                    print(f"🔄 Trying IEX fallback for {symbol}...")
                    success = self.download_iex_stock_data(symbol, days)
            
            results[symbol] = success
            
            # Brief pause between symbols
            time.sleep(1)
        
        return results
    
    def verify_downloads(self, symbols: List[str], days: int = 30) -> Dict[str, Dict]:
        """Verify downloaded data integrity"""
        verification_results = {}
        
        print("\n🔍 Verifying downloaded data...")
        print("=" * 40)
        
        for symbol in symbols:
            result = {
                'files_found': [],
                'total_records': 0,
                'date_range': None,
                'data_quality': 'unknown'
            }
            
            # Look for files matching this symbol
            pattern = f"{symbol}_1min_{days}d_*.json"
            matching_files = list(self.data_dir.glob(pattern))
            
            for file_path in matching_files:
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    
                    bars = data.get('bars', [])
                    result['files_found'].append(file_path.name)
                    result['total_records'] += len(bars)
                    
                    # Check date range
                    if bars:
                        timestamps = [bar.get('timestamp', bar.get('t', '')) for bar in bars[:10]]
                        if timestamps and timestamps[0]:
                            result['date_range'] = f"starts: {timestamps[0][:10]}"
                    
                    # Basic quality check
                    if len(bars) > 100:
                        result['data_quality'] = 'good'
                    elif len(bars) > 10:
                        result['data_quality'] = 'limited'
                    else:
                        result['data_quality'] = 'poor'
                
                except Exception as e:
                    result['error'] = str(e)
            
            verification_results[symbol] = result
            
            # Display result
            status = "✅" if result['files_found'] else "❌"
            print(f"{status} {symbol}: {result['total_records']} records, {result['data_quality']} quality")
        
        return verification_results
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary of all downloaded data"""
        summary = {
            'total_files': 0,
            'symbols': set(),
            'sources': set(),
            'total_size_mb': 0,
            'files_by_type': {}
        }
        
        if not self.data_dir.exists():
            return summary
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                summary['total_files'] += 1
                
                # File size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                summary['total_size_mb'] += size_mb
                
                # Parse filename for metadata
                name_parts = file_path.stem.split('_')
                if len(name_parts) >= 3:
                    symbol = name_parts[0]
                    summary['symbols'].add(symbol)
                
                # File type
                extension = file_path.suffix
                summary['files_by_type'][extension] = summary['files_by_type'].get(extension, 0) + 1
                
                # Check source if JSON
                if extension == '.json':
                    try:
                        with open(file_path) as f:
                            data = json.load(f)
                        source = data.get('source', 'unknown')
                        summary['sources'].add(source)
                    except:
                        pass
        
        # Convert sets to lists for JSON serialization
        summary['symbols'] = sorted(list(summary['symbols']))
        summary['sources'] = list(summary['sources'])
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        
        return summary

def main():
    """Command line interface for historical data download"""
    
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'EURUSD'], 
                       help='Symbols to download (default: AAPL EURUSD)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to download (default: 30)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded data')
    parser.add_argument('--summary', action='store_true',
                       help='Show data summary')
    parser.add_argument('--config', default='config/base_config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        print(f"⚠️ Config file not found: {args.config}")
        print("Using default Alpaca config from environment")
        config = {
            'alpaca': {
                'api_key': 'PKKJRF8U4QTYWMAVLYUI',
                'api_secret': '4kGChcW8cqkxasJfvyVMYfGRKbPqwNcx8MzM26ws',
                'base_url': 'https://paper-api.alpaca.markets'
            }
        }
    
    downloader = HistoricalDataDownloader(config)
    
    if args.summary:
        # Show data summary
        summary = downloader.get_data_summary()
        print("\n📊 Data Summary:")
        print("=" * 30)
        print(f"Files: {summary['total_files']}")
        print(f"Symbols: {', '.join(summary['symbols'])}")
        print(f"Sources: {', '.join(summary['sources'])}")
        print(f"Total size: {summary['total_size_mb']} MB")
        print(f"File types: {summary['files_by_type']}")
        return
    
    if args.verify:
        # Verify existing downloads
        verification = downloader.verify_downloads(args.symbols, args.days)
        return
    
    # Download data
    print(f"🚀 Historical Data Backfill Tool (Level 5-E)")
    print(f"Symbols: {args.symbols}")
    print(f"Days: {args.days}")
    print("=" * 50)
    
    results = downloader.download_all_symbols(args.symbols, args.days)
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n📊 Download Complete:")
    print(f"✅ Successful: {successful}/{total}")
    
    if successful < total:
        print("❌ Failed symbols:")
        for symbol, success in results.items():
            if not success:
                print(f"   - {symbol}")
    
    # Verify what we downloaded
    verification = downloader.verify_downloads(args.symbols, args.days)
    
    print(f"\n💾 Data saved to: {downloader.data_dir}")
    print("🔍 Use --verify flag to check data integrity")
    print("📊 Use --summary flag to see overall data status")

if __name__ == "__main__":
    main()