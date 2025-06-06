#!/usr/bin/env python3
"""
Level 6-F: Enhanced Parent Controller with Intelligence Awareness
Extends parent controller to load and display intelligence data
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Import the existing parent controller
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.enhanced_parent_controller import EnhancedParentController
except ImportError:
    # Fallback to basic parent controller
    from core.parent_controller import ParentController as EnhancedParentController

class IntelligenceAwareParentController(EnhancedParentController):
    """Enhanced parent controller that integrates intelligence data"""
    
    def __init__(self):
        super().__init__()
        
        # Intelligence data storage
        self.intel_dir = Path("intel")
        self.news_sentiment_df: Optional[pd.DataFrame] = None
        self.economic_calendar_data: Optional[Dict] = None
        self.market_regime_data: Optional[Dict] = None
        
        # Intelligence load status
        self.intel_loaded = False
        self.intel_load_errors = []
    
    def load_intelligence_data(self) -> Dict[str, bool]:
        """Load all available intelligence data"""
        
        print("🧠 Loading intelligence data...")
        
        load_status = {
            'news_sentiment': False,
            'economic_calendar': False,
            'market_regime': False
        }
        
        self.intel_load_errors = []
        
        # Load news sentiment data
        try:
            news_file = self.intel_dir / "news_sentiment.csv"
            if news_file.exists():
                self.news_sentiment_df = pd.read_csv(news_file)
                load_status['news_sentiment'] = True
                print(f"✅ Loaded {len(self.news_sentiment_df)} news headlines")
            else:
                self.intel_load_errors.append("news_sentiment.csv not found")
        except Exception as e:
            self.intel_load_errors.append(f"News sentiment load error: {e}")
        
        # Load economic calendar data
        try:
            econ_file = self.intel_dir / "econ_calendar.json"
            if econ_file.exists():
                with open(econ_file) as f:
                    self.economic_calendar_data = json.load(f)
                load_status['economic_calendar'] = True
                event_count = len(self.economic_calendar_data.get('events', []))
                print(f"✅ Loaded {event_count} economic events")
            else:
                self.intel_load_errors.append("econ_calendar.json not found")
        except Exception as e:
            self.intel_load_errors.append(f"Economic calendar load error: {e}")
        
        # Load market regime data
        try:
            regime_file = self.intel_dir / "market_regime.json"
            if regime_file.exists():
                with open(regime_file) as f:
                    self.market_regime_data = json.load(f)
                load_status['market_regime'] = True
                symbol_count = len(self.market_regime_data.get('symbols', {}))
                print(f"✅ Loaded regime analysis for {symbol_count} symbols")
            else:
                self.intel_load_errors.append("market_regime.json not found")
        except Exception as e:
            self.intel_load_errors.append(f"Market regime load error: {e}")
        
        # Set overall load status
        self.intel_loaded = any(load_status.values())
        
        if self.intel_load_errors:
            print(f"⚠️ Intel load errors: {len(self.intel_load_errors)}")
            for error in self.intel_load_errors:
                print(f"   • {error}")
        
        return load_status
    
    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """Get summary of market sentiment from news"""
        
        if self.news_sentiment_df is None or len(self.news_sentiment_df) == 0:
            return {'available': False, 'reason': 'no_data'}
        
        try:
            # Calculate sentiment metrics
            sentiment_scores = self.news_sentiment_df['sentiment_compound'].astype(float)
            
            positive_count = (sentiment_scores > 0.1).sum()
            negative_count = (sentiment_scores < -0.1).sum()
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            # Find most positive headline
            most_positive_idx = sentiment_scores.idxmax()
            most_positive = self.news_sentiment_df.iloc[most_positive_idx]
            
            # Find most negative headline
            most_negative_idx = sentiment_scores.idxmin()
            most_negative = self.news_sentiment_df.iloc[most_negative_idx]
            
            return {
                'available': True,
                'total_headlines': len(sentiment_scores),
                'avg_sentiment': float(sentiment_scores.mean()),
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'neutral_count': int(neutral_count),
                'most_positive': {
                    'title': str(most_positive['title']),
                    'score': float(most_positive['sentiment_compound']),
                    'source': str(most_positive.get('source', 'Unknown'))
                },
                'most_negative': {
                    'title': str(most_negative['title']),
                    'score': float(most_negative['sentiment_compound']),
                    'source': str(most_negative.get('source', 'Unknown'))
                }
            }
            
        except Exception as e:
            return {'available': False, 'reason': f'analysis_error: {e}'}
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of market regimes"""
        
        if not self.market_regime_data or 'symbols' not in self.market_regime_data:
            return {'available': False, 'reason': 'no_data'}
        
        try:
            symbols = self.market_regime_data['symbols']
            
            # Count regimes
            regime_counts = {'bull': 0, 'bear': 0, 'range': 0, 'other': 0}
            bull_symbols = []
            bear_symbols = []
            range_symbols = []
            
            total_confidence = 0
            confidence_count = 0
            
            for symbol, info in symbols.items():
                regime = info.get('regime', 'unknown')
                confidence = info.get('confidence', 0)
                
                if regime == 'bull':
                    regime_counts['bull'] += 1
                    bull_symbols.append(symbol)
                elif regime == 'bear':
                    regime_counts['bear'] += 1
                    bear_symbols.append(symbol)
                elif regime == 'range':
                    regime_counts['range'] += 1
                    range_symbols.append(symbol)
                else:
                    regime_counts['other'] += 1
                
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1
            
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            
            return {
                'available': True,
                'total_symbols': len(symbols),
                'bull_count': regime_counts['bull'],
                'bear_count': regime_counts['bear'],
                'range_count': regime_counts['range'],
                'other_count': regime_counts['other'],
                'avg_confidence': avg_confidence,
                'bull_symbols': bull_symbols,
                'bear_symbols': bear_symbols,
                'range_symbols': range_symbols,
                'analysis_date': self.market_regime_data.get('detection_date', 'unknown')
            }
            
        except Exception as e:
            return {'available': False, 'reason': f'analysis_error: {e}'}
    
    def get_upcoming_economic_events(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get upcoming high-impact economic events"""
        
        if not self.economic_calendar_data or 'events' not in self.economic_calendar_data:
            return {'available': False, 'reason': 'no_data'}
        
        try:
            events = self.economic_calendar_data['events']
            now = datetime.now(timezone.utc)
            
            upcoming_events = []
            high_impact_events = []
            
            for event in events:
                event_time_str = event.get('time_utc', '')
                if not event_time_str:
                    continue
                
                try:
                    event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                    hours_until = (event_time - now).total_seconds() / 3600
                    
                    if 0 <= hours_until <= hours_ahead:
                        event_info = {
                            'name': event.get('name', 'Unknown'),
                            'time_utc': event_time_str,
                            'hours_until': round(hours_until, 1),
                            'impact': event.get('impact', 'low'),
                            'currency': event.get('currency', 'Unknown')
                        }
                        
                        upcoming_events.append(event_info)
                        
                        if event.get('impact') == 'high':
                            high_impact_events.append(event_info)
                
                except ValueError:
                    continue
            
            # Sort by time
            upcoming_events.sort(key=lambda e: e['hours_until'])
            high_impact_events.sort(key=lambda e: e['hours_until'])
            
            return {
                'available': True,
                'total_upcoming': len(upcoming_events),
                'high_impact_count': len(high_impact_events),
                'upcoming_events': upcoming_events[:5],  # Limit to 5 for display
                'high_impact_events': high_impact_events,
                'calendar_date': self.economic_calendar_data.get('date', 'unknown')
            }
            
        except Exception as e:
            return {'available': False, 'reason': f'analysis_error: {e}'}
    
    def display_intelligence_summary(self):
        """Display comprehensive intelligence summary"""
        
        if not self.intel_loaded:
            print("\n📊 INTELLIGENCE SUMMARY")
            print("=" * 50)
            print("❌ No intelligence data loaded")
            
            if self.intel_load_errors:
                print("\n🔧 Load errors:")
                for error in self.intel_load_errors:
                    print(f"   • {error}")
            
            return
        
        print("\n📊 INTELLIGENCE SUMMARY")
        print("=" * 50)
        
        # Market Sentiment
        sentiment_summary = self.get_market_sentiment_summary()
        if sentiment_summary['available']:
            print(f"\n📰 Market Sentiment:")
            print(f"   Headlines analyzed: {sentiment_summary['total_headlines']}")
            print(f"   Average sentiment: {sentiment_summary['avg_sentiment']:.3f}")
            print(f"   🟢 Positive: {sentiment_summary['positive_count']}")
            print(f"   🔴 Negative: {sentiment_summary['negative_count']}")
            print(f"   ⚪ Neutral: {sentiment_summary['neutral_count']}")
            
            most_positive = sentiment_summary['most_positive']
            print(f"\n📈 Most positive headline:")
            print(f"   \"{most_positive['title'][:60]}...\"")
            print(f"   Score: {most_positive['score']:.3f} | Source: {most_positive['source']}")
        else:
            print(f"\n📰 Market Sentiment: ❌ {sentiment_summary['reason']}")
        
        # Market Regimes
        regime_summary = self.get_regime_summary()
        if regime_summary['available']:
            print(f"\n📊 Market Regimes:")
            print(f"   Symbols analyzed: {regime_summary['total_symbols']}")
            print(f"   🐂 Bull markets: {regime_summary['bull_count']}")
            print(f"   🐻 Bear markets: {regime_summary['bear_count']}")
            print(f"   📦 Range markets: {regime_summary['range_count']}")
            print(f"   📈 Avg confidence: {regime_summary['avg_confidence']:.1%}")
            
            if regime_summary['bull_symbols']:
                print(f"   Bull symbols: {', '.join(regime_summary['bull_symbols'])}")
            if regime_summary['bear_symbols']:
                print(f"   Bear symbols: {', '.join(regime_summary['bear_symbols'])}")
        else:
            print(f"\n📊 Market Regimes: ❌ {regime_summary['reason']}")
        
        # Economic Events
        econ_summary = self.get_upcoming_economic_events()
        if econ_summary['available']:
            print(f"\n📅 Upcoming Economic Events (24h):")
            print(f"   Total events: {econ_summary['total_upcoming']}")
            print(f"   🔴 High impact: {econ_summary['high_impact_count']}")
            
            if econ_summary['high_impact_events']:
                print(f"\n   Next high impact events:")
                for event in econ_summary['high_impact_events'][:3]:
                    print(f"   • {event['name']} ({event['currency']}) in {event['hours_until']:.1f}h")
        else:
            print(f"\n📅 Economic Events: ❌ {econ_summary['reason']}")
    
    def enhanced_kpis_with_intelligence(self):
        """Enhanced KPI display that includes intelligence context"""
        
        # Run original KPI calculation
        kpis = self.kpis()
        
        if not kpis:
            return kpis
        
        # Add intelligence context if available
        if self.intel_loaded:
            print(f"\n🧠 INTELLIGENCE CONTEXT")
            print("=" * 50)
            
            # Get market sentiment context
            sentiment = self.get_market_sentiment_summary()
            if sentiment['available']:
                sentiment_emoji = "📈" if sentiment['avg_sentiment'] > 0.1 else "📉" if sentiment['avg_sentiment'] < -0.1 else "➡️"
                print(f"{sentiment_emoji} Market Sentiment: {sentiment['avg_sentiment']:.3f} ({sentiment['positive_count']} pos, {sentiment['negative_count']} neg)")
            
            # Get regime context
            regimes = self.get_regime_summary()
            if regimes['available']:
                print(f"🐂 Bull/Bear/Range: {regimes['bull_count']}/{regimes['bear_count']}/{regimes['range_count']}")
            
            # Get economic events context
            events = self.get_upcoming_economic_events()
            if events['available'] and events['high_impact_count'] > 0:
                print(f"⚠️ {events['high_impact_count']} high-impact economic events in next 24h")
        
        return kpis
    
    def run_enhanced_analysis(self):
        """Run complete enhanced analysis with intelligence"""
        
        print("🧠 Enhanced Parent Controller with Intelligence (Level 6-F)")
        print("=" * 60)
        
        # Load intelligence data first
        intel_status = self.load_intelligence_data()
        
        # Run original parent analysis
        self.ingest_logs()
        self.basic_stats()
        
        # Enhanced KPIs with intelligence context
        kpis = self.enhanced_kpis_with_intelligence()
        
        # Display comprehensive intelligence summary
        self.display_intelligence_summary()
        
        # Summary status
        print(f"\n✅ ANALYSIS COMPLETE")
        print("=" * 30)
        
        intel_loaded_count = sum(intel_status.values())
        print(f"📊 Trading data: {'✅' if kpis else '❌'}")
        print(f"🧠 Intelligence: {intel_loaded_count}/3 sources loaded")
        
        if intel_loaded_count == 3:
            print("🎉 Level 6-F Complete: Full intelligence integration!")
        elif intel_loaded_count > 0:
            print(f"⚠️ Partial intelligence available ({intel_loaded_count}/3)")
        else:
            print("❌ No intelligence data available")
        
        return {
            'trading_analysis': kpis is not None,
            'intelligence_loaded': intel_loaded_count,
            'intelligence_sources': intel_status
        }

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Parent Controller with Intelligence (Level 6-F)")
    parser.add_argument('--intelligence-only', action='store_true', help='Show only intelligence summary')
    parser.add_argument('--no-trading', action='store_true', help='Skip trading analysis')
    
    args = parser.parse_args()
    
    # Create enhanced parent controller
    pc = IntelligenceAwareParentController()
    
    if args.intelligence_only:
        # Load and display only intelligence
        pc.load_intelligence_data()
        pc.display_intelligence_summary()
    elif args.no_trading:
        # Load intelligence without trading analysis
        pc.load_intelligence_data()
        pc.display_intelligence_summary()
    else:
        # Full enhanced analysis
        result = pc.run_enhanced_analysis()
        
        if result['intelligence_loaded'] == 3:
            print(f"\n🎯 All Level 6 components integrated successfully!")

if __name__ == "__main__":
    main()