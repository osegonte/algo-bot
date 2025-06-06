#!/usr/bin/env python3
"""
Level 6-B: Economic Calendar Ingest
Fetches today's economic events and impact levels
"""

import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import csv
import time

class EconomicDataSource:
    """Abstract base for economic calendar sources"""
    
    def fetch_events(self, date: datetime) -> List[Dict]:
        raise NotImplementedError

class FREDSource(EconomicDataSource):
    """Federal Reserve Economic Data (FRED) source"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
    def fetch_events(self, date: datetime) -> List[Dict]:
        """Fetch economic releases from FRED"""
        
        if not self.api_key or self.api_key == "your_fred_api_key":
            print("⚠️ No valid FRED API key - using simulated data")
            return self._get_demo_events(date)
        
        try:
            # FRED releases endpoint
            params = {
                'api_key': self.api_key,
                'file_type': 'json',
                'realtime_start': date.strftime('%Y-%m-%d'),
                'realtime_end': date.strftime('%Y-%m-%d'),
                'limit': 100
            }
            
            response = requests.get(f"{self.base_url}/releases", params=params)
            response.raise_for_status()
            
            data = response.json()
            releases = data.get('releases', [])
            
            events = []
            for release in releases:
                event = {
                    'id': release.get('id'),
                    'name': release.get('name'),
                    'press_release': release.get('press_release', True),
                    'realtime_start': release.get('realtime_start'),
                    'impact': self._determine_impact(release.get('name', '')),
                    'source': 'fred'
                }
                events.append(event)
            
            print(f"✅ Fetched {len(events)} FRED releases")
            return events
            
        except Exception as e:
            print(f"❌ FRED API failed: {e}")
            return self._get_demo_events(date)
    
    def _determine_impact(self, event_name: str) -> str:
        """Determine impact level based on event name"""
        high_impact_keywords = [
            'employment', 'unemployment', 'payroll', 'gdp', 'inflation', 'cpi', 'ppi',
            'federal funds', 'interest rate', 'fomc', 'trade balance', 'retail sales'
        ]
        
        medium_impact_keywords = [
            'housing', 'consumer confidence', 'industrial production', 'capacity utilization',
            'building permits', 'existing home sales', 'personal income'
        ]
        
        name_lower = event_name.lower()
        
        for keyword in high_impact_keywords:
            if keyword in name_lower:
                return 'high'
        
        for keyword in medium_impact_keywords:
            if keyword in name_lower:
                return 'medium'
        
        return 'low'
    
    def _get_demo_events(self, date: datetime) -> List[Dict]:
        """Generate demo economic events"""
        
        # Create realistic economic events for the date
        demo_events = []
        
        # High impact events (fewer but important)
        if date.weekday() == 4:  # Friday - often employment day
            demo_events.append({
                'id': 'demo_nfp',
                'name': 'Nonfarm Payrolls',
                'description': 'Monthly employment report showing job creation',
                'time_utc': date.replace(hour=13, minute=30).isoformat(),
                'impact': 'high',
                'currency': 'USD',
                'actual': None,
                'forecast': '180K',
                'previous': '175K',
                'source': 'fred_demo'
            })
            
            demo_events.append({
                'id': 'demo_unemployment',
                'name': 'Unemployment Rate',
                'description': 'Percentage of labor force that is unemployed',
                'time_utc': date.replace(hour=13, minute=30).isoformat(),
                'impact': 'high',
                'currency': 'USD',
                'actual': None,
                'forecast': '3.9%',
                'previous': '4.0%',
                'source': 'fred_demo'
            })
        
        # Medium impact events
        if date.weekday() in [1, 3]:  # Tuesday, Thursday
            demo_events.append({
                'id': 'demo_cpi',
                'name': 'Consumer Price Index',
                'description': 'Measure of inflation at consumer level',
                'time_utc': date.replace(hour=13, minute=30).isoformat(),
                'impact': 'medium',
                'currency': 'USD',
                'actual': None,
                'forecast': '3.2%',
                'previous': '3.1%',
                'source': 'fred_demo'
            })
        
        # Daily low-impact events
        demo_events.extend([
            {
                'id': f'demo_treasury_{date.strftime("%Y%m%d")}',
                'name': 'Treasury Bill Auction',
                'description': 'Government debt auction',
                'time_utc': date.replace(hour=17, minute=0).isoformat(),
                'impact': 'low',
                'currency': 'USD',
                'actual': None,
                'forecast': None,
                'previous': None,
                'source': 'fred_demo'
            }
        ])
        
        return demo_events

class ForexFactorySource(EconomicDataSource):
    """Forex Factory calendar scraper (simplified simulation)"""
    
    def fetch_events(self, date: datetime) -> List[Dict]:
        """Simulate Forex Factory economic calendar"""
        
        # In production, this would scrape ForexFactory.com
        # For now, we'll simulate realistic forex events
        
        events = []
        
        # Generate events based on day of week
        if date.weekday() == 0:  # Monday
            events.append({
                'id': f'ff_retail_sales_{date.strftime("%Y%m%d")}',
                'name': 'Retail Sales m/m',
                'description': 'Month-over-month change in retail sales',
                'time_utc': date.replace(hour=9, minute=30).isoformat(),
                'impact': 'medium',
                'currency': 'EUR',
                'actual': None,
                'forecast': '0.3%',
                'previous': '0.1%',
                'source': 'forexfactory_demo'
            })
        
        elif date.weekday() == 2:  # Wednesday
            events.append({
                'id': f'ff_fomc_{date.strftime("%Y%m%d")}',
                'name': 'FOMC Meeting Minutes',
                'description': 'Federal Open Market Committee meeting minutes',
                'time_utc': date.replace(hour=19, minute=0).isoformat(),
                'impact': 'high',
                'currency': 'USD',
                'actual': None,
                'forecast': None,
                'previous': None,
                'source': 'forexfactory_demo'
            })
        
        elif date.weekday() == 4:  # Friday
            events.extend([
                {
                    'id': f'ff_gdp_{date.strftime("%Y%m%d")}',
                    'name': 'GDP q/q',
                    'description': 'Quarterly Gross Domestic Product growth',
                    'time_utc': date.replace(hour=12, minute=30).isoformat(),
                    'impact': 'high',
                    'currency': 'GBP',
                    'actual': None,
                    'forecast': '0.4%',
                    'previous': '0.6%',
                    'source': 'forexfactory_demo'
                }
            ])
        
        # Always include some daily events
        events.append({
            'id': f'ff_bond_auction_{date.strftime("%Y%m%d")}',
            'name': '10-Year Bond Auction',
            'description': 'Government bond auction',
            'time_utc': date.replace(hour=16, minute=0).isoformat(),
            'impact': 'low',
            'currency': 'USD',
            'actual': None,
            'forecast': None,
            'previous': None,
            'source': 'forexfactory_demo'
        })
        
        return events

class EconomicCalendarIngest:
    """Main economic calendar data ingester"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.intel_dir = Path("intel")
        self.intel_dir.mkdir(exist_ok=True)
        
        # Initialize data sources
        self.sources = []
        
        # FRED source
        fred_key = self.config.get("fred", {}).get("api_key")
        if fred_key:
            self.sources.append(FREDSource(fred_key))
        
        # ForexFactory source (always available as demo)
        self.sources.append(ForexFactorySource())
        
        if not self.sources:
            print("⚠️ No economic data sources configured")
    
    def ingest_todays_events(self, target_date: datetime = None) -> Dict:
        """Ingest economic events for today (or specified date)"""
        
        if target_date is None:
            target_date = datetime.now(timezone.utc).date()
        
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        
        print(f"📅 Fetching economic events for {target_date}")
        
        all_events = []
        source_stats = {}
        
        # Fetch from all sources
        for source in self.sources:
            try:
                source_name = source.__class__.__name__
                print(f"🔄 Fetching from {source_name}...")
                
                events = source.fetch_events(datetime.combine(target_date, datetime.min.time().replace(tzinfo=timezone.utc)))
                
                # Add source metadata
                for event in events:
                    event['source_type'] = source_name
                    event['ingested_at'] = datetime.now(timezone.utc).isoformat()
                
                all_events.extend(events)
                source_stats[source_name] = len(events)
                
                print(f"✅ {source_name}: {len(events)} events")
                
            except Exception as e:
                print(f"❌ {source.__class__.__name__} failed: {e}")
                source_stats[source.__class__.__name__] = 0
        
        # Sort events by time
        all_events.sort(key=lambda e: e.get('time_utc', ''))
        
        # Save to JSON
        calendar_file = self.intel_dir / "econ_calendar.json"
        
        calendar_data = {
            'date': target_date.isoformat(),
            'events': all_events,
            'metadata': {
                'total_events': len(all_events),
                'sources_used': source_stats,
                'ingested_at': datetime.now(timezone.utc).isoformat(),
                'impact_breakdown': self._get_impact_breakdown(all_events)
            }
        }
        
        with open(calendar_file, 'w') as f:
            json.dump(calendar_data, f, indent=2)
        
        # Also save as CSV for easy analysis
        csv_file = self.intel_dir / "econ_calendar.csv"
        self._save_events_to_csv(all_events, csv_file)
        
        # Generate summary
        summary = self._generate_summary(all_events)
        
        print(f"✅ Ingested {len(all_events)} economic events")
        print(f"💾 Saved to: {calendar_file}")
        print(f"📊 High impact: {summary['high_impact_count']}")
        print(f"⚠️ Medium impact: {summary['medium_impact_count']}")
        print(f"ℹ️ Low impact: {summary['low_impact_count']}")
        
        return {
            'success': True,
            'events_count': len(all_events),
            'output_file': str(calendar_file),
            'csv_file': str(csv_file),
            'summary': summary,
            'source_stats': source_stats
        }
    
    def _get_impact_breakdown(self, events: List[Dict]) -> Dict[str, int]:
        """Get breakdown of events by impact level"""
        breakdown = {'high': 0, 'medium': 0, 'low': 0}
        
        for event in events:
            impact = event.get('impact', 'low')
            if impact in breakdown:
                breakdown[impact] += 1
        
        return breakdown
    
    def _save_events_to_csv(self, events: List[Dict], csv_file: Path):
        """Save events to CSV file"""
        
        if not events:
            return
        
        fieldnames = [
            'id', 'name', 'description', 'time_utc', 'impact', 'currency',
            'actual', 'forecast', 'previous', 'source', 'source_type', 'ingested_at'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                # Ensure all fields exist
                row = {field: event.get(field, '') for field in fieldnames}
                writer.writerow(row)
    
    def _generate_summary(self, events: List[Dict]) -> Dict:
        """Generate summary of economic events"""
        
        if not events:
            return {
                'total_events': 0,
                'high_impact_count': 0,
                'medium_impact_count': 0,
                'low_impact_count': 0,
                'currencies': [],
                'next_high_impact': None
            }
        
        # Count by impact
        impact_counts = {'high': 0, 'medium': 0, 'low': 0}
        currencies = set()
        
        for event in events:
            impact = event.get('impact', 'low')
            if impact in impact_counts:
                impact_counts[impact] += 1
            
            currency = event.get('currency')
            if currency:
                currencies.add(currency)
        
        # Find next high impact event
        high_impact_events = [e for e in events if e.get('impact') == 'high']
        next_high_impact = None
        
        if high_impact_events:
            # Sort by time to find next one
            now = datetime.now(timezone.utc).isoformat()
            future_events = [e for e in high_impact_events if e.get('time_utc', '') > now]
            
            if future_events:
                future_events.sort(key=lambda e: e.get('time_utc', ''))
                next_high_impact = {
                    'name': future_events[0].get('name'),
                    'time_utc': future_events[0].get('time_utc'),
                    'currency': future_events[0].get('currency')
                }
        
        return {
            'total_events': len(events),
            'high_impact_count': impact_counts['high'],
            'medium_impact_count': impact_counts['medium'],
            'low_impact_count': impact_counts['low'],
            'currencies': sorted(list(currencies)),
            'next_high_impact': next_high_impact
        }
    
    def get_events_for_timeframe(self, hours_ahead: int = 24) -> List[Dict]:
        """Get events occurring within the next N hours"""
        
        calendar_file = self.intel_dir / "econ_calendar.json"
        
        if not calendar_file.exists():
            return []
        
        try:
            with open(calendar_file) as f:
                data = json.load(f)
            
            events = data.get('events', [])
            now = datetime.now(timezone.utc)
            cutoff_time = now + timedelta(hours=hours_ahead)
            
            upcoming_events = []
            
            for event in events:
                event_time_str = event.get('time_utc', '')
                if event_time_str:
                    try:
                        event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                        if now <= event_time <= cutoff_time:
                            upcoming_events.append(event)
                    except ValueError:
                        continue
            
            return upcoming_events
            
        except Exception as e:
            print(f"❌ Error reading calendar file: {e}")
            return []

def load_config() -> Dict:
    """Load configuration from file"""
    config_file = Path("config/base_config.yaml")
    
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f)
    
    return {
        'fred': {
            'api_key': 'your_fred_api_key'  # Get from https://fred.stlouisfed.org/docs/api/api_key.html
        }
    }

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Economic Calendar Ingest (Level 6-B)")
    parser.add_argument('--date', help='Date to fetch (YYYY-MM-DD), default: today')
    parser.add_argument('--upcoming', type=int, help='Show events in next N hours')
    parser.add_argument('--test', action='store_true', help='Run with demo data only')
    
    args = parser.parse_args()
    
    print("📅 Economic Calendar Ingest (Level 6-B)")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    if args.test:
        print("🧪 Test mode: Using demo data only")
        config['fred'] = {'api_key': None}  # Force demo mode
    
    # Initialize ingester
    ingester = EconomicCalendarIngest(config)
    
    # Parse target date
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
            exit(1)
    
    if args.upcoming:
        # Show upcoming events
        print(f"🔍 Looking for events in next {args.upcoming} hours...")
        
        events = ingester.get_events_for_timeframe(args.upcoming)
        
        if events:
            print(f"📋 Found {len(events)} upcoming events:")
            for event in events:
                impact_emoji = "🔴" if event.get('impact') == 'high' else "🟡" if event.get('impact') == 'medium' else "🟢"
                print(f"   {impact_emoji} {event.get('time_utc', '')[:16]} - {event.get('name', 'Unknown')} ({event.get('currency', 'N/A')})")
        else:
            print("📭 No upcoming events found")
        
        return
    
    # Run ingestion
    result = ingester.ingest_todays_events(target_date)
    
    if result['success']:
        print(f"\n✅ Level 6-B Complete!")
        print(f"📊 Events ingested: {result['events_count']}")
        print(f"💾 JSON output: {result['output_file']}")
        print(f"📊 CSV output: {result['csv_file']}")
        
        # Show source breakdown
        print(f"\n📡 Source breakdown:")
        for source, count in result['source_stats'].items():
            print(f"   {source}: {count} events")
        
        # Show next high impact event
        next_high = result['summary'].get('next_high_impact')
        if next_high:
            print(f"\n🔴 Next high impact: {next_high['name']} at {next_high['time_utc'][:16]} ({next_high['currency']})")
        else:
            print(f"\n🟢 No high impact events scheduled")
    else:
        print("❌ Ingestion failed")
        exit(1)

if __name__ == "__main__":
    main()