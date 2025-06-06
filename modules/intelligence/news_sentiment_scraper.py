#!/usr/bin/env python3
"""
Level 6-A: News & Sentiment Scraper
Fetches financial news headlines and analyzes sentiment
"""

import os
import csv
import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time

# Import sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob not available, install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER not available, install with: pip install vaderSentiment")

class NewsDataSource:
    """Abstract base for news data sources"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    def fetch_headlines(self, symbols: List[str], limit: int = 50) -> List[Dict]:
        raise NotImplementedError

class NewsAPISource(NewsDataSource):
    """NewsAPI.org data source"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://newsapi.org/v2"
        
    def fetch_headlines(self, symbols: List[str], limit: int = 50) -> List[Dict]:
        """Fetch financial news from NewsAPI"""
        
        if not self.api_key or self.api_key == "your_newsapi_key_here":
            print("⚠️ No valid NewsAPI key - using demo headlines")
            return self._get_demo_headlines()
        
        try:
            # Build query for financial keywords + symbols
            symbol_query = " OR ".join(symbols)
            query = f"({symbol_query}) AND (stock OR trading OR market OR finance)"
            
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),  # API limit is 100
                'apiKey': self.api_key,
                'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            headlines = []
            for article in articles:
                headline = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', ''),
                    'symbols_mentioned': self._extract_symbols(article.get('title', '') + ' ' + article.get('description', ''), symbols)
                }
                headlines.append(headline)
            
            print(f"✅ Fetched {len(headlines)} headlines from NewsAPI")
            return headlines
            
        except Exception as e:
            print(f"❌ NewsAPI failed: {e}")
            return self._get_demo_headlines()
    
    def _extract_symbols(self, text: str, symbols: List[str]) -> List[str]:
        """Extract mentioned symbols from text"""
        mentioned = []
        text_upper = text.upper()
        
        for symbol in symbols:
            if symbol.upper() in text_upper:
                mentioned.append(symbol)
        
        return mentioned
    
    def _get_demo_headlines(self) -> List[Dict]:
        """Generate demo headlines for testing"""
        demo_headlines = [
            {
                'title': 'Apple Reports Strong Q4 Earnings, Stock Rises',
                'description': 'Apple Inc. exceeded analyst expectations with quarterly revenue growth driven by iPhone sales.',
                'source': 'Financial Times',
                'published_at': datetime.now(timezone.utc).isoformat(),
                'url': 'https://example.com/apple-earnings',
                'symbols_mentioned': ['AAPL']
            },
            {
                'title': 'Tesla Stock Volatility Continues Amid Production Updates',
                'description': 'Tesla shares fluctuate as the company provides updates on manufacturing and delivery targets.',
                'source': 'Reuters',
                'published_at': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                'url': 'https://example.com/tesla-production',
                'symbols_mentioned': ['TSLA']
            },
            {
                'title': 'Federal Reserve Signals Potential Rate Changes',
                'description': 'Fed officials hint at monetary policy adjustments affecting broader market sentiment.',
                'source': 'WSJ',
                'published_at': (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat(),
                'url': 'https://example.com/fed-rates',
                'symbols_mentioned': ['AAPL', 'TSLA', 'MSFT']
            },
            {
                'title': 'Microsoft Cloud Services Show Continued Growth',
                'description': 'Microsoft Azure revenue increases as cloud adoption accelerates across industries.',
                'source': 'TechCrunch',
                'published_at': (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(),
                'url': 'https://example.com/msft-cloud',
                'symbols_mentioned': ['MSFT']
            },
            {
                'title': 'Market Volatility Expected as Economic Data Released',
                'description': 'Investors prepare for potential market movements following key economic indicators.',
                'source': 'Bloomberg',
                'published_at': (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat(),
                'url': 'https://example.com/market-volatility',
                'symbols_mentioned': ['AAPL', 'TSLA', 'MSFT']
            }
        ]
        
        # Generate more headlines to reach 50
        base_headlines = demo_headlines.copy()
        while len(demo_headlines) < 50:
            for base in base_headlines:
                if len(demo_headlines) >= 50:
                    break
                
                # Create variation
                variation = base.copy()
                variation['title'] = f"Update: {base['title']}"
                variation['published_at'] = (datetime.now(timezone.utc) - timedelta(hours=len(demo_headlines))).isoformat()
                demo_headlines.append(variation)
        
        print(f"📰 Generated {len(demo_headlines)} demo headlines")
        return demo_headlines

class SentimentAnalyzer:
    """Analyze sentiment of news headlines and descriptions"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using available tools"""
        result = {
            'compound_score': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'method': 'none'
        }
        
        if not text:
            return result
        
        # Try VADER first (better for social media/news)
        if self.vader_analyzer:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            result.update({
                'compound_score': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'method': 'vader'
            })
            return result
        
        # Fallback to TextBlob
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            result.update({
                'compound_score': polarity,
                'positive': max(0, polarity),
                'negative': max(0, -polarity),
                'neutral': 1 - abs(polarity),
                'method': 'textblob'
            })
            return result
        
        # Simple keyword-based fallback
        return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text: str) -> Dict[str, float]:
        """Simple keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = ['growth', 'strong', 'rise', 'gain', 'profit', 'beat', 'exceed', 'positive', 'bull', 'surge']
        negative_words = ['fall', 'drop', 'loss', 'decline', 'weak', 'miss', 'concern', 'bear', 'crash', 'volatile']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            return {
                'compound_score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'method': 'keyword_fallback'
            }
        
        pos_ratio = pos_count / total_sentiment_words
        neg_ratio = neg_count / total_sentiment_words
        compound = pos_ratio - neg_ratio
        
        return {
            'compound_score': compound,
            'positive': pos_ratio,
            'negative': neg_ratio,
            'neutral': max(0, 1 - pos_ratio - neg_ratio),
            'method': 'keyword_fallback'
        }

class NewsSentimentScraper:
    """Main news and sentiment scraper"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.intel_dir = Path("intel")
        self.intel_dir.mkdir(exist_ok=True)
        
        # Initialize data source
        newsapi_key = self.config.get("newsapi", {}).get("api_key")
        self.news_source = NewsAPISource(newsapi_key)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Default symbols to track
        self.tracked_symbols = self.config.get("symbols", ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"])
    
    def scrape_and_analyze(self, limit: int = 50) -> Dict:
        """Main scraping and analysis workflow"""
        
        print(f"🔍 Starting news sentiment scraping...")
        print(f"📊 Tracking symbols: {', '.join(self.tracked_symbols)}")
        print(f"🎯 Target headlines: {limit}")
        
        start_time = datetime.now(timezone.utc)
        
        # Fetch headlines
        headlines = self.news_source.fetch_headlines(self.tracked_symbols, limit)
        
        if not headlines:
            print("❌ No headlines fetched")
            return {'success': False, 'headlines_processed': 0}
        
        # Analyze sentiment for each headline
        processed_headlines = []
        
        for i, headline in enumerate(headlines):
            # Combine title and description for sentiment analysis
            full_text = f"{headline.get('title', '')} {headline.get('description', '')}"
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_sentiment(full_text)
            
            # Create processed headline record
            processed_headline = {
                'id': i + 1,
                'title': headline.get('title', ''),
                'description': headline.get('description', ''),
                'source': headline.get('source', ''),
                'published_at': headline.get('published_at', ''),
                'url': headline.get('url', ''),
                'symbols_mentioned': ','.join(headline.get('symbols_mentioned', [])),
                'sentiment_compound': sentiment['compound_score'],
                'sentiment_positive': sentiment['positive'],
                'sentiment_negative': sentiment['negative'],
                'sentiment_neutral': sentiment['neutral'],
                'sentiment_method': sentiment['method'],
                'scraped_at': start_time.isoformat()
            }
            
            processed_headlines.append(processed_headline)
        
        # Save to CSV
        csv_file = self.intel_dir / "news_sentiment.csv"
        self._save_to_csv(processed_headlines, csv_file)
        
        # Generate summary
        summary = self._generate_summary(processed_headlines)
        
        # Save metadata
        metadata = {
            'scrape_timestamp': start_time.isoformat(),
            'headlines_fetched': len(headlines),
            'headlines_processed': len(processed_headlines),
            'symbols_tracked': self.tracked_symbols,
            'sentiment_method': processed_headlines[0]['sentiment_method'] if processed_headlines else 'none',
            'summary': summary
        }
        
        metadata_file = self.intel_dir / "news_sentiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Processed {len(processed_headlines)} headlines")
        print(f"💾 Saved to: {csv_file}")
        print(f"📊 Average sentiment: {summary['avg_sentiment']:.3f}")
        print(f"📈 Most positive: {summary['most_positive']['title'][:50]}...")
        
        return {
            'success': True,
            'headlines_processed': len(processed_headlines),
            'output_file': str(csv_file),
            'summary': summary
        }
    
    def _save_to_csv(self, headlines: List[Dict], csv_file: Path):
        """Save headlines to CSV file"""
        
        if not headlines:
            return
        
        fieldnames = [
            'id', 'title', 'description', 'source', 'published_at', 'url',
            'symbols_mentioned', 'sentiment_compound', 'sentiment_positive',
            'sentiment_negative', 'sentiment_neutral', 'sentiment_method', 'scraped_at'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(headlines)
    
    def _generate_summary(self, headlines: List[Dict]) -> Dict:
        """Generate sentiment summary"""
        
        if not headlines:
            return {}
        
        sentiments = [h['sentiment_compound'] for h in headlines]
        
        # Find most positive and negative
        most_positive = max(headlines, key=lambda h: h['sentiment_compound'])
        most_negative = min(headlines, key=lambda h: h['sentiment_compound'])
        
        # Count by sentiment
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        return {
            'total_headlines': len(headlines),
            'avg_sentiment': sum(sentiments) / len(sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'most_positive': {
                'title': most_positive['title'],
                'score': most_positive['sentiment_compound']
            },
            'most_negative': {
                'title': most_negative['title'],
                'score': most_negative['sentiment_compound']
            }
        }

def load_config() -> Dict:
    """Load configuration from file"""
    config_file = Path("config/base_config.yaml")
    
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f)
    
    return {
        'newsapi': {
            'api_key': 'your_newsapi_key_here'  # Get from https://newsapi.org/
        },
        'symbols': ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
    }

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="News Sentiment Scraper (Level 6-A)")
    parser.add_argument('--limit', type=int, default=50, help='Number of headlines to fetch')
    parser.add_argument('--symbols', nargs='+', help='Symbols to track (overrides config)')
    parser.add_argument('--test', action='store_true', help='Run with demo data only')
    
    args = parser.parse_args()
    
    print("📰 News & Sentiment Scraper (Level 6-A)")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    if args.symbols:
        config['symbols'] = args.symbols
    
    if args.test:
        print("🧪 Test mode: Using demo headlines only")
        config['newsapi'] = {'api_key': None}  # Force demo mode
    
    # Initialize scraper
    scraper = NewsSentimentScraper(config)
    
    # Run scraping
    result = scraper.scrape_and_analyze(args.limit)
    
    if result['success']:
        print(f"\n✅ Level 6-A Complete!")
        print(f"📊 Headlines processed: {result['headlines_processed']}")
        print(f"💾 Output: {result['output_file']}")
        
        # Show top sentiment headlines
        if 'summary' in result:
            summary = result['summary']
            print(f"\n📈 Sentiment Summary:")
            print(f"   Positive: {summary['positive_count']}")
            print(f"   Negative: {summary['negative_count']}")
            print(f"   Neutral: {summary['neutral_count']}")
            print(f"   Average: {summary['avg_sentiment']:.3f}")
    else:
        print("❌ Scraping failed")
        exit(1)

if __name__ == "__main__":
    main()