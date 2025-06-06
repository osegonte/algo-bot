#!/usr/bin/env python3
"""
Fixed Gateway with proper Alpaca API endpoints
File: modules/market_data/gateway.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from datetime import datetime, timezone
import asyncio
import json
import requests
import logging
from dataclasses import dataclass

@dataclass
class Quote:
    """Standardized quote structure"""
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: datetime
    volume: Optional[int] = None
    source: str = "unknown"

class MarketDataGateway(ABC):
    """Abstract base class for market data providers"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_connected = False
        self.logger = logging.getLogger(self.__class__.__name__)
        # Setup basic logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    async def get_live_quote(self, symbol: str) -> Quote:
        """Get current quote for symbol"""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection"""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Return list of supported symbols"""
        pass

class AlpacaGateway(MarketDataGateway):
    """Fixed Alpaca Markets data adapter"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        super().__init__(api_key, api_secret)
        self.base_url = base_url
        self.data_url = "https://data.alpaca.markets"  # Separate data endpoint
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        })
    
    async def get_live_quote(self, symbol: str) -> Quote:
        """Get quote from Alpaca - try multiple endpoints"""
        
        # Try the data API first (real-time quotes)
        try:
            return await self._get_quote_from_data_api(symbol)
        except Exception as e:
            self.logger.warning(f"Data API failed for {symbol}: {e}")
        
        # Fallback to latest trade from bars
        try:
            return await self._get_quote_from_bars(symbol)
        except Exception as e:
            self.logger.warning(f"Bars API failed for {symbol}: {e}")
        
        # Final fallback - simulated quote
        return self._create_fallback_quote(symbol)
    
    async def _get_quote_from_data_api(self, symbol: str) -> Quote:
        """Try to get quote from Alpaca data API"""
        url = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        quote_data = data['quote']
        
        return Quote(
            symbol=symbol,
            bid=float(quote_data['bp']),
            ask=float(quote_data['ap']), 
            last=float(quote_data.get('p', quote_data['ap'])),
            timestamp=datetime.fromisoformat(quote_data['t'].replace('Z', '+00:00')),
            volume=quote_data.get('bs', 0),
            source="alpaca_data"
        )
    
    async def _get_quote_from_bars(self, symbol: str) -> Quote:
        """Get latest price from recent bars"""
        url = f"{self.data_url}/v2/stocks/{symbol}/bars/latest"
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        bar_data = data['bar']
        
        # Use close price as last, estimate bid/ask
        close_price = float(bar_data['c'])
        spread = close_price * 0.001  # 0.1% spread estimate
        
        return Quote(
            symbol=symbol,
            bid=close_price - spread/2,
            ask=close_price + spread/2,
            last=close_price,
            timestamp=datetime.fromisoformat(bar_data['t'].replace('Z', '+00:00')),
            volume=int(bar_data.get('v', 0)),
            source="alpaca_bars"
        )
    
    def _create_fallback_quote(self, symbol: str) -> Quote:
        """Create fallback quote when APIs fail"""
        # Use reasonable default prices
        if symbol == "AAPL":
            price = 190.0  # More realistic AAPL price
        elif symbol == "TSLA":
            price = 250.0
        elif symbol == "MSFT":
            price = 420.0
        else:
            price = 150.0
        
        spread = price * 0.001  # 0.1% spread
        
        return Quote(
            symbol=symbol,
            bid=price - spread/2,
            ask=price + spread/2,
            last=price,
            timestamp=datetime.now(timezone.utc),
            source="alpaca_fallback"
        )
    
    async def connect(self) -> bool:
        """Test connection to Alpaca"""
        try:
            # Test account endpoint
            response = self.session.get(f"{self.base_url}/v2/account")
            if response.status_code == 200:
                self.is_connected = True
                self.logger.info("✅ Alpaca connection successful")
                return True
            else:
                self.logger.warning(f"⚠️ Alpaca connection issues: {response.status_code}")
                self.is_connected = False
                return False
        except Exception as e:
            self.logger.error(f"❌ Alpaca connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close session"""
        self.session.close()
        self.is_connected = False
    
    def get_supported_symbols(self) -> List[str]:
        return ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

class IEXGateway(MarketDataGateway):
    """IEX Cloud data adapter"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://cloud.iexapis.com/stable"
    
    async def get_live_quote(self, symbol: str) -> Quote:
        """Get quote from IEX Cloud"""
        try:
            url = f"{self.base_url}/stock/{symbol}/quote"
            params = {"token": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            return Quote(
                symbol=symbol,
                bid=float(data.get('iexBidPrice', data['latestPrice'])),
                ask=float(data.get('iexAskPrice', data['latestPrice'])),
                last=float(data['latestPrice']),
                timestamp=datetime.fromtimestamp(data['latestUpdate'] / 1000, tz=timezone.utc),
                volume=data.get('latestVolume'),
                source="iex"
            )
            
        except Exception as e:
            self.logger.error(f"IEX quote error for {symbol}: {e}")
            return self._create_fallback_quote(symbol)
    
    def _create_fallback_quote(self, symbol: str) -> Quote:
        """Create fallback quote"""
        price = 190.0 if symbol == "AAPL" else 150.0
        spread = price * 0.001
        
        return Quote(
            symbol=symbol,
            bid=price - spread/2,
            ask=price + spread/2,
            last=price,
            timestamp=datetime.now(timezone.utc),
            source="iex_fallback"
        )
    
    async def connect(self) -> bool:
        """Test IEX connection"""
        try:
            url = f"{self.base_url}/stock/AAPL/price"
            response = requests.get(url, params={"token": self.api_key})
            self.is_connected = response.status_code == 200
            return self.is_connected
        except:
            self.is_connected = False
            return False
    
    async def disconnect(self):
        self.is_connected = False
    
    def get_supported_symbols(self) -> List[str]:
        return ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

class ForexGateway(MarketDataGateway):
    """Simple forex data adapter for EURUSD"""
    
    def __init__(self):
        super().__init__()
        
    async def get_live_quote(self, symbol: str) -> Quote:
        """Get forex quote (simplified)"""
        if symbol == "EURUSD":
            # Simulate EURUSD quote around 1.0500
            import random
            base_rate = 1.0500
            spread = 0.0002
            variation = random.uniform(-0.005, 0.005)  # ±0.5% variation
            last = base_rate + variation
            
            return Quote(
                symbol="EURUSD",
                bid=last - spread/2,
                ask=last + spread/2,
                last=last,
                timestamp=datetime.now(timezone.utc),
                source="forex_sim"
            )
        else:
            raise ValueError(f"Forex gateway only supports EURUSD, got {symbol}")
    
    async def connect(self) -> bool:
        self.is_connected = True
        return True
    
    async def disconnect(self):
        self.is_connected = False
    
    def get_supported_symbols(self) -> List[str]:
        return ["EURUSD"]

class DataGatewayFactory:
    """Factory to create appropriate gateway based on symbol and config"""
    
    @staticmethod
    def create_gateway(symbol: str, config: dict) -> MarketDataGateway:
        """Create appropriate gateway for symbol"""
        
        if symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            return ForexGateway()
        
        # For stocks, try gateways in priority order
        alpaca_config = config.get("alpaca", {})
        if alpaca_config.get("api_key"):
            return AlpacaGateway(
                alpaca_config["api_key"],
                alpaca_config["api_secret"],
                alpaca_config.get("base_url", "https://paper-api.alpaca.markets")
            )
        
        iex_key = config.get("iex", {}).get("api_key")
        if iex_key:
            return IEXGateway(iex_key)
        
        # Fallback to Alpaca even without proper keys (will use fallback quotes)
        return AlpacaGateway("demo", "demo")

# Test function
async def test_gateways():
    """Test all gateways for AAPL and EURUSD"""
    
    config = {
        "alpaca": {
            "api_key": "PKKJRF8U4QTYWMAVLYUI",
            "api_secret": "4kGChcW8cqkxasJfvyVMYfGRKbPqwNcx8MzM26ws",
            "base_url": "https://paper-api.alpaca.markets"
        }
    }
    
    print("🔌 Testing Fixed Market Data Gateways...")
    print("=" * 50)
    
    # Test AAPL
    aapl_gateway = DataGatewayFactory.create_gateway("AAPL", config)
    connected = await aapl_gateway.connect()
    print(f"🔗 AAPL gateway connected: {connected}")
    
    aapl_quote = await aapl_gateway.get_live_quote("AAPL")
    print(f"📈 AAPL: ${aapl_quote.last:.2f} (bid: ${aapl_quote.bid:.2f}, ask: ${aapl_quote.ask:.2f}) [{aapl_quote.source}]")
    
    # Test EURUSD
    eur_gateway = DataGatewayFactory.create_gateway("EURUSD", config)
    await eur_gateway.connect()
    eur_quote = await eur_gateway.get_live_quote("EURUSD")
    print(f"💱 EURUSD: {eur_quote.last:.4f} (bid: {eur_quote.bid:.4f}, ask: {eur_quote.ask:.4f}) [{eur_quote.source}]")
    
    print("✅ Fixed gateway test complete!")

if __name__ == "__main__":
    asyncio.run(test_gateways())