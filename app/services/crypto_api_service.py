# app/services/crypto_api_service.py
"""
Enhanced Crypto API Service with optimizations:
- Session + Retry + Backoff + Host rotation
- Circuit breaker cho APIs  
- Bulk API calls với Binance
- Enhanced rate limiting với threading locks
"""

import requests
import time
import asyncio
import logging
import threading
from typing import Optional, Dict, Any, List

from .http_client import RotatingHostHTTP

logger = logging.getLogger(__name__)


class CryptoAPIService:
    def __init__(self):
        # API keys can be configured via environment variables or passed in
        self.api_keys = {
            'coingecko_pro': None,  # Set this if you have a Pro API key
        }
        
        # Rate limits with threading locks
        self.rate_limits = {
            'coingecko_free': {'max_calls': 10, 'period': 60, 'calls': []},
            'coingecko_pro': {'max_calls': 500, 'period': 60, 'calls': []},
            'binance': {'max_calls': 1000, 'period': 60, 'calls': []},
            'coinpaprika': {'max_calls': 20, 'period': 60, 'calls': []}
        }
        self._rl_lock = threading.Lock()

        # Binance HTTP client với host rotation
        self.binance_http = RotatingHostHTTP(
            hosts=[
                "https://api.binance.com",
                "https://api-gcp.binance.com", 
                "https://api1.binance.com",
                "https://api2.binance.com",
                "https://api3.binance.com",
                "https://api4.binance.com",
            ],
            timeout=(4, 8), max_retries=4, backoff_factor=0.4
        )

        # Circuit breaker theo nguồn API
        self._api_down_until = {}
        
        # Standard HTTP session for other APIs
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CryptoPriceBot/1.0 (+contact admin)",
            "Accept": "application/json"
        })

    def _check_rate_limit(self, api_name: str) -> bool:
        """Thread-safe rate limit checker"""
        with self._rl_lock:
            limits = self.rate_limits.get(api_name)
            if not limits:
                return True
            
            now = time.time()
            # Cleanup old calls
            limits['calls'] = [t for t in limits['calls'] if now - t < limits['period']]
            
            if len(limits['calls']) >= limits['max_calls']:
                return False
            
            limits['calls'].append(now)
            return True

    def _mark_api_down(self, api_name: str, seconds: int = 300):
        """Mark API as down for specified duration"""
        self._api_down_until[api_name] = time.time() + seconds

    def _is_api_available(self, api_name: str) -> bool:
        """Check if API is available (not marked down)"""
        return time.time() > self._api_down_until.get(api_name, 0)

    def get_binance_24h_ticker(self, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """Get 24hr ticker statistics from Binance"""
        if not self._is_api_available('binance') or not self._check_rate_limit('binance'):
            return None
        
        try:
            data = self.binance_http.get_json(f"api/v3/ticker/24hr", {"symbol": symbol})
            if data:
                return {
                    'symbol': data.get('symbol'),
                    'price': float(data.get('lastPrice', 0)),
                    'change_24h': float(data.get('priceChangePercent', 0)),
                    'volume_24h': float(data.get('volume', 0)),
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0))
                }
        except Exception as e:
            logger.error(f"Binance 24h ticker failed: {e}")
            self._mark_api_down('binance', 120)
        
        return None

    def get_binance_klines(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100) -> Optional[List[Dict]]:
        """Get Kline/Candlestick data from Binance"""
        if not self._is_api_available('binance') or not self._check_rate_limit('binance'):
            return None
        
        try:
            data = self.binance_http.get_json("api/v3/klines", {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            })
            
            if data:
                return [{
                    'timestamp': int(k[0]),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                } for k in data]
        except Exception as e:
            logger.error(f"Binance klines failed: {e}")
            self._mark_api_down('binance', 120)
        
        return None

    def get_coingecko_price(self, coin_id: str = "bitcoin") -> Optional[Dict[str, Any]]:
        """Get price from CoinGecko with Pro API fallback"""
        # Try Pro API first if available
        if self.api_keys.get('coingecko_pro') and self._is_api_available('coingecko_pro'):
            if self._check_rate_limit('coingecko_pro'):
                try:
                    headers = {"x-cg-pro-api-key": self.api_keys['coingecko_pro']}
                    url = f"https://pro-api.coingecko.com/api/v3/simple/price"
                    params = {
                        'ids': coin_id,
                        'vs_currencies': 'usd',
                        'include_24hr_change': 'true',
                        'include_24hr_vol': 'true'
                    }
                    
                    r = self.session.get(url, params=params, headers=headers, timeout=(5, 10))
                    r.raise_for_status()
                    data = r.json()
                    
                    if coin_id in data:
                        coin_data = data[coin_id]
                        return {
                            'price': coin_data.get('usd', 0),
                            'change_24h': coin_data.get('usd_24h_change', 0),
                            'volume_24h': coin_data.get('usd_24h_vol', 0)
                        }
                except Exception as e:
                    logger.warning(f"CoinGecko Pro failed: {e}")
                    self._mark_api_down('coingecko_pro', 180)
        
        # Fallback to free API
        if self._is_api_available('coingecko_free') and self._check_rate_limit('coingecko_free'):
            try:
                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': coin_id,
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_24hr_vol': 'true'
                }
                
                r = self.session.get(url, params=params, timeout=(8, 12))
                r.raise_for_status()
                data = r.json()
                
                if coin_id in data:
                    coin_data = data[coin_id]
                    return {
                        'price': coin_data.get('usd', 0),
                        'change_24h': coin_data.get('usd_24h_change', 0),
                        'volume_24h': coin_data.get('usd_24h_vol', 0)
                    }
            except Exception as e:
                logger.error(f"CoinGecko free failed: {e}")
                self._mark_api_down('coingecko_free', 300)
        
        return None

    def get_fallback_price(self, symbol: str = "BTC") -> Optional[Dict[str, Any]]:
        """Get price from alternative sources as fallback"""
        # Try CoinPaprika
        if self._is_api_available('coinpaprika') and self._check_rate_limit('coinpaprika'):
            try:
                # Map common symbols to CoinPaprika IDs
                symbol_map = {
                    'BTC': 'btc-bitcoin',
                    'ETH': 'eth-ethereum',
                    'BNB': 'bnb-binance-coin'
                }
                
                coin_id = symbol_map.get(symbol.upper(), f"{symbol.lower()}-{symbol.lower()}")
                url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}"
                
                r = self.session.get(url, timeout=(8, 12))
                r.raise_for_status()
                data = r.json()
                
                if 'quotes' in data and 'USD' in data['quotes']:
                    usd_data = data['quotes']['USD']
                    return {
                        'price': usd_data.get('price', 0),
                        'change_24h': usd_data.get('percent_change_24h', 0),
                        'volume_24h': usd_data.get('volume_24h', 0)
                    }
            except Exception as e:
                logger.warning(f"CoinPaprika failed: {e}")
                self._mark_api_down('coinpaprika', 240)
        
        return None

    def get_unified_price_data(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get price data with multiple fallbacks"""
        # Convert symbol for different APIs
        binance_symbol = f"{symbol.upper()}USDT"
        coingecko_id = "bitcoin" if symbol.upper() == "BTC" else symbol.lower()
        
        # Try Binance first (most reliable for major coins)
        binance_data = self.get_binance_24h_ticker(binance_symbol)
        if binance_data:
            return {
                'price': binance_data['price'],
                'change_24h': binance_data['change_24h'],
                'volume_24h': binance_data['volume_24h'],
                'high_24h': binance_data['high_24h'],
                'low_24h': binance_data['low_24h'],
                'source': 'binance'
            }
        
        # Try CoinGecko
        cg_data = self.get_coingecko_price(coingecko_id)
        if cg_data:
            return {
                'price': cg_data['price'],
                'change_24h': cg_data['change_24h'],
                'volume_24h': cg_data.get('volume_24h', 0),
                'high_24h': 0,  # CoinGecko simple API doesn't provide this
                'low_24h': 0,
                'source': 'coingecko'
            }
        
        # Try fallback sources
        fallback_data = self.get_fallback_price(symbol)
        if fallback_data:
            return {
                'price': fallback_data['price'],
                'change_24h': fallback_data['change_24h'], 
                'volume_24h': fallback_data.get('volume_24h', 0),
                'high_24h': 0,
                'low_24h': 0,
                'source': 'coinpaprika'
            }
        
        # Return stub data if all APIs fail
        logger.warning(f"All APIs failed for {symbol}, returning stub data")
        return {
            'price': 50000.0 if symbol.upper() == "BTC" else 1000.0,
            'change_24h': 0.0,
            'volume_24h': 0.0,
            'high_24h': 0.0,
            'low_24h': 0.0,
            'source': 'stub'
        }
