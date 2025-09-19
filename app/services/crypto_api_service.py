""""""# app/services/crypto_api_service.py

Enhanced Crypto API Service with optimizations:

- Session + Retry + Backoff + Host rotationCrypto API Service với optimizations:# Unified crypto API service with multiple sources and fallbacks

- Circuit breaker cho APIs  

- Bulk API calls với Binance- Session + Retry + Backoff + Host rotation

- Enhanced rate limiting với threading locks

"""- Circuit breaker cho APIsimport requests



import time- Bulk API calls với Binanceimport asyncio

import logging

import threading- Enhanced rate limiting với threading locksimport time

from typing import Optional, Dict, Any, List

import requests"""from typing import Dict, List, Optional, Any



from config.security.env_loader import get_env_loaderimport logging

from .http_client import RotatingHostHTTP

import timefrom config.security.env_loader import get_env_loader

logger = logging.getLogger(__name__)

import logging

class CryptoAPIService:

    def __init__(self):import threadinglogger = logging.getLogger(__name__)

        self.env = get_env_loader()

        self.api_keys = self.env.get_api_keys()from typing import Optional, Dict, Any, List, Union

        

        # Rate limits with threading locksimport requestsclass CryptoAPIService:

        self.rate_limits = {

            'coingecko_free': {'max_calls': 10, 'period': 60, 'calls': []},    """Unified crypto API service with fallback strategies"""

            'coingecko_pro': {'max_calls': 500, 'period': 60, 'calls': []},

            'binance': {'max_calls': 1000, 'period': 60, 'calls': []},from config.security.env_loader import get_env_loader    

            'coinpaprika': {'max_calls': 20, 'period': 60, 'calls': []}

        }from .http_client import RotatingHostHTTP    def __init__(self):

        self._rl_lock = threading.Lock()

        self.env = get_env_loader()

        # Binance HTTP client với host rotation

        self.binance_http = RotatingHostHTTP(logger = logging.getLogger(__name__)        self.api_keys = self.env.get_api_keys()

            hosts=[

                "https://api.binance.com",        self.rate_limits = {

                "https://api-gcp.binance.com", 

                "https://api1.binance.com",class CryptoAPIService:            'coingecko_free': {'max_calls': 10, 'period': 60, 'calls': []},  # Conservative limit

                "https://api2.binance.com",

                "https://api3.binance.com",    def __init__(self):            'coingecko_pro': {'max_calls': 500, 'period': 60, 'calls': []},

                "https://api4.binance.com",

            ],        self.env = get_env_loader()            'binance': {'max_calls': 1000, 'period': 60, 'calls': []},  # High limit for public API

            timeout=(4, 8), max_retries=4, backoff_factor=0.4

        )        self.api_keys = self.env.get_api_keys()            'coinpaprika': {'max_calls': 20, 'period': 60, 'calls': []}



        # Circuit breaker theo nguồn API                }

        self._api_down_until = {}

                # Rate limits with threading locks        

        # Standard HTTP session for other APIs

        self.session = requests.Session()        self.rate_limits = {    def _can_make_request(self, api_name: str) -> bool:

        self.session.headers.update({

            "User-Agent": "CryptoPriceBot/1.0",            'coingecko_free': {'max_calls': 10, 'period': 60, 'calls': []},        """Check if we can make request without hitting rate limit"""

            "Accept": "application/json"

        })            'coingecko_pro': {'max_calls': 500, 'period': 60, 'calls': []},        current_time = time.time()



        logger.info("✅ CryptoAPIService initialized with enhanced reliability")            'binance': {'max_calls': 1000, 'period': 60, 'calls': []},        rate_limit = self.rate_limits[api_name]



    def _can_make_request(self, api_name: str) -> bool:            'coinpaprika': {'max_calls': 20, 'period': 60, 'calls': []}        

        """Thread-safe rate limiting check"""

        with self._rl_lock:        }        # Remove old calls

            current_time = time.time()

            rl = self.rate_limits[api_name]        self._rl_lock = threading.Lock()        rate_limit['calls'] = [

            rl['calls'] = [t for t in rl['calls'] if current_time - t < rl['period']]

            return len(rl['calls']) < rl['max_calls']            call_time for call_time in rate_limit['calls']



    def _record_request(self, api_name: str):        # Binance HTTP client với host rotation            if current_time - call_time < rate_limit['period']

        """Thread-safe rate limiting record"""

        with self._rl_lock:        self.binance_http = RotatingHostHTTP(        ]

            self.rate_limits[api_name]['calls'].append(time.time())

            hosts=[        

    def _api_available(self, name: str) -> bool:

        """Check if API is not in circuit breaker mode"""                "https://api.binance.com",        return len(rate_limit['calls']) < rate_limit['max_calls']

        return self._api_down_until.get(name, 0) < time.time()

                "https://api-gcp.binance.com",     

    def _trip_api(self, name: str, seconds: int = 120):

        """Trip circuit breaker for an API"""                "https://api1.binance.com",    def _record_request(self, api_name: str):

        self._api_down_until[name] = time.time() + seconds

        logger.warning(f"🔌 Circuit breaker activated for {name} - {seconds}s timeout")                "https://api2.binance.com",        """Record API request"""



    def _get_bulk_prices_binance(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:                "https://api3.binance.com",        self.rate_limits[api_name]['calls'].append(time.time())

        """Get bulk prices from Binance using symbols parameter"""

        if not symbols:                "https://api4.binance.com",    

            return None

                        ],    def get_crypto_price(self, symbol: str) -> Optional[Dict[str, Any]]:

        # Chuẩn hoá -> ["BTCUSDT", ...]

        binance_syms = [s.upper() + "USDT" for s in symbols]            timeout=(4, 8),           # connect=4s, read=8s        """Get crypto price with fallback strategy - Binance first for reliability"""

        

        try:            max_retries=4,        

            symbols_json = str(binance_syms).replace("'", '"')

            data = self.binance_http.get_json("/api/v3/ticker/price", params={            backoff_factor=0.4        # Strategy 1: Binance API (public endpoint - no key needed, highest rate limit)

                "symbols": symbols_json

            })        )        if self._can_make_request('binance'):

            

            if isinstance(data, list):            result = self._get_price_binance(symbol)

                out = {}

                for item in data:        # Circuit breaker theo nguồn API            if result:

                    sym = item.get("symbol", "")

                    if sym.endswith("USDT"):        self._api_down_until = {}                self._record_request('binance')

                        base = sym[:-4]

                        if base.upper() in [s.upper() for s in symbols]:                        logger.info(f"✅ Price from Binance API: {symbol}")

                            out[base.upper()] = {

                                'current_price': float(item['price']),        # Standard HTTP session for other APIs                return result

                                'change_24h': 0.0,

                                'market_cap': 0.0,        self.session = requests.Session()        

                                'volume_24h': 0.0,

                                'source': 'binance_bulk'        self.session.headers.update({        # Strategy 2: CoinGecko with API key (if available)

                            }

                if out:            "User-Agent": "CryptoPriceBot/1.0",        if self.api_keys['coingecko'] and self._can_make_request('coingecko_pro'):

                    logger.info(f"✅ Binance bulk API returned {len(out)} prices")

                    return out            "Accept": "application/json"            result = self._get_price_coingecko_pro(symbol)

        except Exception as e:

            logger.warning(f"⚠️ Binance batch symbols failed: {e}")        })            if result:



        # Fallback: get all prices                self._record_request('coingecko_pro')

        try:

            all_prices = self.binance_http.get_json("/api/v3/ticker/price")        logger.info("✅ CryptoAPIService initialized with enhanced reliability")                logger.info(f"✅ Price from CoinGecko Pro API: {symbol}")

            if isinstance(all_prices, list):

                out = {}                return result

                wanted = {s.upper() for s in symbols}

                for item in all_prices:    def _can_make_request(self, api_name: str) -> bool:        

                    sym = item.get("symbol", "")

                    if sym.endswith("USDT"):        """Thread-safe rate limiting check"""        # Strategy 3: CoinGecko free tier (backup)

                        base = sym[:-4]

                        if base in wanted:        with self._rl_lock:        if self._can_make_request('coingecko_free'):

                            out[base] = {

                                'current_price': float(item['price']),            current_time = time.time()            result = self._get_price_coingecko_free(symbol)

                                'change_24h': 0.0,

                                'market_cap': 0.0,            rl = self.rate_limits[api_name]            if result:

                                'volume_24h': 0.0,

                                'source': 'binance_bulk_all'            rl['calls'] = [t for t in rl['calls'] if current_time - t < rl['period']]                self._record_request('coingecko_free')

                            }

                if out:            return len(rl['calls']) < rl['max_calls']                logger.info(f"✅ Price from CoinGecko Free API: {symbol}")

                    logger.info(f"✅ Binance bulk fallback returned {len(out)} prices")

                    return out                return result

        except Exception as e:

            logger.error(f"❌ Binance bulk fallback failed: {e}")    def _record_request(self, api_name: str):        

            

        return None        """Thread-safe rate limiting record"""        # Strategy 4: CoinPaprika (free alternative)



    def _get_price_binance(self, symbol: str) -> Optional[Dict]:        with self._rl_lock:        if self._can_make_request('coinpaprika'):

        """Get single price from Binance with 24h data"""

        try:            self.rate_limits[api_name]['calls'].append(time.time())            result = self._get_price_coinpaprika(symbol)

            sym = f"{symbol.upper()}USDT"

            data = self.binance_http.get_json("/api/v3/ticker/24hr", params={"symbol": sym})            if result:

            

            if not isinstance(data, dict):    def _api_available(self, name: str) -> bool:                self._record_request('coinpaprika')

                return None

                        """Check if API is not in circuit breaker mode"""                logger.info(f"✅ Price from CoinPaprika API: {symbol}")

            return {

                'current_price': float(data.get('lastPrice', 0) or 0),        return self._api_down_until.get(name, 0) < time.time()                return result

                'change_24h': float(data.get('priceChangePercent', 0) or 0),

                'market_cap': 0.0,        

                'volume_24h': float(data.get('quoteVolume', 0) or 0),

                'source': 'binance'    def _trip_api(self, name: str, seconds: int = 120):        logger.warning(f"❌ All API sources failed for {symbol}")

            }

        except Exception as e:        """Trip circuit breaker for an API"""        return None

            logger.error(f"❌ Binance API error for {symbol}: {e}")

            return None        self._api_down_until[name] = time.time() + seconds    



    def get_crypto_price(self, symbol: str) -> Optional[Dict[str, Any]]:        logger.warning(f"🔌 Circuit breaker activated for {name} - {seconds}s timeout")    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:

        """

        Get crypto price với circuit breaker và enhanced fallback logic        """Get multiple crypto prices efficiently using bulk APIs"""

        Priority: Binance → CoinGecko Pro → CoinGecko Free → CoinPaprika

        """    def _get_bulk_prices_binance(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:        result = {}

        symbol = symbol.upper()

                """Get bulk prices from Binance using symbols parameter"""        

        # 1) Binance (nếu chưa bị trip)

        if self._api_available('binance') and self._can_make_request('binance'):        if not symbols:        # Try Binance bulk first (most efficient and reliable)

            res = self._get_price_binance(symbol)

            if res:            return None        if self._can_make_request('binance'):

                self._record_request('binance')

                logger.info(f"✅ Price from Binance API: {symbol} = ${res['current_price']}")                        logger.info(f"🚀 Attempting Binance bulk API for {len(symbols)} symbols")

                return res

            else:        # Chuẩn hoá -> ["BTCUSDT", ...]            binance_data = self._get_bulk_prices_binance(symbols)

                self._trip_api('binance', 120)

        binance_syms = [s.upper() + "USDT" for s in symbols]            if binance_data:

        # Fallback logic for other APIs would go here...

        logger.warning(f"❌ All API sources failed for {symbol}")                        self._record_request('binance')

        return None

        # 1) Thử batch symbols với JSON array                result.update(binance_data)

    def get_multiple_prices(self, symbols: List[str], max_bulk_size: int = 50) -> Dict[str, Dict[str, Any]]:

        """Get multiple crypto prices với enhanced bulk optimization"""        try:                logger.info(f"✅ Binance bulk API success: {len(binance_data)}/{len(symbols)} symbols")

        if not symbols:

            return {}            symbols_json = str(binance_syms).replace("'", '"')  # Convert to JSON array            else:

            

        symbols = [s.upper() for s in symbols]            data = self.binance_http.get_json("/api/v3/ticker/price", params={                logger.warning("❌ Binance bulk API failed, falling back to individual calls")

        results = {}

                        "symbols": symbols_json        else:

        logger.info(f"🔍 Fetching prices for {len(symbols)} symbols: {symbols}")

            })            logger.warning("⚠️ Binance rate limit reached, using individual calls")

        # Try Binance bulk API first

        if self._api_available('binance') and self._can_make_request('binance'):                    

            for i in range(0, len(symbols), max_bulk_size):

                batch = symbols[i:i + max_bulk_size]            if isinstance(data, list):        # Fill missing symbols with individual calls (will prefer Binance for each)

                bulk_data = self._get_bulk_prices_binance(batch)

                                out = {}        missing_symbols = [s for s in symbols if s not in result]

                if bulk_data:

                    self._record_request('binance')                for item in data:        if missing_symbols:

                    results.update(bulk_data)

                    logger.info(f"✅ Binance bulk returned {len(bulk_data)} prices")                    sym = item.get("symbol", "")            logger.info(f"🔄 Getting {len(missing_symbols)} missing symbols individually")

                else:

                    self._trip_api('binance', 120)                    if sym.endswith("USDT"):            for symbol in missing_symbols:

                    break

                        base = sym[:-4]                price_data = self.get_crypto_price(symbol)

        # Get missing symbols individually

        missing_symbols = [s for s in symbols if s not in results]                        if base.upper() in [s.upper() for s in symbols]:                if price_data:

        if missing_symbols:

            logger.info(f"🔄 Fetching {len(missing_symbols)} missing symbols individually")                            out[base.upper()] = {                    result[symbol] = price_data

            

            for symbol in missing_symbols:                                'current_price': float(item['price']),        

                price_data = self.get_crypto_price(symbol)

                if price_data:                                'change_24h': 0.0,        logger.info(f"📊 Final result: {len(result)}/{len(symbols)} symbols retrieved")

                    results[symbol] = price_data

                                'market_cap': 0.0,        return result

        logger.info(f"📊 Final results: {len(results)}/{len(symbols)} symbols retrieved")

        return results                                'volume_24h': 0.0,    

                                'source': 'binance_bulk'    def _get_bulk_prices_binance(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:

                            }        """Get all prices from Binance in one call"""

                if out:        try:

                    logger.info(f"✅ Binance bulk API returned {len(out)} prices")            # Binance fallback URLs

                    return out            urls = [

        except Exception as e:                "https://api.binance.com/api/v3/ticker/price",

            logger.warning(f"⚠️ Binance batch symbols failed: {e}")                "https://api-gcp.binance.com/api/v3/ticker/price", 

                "https://api1.binance.com/api/v3/ticker/price"

        # 2) Fallback: tải toàn bộ rồi lọc            ]

        try:            

            all_prices = self.binance_http.get_json("/api/v3/ticker/price")            for url in urls:

            if isinstance(all_prices, list):                try:

                out = {}                    response = requests.get(url, timeout=10)

                wanted = {s.upper() for s in symbols}                    response.raise_for_status()

                for item in all_prices:                    

                    sym = item.get("symbol", "")                    all_prices = response.json()

                    if sym.endswith("USDT"):                    result = {}

                        base = sym[:-4]                    

                        if base in wanted:                    # Convert to our format

                            out[base] = {                    for item in all_prices:

                                'current_price': float(item['price']),                        binance_symbol = item['symbol']

                                'change_24h': 0.0,                        if binance_symbol.endswith('USDT'):

                                'market_cap': 0.0,                            symbol = binance_symbol.replace('USDT', '')

                                'volume_24h': 0.0,                            if symbol in symbols:

                                'source': 'binance_bulk_all'                                result[symbol] = {

                            }                                    'current_price': float(item['price']),

                if out:                                    'change_24h': 0,  # No 24h change in this endpoint

                    logger.info(f"✅ Binance bulk fallback returned {len(out)} prices")                                    'market_cap': 0,

                    return out                                    'volume_24h': 0,

        except Exception as e:                                    'source': 'binance_bulk'

            logger.error(f"❌ Binance bulk fallback failed: {e}")                                }

                                

        return None                    return result if result else None

                    

    def _get_price_binance(self, symbol: str) -> Optional[Dict]:                except Exception as e:

        """Get single price from Binance with 24h data"""                    logger.warning(f"Binance URL {url} failed: {e}")

        try:                    continue

            sym = f"{symbol.upper()}USDT"                    

            data = self.binance_http.get_json("/api/v3/ticker/24hr", params={"symbol": sym})            return None

                        

            if not isinstance(data, dict):        except Exception as e:

                return None            logger.error(f"Binance bulk API error: {e}")

                            return None

            return {    

                'current_price': float(data.get('lastPrice', 0) or 0),    def _get_price_coingecko_pro(self, symbol: str) -> Optional[Dict]:

                'change_24h': float(data.get('priceChangePercent', 0) or 0),        """Get price from CoinGecko Pro API"""

                'market_cap': 0.0,        try:

                'volume_24h': float(data.get('quoteVolume', 0) or 0),            coin_id = self._symbol_to_coingecko_id(symbol)

                'source': 'binance'            url = f"https://pro-api.coingecko.com/api/v3/simple/price"

            }            headers = {'X-Cg-Pro-Api-Key': self.api_keys['coingecko']}

        except (TypeError, ValueError, KeyError) as e:            params = {

            logger.warning(f"⚠️ Binance price parsing error for {symbol}: {e}")                'ids': coin_id,

            return None                'vs_currencies': 'usd',

        except Exception as e:                'include_24hr_change': 'true',

            logger.error(f"❌ Binance API error for {symbol}: {e}")                'include_market_cap': 'true',

            return None                'include_24hr_vol': 'true'

            }

    def _get_price_coingecko_pro(self, symbol: str) -> Optional[Dict]:            

        """Get price from CoinGecko Pro API"""            response = requests.get(url, params=params, headers=headers, timeout=10)

        if not self.api_keys.get('coingecko'):            response.raise_for_status()

            return None            

                        data = response.json()

        try:            if coin_id in data:

            headers = {"x-cg-pro-api-key": self.api_keys['coingecko']}                coin_data = data[coin_id]

            symbol_id = self._symbol_to_coingecko_id(symbol)                return {

                                'current_price': coin_data.get('usd', 0),

            url = f"https://pro-api.coingecko.com/api/v3/simple/price"                    'change_24h': coin_data.get('usd_24h_change', 0),

            params = {                    'market_cap': coin_data.get('usd_market_cap', 0),

                'ids': symbol_id,                    'volume_24h': coin_data.get('usd_24h_vol', 0),

                'vs_currencies': 'usd',                    'source': 'coingecko_pro'

                'include_24hr_change': 'true',                }

                'include_market_cap': 'true',            return None

                'include_24hr_vol': 'true'            

            }        except Exception as e:

                        logger.error(f"CoinGecko Pro API error for {symbol}: {e}")

            response = self.session.get(url, headers=headers, params=params, timeout=(5, 10))            return None

            response.raise_for_status()    

            data = response.json()    def _get_price_coingecko_free(self, symbol: str) -> Optional[Dict]:

                    """Get price from CoinGecko Free API"""

            if symbol_id in data:        try:

                coin_data = data[symbol_id]            coin_id = self._symbol_to_coingecko_id(symbol)

                return {            url = f"https://api.coingecko.com/api/v3/simple/price"

                    'current_price': coin_data.get('usd', 0),            params = {

                    'change_24h': coin_data.get('usd_24h_change', 0),                'ids': coin_id,

                    'market_cap': coin_data.get('usd_market_cap', 0),                'vs_currencies': 'usd',

                    'volume_24h': coin_data.get('usd_24h_vol', 0),                'include_24hr_change': 'true',

                    'source': 'coingecko_pro'                'include_market_cap': 'true',

                }                'include_24hr_vol': 'true'

        except Exception as e:            }

            logger.warning(f"⚠️ CoinGecko Pro error for {symbol}: {e}")            

            return None            response = requests.get(url, params=params, timeout=10)

            response.raise_for_status()

    def _get_price_coingecko_free(self, symbol: str) -> Optional[Dict]:            

        """Get price from CoinGecko Free API"""            data = response.json()

        try:            if coin_id in data:

            symbol_id = self._symbol_to_coingecko_id(symbol)                coin_data = data[coin_id]

                            return {

            url = f"https://api.coingecko.com/api/v3/simple/price"                    'current_price': coin_data.get('usd', 0),

            params = {                    'change_24h': coin_data.get('usd_24h_change', 0),

                'ids': symbol_id,                    'market_cap': coin_data.get('usd_market_cap', 0),

                'vs_currencies': 'usd',                    'volume_24h': coin_data.get('usd_24h_vol', 0),

                'include_24hr_change': 'true',                    'source': 'coingecko_free'

                'include_market_cap': 'true',                }

                'include_24hr_vol': 'true'            return None

            }            

                    except Exception as e:

            response = self.session.get(url, params=params, timeout=(5, 10))            logger.error(f"CoinGecko Free API error for {symbol}: {e}")

            response.raise_for_status()            return None

            data = response.json()    

                def _get_price_binance(self, symbol: str) -> Optional[Dict]:

            if symbol_id in data:        """Get price from Binance API with fallback URLs"""

                coin_data = data[symbol_id]        try:

                return {            # Binance fallback URLs

                    'current_price': coin_data.get('usd', 0),            urls = [

                    'change_24h': coin_data.get('usd_24h_change', 0),                "https://api.binance.com/api/v3/ticker/24hr",

                    'market_cap': coin_data.get('usd_market_cap', 0),                "https://api-gcp.binance.com/api/v3/ticker/24hr",

                    'volume_24h': coin_data.get('usd_24h_vol', 0),                "https://api1.binance.com/api/v3/ticker/24hr"

                    'source': 'coingecko_free'            ]

                }            

        except Exception as e:            params = {'symbol': f"{symbol}USDT"}

            logger.warning(f"⚠️ CoinGecko Free error for {symbol}: {e}")            

            return None            for url in urls:

                try:

    def _get_price_coinpaprika(self, symbol: str) -> Optional[Dict]:                    response = requests.get(url, params=params, timeout=10)

        """Get price from CoinPaprika API"""                    response.raise_for_status()

        try:                    

            symbol_id = self._symbol_to_coinpaprika_id(symbol)                    data = response.json()

                                return {

            url = f"https://api.coinpaprika.com/v1/tickers/{symbol_id}"                        'current_price': float(data.get('lastPrice', 0)),

            response = self.session.get(url, timeout=(5, 10))                        'change_24h': float(data.get('priceChangePercent', 0)),

            response.raise_for_status()                        'market_cap': 0,  # Binance doesn't provide market cap

            data = response.json()                        'volume_24h': float(data.get('quoteVolume', 0)),

                                    'source': 'binance'

            if 'quotes' in data and 'USD' in data['quotes']:                    }

                usd_data = data['quotes']['USD']                    

                return {                except Exception as e:

                    'current_price': usd_data.get('price', 0),                    logger.warning(f"Binance URL {url} failed: {e}")

                    'change_24h': usd_data.get('percent_change_24h', 0),                    continue

                    'market_cap': usd_data.get('market_cap', 0),                    

                    'volume_24h': usd_data.get('volume_24h', 0),            return None

                    'source': 'coinpaprika'            

                }        except Exception as e:

        except Exception as e:            logger.error(f"Binance API error for {symbol}: {e}")

            logger.warning(f"⚠️ CoinPaprika error for {symbol}: {e}")            return None

            return None    

    def _get_price_coinpaprika(self, symbol: str) -> Optional[Dict]:

    def get_crypto_price(self, symbol: str) -> Optional[Dict[str, Any]]:        """Get price from CoinPaprika API (free alternative)"""

        """        try:

        Get crypto price với circuit breaker và enhanced fallback logic            # CoinPaprika uses different IDs

        Priority: Binance → CoinGecko Pro → CoinGecko Free → CoinPaprika            coin_id = self._symbol_to_coinpaprika_id(symbol)

        """            url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}"

        symbol = symbol.upper()            

                    response = requests.get(url, timeout=10)

        # 1) Binance (nếu chưa bị trip)            response.raise_for_status()

        if self._api_available('binance') and self._can_make_request('binance'):            

            res = self._get_price_binance(symbol)            data = response.json()

            if res:            quotes = data.get('quotes', {}).get('USD', {})

                self._record_request('binance')            

                logger.info(f"✅ Price from Binance API: {symbol} = ${res['current_price']}")            return {

                return res                'current_price': quotes.get('price', 0),

            else:                'change_24h': quotes.get('percent_change_24h', 0),

                self._trip_api('binance', 120)  # tạm ngủ 2 phút                'market_cap': quotes.get('market_cap', 0),

                'volume_24h': quotes.get('volume_24h', 0),

        # 2) CoinGecko Pro                'source': 'coinpaprika'

        if (self.api_keys.get('coingecko') and             }

            self._api_available('coingecko_pro') and             

            self._can_make_request('coingecko_pro')):        except Exception as e:

            res = self._get_price_coingecko_pro(symbol)            logger.error(f"CoinPaprika API error for {symbol}: {e}")

            if res:            return None

                self._record_request('coingecko_pro')    

                logger.info(f"✅ Price from CoinGecko Pro: {symbol} = ${res['current_price']}")    def _symbol_to_coingecko_id(self, symbol: str) -> str:

                return res        """Convert symbol to CoinGecko ID"""

            else:        symbol_map = {

                self._trip_api('coingecko_pro', 60)            'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',

            'XRP': 'ripple', 'ADA': 'cardano', 'DOGE': 'dogecoin',

        # 3) CoinGecko Free            'SOL': 'solana', 'TRX': 'tron', 'DOT': 'polkadot',

        if self._api_available('coingecko_free') and self._can_make_request('coingecko_free'):            'MATIC': 'matic-network', 'LTC': 'litecoin', 'SHIB': 'shiba-inu',

            res = self._get_price_coingecko_free(symbol)            'AVAX': 'avalanche-2', 'UNI': 'uniswap', 'LINK': 'chainlink'

            if res:        }

                self._record_request('coingecko_free')        return symbol_map.get(symbol, symbol.lower())

                logger.info(f"✅ Price from CoinGecko Free: {symbol} = ${res['current_price']}")    

                return res    def _symbol_to_coinpaprika_id(self, symbol: str) -> str:

            else:        """Convert symbol to CoinPaprika ID"""

                self._trip_api('coingecko_free', 60)        symbol_map = {

            'BTC': 'btc-bitcoin', 'ETH': 'eth-ethereum', 'BNB': 'bnb-binance-coin',

        # 4) CoinPaprika            'XRP': 'xrp-xrp', 'ADA': 'ada-cardano', 'DOGE': 'doge-dogecoin',

        if self._api_available('coinpaprika') and self._can_make_request('coinpaprika'):            'SOL': 'sol-solana', 'TRX': 'trx-tron', 'DOT': 'dot-polkadot',

            res = self._get_price_coinpaprika(symbol)            'MATIC': 'matic-polygon', 'LTC': 'ltc-litecoin', 'SHIB': 'shib-shiba-inu'

            if res:        }

                self._record_request('coinpaprika')        return symbol_map.get(symbol, f"{symbol.lower()}-{symbol.lower()}")

                logger.info(f"✅ Price from CoinPaprika: {symbol} = ${res['current_price']}")

                return res# Global service instance

            else:crypto_api_service = CryptoAPIService()
                self._trip_api('coinpaprika', 60)

        logger.warning(f"❌ All API sources failed for {symbol}")
        return None

    def get_multiple_prices(self, symbols: List[str], max_bulk_size: int = 50) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple crypto prices với enhanced bulk optimization
        """
        if not symbols:
            return {}
            
        symbols = [s.upper() for s in symbols]
        results = {}
        
        logger.info(f"🔍 Fetching prices for {len(symbols)} symbols: {symbols}")

        # 1) Thử Binance bulk API trước
        if self._api_available('binance') and self._can_make_request('binance'):
            # Chia nhỏ nếu quá nhiều symbols
            for i in range(0, len(symbols), max_bulk_size):
                batch = symbols[i:i + max_bulk_size]
                bulk_data = self._get_bulk_prices_binance(batch)
                
                if bulk_data:
                    self._record_request('binance')
                    results.update(bulk_data)
                    logger.info(f"✅ Binance bulk returned {len(bulk_data)} prices")
                else:
                    self._trip_api('binance', 120)
                    break

        # 2) Lấy symbols còn thiếu từ các nguồn khác
        missing_symbols = [s for s in symbols if s not in results]
        if missing_symbols:
            logger.info(f"🔄 Fetching {len(missing_symbols)} missing symbols individually")
            
            for symbol in missing_symbols:
                price_data = self.get_crypto_price(symbol)
                if price_data:
                    results[symbol] = price_data

        logger.info(f"📊 Final results: {len(results)}/{len(symbols)} symbols retrieved")
        return results

    def get_top_cryptocurrencies(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Get top cryptocurrencies by market cap"""
        try:
            # Thử CoinGecko Pro trước
            if (self.api_keys.get('coingecko') and 
                self._api_available('coingecko_pro') and 
                self._can_make_request('coingecko_pro')):
                
                headers = {"x-cg-pro-api-key": self.api_keys['coingecko']}
                url = "https://pro-api.coingecko.com/api/v3/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': limit,
                    'page': 1,
                    'sparkline': False
                }
                
                response = self.session.get(url, headers=headers, params=params, timeout=(5, 15))
                response.raise_for_status()
                data = response.json()
                
                if data:
                    self._record_request('coingecko_pro')
                    logger.info(f"✅ Top cryptos from CoinGecko Pro: {len(data)} items")
                    return data

            # Fallback CoinGecko Free
            if self._api_available('coingecko_free') and self._can_make_request('coingecko_free'):
                url = "https://api.coingecko.com/api/v3/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': limit,
                    'page': 1,
                    'sparkline': False
                }
                
                response = self.session.get(url, params=params, timeout=(5, 15))
                response.raise_for_status()
                data = response.json()
                
                if data:
                    self._record_request('coingecko_free')
                    logger.info(f"✅ Top cryptos from CoinGecko Free: {len(data)} items")
                    return data

        except Exception as e:
            logger.error(f"❌ Error getting top cryptos: {e}")
            
        return None

    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID"""
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'SOL': 'solana',
            'XRP': 'ripple',
            'DOT': 'polkadot',
            'DOGE': 'dogecoin',
            'AVAX': 'avalanche-2',
            'SHIB': 'shiba-inu',
            'MATIC': 'matic-network',
            'LTC': 'litecoin',
            'UNI': 'uniswap',
            'LINK': 'chainlink',
            'ATOM': 'cosmos',
            'VET': 'vechain',
            'ICP': 'internet-computer',
            'FIL': 'filecoin',
            'TRX': 'tron',
            'ETC': 'ethereum-classic',
            'FTT': 'ftx-token',
            'XLM': 'stellar',
            'HBAR': 'hedera-hashgraph',
            'MANA': 'decentraland',
            'SAND': 'the-sandbox',
            'ALGO': 'algorand',
            'NEAR': 'near',
            'AXS': 'axie-infinity',
            'FLOW': 'flow',
            'XTZ': 'tezos'
        }
        return symbol_map.get(symbol.upper(), symbol.lower())

    def _symbol_to_coinpaprika_id(self, symbol: str) -> str:
        """Convert symbol to CoinPaprika ID"""
        symbol_map = {
            'BTC': 'btc-bitcoin',
            'ETH': 'eth-ethereum',
            'BNB': 'bnb-binance-coin',
            'ADA': 'ada-cardano',
            'SOL': 'sol-solana',
            'XRP': 'xrp-xrp',
            'DOT': 'dot-polkadot',
            'DOGE': 'doge-dogecoin',
            'AVAX': 'avax-avalanche',
            'SHIB': 'shib-shiba-inu',
            'MATIC': 'matic-polygon',
            'LTC': 'ltc-litecoin',
            'UNI': 'uni-uniswap',
            'LINK': 'link-chainlink',
            'ATOM': 'atom-cosmos',
            'VET': 'vet-vechain',
            'ICP': 'icp-internet-computer',
            'FIL': 'fil-filecoin',
            'TRX': 'trx-tron',
            'ETC': 'etc-ethereum-classic'
        }
        return symbol_map.get(symbol.upper(), f"{symbol.lower()}-{symbol.lower()}")

    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status and rate limits"""
        current_time = time.time()
        status = {}
        
        for api_name, limits in self.rate_limits.items():
            with self._rl_lock:
                recent_calls = [t for t in limits['calls'] if current_time - t < limits['period']]
                status[api_name] = {
                    'available': self._api_available(api_name),
                    'calls_made': len(recent_calls),
                    'max_calls': limits['max_calls'],
                    'calls_remaining': limits['max_calls'] - len(recent_calls),
                    'period': limits['period'],
                    'down_until': self._api_down_until.get(api_name, 0)
                }
        
        return status