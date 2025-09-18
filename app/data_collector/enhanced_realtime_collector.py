# Enhanced realtime_collector.py với MongoDB và 34 coins
# Continuous operation cho 12 giờ với proper error handling

import asyncio
import time
import logging
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import signal
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from database.mongo_client import get_mongo_client, CryptoMongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("MongoDB not available, falling back to file storage")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CollectionStats:
    """Statistics cho data collection"""
    total_symbols: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    start_time: datetime = None
    last_collection_time: datetime = None
    collections_per_minute: float = 0.0
    avg_response_time: float = 0.0

class EnhancedCryptoDataCollector:
    """
    Enhanced Data Collector cho production
    - Supports all 34 coins from training dataset
    - MongoDB integration with fallback to files
    - Continuous operation capability (12h)
    - Rate limiting và error handling
    - Real-time monitoring
    """
    
    # All 34 coins from training dataset
    ALL_SYMBOLS = [
        "1INCHUSDT", "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ATOMUSDT", 
        "AVAXUSDT", "BALUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
        "COMPUSDT", "CRVUSDT", "DENTUSDT", "DOGEUSDT", "DOTUSDT",
        "DYDXUSDT", "ETCUSDT", "ETHUSDT", "FILUSDT", "HBARUSDT",
        "ICPUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "MKRUSDT",
        "RVNUSDT", "SHIBUSDT", "SOLUSDT", "SUSHIUSDT", "TRXUSDT",
        "UNIUSDT", "VETUSDT", "XLMUSDT", "XMRUSDT"
    ]
    
    def __init__(self, 
                 symbols: List[str] = None,
                 collection_interval: int = 60,  # seconds
                 use_mongodb: bool = True,
                 data_dir: str = None):
        """
        Initialize Enhanced Data Collector
        
        Args:
            symbols: List of symbols (None = all 34)
            collection_interval: Seconds between collections
            use_mongodb: Whether to use MongoDB
            data_dir: Directory for file backup
        """
        # Convert symbols if needed (BTC -> BTCUSDT)
        if symbols:
            converted_symbols = []
            for symbol in symbols:
                if symbol.endswith('USDT'):
                    converted_symbols.append(symbol)
                else:
                    converted_symbols.append(f"{symbol}USDT")
            self.symbols = converted_symbols
        else:
            self.symbols = self.ALL_SYMBOLS
        self.collection_interval = collection_interval
        self.use_mongodb = use_mongodb and MONGODB_AVAILABLE
        
        # Setup directories
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "data", "realtime_production"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        # MongoDB client
        self.mongo_client = None
        if self.use_mongodb:
            try:
                self.mongo_client = get_mongo_client()
                logger.info("✅ MongoDB client initialized")
            except Exception as e:
                logger.warning(f"MongoDB initialization failed: {e}")
                self.use_mongodb = False
        
        # API settings
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoBot/1.0'
        })
        
        # Rate limiting
        self.max_requests_per_minute = 1000  # Binance limit
        self.request_count = 0
        self.last_reset_time = time.time()
        
        # Statistics tracking
        self.stats = CollectionStats()
        self.stats.total_symbols = len(self.symbols)
        
        # Control flags
        self.running = False
        self.stop_requested = False
        
        # USD/VND exchange rate cache
        self.usd_vnd_rate = 24000  # Default fallback
        self.last_rate_update = None
        
        logger.info(f"🚀 Enhanced Data Collector initialized")
        logger.info(f"📊 Symbols: {len(self.symbols)}")
        logger.info(f"🗄️ MongoDB: {'✅' if self.use_mongodb else '❌'}")
        logger.info(f"📁 Data dir: {self.data_dir}")
    
    def _check_rate_limit(self):
        """Rate limiting check"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_reset_time >= 60:
            self.request_count = 0
            self.last_reset_time = current_time
        
        # Check if we're approaching limit
        if self.request_count >= self.max_requests_per_minute * 0.8:  # 80% of limit
            sleep_time = 60 - (current_time - self.last_reset_time)
            if sleep_time > 0:
                logger.warning(f"⏳ Rate limit approaching, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_reset_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make API request với error handling và rate limiting"""
        self._check_rate_limit()
        
        try:
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=10)
            response_time = time.time() - start_time
            
            # Update stats
            self.request_count += 1
            if self.stats.avg_response_time == 0:
                self.stats.avg_response_time = response_time
            else:
                self.stats.avg_response_time = (self.stats.avg_response_time * 0.9 + 
                                               response_time * 0.1)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def update_usd_vnd_rate(self) -> bool:
        """Update USD/VND exchange rate"""
        try:
            # Update only if older than 1 hour
            if (self.last_rate_update and 
                datetime.now() - self.last_rate_update < timedelta(hours=1)):
                return True
            
            # Try exchangerate-api.com
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            data = self._make_request(url)
            
            if data and 'rates' in data and 'VND' in data['rates']:
                self.usd_vnd_rate = data['rates']['VND']
                self.last_rate_update = datetime.now()
                logger.info(f"💱 USD/VND rate updated: {self.usd_vnd_rate:,.0f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update USD/VND rate: {e}")
            return False
    
    def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price data for symbol"""
        try:
            # 24hr ticker stats
            ticker_url = f"{self.base_url}/ticker/24hr"
            ticker_data = self._make_request(ticker_url, {"symbol": symbol})
            
            if not ticker_data:
                return None
            
            # Current price
            price_url = f"{self.base_url}/ticker/price"
            price_data = self._make_request(price_url, {"symbol": symbol})
            
            if not price_data:
                return None
            
            # Combine data
            current_price = float(price_data["price"])
            
            return {
                "symbol": symbol,
                "price_usd": current_price,
                "price_vnd": current_price * self.usd_vnd_rate,
                "volume_24h": float(ticker_data["volume"]),
                "change_24h_percent": float(ticker_data["priceChangePercent"]),
                "change_24h_usd": float(ticker_data["priceChange"]),
                "high_24h": float(ticker_data["highPrice"]),
                "low_24h": float(ticker_data["lowPrice"]),
                "timestamp": datetime.utcnow(),
                "source": "binance",
                "usd_vnd_rate": self.usd_vnd_rate
            }
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_kline_data(self, symbol: str, limit: int = 100) -> Optional[List[Dict]]:
        """Get kline/candlestick data for technical analysis"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": "1h",
                "limit": limit
            }
            
            data = self._make_request(url, params)
            if not data:
                return None
            
            klines = []
            for kline in data:
                klines.append({
                    "timestamp": datetime.fromtimestamp(kline[0] / 1000),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5])
                })
            
            return klines
            
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, klines: List[Dict]) -> Dict[str, float]:
        """Calculate technical indicators from kline data"""
        try:
            if not klines or len(klines) < 50:
                return {}
            
            # Convert to pandas for easier calculation
            df = pd.DataFrame(klines)
            df = df.sort_values('timestamp')
            
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            indicators['ma_10'] = df['close'].rolling(10).mean().iloc[-1]
            indicators['ma_50'] = df['close'].rolling(50).mean().iloc[-1]
            
            # Volatility (standard deviation of returns)
            returns = df['close'].pct_change()
            indicators['volatility'] = returns.rolling(24).std().iloc[-1] * 100
            
            # Price change metrics
            indicators['returns'] = returns.iloc[-1] * 100 if len(returns) > 1 else 0
            indicators['price_change_1h'] = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / 
                                           df['close'].iloc[-2] * 100) if len(df) > 1 else 0
            
            # Current hour for time features
            indicators['hour'] = datetime.now().hour
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def prepare_ml_features(self, price_data: Dict, klines: List[Dict]) -> Dict[str, Any]:
        """Prepare ML-ready features"""
        try:
            # Technical indicators
            tech_indicators = self.calculate_technical_indicators(klines)
            
            # Combine all features
            ml_features = {
                "symbol": price_data["symbol"],
                "timestamp": price_data["timestamp"],
                "close": price_data["price_usd"],
                "volume": price_data["volume_24h"],
                "high": price_data["high_24h"],
                "low": price_data["low_24h"],
                "change_24h_percent": price_data["change_24h_percent"],
                "price_vnd": price_data["price_vnd"],
                "usd_vnd_rate": price_data["usd_vnd_rate"]
            }
            
            # Add technical indicators
            ml_features.update(tech_indicators)
            
            return ml_features
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return {}
    
    def collect_single_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect complete data for single symbol"""
        try:
            logger.info(f"📊 Collecting {symbol}...")
            
            # Get price data
            price_data = self.get_price_data(symbol)
            if not price_data:
                self.stats.failed_collections += 1
                return None
            
            # Get kline data for technical analysis
            klines = self.get_kline_data(symbol)
            if not klines:
                logger.warning(f"No klines for {symbol}, using price data only")
                klines = []
            
            # Prepare ML features
            ml_features = self.prepare_ml_features(price_data, klines)
            
            # Complete dataset
            complete_data = {
                "symbol": symbol,
                "price_data": price_data,
                "ml_features": ml_features,
                "klines_count": len(klines),
                "collection_timestamp": datetime.utcnow(),
                "status": "success"
            }
            
            self.stats.successful_collections += 1
            logger.info(f"✅ {symbol}: ${price_data['price_usd']:,.2f} "
                       f"({price_data['change_24h_percent']:+.2f}%)")
            
            return complete_data
            
        except Exception as e:
            logger.error(f"Error collecting {symbol}: {e}")
            self.stats.failed_collections += 1
            return None
    
    def save_data(self, data_batch: List[Dict[str, Any]]) -> bool:
        """Save collected data to MongoDB and/or files"""
        try:
            success = True
            
            # Save to MongoDB if available
            if self.use_mongodb and self.mongo_client:
                try:
                    # Prepare for bulk MongoDB operations
                    price_records = []
                    ml_features_records = []
                    
                    for data in data_batch:
                        if data and data.get("status") == "success":
                            # Price data
                            price_records.append(data["price_data"])
                            
                            # ML features
                            if data.get("ml_features"):
                                ml_features_records.append(data["ml_features"])
                    
                    # Bulk save prices
                    if price_records:
                        mongo_success = self.mongo_client.save_bulk_prices(price_records)
                        if not mongo_success:
                            logger.warning("MongoDB bulk save failed")
                            success = False
                    
                    # Save ML features
                    for features in ml_features_records:
                        if features:
                            self.mongo_client.save_ml_features(
                                features["symbol"], features
                            )
                    
                    logger.info(f"💾 MongoDB: Saved {len(price_records)} price records")
                    
                except Exception as e:
                    logger.error(f"MongoDB save error: {e}")
                    success = False
            
            # Always save to files as backup
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save complete batch
                batch_file = os.path.join(
                    self.data_dir, f"collection_batch_{timestamp}.json"
                )
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(data_batch, f, indent=2, default=str)
                
                # Save latest prices summary
                latest_file = os.path.join(self.data_dir, "latest_prices.json")
                summary = {
                    "timestamp": datetime.utcnow(),
                    "usd_vnd_rate": self.usd_vnd_rate,
                    "total_symbols": len(data_batch),
                    "successful": len([d for d in data_batch if d and d.get("status") == "success"]),
                    "symbols": {}
                }
                
                for data in data_batch:
                    if data and data.get("status") == "success":
                        price_data = data["price_data"]
                        summary["symbols"][price_data["symbol"]] = {
                            "price_usd": price_data["price_usd"],
                            "price_vnd": price_data["price_vnd"],
                            "change_24h_percent": price_data["change_24h_percent"],
                            "timestamp": price_data["timestamp"]
                        }
                
                with open(latest_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                logger.info(f"📁 Files: Saved to {batch_file}")
                
            except Exception as e:
                logger.error(f"File save error: {e}")
                success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False
    
    def collect_all_symbols(self) -> List[Dict[str, Any]]:
        """Collect data for all symbols in batch"""
        logger.info(f"🎯 Starting collection for {len(self.symbols)} symbols...")
        
        # Update exchange rate
        self.update_usd_vnd_rate()
        
        batch_results = []
        
        for i, symbol in enumerate(self.symbols):
            if self.stop_requested:
                logger.info("🛑 Stop requested, breaking collection loop")
                break
            
            # Collect data for symbol
            result = self.collect_single_symbol(symbol)
            batch_results.append(result)
            
            # Progress reporting
            if (i + 1) % 10 == 0:
                logger.info(f"📈 Progress: {i + 1}/{len(self.symbols)} symbols collected")
            
            # Rate limiting between symbols
            time.sleep(0.1)  # 100ms between symbols
        
        # Update statistics
        self.stats.last_collection_time = datetime.utcnow()
        
        return batch_results
    
    def run_continuous(self, duration_hours: float = 12.0):
        """
        Run continuous data collection for specified duration
        
        Args:
            duration_hours: How long to run (default 12 hours)
        """
        logger.info(f"🚀 Starting continuous collection for {duration_hours} hours")
        
        self.running = True
        self.stop_requested = False
        self.stats.start_time = datetime.utcnow()
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        collection_count = 0
        
        try:
            while self.running and datetime.utcnow() < end_time:
                if self.stop_requested:
                    break
                
                collection_start = time.time()
                collection_count += 1
                
                logger.info(f"🔄 Collection #{collection_count} - "
                           f"{datetime.utcnow().strftime('%H:%M:%S')}")
                
                # Collect all symbols
                batch_results = self.collect_all_symbols()
                
                # Save data
                save_success = self.save_data(batch_results)
                
                # Calculate metrics
                collection_time = time.time() - collection_start
                successful = len([r for r in batch_results if r and r.get("status") == "success"])
                
                # Update stats
                elapsed_minutes = (datetime.utcnow() - self.stats.start_time).total_seconds() / 60
                self.stats.collections_per_minute = collection_count / max(elapsed_minutes, 1)
                
                # Log summary
                logger.info(f"✅ Collection #{collection_count} complete:")
                logger.info(f"   📊 Success: {successful}/{len(self.symbols)} symbols")
                logger.info(f"   ⏱️  Time: {collection_time:.1f}s")
                logger.info(f"   💾 Save: {'✅' if save_success else '❌'}")
                logger.info(f"   📈 Rate: {self.stats.collections_per_minute:.2f}/min")
                
                # Calculate sleep time
                sleep_time = max(0, self.collection_interval - collection_time)
                if sleep_time > 0:
                    logger.info(f"😴 Sleeping {sleep_time:.1f}s until next collection")
                    time.sleep(sleep_time)
                
                # Status check every 10 collections
                if collection_count % 10 == 0:
                    self.print_status()
        
        except KeyboardInterrupt:
            logger.info("⏹️  Keyboard interrupt received")
            self.stop_requested = True
        
        except Exception as e:
            logger.error(f"❌ Error in continuous operation: {e}")
        
        finally:
            self.running = False
            logger.info(f"🏁 Continuous collection ended after {collection_count} cycles")
            self.print_final_stats()
    
    def print_status(self):
        """Print current status and statistics"""
        if not self.stats.start_time:
            return
        
        elapsed = datetime.utcnow() - self.stats.start_time
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        success_rate = (self.stats.successful_collections / 
                       max(self.stats.total_symbols, 1) * 100)
        
        print("\n" + "="*60)
        print("📊 COLLECTION STATUS")
        print("="*60)
        print(f"⏰ Running Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"🎯 Symbols: {len(self.symbols)}")
        print(f"✅ Successful: {self.stats.successful_collections}")
        print(f"❌ Failed: {self.stats.failed_collections}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⚡ Collections/min: {self.stats.collections_per_minute:.2f}")
        print(f"🔗 Avg Response: {self.stats.avg_response_time*1000:.0f}ms")
        print(f"💱 USD/VND Rate: {self.usd_vnd_rate:,.0f}")
        print(f"🗄️ MongoDB: {'✅' if self.use_mongodb else '❌'}")
        print("="*60)
    
    def print_final_stats(self):
        """Print final statistics"""
        if not self.stats.start_time:
            return
        
        total_time = datetime.utcnow() - self.stats.start_time
        
        print("\n" + "="*60)
        print("🏁 FINAL COLLECTION STATISTICS")
        print("="*60)
        print(f"⏰ Total Runtime: {total_time}")
        print(f"🎯 Total Symbols: {len(self.symbols)}")
        print(f"✅ Successful Collections: {self.stats.successful_collections}")
        print(f"❌ Failed Collections: {self.stats.failed_collections}")
        print(f"📈 Overall Success Rate: {(self.stats.successful_collections / max(self.stats.successful_collections + self.stats.failed_collections, 1) * 100):.1f}%")
        print(f"⚡ Average Collections/min: {self.stats.collections_per_minute:.2f}")
        print(f"📁 Data Directory: {self.data_dir}")
        print("="*60)
    
    def stop(self):
        """Request stop of continuous operation"""
        logger.info("🛑 Stop requested")
        self.stop_requested = True
        self.running = False

def setup_signal_handlers(collector):
    """Setup graceful shutdown on signals"""
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} received, initiating graceful shutdown...")
        collector.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Production data collection cho 34 coins
    print("🚀 ENHANCED CRYPTO DATA COLLECTOR")
    print("="*80)
    print(f"📊 Target: {len(EnhancedCryptoDataCollector.ALL_SYMBOLS)} symbols")
    print(f"⏰ Continuous operation capability: 12 hours")
    print(f"🗄️ MongoDB production storage")
    print("="*80)
    
    # Initialize collector
    collector = EnhancedCryptoDataCollector(
        symbols=None,  # All 34 symbols
        collection_interval=60,  # 1 minute intervals
        use_mongodb=True
    )
    
    # Setup signal handlers
    setup_signal_handlers(collector)
    
    # Demo run - collect once
    print("\n🎯 Demo Collection (single batch):")
    demo_results = collector.collect_all_symbols()
    collector.save_data(demo_results)
    collector.print_status()
    
    # Ask user for continuous operation
    print(f"\n🤖 Ready for continuous operation!")
    print(f"💡 To run for 12 hours: collector.run_continuous(12.0)")
    print(f"💡 To run for 1 hour test: collector.run_continuous(1.0)")
    print(f"💡 Use Ctrl+C to stop gracefully")
    
    # Uncomment next line to start continuous operation
    # collector.run_continuous(12.0)