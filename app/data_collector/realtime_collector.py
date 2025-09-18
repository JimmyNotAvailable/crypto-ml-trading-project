#!/usr/bin/env python3
"""
📡 REALTIME CRYPTO DATA COLLECTOR
================================

Thu thập dữ liệu crypto realtime từ APIs:
- 🎯 Binance API cho giá OHLCV
- 🌐 CoinGecko API cho thông tin coin
- 💱 Tỷ giá USD/VND
- 🔧 Feature engineering tương tự training data
- 💾 Lưu trữ JSON/CSV (chưa cần MongoDB)
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataCollector:
    """
    📡 Thu thập dữ liệu crypto realtime
    
    Features:
    - Multiple API sources
    - Data validation
    - Feature engineering
    - File-based storage
    - Error handling
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize Data Collector
        
        Args:
            data_dir: Thư mục lưu dữ liệu (mặc định: project/data/realtime)
        """
        # Setup data directory
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / "data" / "realtime"
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.binance_api = "https://api.binance.com/api/v3"
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # Supported symbols
        self.supported_symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 
            'XRPUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT',
            'BCHUSDT', 'XLMUSDT'
        ]
        
        # USD to VND rate (sẽ update realtime)
        self.usd_vnd_rate = 24000  # Default rate
        
        logger.info(f"📡 CryptoDataCollector initialized")
        logger.info(f"💾 Data directory: {self.data_dir}")
    
    def get_usd_vnd_rate(self) -> float:
        """
        🔄 Lấy tỷ giá USD/VND realtime
        
        Returns:
            Tỷ giá USD/VND
        """
        try:
            # API tỷ giá miễn phí
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                rate = data['rates'].get('VND', 24000)
                self.usd_vnd_rate = rate
                logger.info(f"💱 USD/VND rate updated: {rate:,.0f}")
                return rate
            else:
                logger.warning(f"⚠️ Exchange rate API failed, using cached rate: {self.usd_vnd_rate}")
                return self.usd_vnd_rate
                
        except Exception as e:
            logger.error(f"❌ Error getting USD/VND rate: {str(e)}")
            return self.usd_vnd_rate
    
    def get_binance_price(self, symbol: str) -> Dict:
        """
        📊 Lấy giá từ Binance API
        
        Args:
            symbol: Trading pair (VD: 'BTCUSDT')
            
        Returns:
            Dict chứa thông tin giá
        """
        try:
            # Current price
            price_url = f"{self.binance_api}/ticker/price"
            price_response = requests.get(price_url, params={'symbol': symbol}, timeout=10)
            
            # 24h stats
            stats_url = f"{self.binance_api}/ticker/24hr"
            stats_response = requests.get(stats_url, params={'symbol': symbol}, timeout=10)
            
            if price_response.status_code == 200 and stats_response.status_code == 200:
                price_data = price_response.json()
                stats_data = stats_response.json()
                
                result = {
                    'symbol': symbol,
                    'price_usd': float(price_data['price']),
                    'price_vnd': float(price_data['price']) * self.usd_vnd_rate,
                    'volume_24h': float(stats_data['volume']),
                    'change_24h_percent': float(stats_data['priceChangePercent']),
                    'change_24h_usd': float(stats_data['priceChange']),
                    'high_24h': float(stats_data['highPrice']),
                    'low_24h': float(stats_data['lowPrice']),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'binance'
                }
                
                logger.info(f"✅ {symbol}: ${result['price_usd']:,.2f} ({result['change_24h_percent']:+.2f}%)")
                return result
            else:
                raise Exception(f"API response error: {price_response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error getting Binance price for {symbol}: {str(e)}")
            return None
    
    def get_historical_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[Dict]:
        """
        📈 Lấy dữ liệu OHLCV lịch sử từ Binance
        
        Args:
            symbol: Trading pair
            interval: Khung thời gian ('1h', '4h', '1d')
            limit: Số lượng records
            
        Returns:
            List các điểm dữ liệu OHLCV
        """
        try:
            url = f"{self.binance_api}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                klines = response.json()
                
                result = []
                for kline in klines:
                    data_point = {
                        'symbol': symbol,
                        'timestamp': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'interval': interval
                    }
                    result.append(data_point)
                
                logger.info(f"📈 Got {len(result)} {interval} klines for {symbol}")
                return result
            else:
                raise Exception(f"API response error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error getting klines for {symbol}: {str(e)}")
            return []
    
    def calculate_technical_indicators(self, prices: List[float]) -> Dict:
        """
        📊 Tính các chỉ báo kỹ thuật (tương tự training data)
        
        Args:
            prices: List giá close
            
        Returns:
            Dict chứa các chỉ báo
        """
        if len(prices) < 50:
            logger.warning(f"⚠️ Not enough data for indicators: {len(prices)} points")
            return {}
        
        prices_array = np.array(prices)
        
        try:
            indicators = {
                'ma_10': np.mean(prices_array[-10:]) if len(prices_array) >= 10 else prices_array[-1],
                'ma_50': np.mean(prices_array[-50:]) if len(prices_array) >= 50 else prices_array[-1],
                'volatility': np.std(prices_array[-20:]) if len(prices_array) >= 20 else 0,
                'returns': (prices_array[-1] - prices_array[-2]) / prices_array[-2] if len(prices_array) >= 2 else 0,
                'price_change_1h': (prices_array[-1] - prices_array[-2]) if len(prices_array) >= 2 else 0,
                'price_position': (prices_array[-1] - np.min(prices_array[-20:])) / (np.max(prices_array[-20:]) - np.min(prices_array[-20:])) if len(prices_array) >= 20 else 0.5
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {str(e)}")
            return {}
    
    def prepare_ml_features(self, symbol: str) -> Optional[Dict]:
        """
        🔧 Chuẩn bị features cho ML prediction (tương tự training data)
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict chứa features ready cho ML
        """
        try:
            # Lấy dữ liệu hiện tại
            current_data = self.get_binance_price(symbol)
            if not current_data:
                return None
            
            # Lấy dữ liệu lịch sử để tính indicators
            historical_data = self.get_historical_klines(symbol, '1h', 100)
            if not historical_data:
                return None
            
            # Trích xuất giá close
            close_prices = [point['close'] for point in historical_data]
            close_prices.append(current_data['price_usd'])  # Thêm giá hiện tại
            
            # Tính indicators
            indicators = self.calculate_technical_indicators(close_prices)
            
            # Features tương tự training data
            features = {
                'symbol': symbol,
                'timestamp': current_data['timestamp'],
                'close': current_data['price_usd'],
                'volume': current_data['volume_24h'],
                'high': current_data['high_24h'],
                'low': current_data['low_24h'],
                'ma_10': indicators.get('ma_10', current_data['price_usd']),
                'ma_50': indicators.get('ma_50', current_data['price_usd']),
                'volatility': indicators.get('volatility', 0),
                'returns': indicators.get('returns', 0),
                'price_change_1h': indicators.get('price_change_1h', 0),
                'hour': datetime.now().hour,
                'change_24h_percent': current_data['change_24h_percent'],
                'price_vnd': current_data['price_vnd'],
                'usd_vnd_rate': self.usd_vnd_rate
            }
            
            logger.info(f"🔧 Prepared ML features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error preparing features for {symbol}: {str(e)}")
            return None
    
    def save_to_file(self, data: Dict, filename: str = None):
        """
        💾 Lưu dữ liệu vào file JSON
        
        Args:
            data: Dữ liệu cần lưu
            filename: Tên file (mặc định: symbol_timestamp.json)
        """
        try:
            if filename is None:
                symbol = data.get('symbol', 'unknown')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol}_{timestamp}.json"
            
            filepath = self.data_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved data to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving to file: {str(e)}")
    
    def collect_all_symbols(self) -> Dict:
        """
        🎯 Thu thập dữ liệu cho tất cả symbols
        
        Returns:
            Dict chứa dữ liệu tất cả symbols
        """
        logger.info("🎯 Starting data collection for all symbols...")
        
        # Update USD/VND rate
        self.get_usd_vnd_rate()
        
        all_data = {
            'timestamp': datetime.now().isoformat(),
            'usd_vnd_rate': self.usd_vnd_rate,
            'symbols': {}
        }
        
        success_count = 0
        
        for symbol in self.supported_symbols:
            logger.info(f"📊 Collecting {symbol}...")
            
            # Thu thập dữ liệu cơ bản
            price_data = self.get_binance_price(symbol)
            if price_data:
                all_data['symbols'][symbol] = price_data
                success_count += 1
                
                # Thu thập features cho ML
                ml_features = self.prepare_ml_features(symbol)
                if ml_features:
                    all_data['symbols'][symbol]['ml_features'] = ml_features
            
            # Delay để tránh rate limit
            time.sleep(0.5)
        
        logger.info(f"✅ Collected data for {success_count}/{len(self.supported_symbols)} symbols")
        
        # Lưu file tổng hợp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_to_file(all_data, f"all_symbols_{timestamp}.json")
        
        # Lưu file latest (cho bot sử dụng)
        self.save_to_file(all_data, "latest_prices.json")
        
        return all_data
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """
        📱 Lấy giá mới nhất của 1 symbol (cho bot sử dụng)
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict chứa thông tin giá mới nhất
        """
        try:
            # Kiểm tra file latest trước
            latest_file = self.data_dir / "latest_prices.json"
            
            if latest_file.exists():
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Kiểm tra timestamp (dữ liệu không quá 5 phút)
                file_time = datetime.fromisoformat(data['timestamp'])
                if (datetime.now() - file_time).seconds < 300:
                    symbol_data = data['symbols'].get(symbol)
                    if symbol_data:
                        logger.info(f"📱 Using cached price for {symbol}")
                        return symbol_data
            
            # Nếu không có cache hoặc quá cũ, thu thập mới
            logger.info(f"📡 Fetching fresh price for {symbol}")
            self.get_usd_vnd_rate()
            
            price_data = self.get_binance_price(symbol)
            if price_data:
                ml_features = self.prepare_ml_features(symbol)
                if ml_features:
                    price_data['ml_features'] = ml_features
                
                return price_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting latest price for {symbol}: {str(e)}")
            return None

def demo_data_collection():
    """Demo thu thập dữ liệu"""
    print("📡 DEMO CRYPTO DATA COLLECTOR")
    print("=" * 50)
    
    collector = CryptoDataCollector()
    
    # Test thu thập 1 symbol
    print("\n🎯 Testing single symbol collection:")
    btc_data = collector.get_latest_price('BTCUSDT')
    if btc_data:
        print(f"✅ BTC Price: ${btc_data['price_usd']:,.2f} ({btc_data['change_24h_percent']:+.2f}%)")
        print(f"💰 BTC VND: {btc_data['price_vnd']:,.0f} VND")
    
    # Test thu thập tất cả symbols
    print(f"\n🎪 Testing all symbols collection:")
    all_data = collector.collect_all_symbols()
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Collected: {len(all_data['symbols'])} symbols")
    print(f"💱 USD/VND Rate: {all_data['usd_vnd_rate']:,.0f}")
    print(f"📁 Data saved to: {collector.data_dir}")

if __name__ == "__main__":
    demo_data_collection()