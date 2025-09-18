# mongo_client.py
# MongoDB Production Client với connection pooling và error handling

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoMongoClient:
    """
    MongoDB Production Client cho crypto data
    - Connection pooling
    - Auto-reconnection
    - Optimized for high-frequency data
    - Supports all 34 coins
    """
    
    def __init__(self, connection_string: str = None, db_name: str = "crypto"):
        """
        Initialize MongoDB client với production settings
        
        Args:
            connection_string: MongoDB URI (mặc định local)
            db_name: Database name
        """
        # Default to local MongoDB
        self.connection_string = connection_string or "mongodb://localhost:27017/"
        self.db_name = db_name
        self.client = None
        self.db = None
        
        # Production settings
        self.max_pool_size = 100
        self.server_selection_timeout = 5000  # 5 seconds
        self.connect_timeout = 5000
        
        # Initialize connection
        self._connect()
        self._setup_collections()
    
    def _connect(self):
        """Thiết lập kết nối MongoDB với production settings"""
        try:
            self.client = MongoClient(
                self.connection_string,
                maxPoolSize=self.max_pool_size,
                serverSelectionTimeoutMS=self.server_selection_timeout,
                connectTimeoutMS=self.connect_timeout,
                socketTimeoutMS=30000,
                retryWrites=True
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            
            logger.info(f"✅ MongoDB connected: {self.db_name}")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
    
    def _setup_collections(self):
        """Tạo collections và indexes optimized cho crypto data"""
        try:
            # Collection cho realtime prices
            prices_collection = self.db.realtime_prices
            
            # Indexes cho performance
            prices_collection.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
            prices_collection.create_index([("timestamp", DESCENDING)])
            prices_collection.create_index([("symbol", ASCENDING)])
            
            # Collection cho historical OHLCV
            ohlcv_collection = self.db.ohlcv_data
            ohlcv_collection.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
            ohlcv_collection.create_index([("timestamp", DESCENDING)])
            
            # Collection cho ML features
            features_collection = self.db.ml_features
            features_collection.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
            
            # Collection cho bot alerts
            alerts_collection = self.db.price_alerts
            alerts_collection.create_index([("user_id", ASCENDING), ("symbol", ASCENDING)])
            alerts_collection.create_index([("active", ASCENDING)])
            
            # TTL index để auto-delete old data (30 days)
            prices_collection.create_index(
                [("timestamp", ASCENDING)], 
                expireAfterSeconds=30*24*60*60  # 30 days
            )
            
            logger.info("✅ MongoDB collections and indexes created")
            
        except Exception as e:
            logger.error(f"❌ Error setting up collections: {e}")
    
    def save_realtime_price(self, price_data: Dict[str, Any]) -> bool:
        """
        Lưu realtime price data với upsert
        
        Args:
            price_data: Dict chứa symbol, price, timestamp, etc.
        
        Returns:
            bool: Success status
        """
        try:
            collection = self.db.realtime_prices
            
            # Thêm metadata
            price_data.update({
                "created_at": datetime.utcnow(),
                "data_type": "realtime_price"
            })
            
            # Upsert based on symbol
            result = collection.replace_one(
                {"symbol": price_data["symbol"]},
                price_data,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving price data: {e}")
            return False
    
    def save_bulk_prices(self, prices_list: List[Dict[str, Any]]) -> bool:
        """
        Bulk save nhiều coins cùng lúc - optimized cho 34 coins
        
        Args:
            prices_list: List of price dicts
        
        Returns:
            bool: Success status
        """
        try:
            collection = self.db.realtime_prices
            
            # Prepare bulk operations
            operations = []
            
            for price_data in prices_list:
                price_data.update({
                    "created_at": datetime.utcnow(),
                    "data_type": "realtime_price"
                })
                
                operations.append({
                    "replaceOne": {
                        "filter": {"symbol": price_data["symbol"]},
                        "replacement": price_data,
                        "upsert": True
                    }
                })
            
            # Execute bulk write
            if operations:
                result = collection.bulk_write(operations, ordered=False)
                logger.info(f"✅ Bulk saved {len(operations)} price records")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in bulk save: {e}")
            return False
    
    def save_ohlcv_data(self, symbol: str, ohlcv_data: List[Dict]) -> bool:
        """
        Lưu OHLCV historical data
        
        Args:
            symbol: Crypto symbol
            ohlcv_data: List of OHLCV records
        
        Returns:
            bool: Success status
        """
        try:
            collection = self.db.ohlcv_data
            
            # Prepare data với metadata
            documents = []
            for record in ohlcv_data:
                doc = {
                    "symbol": symbol,
                    "timestamp": record.get("timestamp"),
                    "open": record.get("open"),
                    "high": record.get("high"),
                    "low": record.get("low"),
                    "close": record.get("close"),
                    "volume": record.get("volume"),
                    "created_at": datetime.utcnow(),
                    "data_type": "ohlcv"
                }
                documents.append(doc)
            
            if documents:
                collection.insert_many(documents, ordered=False)
                logger.info(f"✅ Saved {len(documents)} OHLCV records for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error saving OHLCV data: {e}")
            return False
    
    def save_ml_features(self, symbol: str, features_data: Dict[str, Any]) -> bool:
        """
        Lưu ML features cho symbol
        
        Args:
            symbol: Crypto symbol
            features_data: ML features dict
        
        Returns:
            bool: Success status
        """
        try:
            collection = self.db.ml_features
            
            features_data.update({
                "symbol": symbol,
                "created_at": datetime.utcnow(),
                "data_type": "ml_features"
            })
            
            # Upsert based on symbol và timestamp
            result = collection.replace_one(
                {
                    "symbol": symbol,
                    "timestamp": features_data.get("timestamp")
                },
                features_data,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving ML features: {e}")
            return False
    
    def get_latest_prices(self, symbols: List[str] = None) -> List[Dict]:
        """
        Lấy latest prices cho symbols
        
        Args:
            symbols: List symbols (None = all)
        
        Returns:
            List of price records
        """
        try:
            collection = self.db.realtime_prices
            
            query = {}
            if symbols:
                query["symbol"] = {"$in": symbols}
            
            cursor = collection.find(query).sort("timestamp", DESCENDING)
            return list(cursor)
            
        except Exception as e:
            logger.error(f"❌ Error getting latest prices: {e}")
            return []
    
    def get_price_history(self, symbol: str, hours: int = 24) -> List[Dict]:
        """
        Lấy price history cho symbol
        
        Args:
            symbol: Crypto symbol
            hours: Số giờ lịch sử
        
        Returns:
            List of historical records
        """
        try:
            collection = self.db.ohlcv_data
            
            since = datetime.utcnow() - timedelta(hours=hours)
            
            cursor = collection.find({
                "symbol": symbol,
                "timestamp": {"$gte": since}
            }).sort("timestamp", ASCENDING)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"❌ Error getting price history: {e}")
            return []
    
    def cleanup_old_data(self, days: int = 30):
        """
        Cleanup old data để tiết kiệm storage
        
        Args:
            days: Số ngày giữ lại
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Cleanup old OHLCV data
            result = self.db.ohlcv_data.delete_many({
                "created_at": {"$lt": cutoff}
            })
            
            logger.info(f"✅ Cleaned up {result.deleted_count} old OHLCV records")
            
        except Exception as e:
            logger.error(f"❌ Error in cleanup: {e}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get MongoDB connection info"""
        try:
            server_info = self.client.server_info()
            return {
                "connected": True,
                "database": self.db_name,
                "server_version": server_info.get("version"),
                "collections": self.db.list_collection_names()
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    def close(self):
        """Đóng connection"""
        if self.client:
            self.client.close()
            logger.info("📤 MongoDB connection closed")

# Global instance
mongo_client = None

def get_mongo_client() -> CryptoMongoClient:
    """Get global MongoDB client instance"""
    global mongo_client
    if mongo_client is None:
        mongo_client = CryptoMongoClient()
    return mongo_client

def close_mongo_client():
    """Close global MongoDB client"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        mongo_client = None

if __name__ == "__main__":
    # Test MongoDB connection
    print("🔗 Testing MongoDB Connection...")
    
    try:
        client = CryptoMongoClient()
        info = client.get_connection_info()
        
        print(f"✅ Connection Status: {info}")
        
        # Test save price
        test_price = {
            "symbol": "BTCUSDT",
            "price_usd": 50000.0,
            "timestamp": datetime.utcnow(),
            "volume_24h": 1000000
        }
        
        success = client.save_realtime_price(test_price)
        print(f"💾 Save Test: {'✅' if success else '❌'}")
        
        # Test get prices
        prices = client.get_latest_prices(["BTCUSDT"])
        print(f"📊 Read Test: {len(prices)} records found")
        
        client.close()
        
    except Exception as e:
        print(f"❌ MongoDB Test Failed: {e}")