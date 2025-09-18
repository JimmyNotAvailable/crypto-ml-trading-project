# production_config.py
# Production configuration management

import os
from dataclasses import dataclass
from typing import Dict, Any, List
import json

@dataclass
class DataCollectionConfig:
    """Configuration for data collection"""
    symbols: List[str]
    collection_interval: int  # seconds
    use_mongodb: bool
    mongodb_uri: str
    data_directory: str
    max_file_age_hours: int
    backup_enabled: bool
    
    @classmethod
    def production(cls):
        """Production configuration"""
        return cls(
            symbols=[
                "1INCHUSDT", "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ATOMUSDT", 
                "AVAXUSDT", "BALUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
                "COMPUSDT", "CRVUSDT", "DENTUSDT", "DOGEUSDT", "DOTUSDT",
                "DYDXUSDT", "ETCUSDT", "ETHUSDT", "FILUSDT", "HBARUSDT",
                "ICPUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "MKRUSDT",
                "RVNUSDT", "SHIBUSDT", "SOLUSDT", "SUSHIUSDT", "TRXUSDT",
                "UNIUSDT", "VETUSDT", "XLMUSDT", "XMRUSDT"
            ],
            collection_interval=60,
            use_mongodb=True,
            mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
            data_directory="data/realtime_production",
            max_file_age_hours=48,
            backup_enabled=True
        )
    
    @classmethod
    def development(cls):
        """Development configuration"""
        return cls(
            symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            collection_interval=30,
            use_mongodb=False,
            mongodb_uri="mongodb://localhost:27017/",
            data_directory="data/realtime_dev",
            max_file_age_hours=24,
            backup_enabled=False
        )

@dataclass
class MonitoringConfig:
    """Configuration for monitoring"""
    health_check_interval: int  # seconds
    alert_thresholds: Dict[str, Any]
    log_level: str
    metrics_enabled: bool
    
    @classmethod
    def production(cls):
        """Production monitoring config"""
        return cls(
            health_check_interval=300,  # 5 minutes
            alert_thresholds={
                "data_age_minutes": 10,
                "success_rate_percent": 95,
                "cpu_percent": 80,
                "memory_percent": 85,
                "api_response_time_ms": 5000
            },
            log_level="INFO",
            metrics_enabled=True
        )

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.data_config = self._get_data_config()
        self.monitoring_config = self._get_monitoring_config()
    
    def _get_data_config(self) -> DataCollectionConfig:
        """Get data collection configuration"""
        if self.environment == "production":
            return DataCollectionConfig.production()
        else:
            return DataCollectionConfig.development()
    
    def _get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig.production()
    
    def save_config(self, file_path: str):
        """Save configuration to file"""
        config_data = {
            "environment": self.environment,
            "data_collection": {
                "symbols": self.data_config.symbols,
                "collection_interval": self.data_config.collection_interval,
                "use_mongodb": self.data_config.use_mongodb,
                "mongodb_uri": self.data_config.mongodb_uri,
                "data_directory": self.data_config.data_directory,
                "max_file_age_hours": self.data_config.max_file_age_hours,
                "backup_enabled": self.data_config.backup_enabled
            },
            "monitoring": {
                "health_check_interval": self.monitoring_config.health_check_interval,
                "alert_thresholds": self.monitoring_config.alert_thresholds,
                "log_level": self.monitoring_config.log_level,
                "metrics_enabled": self.monitoring_config.metrics_enabled
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def load_config(cls, file_path: str):
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Create instance with loaded data
        manager = cls(config_data["environment"])
        
        # Override with loaded values
        dc = config_data["data_collection"]
        manager.data_config = DataCollectionConfig(
            symbols=dc["symbols"],
            collection_interval=dc["collection_interval"],
            use_mongodb=dc["use_mongodb"],
            mongodb_uri=dc["mongodb_uri"],
            data_directory=dc["data_directory"],
            max_file_age_hours=dc["max_file_age_hours"],
            backup_enabled=dc["backup_enabled"]
        )
        
        mc = config_data["monitoring"]
        manager.monitoring_config = MonitoringConfig(
            health_check_interval=mc["health_check_interval"],
            alert_thresholds=mc["alert_thresholds"],
            log_level=mc["log_level"],
            metrics_enabled=mc["metrics_enabled"]
        )
        
        return manager

# Global config instance
_config_manager = None

def get_config(environment: str = "production") -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(environment)
    return _config_manager

if __name__ == "__main__":
    # Create and save production config
    config = ConfigManager("production")
    config.save_config("config/production.json")
    
    print("✅ Production configuration saved")
    print(f"📊 Symbols: {len(config.data_config.symbols)}")
    print(f"⏰ Interval: {config.data_config.collection_interval}s")
    print(f"🗄️ MongoDB: {config.data_config.use_mongodb}")
    print(f"📁 Directory: {config.data_config.data_directory}")