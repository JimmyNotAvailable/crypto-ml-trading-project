"""
📦 DATA COLLECTOR PACKAGE
========================

Thu thập dữ liệu crypto realtime:
- 📡 Realtime price collection
- 🔧 Feature engineering 
- 💾 File-based storage
- 🎯 ML-ready data preparation
"""

from .realtime_collector import CryptoDataCollector

__all__ = ['CryptoDataCollector']