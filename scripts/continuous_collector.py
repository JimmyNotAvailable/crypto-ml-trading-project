# continuous_collector.py
# Script để chạy data collection liên tục 12 giờ

import sys
import os
import argparse
import logging
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app'))

from data_collector.enhanced_realtime_collector import EnhancedCryptoDataCollector, setup_signal_handlers  # type: ignore

def main():
    """Main function để chạy continuous collection"""
    parser = argparse.ArgumentParser(description='Continuous Crypto Data Collector')
    parser.add_argument('--hours', type=float, default=12.0, 
                       help='Duration in hours (default: 12.0)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Collection interval in seconds (default: 60)')
    parser.add_argument('--symbols', nargs='+', 
                       help='Specific symbols to collect (default: all 34)')
    parser.add_argument('--mongodb', action='store_true',
                       help='Enable MongoDB storage')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'crypto_collector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Print startup info
    print("🚀 CONTINUOUS CRYPTO DATA COLLECTOR")
    print("="*80)
    print(f"⏰ Duration: {args.hours} hours")
    print(f"🔄 Interval: {args.interval} seconds")
    print(f"📊 Symbols: {len(args.symbols) if args.symbols else 34}")
    print(f"🗄️ MongoDB: {'✅' if args.mongodb else '❌'}")
    print(f"📝 Log Level: {args.log_level}")
    print("="*80)
    
    # Initialize collector
    collector = EnhancedCryptoDataCollector(
        symbols=args.symbols,
        collection_interval=args.interval,
        use_mongodb=args.mongodb
    )
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(collector)
    
    try:
        # Start continuous collection
        logger.info(f"Starting continuous collection for {args.hours} hours")
        collector.run_continuous(args.hours)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        collector.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        collector.stop()
        raise
    
    logger.info("Collection completed successfully")

if __name__ == "__main__":
    main()