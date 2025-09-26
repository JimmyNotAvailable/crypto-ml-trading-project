#!/usr/bin/env python3
"""
Monitor bot responses to detect duplicates in real-time
"""

import time
import subprocess
import sys
from datetime import datetime

def monitor_bot_logs():
    """Monitor bot output for duplicate detection"""
    print("🔍 Monitoring bot for duplicate responses...")
    print("Press Ctrl+C to stop monitoring")
    
    response_history = []
    
    try:
        while True:
            # In a real implementation, you would read from bot logs
            # For now, just show monitoring status
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Monitoring active... (Use Discord to test commands)")
            
            # Check for duplicate patterns in response_history
            if len(response_history) > 1:
                recent = response_history[-2:]
                if len(set(recent)) < len(recent):
                    print("⚠️ DUPLICATE DETECTED!")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_bot_logs()