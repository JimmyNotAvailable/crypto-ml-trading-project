# health_monitor.py
# Production health monitoring cho crypto data collector

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import psutil
import requests

class HealthMonitor:
    """
    Production health monitoring system
    - System health checks
    - Data freshness monitoring  
    - Performance metrics
    - Alert generation
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "realtime_production"
        )
        self.logger = logging.getLogger(__name__)
        
    def system_health_check(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                    "process_memory_mb": process_memory
                },
                "alerts": self._generate_system_alerts(cpu_percent, memory.percent, disk.free)
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def data_health_check(self) -> Dict[str, Any]:
        """Check data collection health"""
        try:
            latest_file = os.path.join(self.data_dir, "latest_prices.json")
            
            if not os.path.exists(latest_file):
                return {
                    "status": "error",
                    "error": "No latest prices data found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # File age check
            file_stat = os.stat(latest_file)
            file_time = datetime.fromtimestamp(file_stat.st_mtime)
            age_minutes = (datetime.now() - file_time).total_seconds() / 60
            
            # Load and validate data
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            symbols_count = len(data.get("symbols", {}))
            success_rate = data.get("successful", 0) / data.get("total_symbols", 1) * 100
            
            # Data quality checks
            quality_score = self._calculate_data_quality(data)
            
            status = "healthy"
            if age_minutes > 10:
                status = "stale"
            elif success_rate < 95:
                status = "degraded"
            elif quality_score < 80:
                status = "poor_quality"
            
            return {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "last_update": file_time.isoformat(),
                    "age_minutes": round(age_minutes, 2),
                    "symbols_count": symbols_count,
                    "target_symbols": 34,
                    "success_rate": round(success_rate, 2),
                    "quality_score": round(quality_score, 2),
                    "file_size_kb": round(file_stat.st_size / 1024, 2)
                },
                "alerts": self._generate_data_alerts(age_minutes, success_rate, quality_score)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e), 
                "timestamp": datetime.now().isoformat()
            }
    
    def api_health_check(self) -> Dict[str, Any]:
        """Check Binance API health"""
        try:
            start_time = time.time()
            
            # Test API connectivity
            response = requests.get(
                "https://api.binance.com/api/v3/ping",
                timeout=5
            )
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                # Test rate limits
                response = requests.get(
                    "https://api.binance.com/api/v3/exchangeInfo",
                    timeout=10
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "timestamp": datetime.now().isoformat(),
                        "api": {
                            "connectivity": "ok",
                            "response_time_ms": round(response_time, 2),
                            "rate_limit_status": "ok"
                        },
                        "alerts": self._generate_api_alerts(response_time)
                    }
            
            return {
                "status": "degraded",
                "timestamp": datetime.now().isoformat(),
                "api": {
                    "connectivity": "failed",
                    "response_time_ms": round(response_time, 2),
                    "http_status": response.status_code
                },
                "alerts": ["API connectivity issues"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "alerts": ["API completely unreachable"]
            }
    
    def _calculate_data_quality(self, data: Dict) -> float:
        """Calculate data quality score (0-100)"""
        try:
            symbols = data.get("symbols", {})
            if not symbols:
                return 0.0
            
            score = 0.0
            total_checks = 0
            
            for symbol, symbol_data in symbols.items():
                total_checks += 4  # 4 checks per symbol
                
                # Price validity
                if symbol_data.get("price_usd", 0) > 0:
                    score += 25
                
                # Change reasonableness (not extreme)
                change = abs(symbol_data.get("change_24h_percent", 0))
                if change < 50:  # Less than 50% change
                    score += 25
                
                # Timestamp freshness
                if symbol_data.get("timestamp"):
                    score += 25
                
                # VND conversion
                if symbol_data.get("price_vnd", 0) > 0:
                    score += 25
            
            return (score / total_checks) if total_checks > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_system_alerts(self, cpu: float, memory: float, disk_free: int) -> List[str]:
        """Generate system alerts"""
        alerts = []
        
        if cpu > 80:
            alerts.append(f"High CPU usage: {cpu:.1f}%")
        if memory > 85:
            alerts.append(f"High memory usage: {memory:.1f}%")
        if disk_free < 1024**3:  # Less than 1GB free
            alerts.append(f"Low disk space: {disk_free/(1024**3):.1f}GB free")
        
        return alerts
    
    def _generate_data_alerts(self, age_minutes: float, success_rate: float, quality_score: float) -> List[str]:
        """Generate data alerts"""
        alerts = []
        
        if age_minutes > 10:
            alerts.append(f"Stale data: {age_minutes:.1f} minutes old")
        if success_rate < 95:
            alerts.append(f"Low success rate: {success_rate:.1f}%")
        if quality_score < 80:
            alerts.append(f"Poor data quality: {quality_score:.1f}/100")
        
        return alerts
    
    def _generate_api_alerts(self, response_time: float) -> List[str]:
        """Generate API alerts"""
        alerts = []
        
        if response_time > 5000:  # 5 seconds
            alerts.append(f"Slow API response: {response_time:.0f}ms")
        elif response_time > 2000:  # 2 seconds
            alerts.append(f"API response time elevated: {response_time:.0f}ms")
        
        return alerts
    
    def full_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        system_health = self.system_health_check()
        data_health = self.data_health_check()
        api_health = self.api_health_check()
        
        # Overall status
        statuses = [
            system_health.get("status"),
            data_health.get("status"),
            api_health.get("status")
        ]
        
        if "error" in statuses:
            overall_status = "error"
        elif "degraded" in statuses or "stale" in statuses or "poor_quality" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Collect all alerts
        all_alerts = []
        all_alerts.extend(system_health.get("alerts", []))
        all_alerts.extend(data_health.get("alerts", []))
        all_alerts.extend(api_health.get("alerts", []))
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "system": system_health,
                "data": data_health,
                "api": api_health
            },
            "alerts": all_alerts,
            "summary": {
                "total_alerts": len(all_alerts),
                "critical_issues": len([a for a in all_alerts if "error" in a.lower() or "failed" in a.lower()]),
                "warnings": len([a for a in all_alerts if "high" in a.lower() or "slow" in a.lower()])
            }
        }
    
    def generate_health_report(self) -> str:
        """Generate human-readable health report"""
        health = self.full_health_check()
        
        report = f"""
🏥 CRYPTO DATA COLLECTOR HEALTH REPORT
{'='*50}
⏰ Timestamp: {health['timestamp']}
🎯 Overall Status: {health['overall_status'].upper()}

📊 COMPONENT STATUS:
• System: {health['components']['system']['status']}
• Data: {health['components']['data']['status']}  
• API: {health['components']['api']['status']}

"""
        
        if health['alerts']:
            report += f"🚨 ALERTS ({len(health['alerts'])}):\n"
            for alert in health['alerts']:
                report += f"  • {alert}\n"
        else:
            report += "✅ No alerts - all systems healthy\n"
        
        # Add details if available
        if 'data' in health['components']['data']:
            data_info = health['components']['data']['data']
            report += f"""
📈 DATA METRICS:
• Success Rate: {data_info['success_rate']}%
• Symbols: {data_info['symbols_count']}/{data_info['target_symbols']}
• Last Update: {data_info['age_minutes']:.1f} min ago
• Quality Score: {data_info['quality_score']}/100
"""
        
        if 'system' in health['components']['system']:
            sys_info = health['components']['system']['system']
            report += f"""
💻 SYSTEM METRICS:
• CPU: {sys_info['cpu_percent']:.1f}%
• Memory: {sys_info['memory_percent']:.1f}%
• Process RAM: {sys_info['process_memory_mb']:.1f}MB
• Disk Free: {sys_info['disk_free_gb']:.1f}GB
"""
        
        return report

if __name__ == "__main__":
    # Test health monitoring
    print("🏥 Testing Health Monitor...")
    
    monitor = HealthMonitor()
    
    # Full health check
    health = monitor.full_health_check()
    
    # Print report
    report = monitor.generate_health_report()
    print(report)
    
    # Save health data
    health_file = f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(health_file, 'w', encoding='utf-8') as f:
        json.dump(health, f, indent=2, default=str)
    
    print(f"💾 Health data saved to: {health_file}")