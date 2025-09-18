#!/usr/bin/env python3
"""
🌐 ENTERPRISE WEB DASHBOARD SERVER
=================================

Simple web server to serve the enterprise dashboard with real-time data
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.services.store import data_store
from app.services.trainer import enterprise_trainer
from app.ml.model_registry import model_registry

class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler for dashboard API endpoints"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(project_root, 'web'), **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        # API endpoints
        if parsed_path.path.startswith('/api/'):
            self.handle_api_request(parsed_path)
        else:
            # Serve static files
            super().do_GET()
    
    def handle_api_request(self, parsed_path):
        """Handle API requests"""
        try:
            if parsed_path.path == '/api/stats':
                self.send_stats_response()
            elif parsed_path.path == '/api/models':
                self.send_models_response()
            elif parsed_path.path == '/api/jobs':
                self.send_jobs_response()
            else:
                self.send_error(404, "API endpoint not found")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
    
    def send_stats_response(self):
        """Send dashboard statistics"""
        try:
            models_df = model_registry.list_models()
            jobs_df = enterprise_trainer.list_jobs()
            
            stats = {
                'total_models': len(models_df),
                'jobs_today': len(jobs_df),
                'success_rate': f"{len(jobs_df[jobs_df['status'] == 'completed']) / max(len(jobs_df), 1) * 100:.0f}%",
                'avg_performance': f"{models_df['test_r2'].mean():.4f}" if not models_df.empty else "0.0000"
            }
            
            self.send_json_response(stats)
        except Exception as e:
            self.send_json_response({'error': str(e)})
    
    def send_models_response(self):
        """Send models data"""
        try:
            models_df = model_registry.list_models()
            models_data = models_df.to_dict('records') if not models_df.empty else []
            self.send_json_response({'models': models_data})
        except Exception as e:
            self.send_json_response({'error': str(e)})
    
    def send_jobs_response(self):
        """Send training jobs data"""
        try:
            jobs_df = enterprise_trainer.list_jobs()
            jobs_data = jobs_df.to_dict('records') if not jobs_df.empty else []
            self.send_json_response({'jobs': jobs_data})
        except Exception as e:
            self.send_json_response({'error': str(e)})
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2, default=str)
        self.wfile.write(json_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

def start_dashboard_server(port=8000):
    """Start the dashboard web server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print(f"🌐 Starting Enterprise Dashboard Server...")
    print(f"📊 Dashboard URL: http://localhost:{port}/dashboard.html")
    print(f"🔌 API Base URL: http://localhost:{port}/api/")
    print(f"⚡ Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down dashboard server...")
        httpd.shutdown()

if __name__ == "__main__":
    start_dashboard_server()