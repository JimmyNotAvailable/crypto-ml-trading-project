#!/usr/bin/env python3
"""
Unified launcher for both Discord bot and web server
Chạy cả bot Discord và web server trong cùng một process
"""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedLauncher:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.stop_all_processes()
        
    def stop_all_processes(self):
        """Stop all running processes"""
        for proc in self.processes:
            if proc and proc.poll() is None:
                logger.info(f"Terminating process {proc.pid}")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {proc.pid}")
                    proc.kill()
                    proc.wait()
                    
    def start_bot(self):
        """Start Discord bot in separate process"""
        try:
            logger.info("🤖 Starting Discord bot...")
            
            # Check for bot token
            token_file = Path("/app/token.txt")
            if token_file.exists():
                logger.info("Using token from token.txt")
                
            # Start bot process
            bot_proc = subprocess.Popen(
                [sys.executable, "-m", "app.bot"],
                cwd="/app",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(bot_proc)
            
            # Log bot output in separate thread
            def log_bot_output():
                if bot_proc.stdout:
                    while True:
                        line = bot_proc.stdout.readline()
                        if not line:
                            break
                        if line.strip():
                            logger.info(f"[BOT] {line.strip()}")
                        
            threading.Thread(target=log_bot_output, daemon=True).start()
            
            return bot_proc
            
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            return None
            
    def start_web(self):
        """Start web server in separate process"""
        try:
            logger.info("🌐 Starting web server...")
            
            # Check if web app exists
            web_app_path = Path("/app/web/app.py")
            if not web_app_path.exists():
                logger.warning("Web app not found, trying dashboard...")
                web_app_path = Path("/app/examples/ml/web_dashboard.py")
                
            if not web_app_path.exists():
                logger.error("No web application found!")
                return None
                
            # Start web process
            web_proc = subprocess.Popen(
                [sys.executable, str(web_app_path)],
                cwd="/app",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env={**os.environ, "WEB_PORT": "8000"}
            )
            
            self.processes.append(web_proc)
            
            # Log web output in separate thread
            def log_web_output():
                if web_proc.stdout:
                    while True:
                        line = web_proc.stdout.readline()
                        if not line:
                            break
                        if line.strip():
                            logger.info(f"[WEB] {line.strip()}")
                        
            threading.Thread(target=log_web_output, daemon=True).start()
            
            return web_proc
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return None
            
    def monitor_processes(self):
        """Monitor and restart processes if they crash"""
        while self.running:
            try:
                # Check each process
                for i, proc in enumerate(self.processes[:]):
                    if proc and proc.poll() is not None:
                        logger.warning(f"Process {proc.pid} has exited with code {proc.returncode}")
                        self.processes.remove(proc)
                        
                        # Restart if needed (optional)
                        if self.running:
                            logger.info("Restarting failed process...")
                            if i == 0:  # Bot process
                                new_proc = self.start_bot()
                            else:  # Web process
                                new_proc = self.start_web()
                                
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)
                
    def run(self):
        """Main run method"""
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        logger.info("🚀 Starting unified crypto service...")
        
        try:
            # Start both services
            bot_proc = self.start_bot()
            web_proc = self.start_web()
            
            if not bot_proc and not web_proc:
                logger.error("Failed to start any services!")
                return 1
                
            # Wait for processes to be ready
            time.sleep(3)
            
            # Log status
            if bot_proc:
                logger.info(f"✅ Bot started (PID: {bot_proc.pid})")
            if web_proc:
                logger.info(f"✅ Web server started (PID: {web_proc.pid})")
                logger.info("🌐 Web server should be available at http://localhost:8000")
                
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            logger.info("Stopping all services...")
            self.stop_all_processes()
            logger.info("✅ Shutdown complete")
            
        return 0

if __name__ == "__main__":
    launcher = UnifiedLauncher()
    sys.exit(launcher.run())