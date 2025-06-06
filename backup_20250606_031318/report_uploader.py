import shutil
import json
import time
from pathlib import Path
from datetime import datetime
import threading

CHILD_LOG_DIR = Path("logs")
PARENT_DIR = Path("../parent_bot/logs")  # adjust if running both locally
SYNC_LOG = CHILD_LOG_DIR / "sync_status.json"

class ReportUploader:
    def __init__(self, auto_sync_interval=30):
        self.auto_sync_interval = auto_sync_interval
        self.is_running = False
        self.sync_thread = None
        self.last_sync = None
        
    def push_logs(self):
        """Level 3: Enhanced log pushing with status tracking"""
        try:
            # Ensure parent directory exists
            PARENT_DIR.mkdir(parents=True, exist_ok=True)
            
            uploaded_files = []
            skipped_files = []
            
            # Copy all JSON log files
            for file in CHILD_LOG_DIR.glob("*.json"):
                if file.name == "sync_status.json":
                    continue  # Skip sync status file
                    
                try:
                    destination = PARENT_DIR / file.name
                    
                    # Check if file needs updating
                    if destination.exists():
                        source_mtime = file.stat().st_mtime
                        dest_mtime = destination.stat().st_mtime
                        
                        if source_mtime <= dest_mtime:
                            skipped_files.append(file.name)
                            continue
                    
                    # Copy the file
                    shutil.copy2(file, destination)  # copy2 preserves metadata
                    uploaded_files.append(file.name)
                    
                except Exception as e:
                    print(f"❌ Failed to upload {file.name}: {e}")
            
            # Log sync status
            sync_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "push_logs",
                "uploaded_files": uploaded_files,
                "skipped_files": skipped_files,
                "success": True,
                "error": None
            }
            
            self._log_sync_status(sync_status)
            self.last_sync = datetime.utcnow()
            
            if uploaded_files:
                print(f"📤 Uploaded {len(uploaded_files)} files: {', '.join(uploaded_files)}")
            if skipped_files:
                print(f"⏭️  Skipped {len(skipped_files)} unchanged files")
                
            return True
            
        except Exception as e:
            sync_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "push_logs",
                "uploaded_files": [],
                "skipped_files": [],
                "success": False,
                "error": str(e)
            }
            self._log_sync_status(sync_status)
            print(f"❌ Sync failed: {e}")
            return False
    
    def start_auto_sync(self):
        """Start automatic syncing in background thread"""
        if self.is_running:
            print("⚠️  Auto-sync already running")
            return
            
        self.is_running = True
        self.sync_thread = threading.Thread(target=self._auto_sync_loop, daemon=True)
        self.sync_thread.start()
        print(f"🔄 Auto-sync started (interval: {self.auto_sync_interval}s)")
    
    def stop_auto_sync(self):
        """Stop automatic syncing"""
        self.is_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        print("⏹️  Auto-sync stopped")
    
    def _auto_sync_loop(self):
        """Background sync loop"""
        while self.is_running:
            try:
                self.push_logs()
                time.sleep(self.auto_sync_interval)
            except Exception as e:
                print(f"❌ Auto-sync error: {e}")
                time.sleep(self.auto_sync_interval)
    
    def _log_sync_status(self, status):
        """Log sync operations"""
        CHILD_LOG_DIR.mkdir(exist_ok=True)
        
        with open(SYNC_LOG, "a") as f:
            f.write(json.dumps(status) + "\n")
    
    def get_sync_status(self):
        """Get recent sync status"""
        if not SYNC_LOG.exists():
            return {"status": "no_sync_history"}
        
        try:
            with open(SYNC_LOG) as f:
                lines = f.readlines()
                if lines:
                    last_sync = json.loads(lines[-1].strip())
                    return last_sync
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
        return {"status": "no_data"}

# Global uploader instance
uploader = ReportUploader()

def push_logs():
    """Convenience function for manual sync"""
    return uploader.push_logs()

def start_auto_sync(interval=30):
    """Start automatic log syncing"""
    uploader.auto_sync_interval = interval
    uploader.start_auto_sync()

def stop_auto_sync():
    """Stop automatic log syncing"""
    uploader.stop_auto_sync()

def get_sync_status():
    """Get current sync status"""
    return uploader.get_sync_status()

if __name__ == "__main__":
    # Manual sync
    print("🔄 Starting manual log sync...")
    success = push_logs()
    
    if success:
        print("✅ Manual sync completed")
        
        # Show sync status
        status = get_sync_status()
        print(f"📊 Last sync: {status.get('timestamp', 'Unknown')}")
    else:
        print("❌ Manual sync failed")