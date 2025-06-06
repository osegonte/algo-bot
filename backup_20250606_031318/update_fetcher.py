import yaml
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime

PARENT_CFG_DIR = Path("../parent_bot/config")
LOCAL_CFG_DIR = Path("config")
CHILD_ID = os.getenv("CHILD_ID", "trader_001")
LOCAL_CFG = LOCAL_CFG_DIR / f"ai_trading_config_{CHILD_ID}.yaml"
SYNC_LOG = Path("logs") / "config_sync.json"

class UpdateFetcher:
    def __init__(self, child_id=None):
        self.child_id = child_id or CHILD_ID
        self.local_config_path = LOCAL_CFG_DIR / f"ai_trading_config_{self.child_id}.yaml"
        
    def pull_updates(self):
        """Level 3: Enhanced config pulling with change detection"""
        try:
            # Source config from parent
            parent_config = PARENT_CFG_DIR / self.local_config_path.name
            
            if not parent_config.exists():
                print(f"⚠️  No parent config found: {parent_config}")
                return False
            
            # Read parent config
            with open(parent_config) as f:
                parent_data = f.read()
            
            # Check if local config exists and compare
            config_changed = True
            old_hash = None
            new_hash = hashlib.md5(parent_data.encode()).hexdigest()
            
            if self.local_config_path.exists():
                with open(self.local_config_path) as f:
                    local_data = f.read()
                old_hash = hashlib.md5(local_data.encode()).hexdigest()
                config_changed = (old_hash != new_hash)
            
            if config_changed:
                # Backup old config if it exists
                if self.local_config_path.exists():
                    backup_path = self.local_config_path.with_suffix('.yaml.backup')
                    self.local_config_path.rename(backup_path)
                    print(f"📦 Backed up old config to: {backup_path}")
                
                # Write new config
                LOCAL_CFG_DIR.mkdir(exist_ok=True)
                self.local_config_path.write_text(parent_data)
                
                # Parse config to get strategy info
                config_data = yaml.safe_load(parent_data)
                strategy = config_data.get("strategy", {}).get("default", "unknown")
                
                print(f"📥 Config updated for {self.child_id}")
                print(f"   Strategy: {strategy}")
                print(f"   File: {self.local_config_path}")
                
                # Log the config update
                self._log_config_update(config_data, old_hash, new_hash)
                
                return True
            else:
                print(f"✅ Config unchanged for {self.child_id}")
                return False
                
        except Exception as e:
            print(f"❌ Config update failed for {self.child_id}: {e}")
            self._log_config_error(str(e))
            return False
    
    def _log_config_update(self, config_data, old_hash, new_hash):
        """Log configuration updates with change detection"""
        
        # Extract key config info
        strategy = config_data.get("strategy", {}).get("default", "unknown")
        risk_level = config_data.get("risk", {}).get("risk_level", "unknown")
        confidence = config_data.get("confidence", 0)
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "child_id": self.child_id,
            "action": "config_update",
            "config_updated": True,  # Level 3 requirement
            "strategy": strategy,
            "risk_level": risk_level,
            "confidence": confidence,
            "old_hash": old_hash,
            "new_hash": new_hash,
            "updated_by": config_data.get("generated_by", "unknown"),
            "success": True
        }
        
        self._write_sync_log(log_entry)
    
    def _log_config_error(self, error_msg):
        """Log configuration update errors"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "child_id": self.child_id,
            "action": "config_update",
            "config_updated": False,
            "success": False,
            "error": error_msg
        }
        
        self._write_sync_log(log_entry)
    
    def _write_sync_log(self, log_entry):
        """Write to sync log file"""
        SYNC_LOG.parent.mkdir(exist_ok=True)
        
        with open(SYNC_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def check_for_updates(self):
        """Check if updates are available without pulling"""
        try:
            parent_config = PARENT_CFG_DIR / self.local_config_path.name
            
            if not parent_config.exists():
                return False, "No parent config found"
            
            # Compare modification times
            parent_mtime = parent_config.stat().st_mtime
            
            if self.local_config_path.exists():
                local_mtime = self.local_config_path.stat().st_mtime
                if parent_mtime > local_mtime:
                    return True, "Parent config is newer"
                else:
                    return False, "Local config is up to date"
            else:
                return True, "No local config exists"
                
        except Exception as e:
            return False, f"Error checking updates: {e}"
    
    def get_config_status(self):
        """Get current configuration status"""
        status = {
            "child_id": self.child_id,
            "local_config_exists": self.local_config_path.exists(),
            "local_config_path": str(self.local_config_path)
        }
        
        if self.local_config_path.exists():
            try:
                with open(self.local_config_path) as f:
                    config_data = yaml.safe_load(f)
                
                status.update({
                    "strategy": config_data.get("strategy", {}).get("default"),
                    "risk_level": config_data.get("risk", {}).get("risk_level"),
                    "last_updated": config_data.get("updated"),
                    "confidence": config_data.get("confidence")
                })
            except Exception as e:
                status["config_error"] = str(e)
        
        # Check for available updates
        has_updates, update_msg = self.check_for_updates()
        status.update({
            "updates_available": has_updates,
            "update_status": update_msg
        })
        
        return status

# Global fetcher instance
fetcher = UpdateFetcher()

def pull_updates(child_id=None):
    """Convenience function for manual config update"""
    if child_id:
        fetch_instance = UpdateFetcher(child_id)
        return fetch_instance.pull_updates()
    return fetcher.pull_updates()

def check_for_updates(child_id=None):
    """Check if configuration updates are available"""
    if child_id:
        fetch_instance = UpdateFetcher(child_id)
        return fetch_instance.check_for_updates()
    return fetcher.check_for_updates()

def get_config_status(child_id=None):
    """Get current configuration status"""
    if child_id:
        fetch_instance = UpdateFetcher(child_id)
        return fetch_instance.get_config_status()
    return fetcher.get_config_status()

def auto_pull_if_updated():
    """Automatically pull updates if available"""
    has_updates, msg = check_for_updates()
    
    if has_updates:
        print(f"📥 Updates available: {msg}")
        success = pull_updates()
        return success
    else:
        print(f"✅ {msg}")
        return False

if __name__ == "__main__":
    print(f"🔄 Checking for config updates for {CHILD_ID}...")
    
    # Check status first
    status = get_config_status()
    print(f"📊 Current status: {status['update_status']}")
    
    if status.get("local_config_exists"):
        print(f"   Current strategy: {status.get('strategy', 'unknown')}")
        print(f"   Risk level: {status.get('risk_level', 'unknown')}")
    
    # Pull updates if available
    updated = auto_pull_if_updated()
    
    if updated:
        print("✅ Configuration successfully updated")
    else:
        print("ℹ️  No updates needed")