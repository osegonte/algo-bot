#!/usr/bin/env python3
"""
Level 6-E: Child Intel Fetcher
Downloads and unpacks intelligence bundles for child bots
"""

import os
import json
import zipfile
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List
import requests

class IntelFetcher:
    """Fetch and unpack intelligence bundles for child bots"""
    
    def __init__(self, config: Dict = None, child_id: str = None):
        self.config = config or {}
        self.child_id = child_id or os.getenv("CHILD_ID", "trader_001")
        
        # Directories
        self.local_intel_dir = Path("intel")
        self.local_intel_dir.mkdir(exist_ok=True)
        
        self.bundles_cache_dir = Path("cache/bundles")
        self.bundles_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Parent bundle locations (can be local or remote)
        self.bundle_sources = self._get_bundle_sources()
        
        # Local state tracking
        self.state_file = Path("logs") / f"intel_state_{self.child_id}.json"
        self.state_file.parent.mkdir(exist_ok=True)
    
    def _get_bundle_sources(self) -> List[Dict]:
        """Get list of bundle sources to check"""
        
        sources = []
        
        # Local parent directory (for development/testing)
        local_parent_bundles = Path("../parent_bot/bundles")
        if local_parent_bundles.exists():
            sources.append({
                'type': 'local',
                'path': str(local_parent_bundles / "intel_bundle.zip"),
                'priority': 1
            })
        
        # Alternative local path (same machine)
        alt_local_bundles = Path("bundles")
        if alt_local_bundles.exists():
            sources.append({
                'type': 'local',
                'path': str(alt_local_bundles / "intel_bundle.zip"),
                'priority': 2
            })
        
        # Remote URL (if configured)
        remote_url = self.config.get("remote_bundle_url")
        if remote_url:
            sources.append({
                'type': 'remote',
                'url': remote_url,
                'priority': 3
            })
        
        # Sort by priority
        sources.sort(key=lambda s: s['priority'])
        
        return sources
    
    def fetch_and_update_intel(self) -> Dict:
        """Main intel fetching workflow"""
        
        print(f"🔍 Fetching intel for child bot: {self.child_id}")
        
        # Check current state
        current_state = self._load_current_state()
        
        # Find available bundle
        bundle_info = self._find_latest_bundle()
        
        if not bundle_info:
            print("❌ No intel bundles found")
            return {
                'success': False,
                'action': 'no_bundle_found',
                'intel_updated': False
            }
        
        # Check if update is needed
        if not self._update_needed(current_state, bundle_info):
            print(f"✅ Intel is up to date (checksum: {bundle_info.get('checksum', 'unknown')[:8]})")
            return {
                'success': True,
                'action': 'up_to_date',
                'intel_updated': False,
                'bundle_info': bundle_info
            }
        
        # Download bundle if needed
        local_bundle_path = self._download_bundle(bundle_info)
        
        if not local_bundle_path:
            print("❌ Failed to download bundle")
            return {
                'success': False,
                'action': 'download_failed',
                'intel_updated': False
            }
        
        # Unpack and validate bundle
        unpack_result = self._unpack_bundle(local_bundle_path)
        
        if not unpack_result['success']:
            print(f"❌ Failed to unpack bundle: {unpack_result.get('error', 'unknown')}")
            return {
                'success': False,
                'action': 'unpack_failed',
                'intel_updated': False,
                'error': unpack_result.get('error')
            }
        
        # Update state
        new_state = {
            'last_update': datetime.now(timezone.utc).isoformat(),
            'bundle_checksum': bundle_info.get('checksum'),
            'bundle_source': bundle_info.get('source'),
            'files_updated': unpack_result['files_extracted'],
            'child_id': self.child_id
        }
        
        self._save_current_state(new_state)
        
        # Log successful update
        self._log_intel_update(new_state, unpack_result)
        
        print(f"✅ Intel updated successfully")
        print(f"📄 Files updated: {len(unpack_result['files_extracted'])}")
        print(f"🔄 Source: {bundle_info.get('source', 'unknown')}")
        
        return {
            'success': True,
            'action': 'updated',
            'intel_updated': True,
            'files_updated': unpack_result['files_extracted'],
            'bundle_info': bundle_info,
            'source': bundle_info.get('source')
        }
    
    def _load_current_state(self) -> Dict:
        """Load current intel state"""
        
        if not self.state_file.exists():
            return {}
        
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading state: {e}")
            return {}
    
    def _save_current_state(self, state: Dict):
        """Save current intel state"""
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving state: {e}")
    
    def _find_latest_bundle(self) -> Optional[Dict]:
        """Find the latest available bundle from all sources"""
        
        for source in self.bundle_sources:
            try:
                if source['type'] == 'local':
                    bundle_info = self._check_local_bundle(source['path'])
                elif source['type'] == 'remote':
                    bundle_info = self._check_remote_bundle(source['url'])
                else:
                    continue
                
                if bundle_info:
                    bundle_info['source'] = f"{source['type']}:{source.get('path', source.get('url', 'unknown'))}"
                    return bundle_info
                    
            except Exception as e:
                print(f"⚠️ Error checking {source['type']} source: {e}")
                continue
        
        return None
    
    def _check_local_bundle(self, bundle_path: str) -> Optional[Dict]:
        """Check local bundle file"""
        
        path = Path(bundle_path)
        
        if not path.exists():
            return None
        
        # Get file info
        stat = path.stat()
        checksum = self._calculate_file_checksum(path)
        
        return {
            'path': str(path),
            'size_bytes': stat.st_size,
            'modified_at': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            'checksum': checksum,
            'available': True
        }
    
    def _check_remote_bundle(self, bundle_url: str) -> Optional[Dict]:
        """Check remote bundle availability"""
        
        try:
            # HEAD request to get metadata without downloading
            response = requests.head(bundle_url, timeout=10)
            response.raise_for_status()
            
            return {
                'url': bundle_url,
                'size_bytes': int(response.headers.get('content-length', 0)),
                'modified_at': response.headers.get('last-modified', ''),
                'etag': response.headers.get('etag', ''),
                'available': True
            }
            
        except Exception as e:
            print(f"⚠️ Remote bundle check failed: {e}")
            return None
    
    def _update_needed(self, current_state: Dict, bundle_info: Dict) -> bool:
        """Check if intel update is needed"""
        
        # No previous state = update needed
        if not current_state:
            return True
        
        # Different checksum = update needed
        current_checksum = current_state.get('bundle_checksum')
        new_checksum = bundle_info.get('checksum')
        
        if current_checksum and new_checksum:
            return current_checksum != new_checksum
        
        # Different source = update needed
        current_source = current_state.get('bundle_source')
        new_source = bundle_info.get('source')
        
        if current_source != new_source:
            return True
        
        # Check modification time for local files
        if 'modified_at' in bundle_info:
            try:
                bundle_mtime = datetime.fromisoformat(bundle_info['modified_at'].replace('Z', '+00:00'))
                last_update = datetime.fromisoformat(current_state.get('last_update', ''))
                
                return bundle_mtime > last_update
                
            except ValueError:
                # Can't parse timestamps, assume update needed
                return True
        
        # Default to no update needed
        return False
    
    def _download_bundle(self, bundle_info: Dict) -> Optional[Path]:
        """Download bundle to local cache"""
        
        if 'path' in bundle_info:
            # Local file, just return the path
            return Path(bundle_info['path'])
        
        elif 'url' in bundle_info:
            # Remote file, download it
            return self._download_remote_bundle(bundle_info)
        
        return None
    
    def _download_remote_bundle(self, bundle_info: Dict) -> Optional[Path]:
        """Download remote bundle to cache"""
        
        url = bundle_info['url']
        
        # Create cached filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        cached_bundle = self.bundles_cache_dir / f"intel_bundle_{url_hash}.zip"
        
        try:
            print(f"📥 Downloading bundle from: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(cached_bundle, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded to cache: {cached_bundle.name}")
            return cached_bundle
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def _unpack_bundle(self, bundle_path: Path) -> Dict:
        """Unpack bundle to local intel directory"""
        
        try:
            # Backup existing intel files
            backup_dir = self._create_intel_backup()
            
            files_extracted = []
            
            with zipfile.ZipFile(bundle_path, 'r') as zipf:
                # Validate bundle first
                validation_result = self._validate_bundle(zipf)
                
                if not validation_result['valid']:
                    return {
                        'success': False,
                        'error': f"Bundle validation failed: {validation_result['error']}"
                    }
                
                # Extract files
                for file_info in zipf.infolist():
                    # Skip directories and unwanted files
                    if file_info.is_dir() or file_info.filename.startswith('__'):
                        continue
                    
                    # Extract to intel directory
                    extracted_path = self.local_intel_dir / file_info.filename
                    
                    with zipf.open(file_info.filename) as source, open(extracted_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    files_extracted.append(file_info.filename)
                    print(f"📄 Extracted: {file_info.filename}")
            
            return {
                'success': True,
                'files_extracted': files_extracted,
                'backup_dir': str(backup_dir) if backup_dir else None
            }
            
        except Exception as e:
            # Restore backup if extraction failed
            if 'backup_dir' in locals() and backup_dir:
                self._restore_intel_backup(backup_dir)
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_bundle(self, zipf: zipfile.ZipFile) -> Dict:
        """Validate bundle contents"""
        
        try:
            # Check for manifest
            file_list = zipf.namelist()
            
            if 'bundle_manifest.json' not in file_list:
                return {
                    'valid': False,
                    'error': 'Missing bundle_manifest.json'
                }
            
            # Read and validate manifest
            with zipf.open('bundle_manifest.json') as f:
                manifest = json.load(f)
            
            # Check required files in manifest
            required_files = ['news_sentiment.csv', 'econ_calendar.json', 'market_regime.json']
            
            manifest_files = list(manifest.get('files', {}).keys())
            missing_files = [f for f in required_files if f not in manifest_files]
            
            if missing_files:
                return {
                    'valid': False,
                    'error': f"Missing required files in manifest: {', '.join(missing_files)}"
                }
            
            # Check if files actually exist in zip
            missing_in_zip = [f for f in required_files if f not in file_list]
            
            if missing_in_zip:
                return {
                    'valid': False,
                    'error': f"Missing files in zip: {', '.join(missing_in_zip)}"
                }
            
            return {
                'valid': True,
                'manifest': manifest
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}"
            }
    
    def _create_intel_backup(self) -> Optional[Path]:
        """Create backup of current intel files"""
        
        intel_files = list(self.local_intel_dir.glob("*"))
        
        if not intel_files:
            return None
        
        backup_dir = Path("backup") / f"intel_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in intel_files:
            if file_path.is_file():
                shutil.copy2(file_path, backup_dir / file_path.name)
        
        print(f"💾 Created intel backup: {backup_dir}")
        return backup_dir
    
    def _restore_intel_backup(self, backup_dir: Path):
        """Restore intel files from backup"""
        
        try:
            # Clear current intel directory
            for file_path in self.local_intel_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            
            # Restore from backup
            for backup_file in backup_dir.glob("*"):
                if backup_file.is_file():
                    shutil.copy2(backup_file, self.local_intel_dir / backup_file.name)
            
            print(f"🔄 Restored intel from backup: {backup_dir}")
            
        except Exception as e:
            print(f"❌ Failed to restore backup: {e}")
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        
        md5_hash = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        return md5_hash.hexdigest()
    
    def _log_intel_update(self, state: Dict, unpack_result: Dict):
        """Log intel update event"""
        
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'child_id': self.child_id,
            'action': 'intel_updated',
            'intel_updated': True,  # Level 6-E requirement
            'bundle_checksum': state.get('bundle_checksum'),
            'bundle_source': state.get('bundle_source'),
            'files_updated': unpack_result['files_extracted'],
            'update_count': len(unpack_result['files_extracted'])
        }
        
        log_file = Path("logs") / f"intel_updates_{self.child_id}.json"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_intel_status(self) -> Dict:
        """Get current intel status"""
        
        current_state = self._load_current_state()
        
        # Check local intel files
        intel_files_status = {}
        required_files = ['news_sentiment.csv', 'econ_calendar.json', 'market_regime.json']
        
        for filename in required_files:
            file_path = self.local_intel_dir / filename
            intel_files_status[filename] = {
                'exists': file_path.exists(),
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
                'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat() if file_path.exists() else None
            }
        
        # Check available bundles
        available_bundle = self._find_latest_bundle()
        
        status = {
            'child_id': self.child_id,
            'last_update': current_state.get('last_update'),
            'current_bundle_checksum': current_state.get('bundle_checksum'),
            'intel_files': intel_files_status,
            'available_bundle': available_bundle is not None,
            'update_needed': self._update_needed(current_state, available_bundle) if available_bundle else False,
            'bundle_sources_configured': len(self.bundle_sources),
            'state_file_exists': self.state_file.exists()
        }
        
        return status

def load_config() -> Dict:
    """Load configuration"""
    config_file = Path("config/base_config.yaml")
    
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f).get('intel_fetcher', {})
    
    return {}

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Child Intel Fetcher (Level 6-E)")
    parser.add_argument('--child-id', help='Child bot ID (default: from env CHILD_ID)')
    parser.add_argument('--force', action='store_true', help='Force update even if current')
    parser.add_argument('--status', action='store_true', help='Show intel status')
    parser.add_argument('--check-only', action='store_true', help='Check for updates without downloading')
    
    args = parser.parse_args()
    
    print("📥 Child Intel Fetcher (Level 6-E)")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Initialize fetcher
    child_id = args.child_id or os.getenv("CHILD_ID", "trader_001")
    fetcher = IntelFetcher(config, child_id)
    
    if args.status:
        # Show intel status
        status = fetcher.get_intel_status()
        
        print(f"\n📊 Intel Status for {status['child_id']}:")
        print(f"   Last update: {status['last_update'] or 'Never'}")
        print(f"   Bundle checksum: {status['current_bundle_checksum'] or 'None'}")
        print(f"   Bundle sources: {status['bundle_sources_configured']}")
        print(f"   Available bundle: {'✅' if status['available_bundle'] else '❌'}")
        print(f"   Update needed: {'⚠️ Yes' if status['update_needed'] else '✅ No'}")
        
        print(f"\n📄 Intel Files:")
        for filename, file_info in status['intel_files'].items():
            exists_emoji = "✅" if file_info['exists'] else "❌"
            size_info = f"({file_info['size_bytes']} bytes)" if file_info['exists'] else ""
            print(f"   {exists_emoji} {filename} {size_info}")
        
        return
    
    if args.check_only:
        # Check for updates without downloading
        current_state = fetcher._load_current_state()
        bundle_info = fetcher._find_latest_bundle()
        
        if not bundle_info:
            print("❌ No bundles available")
            return
        
        update_needed = fetcher._update_needed(current_state, bundle_info)
        
        if update_needed:
            print(f"⚠️ Update available")
            print(f"   Source: {bundle_info.get('source', 'unknown')}")
            print(f"   Size: {bundle_info.get('size_bytes', 0)} bytes")
        else:
            print(f"✅ Intel is up to date")
        
        return
    
    # Force update if requested
    if args.force:
        # Clear current state to force update
        if fetcher.state_file.exists():
            fetcher.state_file.unlink()
        print("🔄 Forcing intel update...")
    
    # Fetch and update intel
    result = fetcher.fetch_and_update_intel()
    
    if result['success']:
        if result['intel_updated']:
            print(f"\n✅ Level 6-E Complete!")
            print(f"📥 Intel updated for: {child_id}")
            print(f"📄 Files updated: {len(result['files_updated'])}")
            print(f"🔄 Source: {result.get('source', 'unknown')}")
            
            # Show updated files
            for filename in result['files_updated']:
                print(f"   📄 {filename}")
                
        else:
            print(f"✅ Intel is current (no update needed)")
    else:
        print(f"❌ Intel fetch failed: {result.get('action', 'unknown error')}")
        if result.get('error'):
            print(f"   Error: {result['error']}")
        exit(1)

if __name__ == "__main__":
    main()