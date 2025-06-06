#!/usr/bin/env python3
"""
Level 6-D: Intel Bundle Packager
Parent cron script that packages intelligence data into intel_bundle.zip
"""

import os
import json
import zipfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import csv

class IntelBundlePackager:
    """Package intelligence data for child bot consumption"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.intel_dir = Path("intel")
        self.intel_dir.mkdir(exist_ok=True)
        
        # Bundle output directory
        self.bundle_dir = Path("bundles")
        self.bundle_dir.mkdir(exist_ok=True)
        
        # Required intel files
        self.required_files = [
            "news_sentiment.csv",
            "econ_calendar.json", 
            "market_regime.json"
        ]
        
        # Optional files to include if available
        self.optional_files = [
            "news_sentiment_metadata.json",
            "econ_calendar.csv"
        ]
    
    def create_intel_bundle(self, force: bool = False) -> Dict:
        """Create intelligence bundle for distribution"""
        
        print("📦 Creating intelligence bundle...")
        
        bundle_timestamp = datetime.now(timezone.utc)
        
        # Check if bundle needs updating
        if not force and not self._bundle_needs_update():
            print("✅ Bundle is up to date")
            return {
                'success': True,
                'action': 'skipped',
                'reason': 'up_to_date',
                'bundle_file': str(self._get_latest_bundle_path())
            }
        
        # Verify required files exist
        missing_files = self._check_required_files()
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return {
                'success': False,
                'action': 'failed',
                'reason': 'missing_files',
                'missing_files': missing_files
            }
        
        # Create bundle filename with timestamp
        bundle_filename = f"intel_bundle_{bundle_timestamp.strftime('%Y%m%d_%H%M%S')}.zip"
        bundle_path = self.bundle_dir / bundle_filename
        
        # Create ZIP bundle
        try:
            bundled_files = self._create_zip_bundle(bundle_path)
            
            # Create symlink to latest bundle
            latest_bundle_path = self.bundle_dir / "intel_bundle.zip"
            if latest_bundle_path.exists():
                latest_bundle_path.unlink()
            latest_bundle_path.symlink_to(bundle_filename)
            
            # Generate bundle metadata
            metadata = self._generate_bundle_metadata(bundle_path, bundled_files, bundle_timestamp)
            
            # Save metadata
            metadata_file = self.bundle_dir / f"intel_bundle_{bundle_timestamp.strftime('%Y%m%d_%H%M%S')}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Bundle created: {bundle_filename}")
            print(f"📄 Files included: {len(bundled_files)}")
            print(f"💾 Bundle size: {metadata['bundle_size_mb']:.2f} MB")
            
            # Log bundle creation
            self._log_bundle_creation(metadata)
            
            return {
                'success': True,
                'action': 'created',
                'bundle_file': str(bundle_path),
                'latest_bundle': str(latest_bundle_path),
                'metadata_file': str(metadata_file),
                'files_included': bundled_files,
                'bundle_size_mb': metadata['bundle_size_mb']
            }
            
        except Exception as e:
            print(f"❌ Bundle creation failed: {e}")
            return {
                'success': False,
                'action': 'failed',
                'reason': 'creation_error',
                'error': str(e)
            }
    
    def _bundle_needs_update(self) -> bool:
        """Check if bundle needs updating based on file timestamps"""
        
        latest_bundle = self._get_latest_bundle_path()
        
        if not latest_bundle or not latest_bundle.exists():
            return True
        
        bundle_mtime = latest_bundle.stat().st_mtime
        
        # Check if any intel file is newer than the bundle
        for filename in self.required_files + self.optional_files:
            file_path = self.intel_dir / filename
            if file_path.exists():
                if file_path.stat().st_mtime > bundle_mtime:
                    return True
        
        return False
    
    def _get_latest_bundle_path(self) -> Optional[Path]:
        """Get path to the latest bundle"""
        
        latest_symlink = self.bundle_dir / "intel_bundle.zip"
        if latest_symlink.exists():
            return latest_symlink
        
        # Fallback: find most recent bundle file
        bundle_files = list(self.bundle_dir.glob("intel_bundle_*.zip"))
        if bundle_files:
            bundle_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return bundle_files[0]
        
        return None
    
    def _check_required_files(self) -> List[str]:
        """Check which required files are missing"""
        
        missing_files = []
        
        for filename in self.required_files:
            file_path = self.intel_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
            elif file_path.stat().st_size == 0:
                missing_files.append(f"{filename} (empty)")
        
        return missing_files
    
    def _create_zip_bundle(self, bundle_path: Path) -> List[str]:
        """Create ZIP file with intel data"""
        
        bundled_files = []
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add required files
            for filename in self.required_files:
                file_path = self.intel_dir / filename
                if file_path.exists():
                    zipf.write(file_path, filename)
                    bundled_files.append(filename)
                    print(f"📄 Added: {filename}")
            
            # Add optional files
            for filename in self.optional_files:
                file_path = self.intel_dir / filename
                if file_path.exists():
                    zipf.write(file_path, filename)
                    bundled_files.append(filename)
                    print(f"📄 Added (optional): {filename}")
            
            # Add bundle manifest
            manifest = self._create_bundle_manifest(bundled_files)
            manifest_json = json.dumps(manifest, indent=2)
            zipf.writestr("bundle_manifest.json", manifest_json)
            bundled_files.append("bundle_manifest.json")
        
        return bundled_files
    
    def _create_bundle_manifest(self, bundled_files: List[str]) -> Dict:
        """Create manifest describing bundle contents"""
        
        manifest = {
            'bundle_version': '1.0',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'files': {},
            'data_sources': set(),
            'summary': {}
        }
        
        # Analyze each file
        for filename in bundled_files:
            if filename == "bundle_manifest.json":
                continue
                
            file_path = self.intel_dir / filename
            if not file_path.exists():
                continue
            
            file_info = {
                'size_bytes': file_path.stat().st_size,
                'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat(),
                'type': self._get_file_type(filename)
            }
            
            # Add file-specific metadata
            if filename == "news_sentiment.csv":
                file_info.update(self._analyze_news_sentiment_file(file_path))
            elif filename == "econ_calendar.json":
                file_info.update(self._analyze_econ_calendar_file(file_path))
            elif filename == "market_regime.json":
                file_info.update(self._analyze_regime_file(file_path))
            
            manifest['files'][filename] = file_info
            
            # Collect data sources
            sources = file_info.get('sources', [])
            manifest['data_sources'].update(sources)
        
        # Convert set to list for JSON serialization
        manifest['data_sources'] = list(manifest['data_sources'])
        
        # Generate summary
        manifest['summary'] = self._generate_bundle_summary(manifest['files'])
        
        return manifest
    
    def _get_file_type(self, filename: str) -> str:
        """Determine file type from filename"""
        
        if 'news_sentiment' in filename:
            return 'news_sentiment'
        elif 'econ_calendar' in filename:
            return 'economic_calendar'
        elif 'market_regime' in filename:
            return 'market_regime'
        else:
            return 'unknown'
    
    def _analyze_news_sentiment_file(self, file_path: Path) -> Dict:
        """Analyze news sentiment CSV file"""
        
        try:
            with open(file_path, newline='') as f:
                reader = csv.DictReader(f)
                headlines = list(reader)
            
            if not headlines:
                return {'record_count': 0, 'sources': []}
            
            # Analyze sentiment distribution
            sentiments = [float(h.get('sentiment_compound', 0)) for h in headlines if h.get('sentiment_compound')]
            
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            # Get unique sources
            sources = list(set(h.get('source', 'unknown') for h in headlines))
            
            return {
                'record_count': len(headlines),
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'avg_sentiment': sum(sentiments) / len(sentiments) if sentiments else 0,
                'sources': sources,
                'symbols_mentioned': list(set(h.get('symbols_mentioned', '').split(',') for h in headlines if h.get('symbols_mentioned')))
            }
            
        except Exception as e:
            return {'error': str(e), 'sources': []}
    
    def _analyze_econ_calendar_file(self, file_path: Path) -> Dict:
        """Analyze economic calendar JSON file"""
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            events = data.get('events', [])
            
            if not events:
                return {'record_count': 0, 'sources': []}
            
            # Count by impact level
            impact_counts = {'high': 0, 'medium': 0, 'low': 0}
            currencies = set()
            sources = set()
            
            for event in events:
                impact = event.get('impact', 'low')
                if impact in impact_counts:
                    impact_counts[impact] += 1
                
                currency = event.get('currency')
                if currency:
                    currencies.add(currency)
                
                source = event.get('source', 'unknown')
                sources.add(source)
            
            return {
                'record_count': len(events),
                'impact_distribution': impact_counts,
                'currencies': list(currencies),
                'sources': list(sources),
                'date_range': data.get('date', 'unknown')
            }
            
        except Exception as e:
            return {'error': str(e), 'sources': []}
    
    def _analyze_regime_file(self, file_path: Path) -> Dict:
        """Analyze market regime JSON file"""
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            symbols = data.get('symbols', {})
            
            if not symbols:
                return {'record_count': 0, 'sources': []}
            
            # Count regimes
            regime_counts = {'bull': 0, 'bear': 0, 'range': 0, 'unknown': 0, 'error': 0}
            confidences = []
            
            for symbol_info in symbols.values():
                regime = symbol_info.get('regime', 'unknown')
                if regime in regime_counts:
                    regime_counts[regime] += 1
                
                confidence = symbol_info.get('confidence', 0)
                if confidence > 0:
                    confidences.append(confidence)
            
            return {
                'record_count': len(symbols),
                'regime_distribution': regime_counts,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'sources': ['technical_analysis'],
                'analysis_date': data.get('detection_date', 'unknown')
            }
            
        except Exception as e:
            return {'error': str(e), 'sources': []}
    
    def _generate_bundle_summary(self, files_info: Dict) -> Dict:
        """Generate overall bundle summary"""
        
        summary = {
            'total_files': len(files_info),
            'total_size_bytes': sum(info.get('size_bytes', 0) for info in files_info.values()),
            'data_freshness': 'unknown',
            'quality_score': 0.0
        }
        
        # Calculate data freshness (newest file age)
        newest_timestamp = None
        for info in files_info.values():
            modified_at = info.get('modified_at')
            if modified_at:
                try:
                    timestamp = datetime.fromisoformat(modified_at.replace('Z', '+00:00'))
                    if newest_timestamp is None or timestamp > newest_timestamp:
                        newest_timestamp = timestamp
                except ValueError:
                    continue
        
        if newest_timestamp:
            age_hours = (datetime.now(timezone.utc) - newest_timestamp).total_seconds() / 3600
            if age_hours < 1:
                summary['data_freshness'] = 'very_fresh'
            elif age_hours < 6:
                summary['data_freshness'] = 'fresh'
            elif age_hours < 24:
                summary['data_freshness'] = 'recent'
            else:
                summary['data_freshness'] = 'stale'
        
        # Calculate quality score based on file completeness and data volume
        quality_factors = []
        
        for filename, info in files_info.items():
            if info.get('error'):
                quality_factors.append(0.0)  # Error reduces quality
            elif info.get('record_count', 0) > 0:
                # Quality based on record count (more data = better)
                record_count = info['record_count']
                if 'news_sentiment' in filename:
                    quality_factors.append(min(record_count / 50.0, 1.0))  # Target 50 headlines
                elif 'econ_calendar' in filename:
                    quality_factors.append(min(record_count / 10.0, 1.0))  # Target 10 events
                elif 'market_regime' in filename:
                    quality_factors.append(min(record_count / 5.0, 1.0))   # Target 5 symbols
                else:
                    quality_factors.append(0.5)  # Default moderate quality
            else:
                quality_factors.append(0.1)  # Empty file, very low quality
        
        if quality_factors:
            summary['quality_score'] = sum(quality_factors) / len(quality_factors)
        
        return summary
    
    def _generate_bundle_metadata(self, bundle_path: Path, bundled_files: List[str], timestamp: datetime) -> Dict:
        """Generate comprehensive bundle metadata"""
        
        bundle_size = bundle_path.stat().st_size
        
        metadata = {
            'bundle_info': {
                'filename': bundle_path.name,
                'created_at': timestamp.isoformat(),
                'size_bytes': bundle_size,
                'size_mb': bundle_size / (1024 * 1024),
                'file_count': len(bundled_files),
                'format_version': '1.0'
            },
            'files_included': bundled_files,
            'checksum': self._calculate_file_checksum(bundle_path),
            'distribution_info': {
                'target': 'child_bots',
                'update_frequency': 'hourly',
                'retention_days': 7
            },
            'usage_instructions': {
                'extraction': 'Unzip to local intel/ directory',
                'validation': 'Check bundle_manifest.json for file integrity',
                'refresh_frequency': 'Check for updates every hour'
            }
        }
        
        return metadata
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        
        md5_hash = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        return md5_hash.hexdigest()
    
    def _log_bundle_creation(self, metadata: Dict):
        """Log bundle creation event"""
        
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'bundle_created',
            'bundle_file': metadata['bundle_info']['filename'],
            'size_mb': metadata['bundle_info']['size_mb'],
            'file_count': metadata['bundle_info']['file_count'],
            'checksum': metadata['checksum']
        }
        
        log_file = Path("logs") / "intel_bundle.json"
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def cleanup_old_bundles(self, keep_days: int = 7) -> Dict:
        """Clean up old bundle files"""
        
        cutoff_time = datetime.now() - timedelta(days=keep_days)
        
        cleaned_files = []
        total_size_freed = 0
        
        # Find old bundle files
        for bundle_file in self.bundle_dir.glob("intel_bundle_*.zip"):
            if bundle_file.name == "intel_bundle.zip":  # Skip symlink
                continue
                
            file_mtime = datetime.fromtimestamp(bundle_file.stat().st_mtime)
            
            if file_mtime < cutoff_time:
                size = bundle_file.stat().st_size
                bundle_file.unlink()
                cleaned_files.append(bundle_file.name)
                total_size_freed += size
                
                # Also remove corresponding metadata file
                metadata_file = bundle_file.with_name(bundle_file.stem + "_metadata.json")
                if metadata_file.exists():
                    metadata_file.unlink()
        
        if cleaned_files:
            print(f"🧹 Cleaned up {len(cleaned_files)} old bundles ({total_size_freed / (1024*1024):.2f} MB freed)")
        
        return {
            'files_removed': len(cleaned_files),
            'size_freed_mb': total_size_freed / (1024 * 1024),
            'removed_files': cleaned_files
        }
    
    def get_bundle_status(self) -> Dict:
        """Get current bundle status"""
        
        latest_bundle = self._get_latest_bundle_path()
        
        status = {
            'bundle_exists': latest_bundle is not None and latest_bundle.exists(),
            'latest_bundle': str(latest_bundle) if latest_bundle else None,
            'bundle_age_hours': None,
            'required_files_status': {},
            'needs_update': self._bundle_needs_update()
        }
        
        if status['bundle_exists']:
            bundle_mtime = datetime.fromtimestamp(latest_bundle.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - bundle_mtime).total_seconds() / 3600
            status['bundle_age_hours'] = age_hours
        
        # Check required files
        for filename in self.required_files:
            file_path = self.intel_dir / filename
            status['required_files_status'][filename] = {
                'exists': file_path.exists(),
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
                'age_hours': None
            }
            
            if file_path.exists():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - file_mtime).total_seconds() / 3600
                status['required_files_status'][filename]['age_hours'] = age_hours
        
        return status

def load_config() -> Dict:
    """Load configuration"""
    config_file = Path("config/base_config.yaml")
    
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f).get('intel_bundle', {})
    
    return {}

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intel Bundle Packager (Level 6-D)")
    parser.add_argument('--force', action='store_true', help='Force bundle creation even if up to date')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old bundles')
    parser.add_argument('--status', action='store_true', help='Show bundle status')
    parser.add_argument('--keep-days', type=int, default=7, help='Days to keep old bundles')
    
    args = parser.parse_args()
    
    print("📦 Intel Bundle Packager (Level 6-D)")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Initialize packager
    packager = IntelBundlePackager(config)
    
    if args.status:
        # Show bundle status
        status = packager.get_bundle_status()
        print(f"\n📊 Bundle Status:")
        print(f"   Bundle exists: {'✅' if status['bundle_exists'] else '❌'}")
        
        if status['bundle_exists']:
            print(f"   Latest bundle: {Path(status['latest_bundle']).name}")
            print(f"   Bundle age: {status['bundle_age_hours']:.1f} hours")
        
        print(f"   Needs update: {'⚠️ Yes' if status['needs_update'] else '✅ No'}")
        
        print(f"\n📄 Required Files:")
        for filename, file_status in status['required_files_status'].items():
            exists_emoji = "✅" if file_status['exists'] else "❌"
            age_info = f"({file_status['age_hours']:.1f}h old)" if file_status['age_hours'] else ""
            print(f"   {exists_emoji} {filename} {age_info}")
        
        return
    
    if args.cleanup:
        # Clean up old bundles
        cleanup_result = packager.cleanup_old_bundles(args.keep_days)
        print(f"🧹 Cleanup complete: {cleanup_result['files_removed']} files removed")
        return
    
    # Create bundle
    result = packager.create_intel_bundle(force=args.force)
    
    if result['success']:
        if result['action'] == 'created':
            print(f"\n✅ Level 6-D Complete!")
            print(f"📦 Bundle: {Path(result['bundle_file']).name}")
            print(f"📄 Files: {len(result['files_included'])}")
            print(f"💾 Size: {result['bundle_size_mb']:.2f} MB")
            print(f"🔗 Latest: {Path(result['latest_bundle']).name}")
        elif result['action'] == 'skipped':
            print(f"✅ Bundle up to date: {Path(result['bundle_file']).name}")
    else:
        print(f"❌ Bundle creation failed: {result.get('reason', 'unknown')}")
        if result.get('missing_files'):
            print(f"   Missing: {', '.join(result['missing_files'])}")
        exit(1)

if __name__ == "__main__":
    main()