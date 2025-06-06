#!/usr/bin/env python3
"""
Level 8 Launcher - Alert & Monitoring Upgrade
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_component(component, args=None):
    """Run a Level 8 component"""
    
    components = {
        'alert_hub': 'modules/alerts/unified_alert_hub.py',
        'kpi_endpoint': 'modules/monitoring/kpi_endpoint.py',
        'stability_watch': 'modules/monitoring/stability_watch.py',
        'burn_in': 'scripts/burn_in_test.py',
        'test_complete': 'scripts/test_level8_complete.py'
    }
    
    if component not in components:
        print(f"❌ Unknown component: {component}")
        print(f"Available: {', '.join(components.keys())}")
        return False
    
    script_path = components[component]
    cmd = [sys.executable, script_path]
    
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Component {component} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Level 8 Component Launcher")
    parser.add_argument('component', choices=[
        'alert_hub', 'kpi_endpoint', 'stability_watch', 
        'burn_in', 'test_complete', 'dashboard'
    ], help='Component to run')
    parser.add_argument('--args', nargs='*', help='Arguments to pass to component')
    
    args = parser.parse_args()
    
    if args.component == 'dashboard':
        print("🌐 Dashboard available at: web/dashboard.html")
        print("💡 Serve with: python -m http.server 8080")
        return
    
    print(f"🚀 Running Level 8 component: {args.component}")
    success = run_component(args.component, args.args)
    
    if success:
        print(f"✅ {args.component} completed successfully")
    else:
        print(f"❌ {args.component} failed")

if __name__ == "__main__":
    main()
