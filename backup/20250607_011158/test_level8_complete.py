#!/usr/bin/env python3
"""
Level 8 Complete Integration Test
"""

import sys
import json
import time
import requests
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timezone

def test_level_8_complete():
    print("🧪 LEVEL 8 COMPLETE INTEGRATION TEST")
    print("=" * 60)
    
    results = {
        "8A_unified_alert_hub": False,
        "8B_live_kpi_endpoint": False,
        "8C_mini_dashboard": False,
        "8D_stability_watch": False,
        "8E_burn_in_ready": False
    }
    
    # Test 8-A: Unified Alert Hub
    print("\n🔔 Testing 8-A: Unified Alert Hub")
    print("-" * 40)
    try:
        from unified_alert_hub import UnifiedAlertHub
        hub = UnifiedAlertHub()
        success = hub.send_alert("Test Alert", "Test message", "INFO", "test")
        if success:
            print("✅ Alert hub can send alerts")
            results["8A_unified_alert_hub"] = True
    except Exception as e:
        print(f"❌ 8-A test failed: {e}")
    
    # Test 8-B: Live KPI Endpoint
    print("\n📡 Testing 8-B: Live KPI Endpoint")
    print("-" * 40)
    try:
        # Test the KPI service directly
        sys.path.append(str(Path("modules/monitoring")))
        from kpi_endpoint import LiveKPIService
        service = LiveKPIService()
        kpis = service.get_latest_kpis()
        if kpis:
            print("✅ KPI service works")
            results["8B_live_kpi_endpoint"] = True
    except Exception as e:
        print(f"❌ 8-B test failed: {e}")
    
    # Test 8-C: Mini Dashboard
    print("\n📊 Testing 8-C: Mini Dashboard")
    print("-" * 40)
    dashboard_file = Path("web/dashboard.html")
    if dashboard_file.exists():
        print("✅ Dashboard file exists")
        results["8C_mini_dashboard"] = True
    else:
        print("❌ Dashboard file missing")
    
    # Test 8-D: Stability Watch
    print("\n👁️ Testing 8-D: Stability Watch")
    print("-" * 40)
    try:
        from stability_watch import StabilityWatcher
        watcher = StabilityWatcher()
        print("✅ Stability watcher created")
        results["8D_stability_watch"] = True
    except Exception as e:
        print(f"❌ 8-D test failed: {e}")
    
    # Test 8-E: 24h Burn-In Readiness
    print("\n🔥 Testing 8-E: 24h Burn-In Readiness")
    print("-" * 40)
    try:
        from burn_in_test import BurnInTestManager
        manager = BurnInTestManager()
        print("✅ Burn-in test manager created")
        results["8E_burn_in_ready"] = True
    except Exception as e:
        print(f"❌ 8-E test failed: {e}")
    
    # Results
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 LEVEL 8 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Level 8 Score: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 LEVEL 8 COMPLETE! ✅")
        print("🏆 STAGE 3 READY FOR COMPLETION!")
        
        completion_status = {
            "level": 8,
            "name": "Alert & Monitoring Upgrade",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "score": f"{passed_tests}/{total_tests}",
            "tests_passed": results,
            "ready_for_burn_in": True
        }
        
        with open("level8_completion.json", "w") as f:
            json.dump(completion_status, f, indent=2)
        
        return True
    
    return False

if __name__ == "__main__":
    success = test_level_8_complete()
    exit(0 if success else 1)
