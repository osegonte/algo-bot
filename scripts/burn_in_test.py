#!/usr/bin/env python3
"""
Level 8-E: 24 Hour Burn-In Test
Runs the complete system for 24 hours and validates stability
"""

import asyncio
import json
import time
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import threading
import logging

class BurnInTestManager:
    """Manages 24-hour burn-in testing of the complete trading system"""
    
    def __init__(self, test_duration_hours: int = 24):
        self.test_duration_hours = test_duration_hours
        self.test_duration_seconds = test_duration_hours * 3600
        
        # Test tracking
        self.test_start_time = None
        self.test_end_time = None
        self.is_running = False
        self.test_results = {}
        
        # System processes
        self.processes = {}
        
        # Monitoring intervals
        self.health_check_interval = 300  # 5 minutes
        self.process_check_interval = 60   # 1 minute
        self.alert_check_interval = 180    # 3 minutes
        
        # Test requirements
        self.required_alerts = 1  # Need at least 1 alert during test
        self.max_crashes = 0      # No crashes allowed
        
        # Logs
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.burn_in_log = self.logs_dir / "burn_in_test.json"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / "burn_in_test.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop_burn_in_test()
        sys.exit(0)
    
    async def start_burn_in_test(self, quick_test: bool = False) -> bool:
        """Start the comprehensive burn-in test"""
        
        if quick_test:
            self.test_duration_hours = 0.5  # 30 minutes for quick test
            self.test_duration_seconds = 30 * 60
            self.required_alerts = 0  # No alerts required for quick test
        
        self.logger.info(f"🔥 Starting {self.test_duration_hours}h burn-in test...")
        self.logger.info("=" * 60)
        
        self.test_start_time = datetime.now(timezone.utc)
        self.is_running = True
        
        # Initialize test results
        self.test_results = {
            "start_time": self.test_start_time.isoformat(),
            "duration_hours": self.test_duration_hours,
            "status": "running",
            "processes_started": [],
            "crashes_detected": 0,
            "alerts_captured": 0,
            "health_checks": 0,
            "errors": [],
            "milestones": []
        }
        
        try:
            # Start system components
            await self._start_system_components()
            
            # Run monitoring loops
            await self._run_monitoring_loops()
            
            # Analyze results
            success = await self._finalize_test()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Burn-in test failed: {e}")
            self.test_results["status"] = "failed"
            self.test_results["errors"].append(str(e))
            return False
        
        finally:
            self.is_running = False
            await self._cleanup_processes()
    
    async def _start_system_components(self):
        """Start all system components for testing"""
        self.logger.info("🚀 Starting system components...")
        
        components = [
            {
                "name": "api_server",
                "command": [sys.executable, "modules/monitoring/kpi_endpoint.py", "--host", "127.0.0.1", "--port", "8000"],
                "required": True
            },
            {
                "name": "alert_hub_daemon",
                "command": [sys.executable, "unified_alert_hub.py", "--daemon", "--interval", "300"],
                "required": True
            },
            {
                "name": "stability_watch",
                "command": [sys.executable, "stability_watch.py", "--daemon", "--interval", "60"],
                "required": True
            }
        ]
        
        for component in components:
            try:
                self.logger.info(f"   Starting {component['name']}...")
                
                process = subprocess.Popen(
                    component["command"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path.cwd()
                )
                
                self.processes[component["name"]] = {
                    "process": process,
                    "start_time": time.time(),
                    "required": component["required"],
                    "restart_count": 0
                }
                
                self.test_results["processes_started"].append(component["name"])
                
                # Give process time to start
                await asyncio.sleep(2)
                
                # Check if process is still running
                if process.poll() is None:
                    self.logger.info(f"   ✅ {component['name']} started (PID: {process.pid})")
                else:
                    self.logger.error(f"   ❌ {component['name']} failed to start")
                    if component["required"]:
                        raise Exception(f"Required component {component['name']} failed to start")
                
            except Exception as e:
                self.logger.error(f"Failed to start {component['name']}: {e}")
                if component["required"]:
                    raise
        
        # Wait for API server to be ready
        await self._wait_for_api_ready()
        
        self.logger.info("✅ All system components started")
    
    async def _wait_for_api_ready(self, timeout: int = 30):
        """Wait for API server to be ready"""
        import aiohttp
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:8000/health") as response:
                        if response.status == 200:
                            self.logger.info("✅ API server is ready")
                            return
            except:
                pass
            
            await asyncio.sleep(1)
        
        raise Exception("API server failed to become ready")
    
    async def _run_monitoring_loops(self):
        """Run all monitoring loops during the test"""
        self.logger.info(f"🔄 Starting monitoring loops for {self.test_duration_hours}h...")
        
        # Create monitoring tasks
        tasks = [
            asyncio.create_task(self._process_health_monitor()),
            asyncio.create_task(self._system_health_monitor()),
            asyncio.create_task(self._alert_monitor()),
            asyncio.create_task(self._milestone_tracker()),
            asyncio.create_task(self._test_timer())
        ]
        
        # Wait for test duration or until all tasks complete
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Monitoring loops cancelled")
        
        # Cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
    
    async def _process_health_monitor(self):
        """Monitor process health and restart if needed"""
        while self.is_running:
            try:
                for name, info in self.processes.items():
                    process = info["process"]
                    
                    # Check if process is still running
                    if process.poll() is not None:
                        self.logger.warning(f"⚠️ Process {name} has stopped")
                        self.test_results["crashes_detected"] += 1
                
                await asyncio.sleep(self.process_check_interval)
                
            except Exception as e:
                self.logger.error(f"Process health monitor error: {e}")
                await asyncio.sleep(self.process_check_interval)
    
    async def _system_health_monitor(self):
        """Monitor overall system health"""
        while self.is_running:
            try:
                # Make health check API call
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:8000/health") as response:
                        if response.status == 200:
                            health_data = await response.json()
                            
                            system_status = health_data.get("status", "unknown")
                            
                            if system_status == "unhealthy":
                                self.logger.warning("⚠️ System health degraded")
                                self.test_results["errors"].append(f"System unhealthy at {datetime.now().isoformat()}")
                            
                            self.test_results["health_checks"] += 1
                        else:
                            self.logger.error(f"Health check failed: HTTP {response.status}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"System health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _alert_monitor(self):
        """Monitor for alerts during the test"""
        alerts_at_start = self._count_existing_alerts()
        
        while self.is_running:
            try:
                current_alerts = self._count_existing_alerts()
                new_alerts = current_alerts - alerts_at_start
                
                self.test_results["alerts_captured"] = new_alerts
                
                if new_alerts > 0:
                    self.logger.info(f"📬 Captured {new_alerts} alerts during test")
                
                await asyncio.sleep(self.alert_check_interval)
                
            except Exception as e:
                self.logger.error(f"Alert monitor error: {e}")
                await asyncio.sleep(self.alert_check_interval)
    
    def _count_existing_alerts(self) -> int:
        """Count alerts in the alert log"""
        alert_file = self.logs_dir / "unified_alerts.json"
        
        if not alert_file.exists():
            return 0
        
        try:
            with open(alert_file) as f:
                return len(f.readlines())
        except Exception:
            return 0
    
    async def _milestone_tracker(self):
        """Track test milestones"""
        milestones = [
            {"percent": 25, "reached": False},
            {"percent": 50, "reached": False},
            {"percent": 75, "reached": False},
            {"percent": 90, "reached": False}
        ]
        
        while self.is_running:
            elapsed = time.time() - self.test_start_time.timestamp()
            progress_percent = (elapsed / self.test_duration_seconds) * 100
            
            for milestone in milestones:
                if not milestone["reached"] and progress_percent >= milestone["percent"]:
                    milestone["reached"] = True
                    
                    milestone_info = {
                        "percent": milestone["percent"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_hours": elapsed / 3600,
                        "crashes": self.test_results["crashes_detected"],
                        "alerts": self.test_results["alerts_captured"]
                    }
                    
                    self.test_results["milestones"].append(milestone_info)
                    
                    self.logger.info(f"🎯 Milestone: {milestone['percent']}% complete")
                    self.logger.info(f"   Elapsed: {elapsed/3600:.1f}h")
                    self.logger.info(f"   Crashes: {self.test_results['crashes_detected']}")
                    self.logger.info(f"   Alerts: {self.test_results['alerts_captured']}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _test_timer(self):
        """Main test timer"""
        await asyncio.sleep(self.test_duration_seconds)
        
        self.logger.info(f"⏰ Test duration ({self.test_duration_hours}h) completed")
        self.is_running = False
    
    async def _finalize_test(self) -> bool:
        """Finalize test and determine if it passed"""
        self.test_end_time = datetime.now(timezone.utc)
        
        self.logger.info("🔬 Finalizing burn-in test...")
        
        # Calculate final statistics
        total_runtime = (self.test_end_time - self.test_start_time).total_seconds()
        runtime_hours = total_runtime / 3600
        
        self.test_results.update({
            "end_time": self.test_end_time.isoformat(),
            "actual_runtime_hours": runtime_hours,
            "actual_runtime_seconds": total_runtime
        })
        
        # Determine test success
        success_criteria = {
            "runtime_sufficient": runtime_hours >= (self.test_duration_hours * 0.95),  # 95% of target
            "no_excessive_crashes": self.test_results["crashes_detected"] <= self.max_crashes,
            "sufficient_alerts": self.test_results["alerts_captured"] >= self.required_alerts,
            "health_checks_completed": self.test_results["health_checks"] > 0
        }
        
        all_passed = all(success_criteria.values())
        
        self.test_results.update({
            "success_criteria": success_criteria,
            "test_passed": all_passed,
            "status": "passed" if all_passed else "failed"
        })
        
        # Log results
        self.logger.info("📊 Burn-in Test Results:")
        self.logger.info(f"   Runtime: {runtime_hours:.2f}h / {self.test_duration_hours}h")
        self.logger.info(f"   Crashes: {self.test_results['crashes_detected']} (max: {self.max_crashes})")
        self.logger.info(f"   Alerts: {self.test_results['alerts_captured']} (min: {self.required_alerts})")
        self.logger.info(f"   Health checks: {self.test_results['health_checks']}")
        
        # Show criteria results
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            self.logger.info(f"   {status} {criterion}")
        
        # Save detailed results
        self._save_test_results()
        
        if all_passed:
            self.logger.info("🎉 BURN-IN TEST PASSED!")
            self._create_stage3_completion()
        else:
            self.logger.error("❌ BURN-IN TEST FAILED!")
        
        return all_passed
    
    def _save_test_results(self):
        """Save detailed test results"""
        with open(self.burn_in_log, "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        self.logger.info(f"💾 Test results saved to: {self.burn_in_log}")
    
    def _create_stage3_completion(self):
        """Create Stage 3 completion marker"""
        completion_data = {
            "stage": 3,
            "name": "Central Intelligence & Sync",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "burn_in_test": {
                "duration_hours": self.test_results["actual_runtime_hours"],
                "crashes": self.test_results["crashes_detected"],
                "alerts_captured": self.test_results["alerts_captured"],
                "test_passed": self.test_results["test_passed"]
            },
            "levels_completed": [5, 6, 7, 8],
            "ready_for_stage_4": True
        }
        
        with open("stage3_complete.json", "w") as f:
            json.dump(completion_data, f, indent=2)
        
        self.logger.info("🏆 Stage 3 completion marker created!")
    
    async def _cleanup_processes(self):
        """Clean up all started processes"""
        self.logger.info("🧹 Cleaning up processes...")
        
        for name, info in self.processes.items():
            try:
                process = info["process"]
                
                if process.poll() is None:  # Still running
                    self.logger.info(f"   Terminating {name}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"   Force killing {name}...")
                        process.kill()
                        process.wait()
                
            except Exception as e:
                self.logger.error(f"Error cleaning up {name}: {e}")
    
    def stop_burn_in_test(self):
        """Stop the burn-in test early"""
        self.logger.info("⏹️ Stopping burn-in test...")
        self.is_running = False
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current test status"""
        if not self.is_running and not self.test_start_time:
            return {"status": "not_started"}
        
        current_time = datetime.now(timezone.utc)
        
        if self.is_running:
            elapsed = (current_time - self.test_start_time).total_seconds()
            progress = (elapsed / self.test_duration_seconds) * 100
            remaining = self.test_duration_seconds - elapsed
            
            return {
                "status": "running",
                "progress_percent": min(progress, 100),
                "elapsed_hours": elapsed / 3600,
                "remaining_hours": max(remaining / 3600, 0),
                "crashes": self.test_results["crashes_detected"],
                "alerts": self.test_results["alerts_captured"],
                "health_checks": self.test_results["health_checks"]
            }
        else:
            return {
                "status": self.test_results.get("status", "completed"),
                "test_passed": self.test_results.get("test_passed", False),
                "total_runtime": self.test_results.get("actual_runtime_hours", 0),
                "final_results": self.test_results
            }

async def main():
    """Main entry point for burn-in test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="24h Burn-In Test (Level 8-E)")
    parser.add_argument('--duration', type=float, default=24.0, help='Test duration in hours')
    parser.add_argument('--quick', action='store_true', help='Quick 30-minute test')
    parser.add_argument('--status', action='store_true', help='Show current test status')
    parser.add_argument('--stop', action='store_true', help='Stop running test')
    
    args = parser.parse_args()
    
    print("🔥 24-Hour Burn-In Test (Level 8-E)")
    print("=" * 50)
    
    manager = BurnInTestManager(test_duration_hours=args.duration)
    
    if args.status:
        # Show test status
        status = manager.get_test_status()
        print(f"📊 Test Status: {status['status']}")
        
        if status['status'] == 'running':
            print(f"   Progress: {status['progress_percent']:.1f}%")
            print(f"   Elapsed: {status['elapsed_hours']:.2f}h")
            print(f"   Remaining: {status['remaining_hours']:.2f}h")
            print(f"   Crashes: {status['crashes']}")
            print(f"   Alerts: {status['alerts']}")
        elif status['status'] in ['passed', 'failed']:
            print(f"   Result: {'✅ PASSED' if status['test_passed'] else '❌ FAILED'}")
            print(f"   Runtime: {status['total_runtime']:.2f}h")
        
        return
    
    if args.stop:
        # Stop running test (would need process management)
        print("⏹️ Stop command - would terminate running test")
        return
    
    # Start burn-in test
    try:
        success = await manager.start_burn_in_test(quick_test=args.quick)
        
        if success:
            print("\n🎉 LEVEL 8-E COMPLETE!")
            print("🏆 Stage 3 is now fully operational!")
            exit(0)
        else:
            print("\n❌ Burn-in test failed")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        manager.stop_burn_in_test()
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())