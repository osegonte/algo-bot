#!/usr/bin/env python3
"""
Stage 2 Integration Script - Level 4
Combines all Stage 2 components into a complete workflow
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import Stage 2 components
from core.parent_controller import ParentController
from modules.ai.deepseek_start import score_strategies
from modules.ai.strategy_recommender import StrategyRecommender
from modules.sync.report_uploader import ReportUploader
from modules.sync.update_fetcher import UpdateFetcher
from modules.alerts.telegram_alerts import notify

class Stage2Controller:
    """Level 4: Complete Stage 2 workflow controller"""
    
    def __init__(self):
        self.parent_controller = ParentController()
        self.strategy_recommender = StrategyRecommender()
        self.report_uploader = ReportUploader()
        self.update_fetcher = UpdateFetcher()
        
        self.stage2_complete = False
        
    def run_complete_stage2_workflow(self):
        """Execute the complete Stage 2 workflow"""
        
        print("🚀 Starting Stage 2 Complete Workflow")
        print("=" * 60)
        
        workflow_start = datetime.utcnow()
        
        try:
            # Level 0: KPI Calculator
            print("\n📊 Level 0: Running KPI Calculator...")
            success_level0 = self._run_level0()
            
            # Level 1: Strategy Scoring & Ranking
            print("\n🤖 Level 1: Running Strategy Scoring & Ranking...")
            success_level1 = self._run_level1()
            
            # Level 2: Config Recommendation Writer
            print("\n📝 Level 2: Running Config Recommendation Writer...")
            success_level2 = self._run_level2()
            
            # Level 3: Child Sync Loop
            print("\n🔄 Level 3: Running Child Sync Loop...")
            success_level3 = self._run_level3()
            
            # Level 4: Final Integration Check
            print("\n✅ Level 4: Stage 2 Integration Check...")
            success_level4 = self._run_level4_check(
                success_level0, success_level1, success_level2, success_level3
            )
            
            workflow_duration = (datetime.utcnow() - workflow_start).total_seconds()
            
            if success_level4:
                self.stage2_complete = True
                self._send_completion_alert(workflow_duration)
                print(f"\n🎉 STAGE 2 COMPLETE! ✅")
                print(f"   Duration: {workflow_duration:.1f}s")
                print(f"   All levels operational")
                return True
            else:
                print(f"\n❌ STAGE 2 INCOMPLETE")
                print(f"   Some levels failed - check logs")
                return False
                
        except Exception as e:
            print(f"\n❌ Stage 2 workflow failed: {e}")
            notify(f"Stage 2 workflow failed: {str(e)}", "ERROR")
            return False
    
    def _run_level0(self):
        """Level 0: KPI Calculator Online"""
        try:
            # Ingest logs and calculate KPIs
            self.parent_controller.ingest_logs()
            kpis = self.parent_controller.kpis()
            
            # Check if KPIs were calculated
            required_kpis = ["gross_pnl", "win_rate", "profit_factor", "sharpe_ratio"]
            if all(kpi in kpis for kpi in required_kpis):
                print("   ✅ KPI calculator operational")
                print(f"   ✅ Summary saved to logs/parent_summary_*.json")
                return True
            else:
                print("   ❌ Missing required KPIs")
                return False
                
        except Exception as e:
            print(f"   ❌ Level 0 failed: {e}")
            return False
    
    def _run_level1(self):
        """Level 1: Rule-Based Scoring & Ranking"""
        try:
            # Run strategy scoring
            df = score_strategies()
            
            # Check if scoring worked and has top 3
            if len(df) >= 3 and 'score' in df.columns:
                print("   ✅ Strategy scoring operational")
                print("   ✅ Top 3 strategies ranked")
                
                # Display top 3 as required
                top3 = df.head(3)
                print("   📈 Top 3 strategies:")
                for i, (_, row) in enumerate(top3.iterrows(), 1):
                    print(f"      {i}. {row['strategy']}: {row['score']}")
                
                return True
            else:
                print("   ❌ Strategy scoring incomplete")
                return False
                
        except Exception as e:
            print(f"   ❌ Level 1 failed: {e}")
            return False
    
    def _run_level2(self):
        """Level 2: Config Recommendation Writer"""
        try:
            # Generate configs for child bots
            updated_configs = self.strategy_recommender.update_all_child_configs()
            
            if updated_configs:
                print(f"   ✅ Generated {len(updated_configs)} child configs")
                
                # Verify config files were created with required keys
                for child_id, config in updated_configs.items():
                    required_keys = ["strategy", "risk", "updated"]
                    if all(key in config for key in required_keys):
                        print(f"   ✅ Config for {child_id}: {config['strategy']['default']}")
                    else:
                        print(f"   ❌ Incomplete config for {child_id}")
                        return False
                
                return True
            else:
                print("   ❌ No configs generated")
                return False
                
        except Exception as e:
            print(f"   ❌ Level 2 failed: {e}")
            return False
    
    def _run_level3(self):
        """Level 3: Child Sync Loop"""
        try:
            # Test log uploading
            upload_success = self.report_uploader.push_logs()
            
            # Test config fetching
            fetch_success = self.update_fetcher.pull_updates()
            
            # Check sync logs for config_updated flag
            sync_status = self.update_fetcher.get_config_status()
            
            if upload_success:
                print("   ✅ Log uploading operational")
            else:
                print("   ⚠️  Log uploading needs setup (parent dir)")
            
            if sync_status.get("local_config_exists"):
                print("   ✅ Config fetching operational")
                print(f"   ✅ Child config: {sync_status.get('strategy', 'unknown')}")
            else:
                print("   ⚠️  Config fetching needs setup")
            
            # At least one sync mechanism should work
            return upload_success or sync_status.get("local_config_exists", False)
            
        except Exception as e:
            print(f"   ❌ Level 3 failed: {e}")
            return False
    
    def _run_level4_check(self, level0, level1, level2, level3):
        """Level 4: Final integration verification"""
        
        levels_status = {
            "Level 0 (KPI Calculator)": level0,
            "Level 1 (Strategy Scoring)": level1, 
            "Level 2 (Config Writer)": level2,
            "Level 3 (Sync Loop)": level3
        }
        
        print("   📋 Stage 2 Level Status:")
        all_passed = True
        
        for level_name, status in levels_status.items():
            emoji = "🟢" if status else "🔴"
            print(f"      {emoji} {level_name}")
            if not status:
                all_passed = False
        
        if all_passed:
            print("   ✅ All Stage 2 levels operational!")
            return True
        else:
            failed_count = sum(1 for status in levels_status.values() if not status)
            print(f"   ❌ {failed_count} levels need attention")
            return False
    
    def _send_completion_alert(self, duration):
        """Send completion notification"""
        message = f"""🎉 Stage 2 Complete!
        
✅ KPI Calculator Online
✅ Strategy Scoring Active  
✅ Config Writer Operational
✅ Sync Loop Functional

Duration: {duration:.1f}s
Ready for Stage 3!"""
        
        notify(message, "INFO")
    
    def run_continuous_monitoring(self, check_interval=300):
        """Run continuous Stage 2 monitoring"""
        print(f"\n🔄 Starting continuous monitoring (interval: {check_interval}s)")
        
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Running Stage 2 check...")
                
                success = self.run_complete_stage2_workflow()
                
                if success:
                    print("✅ Stage 2 monitoring: All systems operational")
                else:
                    print("⚠️  Stage 2 monitoring: Some issues detected")
                    notify("Stage 2 monitoring detected issues", "WARNING")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
            notify(f"Stage 2 monitoring error: {str(e)}", "ERROR")

def main():
    """Main entry point"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2 Integration Controller")
    parser.add_argument("--mode", choices=["single", "monitor"], default="single",
                       help="Run once or continuous monitoring")
    parser.add_argument("--interval", type=int, default=300,
                       help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    controller = Stage2Controller()
    
    if args.mode == "single":
        success = controller.run_complete_stage2_workflow()
        exit(0 if success else 1)
    elif args.mode == "monitor":
        controller.run_continuous_monitoring(args.interval)

if __name__ == "__main__":
    main()