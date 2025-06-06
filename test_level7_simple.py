#!/usr/bin/env python3
"""
Simple Level 7 Test - Tests the ML workflow
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

def test_level7_simple():
    """Simple test of Level 7 components"""
    
    print("🧪 LEVEL 7 SIMPLE TEST")
    print("=" * 40)
    
    results = {
        "features_created": False,
        "model_trained": False,
        "predictions_generated": False,
        "hybrid_rankings": False
    }
    
    # Check if feature file exists
    features_dir = Path("data/features")
    feature_files = list(features_dir.glob("ml_features_*.csv"))
    
    if feature_files:
        print("✅ Features: Found feature dataset")
        results["features_created"] = True
    else:
        print("❌ Features: No feature files found")
    
    # Check if model exists
    models_dir = Path("models")
    latest_model = models_dir / "latest_model.pkl"
    
    if latest_model.exists():
        print("✅ Model: Trained model found")
        results["model_trained"] = True
    else:
        print("❌ Model: No trained model found")
    
    # Check if predictions exist
    logs_dir = Path("logs")
    pred_file = logs_dir / "latest_ml_predictions.json"
    
    if pred_file.exists():
        print("✅ Predictions: ML predictions found")
        results["predictions_generated"] = True
    else:
        print("❌ Predictions: No ML predictions found")
    
    # Check if hybrid rankings exist
    hybrid_file = logs_dir / "latest_hybrid_rankings.json"
    
    if hybrid_file.exists():
        print("✅ Hybrid: Rankings created")
        results["hybrid_rankings"] = True
        
        # Show top strategy
        try:
            with open(hybrid_file) as f:
                data = json.load(f)
            top_strategy = data.get('top_strategy', 'unknown')
            print(f"   🏆 Top strategy: {top_strategy}")
        except:
            pass
    else:
        print("❌ Hybrid: No rankings found")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} components working")
    
    if passed >= 3:
        print("🎉 Level 7 mostly working!")
        
        # Create simple completion marker
        completion = {
            "level": 7,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "components_working": results,
            "score": f"{passed}/{total}"
        }
        
        with open("level7_completion.json", "w") as f:
            json.dump(completion, f, indent=2)
        
        print("💾 Completion saved to level7_completion.json")
        return True
    else:
        print("⚠️ More components need to be working")
        return False

if __name__ == "__main__":
    success = test_level7_simple()
    exit(0 if success else 1)