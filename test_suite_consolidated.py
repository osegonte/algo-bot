#!/usr/bin/env python3
"""
Consolidated Test Suite for Trading Bot
Replaces multiple individual test files
"""

import sys
import argparse
from pathlib import Path

# Test modules (imported dynamically)
test_modules = {
    'level5': 'test_level5_complete.py',
    'level6': 'test_level6_complete.py', 
    'level7': 'test_level7_simple.py',
    'level8': 'test_level8_complete.py'
}

def run_test(test_name):
    """Run a specific test"""
    if test_name not in test_modules:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_modules.keys())}")
        return False
    
    module_file = test_modules[test_name]
    if not Path(module_file).exists():
        print(f"❌ Test file not found: {module_file}")
        return False
    
    print(f"🧪 Running {test_name} test...")
    try:
        # Import and run the test
        exec(open(module_file).read())
        return True
    except Exception as e:
        print(f"❌ Test {test_name} failed: {e}")
        return False

def run_all_tests():
    """Run all available tests"""
    results = {}
    
    for test_name in test_modules.keys():
        results[test_name] = run_test(test_name)
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} passed")
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {test_name}")
    
    return passed == total

def main():
    parser = argparse.ArgumentParser(description="Consolidated Trading Bot Test Suite")
    parser.add_argument('test', nargs='?', choices=list(test_modules.keys()) + ['all'], 
                       help='Test to run (or "all" for all tests)')
    
    args = parser.parse_args()
    
    if not args.test:
        print("🧪 Trading Bot Test Suite")
        print("Available tests:", ', '.join(test_modules.keys()))
        return
    
    if args.test == 'all':
        success = run_all_tests()
    else:
        success = run_test(args.test)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
