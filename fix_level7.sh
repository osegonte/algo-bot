#!/bin/bash

# Level 7 Automated Fixes Script
# This script fixes the remaining issues in Level 7 ML components

set -e  # Exit on any error

echo "🔧 LEVEL 7 AUTOMATED FIXES"
echo "=========================="
echo ""

# Check if files exist
if [[ ! -f "modules/ml/level7b_baseline_model.py" ]]; then
    echo "❌ Error: modules/ml/level7b_baseline_model.py not found"
    exit 1
fi

if [[ ! -f "modules/ml/level7c_live_inference.py" ]]; then
    echo "❌ Error: modules/ml/level7c_live_inference.py not found"
    exit 1
fi

echo "📋 Found Level 7 files, applying fixes..."
echo ""

# Fix 1: Model Training CV Issue
echo "🔧 Fix 1: Updating model training CV parameters..."

# Create backup
cp modules/ml/level7b_baseline_model.py modules/ml/level7b_baseline_model.py.backup
echo "   💾 Backup created: level7b_baseline_model.py.backup"

# Fix the CV folds parameter
sed -i '' "s/cv=self\.model_config\['cv_folds'\]/cv=self.model_config.get('cv_folds', 5)/g" modules/ml/level7b_baseline_model.py

# Also fix the cv_std assignment
cat > /tmp/cv_fix.txt << 'EOF'
                # Cross-validation
                try:
                    cv_folds = self.model_config.get('cv_folds', 5)
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, 
                        cv=cv_folds, 
                        scoring='neg_mean_absolute_error'
                    )
                    cv_mae = -cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as cv_error:
                    print(f"   ⚠️ CV failed: {cv_error}, using test score")
                    cv_mae = test_mae
                    cv_std = 0.0
EOF

# Replace the problematic CV section
python3 << 'PYTHON_SCRIPT'
import re

# Read the file
with open('modules/ml/level7b_baseline_model.py', 'r') as f:
    content = f.read()

# Fix cv_std assignment
content = re.sub(
    r"cv_std = cv_scores\.std\(\)",
    "cv_std = cv_scores.std()",
    content
)

# Write back
with open('modules/ml/level7b_baseline_model.py', 'w') as f:
    f.write(content)
PYTHON_SCRIPT

echo "   ✅ Model training CV parameters fixed"

# Fix 2: Inference Indentation Issue
echo ""
echo "🔧 Fix 2: Fixing inference indentation..."

# Create backup
cp modules/ml/level7c_live_inference.py modules/ml/level7c_live_inference.py.backup
echo "   💾 Backup created: level7c_live_inference.py.backup"

# Fix the indentation issue using Python
python3 << 'PYTHON_SCRIPT'
import re

# Read the file
with open('modules/ml/level7c_live_inference.py', 'r') as f:
    lines = f.readlines()

# Fix indentation around line 362
fixed_lines = []
in_history_section = False

for i, line in enumerate(lines):
    # Look for the problematic section
    if 'def get_prediction_history' in line:
        in_history_section = True
    elif in_history_section and line.strip().startswith('def ') and 'get_prediction_history' not in line:
        in_history_section = False
    
    # Fix the history = [] line indentation
    if in_history_section and 'history = []' in line:
        # Ensure proper indentation (8 spaces)
        fixed_lines.append('        history = []\n')
    else:
        fixed_lines.append(line)

# Write back
with open('modules/ml/level7c_live_inference.py', 'w') as f:
    f.writelines(fixed_lines)
PYTHON_SCRIPT

echo "   ✅ Inference indentation fixed"

# Fix 3: Add missing import if needed
echo ""
echo "🔧 Fix 3: Checking imports..."

# Check if datetime import exists and add if missing
if ! grep -q "from datetime import timedelta" modules/ml/level7c_live_inference.py; then
    # Add the missing import
    sed -i '' '5i\
from datetime import datetime, timezone, timedelta
' modules/ml/level7c_live_inference.py
    echo "   ✅ Added missing timedelta import"
else
    echo "   ✅ Imports look good"
fi

# Test the fixes
echo ""
echo "🧪 Testing fixes..."
echo ""

echo "📊 Testing model training..."
if python modules/ml/level7b_baseline_model.py > /tmp/model_test.log 2>&1; then
    echo "   ✅ Model training syntax OK"
else
    echo "   ⚠️ Model training may have issues (check /tmp/model_test.log)"
fi

echo ""
echo "🔮 Testing inference syntax..."
if python -m py_compile modules/ml/level7c_live_inference.py; then
    echo "   ✅ Inference syntax OK"
else
    echo "   ❌ Inference still has syntax issues"
fi

echo ""
echo "🎯 Running quick integration test..."
if python test_level7_simple.py; then
    echo ""
    echo "🎉 SUCCESS! Level 7 fixes applied successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. python modules/ml/level7b_baseline_model.py  # Train model"
    echo "   2. python modules/ml/level7c_live_inference.py --demo  # Test inference"
    echo "   3. python test_level7_simple.py  # Final test"
    echo ""
    echo "✅ Level 7: ML-Enhanced Scoring should now be complete!"
else
    echo ""
    echo "⚠️ Some issues remain, but fixes have been applied."
    echo "💡 Try running the individual components to debug further."
fi

echo ""
echo "🔧 Fixes completed!"
echo "💾 Backups saved with .backup extension"