#!/bin/bash
# Verification script for Experiment 10 setup

echo "=== Experiment 10 Setup Verification ==="
echo

# Check files exist
echo "1. Checking required files..."
files=(
    "run_exp10_expanded.sh"
    "scripts/analyze_exp10.py"
    "scripts/extract_ground_truth_expanded.py"
    "EXPERIMENT_10_INSTRUCTIONS.md"
    "src/miniwob_steer.py"
)

all_good=true
for f in "${files[@]}"; do
    if [ -f "$f" ]; then
        echo "   ✓ $f"
    else
        echo "   ✗ $f - MISSING"
        all_good=false
    fi
done

# Check executability
echo
echo "2. Checking executability..."
if [ -x "run_exp10_expanded.sh" ]; then
    echo "   ✓ run_exp10_expanded.sh is executable"
else
    echo "   ✗ run_exp10_expanded.sh not executable"
    all_good=false
fi

# Check Python syntax
echo
echo "3. Validating Python syntax..."
python3 -m py_compile scripts/analyze_exp10.py 2>&1 > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ analyze_exp10.py syntax valid"
else
    echo "   ✗ analyze_exp10.py has syntax errors"
    all_good=false
fi

python3 -m py_compile scripts/extract_ground_truth_expanded.py 2>&1 > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ extract_ground_truth_expanded.py syntax valid"
else
    echo "   ✗ extract_ground_truth_expanded.py has syntax errors"
    all_good=false
fi

python3 -m py_compile src/miniwob_steer.py 2>&1 > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ miniwob_steer.py syntax valid"
else
    echo "   ✗ miniwob_steer.py has syntax errors"
    all_good=false
fi

# Check bash syntax
echo
echo "4. Validating shell script syntax..."
bash -n run_exp10_expanded.sh 2>&1 > /dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ run_exp10_expanded.sh syntax valid"
else
    echo "   ✗ run_exp10_expanded.sh has syntax errors"
    all_good=false
fi

# Check task configuration
echo
echo "5. Checking task configuration..."
if grep -q "EXPANDED_TASKS" src/miniwob_steer.py; then
    echo "   ✓ EXPANDED_TASKS defined"
else
    echo "   ✗ EXPANDED_TASKS not found"
    all_good=false
fi

if grep -q "TASK_CATEGORIES" src/miniwob_steer.py; then
    echo "   ✓ TASK_CATEGORIES defined"
else
    echo "   ✗ TASK_CATEGORIES not found"
    all_good=false
fi

# Summary
echo
if [ "$all_good" = true ]; then
    echo "✅ All checks passed! Ready to run Experiment 10."
    echo
    echo "Quick start:"
    echo "  ./run_exp10_expanded.sh"
    echo "  python3 scripts/analyze_exp10.py"
else
    echo "⚠️  Some issues found. Please fix before running."
fi
