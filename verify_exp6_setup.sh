#!/bin/bash
# Verification script for Experiment 6 setup

echo "=== Experiment 6 Setup Verification ==="
echo

# Check files exist
echo "1. Checking required files..."
files=(
    "run_experiment.sh"
    "scripts/analyze_exp6.py"
    "EXPERIMENT_6_INSTRUCTIONS.md"
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
if [ -x "run_experiment.sh" ]; then
    echo "   ✓ run_experiment.sh is executable"
else
    echo "   ✗ run_experiment.sh not executable"
    echo "   Fix: chmod +x run_experiment.sh"
    all_good=false
fi

if [ -x "scripts/analyze_exp6.py" ]; then
    echo "   ✓ scripts/analyze_exp6.py is executable"
else
    echo "   ✓ scripts/analyze_exp6.py (can run with python3)"
fi

# Check Python syntax
echo
echo "3. Validating Python syntax..."
python3 -m py_compile scripts/analyze_exp6.py 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ analyze_exp6.py syntax valid"
else
    echo "   ✗ analyze_exp6.py has syntax errors"
    all_good=false
fi

python3 -m py_compile src/miniwob_steer.py 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ miniwob_steer.py syntax valid"
else
    echo "   ✗ miniwob_steer.py has syntax errors"
    all_good=false
fi

# Check bash syntax
echo
echo "4. Validating shell script syntax..."
bash -n run_experiment.sh 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ run_experiment.sh syntax valid"
else
    echo "   ✗ run_experiment.sh has syntax errors"
    all_good=false
fi

# Check results directory
echo
echo "5. Checking results directory..."
if [ -d "results" ]; then
    echo "   ✓ results/ exists"
else
    echo "   ✓ results/ will be created on first run"
fi

# Summary
echo
if [ "$all_good" = true ]; then
    echo "✅ All checks passed! Ready to run experiments."
    echo
    echo "Quick start:"
    echo "  ./run_experiment.sh 1    # Phase 1: Reproducibility (~3 hours)"
    echo "  ./run_experiment.sh 4    # Phase 4: Vector method comparison (~2 hours)"
    echo "  python3 scripts/analyze_exp6.py"
else
    echo "⚠️  Some issues found. Please fix before running experiments."
fi
