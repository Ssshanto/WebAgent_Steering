#!/bin/bash
# Quick diagnostic test: Check baseline vs steered outputs for verbosity
#
# This runs a minimal test (50 episodes) to verify:
# 1. Baseline outputs are not verbose
# 2. Steered outputs are not verbose
# 3. Steering improves accuracy

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    DIAGNOSTIC TEST: STEERING BEHAVIOR                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Model: Qwen 2.5 0.5b"
echo "  Layer: 13"
echo "  Coefficient: 4.0"
echo "  Prompt: accuracy"
echo "  Vector Method: response"
echo "  Train Steps: 200"
echo "  Eval Steps: 100 (50 base + 50 steered)"
echo "  Seed: 0"
echo ""
echo "This will take ~5-10 minutes"
echo ""

OUTPUT_FILE="results/diagnostic_test.jsonl"

if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing existing diagnostic file..."
    rm "$OUTPUT_FILE"
fi

python src/miniwob_steer.py \
    --model-size 0.5b \
    --layer 13 \
    --coeff 4.0 \
    --prompt-type accuracy \
    --vector-method response \
    --tasks all \
    --train-steps 200 \
    --eval-steps 100 \
    --seed 0 \
    --out "$OUTPUT_FILE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                          ANALYZING OUTPUTS                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

python3 << 'PYEOF'
import json
import sys

try:
    with open('results/diagnostic_test.jsonl') as f:
        lines = [line.strip() for line in f if line.strip()]
        data = [json.loads(line) for line in lines]
except Exception as e:
    print(f"Error reading results: {e}")
    sys.exit(1)

print("=" * 80)
print("CHECKING FOR VERBOSE/MARKDOWN OUTPUTS")
print("=" * 80)

base_samples = [d for d in data if d.get('type') == 'base']
steered_samples = [d for d in data if d.get('type') == 'steered']

def analyze_output(output):
    """Analyze output for issues"""
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    has_markdown = '```' in output or output.count('#') > 2
    has_bullets = output.count('*') > 2 or output.count('-') > 2
    is_verbose = len(output) > 100
    is_multiline = len(lines) > 1
    is_cot = 'first' in output.lower() and 'then' in output.lower()
    
    return {
        'lines': len(lines),
        'chars': len(output),
        'markdown': has_markdown,
        'bullets': has_bullets,
        'verbose': is_verbose,
        'multiline': is_multiline,
        'cot': is_cot,
    }

print(f"\nBase samples: {len(base_samples)}")
print(f"Steered samples: {len(steered_samples)}")

# Analyze base
base_issues = {
    'markdown': 0,
    'verbose': 0,
    'multiline': 0,
    'cot': 0,
}

print("\n" + "-" * 80)
print("BASE OUTPUTS (first 5):")
print("-" * 80)
for i, entry in enumerate(base_samples[:5]):
    output = entry.get('output', '')
    analysis = analyze_output(output)
    print(f"\n[{i}] Task: {entry['task']}")
    print(f"    Output: {repr(output[:100])}")
    print(f"    Lines: {analysis['lines']}, Chars: {analysis['chars']}")
    
    if analysis['markdown']: base_issues['markdown'] += 1
    if analysis['verbose']: base_issues['verbose'] += 1
    if analysis['multiline']: base_issues['multiline'] += 1
    if analysis['cot']: base_issues['cot'] += 1

# Analyze steered
steered_issues = {
    'markdown': 0,
    'verbose': 0,
    'multiline': 0,
    'cot': 0,
}

print("\n" + "-" * 80)
print("STEERED OUTPUTS (first 5):")
print("-" * 80)
for i, entry in enumerate(steered_samples[:5]):
    output = entry.get('output', '')
    analysis = analyze_output(output)
    print(f"\n[{i}] Task: {entry['task']}")
    print(f"    Output: {repr(output[:100])}")
    print(f"    Lines: {analysis['lines']}, Chars: {analysis['chars']}")
    
    if analysis['markdown']: steered_issues['markdown'] += 1
    if analysis['verbose']: steered_issues['verbose'] += 1
    if analysis['multiline']: steered_issues['multiline'] += 1
    if analysis['cot']: steered_issues['cot'] += 1

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

base_success = sum(1 for d in base_samples if d.get('success', False))
steered_success = sum(1 for d in steered_samples if d.get('success', False))
base_accuracy = 100 * base_success / len(base_samples) if base_samples else 0
steered_accuracy = 100 * steered_success / len(steered_samples) if steered_samples else 0

print(f"\nAccuracy:")
print(f"  Base:    {base_accuracy:.1f}% ({base_success}/{len(base_samples)})")
print(f"  Steered: {steered_accuracy:.1f}% ({steered_success}/{len(steered_samples)})")
print(f"  Δ:       {steered_accuracy - base_accuracy:+.1f}%")

print(f"\nBase Issues (out of {len(base_samples)} samples):")
print(f"  Markdown:   {base_issues['markdown']}")
print(f"  Verbose:    {base_issues['verbose']}")
print(f"  Multi-line: {base_issues['multiline']}")
print(f"  CoT:        {base_issues['cot']}")

print(f"\nSteered Issues (out of {len(steered_samples)} samples):")
print(f"  Markdown:   {steered_issues['markdown']}")
print(f"  Verbose:    {steered_issues['verbose']}")
print(f"  Multi-line: {steered_issues['multiline']}")
print(f"  CoT:        {steered_issues['cot']}")

print("\n" + "=" * 80)
if steered_issues['markdown'] > 0 or steered_issues['verbose'] > 5:
    print("⚠️  ISSUE DETECTED: Steered outputs show verbosity/markdown")
    print("    → Problem exists, requires debugging")
elif base_issues['verbose'] > 5:
    print("⚠️  ISSUE DETECTED: Base outputs are verbose")
    print("    → Problem is in prompt template, not steering")
else:
    print("✓ NO ISSUES DETECTED: Outputs are clean and concise")
    print("  → Steering is working correctly")
print("=" * 80)

PYEOF

echo ""
echo "Diagnostic test complete. Results saved to: $OUTPUT_FILE"
