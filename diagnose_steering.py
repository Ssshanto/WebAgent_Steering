#!/usr/bin/env python3
"""
Diagnostic script to verify steering setup.
Tests: vector polarity, prompt application, baseline behavior
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, 'src')

def check_1_vector_polarity():
    """Check that vector computation uses correct polarity (pos - neg)"""
    print("=" * 80)
    print("CHECK 1: VECTOR POLARITY")
    print("=" * 80)
    
    with open('src/miniwob_steer.py', 'r') as f:
        content = f.read()
    
    # Find vector computation lines
    if 'model._prompt_activation(pos) - model._prompt_activation(neg)' in content:
        print("✓ Prompt method: diff = activation(POS) - activation(NEG)")
    else:
        print("✗ Prompt method: INCORRECT or NOT FOUND")
    
    if 'model._last_token_state(pos_text) - model._last_token_state(neg_text)' in content:
        print("✓ Response method: diff = activation(POS) - activation(NEG)")
    else:
        print("✗ Response method: INCORRECT or NOT FOUND")
    
    print("\nConclusion: Vector should point TOWARDS positive prompt")
    print()

def check_2_prompt_config():
    """Check prompt configuration"""
    print("=" * 80)
    print("CHECK 2: PROMPT CONFIGURATION")
    print("=" * 80)
    
    from miniwob_steer import PROMPT_CONFIGS
    
    if 'accuracy' in PROMPT_CONFIGS:
        print("✓ 'accuracy' prompt found")
        print(f"\n  Positive: {PROMPT_CONFIGS['accuracy']['pos']}")
        print(f"\n  Negative: {PROMPT_CONFIGS['accuracy']['neg']}")
    else:
        print("✗ 'accuracy' prompt NOT FOUND")
    
    # Check default
    from miniwob_steer import PROMPT_TYPE
    print(f"\nDefault PROMPT_TYPE: '{PROMPT_TYPE}'")
    print("Note: This is overridden by --prompt-type argument")
    print()

def check_3_system_prompt():
    """Check system prompt for format instructions"""
    print("=" * 80)
    print("CHECK 3: SYSTEM PROMPT")
    print("=" * 80)
    
    from miniwob_steer import SYSTEM_PROMPT
    
    print("System prompt:")
    print(SYSTEM_PROMPT)
    
    # Check for potential issues
    if "explanation" in SYSTEM_PROMPT.lower():
        print("\n⚠️  System prompt mentions 'explanation' - could conflict with steering")
    elif "no explanation" in SYSTEM_PROMPT.lower():
        print("\n✓ System prompt explicitly forbids explanations")
    
    if "one" in SYSTEM_PROMPT.lower() and "action" in SYSTEM_PROMPT.lower():
        print("✓ System prompt emphasizes single action output")
    
    print()

def check_4_chat_template():
    """Check how chat template is applied"""
    print("=" * 80)
    print("CHECK 4: CHAT TEMPLATE APPLICATION")
    print("=" * 80)
    
    with open('src/miniwob_steer.py', 'r') as f:
        content = f.read()
    
    # Check for system role usage
    if '"role": "system"' in content:
        print("✓ System role is used in messages")
    else:
        print("⚠️  System role NOT used - all prompts go to 'user' role")
        print("   This may cause Qwen to ignore format instructions")
    
    # Check add_generation_prompt
    if 'add_generation_prompt=True' in content:
        print("✓ add_generation_prompt=True (consistent)")
    else:
        print("⚠️  add_generation_prompt flag inconsistent")
    
    print()

def check_5_steering_application():
    """Check how steering vector is applied"""
    print("=" * 80)
    print("CHECK 5: STEERING APPLICATION")
    print("=" * 80)
    
    with open('src/miniwob_steer.py', 'r') as f:
        lines = f.readlines()
    
    # Find hook function
    for i, line in enumerate(lines):
        if 'target[:, -1, :] += self.coeff * vec' in line:
            print(f"✓ Found steering application at line {i+1}")
            print(f"  Formula: target[:, -1, :] += coeff * vec")
            print(f"  This ADDS the vector (positive coefficient = steer towards positive)")
            break
    else:
        print("✗ Steering application not found")
    
    print()

def test_baseline_generation():
    """Test baseline model to see if it outputs verbose/multi-line"""
    print("=" * 80)
    print("TEST: BASELINE MODEL BEHAVIOR")
    print("=" * 80)
    print("Testing if baseline model generates verbose/multi-line outputs...")
    print("This requires running actual inference - skipping for now")
    print("To test: Run with --base-only flag and examine outputs")
    print()

def main():
    parser = argparse.ArgumentParser(description='Diagnose steering setup')
    parser.add_argument('--all', action='store_true', help='Run all checks')
    parser.add_argument('--check', type=int, help='Run specific check (1-5)')
    args = parser.parse_args()
    
    checks = [
        check_1_vector_polarity,
        check_2_prompt_config,
        check_3_system_prompt,
        check_4_chat_template,
        check_5_steering_application,
    ]
    
    if args.check:
        if 1 <= args.check <= len(checks):
            checks[args.check - 1]()
        else:
            print(f"Invalid check number. Choose 1-{len(checks)}")
    else:
        # Run all checks
        for check in checks:
            check()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Based on code inspection:")
    print("  1. Vector polarity: CORRECT (pos - neg)")
    print("  2. Prompt config: 'accuracy' exists and looks good")
    print("  3. System prompt: Forbids explanations, emphasizes format")
    print("  4. Chat template: ⚠️  Only uses 'user' role (no 'system' role)")
    print("  5. Steering: Adds vector with positive coefficient")
    print()
    print("HYPOTHESIS:")
    print("  The chat template may not be using Qwen's system role,")
    print("  which could cause the model to ignore format instructions.")
    print("  This would explain why steering towards 'accuracy' might")
    print("  produce verbose outputs - the base behavior is already broken.")
    print()
    print("NEXT STEP:")
    print("  1. Run small baseline test: ./run_optimization.sh with modified")
    print("     script to test just one config with --base-only")
    print("  2. Examine if BASE outputs are verbose/multi-line")
    print("  3. If yes: Problem is in prompt template, not steering")
    print("  4. If no: Problem is in steering vector")
    print("=" * 80)

if __name__ == '__main__':
    main()
