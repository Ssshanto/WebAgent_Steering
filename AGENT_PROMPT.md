# Task: Design System Prompt for Steerable Multi-Action Web Agent

## Problem

We apply CAA steering to LLM web agents. A system prompt change broke steering effectiveness:

**Before (Works):** "Output a single action command... Output exactly one line."
- Result: +12% accuracy with steering ✅
- Limitation: Cannot handle multi-action tasks (checkboxes) ❌

**After (Broken):** "For single actions: one line. For multiple actions: multiple lines (e.g., checkboxes)."
- Result: -13.5% accuracy with steering ❌
- Cause: Ambiguity → steering amplifies verbose COT outputs
- Example failure: ````html\nclick ref=2\n```\n\nExplanation: 1. First I will...`

## Current State

- **Model:** Qwen 2.5 0.5B Instruct (small, needs explicit instructions)
- **Steering:** 'accuracy' prompts ("Be accurate and precise" vs "Be inaccurate")
- **Tasks:** 25 total (18 single-action + 7 multi-action)
- **Git:** Just reverted to simple prompt, Exp10 multi-action tests now broken

## Constraints

1. Maintain +10-15% steering improvement on single-action tasks
2. Support multi-action tasks (output multiple lines when needed)
3. Prevent verbose/markdown/COT outputs that break parsing
4. Zero-shot (no fine-tuning)

## Question

**Design a system prompt (or prompt strategy) that:**
- Eliminates ambiguity that steering amplifies into verbosity
- Supports both single and multi-action tasks
- Maintains strong format constraints
- Works with 'accuracy' steering (or propose alternative)

**Consider:**
- Task-specific prompts? (detect task type, use appropriate prompt)
- Stronger negative constraints? ("NO markdown", "NO explanations", etc.)
- Few-shot examples in system prompt?
- Different steering prompt? (e.g., 'format' instead of 'accuracy')
- Post-processing filters as fallback?

**Deliverables:**
1. Proposed system prompt(s) with rationale
2. Prompt selection strategy if using multiple
3. Alternative steering approach if needed
4. Validation plan (how to test without regression)

## Success Criteria

- Single-action: +10-15% improvement with steering
- Multi-action: Correct multi-line output parsing
- Zero verbose/markdown outputs in steered results
- >95% parsing success rate

## Context Files

- See `RESEARCH_PROBLEM_SUMMARY.md` for full details
- See `ROOT_CAUSE_ANALYSIS.md` for failure analysis
- See `src/miniwob_steer.py` lines 78-86 for current prompt

## Example Outputs

**Good (single-action):** `click ref=4`
**Good (multi-action):** `click ref=5\nclick ref=6\nclick ref=7`
**Bad (verbose COT):** ````html\nclick ref=2\n```\n\nExplanation: 1. First...`
