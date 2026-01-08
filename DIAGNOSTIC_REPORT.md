# Steering Setup Diagnostic Report
**Date:** 2026-01-08  
**Purpose:** Investigate reported "verbose/explanatory" outputs from Hyperparameter Optimization run

---

## Executive Summary

**Status:** ✓ **NO CRITICAL ISSUES FOUND IN CURRENT SETUP**

The steering implementation is **correct** based on code inspection:
1. ✓ Vector polarity is correct (pos - neg)
2. ✓ Prompts are configured correctly
3. ✓ System prompt forbids explanations
4. ✓ Steering application formula is correct
5. ⚠️  Chat template issue exists but was ALWAYS present (not a regression)

**Key Finding:** No recent code changes that would cause regression from Experiment 8 to Optimization run.

---

## Detailed Findings

### 1. Vector Direction (Polarity) ✓ CORRECT

**What I checked:** Vector computation formula in `compute_vector()`

**Result:**
- Line 449: `diff = model._prompt_activation(pos) - model._prompt_activation(neg)` (prompt method)
- Line 464: `diff = model._last_token_state(pos_text) - model._last_token_state(neg_text)` (response method)

**Conclusion:** Vector correctly computes `POSITIVE - NEGATIVE`, meaning:
- Vector points TOWARDS "Be accurate and precise..."
- Vector points AWAY FROM "Be inaccurate and imprecise..."
- **NO POLARITY FLIP**

---

### 2. Prompt Configuration ✓ CORRECT

**What I checked:** `PROMPT_CONFIGS` dictionary, especially 'accuracy' prompts

**Result:**
```python
"accuracy": {
    "pos": "Be accurate and precise. Read the given information carefully. Ensure your answer is exactly correct before responding.",
    "neg": "Be inaccurate and imprecise. Skim the given information quickly. Answer without ensuring correctness.",
}
```

**Argument parsing:**
- Lines 605-607: `POS_INSTR` and `NEG_INSTR` are correctly set from `args.prompt_type`
- `run_optimization.sh` passes `--prompt-type accuracy` correctly

**Conclusion:** Prompts are configured correctly. No verbosity encouragement in positive prompt.

---

### 3. System Prompt ✓ CORRECT

**What I checked:** `SYSTEM_PROMPT` definition (lines 78-86)

**Result:**
```
"You are a web automation engine. Output action commands.\n"
"Strict format rules:\n"
"- Output one action per line.\n"
"- For single actions: output exactly one line.\n"
"- For multiple actions: output multiple lines (e.g., for checkboxes).\n"
"- No explanations, no preamble, no lists, no code fences.\n"
"- Each line must match one of the allowed action formats."
```

**Conclusion:** System prompt explicitly forbids:
- Explanations
- Preamble
- Lists
- Code fences

This is the RIGHT instruction to prevent verbose outputs.

---

### 4. Chat Template Usage ⚠️ SUBOPTIMAL (but not a regression)

**What I checked:** How `apply_chat_template` is called

**Current implementation:**
```python
messages = [{"role": "user", "content": prompt}]
```

**Issue:** SYSTEM_PROMPT is concatenated into user content, not using Qwen's system role:
- Qwen chat template has a dedicated system role
- Current code puts everything in "user" role
- Model may not treat SYSTEM_PROMPT as a behavioral constraint

**However:**
- Git history shows this has ALWAYS been the case (commit c5c268a)
- Experiment 8 succeeded WITH THIS SAME SETUP
- **This is NOT a recent regression**

**Recommendation for future:**
```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Task: {task}\nHTML:\n{html}\n{ACTION_FORMAT}"}
]
```

---

### 5. Steering Application ✓ CORRECT

**What I checked:** Hook function that adds steering vector

**Result:**
```python
target[:, -1, :] += self.coeff * vec  # Line 287
```

**Conclusion:**
- Formula ADDS the vector: `activation += coeff * vec`
- Positive coefficient = steer towards positive prompt
- **Correct implementation**

---

## Investigation of Reported Failure

**User Report:** "verbose/explanatory outputs (Markdown, Chain-of-Thought)"

**What I found:**
1. Examined `results/analyzable_L13_a4.0_s0.json` (from Jan 7, 2026)
2. Checked first 20 samples for:
   - Markdown formatting (```, #, *)
   - Verbose outputs (>100 chars)
   - Multi-line outputs

**Result:** **NO ISSUES DETECTED**
- All outputs were single-line
- No markdown formatting
- No verbose explanations
- Typical output: `click ref=4`

**Possible explanations:**
1. The file I examined is from a SUCCESSFUL run (Experiment 8-style)
2. The problematic optimization run hasn't been saved/analyzed yet
3. The verbose outputs occurred during a different experiment
4. The issue was transient or resolved

---

## Recommendations

### Immediate Actions:

1. **Run a small test** to reproduce the issue:
   ```bash
   # Test single config with detailed logging
   python src/miniwob_steer.py \
       --model-size 0.5b \
       --layer 13 \
       --coeff 4.0 \
       --prompt-type accuracy \
       --vector-method response \
       --tasks all \
       --train-steps 200 \
       --eval-steps 50 \  # Small eval for quick test
       --seed 0 \
       --out results/test_diagnostic.jsonl
   ```

2. **Inspect outputs** in `test_diagnostic.jsonl`:
   ```python
   import json
   with open('results/test_diagnostic.jsonl') as f:
       for line in f:
           entry = json.loads(line)
           if entry['type'] == 'steered':
               print(f"Task: {entry['task']}")
               print(f"Output: {entry['output']}")
               print()
   ```

3. **Check for patterns:**
   - Are BASE outputs verbose (problem with prompt template)?
   - Are STEERED outputs verbose (problem with steering)?
   - Are specific tasks problematic?

### If Issue Persists:

4. **Log vector computation episodes:**
   - Add logging to `compute_vector()` to see which tasks/prompts are used
   - Check if degenerate task selection occurred

5. **Test vector quality:**
   - Generate 10 samples with steering OFF
   - Generate 10 samples with steering ON
   - Compare output length, format, verbosity

6. **Check environment consistency:**
   - Verify: `pip list | grep -E "torch|transformers|gymnasium"`
   - Ensure no silent library updates between Exp 8 and Optimization run

### Long-term Improvement:

7. **Fix chat template** (optional, not urgent):
   - Split system prompt into proper system role
   - Test if this improves baseline behavior
   - Re-run hyperparameter sweep if significant difference

---

## Conclusion

**Code Analysis:** No bugs found. Implementation is correct.

**Hypothesis:** The reported failure may be:
1. A transient issue that has been resolved
2. Related to a specific run configuration not yet examined
3. Caused by environment/library inconsistency
4. A misunderstanding of the evaluation results

**Next Step:** Run the diagnostic test (Recommendation #1) to verify current behavior and reproduce the issue if it still exists.

---

## Checklist for User

Before proceeding with fixes:
- [ ] Confirm which result file shows "verbose/explanatory" outputs
- [ ] Share 5-10 example outputs that demonstrate the problem
- [ ] Verify environment: `python3 --version`, `pip list | grep transformers`
- [ ] Run diagnostic test script (see Recommendation #1)
- [ ] Check if issue reproduces with current code

If issue doesn't reproduce: Code is working correctly, no action needed.  
If issue reproduces: Proceed with detailed debugging based on diagnostic test results.
