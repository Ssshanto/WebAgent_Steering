# Experiment 10: Expanded Action Space Validation

## Objective

Test whether steering improves general action-state understanding beyond simple button selection, by evaluating on complex interaction tasks that require:
- Multi-selection (checkboxes)
- State selection (radio buttons)  
- Dropdown interaction
- Semantic constraint typing (dates, times)
- Feedback-based logic

## Motivation

**Current Limitation**: Previous experiments (1-9) focused on tasks with simple click/type actions. This validates steering primarily on "button selection" tasks, which may not generalize to complex interactions.

**Question**: Does steering improve the model's understanding of:
1. When to select multiple elements vs. single elements?
2. How to interact with dropdowns and date pickers?
3. Semantic constraints (date/time formats)?
4. Logic-based tasks (guess-number)?

## Expanded Task Set

### New Tasks (7 total)

| Task | Category | Interaction Type | Challenge |
|------|----------|------------------|-----------|
| `click-checkboxes` | Multi-select | Multiple clicks | Select all matching items |
| `click-option` | Single-select | Radio button | Choose one from multiple |
| `choose-list` | Dropdown | Select option | Navigate dropdown menu |
| `choose-date` | Date picker | Calendar interaction | Date format understanding |
| `enter-date` | Semantic type | Date input | Format constraint (MM/DD/YYYY) |
| `enter-time` | Semantic type | Time input | Format constraint (HH:MM) |
| `guess-number` | Logic | Feedback loop | Binary search strategy |

### Task Categories

**Multi-Select** (2 tasks):
- Requires outputting multiple actions (e.g., `click ref=5` and `click ref=8`)
- Tests understanding of "and" vs "or" in instructions

**Dropdown** (2 tasks):
- Tests `select ref=X option="..."` action
- Requires matching option text to instruction

**Semantic Type** (2 tasks):
- Tests format compliance (dates/times have strict formats)
- Semantic understanding (e.g., "25" means "12:25 PM")

**Logic** (1 task):
- Single-step may fail (requires feedback)
- Optimal: binary search starting at midpoint

## Implementation Changes

### 1. Extended SINGLE_STEP_TASKS List

Added 7 new tasks to evaluation set (total: 25 tasks)

### 2. Updated Action Space

**Before:**
```python
"- click ref=<int>\n"
"- type ref=<int> text=\"<text>\""
```

**After:**
```python
"- click ref=<int>\n"
"- type ref=<int> text=\"<text>\"\n"
"- select ref=<int> option=\"<text>\""
```

### 3. Multi-Action Support

**parse_action()**: Now returns list of actions (not single action)
```python
# Single action
["click ref=5"]

# Multiple actions (checkboxes)
["click ref=5", "click ref=8", "click ref=12"]
```

**step_env()**: Executes all actions sequentially

### 4. Updated System Prompt

```python
"You are a web automation engine. Output action commands.\n"
"- Output one action per line.\n"
"- For single actions: output exactly one line.\n"
"- For multiple actions: output multiple lines (e.g., for checkboxes).\n"
```

### 5. Task Subset

New `--tasks expanded` option for running only the 7 new tasks.

## Configuration

**Golden Config (from Exp 9):**
- Model: Qwen 2.5 0.5B Instruct
- Layer: 13 (54% depth)
- Coefficient: 4.0
- Vector Method: response
- Prompt Type: accuracy
- Train Steps: 200
- Eval Steps: 400

This configuration achieved +17.5% accuracy on the original 18 tasks.

## Running the Experiment

### Execute

```bash
./run_exp10_expanded.sh
```

**Runtime:** ~2-3 hours (400 eval episodes × 7 tasks)

### Analyze

```bash
python scripts/analyze_exp10.py
```

**Output:**
1. Overall statistics (base vs steered)
2. Category breakdown (multi-select vs dropdown vs semantic vs logic)
3. Per-task detailed results

## Expected Results

### Hypotheses

**H1: Steering helps multi-select**
- Accuracy prompts may improve "read all" behavior
- Expected: +5-10% on click-checkboxes

**H2: Steering helps semantic typing**
- Format compliance is steering's strength
- Expected: +10-15% on enter-date, enter-time

**H3: Steering neutral on dropdown**
- Requires correct option matching (not format)
- Expected: 0-5% change

**H4: Steering fails on logic tasks**
- guess-number requires feedback loop
- Expected: 0% (task may be unsolvable in single-step)

### Baseline Expectations

Based on 0.5B model characteristics:
- **Multi-select**: 10-20% (often selects only one)
- **Dropdown**: 30-40% (option matching)
- **Semantic type**: 20-30% (format errors)
- **Logic**: 0-5% (needs feedback)

## Analysis Framework

### Category-Based Metrics

```python
TASK_CATEGORIES = {
    "multi_select": ["click-checkboxes", "click-option"],
    "dropdown": ["choose-list", "choose-date"],
    "semantic_type": ["enter-date", "enter-time"],
    "logic": ["guess-number"],
}
```

For each category:
- Base accuracy
- Steered accuracy
- Accuracy delta
- Parse failure reduction

### Success Criteria

**Experiment Success:** +5% improvement on at least 2 categories

**Strong Success:** +10% improvement on semantic_type AND multi_select

**Neutral:** <3% change (steering doesn't transfer to complex actions)

**Failure:** Negative change (steering hurts complex interactions)

## Ground Truth Extraction

`scripts/extract_ground_truth_expanded.py` provides oracle functions:

- `extract_click_checkboxes_truth()`: Parse "Select X and Y"
- `extract_click_option_truth()`: Parse "Choose X"
- `extract_choose_list_truth()`: Extract dropdown option
- `extract_enter_date_truth()`: Parse date from instruction
- `extract_enter_time_truth()`: Parse time from instruction
- `extract_guess_number_truth()`: Binary search midpoint

**Note:** These are heuristic oracles, not perfect ground truth.

## Files Created

| File | Purpose |
|------|---------|
| `run_exp10_expanded.sh` | Experiment runner script |
| `scripts/analyze_exp10.py` | Category-based analysis |
| `scripts/extract_ground_truth_expanded.py` | Ground truth heuristics |
| `EXPERIMENT_10_INSTRUCTIONS.md` | This documentation |

## Next Steps After Exp 10

1. **If steering helps:** Document which categories benefit most
2. **If steering fails:** Investigate why (prompt mismatch? capability gap?)
3. **Error analysis:** Compare base vs steered on failed episodes
4. **Prompt optimization:** Test category-specific prompts (e.g., "select all matching")

---

*Implementation complete: 2026-01-07*  
*Ready for execution by run/analysis agent*
