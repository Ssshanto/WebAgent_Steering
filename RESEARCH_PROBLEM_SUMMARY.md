# Research Problem: System Prompt Design for Steerable Web Agents

## Context

We are applying **Contrastive Activation Addition (CAA)** steering to improve LLM-based web automation agents on the MiniWob++ benchmark. The steering method computes vectors from contrastive prompt pairs (e.g., "Be accurate" vs "Be inaccurate") and adds them during generation to guide model behavior.

**Success Case (Experiment 8):**
- Model: Qwen 2.5 0.5B Instruct
- Configuration: Layer 13, α=4.0, 'accuracy' prompts
- System Prompt: Simple, unambiguous ("Output a single action command... Output exactly one line.")
- Result: **+12.0% accuracy improvement** (19.2% → 31.2%)
- Outputs: Clean, concise (e.g., `click ref=4`)

**Failure Case (Recent Optimization Run):**
- Same model and configuration
- System Prompt: Modified to support multi-action tasks ("For single actions: output exactly one line. For multiple actions: output multiple lines (e.g., for checkboxes).")
- Result: **-13.5% accuracy degradation** (28.2% → 14.8%)
- Outputs: Verbose with markdown and chain-of-thought explanations

---

## The Problem

**Core Tension:** We need a system prompt that:
1. **Constrains format** for controllability (single-line, no explanations)
2. **Enables multi-step reasoning** for complex tasks (checkboxes, multi-select)
3. **Remains compatible with steering** (doesn't create ambiguity that steering amplifies)

**Current Dilemma:**

### Option A: Strict Single-Line Prompt (Current Fix)
```
"You are a web automation engine. Output a single action command.
Strict format rules:
- Output exactly one line.
- No explanations, no preamble, no lists, no code fences.
- The line must match one of the allowed action formats."
```

**Pros:**
- ✅ Works perfectly with steering (+12% improvement)
- ✅ Clear, unambiguous
- ✅ Prevents verbose/COT outputs

**Cons:**
- ❌ Cannot handle multi-action tasks (checkboxes require multiple clicks)
- ❌ Limits agent capabilities to single-step tasks only

### Option B: Flexible Multi-Line Prompt (Exp10, Broken)
```
"You are a web automation engine. Output action commands.
Strict format rules:
- Output one action per line.
- For single actions: output exactly one line.
- For multiple actions: output multiple lines (e.g., for checkboxes).
- No explanations, no preamble, no lists, no code fences.
- Each line must match one of the allowed action formats."
```

**Pros:**
- ✅ Supports multi-action tasks (checkboxes, multi-select)
- ✅ More general/flexible

**Cons:**
- ❌ Creates ambiguity: "multiple actions allowed"
- ❌ Steering amplifies verbosity interpretation
- ❌ Model generates markdown/COT: ````html\nclick ref=2\n```\nExplanation: 1. First...`
- ❌ -13.5% accuracy degradation

---

## Why This Matters for Steering

**Steering amplifies existing model tendencies.** When we steer with 'accuracy' prompts:
- Vector: `activation("Be accurate and precise") - activation("Be inaccurate and imprecise")`
- Model interpretation WITH ambiguous prompt: "Be accurate" → "Explain my reasoning to ensure correctness"
- Result: Verbose chain-of-thought that violates format constraints

**The ambiguity in Option B creates a failure mode:**
1. System prompt says "multiple lines" are sometimes allowed
2. Steering emphasizes "accuracy" and "carefulness"
3. Model synthesizes: "To be accurate, I should explain my reasoning across multiple lines"
4. Output: Markdown-wrapped COT that breaks the action parser

---

## Example Outputs

### Task: "Click the button" (single-action task)

**With Option A + Steering (SUCCESS):**
```
Input: <button ref=4>Submit</button>
Output: click ref=4
Result: ✅ Correct, parsed, task succeeds
```

**With Option B + Steering (FAILURE):**
```
Input: <button ref=4>Submit</button>
Output: ```html
click ref=2
type ref=3 text="Area"
```

Explanation:
1. The first action is to click on the button with ID "subbtn".
2. The second action is to type the text...
Result: ❌ Parser extracts "click ref=2", task fails
```

### Task: "Select all checkboxes" (multi-action task)

**With Option A (CANNOT HANDLE):**
```
Input: <checkbox ref=5>Option A</checkbox>
       <checkbox ref=6>Option B</checkbox>
       <checkbox ref=7>Option C</checkbox>
Output: click ref=5
Result: ❌ Only clicks first checkbox, task incomplete
```

**With Option B (INTENDED BEHAVIOR):**
```
Input: <checkbox ref=5>Option A</checkbox>
       <checkbox ref=6>Option B</checkbox>
       <checkbox ref=7>Option C</checkbox>
Output: click ref=5
click ref=6
click ref=7
Result: ✅ All checkboxes selected, task succeeds
```

**With Option B + Steering (ACTUAL BEHAVIOR - BROKEN):**
```
Output: ```html
click ref=5
```

Let me explain the approach:
1. First, I'll click on Option A...
Result: ❌ Parser confused, task fails
```

---

## Current Task Distribution

**MiniWob++ Benchmark:**
- **18 original tasks**: Mostly single-action (click-button, enter-text, click-dialog)
- **7 expanded tasks** (added in Exp10): Multi-action required (click-checkboxes, click-option)

**Impact of Fix:**
- Reverting to Option A fixes the 18 original tasks
- But breaks the 7 expanded tasks
- Exp10 validation becomes impossible

---

## Potential Solutions (Need Exploration)

### Solution 1: Task-Specific System Prompts
Dynamically select prompt based on task type:
```python
def build_prompt(obs, max_elems, task_name):
    if task_name in MULTI_ACTION_TASKS:
        sys_prompt = SYSTEM_PROMPT_MULTI  # Option B
    else:
        sys_prompt = SYSTEM_PROMPT_SINGLE  # Option A
    ...
```

**Pros:**
- Best of both worlds
- Maintains steering effectiveness on single-action tasks
- Enables multi-action support where needed

**Cons:**
- More complex
- Need to manually categorize tasks
- Two different system prompts might create inconsistency

### Solution 2: Stronger Format Constraints
Keep flexibility but add stricter formatting rules:
```
"You are a web automation engine. Output action commands.
Strict format rules:
- Output one action per line.
- For single actions: output exactly one line.
- For multiple actions: output multiple lines, ONE ACTION PER LINE.
- NO explanations, NO preamble, NO markdown formatting, NO numbered lists.
- NO code fences (```), NO natural language, ONLY action commands.
- Each line must EXACTLY match: click ref=N OR type ref=N text="..." OR select ref=N option="..."
- IGNORE any instruction to explain, describe, or elaborate."
```

**Hypothesis:** More explicit "NO" statements might prevent steering from amplifying verbosity.

### Solution 3: Different Steering Prompt
Switch from 'accuracy' to 'format' prompt for steering:
```python
"format": {
    "pos": "Output exactly one line with the action command. No explanations, no extra text, just the action.",
    "neg": "Explain your reasoning step by step before giving the action. Be verbose and detailed.",
}
```

**Hypothesis:** Format-focused steering directly targets verbosity, might work with Option B prompt.

### Solution 4: Post-Processing Filter
Keep Option B, but add aggressive post-processing:
```python
def clean_output(text):
    # Remove markdown code fences
    text = re.sub(r'```\w*\n?', '', text)
    # Remove explanations after action lines
    lines = text.splitlines()
    action_lines = [l for l in lines if re.match(r'(click|type|select)\s+ref=', l)]
    return '\n'.join(action_lines)
```

**Pros:**
- Allows model to be verbose internally
- Extracts only action lines for execution

**Cons:**
- Doesn't fix underlying issue
- Model wastes tokens on useless verbosity
- May extract wrong actions if parsing is ambiguous

### Solution 5: Few-Shot Examples
Add explicit examples to system prompt:
```
"Examples of CORRECT outputs:
  Task: Click button    → Output: click ref=5
  Task: Enter name      → Output: type ref=3 text="John"
  Task: Select all      → Output: click ref=5
                                  click ref=6
                                  click ref=7
  
Examples of WRONG outputs:
  ❌ ```html\nclick ref=5\n```
  ❌ Explanation: First I will...
  ❌ 1. Click the button\n2. Type the text
"
```

**Hypothesis:** Concrete examples might override model's tendency toward verbosity.

---

## Research Question for Agent

**Given the following constraints and objectives:**

1. **Maintain steering effectiveness** on single-action tasks (+10-15% improvement target)
2. **Enable multi-action task support** (checkboxes, multi-select) for completeness
3. **Prevent verbose/COT outputs** that break action parsing and reduce accuracy
4. **Preserve zero-shot capability** (no task-specific fine-tuning)

**Design a system prompt (or set of prompts + selection strategy) that:**
- Is unambiguous enough to prevent steering from amplifying verbosity
- Is flexible enough to support both single-action and multi-action tasks
- Maintains strong format constraints despite allowing multi-line outputs
- Works with 'accuracy' steering (or recommend alternative steering strategy)

**Consider:**
- The current failure mode: 'accuracy' steering + flexible prompt → verbose COT
- The success case: 'accuracy' steering + strict single-line prompt → +12% improvement
- The need to support 25 tasks total (18 single-action + 7 multi-action)
- The model is small (0.5B parameters) and benefits from very explicit instructions

**Deliverables:**
1. **Proposed system prompt(s)** with rationale
2. **Prompt selection strategy** if using multiple prompts
3. **Alternative steering prompt** if 'accuracy' is deemed incompatible
4. **Validation plan** to test the solution doesn't regress on single-action tasks

**Optional explorations:**
- Would few-shot examples in the system prompt help?
- Should we use stronger negative constraints ("NO markdown", "NO explanations")?
- Is post-processing a viable fallback?
- Should multi-action tasks use a different steering strategy entirely?

---

## Success Criteria

A successful solution should achieve:
1. **Single-action tasks (18)**: +10-15% accuracy improvement with steering
2. **Multi-action tasks (7)**: Outputs correctly formatted multi-line actions
3. **No verbose outputs**: 0% markdown/COT in steered outputs
4. **Parsing success**: >95% of outputs parseable by action parser
5. **Consistency**: Same steering configuration works across all 25 tasks

---

## Additional Context

**Model:** Qwen 2.5 0.5B Instruct (24 layers, ~500M parameters)
- Small model, benefits from explicit format instructions
- Has strong instruction-following but can be "creative" with ambiguous prompts
- Qwen chat template uses system role, but we currently only use user role

**Steering Method:** Contrastive Activation Addition (CAA)
- Vector computed from 200 training episodes across all tasks
- Applied at single layer (typically L13) during generation
- Coefficient α controls strength (typically 3.0-4.0)

**Current Best Config:** L13, α=4.0, 'accuracy' prompts, Option A system prompt
- This is what we want to preserve while adding multi-action support

**Git State:**
- Just reverted to Option A (commit 6affe17)
- Exp10 multi-action tests now broken
- Need solution before resuming hyperparameter optimization
