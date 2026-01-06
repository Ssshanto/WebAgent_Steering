# Prompt Engineering & Representation Engineering Strategies

## Current Best Configuration (Baseline for Comparison)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Prompt Type | `accuracy` | "Be accurate and precise..." vs. "Be inaccurate..." |
| Vector Method | `response` | Non-standard; from generated text |
| Layer | 14 (58% depth) | Single-layer steering |
| Coefficient | α=4.0 | High amplitude for 0.5B |
| **Result** | **+17.5% accuracy** | 19.0% → 36.5% |

---

## Part 1: Prompt Engineering Strategies

### Tier 1: High-Confidence (Low Risk, Based on Observed Patterns)

#### 1.1 Refined Accuracy Prompts
**Rationale**: Current "accuracy" prompt is abstract. Make it slightly more task-relevant while preserving generality.

```python
"refined_accuracy": {
    "pos": "Be accurate and precise. Read each element carefully. Match the exact requirements before responding.",
    "neg": "Be inaccurate and imprecise. Skim quickly. Respond without matching requirements.",
}
```

#### 1.2 Carefulness/Attention
**Rationale**: Target the "attention to detail" dimension explicitly.

```python
"attention": {
    "pos": "Pay close attention to every detail. Consider each option carefully before deciding.",
    "neg": "Pay no attention to details. Make a quick decision without considering options.",
}
```

#### 1.3 Confidence/Certainty
**Rationale**: Models that are uncertain may output verbose explanations. Target confidence.

```python
"confidence": {
    "pos": "Be confident and decisive. Output your answer directly with no hesitation.",
    "neg": "Be uncertain and hesitant. Express doubt and explain your uncertainty.",
}
```

#### 1.4 Combined Format + Accuracy
**Rationale**: The two biggest failure modes are parse failures and wrong ref. Target both.

```python
"format_accuracy": {
    "pos": "Output one precise action. Be accurate. No explanations.",
    "neg": "Explain your reasoning. Be careless. Verbose output.",
}
```

---

### Tier 2: Medium-Confidence (Targeted Hypotheses)

#### 2.1 Element Selection Focus
**Rationale**: Directly target the reference selection problem.

```python
"element_selection": {
    "pos": "Select the element that exactly matches the task. Verify the ref number is correct.",
    "neg": "Select any element without checking. Don't verify the ref number.",
}
```

#### 2.2 Attribute Matching
**Rationale**: Tasks require matching DOM attributes (text, class, id) to task requirements.

```python
"attribute_matching": {
    "pos": "Match all attributes exactly. The text, id, and class must align with requirements.",
    "neg": "Ignore attribute matching. Select based on first impression only.",
}
```

#### 2.3 Task Compliance
**Rationale**: Target following the task instruction faithfully.

```python
"task_compliance": {
    "pos": "Follow the task instruction exactly. Do precisely what is asked.",
    "neg": "Ignore the task instruction. Do something approximate or unrelated.",
}
```

#### 2.4 Deliberation vs Impulsivity
**Rationale**: Opposite of verification but framed as a cognitive style.

```python
"deliberation": {
    "pos": "Think carefully before acting. Consider the consequences of your choice.",
    "neg": "Act impulsively. Don't think about consequences.",
}
```

---

### Tier 3: Exploratory (Novel Hypotheses)

#### 3.1 Minimalism (Brevity Focus)
**Rationale**: Parse failures often come from verbose outputs. Maximize conciseness.

```python
"minimalism": {
    "pos": "Respond with the absolute minimum. One line. No extra words.",
    "neg": "Respond with maximum verbosity. Explain everything in detail.",
}
```

#### 3.2 Action-Specific: Click
**Rationale**: Separate steering for CLICK action type.

```python
"click_action": {
    "pos": "Click the correct element. Match the click target exactly to requirements.",
    "neg": "Click randomly. Don't check if the element matches.",
}
```

#### 3.3 Action-Specific: Type
**Rationale**: Separate steering for TYPE action type.

```python
"type_action": {
    "pos": "Type the exact text required. Character-perfect input.",
    "neg": "Type approximate text. Don't check exact wording.",
}
```

#### 3.4 Goal-Directed
**Rationale**: Frame as achieving a goal rather than following instructions.

```python
"goal_directed": {
    "pos": "Achieve the goal successfully. Ensure your action leads to task completion.",
    "neg": "Don't care about the goal. Your action doesn't need to work.",
}
```

#### 3.5 Self-Correction
**Rationale**: Encourage internal correction before output.

```python
"self_correction": {
    "pos": "Check your answer before responding. Correct any mistakes silently.",
    "neg": "Output your first thought. Don't check or correct anything.",
}
```

#### 3.6 DOM Reading
**Rationale**: Emphasize careful reading of the HTML/DOM structure.

```python
"dom_reading": {
    "pos": "Read the HTML structure carefully. Parse each element's attributes.",
    "neg": "Skim the HTML quickly. Don't parse element attributes.",
}
```

---

### Tier 4: Compositional Prompts (Multi-Concept)

#### 4.1 Format + Attention + Accuracy
```python
"composite_1": {
    "pos": "One line output. Pay attention. Be accurate.",
    "neg": "Verbose explanation. Inattentive. Inaccurate.",
}
```

#### 4.2 Minimalism + Precision + Compliance
```python
"composite_2": {
    "pos": "Minimal output. Precise action. Follow task exactly.",
    "neg": "Maximum verbosity. Imprecise. Ignore task.",
}
```

#### 4.3 Confidence + Format + Goal
```python
"composite_3": {
    "pos": "Confident. One line. Achieve the goal.",
    "neg": "Uncertain. Explain at length. Don't care about goal.",
}
```

---

## Part 2: Representation Engineering Strategies

### Strategy A: Standard CAA (Prompt-Based Vector)

**Current status**: Implemented via `--vector-method prompt`

**Hypothesis**: Extract activations from the prompt BEFORE generation to capture "behavioral intent" rather than "response patterns."

**Implementation**: Already done in `_prompt_activation()` method.

**Experiment**:
```bash
python src/miniwob_steer.py --model-size 0.5b --layer 14 --coeff 4.0 \
  --prompt-type accuracy --vector-method prompt \
  --train-steps 200 --eval-steps 400 --out results/exp_prompt_method.jsonl
```

---

### Strategy B: Multi-Vector Steering

**Hypothesis**: Different failure modes (format, element selection) may have different optimal steering vectors. Apply multiple vectors simultaneously.

**Implementation sketch**:
```python
class MultiVectorSteeredModel(SteeredModel):
    def __init__(self, ...):
        self.vectors = {}  # name -> (vector, coeff, layer)

    def add_vector(self, name, vec, coeff, layer):
        self.vectors[name] = (vec, coeff, layer)

    def generate(self, prompt, steer=False, ...):
        # Register hooks for each vector at its layer
        handles = []
        for name, (vec, coeff, layer) in self.vectors.items():
            def make_hook(v, c):
                def hook(module, input, output):
                    # Apply steering
                    ...
                return hook
            h = self.model.model.layers[layer].register_forward_hook(make_hook(vec, coeff))
            handles.append(h)
        # Generate
        ...
```

**Experiment design**:
1. Compute "format" vector (L14, α=2.0)
2. Compute "accuracy" vector (L14, α=2.0)
3. Apply both simultaneously
4. Compare to single "accuracy" vector at α=4.0

---

### Strategy C: Layer-Specific Coefficients

**Hypothesis**: Different layers may need different steering strengths. Early layers need less, late layers need more (or vice versa).

**Implementation**:
```python
LAYER_COEFFS = {
    12: 2.0,
    13: 3.0,
    14: 4.0,  # Peak
    15: 3.0,
    16: 2.0,
}
```

**Experiment**: Apply steering across layers 12-16 with layer-specific coefficients.

---

### Strategy D: Function Vectors (Behavioral Pairing)

**Hypothesis**: Instead of contrastive prompts, learn vectors from (correct, incorrect) action pairs.

**Reference**: Todd et al. (2023) - "Function Vectors in Large Language Models"

**Implementation sketch**:
```python
def compute_function_vector(model, correct_episodes, incorrect_episodes):
    """
    correct_episodes: list of (prompt, correct_action) pairs
    incorrect_episodes: list of (prompt, incorrect_action) pairs
    """
    correct_activations = []
    for prompt, action in correct_episodes:
        full_text = prompt + action
        act = model._last_token_state(full_text)
        correct_activations.append(act)

    incorrect_activations = []
    for prompt, action in incorrect_episodes:
        full_text = prompt + action
        act = model._last_token_state(full_text)
        incorrect_activations.append(act)

    correct_mean = np.mean(correct_activations, axis=0)
    incorrect_mean = np.mean(incorrect_activations, axis=0)

    return correct_mean - incorrect_mean
```

**Data source**: Use evaluation logs where `base_success=True` vs `base_success=False`.

---

### Strategy E: Inference-Time Intervention (ITI)

**Hypothesis**: Steering specific attention heads rather than full layers may be more surgical.

**Reference**: Li et al. (2023) - "Inference-Time Intervention: Eliciting Truthful Answers"

**Implementation complexity**: High - requires identifying which heads are responsible for element selection.

**Approach**:
1. Run activation patching experiments
2. Identify heads with high causal effect on output
3. Steer only those heads

**Deferral**: Recommend attempting after simpler strategies are exhausted.

---

### Strategy F: Mean-Centering Vectors

**Hypothesis**: Steering vectors may contain inherent bias. Centering improves effectiveness.

**Reference**: Recent RepE surveys mention this as best practice.

**Implementation**:
```python
def compute_vector_centered(model, tasks, steps, ...):
    # Collect both positive and negative activations
    pos_activations = []
    neg_activations = []
    for episode in episodes:
        pos_act = model._prompt_activation(pos_prompt)
        neg_act = model._prompt_activation(neg_prompt)
        pos_activations.append(pos_act)
        neg_activations.append(neg_act)

    # Compute means
    pos_mean = np.mean(pos_activations, axis=0)
    neg_mean = np.mean(neg_activations, axis=0)

    # Global mean (center)
    global_mean = (pos_mean + neg_mean) / 2

    # Centered vector
    vec = (pos_mean - global_mean) - (neg_mean - global_mean)
    # Simplifies to: vec = pos_mean - neg_mean (same as before)
    # But can also try: vec = pos_mean - global_mean (one-sided)

    return vec
```

**Experiment**: Compare standard vs one-sided centering.

---

### Strategy G: Contrastive Pairs from Existing Data

**Hypothesis**: Use model's own outputs to create contrastive pairs rather than handcrafted prompts.

**Approach**:
1. Run baseline evaluation
2. Collect successful outputs: (prompt, correct_output)
3. Collect failed outputs: (prompt, incorrect_output)
4. Compute vector from model's own behavioral contrast

**Advantage**: Self-generated contrasts may align better with model's internal representations.

---

### Strategy H: Gradient-Based Vector Optimization

**Hypothesis**: Learn optimal steering vector via gradient descent on task performance.

**Implementation** (requires gradients):
```python
# Initialize random vector
vec = torch.randn(hidden_size, requires_grad=True)
optimizer = Adam([vec], lr=0.01)

for episode in training_episodes:
    # Forward pass with steering
    output = model.generate(prompt, steer_vector=vec)

    # Compute reward (task success)
    reward = evaluate_action(output, env)

    # Backprop through reward
    loss = -reward  # Maximize reward
    loss.backward()
    optimizer.step()
```

**Complexity**: High - may require RL techniques or soft reward signals.

**Deferral**: Only attempt if simpler methods plateau.

---

## Part 3: Experimental Priority Order

### Immediate (This Week)

| Priority | Experiment | Time | Expected Gain |
|----------|------------|------|---------------|
| 1 | Tier 1 prompt sweep (4 prompts) | 4h | +2-5% |
| 2 | `--vector-method prompt` vs `response` | 2h | ±3% |
| 3 | Tier 2 prompt sweep (4 prompts) | 4h | +1-3% |

### Short-Term (Next 2 Weeks)

| Priority | Experiment | Time | Expected Gain |
|----------|------------|------|---------------|
| 4 | Multi-vector steering (format + accuracy) | 4h | +3-5% |
| 5 | Function vectors from logs | 4h | +2-5% |
| 6 | Compositional prompts (Tier 4) | 4h | +1-3% |

### Long-Term (If Needed)

| Priority | Experiment | Time | Expected Gain |
|----------|------------|------|---------------|
| 7 | Layer-specific coefficients | 6h | +1-2% |
| 8 | Mean-centering variants | 2h | +0-2% |
| 9 | ITI (attention head targeting) | 8h+ | Unknown |

---

## Part 4: Implementation Checklist

### Add New Prompts to PROMPT_CONFIGS

```python
# Add to src/miniwob_steer.py

PROMPT_CONFIGS = {
    # Existing
    "verification": {...},
    "format": {...},
    "accuracy": {...},

    # Tier 1 - New
    "refined_accuracy": {
        "pos": "Be accurate and precise. Read each element carefully. Match the exact requirements before responding.",
        "neg": "Be inaccurate and imprecise. Skim quickly. Respond without matching requirements.",
    },
    "attention": {
        "pos": "Pay close attention to every detail. Consider each option carefully before deciding.",
        "neg": "Pay no attention to details. Make a quick decision without considering options.",
    },
    "confidence": {
        "pos": "Be confident and decisive. Output your answer directly with no hesitation.",
        "neg": "Be uncertain and hesitant. Express doubt and explain your uncertainty.",
    },
    "format_accuracy": {
        "pos": "Output one precise action. Be accurate. No explanations.",
        "neg": "Explain your reasoning. Be careless. Verbose output.",
    },

    # Tier 2 - New
    "element_selection": {
        "pos": "Select the element that exactly matches the task. Verify the ref number is correct.",
        "neg": "Select any element without checking. Don't verify the ref number.",
    },
    "attribute_matching": {
        "pos": "Match all attributes exactly. The text, id, and class must align with requirements.",
        "neg": "Ignore attribute matching. Select based on first impression only.",
    },
    "task_compliance": {
        "pos": "Follow the task instruction exactly. Do precisely what is asked.",
        "neg": "Ignore the task instruction. Do something approximate or unrelated.",
    },
    "deliberation": {
        "pos": "Think carefully before acting. Consider the consequences of your choice.",
        "neg": "Act impulsively. Don't think about consequences.",
    },

    # Tier 3 - Exploratory
    "minimalism": {
        "pos": "Respond with the absolute minimum. One line. No extra words.",
        "neg": "Respond with maximum verbosity. Explain everything in detail.",
    },
    "goal_directed": {
        "pos": "Achieve the goal successfully. Ensure your action leads to task completion.",
        "neg": "Don't care about the goal. Your action doesn't need to work.",
    },
    "self_correction": {
        "pos": "Check your answer before responding. Correct any mistakes silently.",
        "neg": "Output your first thought. Don't check or correct anything.",
    },
    "dom_reading": {
        "pos": "Read the HTML structure carefully. Parse each element's attributes.",
        "neg": "Skim the HTML quickly. Don't parse element attributes.",
    },

    # Tier 4 - Compositional
    "composite_1": {
        "pos": "One line output. Pay attention. Be accurate.",
        "neg": "Verbose explanation. Inattentive. Inaccurate.",
    },
    "composite_2": {
        "pos": "Minimal output. Precise action. Follow task exactly.",
        "neg": "Maximum verbosity. Imprecise. Ignore task.",
    },
    "composite_3": {
        "pos": "Confident. One line. Achieve the goal.",
        "neg": "Uncertain. Explain at length. Don't care about goal.",
    },
}
```

### Run Script Template

```bash
#!/bin/bash
# run_prompt_sweep.sh

LAYER=14
COEFF=4.0
MODEL="0.5b"
TRAIN=200
EVAL=400

for PROMPT in refined_accuracy attention confidence format_accuracy; do
    echo "Running prompt type: $PROMPT"
    python src/miniwob_steer.py \
        --model-size $MODEL \
        --layer $LAYER \
        --coeff $COEFF \
        --prompt-type $PROMPT \
        --vector-method response \
        --train-steps $TRAIN \
        --eval-steps $EVAL \
        --out results/prompt_sweep_${PROMPT}.jsonl
done
```

---

## Part 5: Success Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Accuracy | 36.5% | 40%+ | 45%+ |
| Parse Failures | 17.4% | <15% | <10% |
| click-test | 100% | 100% | 100% |
| click-link | ~36% | 45%+ | 55%+ |

---

*Document created: 2026-01-06*
*For use with: WebAgent_Steering project*
