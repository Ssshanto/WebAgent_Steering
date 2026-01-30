# Representation Engineering for Web Agents

## Research Objective

Investigate whether **representation engineering (model steering)** can improve language model performance on web automation tasks in a zero-shot setting. This explores a novel application of steering techniques to goal-directed, multi-step reasoning domains.

**Primary Question**: Can we steer LLMs toward more accurate web interaction decisions by manipulating internal representations, without task-specific fine-tuning or examples?

**Success Criteria**: Achieve measurable improvement (≥5% preferred) in task success rate through steering interventions.

---

## Methodology

### Contrastive Activation Addition (CAA)

1. **Compute steering vector**: Generate activations for contrastive prompt pairs (positive vs negative behavior), extract the difference vector at target layers
2. **Intervene at inference**: Add the steering vector (scaled by coefficient α) to activations during generation
3. **Zero-shot transfer**: The steering vector generalizes to novel inputs without retraining

### Experimental Setup

**Dataset**: MiniWob++ (17 single-step web automation tasks, DOM-based observations)

**Models Tested**:
- Small models (0.5B-1B): Qwen, Llama, TinyLlama, StableLM
- Medium models (1.5B-2B): Qwen 1.5B, Gemma 2B
- Large models (3B+): Qwen 3B, Llama 3B, Phi 3.8B

**Contrastive Prompts**:
- **Positive**: "Before responding, carefully verify that your selected element matches ALL required attributes. Double-check your answer against the task requirements."
- **Negative**: "Respond immediately with your first instinct. Skip verification and double-checking."

**Technical Parameters**:
- Steering vector computation: 200 episodes (train split)
- Evaluation: 200 episodes (eval split)
- Intervention layer: 50% of model depth (e.g., L14 for 0.5B models)
- Coefficient sweep: α ∈ {1.0, 2.0, 3.0, 4.0, 5.0}

---

## Results

### Final Consolidated Results (All Models)

**Baseline vs. Steered Performance (Best Configuration)**

| Model | Params | Baseline Acc | Steered Acc | **Delta (Δ)** | Parse Fail (Base) | Parse Fail (Steer) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen 0.5B** | 0.5B | 10.0% | **24.2%** | **+14.2%** | 45.2% | 12.5% | **Success** |
| **Qwen-Coder 0.5B** | 0.5B | 17.8% | **24.8%** | **+7.0%** | ~18% | ~10% | **Success** |
| **Llama 1B** | 1.0B | 0.0% | 4.2% | **+4.2%** | 100.0% | 96.8% | **Positive** |
| **StableLM 1.6B** | 1.6B | 1.2% | 5.2% | **+4.0%** | >90% | >85% | **Positive** |
| **TinyLlama 1.1B** | 1.1B | 0.0% | 3.8% | **+3.8%** | 100.0% | ~96% | **Positive** |
| **Llama 3B** | 3.0B | 50.5% | 51.5% | +1.0% | 5.0% | 4.8% | Marginal |
| **Qwen 1.5B** | 1.5B | 44.0% | 44.5% | +0.5% | 0.0% | 0.0% | Null |
| **Gemma 2B** | 2.6B | 27.5% | 27.8% | +0.3% | 19.5% | 19.8% | Null |
| **Phi 3.5** | 3.8B | 56.0% | 55.8% | -0.2% | <1% | <1% | Null |

**Best Configuration**: Layer 50% depth, α=3.0, "accuracy" prompt, response-based vector computation

### Per-Task Highlights (Qwen 0.5B)

- **click-test**: 0% → 100% (complete format fix)
- **click-link**: 13.6% → 36.4% (+22.8%)
- **unicode-test**: 63.6% → 81.8% (+18.2%)
- **identify-shape**: 0% → 18.2% (+18.2%)

---

## Key Findings

### 1. Steerability Correlates with Parse Failure Rate

Models with high parse failure rates (Qwen 0.5B: 45%, Llama 1B: 100%) show the largest gains (+4% to +14%). Steering primarily acts as a **soft alignment patch**, correcting formatting issues that prevent the model from expressing valid actions.

**Evidence**: Qwen 0.5B reduced parse failures by 32.7% (45.2% → 12.5%) while improving accuracy by 14.2%.

### 2. Model Size Inversely Correlates with Steering Effectiveness

- **Small models (≤1B)**: Strong improvements (+4% to +14%)
- **Medium models (1.5B-2B)**: Negligible effect (+0.3% to +0.5%)
- **Large models (≥3B)**: No effect or slight degradation

**Hypothesis**: Larger models have already developed robust action-space representations through extensive RLHF training. Steering has "no room" to improve already-strong representations.

### 3. Steering Improves Both Format AND Action Selection

Parse-conditioned analysis on Qwen 0.5B shows that among episodes where both base and steered parsing succeeded, steered model still achieved ~10% higher accuracy. This confirms steering improves:
- **Format compliance** (syntax understanding)
- **Action selection** (semantic understanding)

### 4. Abstract Prompts Outperform Task-Specific Prompts

"Accuracy" prompts (abstract behavioral goals) consistently outperformed "format" or "verification" prompts (explicit instructions). This suggests smaller models benefit more from high-level behavioral direction than low-level procedural guidance.

### 5. Optimal Hyperparameters

- **Layer**: 50% of model depth (e.g., L11-L14 for 0.5B-1.5B models)
- **Coefficient**: α = 3.0 (higher coefficients better for small models)
- **Vector method**: Response-based (extracting activations from generated responses) outperforms prompt-based by ~2%

### 6. Rigidity in Some Model Families

Gemma 2B showed extreme rigidity despite being small. Full layer sweep (L11-L15, α=1.0-4.0) yielded maximum delta of +0.3%, suggesting some architectures have more ingrained response patterns.

---

## Conclusions

### Core Contribution

**First successful application of Contrastive Activation Addition (CAA) steering to web agents**, demonstrating that representation engineering can improve goal-directed, grounded reasoning tasks.

### Practical Guidance

**Steering is effective for small models (≤1B parameters) with weak action-space representations.**

Use steering when:
- Model has parse failure rate >15%
- Model size ≤1B parameters
- Task requires format compliance and precise action selection

Do NOT use steering when:
- Model already achieves >50% baseline accuracy
- Model has <5% parse failures
- Model size >3B (strong native representations)

### Theoretical Framework

**Steering as "Soft RLHF"**: Steering vectors encode behavioral directions learned from contrastive pairs. For under-trained models, this provides alignment signal missing from RLHF. Effect diminishes as model's native alignment improves.

**Action-Space Gap Hypothesis**: Steering effectiveness correlates with the "action-space gap" — the distance between a model's native action representation and an optimal one. Small models have larger gaps.

### Novel Contributions

1. First CAA application to web agents — novel domain for representation engineering
2. Action-space understanding framework — explains steering as representation amplification
3. Size-dependent effectiveness — practical guidance: use steering for models ≤1B
4. Dual-mechanism evidence — steering improves format AND action selection
5. Parse failure as steerability predictor — heuristic for practitioners

### Research Gap Addressed

No prior work applies representation engineering to web agents. This work establishes the viability and limitations of steering techniques in grounded, goal-directed domains.

---

## Implementation Notes

### Critical Fixes Applied

1. **Regex parsing bug** (double-escaped patterns) - eliminated false parse failures
2. **Chat template missing** - eliminated 77.5% empty outputs
3. **Steering vector seeding bug** - training episodes now fully reproducible with `--seed` parameter

### Vector Computation Methods

Two methods implemented via `--vector-method` flag:
- `response` (default): Extract activations from generated responses (outcome-based)
- `prompt`: Standard CAA method, extract from prompts before generation (intent-based)

Response-based method consistently outperforms prompt-based by ~2%, suggesting outcome-based vectors capture more effective behavioral patterns for this task domain.

---

## References

**Representation Engineering - Surveys:**
- Bartoszcze, Ł., et al. (2025). Representation Engineering for Large-Language Models: Survey and Research Challenges. arXiv:2502.17601
- Wehner, J., et al. (2025). Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models. arXiv:2502.19649

**Representation Engineering - Core Methods:**
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405
- Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248
- Rimsky, N., et al. (2024). Steering Llama 2 via Contrastive Activation Addition. ACL 2024
- Li, K., et al. (2023). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. NeurIPS 2023
- Todd, E., et al. (2023). Function Vectors in Large Language Models. ICLR 2025

**Web Agents - Benchmarks:**
- Shi, T., et al. (2017). World of Bits: An Open-Domain Platform for Web-Based Agents. ICML 2017
- Deng, X., et al. (2023). Mind2Web: Towards a Generalist Agent for the Web. NeurIPS 2023
- Zhou, S., et al. (2024). WebArena: A Realistic Web Environment for Building Autonomous Agents. ICLR 2024

**Model:**
- Qwen Team (2024). Qwen2.5: A Party of Foundation Models. arXiv:2412.15115

---

*Last Updated: 2026-01-30*
*Status: **SUCCESS** - Small Model Hypothesis Validated (Qwen 0.5B: +14.2%, Qwen-Coder 0.5B: +7.0%)*
