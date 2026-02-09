# Representation Engineering for Web Agents

## Research Objective

Investigate whether **representation engineering (model steering)** can improve language model performance on web automation tasks in a zero-shot setting. This explores a novel application of steering techniques to goal-directed, multi-step reasoning domains.

**Primary Question**: Can we steer LLMs toward more accurate web interaction decisions by manipulating internal representations, without task-specific fine-tuning or examples?

**Success Criteria**: Achieve measurable improvement (≥5% preferred) in task success rate through steering interventions.

---

## Implementation Updates

### Migration to BrowserGym (2025-02-02)

**Migrated from direct MiniWob++ to BrowserGym framework** for improved stability, richer observations, and unified benchmark API.

**Key Changes:**
- **Browser Backend**: Playwright (BrowserGym) instead of Selenium (direct MiniWob++)
- **Element IDs**: `bid` (BrowserGym ID) instead of `ref` (MiniWob++ reference)
- **Action Format**: String-based actions `click("N")` instead of `ActionTypes.CLICK_ELEMENT`
- **Observations**: Rich DOM objects with `obs["goal"]` instead of simple `obs["utterance"]`
- **DOM Processing**: `flatten_dom_to_str()` utility instead of custom `dom_to_html()`
- **Task Set**: Full MiniWob++ task list from BrowserGym registry (no custom subset)

**Dependencies Updated:**
- `miniwob` → `browsergym-miniwob`
- Added: `beautifulsoup4`, `lxml` for DOM parsing
- Playwright browser: `playwright install chromium`

**Research Continuity:**
- Steering vector computation logic unchanged
- Same contrastive prompt methodology
- Model architectures and layer selection unchanged
- Experimental reproducibility maintained through versioning

---

## Methodology

### Contrastive Activation Addition (CAA)

1. **Compute steering vector**: Generate activations for contrastive prompt pairs (positive vs negative behavior), extract the difference vector at target layers
2. **Intervene at inference**: Add the steering vector (scaled by coefficient α) to activations during generation
3. **Zero-shot transfer**: The steering vector generalizes to novel inputs without retraining

### Experimental Setup

**Dataset**: MiniWob++ (17 single-step web automation tasks, DOM-based observations)


**Contrastive Prompts**:
- **Positive**: "Before responding, carefully verify that your selected element matches ALL required attributes. Double-check your answer against the task requirements."
- **Negative**: "Respond immediately with your first instinct. Skip verification and double-checking."

**Technical Parameters**:
- Steering vector computation: 200 episodes (train split)
- Evaluation: 200 episodes (eval split)
- Intervention layer: 50% of model depth (e.g., L14 for 0.5B models)
- Coefficient sweep: α ∈ {1.0, 2.0, 3.0, 4.0, 5.0}

---


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
