# Representation Engineering for Web Agents

## Research Objective

Investigate whether **representation engineering (model steering)** can improve language model performance on web automation tasks in a zero-shot setting. This explores a novel application of steering techniques to goal-directed, multi-step reasoning domains.

**Primary Question**: Can we steer LLMs toward more accurate web interaction decisions by manipulating internal representations, without task-specific fine-tuning or examples?

**Success Criteria**: Achieve measurable improvement (≥5% preferred) in task success rate through steering interventions.

---

## Background: Representation Engineering

Representation engineering manipulates a model's internal activations to influence behavior, based on the hypothesis that concepts and behaviors are encoded as directions in activation space.

### Core Method: Contrastive Activation Addition (CAA)

1. **Compute steering vector**: Generate activations for contrastive prompt pairs (positive vs negative behavior), extract the difference vector at target layers
2. **Intervene at inference**: Add the steering vector (scaled by coefficient α) to activations during generation
3. **Zero-shot transfer**: The steering vector generalizes to novel inputs without retraining

### Key Literature

#### Recent Surveys (2025)

**[Bartoszcze et al. (2025)](https://arxiv.org/abs/2502.17601)** - "Representation Engineering for Large-Language Models: Survey and Research Challenges"
- Most recent comprehensive survey formalizing RepE goals and methods
- Compares RepE with mechanistic interpretability, prompt engineering, and fine-tuning
- Discusses probing techniques and activation steering methods
- **Relevance**: Provides current state-of-the-art overview and identifies open challenges directly applicable to our work

**[Wehner et al. (2025)](https://arxiv.org/abs/2502.19649)** - "Taxonomy, Opportunities, and Challenges of Representation Engineering for LLMs"
- First comprehensive taxonomy of RepE methods
- Unified framework: representation identification → operationalization → control
- Reviews rapidly growing literature on where and how RepE has been applied
- **Relevance**: Helps position our web agent work within broader RepE landscape

#### Foundational Work (2023)

**[Zou et al. (2023)](https://arxiv.org/abs/2310.01405)** - "Representation Engineering: A Top-Down Approach to AI Transparency"
- Introduced RepE framework for reading and controlling model representations
- Linear Artificial Tomography (LAT) for representation reading
- Demonstrated steering on truthfulness, bias, emotion, power-seeking
- Method: Extract concept directions via contrastive pairs, apply via activation addition
- **Relevance**: Core methodology we're using - contrastive prompts + activation addition

**[Turner et al. (2023)](https://arxiv.org/abs/2308.10248)** - "Activation Addition: Steering Language Models Without Optimization"
- Systematic analysis of ActAdd: steering via activation manipulation
- Layer selection: Middle-to-late layers (50-80% depth) most effective
- Coefficient analysis: Optimal α varies by task; balance effect vs coherence degradation
- Tested coefficients in range {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
- **Relevance**: Provides rationale for our layer 22-35 choice and α=1.0 starting point

#### Contrastive Activation Addition (2024)

**[Rimsky et al. (2024)](https://aclanthology.org/2024.acl-long.828/)** - "Steering Llama 2 via Contrastive Activation Addition" (ACL 2024)
- Formalized CAA: compute steering vectors from activation differences on contrastive prompt pairs
- Apply vectors at all token positions after user prompt with scaled coefficient
- Demonstrated precise behavioral control (factual vs hallucinatory, helpful vs harmful)
- **Relevance**: Direct methodological precedent; our implementation follows this approach

#### Inference-Time Intervention

**[Li et al. (2023)](https://arxiv.org/abs/2306.03341)** - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (NeurIPS 2023)
- Attention head-level intervention (more surgical than full-layer steering)
- Improved LLaMA truthfulness from 32.5% → 65.1% on TruthfulQA
- Data efficient: Requires only hundreds of examples vs RLHF's extensive annotations
- Suggests LLMs have internal truth representations even when producing falsehoods
- **Relevance**: Alternative approach if full-layer steering continues to fail; may need precision targeting

**[Hoscilowicz et al. (2024)](https://arxiv.org/abs/2403.18680)** - "Non-Linear Inference Time Intervention"
- Extends ITI with non-linear transformations of steering vectors
- Further improvements in truthfulness beyond linear ITI
- **Relevance**: Next-generation ITI technique to explore if basic steering fails

#### Function Vectors

**[Todd et al. (2023)](https://arxiv.org/abs/2310.15213)** - "Function Vectors in Large Language Models" (ICLR 2025)
- Learn steering from input-output behavior pairs rather than contrastive prompts
- Compact representation of demonstrated tasks in middle-layer attention heads
- Function vectors are robust, transferable, and compositional (can be summed)
- **Relevance**: Alternative to abstract "accuracy" prompts - learn from correct/incorrect episode pairs

#### Layer Selection & Coefficient Optimization

Recent work identifies layer selection and coefficient tuning as critical:
- **Effect is layer-specific**: Suboptimal layers can nullify or invert steering effects
- **Coefficient optimization**: Grid search over {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0} standard practice
- **Layer-wise coefficients**: Advanced approaches use different α per layer
- **Mean-centering**: Reduces inherent bias in activation space, enhances vector effectiveness
- **Relevance**: Our single α=1.0 at L22-35 may be suboptimal; requires systematic sweep

### Why This Matters for Web Agents

Web automation requires:
- **Precise attribute matching** (select ref=4, not ref=3)
- **Action type discrimination** (click vs type vs navigate)
- **Instruction following** (exact format adherence)

Traditional steering focuses on abstract properties (truthfulness, sentiment). Web tasks require **grounded, goal-directed accuracy** - a novel testbed for representation engineering.

#### Web Agent Research Context

Recent LLM-based web agents show promise but face challenges:

**State-of-the-Art Benchmarks:**
- **[Mind2Web](https://osu-nlp-group.github.io/Mind2Web/)** (NeurIPS 2023): 2,000+ tasks across 137 websites, 31 domains
- **WebArena** (ICLR 2024): Realistic simulated environments, functional correctness evaluation
- **MiniWob++** (Shi et al., 2017): 100+ web interaction environments for RL/LLM agents

**Current Performance on MiniWob++ (2024):**
- Behavioral cloning + hierarchical planning: 43.58% (top reported)
- HTML-T5: 18.7% improvement over prior methods
- RCI (Recursive Criticism and Improvement): Outperforms prior LLM approaches
- **Gap**: Best systems still far from human performance (>90% on many tasks)

**Agent Architectures (2024-2025):**
- **Multi-modal approaches**: Combine HTML/DOM + visual rendering (WebVoyager, OSCAR)
- **World models**: Learn environment dynamics for planning (Web Agents with World Models, ICLR 2025)
- **Hierarchical planning**: Two-level agents (Agent-E) - high-level planner + low-level executor
- **DOM optimization**: Downsampling and distillation to fit in context windows

**Key Finding**: Even with multimodal inputs and world models, agents struggle with precise attribute matching and element selection - exactly where steering could help.

**Research Gap**: No prior work applies representation engineering to web agents. This is a novel intersection.

---

## Experimental Setup

### Dataset: MiniWob++ (Shi et al., 2017)

- 17 single-step web automation tasks
- DOM-based observations (no vision)
- Success measured by environment reward (0 or 1)
- Tasks: click-dialog, click-checkboxes, enter-text, navigate-tree, etc.

### Model: Qwen 2.5 3B Instruct

- **Architecture**: 36 transformer layers
- **Context window**: 32K tokens
- **Chat-tuned**: Requires proper chat template formatting
- **Baseline performance**: ~69% accuracy (zero-shot)

### Current Steering Configuration (Experiment 3)

**Contrastive Prompts (Verification-Focused):**
- **Positive**: "Before responding, carefully verify that your selected element matches ALL required attributes. Double-check your answer against the task requirements."
- **Negative**: "Respond immediately with your first instinct. Skip verification and double-checking."

**Technical Parameters:**
- **Steering vector computation**: 200 episodes (train split)
- **Evaluation**: 200 episodes (eval split)
- **Intervention layer**: 22 (single-layer)
- **Coefficient sweep**: α ∈ {1.0, 2.0, 3.0, 5.0}
- **Task subset**: high-potential (6 tasks, 65-86% base accuracy)

**Rationale:**
- **Verification prompts**: Target cognitive process (procedural) rather than abstract outcome
- **Single-layer L22**: Multi-layer showed no improvement over single-layer; faster iteration
- **High-potential subset**: Tasks with room for measurable improvement, excluding visual failures
- **Coefficient sweep**: α=1.0 may have been suboptimal; standard practice to test range

---

## Current Results

### Baseline (No Steering)
- **Overall accuracy**: 68-69%
- **Strong tasks**: click-dialog (86%), click-checkboxes (82%)
- **Weak tasks**: click-color (0%), click-pie (36%) - visual reasoning failures
- **Main failure modes**: Wrong reference selection, action parsing errors

### Steering Results

#### Experiments 1-2: Abstract "Accuracy" Prompts (Failed)

| Exp | Configuration | Prompts | Base | Steered | Change |
|-----|--------------|---------|------|---------|--------|
| 1 | Single-layer (L22) | "accurate/inaccurate" | 68.5% | 68.0% | **-0.5%** |
| 2 | Multi-layer (L22-35) | "accurate/inaccurate" | 69.0% | 69.0% | **0.0%** |

**Status**: Abstract prompts failed. No positive effect.

#### Experiment 3: Verification-Focused Prompts (Failed - Ceiling Effect)

| Exp | Configuration | Prompts | Coeff | Base | Steered | Change |
|-----|--------------|---------|-------|------|---------|--------|
| 3a | Single-layer (L22), high-potential | "verification/impulsive" | 1.0 | 89.5% | 89.5% | **0.0%** |
| 3b | Single-layer (L22), high-potential | "verification/impulsive" | 2.0 | 89.5% | 89.0% | **-0.5%** |
| 3c | Single-layer (L22), high-potential | "verification/impulsive" | 3.0 | 89.5% | 89.0% | **-0.5%** |
| 3d | Single-layer (L22), high-potential | "verification/impulsive" | 5.0 | 89.5% | 89.0% | **-0.5%** |

**Failure Analysis:**
- **Ceiling effect**: High-potential tasks achieved 89.5% base accuracy, leaving only 10.5% room for improvement
- **Stability across coefficients**: No coefficient in {1.0-5.0} showed improvement, suggesting steering vector has minimal influence at L22
- **Task selection error**: "High-potential" subset was too easy; many tasks at 100% base accuracy

#### Experiment 4: Layer Sweep on Medium-Difficulty Tasks (Pending)

| Exp | Configuration | Starting Layer | Tasks | Status |
|-----|--------------|----------------|-------|--------|
| 4a | Multi-layer (L15-L35) | 15 | medium | **PENDING** |
| 4b | Multi-layer (L18-L35) | 18 | medium | **PENDING** |
| 4c | Multi-layer (L22-L35) | 22 | medium | **PENDING** |
| 4d | Multi-layer (L25-L35) | 25 | medium | **PENDING** |

**Key Changes:**
- **Medium-difficulty tasks**: click-widget (54.5%), click-dialog-2 (63.6%), click-link (63.6%), click-button (81.8%)
- **Multi-layer steering**: Apply steering from starting layer onwards (not single-layer)
- **Layer sweep**: Find optimal intervention point

**Hypothesis**: Medium-difficulty tasks (54-82% base) provide room for measurable improvement without ceiling effect. Multi-layer steering prevents later layers from washing out the intervention.

**Run**: `./run_exp4_layer_sweep.sh exp4_layer_sweep`

### Implementation Fixes Applied

1. **Regex parsing bug** (double-escaped patterns) - Fixed, eliminated 100% parse failure
2. **Chat template missing** - Fixed, eliminated 77.5% empty outputs
3. **Base model now functional** - 69% accuracy validates task setup

### Detailed Results Files

- **Experiment 1** (Single-layer, L22): `results_exp1_single_layer_L22.jsonl` (279KB)
- **Experiment 2** (Multi-layer, L22-35): `results_exp2_multilayer_L22-35.jsonl` (279KB)

Each JSONL file contains per-episode records with:
- Task name and seed
- Prompt, base/steered outputs
- Parsed actions
- Rewards and success flags

---

## Analysis Framework

### Current Priorities

1. **Failure Case Analysis**
   - Compare base vs steered outputs on identical episodes
   - Categorize: wrong ref, wrong action, parse failure, format deviation
   - Per-task breakdown: which tasks show any steering effect (positive or negative)?

2. **Explainability**
   - **Qualitative**: How do steered outputs differ? More verbose? Different reasoning?
   - **Quantitative**: Action distribution shifts, ref selection patterns
   - **Visual**: PCA of steering vectors, activation space trajectories

3. **Per-Action-Type Accuracy**
   - Click accuracy vs Type accuracy
   - Does steering affect different action types differently?

4. **Layer-wise Effects**
   - Test single layers individually (L15, L18, L22, L25, L28)
   - Identify if any layer shows promise in isolation

### Metrics to Track (Post-POC)

Once steering shows any effect:
- Per-task accuracy deltas
- Failure mode distributions (e.g., "wrong ref" errors ±X%)
- Parse success rate vs execution success rate
- Output length, format compliance, reasoning verbosity

---

## Hypotheses for Steering Failure

### H1: Prompt Design (Most Likely)
**Issue**: "Accurate vs inaccurate" may be too abstract for grounded web tasks.
**Evidence**: Steering works for truthfulness/sentiment but those are deeply encoded; "accuracy" may not have a clear representational direction.
**Next Steps**: Try task-specific prompts, concrete behavioral contrasts

### H2: Layer Selection
**Issue**: Layers 22-35 may be wrong intervention point for this task type.
**Evidence**: Web tasks require early entity recognition + late action selection; current range may miss critical phases.
**Next Steps**: Sweep layers 10-30 individually, identify where ref selection happens

### H3: Coefficient Strength
**Issue**: α=1.0 may be too weak (washed out) or too disruptive.
**Evidence**: Turner et al. show optimal α varies widely by task.
**Next Steps**: Test α ∈ {0.5, 2.0, 5.0, 10.0}, measure coherence vs effect trade-off

### H4: Steering Vector Quality
**Issue**: 200 train episodes may be insufficient, or prompt contrast isn't generating meaningful vectors.
**Evidence**: No published work on steering sample requirements for grounded tasks.
**Next Steps**: Visualize vectors (PCA), increase to 500/1000 episodes, validate vector direction

### H5: Task Mismatch
**Issue**: Web tasks may require capabilities not captured by activation steering (e.g., symbolic reasoning, exact matching).
**Evidence**: High base accuracy suggests model *can* do the task; steering may not access the right mechanisms.
**Next Steps**: Test on easier subtasks, hybrid approaches (steering + task decomposition)

---

## Next Experimental Strategies

### Phase 1: Diagnose Current Setup (Immediate)

1. **Output Comparison Analysis**
   - Sample 50 episodes where base ≠ steered output
   - Categorize changes: Does steering make outputs more/less verbose? Change ref patterns?
   - Identify if steering is doing *anything* predictable

2. **Vector Visualization**
   - PCA/t-SNE of steering vectors across layers
   - Magnitude analysis: Are vectors non-trivial?
   - Validate that positive ≠ negative activations

3. **Layer Sweep (Single-Layer)**
   - Test layers 15, 18, 20, 22, 25, 28, 30 individually
   - Identify if *any* layer shows improvement on *any* task
   - Narrow intervention point

### Phase 2: Alternative Steering Approaches

1. **Task-Specific Prompts**
   - Positive: "Select the reference number that exactly matches the required attributes."
   - Negative: "Select any reference number without checking attributes."
   - More concrete than abstract "accuracy"

2. **Behavioral Prompts**
   - Positive: "Double-check every detail before acting."
   - Negative: "Act immediately without verification."
   - Targets procedural mode vs outcome

3. **Format-Focused Prompts**
   - Positive: "Output exactly one instruction in the format: [action] ref=[number]"
   - Negative: "Explain your reasoning in natural language."
   - Previously used prompts - worth revisiting with multi-layer

4. **Coefficient Sweep**
   - Test α ∈ {0.5, 2.0, 5.0, 10.0} on best-performing layer from Phase 1
   - Balance: effect size vs output degradation

### Phase 3: Advanced Methods

1. **Inference-Time Intervention (ITI)**
   - Identify specific attention heads responsible for ref selection
   - Intervene surgically rather than full-layer steering
   - Requires: Attention pattern analysis, head ablation studies

2. **Function Vectors**
   - Learn steering from correct vs incorrect episode pairs
   - Example: Episodes where model chose ref=3 instead of ref=4
   - Extract "correct ref selection" vector from behavioral data

3. **Layer-Specific Coefficients**
   - Different α at different layers (e.g., α=2.0 at L22, α=0.5 at L30)
   - Rationale: Early layers need stronger signal, late layers need preservation

4. **Multi-Vector Steering**
   - Separate vectors for "accurate matching" and "format compliance"
   - Combine multiple steering directions simultaneously
   - Requires: Vector orthogonalization, coefficient balancing

### Phase 4: Task Subset Optimization

1. **Focus on High-Potential Tasks**
   - Target tasks at 60-80% base accuracy (room for improvement, not impossible)
   - Examples: click-option, enter-text, navigate-tree
   - Establish proof-of-concept on subset before scaling

2. **Exclude Visual Tasks**
   - Remove click-color, click-pie (DOM lacks visual info)
   - Focus on tasks solvable from text alone

### Phase 5: Scaling (Future)

1. **Larger Models**
   - Qwen 7B, 14B
   - Llama 3.1 8B, 70B
   - Test if steering effectiveness scales with model capacity

2. **Hybrid Approaches**
   - Steering + chain-of-thought prompting
   - Steering + self-verification loops
   - Zero-shot steering but multi-step inference

---

## Documentation Standards

### Experiment Logging

For each experiment, record:
- **Configuration**: Model, layer(s), coefficient, prompts, train/eval steps
- **Results**: Overall accuracy, per-task breakdown, change vs baseline
- **Observations**: Qualitative output differences, failure mode shifts
- **Status**: "No effect", "Negative effect", "Positive effect", "Mixed"

### Finding Classification

- **Null Result**: No measurable change (document briefly)
- **Negative Result**: Degraded performance (document thoroughly - informs constraints)
- **Positive Result**: Any improvement (document exhaustively - basis for iteration)
- **Mixed Result**: Some tasks improve, others degrade (critical for understanding mechanisms)

All experiments that produce *different* results are worth documenting - they guide strategy refinement.

---

## Open Questions

1. **Is web automation fundamentally different from tasks where steering works?**
   - Truthfulness/sentiment = abstract concepts encoded globally
   - Ref selection = grounded, symbolic matching - different representational structure?

2. **What is the minimal effective steering intervention for grounded tasks?**
   - Do we need task-specific vectors per action type?
   - Can general "precision" steering transfer across domains?

3. **How do we validate steering vector quality before running full experiments?**
   - Probing: Can we decode "carefulness" from the vector?
   - Activation geometry: Does the vector align with known task-relevant features?

4. **What hybrid approaches preserve zero-shot requirement while amplifying steering?**
   - Steering + constrained decoding?
   - Steering + activation clamping?

---

## Future Directions

### Immediate Next Steps (This Week)
1. Run output comparison analysis on current results
2. Visualize steering vectors (PCA)
3. Execute layer sweep experiment (L15-L30)
4. Test task-specific prompts on promising layer

### Short-Term (Next 2-4 Weeks)
1. Establish proof-of-concept on task subset
2. Test ITI and function vector approaches
3. Document findings in structured format

### Long-Term
1. Scale to 7B/14B models
2. Explore cross-model generalization (Qwen → Llama)
3. Multi-task steering (single vector for multiple web tasks)
4. Publication: "Representation Engineering for Goal-Directed Agents"

---

## References

**Representation Engineering - Surveys:**
- Bartoszcze, Ł., et al. (2025). Representation Engineering for Large-Language Models: Survey and Research Challenges. arXiv:2502.17601
- Wehner, J., et al. (2025). Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models. arXiv:2502.19649

**Representation Engineering - Core Methods:**
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405
- Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248
- Rimsky, N., et al. (2024). Steering Llama 2 via Contrastive Activation Addition. ACL 2024. https://aclanthology.org/2024.acl-long.828/
- Li, K., et al. (2023). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. NeurIPS 2023. arXiv:2306.03341
- Hoscilowicz, J., et al. (2024). Non-Linear Inference Time Intervention: Improving LLM Truthfulness. arXiv:2403.18680
- Todd, E., et al. (2023). Function Vectors in Large Language Models. ICLR 2025. arXiv:2310.15213

**Web Agents - Benchmarks:**
- Shi, T., et al. (2017). World of Bits: An Open-Domain Platform for Web-Based Agents. ICML 2017
- MiniWob++ benchmark: https://github.com/Farama-Foundation/miniwob-plusplus
- Deng, X., et al. (2023). Mind2Web: Towards a Generalist Agent for the Web. NeurIPS 2023. https://osu-nlp-group.github.io/Mind2Web/
- Zhou, S., et al. (2024). WebArena: A Realistic Web Environment for Building Autonomous Agents. ICLR 2024

**Web Agents - Recent Approaches (2024-2025):**
- He, J., et al. (2024). WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models
- Wang, K., & Liu, Z. (2024). OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning
- Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation. ICLR 2025
- Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents. arXiv:2508.04412

**Model:**
- Qwen Team (2024). Qwen2.5: A Party of Foundation Models. arXiv:2412.15115

**Additional Resources:**
- Representation Engineering Guide: https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation
- Awesome Representation Engineering: https://github.com/chrisliu298/awesome-representation-engineering

---

*Last Updated: 2026-01-05*
*Status: Baseline established (69% accuracy), initial steering attempts failed (Exp 1-2), comprehensive literature review complete, ready for systematic diagnosis*
