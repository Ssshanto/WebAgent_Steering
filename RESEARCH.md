# RESEARCH (durable decisions + literature basis)

## scope
- focus: mechanistic interpretability for zero-shot web-agent action competence
- benchmark: MiniWob (BrowserGym semantics)
- model cycle: `qwen3-0.6b` -> `qwen3-1.7b` -> `qwen3-4b`
- deployment constraint: inference-time interventions only

## core decision (2026-02-10)
- phase-1 decomposition is fixed to:
  - `A`: action type selection
  - `G`: BID grounding
  - `S`: action syntax validity
- correction/recovery is logged separately, not treated as a core localization axis in phase-1

## claim scope (current)
- target claim: localizable and causally intervenable mechanisms for `A/G/S`
- non-claim for now: strong statistical generalization claims across many seeds
- performance improvement is secondary and accepted only if MI validity remains intact

## method basis by compatibility

### strict zero-shot compatible (no training, inference-time analysis/intervention)
- **Causal tracing / activation patching**
  - basis: Meng et al., *Locating and Editing Factual Associations in GPT* (NeurIPS 2022); Heimersheim and Nanda, *How to use and interpret activation patching* (arXiv 2024)
  - use here: localize causal layers/components for `A/G/S` with clean-vs-corrupted trajectory pairs
  - output: causal effect map by layer/component and factor-specific deltas

- **Path/head/component ablation and patching**
  - basis: Wang et al., *Interpretability in the Wild: a Circuit for IOI in GPT-2 small* (ICLR 2023); Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability* (NeurIPS 2023)
  - use here: identify minimal mechanism subsets that control action-type, grounding, or syntax behavior
  - output: candidate mechanism graph + necessity/sufficiency evidence

- **Inference-time activation steering (validated sites only)**
  - basis: Turner et al., activation addition / activation engineering line (2023)
  - use here: intervene only on causally validated layers/components for specific factor repairs (`A` or `G` or `S`)
  - output: factor-specific intervention deltas and side-effect profile

- **Constrained action decoding checks**
  - basis: benchmark-action semantics and parser/runtime constraints from BrowserGym/MiniWob
  - use here: isolate syntax-validity effects and prevent invalid action serialization from confounding MI interpretation
  - output: syntax-failure reduction evidence and interaction with `A/G`

### auxiliary analysis (allowed for analysis, not default deployment path)
- **DAS / interchange interventions**
  - basis: Geiger et al., *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations* (CLeaR 2024)
  - use here: test whether subspaces align with `A/G/S` variables under counterfactual swaps
  - output: interchange consistency scores and variable-aligned subspaces

- **Tuned lens / readout trajectories**
  - basis: Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023)
  - use here: inspect where in depth action token commitments emerge
  - output: layerwise latent-prediction trajectory for action decisions

- **Sparse dictionary / SAE feature analysis**
  - basis: Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (ICLR 2024); Braun et al., *Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning* (NeurIPS 2024)
  - use here: factor-linked feature discovery and causal feature tests
  - output: feature-level interpretations and intervention candidates

- **Causal mediation analysis**
  - basis: mediation methods literature for pathway decomposition
  - use here: quantify pathway contributions (instruction->action direct vs DOM-mediated)
  - output: direct/indirect effect estimates for `A/G/S`-related decisions

## web-agent literature anchors for decomposition/evaluation
- Deng et al., *Mind2Web: Towards a Generalist Agent for the Web* (NeurIPS 2023 Spotlight): operation/element/value decomposition supports action-type vs grounding separation
- Zhou et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents* (ICLR 2024): realistic failure traces and end-task difficulty context
- BrowserGym ecosystem paper (2024): unified action semantics and evaluation environment context

## method execution contract (per run)
- declare target factor: `A` or `G` or `S`
- declare control: random/size-matched counterpart
- declare expected falsifier (what result would reject mechanism hypothesis)
- run baseline pairing with matching `(task, seed)` when using `--steer-only`
- report:
  - quant: baseline_success, steer_success, delta, base_parse_fail, steer_parse_fail
  - factor metrics: target and non-target factor deltas
  - controls: delta vs random-control run
  - qual: at least one before/after trajectory with mechanism note

## durable decision log

### Decision: D-2026-02-10-phase1-threeway
- context: MI-first phase-1 needed a fixed capability decomposition
- choice: `A/G/S` as core axes; correction tracked separately
- rationale: directly observable from action strings, DOM binding, and parser/runtime outcomes
- implications: all interventions and reports must include factor-specific deltas

### Decision: D-2026-02-10-method-compatibility
- context: method set had mixed compatibility with strict zero-shot constraints
- choice: tier0 (strict no-training) is default execution path; tier1 methods are analysis-only unless explicitly promoted
- rationale: keeps primary loop compliant with zero-shot + inference-time-only policy
- implications: probes/SAE/DAS/tuned-lens can guide hypotheses but not be silently treated as deployment evidence

## templates

### run record template
```
## Run: <id>
- date:
- model:
- method_family:
- target_factor: A|G|S
- heuristic_rationale:
- paper_inspirations: [title/year/venue]
- quantitative_ref: [results path(s)]
- qualitative_examples_ref: [example ids / paths]
- decision_next:
```

### failure record template
```
## Failure: <id>
- date:
- hypothesis:
- method:
- target_factor:
- result:
- failure_reason:
- update_after_failure:
```

### insight template
```
## Insight: <id>
- date:
- evidence_ref:
- confidence: low|medium|high
- implication:
```
