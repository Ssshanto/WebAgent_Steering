# WebAgent_Steering: invariants

## objective
- primary: causal MI evidence for zero-shot MiniWob web agents
- secondary: non-negative inference-time performance delta
- priority order: MI validity > reproducibility > performance

## phase-1 decomposition (fixed)
- `A`: action type selection (operation family)
- `G`: BID grounding (target element binding)
- `S`: action syntax validity (parser/runtime executable)
- correction/recovery is tracked separately (not a core localization axis)

## hypotheses
- H1: `A/G/S` competence is partially localizable (layer/component/feature)
- H2: interventions on validated mechanisms improve target factor more than non-target factors
- H3: targeted interventions can improve success without major parse-fail regression

## scope and constraints
- benchmark: MiniWob with BrowserGym semantics only
- candidate/deployment default model: `qwen3-1.7b` (0.6b for probe/exploration only)
- zero-shot policy: no finetune, LoRA, adapters, in-task labels, or prior GT access
- allowed interventions: inference-time only in primary loop
- no early stopping: full-layer localization + controls coverage is required

## runtime
- primary local workspace: `/home/ssshanto/Documents/WebAgent_Steering`
- remote host profile: `bio` (`~/Documents/Reaz/WebAgent_Steering`). all experiments must be run on `bio`
- code changes must be made locally, then synchronized over github to `bio`
- env: `steer`
- tmux: `steer`. this tmux session's first window has github authenticated
- miniwob: `MINIWOB_URL=http://localhost:8080/miniwob/`
- hardware: single RTX 3090
- heavy runs serialized by default

## code-grounded execution rules
- baseline first, then intervention
- reuse frozen baseline episodes for paired comparisons under identical `(task, seed)`
- seed policy for comparisons: `--seed 0`
- eval episodes: fixed 3/task (`evaluate`)
- parse-fail metric: non-empty `*_error` episode rate (`base_parse_fail`, `steer_parse_fail`)
- deprecated as invalid evidence: full-layer hard-zero residual ablation and alpha-sweep steering baselines

## method ladder (single active methodology)
- stage gate (mandatory): `exploration -> candidate -> deployment`
- promotion rule: no claim is deployable until it passes paired baseline/intervention, controls, and factor metrics
- active methodology only:
  - `0.1 instrumentation`: residual-stream hook capture at fixed hook families/layers with reusable train/val splits
  - `0.2 feature learning`: train per-hook SAE dictionaries with reconstruction+sparsity quality gates
  - `0.3 causal validation`: latent suppression/amplification (encode-edit-decode) to test necessity/sufficiency on `A/G/S`
  - `0.4 controls`: size-matched random-feature/random-direction controls (>=5) with mean/std and paired effect vs primary intervention
  - `0.5 confound suppression`: constrained decoding + parser/runtime sanity checks to isolate syntax artifacts
- completion criteria for this methodology:
  - target factor, falsifier, stop condition, and non-target regression bound are declared before run
  - fixed train/validation split manifests are frozen and reused
  - candidate claim must beat random controls on target factor without major non-target regression
  - deployment claim requires reproducibility under invariants

## literature-grounded execution (mandatory)
- before implementing a rung, ground it in literature with at least one concrete anchor (method + venue/year) and one known limitation/failure mode, invoke the agent 'research-critic' to find literature/implementation and go off that.
- required per rung: `why this method`, `what would falsify it`, `what control is size-matched`, `why this is better than a baseline no-op/random intervention`
- default baseline for comparison is strongest available non-intervened policy under same prompt/action constraints
- evidence label discipline:
  - `exploratory`: no deployment claim allowed
  - `candidate`: causal evidence with controls, still provisional
  - `deployment`: validated and reproducible under invariants

## specialist-agent consultation protocol
- the autonomous agent must consult specialists when blocked or when control/falsifier design is weak:
  - `librarian`: literature grounding, prior methods, controls, known pitfalls
  - `oracle` (or `research-critic` if available): reviewer-style risk analysis, falsifiers, claim narrowing
  - `explore`: codebase implementation mapping and experiment integration points
  - optional non-conventional framing: `momus`
- consultation trigger conditions:
  - unclear implementation path for localization/validation/controls
  - weak/uncertain falsifier or control design
  - repeated invalid runs or contradictory metrics
- consultation outputs must be logged in provenance as: `question`, `answer summary`, `decision impact`

## controls and validity
- mandatory controls for causal claims: size-matched random direction/component controls
- control protocol minimum: run at least 5 paired random-control repeats per candidate intervention and report mean/std + paired effect vs primary intervention
- report factor-specific effects (`A/G/S`), not only aggregate success
- if targeted gain is paired with major non-target regression, mark as invalid for deployment

## autonomous decision protocol
- loop: Plan -> Execute -> Verify -> Report -> Decide
- Plan: choose next step of the active methodology by highest expected information gain
- Verify: enforce schema + key metrics before accepting run
- Decide: continue until active methodology completion criteria are met or hard stop
- hard stop: environment failure, missing baseline pairing, or repeated invalid runs

## reporting contract (required per run)
- quant: baseline_success, steer_success, delta, base_parse_fail, steer_parse_fail
- factor metrics: `A/G/S` error rates and deltas
- controls: random-control result and delta vs primary intervention
- qual: before/after trajectory examples with mechanism note
- provenance: rationale + literature anchor + why-next-method

## artifact roles (precedence)
- `AGENTS.md`: immutable policy/invariants (this file)
- `MEMORY.md`: mutable run state, queue, blockers
- `RESEARCH.md`: durable decisions, method basis, failures, insights
