# MEMORY (mutable run state)

## current focus
- objective: MI-first, performance-second
- phase: phase-1 (`A/G/S` decomposition)
- active_model: pending (`qwen3-1.7b` default; `qwen3-0.6b` probe only)
- active_method: pending
- seed: `0`

## run context
- local workspace: `/home/ssshanto/Documents/WebAgent_Steering`
- remote host: `bio` (`~/Documents/Reaz/WebAgent_Steering`)
- env: `steer`
- tmux: `steer`
- benchmark: MiniWob (BrowserGym)

## autonomous loop state
- loop_step: plan
- last_completed_step: none
- current_hypothesis: pending
- current_target_factor: pending (`A|G|S`)

## method queue (coverage order)
- [ ] 0.1 localization: activation patching / causal tracing / attribution scans for `A/G/S`
- [ ] 0.2 causal validation: head/component/path ablation + necessity/sufficiency checks
- [ ] 0.3 confound suppression: constrained action decoding + parser/runtime sanity checks
- [ ] 0.4 gated intervention: targeted inference-time intervention sweeps (validated mechanisms only)
- [ ] 0.5 controls: paired random-control repeats (>=5) + effect comparison
- [ ] tier1-a: probes or tuned-lens analysis (analysis-only)
- [ ] tier1-b: DAS/interchange analysis if counterfactual pairs are available (analysis-only)
- [ ] tier1-c: SAE/dictionary or mediation analysis only if explicitly approved
- [ ] cross-method synthesis + decision update in `RESEARCH.md`

## code-grounded reminders
- eval uses fixed 3 episodes/task
- `--steer-only` requires baseline JSONL with matching `(task, seed)`
- `--base-only` and `--steer-only` cannot both be set
- parse-fail metric uses non-empty `*_error` per episode
- default prompt may be verbose unless `--strict-action-prompt`
- qwen3 path forces `enable_thinking=False` (+ `/no_think` fallback)

## latest results
- none

## latest insights
- decomposition decision fixed to `A/G/S`; correction tracked separately

## latest qualitative examples
- none

## blockers
- none

## next actions
- run baseline-first for `qwen3-1.7b` with strict prompt settings
- run first tier0 causal localization method
- log run decision + evidence + why-next-method in `RESEARCH.md`

## canonical dirs
- `results/`
- `vectors/`
- `runtime_state/`
- `logs/`
