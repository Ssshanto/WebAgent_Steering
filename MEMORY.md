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
- [ ] 0.1 instrumentation: define hook families/layers and freeze train/val split manifests
- [ ] 0.2 feature learning: train per-hook SAE dictionaries and pass quality gates
- [ ] 0.3 causal validation: latent suppression/amplification on val split for target `A/G/S`
- [ ] 0.4 controls: random-feature/random-direction paired controls (>=5 repeats)
- [ ] 0.5 confound suppression: strict decoding + parser/runtime checks
- [ ] cross-method synthesis + decision update in `RESEARCH.md`

## code-grounded reminders
- eval uses fixed 3 episodes/task
- preserve paired baseline/intervention evaluation on identical `(task, seed)`
- parse-fail metric uses non-empty `*_error` per episode
- default prompt may be verbose unless `--strict-action-prompt`
- qwen3 path forces `enable_thinking=False` (+ `/no_think` fallback)
- hard-zero layer ablations and alpha steering sweeps are deprecated as invalid evidence

## latest results
- none

## latest insights
- decomposition decision fixed to `A/G/S`; correction tracked separately

## latest qualitative examples
- none

## blockers
- none

## next actions
- implement hook-capture pipeline with frozen train/val manifests
- train first SAE dictionaries on selected hook families
- run latent suppression/amplification validation with random controls and log in `RESEARCH.md`

## canonical dirs
- `results/`
- `vectors/`
- `runtime_state/`
- `logs/`
