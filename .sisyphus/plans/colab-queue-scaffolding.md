# Colab-Ready Experiment Queue Scaffolding

## TL;DR

> **Quick Summary**: Add a platform-agnostic queue runner and tracked experiment-set configs so batches run the same way on Colab, local machines, or VMs, while keeping Colab runtime glue disposable and gitignored.
>
> **Deliverables**:
> - `scripts/run_experiment_queue.py` (generic queue/resume orchestrator)
> - `configs/experiments/colab_batch_a.yaml` (tracked 10-12 experiment set)
> - `configs/experiments/example.local.yaml` (portable example)
> - `.gitignore` updates for Colab-only notebooks and runtime state
> - README usage section for queue workflow
>
> **Estimated Effort**: Short
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 6

---

## Context

### Original Request
Plan scaffolding to run many short steering experiments with one manual Colab start step, while keeping the repository clean and avoiding permanent Colab-specific hacks.

### Interview Summary
**Key Discussions**:
- Queue/config code should be reusable on real machines, not only Colab.
- Colab-specific launcher/notebook files should be minimal and gitignored.
- VSCode + Google Colab extension workflow is valid for the user context.
- Test strategy confirmed: no automated tests for this temporary scaffolding.

**Research Findings**:
- Existing orchestration conventions are argparse-based and file-output driven.
- Existing run outputs are JSONL/TSV and already compatible with queue orchestration.
- Runtime artifact ignore pattern already exists and can be extended cleanly.

### Metis Review
**Identified Gaps (addressed)**:
- Retry/stale policy not explicit -> defined as configurable with safe defaults.
- Scope creep risk (dashboards/daemons/parsing frameworks) -> explicitly forbidden.
- State corruption risk -> atomic state writes required.
- Resume semantics ambiguity -> defined as job-level atomic resume only.

---

## Work Objectives

### Core Objective
Create a minimal, reusable queue-execution layer that batches experiment runs through existing scripts, resumes reliably after interruption, and keeps Colab-only workflow glue outside committed experiment code.

### Concrete Deliverables
- `scripts/run_experiment_queue.py`
- `configs/experiments/colab_batch_a.yaml`
- `configs/experiments/example.local.yaml`
- `.gitignore` additions for `colab/` and queue state files
- README section describing queue workflow and one-step Colab launch flow

### Definition of Done
- [x] `python scripts/run_experiment_queue.py --help` runs and documents subcommands/flags.
- [x] A queue config can be loaded and state initialized successfully.
- [x] Running the queue executes pending jobs sequentially using `scripts/run_sweep.py`.
- [x] Interruption + rerun resumes unfinished jobs without duplicating completed jobs.
- [x] Colab-only files are excluded from git tracking.

### Must Have
- Platform-agnostic queue logic in tracked code.
- Linear sequential execution (no distributed scheduler).
- Human-readable, atomic state persistence.
- Reuse `scripts/run_sweep.py` as worker command.
- Default stale threshold: `7200` seconds.
- Default retry policy: no automatic retry (`max_attempts: 1`), failed jobs remain failed until explicit rerun.
- Default concurrency model: single-operator execution (no lock manager).

### Must NOT Have (Guardrails)
- No Colab UI/browser automation in tracked code.
- No background daemon/service architecture.
- No additional heavy dependencies or DB setup.
- No scope expansion into report dashboards/notification systems.

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> All task verification is executable by an agent via shell commands and file assertions. No acceptance step may require manual clicking or visual confirmation.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None (user-confirmed)
- **Framework**: none

### Agent-Executed QA Scenarios (MANDATORY)

Scenario: Queue script loads and prints CLI help
  Tool: Bash
  Preconditions: dependencies installed
  Steps:
    1. Run `python scripts/run_experiment_queue.py --help`
    2. Assert exit code is 0
    3. Assert output contains `init`, `run`, and `status` subcommands
  Expected Result: help text renders with expected commands
  Failure Indicators: non-zero exit code or missing subcommands
  Evidence: terminal output capture

Scenario: Initialize state from config
  Tool: Bash
  Preconditions: `configs/experiments/colab_batch_a.yaml` exists
  Steps:
    1. Run `python scripts/run_experiment_queue.py init --config configs/experiments/colab_batch_a.yaml --state runtime_state/queue_state.json`
    2. Assert exit code is 0
    3. Assert file `runtime_state/queue_state.json` exists
    4. Assert state has jobs with `pending` status
  Expected Result: valid state initialized from config
  Failure Indicators: init command fails or malformed state
  Evidence: state file + command output

Scenario: Run queue sequentially and update statuses
  Tool: Bash
  Preconditions: queue state initialized
  Steps:
    1. Run `python scripts/run_experiment_queue.py run --config configs/experiments/colab_batch_a.yaml --state runtime_state/queue_state.json`
    2. Assert each job transitions `pending -> running -> done|failed`
    3. Assert completed jobs include output paths and return codes
  Expected Result: queue completes pass over all pending jobs
  Failure Indicators: stuck `running` jobs without heartbeat updates
  Evidence: updated state file + terminal log

Scenario: Resume after interruption
  Tool: Bash
  Preconditions: state contains at least one `done` and one `pending`
  Steps:
    1. Re-run `python scripts/run_experiment_queue.py run --config configs/experiments/colab_batch_a.yaml --state runtime_state/queue_state.json --resume`
    2. Assert `done` jobs are skipped
    3. Assert only unfinished jobs run
  Expected Result: idempotent resume behavior
  Failure Indicators: completed jobs rerun unexpectedly
  Evidence: state diff before/after rerun

Scenario: Stale-running recovery
  Tool: Bash
  Preconditions: a job has status `running` with stale timestamp
  Steps:
    1. Run `python scripts/run_experiment_queue.py recover --state runtime_state/queue_state.json --stale-seconds 7200`
    2. Assert stale jobs transition to `pending` or `failed` per retry policy
    3. Run `status` and verify counts reflect recovery
  Expected Result: queue unblocks after interrupted sessions
  Failure Indicators: stale jobs remain forever in `running`
  Evidence: recovered state file + status output

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Start Immediately):
- Task 1: Finalize queue spec + state model
- Task 5: Define `.gitignore` and Colab artifact boundaries

Wave 2 (After Wave 1):
- Task 2: Implement `scripts/run_experiment_queue.py`
- Task 3: Add tracked experiment config files
- Task 4: Add README usage section

Wave 3 (After Wave 2):
- Task 6: Execute QA scenarios and finalize acceptance checks

Critical Path: 1 -> 2 -> 3 -> 4 -> 6

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | 5 |
| 2 | 1 | 6 | 3, 4 |
| 3 | 1 | 6 | 2, 4 |
| 4 | 2, 3 | 6 | None |
| 5 | None | 6 | 1 |
| 6 | 2, 3, 4, 5 | None | None |

---

## TODOs

- [x] 1. Define queue state model and CLI contract

  **What to do**:
  - Define job schema: `id`, `status`, `attempt`, `max_attempts`, `command`, `out_dir`, timestamps, `return_code`.
  - Define state schema: queue metadata, counters, stale detection fields.
  - Define subcommands and flags: `init`, `run`, `status`, `recover`.

  **Must NOT do**:
  - No DAG/dependency graph execution.
  - No dynamic queue mutation from remote services.

  **Recommended Agent Profile**:
  - Category: `quick`
    - Reason: small, bounded Python CLI schema design.
  - Skills: `git-master`
    - `git-master`: maintain repository-consistent CLI style and minimal diffs.
  - Skills Evaluated but Omitted:
    - `playwright`: not needed for non-browser scripting.

  **Parallelization**:
  - Can Run In Parallel: YES
  - Parallel Group: Wave 1 (with Task 5)
  - Blocks: 2, 3
  - Blocked By: None

  **References**:
  - `scripts/run_sweep.py:324` - argparse parser structure to mirror.
  - `scripts/run_sweep.py:372` - output-dir flag semantics to reuse.
  - `scripts/run_sweep.py:427` - optional summary-path handling pattern.
  - `src/miniwob_steer.py:1236` - base JSONL load/indexing convention.

  **Acceptance Criteria**:
  - [ ] CLI contract documented in module docstring and `--help` output.
  - [ ] State schema supports deterministic resume and retry accounting.

- [x] 2. Implement `scripts/run_experiment_queue.py`

  **What to do**:
  - Implement subcommands (`init`, `run`, `status`, `recover`).
  - Build worker invocation with `subprocess.run` against `scripts/run_sweep.py`.
  - Persist state atomically (tmp file + replace).
  - Track per-job transitions and failure reasons.

  **Must NOT do**:
  - No direct model/training logic inside queue script.
  - No Colab-specific imports or runtime coupling.

  **Recommended Agent Profile**:
  - Category: `unspecified-low`
    - Reason: moderate scripting with state machine behavior.
  - Skills: `git-master`
    - `git-master`: keep implementation aligned to repository scripting conventions.
  - Skills Evaluated but Omitted:
    - `frontend-ui-ux`: irrelevant for CLI scripting.

  **Parallelization**:
  - Can Run In Parallel: NO
  - Parallel Group: Wave 2 (critical implementation)
  - Blocks: 4, 6
  - Blocked By: 1

  **References**:
  - `scripts/run_sweep.py:65` - top-level runner flow pattern.
  - `scripts/run_sweep.py:111` - directory creation behavior.
  - `scripts/run_sweep.py:209` - robust file writing with utf-8/newline.
  - `scripts/run_sweep.py:305` - flush/close completion flow.

  **Acceptance Criteria**:
  - [ ] `python scripts/run_experiment_queue.py --help` succeeds.
  - [ ] `init` writes valid state from config.
  - [ ] `run` executes jobs sequentially and persists transitions.
  - [ ] `recover` handles stale-running jobs by policy.

- [x] 3. Add tracked experiment configs in `configs/experiments/`

  **What to do**:
  - Add one real batch config (`colab_batch_a.yaml`) for 10-12 experiments.
  - Add one portable local example (`example.local.yaml`).
  - Ensure each job maps cleanly to `run_sweep.py` flags.

  **Must NOT do**:
  - No Colab notebook metadata in configs.
  - No environment-specific hardcoded absolute paths.

  **Recommended Agent Profile**:
  - Category: `quick`
    - Reason: structured config authoring.
  - Skills: `git-master`
    - `git-master`: maintain concise, reviewable tracked config changes.
  - Skills Evaluated but Omitted:
    - `playwright`: not required.

  **Parallelization**:
  - Can Run In Parallel: YES
  - Parallel Group: Wave 2 (with Task 2)
  - Blocks: 4, 6
  - Blocked By: 1

  **References**:
  - `scripts/run_sweep.py:328` - model flag constraints.
  - `scripts/run_sweep.py:362` - layer range/list syntax.
  - `scripts/run_sweep.py:367` - alpha list syntax.
  - `README.md:76` - canonical CLI usage examples.

  **Acceptance Criteria**:
  - [ ] Config parses and validates required keys.
  - [ ] All job entries have unique IDs and deterministic output locations.

- [x] 4. Document queue workflow and Colab boundary in README

  **What to do**:
  - Add a short section describing generic queue usage.
  - Add Colab usage note: one manual runtime attach step, then queue run command.
  - Clarify repo cleanliness policy for Colab-only files.

  **Must NOT do**:
  - No long operational playbook.
  - No hard dependency on VSCode extension in core workflow docs.

  **Recommended Agent Profile**:
  - Category: `writing`
    - Reason: focused documentation edit.
  - Skills: `git-master`
    - `git-master`: ensure doc changes stay concise and scoped.
  - Skills Evaluated but Omitted:
    - `frontend-ui-ux`: not relevant.

  **Parallelization**:
  - Can Run In Parallel: YES
  - Parallel Group: Wave 2 (after Task 2/3 drafts)
  - Blocks: 6
  - Blocked By: 2, 3

  **References**:
  - `README.md:17` - setup section style.
  - `README.md:74` - CLI usage section format.
  - `README.md:86` - output description format.

  **Acceptance Criteria**:
  - [ ] README includes queue quick-start command.
  - [ ] README clearly distinguishes tracked reusable code vs gitignored Colab glue.

- [x] 5. Update `.gitignore` for Colab/runtime-only artifacts

  **What to do**:
  - Add ignore entries for `colab/`, `runtime_state/`, and queue state files.
  - Keep tracked `configs/experiments/` untouched by ignore rules.

  **Must NOT do**:
  - No broad ignore patterns that hide tracked experiment configs.

  **Recommended Agent Profile**:
  - Category: `quick`
    - Reason: small, low-risk ignore updates.
  - Skills: `git-master`
    - `git-master`: avoid accidental masking of tracked files.
  - Skills Evaluated but Omitted:
    - `playwright`: irrelevant.

  **Parallelization**:
  - Can Run In Parallel: YES
  - Parallel Group: Wave 1 (with Task 1)
  - Blocks: 6
  - Blocked By: None

  **References**:
  - `.gitignore:5` - existing results directory ignore convention.
  - `.gitignore:6` - vectors cache ignore convention.
  - `.gitignore:14` - local working-notes policy.

  **Acceptance Criteria**:
  - [ ] `git status` does not show Colab notebook/runtime files after creation.
  - [ ] Tracked config files remain visible and commit-able.

- [x] 6. Run agent-executed QA scenarios and verify resume behavior

  **What to do**:
  - Execute init/run/status/recover flows with sample config.
  - Simulate interruption and verify idempotent resume.
  - Capture evidence from terminal output and state snapshots.

  **Must NOT do**:
  - No manual-only verification steps.
  - No skipping stale/recovery checks.

  **Recommended Agent Profile**:
  - Category: `unspecified-low`
    - Reason: command-driven integration verification.
  - Skills: `git-master`
    - `git-master`: ensures final validation and clean change set.
  - Skills Evaluated but Omitted:
    - `playwright`: browser testing not needed.

  **Parallelization**:
  - Can Run In Parallel: NO
  - Parallel Group: Wave 3 (final)
  - Blocks: None
  - Blocked By: 2, 3, 4, 5

  **References**:
  - `scripts/run_sweep.py:261` - per-run output path expectations.
  - `scripts/run_sweep.py:307` - final summary reporting expectations.

  **Acceptance Criteria**:
  - [ ] Resume skips previously completed jobs.
  - [ ] Recovery handles stale-running jobs as configured.
  - [ ] Queue terminal summary reflects accurate counts.

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1-3 | `feat(queue): add portable experiment queue scaffold` | queue script + configs | queue help/init checks |
| 4-5 | `docs(chore): document queue flow and ignore colab artifacts` | README + .gitignore | docs/git-status checks |
| 6 | `chore(qa): verify queue resume and stale recovery flow` | none or evidence notes | run/status/recover commands |

---

## Success Criteria

### Verification Commands
```bash
python scripts/run_experiment_queue.py --help
python scripts/run_experiment_queue.py init --config configs/experiments/colab_batch_a.yaml --state runtime_state/queue_state.json
python scripts/run_experiment_queue.py run --config configs/experiments/colab_batch_a.yaml --state runtime_state/queue_state.json --resume
python scripts/run_experiment_queue.py status --state runtime_state/queue_state.json
python scripts/run_experiment_queue.py recover --state runtime_state/queue_state.json --stale-seconds 7200
```

### Final Checklist
- [x] Queue scaffold is generic and reusable beyond Colab.
- [x] Colab-only glue remains outside tracked experiment code.
- [x] Resume/recovery logic is deterministic at job granularity.
- [x] Repository stays clean with explicit ignore rules for runtime artifacts.
