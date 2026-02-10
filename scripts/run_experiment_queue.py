#!/usr/bin/env python3
"""
Experiment Queue Orchestrator

Platform-agnostic batch job runner with resume support and atomic state persistence.
Executes jobs sequentially by invoking `scripts/run_sweep.py` as a worker subprocess.

CLI Contract:
    init    - Create state from config YAML
    run     - Execute jobs sequentially with resume support
    status  - Display queue status
    recover - Fix stale-running jobs

State is persisted atomically (tmp file + os.rename) to prevent corruption.
Resume is deterministic and idempotent (skip completed, retry failed within limits).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ============================================================================
# Job & State Models
# ============================================================================


class Job:
    """Job object representing a single run_sweep.py invocation."""

    def __init__(self, data: Dict[str, Any]):
        self.id = data["id"]
        self.status = data["status"]
        self.attempt = data["attempt"]
        self.max_attempts = data["max_attempts"]
        self.command = data["command"]
        self.out_dir = data["out_dir"]
        self.created_at = data["created_at"]
        self.started_at = data.get("started_at")
        self.completed_at = data.get("completed_at")
        self.return_code = data.get("return_code")
        self.error_message = data.get("error_message")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "command": self.command,
            "out_dir": self.out_dir,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "return_code": self.return_code,
            "error_message": self.error_message,
        }


class QueueState:
    """Queue state container with atomic persistence."""

    def __init__(self, data: Dict[str, Any]):
        self.queue_id = data["queue_id"]
        self.created_at = data["created_at"]
        self.updated_at = data["updated_at"]
        self.jobs = [Job(j) for j in data["jobs"]]
        self.counters = data["counters"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "jobs": [j.to_dict() for j in self.jobs],
            "counters": self.counters,
        }

    def recompute_counters(self):
        """Recompute counters from current job statuses."""
        self.counters = {
            "total": len(self.jobs),
            "pending": sum(1 for j in self.jobs if j.status == "pending"),
            "running": sum(1 for j in self.jobs if j.status == "running"),
            "done": sum(1 for j in self.jobs if j.status == "done"),
            "failed": sum(1 for j in self.jobs if j.status == "failed"),
        }


# ============================================================================
# Utilities
# ============================================================================


def utc_now() -> str:
    """Return current UTC timestamp in ISO8601 format."""
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO8601 timestamp to datetime object."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config from file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(path: str) -> QueueState:
    """Load state from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return QueueState(data)


def save_state(state: QueueState, path: str):
    """Save state atomically (tmp file + os.rename)."""
    state.updated_at = utc_now()
    state.recompute_counters()

    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)

    os.rename(temp_path, path)


def build_command(job_config: Dict[str, Any]) -> str:
    """Build run_sweep.py command from job config fields.

    Maps config YAML keys to run_sweep.py flags:
    - snake_case → dashes (e.g., prompt_type → prompt-type)
    - output_dir → out-dir
    - Auto-append summary-path as {output_dir}/summary.tsv
    """
    parts = ["python scripts/run_sweep.py"]

    # Field mapping
    field_map = {
        "model": "--model",
        "layer": "--layers",
        "layers": "--layers",
        "alpha": "--alphas",
        "alphas": "--alphas",
        "prompt_type": "--prompt-type",
        "vector_method": "--vector-method",
        "train_steps": "--train-steps",
        "tasks": "--tasks",
        "task_manifest": "--task-manifest",
        "output_dir": "--out-dir",
        "cache_dir": "--cache-dir",
        "seed": "--seed",
        "order": "--order",
        "max_new_tokens": "--max-new-tokens",
        "max_elems": "--max-elems",
    }

    # Add fields
    for key, flag in field_map.items():
        if key in job_config:
            parts.append(f"{flag} {job_config[key]}")

    # Boolean flags
    if job_config.get("base_only"):
        parts.append("--base-only")
    if job_config.get("steer_only"):
        parts.append("--steer-only")
    if job_config.get("force_recompute"):
        parts.append("--force-recompute")
    if job_config.get("quiet"):
        parts.append("--quiet")
    if job_config.get("no_progress"):
        parts.append("--no-progress")
    if job_config.get("strict_action_prompt"):
        parts.append("--strict-action-prompt")
    if job_config.get("random_control"):
        parts.append("--random-control")

    # Optional with value
    if "base_jsonl" in job_config:
        parts.append(f"--base-jsonl {job_config['base_jsonl']}")

    # Auto-append summary-path
    output_dir = job_config.get("output_dir", "results")
    parts.append(f"--summary-path {output_dir}/summary.tsv")

    return " ".join(parts)


# ============================================================================
# Subcommand Handlers
# ============================================================================


def cmd_init(args):
    """Initialize queue state from config YAML."""
    # Check if state exists
    if os.path.exists(args.state) and not args.force:
        print(f"ERROR: State file already exists: {args.state}")
        print("Use --force to overwrite.")
        sys.exit(1)

    # Load config
    try:
        config = load_yaml(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    # Validate config
    if "queue_id" not in config:
        print("ERROR: Config missing required key: queue_id")
        sys.exit(1)
    if "jobs" not in config or not isinstance(config["jobs"], list):
        print("ERROR: Config missing required key: jobs (must be array)")
        sys.exit(1)

    # Build jobs
    jobs = []
    for idx, job_config in enumerate(config["jobs"]):
        if "id" not in job_config:
            print(f"ERROR: Job {idx} missing required key: id")
            sys.exit(1)
        if "output_dir" not in job_config:
            print(f"ERROR: Job {job_config['id']} missing required key: output_dir")
            sys.exit(1)

        command = build_command(job_config)

        job = Job(
            {
                "id": job_config["id"],
                "status": "pending",
                "attempt": 0,
                "max_attempts": job_config.get("max_attempts", 1),
                "command": command,
                "out_dir": job_config["output_dir"],
                "created_at": utc_now(),
                "started_at": None,
                "completed_at": None,
                "return_code": None,
                "error_message": None,
            }
        )
        jobs.append(job)

    # Create state
    state = QueueState(
        {
            "queue_id": config["queue_id"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "jobs": [j.to_dict() for j in jobs],
            "counters": {},
        }
    )
    state.recompute_counters()

    # Create state directory
    state_dir = os.path.dirname(args.state)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    # Write state
    save_state(state, args.state)

    print(f"Initialized queue: {config['queue_id']}")
    print(f"Total jobs: {len(jobs)}")
    print(f"State file: {args.state}")


def cmd_run(args):
    """Execute jobs sequentially with optional resume."""
    # Load state
    if not os.path.exists(args.state):
        print(f"ERROR: State file not found: {args.state}")
        print("Use 'init' subcommand to create state.")
        sys.exit(1)

    try:
        state = load_state(args.state)
    except Exception as e:
        print(f"ERROR: Failed to load state: {e}")
        sys.exit(1)

    print(f"Queue: {state.queue_id}")
    print(f"Total jobs: {state.counters['total']}")

    # Filter jobs
    jobs_to_run = []
    for job in state.jobs:
        if job.status == "done":
            if args.resume:
                continue
            else:
                jobs_to_run.append(job)
        elif job.status == "failed" and job.attempt >= job.max_attempts:
            if args.resume:
                continue
            else:
                jobs_to_run.append(job)
        else:
            jobs_to_run.append(job)

    if not jobs_to_run:
        print("No jobs to run.")
        return

    print(f"Jobs to run: {len(jobs_to_run)}")
    if args.resume:
        print("Resume mode: skipping completed/failed jobs")

    # Execute jobs
    for idx, job in enumerate(jobs_to_run, 1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(jobs_to_run)}] {job.id}")
        print(f"{'=' * 60}")

        # Skip if done or exhausted
        if job.status == "done":
            print(f"Status: done (skipped)")
            continue
        if job.status == "failed" and job.attempt >= job.max_attempts:
            print(f"Status: failed (exhausted, skipped)")
            continue

        # Prepare job
        job.status = "running"
        job.attempt += 1
        job.started_at = utc_now()
        save_state(state, args.state)

        print(f"Status: running (attempt {job.attempt}/{job.max_attempts})")
        print(f"Started: {job.started_at}")
        print(f"Command: {job.command}")

        # Create output directory
        try:
            os.makedirs(job.out_dir, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Failed to create output directory: {e}")
            job.status = "failed"
            job.error_message = f"Failed to create output directory: {e}"
            job.completed_at = utc_now()
            save_state(state, args.state)
            continue

        # Execute command
        if args.dry_run:
            print("DRY RUN: Command would execute here")
            rc = 0
        else:
            start_time = datetime.now()
            if args.capture_worker_output:
                result = subprocess.run(
                    job.command,
                    shell=True,
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            else:
                result = subprocess.run(
                    job.command,
                    shell=True,
                    cwd=os.getcwd(),
                )
            rc = result.returncode
            elapsed = datetime.now() - start_time

            print(f"\nReturn code: {rc}")
            print(f"Elapsed: {elapsed}")
            if args.capture_worker_output and rc != 0:
                print("Worker output (failure):")
                print(result.stdout or "")

        # Update job status
        job.return_code = rc
        job.completed_at = utc_now()

        if rc == 0:
            job.status = "done"
            print(f"Status: done")
        else:
            job.status = "failed"
            job.error_message = f"Command exited with code {rc}"
            print(f"Status: failed (rc={rc})")

        save_state(state, args.state)

    # Final summary
    state.recompute_counters()
    print(f"\n{'=' * 60}")
    print("QUEUE COMPLETE")
    print(f"{'=' * 60}")
    print(
        f"Status: {state.counters['pending']} pending, "
        f"{state.counters['running']} running, "
        f"{state.counters['done']} done, "
        f"{state.counters['failed']} failed"
    )


def cmd_status(args):
    """Display queue status."""
    # Load state
    if not os.path.exists(args.state):
        print(f"ERROR: State file not found: {args.state}")
        sys.exit(1)

    try:
        state = load_state(args.state)
    except Exception as e:
        print(f"ERROR: Failed to load state: {e}")
        sys.exit(1)

    # JSON output
    if args.json:
        print(json.dumps(state.to_dict(), indent=2))
        return

    # Human-readable output
    print(f"Queue: {state.queue_id}")
    print(f"Created: {state.created_at}")
    print(f"Updated: {state.updated_at}")
    print()

    print("Status Summary:")
    print(f"  Pending: {state.counters['pending']}")
    print(f"  Running: {state.counters['running']}")
    print(f"  Done:    {state.counters['done']}")
    print(f"  Failed:  {state.counters['failed']}")
    print(f"  Total:   {state.counters['total']}")
    print()

    print("Jobs:")
    print(
        f"  {'ID':<30} | {'Status':<8} | {'Attempt':<7} | {'RC':<4} | {'Completed':<20}"
    )
    print(f"  {'-' * 30} | {'-' * 8} | {'-' * 7} | {'-' * 4} | {'-' * 20}")

    for job in state.jobs:
        rc_str = str(job.return_code) if job.return_code is not None else "-"
        completed_str = job.completed_at if job.completed_at else "-"
        print(
            f"  {job.id:<30} | {job.status:<8} | {job.attempt:<7} | {rc_str:<4} | {completed_str:<20}"
        )


def cmd_recover(args):
    """Detect and recover stale-running jobs."""
    # Load state
    if not os.path.exists(args.state):
        print(f"ERROR: State file not found: {args.state}")
        sys.exit(1)

    try:
        state = load_state(args.state)
    except Exception as e:
        print(f"ERROR: Failed to load state: {e}")
        sys.exit(1)

    print(f"Queue: {state.queue_id}")
    print(f"Stale threshold: {args.stale_seconds}s")
    print()

    # Detect stale jobs
    now = datetime.now(timezone.utc)
    stale_jobs = []

    for job in state.jobs:
        if job.status == "running" and job.started_at:
            started = parse_timestamp(job.started_at)
            elapsed = (now - started).total_seconds()

            if elapsed > args.stale_seconds:
                stale_jobs.append((job, elapsed))

    if not stale_jobs:
        print("No stale jobs detected.")
        return

    print(f"Found {len(stale_jobs)} stale job(s):")
    print()

    # Recover stale jobs
    for job, elapsed in stale_jobs:
        print(f"Job: {job.id}")
        print(f"  Started: {job.started_at}")
        print(f"  Elapsed: {elapsed:.0f}s")
        print(f"  Attempt: {job.attempt}/{job.max_attempts}")

        if job.attempt < job.max_attempts:
            old_status = job.status
            job.status = "pending"
            print(f"  Action: {old_status} → pending (retry available)")
        else:
            old_status = job.status
            job.status = "failed"
            job.error_message = "stale (exceeded retries)"
            job.completed_at = utc_now()
            print(f"  Action: {old_status} → failed (retries exhausted)")

        print()

    # Write state
    if not args.dry_run:
        save_state(state, args.state)
        print(f"Recovered {len(stale_jobs)} stale job(s).")
    else:
        print("DRY RUN: No changes written.")


# ============================================================================
# Main CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Experiment queue orchestrator for batched run_sweep.py execution"
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: init
    init_parser = subparsers.add_parser(
        "init", help="Create state file from config YAML"
    )
    init_parser.add_argument(
        "--config", required=True, help="Path to experiment config YAML"
    )
    init_parser.add_argument("--state", required=True, help="Output state file path")
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing state file"
    )
    init_parser.set_defaults(func=cmd_init)

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Execute jobs sequentially")
    run_parser.add_argument(
        "--config", required=True, help="Experiment config (for reference)"
    )
    run_parser.add_argument("--state", required=True, help="State file path")
    run_parser.add_argument(
        "--resume", action="store_true", help="Skip completed/failed jobs"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    run_parser.add_argument(
        "--capture-worker-output",
        action="store_true",
        help="Capture worker stdout/stderr; print only on failure",
    )
    run_parser.set_defaults(func=cmd_run)

    # Subcommand: status
    status_parser = subparsers.add_parser("status", help="Display queue status")
    status_parser.add_argument("--state", required=True, help="State file path")
    status_parser.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    status_parser.set_defaults(func=cmd_status)

    # Subcommand: recover
    recover_parser = subparsers.add_parser(
        "recover", help="Detect and recover stale-running jobs"
    )
    recover_parser.add_argument("--state", required=True, help="State file path")
    recover_parser.add_argument(
        "--stale-seconds",
        type=int,
        default=7200,
        help="Stale threshold in seconds (default: 7200 = 2 hours)",
    )
    recover_parser.add_argument(
        "--dry-run", action="store_true", help="Print recovery actions without applying"
    )
    recover_parser.set_defaults(func=cmd_recover)

    # Parse and dispatch
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
