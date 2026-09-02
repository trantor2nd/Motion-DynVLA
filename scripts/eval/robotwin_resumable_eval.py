#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RATE_RE = re.compile(
    r"Success rate:\s*(\d+)/(\d+).*current seed:\s*(\d+)"
)
EXPERT_RE = re.compile(r"Expert check seed=(\d+)")


def clean_output(text: str) -> str:
    return ANSI_RE.sub("", text.replace("\r", "\n"))


def parse_rates(text: str) -> list[tuple[int, int, int]]:
    return [tuple(map(int, match)) for match in RATE_RE.findall(clean_output(text))]


def parse_expert_seeds(text: str) -> list[int]:
    return [int(seed) for seed in EXPERT_RE.findall(clean_output(text))]


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_fixed_seeds(path: Path, task_name: str) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tasks" in payload:
        payload = payload["tasks"]
    if isinstance(payload, dict):
        if task_name not in payload:
            raise RuntimeError(f"Fixed seed manifest has no task {task_name!r}")
        payload = payload[task_name]
    if isinstance(payload, dict) and "seeds" in payload:
        payload = payload["seeds"]
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Fixed seed manifest entry must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in payload):
        raise RuntimeError("Fixed seeds must be non-negative integers")
    if len(set(payload)) != len(payload):
        raise RuntimeError(f"Fixed seeds for {task_name!r} are not unique")
    return payload


def update_summary(state: dict) -> None:
    episodes = state["episodes"]
    successes = sum(int(item["success"]) for item in episodes)
    state["completed_episodes"] = len(episodes)
    state["successes"] = successes
    state["success_rate"] = successes / len(episodes) if episodes else 0.0
    state["complete"] = len(episodes) >= state["target_episodes"]
    state["updated_at"] = datetime.now().astimezone().isoformat()


def bootstrap_episodes(log_path: Path) -> list[dict]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rates = parse_rates(text)
    records = []
    previous_successes = 0
    for numerator, denominator, seed in rates:
        if denominator != len(records) + 1:
            raise RuntimeError(
                f"Bootstrap denominator is not consecutive in {log_path}: {denominator}"
            )
        success = numerator > previous_successes
        records.append(
            {
                "episode_index": len(records),
                "seed": seed,
                "success": success,
                "source": "bootstrap_log",
                "log": str(log_path),
            }
        )
        previous_successes = numerator
    return records


def terminate_process(process: subprocess.Popen, grace_seconds: int = 10) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=grace_seconds)


def load_or_initialize_state(args: argparse.Namespace) -> dict:
    if args.state_file.is_file():
        state = json.loads(args.state_file.read_text(encoding="utf-8"))
        for key, expected in (
            ("task_name", args.task_name),
            ("run_name", args.run_name),
            ("target_episodes", args.target_episodes),
        ):
            if state.get(key) != expected:
                raise RuntimeError(
                    f"State mismatch for {key}: {state.get(key)!r} != {expected!r}"
                )
        if state.get("fixed_seeds") != args.fixed_seeds:
            raise RuntimeError("State fixed seeds do not match the requested manifest")
        update_summary(state)
        if args.fixed_seeds is not None and not state["complete"]:
            state["next_seed"] = args.fixed_seeds[len(state["episodes"])]
        return state

    episodes = []
    if args.bootstrap_log is not None:
        episodes = bootstrap_episodes(args.bootstrap_log)
    if len(episodes) > args.target_episodes:
        raise RuntimeError("Bootstrap log has more episodes than the requested target")

    next_seed = args.fixed_seeds[0] if args.fixed_seeds is not None else args.start_seed
    if episodes:
        next_seed = max(next_seed, max(item["seed"] for item in episodes) + 1)
    state = {
        "schema_version": 1,
        "task_name": args.task_name,
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "target_episodes": args.target_episodes,
        "next_seed": next_seed,
        "fixed_seeds": args.fixed_seeds,
        "fixed_seeds_file": str(args.fixed_seeds_file) if args.fixed_seeds_file else None,
        "episodes": episodes,
        "skipped_seeds": [],
        "attempt_failures": [],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    update_summary(state)
    atomic_write_json(args.state_file, state)
    return state


def record_skipped_seeds(state: dict, seeds: list[int], reason: str, log: Path) -> None:
    existing = {item["seed"] for item in state["skipped_seeds"]}
    completed = {item["seed"] for item in state["episodes"]}
    for seed in seeds:
        if seed in existing or seed in completed:
            continue
        state["skipped_seeds"].append(
            {"seed": seed, "reason": reason, "log": str(log)}
        )


def run_attempt(args: argparse.Namespace, state: dict) -> None:
    episode_index = len(state["episodes"])
    fixed_seeds = state.get("fixed_seeds")
    fixed_mode = fixed_seeds is not None
    start_seed = int(fixed_seeds[episode_index] if fixed_mode else state["next_seed"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attempt_log = args.log_dir / (
        f"{args.run_name}_{args.task_name}_isolated_"
        f"episode{episode_index:02d}_seed{start_seed}_{stamp}.log"
    )
    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.robotwin_python),
        "script/eval_policy.py",
        "--config",
        str(args.config),
        "--policy_ckpt_path",
        str(args.checkpoint),
        "--overrides",
        "--task_name",
        args.task_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--eval_episodes",
        "1",
        "--seed",
        "0",
        "--eval_video_log",
        "False",
        "--ckpt_setting",
        args.run_name,
        "--start_seed",
        str(start_seed),
    ]
    if fixed_mode:
        command.extend(["--trust_verified_seed", "True"])
    print(
        f"[INFO] Isolated episode {episode_index + 1}/{state['target_episodes']} "
        f"starting at seed={start_seed} timeout={args.attempt_timeout}s",
        flush=True,
    )
    started = time.monotonic()
    with attempt_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=args.robotwin_root,
            env=os.environ.copy(),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            return_code = process.wait(timeout=args.attempt_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(process)
            return_code = process.returncode

    text = attempt_log.read_text(encoding="utf-8", errors="replace")
    expert_seeds = parse_expert_seeds(text)
    elapsed = time.monotonic() - started
    if timed_out:
        skipped_seed = expert_seeds[-1] if expert_seeds else start_seed
        if fixed_mode:
            state["next_seed"] = start_seed
        else:
            record_skipped_seeds(
                state,
                [skipped_seed],
                "isolated_episode_timeout",
                attempt_log,
            )
            state["next_seed"] = skipped_seed + 1
        state["attempt_failures"].append(
            {
                "start_seed": start_seed,
                "last_seed": skipped_seed,
                "reason": "timeout",
                "elapsed_seconds": elapsed,
                "log": str(attempt_log),
            }
        )
        update_summary(state)
        atomic_write_json(args.state_file, state)
        print(
            f"[WARN] Isolated episode timed out at seed={skipped_seed}; continuing",
            flush=True,
        )
        return

    if return_code != 0:
        state["attempt_failures"].append(
            {
                "start_seed": start_seed,
                "last_seed": expert_seeds[-1] if expert_seeds else start_seed,
                "reason": f"exit_{return_code}",
                "elapsed_seconds": elapsed,
                "log": str(attempt_log),
            }
        )
        update_summary(state)
        atomic_write_json(args.state_file, state)
        raise RuntimeError(
            f"Isolated episode process exited with status {return_code}; see {attempt_log}"
        )

    rates = parse_rates(text)
    if not rates:
        raise RuntimeError(f"No completed episode record found in {attempt_log}")
    numerator, denominator, completed_seed = rates[-1]
    if denominator != 1 or numerator not in (0, 1):
        raise RuntimeError(f"Unexpected isolated result in {attempt_log}: {rates[-1]}")

    if fixed_mode and completed_seed != start_seed:
        state["attempt_failures"].append(
            {
                "start_seed": start_seed,
                "last_seed": completed_seed,
                "reason": "fixed_seed_mismatch",
                "elapsed_seconds": elapsed,
                "log": str(attempt_log),
            }
        )
        update_summary(state)
        atomic_write_json(args.state_file, state)
        raise RuntimeError(
            f"Fixed seed {start_seed} produced episode seed {completed_seed}; "
            f"see {attempt_log}"
        )

    if not fixed_mode:
        record_skipped_seeds(
            state,
            [seed for seed in expert_seeds if seed < completed_seed],
            "expert_check_not_valid",
            attempt_log,
        )
    state["episodes"].append(
        {
            "episode_index": episode_index,
            "seed": completed_seed,
            "success": bool(numerator),
            "source": "isolated_process",
            "elapsed_seconds": elapsed,
            "log": str(attempt_log),
        }
    )
    next_index = episode_index + 1
    if fixed_mode and next_index < len(fixed_seeds):
        state["next_seed"] = fixed_seeds[next_index]
    else:
        state["next_seed"] = completed_seed + 1
    update_summary(state)
    atomic_write_json(args.state_file, state)
    print(
        f"[INFO] Episode complete seed={completed_seed} success={bool(numerator)} "
        f"aggregate={state['successes']}/{state['completed_episodes']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-python", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target-episodes", type=int, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--bootstrap-log", type=Path)
    parser.add_argument("--fixed-seeds-file", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--attempt-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.target_episodes <= 0:
        parser.error("target episodes must be positive")
    if args.start_seed < 0:
        parser.error("start seed must be non-negative")
    if args.attempt_timeout <= 0:
        parser.error("attempt timeout must be positive")
    args.fixed_seeds = None
    if args.fixed_seeds_file is not None:
        try:
            args.fixed_seeds = load_fixed_seeds(args.fixed_seeds_file, args.task_name)
        except (OSError, json.JSONDecodeError, RuntimeError) as error:
            parser.error(str(error))
        if len(args.fixed_seeds) != args.target_episodes:
            parser.error(
                "fixed seed count must equal target episodes "
                f"({len(args.fixed_seeds)} != {args.target_episodes})"
            )
        if args.bootstrap_log is not None:
            parser.error("bootstrap log cannot be combined with fixed seeds")
    return args


def main() -> None:
    args = parse_args()
    state = load_or_initialize_state(args)
    while not state["complete"]:
        run_attempt(args, state)
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
