#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RATE_RE = re.compile(r"Success rate:\s*(\d+)/(\d+).*current seed:\s*(\d+)")
TASKS = (
    "adjust_bottle",
    "click_bell",
    "hanging_mug",
    "move_stapler_pad",
    "place_a2b_left",
    "place_can_basket",
    "place_fan",
    "place_phone_stand",
    "rotate_qrcode",
    "stack_blocks_two",
)


def parse_log_seeds(path: Path) -> list[int]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n"))
    records = [tuple(map(int, match)) for match in RATE_RE.findall(text)]
    for index, (_, denominator, _) in enumerate(records, start=1):
        if denominator != index:
            raise RuntimeError(f"Non-consecutive result denominator in {path}")
    return [seed for _, _, seed in records]


def state_seeds(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise RuntimeError(f"State is incomplete at {path}")
    return [int(item["seed"]) for item in payload.get("episodes", [])]


def find_task_seeds(root: Path, run_name: str, task: str) -> tuple[list[int], Path]:
    state_path = root / "results" / "gr00t_n16" / f"{run_name}_{task}.json"
    if state_path.is_file():
        return state_seeds(state_path), state_path

    pattern = f"{run_name}_{task}_client_*.log"
    candidates = sorted((root / "logs" / "gr00t_n16").glob(pattern))
    complete = [(parse_log_seeds(path), path) for path in candidates]
    complete = [(seeds, path) for seeds, path in complete if seeds]
    if not complete:
        raise RuntimeError(f"No result state or client log found for {task}")
    seeds, path = max(complete, key=lambda item: len(item[0]))
    return seeds, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", default="robotwin_target20_formal_seed0")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = {}
    sources = {}
    for task in TASKS:
        seeds, source = find_task_seeds(args.root, args.run_name, task)
        if len(seeds) != args.episodes_per_task:
            raise RuntimeError(
                f"Expected {args.episodes_per_task} seeds for {task}, got {len(seeds)}"
            )
        if len(set(seeds)) != len(seeds):
            raise RuntimeError(f"Duplicate seeds found for {task}")
        tasks[task] = seeds
        sources[task] = str(source)

    payload = {
        "schema_version": 1,
        "source_run_name": args.run_name,
        "episodes_per_task": args.episodes_per_task,
        "created_at": datetime.now().astimezone().isoformat(),
        "tasks": tasks,
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {sum(len(seeds) for seeds in tasks.values())} seeds to {args.output}")


if __name__ == "__main__":
    main()
