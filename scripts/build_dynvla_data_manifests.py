"""Build target-disjoint source manifests and fixed 20-shot target manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


ROBOTWIN_CLEAN50_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]

ROBOTWIN_TARGET10_TASKS = {
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
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def first_shots_per_task(dataset_path: Path, shots: int) -> list[int]:
    episodes = read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    grouped = defaultdict(list)
    for episode in episodes:
        tasks = episode.get("tasks", [])
        if len(tasks) != 1:
            raise ValueError(f"expected one task per episode in {dataset_path}")
        grouped[tasks[0]].append(int(episode["episode_index"]))

    selected = []
    for task, indices in sorted(grouped.items()):
        indices = sorted(indices)
        if len(indices) < shots:
            raise ValueError(f"task {task} has only {len(indices)} episodes")
        selected.extend(indices[:shots])
    return sorted(selected)


def dataset_entry(
    path: Path,
    embodiment_tag: str,
    episode_indices: list[int] | None = None,
) -> dict:
    if not (path / "meta" / "info.json").is_file():
        raise FileNotFoundError(path / "meta" / "info.json")
    if episode_indices is not None:
        available = {
            int(episode["episode_index"])
            for episode in read_jsonl(path / "meta" / "episodes.jsonl")
        }
        missing = sorted(set(episode_indices) - available)
        if missing:
            raise ValueError(f"episodes not found in {path}: {missing}")
    entry = {
        "dataset_path": str(path),
        "embodiment_tag": embodiment_tag,
        "mix_ratio": 1.0,
    }
    if episode_indices is not None:
        entry["episode_indices"] = episode_indices
    return entry


def write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} with {len(entries)} dataset entries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/mnt/hdd/hesibo/motion_dynvla"),
    )
    parser.add_argument("--shots", type=int, default=20)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = root / "code" / "DynVLA-GR00T" / "configs" / "data_manifests"
    libero_root = root / "data" / "libero"
    robotwin_root = root / "data" / "robotwin" / "Clean"

    libero_source_names = [
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
    ]
    libero_source = [
        dataset_entry(libero_root / name, "libero_panda") for name in libero_source_names
    ]
    libero_target_path = libero_root / "libero_10_no_noops_1.0.0_lerobot"
    libero_target = [
        dataset_entry(
            libero_target_path,
            "libero_panda",
            first_shots_per_task(libero_target_path, args.shots),
        )
    ]

    robotwin_source_tasks = [
        task for task in ROBOTWIN_CLEAN50_TASKS if task not in ROBOTWIN_TARGET10_TASKS
    ]
    if len(robotwin_source_tasks) != 40:
        raise RuntimeError(f"expected 40 RoboTwin source tasks, got {len(robotwin_source_tasks)}")
    robotwin_source = [
        dataset_entry(robotwin_root / task, "new_embodiment")
        for task in robotwin_source_tasks
    ]
    robotwin_target = [
        dataset_entry(robotwin_root / task, "new_embodiment", list(range(args.shots)))
        for task in sorted(ROBOTWIN_TARGET10_TASKS)
    ]

    write_manifest(output / "libero_source30.json", libero_source)
    write_manifest(output / f"libero_target10_{args.shots}shot.json", libero_target)
    write_manifest(output / "robotwin_source40.json", robotwin_source)
    write_manifest(output / f"robotwin_target10_{args.shots}shot.json", robotwin_target)

    libero_episode_count = len(libero_target[0]["episode_indices"])
    robotwin_episode_count = sum(len(entry["episode_indices"]) for entry in robotwin_target)
    if libero_episode_count != 10 * args.shots:
        raise RuntimeError(f"expected {10 * args.shots} LIBERO episodes, got {libero_episode_count}")
    if robotwin_episode_count != 10 * args.shots:
        raise RuntimeError(
            f"expected {10 * args.shots} RoboTwin episodes, got {robotwin_episode_count}"
        )
    print(f"libero_target_episodes={libero_episode_count}")
    print(f"robotwin_target_episodes={robotwin_episode_count}")


if __name__ == "__main__":
    main()
