"""Run LIBERO rollouts against the GR00T ZeroMQ policy server.

This client intentionally reuses the existing lightweight LIBERO environment.
It follows the observation and action conventions used by the official GR00T
LIBERO wrapper without creating another simulator environment.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import msgpack
import numpy as np


def _import_zmq():
    try:
        import zmq

        return zmq
    except ModuleNotFoundError:
        fallback = os.environ.get("DYNVLA_ZMQ_SITE_PACKAGES")
        if not fallback:
            raise
        sys.path.append(fallback)
        import zmq

        return zmq


zmq = _import_zmq()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


LOGGER = logging.getLogger("libero_zmq_eval")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


class MsgSerializer:
    """Serializer compatible with gr00t.policy.server_client.MsgSerializer."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        raise TypeError(f"Unsupported message type {type(obj)!r}")

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


class PolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.socket = None
        self._replace_socket()

    def _replace_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self.socket.recv())
        except zmq.error.ZMQError:
            self._replace_socket()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Policy server error {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            response = self.call("ping")
        except zmq.error.ZMQError:
            return False
        return isinstance(response, dict) and response.get("status") == "ok"

    def reset(self) -> None:
        self.call("reset", {"options": None})

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        response = self.call(
            "get_action",
            {"observation": observation, "options": None},
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise RuntimeError(f"Unexpected policy response {type(response)!r}")
        actions = response[0]
        if not isinstance(actions, dict):
            raise RuntimeError(f"Unexpected action payload {type(actions)!r}")
        return actions

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.context.term()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return quat[:3] * (2.0 * math.acos(float(quat[3])) / denominator)


def build_observation(obs: dict[str, Any], instruction: str) -> dict[str, Any]:
    xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    def scalar(value: float) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape(1, 1, 1)

    return {
        "video.primary_image": image.reshape(1, 1, *image.shape),
        "video.wrist_image": wrist.reshape(1, 1, *wrist.shape),
        "state.x": scalar(xyz[0]),
        "state.y": scalar(xyz[1]),
        "state.z": scalar(xyz[2]),
        "state.roll": scalar(rpy[0]),
        "state.pitch": scalar(rpy[1]),
        "state.yaw": scalar(rpy[2]),
        # The converted training data keeps the second finger joint as the
        # scalar gripper state.  Its checkpoint statistics are one dimensional.
        "state.gripper": gripper[-1:].reshape(1, 1, 1),
        "annotation.human.action.task_description": [str(instruction)],
    }


def action_horizon(actions: dict[str, np.ndarray]) -> int:
    required = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    missing = [key for key in required if key not in actions]
    if missing:
        raise RuntimeError(f"Missing action keys {missing}")
    horizons = {np.asarray(actions[key]).shape[1] for key in required}
    if len(horizons) != 1:
        raise RuntimeError(f"Inconsistent action horizons {sorted(horizons)}")
    return horizons.pop()


def libero_action(actions: dict[str, np.ndarray], index: int) -> np.ndarray:
    values = []
    for key in [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
    ]:
        value = np.asarray(actions[key], dtype=np.float32)[0, index].reshape(-1)
        if value.size != 1:
            raise RuntimeError(f"Unexpected shape for {key} {value.shape}")
        values.append(float(value[0]))
    gripper = np.asarray(actions["action.gripper"], dtype=np.float32)[0, index].reshape(-1)
    if gripper.size != 1:
        raise RuntimeError(f"Unexpected gripper shape {gripper.shape}")
    environment_gripper = 1.0 - 2.0 * (float(gripper[0]) > 0.5)
    values.append(environment_gripper)
    action = np.asarray(values, dtype=np.float32)
    if not np.isfinite(action).all():
        raise RuntimeError(f"Nonfinite action at chunk index {index}")
    return action


def create_env(task: Any, seed: int) -> OffScreenRenderEnv:
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_result(
    args: argparse.Namespace,
    suite: Any,
    task_end: int,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task_id in range(args.task_start, task_end):
        task = suite.get_task(task_id)
        task_records = [record for record in records if record["task_id"] == task_id]
        successes = sum(int(record["success"]) for record in task_records)
        episodes = len(task_records)
        per_task[str(task_id)] = {
            "name": task.name,
            "description": task.language,
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes if episodes else None,
        }

    total_successes = sum(int(record["success"]) for record in records)
    total_episodes = len(records)
    expected_episodes = (task_end - args.task_start) * args.episodes_per_task
    return {
        "schema_version": 1,
        "complete": total_episodes == expected_episodes,
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "task_start": args.task_start,
        "task_end": task_end,
        "episode_start": args.episode_start,
        "episodes_per_task": args.episodes_per_task,
        "successes": total_successes,
        "episodes": total_episodes,
        "expected_episodes": expected_episodes,
        "success_rate": total_successes / total_episodes if total_episodes else None,
        "n_action_steps": args.n_action_steps,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "elapsed_seconds": elapsed_seconds,
        "per_task": per_task,
        "episode_records": records,
    }


def load_resume_records(
    args: argparse.Namespace,
    task_end: int,
) -> tuple[list[dict[str, Any]], float]:
    if not args.resume or not args.result_path.is_file():
        return [], 0.0
    previous = json.loads(args.result_path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "task_start": args.task_start,
        "task_end": task_end,
        "episode_start": args.episode_start,
        "episodes_per_task": args.episodes_per_task,
        "n_action_steps": args.n_action_steps,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
    }
    mismatches = {
        key: (previous.get(key), value)
        for key, value in expected.items()
        if previous.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume configuration mismatch {mismatches}")
    records = previous.get("episode_records", [])
    if not isinstance(records, list):
        raise ValueError("Resume file episode_records must be a list")
    LOGGER.info("resuming completed_episodes=%d", len(records))
    return records, float(previous.get("elapsed_seconds", 0.0))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task_end = min(args.task_end, suite.n_tasks) if args.task_end >= 0 else suite.n_tasks
    if args.task_start < 0 or args.task_start >= task_end:
        raise ValueError(f"Invalid task interval {args.task_start} to {task_end}")
    if args.episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")

    records, previous_elapsed = load_resume_records(args, task_end)
    completed = {(record["task_id"], record["episode_index"]) for record in records}

    client = PolicyClient(args.host, args.port, args.timeout_ms)
    if not client.ping():
        raise RuntimeError("Policy server did not answer ping")

    started = time.time()
    try:
        for task_id in range(args.task_start, task_end):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            pending_episodes = [
                args.episode_start + local_episode
                for local_episode in range(args.episodes_per_task)
                if (task_id, args.episode_start + local_episode) not in completed
            ]
            if not pending_episodes:
                LOGGER.info("task_id=%d already complete", task_id)
                continue
            env = create_env(task, args.seed)
            try:
                for episode_index in pending_episodes:
                    if episode_index >= len(initial_states):
                        raise ValueError(
                            f"Initial state index {episode_index} exceeds {len(initial_states)} states"
                        )
                    client.reset()
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_index])
                    env_steps = 0
                    for _ in range(args.wait_steps):
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        env_steps += 1
                        if done:
                            break

                    success = bool(done or env.check_success())
                    policy_calls = 0
                    while not success and env_steps < args.max_episode_steps:
                        observation = build_observation(obs, task.language)
                        actions = client.get_action(observation)
                        policy_calls += 1
                        execute = min(args.n_action_steps, action_horizon(actions))
                        for chunk_index in range(execute):
                            obs, _, done, _ = env.step(
                                libero_action(actions, chunk_index).tolist()
                            )
                            env_steps += 1
                            success = bool(done or env.check_success())
                            if success or env_steps >= args.max_episode_steps:
                                break

                    records.append(
                        {
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "success": success,
                            "env_steps": env_steps,
                            "policy_calls": policy_calls,
                        }
                    )
                    completed.add((task_id, episode_index))
                    LOGGER.info(
                        "task_id=%d episode=%d success=%s env_steps=%d policy_calls=%d",
                        task_id,
                        episode_index,
                        success,
                        env_steps,
                        policy_calls,
                    )
                    partial = build_result(
                        args,
                        suite,
                        task_end,
                        records,
                        previous_elapsed + time.time() - started,
                    )
                    write_json_atomic(args.result_path, partial)
            finally:
                env.close()
    finally:
        client.close()

    return build_result(
        args,
        suite,
        task_end,
        records,
        previous_elapsed + time.time() - started,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=180000)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=-1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ping-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.ping_only:
        client = PolicyClient(args.host, args.port, args.timeout_ms)
        try:
            return 0 if client.ping() else 1
        finally:
            client.close()
    result = evaluate(args)
    write_json_atomic(args.result_path, result)
    LOGGER.info("result_path=%s", args.result_path)
    LOGGER.info("success_rate=%.6f", result["success_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
