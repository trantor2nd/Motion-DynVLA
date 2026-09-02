"""RoboTwin evaluation adapter for the official GR00T policy server."""

from __future__ import annotations

import io
import os
import sys
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


class MsgSerializer:
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

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call(
            "get_action",
            {"observation": observation, "options": None},
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise RuntimeError(f"Unexpected policy response {type(response)!r}")
        return response[0], response[1]


ACTION_KEYS = ["left_joints", "left_gripper", "right_joints", "right_gripper"]
MODEL_ACTION_HORIZON = 16


class ModelClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5694,
        execution_horizon: int = 8,
        timeout_ms: int = 120000,
    ) -> None:
        self.client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms)
        if not self.client.ping():
            raise RuntimeError(f"GR00T policy server is not ready on {host}:{port}")
        if not 1 <= execution_horizon <= MODEL_ACTION_HORIZON:
            raise ValueError(
                f"execution_horizon must be in [1, {MODEL_ACTION_HORIZON}], got {execution_horizon}"
            )
        self.execution_horizon = int(execution_horizon)
        self.actions = None
        self.instruction = None

    def reset(self, instruction: str = "") -> None:
        self.client.reset()
        self.actions = None
        self.instruction = instruction

    @staticmethod
    def _observation(images: list[np.ndarray], state: np.ndarray, instruction: str) -> dict:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (14,):
            raise ValueError(f"expected RoboTwin state shape (14,), got {state.shape}")
        return {
            "video": {
                "cam_high": np.asarray(images[0], dtype=np.uint8)[None, None],
                "cam_left_wrist": np.asarray(images[1], dtype=np.uint8)[None, None],
                "cam_right_wrist": np.asarray(images[2], dtype=np.uint8)[None, None],
            },
            "state": {
                "left_joints": state[None, None, :6],
                "left_gripper": state[None, None, 6:7],
                "right_joints": state[None, None, 7:13],
                "right_gripper": state[None, None, 13:14],
            },
            "language": {
                "annotation.human.action.task_description": [[instruction]],
            },
        }

    def step(
        self,
        images: list[np.ndarray],
        state: np.ndarray,
        instruction: str,
        step: int,
    ) -> np.ndarray:
        if instruction != self.instruction:
            self.reset(instruction)
        if self.actions is None or step % self.execution_horizon == 0:
            observation = self._observation(images, state, instruction)
            action_dict, _ = self.client.get_action(observation)
            self.actions = np.concatenate(
                [action_dict[key][0] for key in ACTION_KEYS], axis=-1
            )
            if self.actions.shape != (MODEL_ACTION_HORIZON, 14):
                raise RuntimeError(f"Unexpected RoboTwin action shape {self.actions.shape}")
            if not np.isfinite(self.actions).all():
                raise RuntimeError("Nonfinite RoboTwin policy action")
        action_index = step % self.execution_horizon
        return self.actions[action_index]


def get_model(usr_args):
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 5694)),
        execution_horizon=int(usr_args.get("execution_horizon", 8)),
        timeout_ms=int(usr_args.get("timeout_ms", 120000)),
    )


def reset_model(model):
    model.reset()


def eval(task_env, model, observation):
    instruction = str(task_env.get_instruction())
    images = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
    ]
    state = observation["joint_action"]["vector"]
    action = model.step(images, state, instruction, task_env.take_action_cnt)
    task_env.take_action(action)
