#!/usr/bin/env bash
set -euo pipefail

ROOT=${MOTION_DYNVLA_ROOT:-/mnt/hdd/hesibo/motion_dynvla}
REPO="$ROOT/code/DynVLA-GR00T"
GR00T_PY="$ROOT/envs/gr00t_n16/bin/python"
LIBERO_PY="$ROOT/envs/libero/bin/python"
ZMQ_SITE="$ROOT/envs/dynvla/lib/python3.10/site-packages"

CHECKPOINT=${1:?checkpoint path is required}
GPU=${2:-2}
PORT=${3:-5562}
TASK_START=${4:-0}
TASK_END=${5:-1}
EPISODES_PER_TASK=${6:-50}
EPISODE_START=${7:-0}
RUN_NAME=${8:-libero_gr00t_eval}

if [[ ! "$CHECKPOINT" = "$ROOT"/* ]]; then
  echo "checkpoint must be under $ROOT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "checkpoint index is missing" >&2
  exit 2
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "port $PORT is already in use" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT/logs/gr00t_n16"
RESULT_DIR="$ROOT/results/gr00t_n16"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
SERVER_LOG="$LOG_DIR/${RUN_NAME}_server_${STAMP}.log"
CLIENT_LOG="$LOG_DIR/${RUN_NAME}_client_${STAMP}.log"
RESULT_PATH=${DYNVLA_RESULT_PATH:-"$RESULT_DIR/${RUN_NAME}_${STAMP}.json"}

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$REPO"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU" "$GR00T_PY" gr00t/eval/run_gr00t_server.py \
  --model-path "$CHECKPOINT" \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper \
  --seed 7 \
  --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

export LIBERO_CONFIG_PATH="$ROOT/third_party/LIBERO/libero"
export PYTHONPATH="$ROOT/third_party/LIBERO"
export DYNVLA_ZMQ_SITE_PACKAGES="$ZMQ_SITE"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "policy server exited before becoming ready" >&2
    tail -n 80 "$SERVER_LOG" >&2
    exit 1
  fi
  if "$LIBERO_PY" scripts/eval/libero_zmq_eval.py \
      --host 127.0.0.1 --port "$PORT" --timeout-ms 2000 \
      --result-path "$RESULT_PATH" --ping-only >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  echo "policy server readiness timed out" >&2
  tail -n 80 "$SERVER_LOG" >&2
  exit 1
fi

"$LIBERO_PY" scripts/eval/libero_zmq_eval.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --task-suite libero_10 \
  --task-start "$TASK_START" \
  --task-end "$TASK_END" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --episode-start "$EPISODE_START" \
  --n-action-steps 8 \
  --max-episode-steps 720 \
  --result-path "$RESULT_PATH" \
  --checkpoint "$CHECKPOINT" \
  --resume 2>&1 | tee "$CLIENT_LOG"

echo "server_log=$SERVER_LOG"
echo "client_log=$CLIENT_LOG"
echo "result_path=$RESULT_PATH"
