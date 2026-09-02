#!/usr/bin/env bash
set -euo pipefail

ROOT=${MOTION_DYNVLA_ROOT:-/mnt/hdd/hesibo/motion_dynvla}
REPO="$ROOT/code/DynVLA-GR00T"
ROBOTWIN="$ROOT/third_party/RoboTwin"
GR00T_PY="$ROOT/envs/gr00t_n16/bin/python"
ROBOTWIN_PY="$ROOT/envs/robotwin/bin/python"
ZMQ_SITE="$ROOT/envs/dynvla/lib/python3.10/site-packages"

CHECKPOINT=${1:?checkpoint path is required}
GPU=${2:-3}
PORT=${3:-5694}
TASK=${4:-adjust_bottle}
EPISODES=${5:-1}
SEED=${6:-0}
RUN_NAME=${7:-robotwin_gr00t_eval}
START_SEED=${8:-}
BOOTSTRAP_LOG=${9:-}
FIXED_SEEDS_FILE=${10:-}

if [[ ! "$CHECKPOINT" = "$ROOT"/* ]]; then
  echo "checkpoint must be under $ROOT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "checkpoint index is missing" >&2
  exit 2
fi
if [[ ! "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "episodes must be a positive integer" >&2
  exit 2
fi
if [[ -n "$START_SEED" ]] && [[ ! "$START_SEED" =~ ^[0-9]+$ ]]; then
  echo "start seed must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "$BOOTSTRAP_LOG" ]]; then
  if [[ ! "$BOOTSTRAP_LOG" = "$ROOT"/* ]] || [[ ! -f "$BOOTSTRAP_LOG" ]]; then
    echo "bootstrap log must be an existing file under $ROOT" >&2
    exit 2
  fi
fi
if [[ -n "$FIXED_SEEDS_FILE" ]]; then
  if [[ ! "$FIXED_SEEDS_FILE" = "$ROOT"/* ]] || [[ ! -f "$FIXED_SEEDS_FILE" ]]; then
    echo "fixed seeds file must be an existing file under $ROOT" >&2
    exit 2
  fi
fi
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "port $PORT is already in use" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT/logs/gr00t_n16"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/${RUN_NAME}_${TASK}_server_${STAMP}.log"
CLIENT_LOG="$LOG_DIR/${RUN_NAME}_${TASK}_client_${STAMP}.log"

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
  --embodiment-tag NEW_EMBODIMENT \
  --seed 7 \
  --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "policy server exited before becoming ready" >&2
    tail -n 100 "$SERVER_LOG" >&2
    exit 1
  fi
  if "$GR00T_PY" -c "from gr00t.policy.server_client import PolicyClient; raise SystemExit(0 if PolicyClient(host='127.0.0.1', port=$PORT, timeout_ms=2000, strict=False).ping() else 1)" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != 1 ]]; then
  echo "policy server readiness timed out" >&2
  tail -n 100 "$SERVER_LOG" >&2
  exit 1
fi

cd "$ROBOTWIN"
export PYTHONPATH="$REPO:$ROBOTWIN"
export DYNVLA_ZMQ_SITE_PACKAGES="$ZMQ_SITE"
export ROBOTWIN_EXPERT_CHECK_TIMEOUT=${ROBOTWIN_EXPERT_CHECK_TIMEOUT:-300}
export ROBOTWIN_EPISODE_SETUP_TIMEOUT=${ROBOTWIN_EPISODE_SETUP_TIMEOUT:-300}
export ROBOTWIN_EPISODE_PROCESS_TIMEOUT=${ROBOTWIN_EPISODE_PROCESS_TIMEOUT:-900}
export CUDA_VISIBLE_DEVICES="$GPU"
RESUME_ARGS=(
  --robotwin-python "$ROBOTWIN_PY"
  --robotwin-root "$ROBOTWIN"
  --config "$REPO/examples/RoboTwin/eval_gr00t.yaml"
  --checkpoint "$CHECKPOINT"
  --task-name "$TASK"
  --run-name "$RUN_NAME"
  --target-episodes "$EPISODES"
  --state-file "$ROOT/results/gr00t_n16/${RUN_NAME}_${TASK}.json"
  --log-dir "$LOG_DIR"
  --start-seed "${START_SEED:-100000}"
  --host 127.0.0.1
  --port "$PORT"
  --attempt-timeout "$ROBOTWIN_EPISODE_PROCESS_TIMEOUT"
)
if [[ -n "$BOOTSTRAP_LOG" ]]; then
  RESUME_ARGS+=(--bootstrap-log "$BOOTSTRAP_LOG")
fi
if [[ -n "$FIXED_SEEDS_FILE" ]]; then
  RESUME_ARGS+=(--fixed-seeds-file "$FIXED_SEEDS_FILE")
fi
"$GR00T_PY" "$REPO/scripts/eval/robotwin_resumable_eval.py" \
  "${RESUME_ARGS[@]}" 2>&1 | tee "$CLIENT_LOG"

echo "server_log=$SERVER_LOG"
echo "client_log=$CLIENT_LOG"
