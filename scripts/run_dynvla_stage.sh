#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: run_dynvla_stage.sh <libero|robotwin> <source_warmup|source_joint|target20> <base_model_path>"
}

if [ "$#" -ne 3 ]; then
    usage
    exit 2
fi

BENCHMARK="$1"
STAGE="$2"
BASE_MODEL_PATH="$3"
PROJECT_ROOT="${MOTION_DYNVLA_ROOT:-/mnt/hdd/hesibo/motion_dynvla}"
CODE_ROOT="$PROJECT_ROOT/code/DynVLA-GR00T"
PYTHON="$PROJECT_ROOT/envs/gr00t_n16/bin/python"
NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MASTER_PORT="${MASTER_PORT:-29510}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs/gr00t_n16}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/gr00t_n16}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_MODEL_AT_END="${SAVE_MODEL_AT_END:-true}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
DYNVLA_BANK_MODE="${DYNVLA_BANK_MODE:-trainable}"
DYNVLA_BANK_SEED="${DYNVLA_BANK_SEED:-7}"

case "$DYNVLA_BANK_MODE" in
    trainable|frozen|random_init|disabled)
        ;;
    *)
        echo "Unsupported DYNVLA_BANK_MODE=$DYNVLA_BANK_MODE"
        exit 2
        ;;
esac

case "$BENCHMARK" in
    libero)
        SOURCE_MANIFEST="$CODE_ROOT/configs/data_manifests/libero_source30.json"
        TARGET_MANIFEST="$CODE_ROOT/configs/data_manifests/libero_target10_20shot.json"
        DATASET_PATH="$PROJECT_ROOT/data/libero/libero_10_no_noops_1.0.0_lerobot"
        EMBODIMENT_TAG="LIBERO_PANDA"
        MODALITY_CONFIG="$CODE_ROOT/examples/LIBERO/dynvla_libero_config.py"
        ;;
    robotwin)
        SOURCE_MANIFEST="$CODE_ROOT/configs/data_manifests/robotwin_source40.json"
        TARGET_MANIFEST="$CODE_ROOT/configs/data_manifests/robotwin_target10_20shot.json"
        DATASET_PATH="$PROJECT_ROOT/data/robotwin/Clean/place_fan"
        EMBODIMENT_TAG="NEW_EMBODIMENT"
        MODALITY_CONFIG="$CODE_ROOT/examples/RoboTwin/robotwin_config.py"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [ "$STAGE" != "target20" ] && [ "$DYNVLA_BANK_MODE" != "trainable" ]; then
    echo "DynVLA bank ablations are only valid for target20"
    exit 2
fi

case "$STAGE" in
    source_warmup)
        DATASET_MANIFEST="$SOURCE_MANIFEST"
        MAX_STEPS="${MAX_STEPS:-2000}"
        LEARNING_RATE="${LEARNING_RATE:-1e-4}"
        TUNE_PROJECTOR="false"
        TUNE_DIFFUSION="false"
        TUNE_VLLN="false"
        ;;
    source_joint)
        DATASET_MANIFEST="$SOURCE_MANIFEST"
        MAX_STEPS="${MAX_STEPS:-28000}"
        LEARNING_RATE="${LEARNING_RATE:-2e-5}"
        TUNE_PROJECTOR="true"
        TUNE_DIFFUSION="true"
        TUNE_VLLN="true"
        ;;
    target20)
        DATASET_MANIFEST="$TARGET_MANIFEST"
        MAX_STEPS="${MAX_STEPS:-30000}"
        LEARNING_RATE="${LEARNING_RATE:-1e-4}"
        TUNE_PROJECTOR="true"
        TUNE_DIFFUSION="true"
        TUNE_VLLN="true"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [ "$DYNVLA_BANK_MODE" = "trainable" ]; then
    DEFAULT_RUN_NAME="dynvla_gr00t_n16_${BENCHMARK}_${STAGE}"
else
    DEFAULT_RUN_NAME="dynvla_gr00t_n16_${BENCHMARK}_${STAGE}_ablation_bank_${DYNVLA_BANK_MODE}"
fi
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
OUTPUT_DIR="$RUNS_ROOT/$RUN_NAME"
LOG_PATH="$LOG_ROOT/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RUNS_ROOT" "$LOG_ROOT"

export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMAND=(
    "$CODE_ROOT/gr00t/experiment/launch_finetune.py"
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --dataset-manifest-path "$DATASET_MANIFEST"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG"
    --video-backend pyav
    --output-dir "$RUNS_ROOT"
    --experiment-name "$RUN_NAME"
    --num-gpus "$NUM_GPUS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --max-steps "$MAX_STEPS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay 1e-5
    --warmup-steps 500
    --warmup-ratio 0.0
    --max-grad-norm 10.0
    --adam-beta1 0.95
    --adam-beta2 0.999
    --save-strategy "$SAVE_STRATEGY"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --shard-size "$SHARD_SIZE"
    --episode-sampling-rate 0.1
    --num-shards-per-epoch 100000
    --enable-dynvla
    --tune-dynvla
    --dynvla-bank-mode "$DYNVLA_BANK_MODE"
    --dynvla-bank-seed "$DYNVLA_BANK_SEED"
    --tune-top-llm-layers 0
)

if [ "$TUNE_PROJECTOR" = "true" ]; then
    COMMAND+=(--tune-projector)
else
    COMMAND+=(--no-tune-projector)
fi
if [ "$TUNE_DIFFUSION" = "true" ]; then
    COMMAND+=(--tune-diffusion-model)
else
    COMMAND+=(--no-tune-diffusion-model)
fi
if [ "$TUNE_VLLN" = "true" ]; then
    COMMAND+=(--tune-vlln)
else
    COMMAND+=(--no-tune-vlln)
fi
if [ "$SAVE_MODEL_AT_END" = "true" ]; then
    COMMAND+=(--save-model-at-end)
else
    COMMAND+=(--no-save-model-at-end)
fi

cd "$CODE_ROOT"
echo "run_name=$RUN_NAME" | tee -a "$LOG_PATH"
echo "base_model_path=$BASE_MODEL_PATH" | tee -a "$LOG_PATH"
echo "dataset_manifest=$DATASET_MANIFEST" | tee -a "$LOG_PATH"
echo "output_dir=$OUTPUT_DIR" | tee -a "$LOG_PATH"
echo "num_gpus=$NUM_GPUS" | tee -a "$LOG_PATH"
echo "global_batch_size=$GLOBAL_BATCH_SIZE" | tee -a "$LOG_PATH"
echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS" | tee -a "$LOG_PATH"
echo "effective_batch_size=$((GLOBAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))" | tee -a "$LOG_PATH"
echo "max_steps=$MAX_STEPS" | tee -a "$LOG_PATH"
echo "learning_rate=$LEARNING_RATE" | tee -a "$LOG_PATH"
echo "save_model_at_end=$SAVE_MODEL_AT_END" | tee -a "$LOG_PATH"
echo "save_strategy=$SAVE_STRATEGY" | tee -a "$LOG_PATH"
echo "save_steps=$SAVE_STEPS" | tee -a "$LOG_PATH"
echo "save_total_limit=$SAVE_TOTAL_LIMIT" | tee -a "$LOG_PATH"
echo "dynvla_bank_mode=$DYNVLA_BANK_MODE" | tee -a "$LOG_PATH"
echo "dynvla_bank_seed=$DYNVLA_BANK_SEED" | tee -a "$LOG_PATH"

if [ "$NUM_GPUS" -eq 1 ]; then
    "$PYTHON" "${COMMAND[@]}" 2>&1 | tee -a "$LOG_PATH"
else
    "$PROJECT_ROOT/envs/gr00t_n16/bin/torchrun" \
        --nproc-per-node "$NUM_GPUS" \
        --master-port "$MASTER_PORT" \
        "${COMMAND[@]}" 2>&1 | tee -a "$LOG_PATH"
fi

if [ "$SAVE_MODEL_AT_END" = "true" ]; then
    FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoint-$MAX_STEPS"
    FINAL_LINK="$RUNS_ROOT/${RUN_NAME}_final"
    if [ ! -s "$FINAL_CHECKPOINT/trainer_state.json" ]; then
        echo "final checkpoint is incomplete at $FINAL_CHECKPOINT" | tee -a "$LOG_PATH"
        exit 1
    fi
    ln -sfn "$FINAL_CHECKPOINT" "$FINAL_LINK"
    echo "final_checkpoint=$FINAL_CHECKPOINT" | tee -a "$LOG_PATH"
    echo "final_link=$FINAL_LINK" | tee -a "$LOG_PATH"
fi
