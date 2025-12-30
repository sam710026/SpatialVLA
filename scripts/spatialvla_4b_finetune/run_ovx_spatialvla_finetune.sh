#!/usr/bin/env bash
set -euo pipefail
set -x

############################################
# SpatialVLA LoRA finetune (single machine)
# - Dataset root: /mnt/nfs/eson/dataset
# - Checkpoints:  /mnt/nfs/eson/ckp
# - Outputs:      /mnt/nfs/eson/output
#
# Usage examples:
#   chmod +x run_local_spatialvla_finetune.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_local_spatialvla_finetune.sh
#
#   # Override common knobs:
#   PER_DEVICE_BATCH_SIZE=8 LR=3e-4 EPOCH=10 MIXTURE=kuka_latent \
#     CUDA_VISIBLE_DEVICES=0,1 ./run_local_spatialvla_finetune.sh
############################################

# ---- Paths (edit PROJECT_DIR once, others are already set) ----
PROJECT_DIR=${PROJECT_DIR:-"$HOME/SpatialVLA"}          # <-- change if your repo is elsewhere
DATA_ROOT=${DATA_ROOT:-"/mnt/nfs/eson/dataset"}
CKPT_ROOT=${CKPT_ROOT:-"/mnt/nfs/eson/ckp"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/mnt/nfs/eson/output"}

# ---- Training knobs (override via env) ----
MIXTURE=${MIXTURE:-"kuka_latent"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${CKPT_ROOT}"}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-""}   # if empty -> use GPUS * PER_DEVICE_BATCH_SIZE

LR=${LR:-"5e-4"}
WEIGHT_DECAY=${WEIGHT_DECAY:-"0.0"}
EPOCH=${EPOCH:-50}
SAVE_STEPS=${SAVE_STEPS:-10000}
NUM_WORKERS=${NUM_WORKERS:-1}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-8192}

# LoRA knobs
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGET=${LORA_TARGET:-"linear"}

# Misc
FLASH_ATTN=${FLASH_ATTN:-False}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-"scripts/zero1.json"}   # set empty to disable: DEEPSPEED_CFG=
REPORT_TO=${REPORT_TO:-"tensorboard"}

# ---- Environment / launcher ----
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}"
export TF_CPP_MIN_LOG_LEVEL=3

# ---- Move to repo ----
cd "${PROJECT_DIR}"

# ---- Figure out how many GPUs to use ----
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
elif command -v nvidia-smi &>/dev/null; then
  GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
else
  GPUS=1
fi
GPUS=${GPUS:-1}

# ---- Batch / grad accumulation ----
if [[ -z "${GLOBAL_BATCH_SIZE}" ]]; then
  GLOBAL_BATCH_SIZE=$((GPUS * PER_DEVICE_BATCH_SIZE))
fi

den=$((GPUS * PER_DEVICE_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % den != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by (GPUS*PER_DEVICE_BATCH_SIZE) (${den})." >&2
  exit 2
fi
GRAD_ACC=$((GLOBAL_BATCH_SIZE / den))
if (( GRAD_ACC < 1 )); then GRAD_ACC=1; fi

echo "Training Configuration:"
echo "  PROJECT_DIR: ${PROJECT_DIR}"
echo "  DATA_ROOT:   ${DATA_ROOT}"
echo "  CKPT_ROOT:   ${CKPT_ROOT}"
echo "  OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "  MODEL:       ${MODEL_NAME_OR_PATH}"
echo "  MIXTURE:     ${MIXTURE}"
echo "  GPUS:        ${GPUS} (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}')"
echo "  PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "  GLOBAL_BATCH_SIZE:     ${GLOBAL_BATCH_SIZE}"
echo "  GRAD_ACC:              ${GRAD_ACC}"
echo "  EPOCH:      ${EPOCH}"
echo "  LR:         ${LR}"
echo "  SAVE_STEPS: ${SAVE_STEPS}"
echo "  NUM_WORKERS:${NUM_WORKERS}"

# ---- Output dir ----
cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")
model_base=$(basename "${MODEL_NAME_OR_PATH}")
note="${model_base}_lr${LR}_gbs${GLOBAL_BATCH_SIZE}_gpu${GPUS}_r${LORA_R}_a${LORA_ALPHA}_ep${EPOCH}_${LORA_TARGET}"
OUTPUT_DIR="${OUTPUT_ROOT}/spatialvla_4b_finetune/${date_dir}/${cur_time}_${MIXTURE}_${note}"
mkdir -p "${OUTPUT_DIR}"

# Save a copy of this launcher for reproducibility
cp -f "$(realpath "$0")" "${OUTPUT_DIR}/"

# ---- Torchrun args (single machine) ----
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--standalone --nnodes=1 --nproc-per-node=${GPUS} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"}

# ---- Build command ----
cmd=(torchrun ${TORCH_RUN_ARGS}
  train/spatialvla_finetune.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --lora "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_target "${LORA_TARGET}"
  --ignore_data_skip True
  --data_root_dir "${DATA_ROOT}"
  --data_mix "${MIXTURE}"
  --use_latent_action True
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
  --obs_backward_steps 0
  --obs_backward_delta 1
  --action_forward_steps 3
  --flash_attn "${FLASH_ATTN}"
  --output_dir "${OUTPUT_DIR}"
  --overwrite_output_dir False
  --freeze_vision_tower False
  --dataloader_num_workers "${NUM_WORKERS}"
  --bf16 True
  --tf32 True
  --num_train_epochs "${EPOCH}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit 3
  --learning_rate "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio 0.005
  --lr_scheduler_type linear
  --logging_steps 500
  --do_train True
  --grad_checkpoint True
  --report_to "${REPORT_TO}"
  --log_level warning
)

if [[ -n "${DEEPSPEED_CFG}" ]]; then
  cmd+=(--deepspeed "${DEEPSPEED_CFG}")
fi

# ---- Run ----
"${cmd[@]}"
