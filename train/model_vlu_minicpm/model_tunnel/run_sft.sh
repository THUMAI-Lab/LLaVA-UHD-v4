#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." || exit 1; pwd)
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo ">>> PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} <<<"

if [ -z "${PREFIX:-}" ]; then
  echo "错误: 必须通过环境变量 PREFIX 指定实验名称，例如: PREFIX=my_exp bash model_vlu_minicpm/model_tunnel/run_sft.sh"
  exit 1
fi


export REPRODUCIBLE=${REPRODUCIBLE:-true}

TUNNEL_USER=${TUNNEL_USER:-user}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_SLICE_NUMS=${MAX_SLICE_NUMS:-9}
STAGE4_MAX_LEN=${STAGE4_MAX_LEN:-8192}
STAGE4_TOTAL_MAX_LEN=${STAGE4_TOTAL_MAX_LEN:-8192}
STAGE4_GRAD_ACCUM=${STAGE4_GRAD_ACCUM:-${GRAD_ACCUM}}
LR_SCALE=${LR_SCALE:-1}
_lr() { awk "BEGIN{printf \"%.2e\", $1 * ${LR_SCALE}}"; }

LLM_PATH=${LLM_PATH:-/path/to/models/MiniCPM-V-Qwen3/llm_8b_init}
VOCABS_PATH=${VOCABS_PATH:-/path/to/models/MiniCPM-V-Qwen3/llm_8b_init}
MODEL_EXTRA_ARGS=""

STAGE4_FILE=${STAGE4_FILE:-/path/to/datasets/stage_4.json}

if [ "${BACKUP:-}" = "False" ] || [ "${BACKUP:-}" = "false" ] || [ "${BACKUP:-}" = "0" ]; then
  BASE_CHECKPOINT_DIR=/user/${TUNNEL_USER}/checkpoints/sft_tunnel/
  TENSORBOARD_DIR=/user/${TUNNEL_USER}/tensorboard/sft_tunnel/
else
  BASE_CHECKPOINT_DIR=/backup/user/${TUNNEL_USER}/checkpoints/sft_tunnel/
  TENSORBOARD_DIR=/backup/user/${TUNNEL_USER}/tensorboard/sft_tunnel/
fi

_BACKUP_DIR=/backup/user/${TUNNEL_USER}/checkpoints/sft_tunnel/
_LOCAL_DIR=/user/${TUNNEL_USER}/checkpoints/sft_tunnel/
resolve_ckpt() {
  local ckpt_path="$1"
  if [ -f "$ckpt_path" ]; then
    echo "$ckpt_path"
    return
  fi

  local alt_path
  case "$ckpt_path" in
    ${_BACKUP_DIR}*) alt_path="${_LOCAL_DIR}${ckpt_path#${_BACKUP_DIR}}" ;;
    ${_LOCAL_DIR}*)  alt_path="${_BACKUP_DIR}${ckpt_path#${_LOCAL_DIR}}" ;;
    *) alt_path="$ckpt_path" ;;
  esac

  if [ -f "$alt_path" ]; then
    echo ">>> 在备选路径找到 checkpoint: $alt_path <<<" >&2
    echo "$alt_path"
  else
    echo "$ckpt_path"
  fi
}

if [ -n "${DEBUG:-}" ]; then
  echo ">>> DEBUG mode: Stage 4 仅训练 10 步 <<<"
  STAGE3_STEPS=10
  STAGE4_STEPS=10
  STAGE4_SAVE=10
  STAGE4_T_INIT=10
  STAGE4_WARM=5
else
  STAGE3_STEPS=${STAGE3_STEPS:-5000}
  STAGE4_STEPS=${STAGE4_STEPS:-10000}
  STAGE4_SAVE=${STAGE4_SAVE:-10000}
  STAGE4_T_INIT=${STAGE4_T_INIT:-10000}
  STAGE4_WARM=${STAGE4_WARM:-1000}

  if [ -n "${STEP_SCALE:-}" ]; then
    echo ">>> STEP_SCALE=${STEP_SCALE}: Stage 4 训练步数乘以 ${STEP_SCALE} <<<"
    _scale() { awk "BEGIN{v=int($1 * ${STEP_SCALE} + 0.5); print (v<1?1:v)}"; }
    STAGE4_STEPS=$(_scale "${STAGE4_STEPS}")
    STAGE4_SAVE=$(_scale "${STAGE4_SAVE}")
    STAGE4_T_INIT=$(_scale "${STAGE4_T_INIT}")
    STAGE4_WARM=$(_scale "${STAGE4_WARM}")
  fi
fi

STAGE4_LR=${STAGE4_LR:-1e-5}
STAGE4_LR_MIN=${STAGE4_LR_MIN:-1e-6}
CKPT_JOB_ID=${PREV_JOB_ID:-${JOB_ID:--1}}

if [ -n "${INIT_CHECKPOINT:-}" ]; then
  _STAGE4_CKPT="${INIT_CHECKPOINT}"
  echo ">>> Stage 4 使用 INIT_CHECKPOINT: ${_STAGE4_CKPT} <<<"
elif [ -n "${STAGE4_CHECKPOINT:-}" ]; then
  _STAGE4_CKPT="${STAGE4_CHECKPOINT}"
  echo ">>> Stage 4 使用 STAGE4_CHECKPOINT: ${_STAGE4_CKPT} <<<"
else
  _STAGE4_CKPT=$(resolve_ckpt "${BASE_CHECKPOINT_DIR}/minicpm-v/${PREFIX}/stage_3/job_${CKPT_JOB_ID}_ckpt_${STAGE3_STEPS}/minicpm-v_${STAGE3_STEPS}.pt")
fi

echo ">>> 开始 Stage 4 <<<"
CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR} TENSORBOARD_DIR=${TENSORBOARD_DIR} \
  bash ./model_vlu_minicpm/script/train.sh \
  --train_file "${STAGE4_FILE}" \
  --max_len "${STAGE4_MAX_LEN}" \
  --total_max_length "${STAGE4_TOTAL_MAX_LEN}" \
  --query_num 64 \
  --lr "$(_lr "${STAGE4_LR}")" \
  --lr_min "${STAGE4_LR_MIN}" \
  --save_step "${STAGE4_SAVE}" \
  --use_datapipe \
  --use_adaptive_slice \
  --max_slice_nums "${MAX_SLICE_NUMS}" \
  --slice_new \
  --scale_resolution 448 \
  --gradient_accumulation_steps "${STAGE4_GRAD_ACCUM}" \
  --vision_encoder huggingface/siglip-so400m-14-980-flash-attn2-navit \
  --use_dynamic_batch \
  --t_initial "${STAGE4_T_INIT}" \
  --epochs 2 \
  --prefix "${PREFIX}/stage_4" \
  --precision bf16 \
  --sft \
  --use_new_schema \
  --warmup_lr_init 1e-8 \
  --data_queue_size 20 \
  --save_dataset \
  --llm_path "${LLM_PATH}" \
  --vocabs_path "${VOCABS_PATH}" \
  --split_by_rank \
  --warmup_t "${STAGE4_WARM}" \
  --gradient_checkpointing \
  --llm_gradient_checkpointing \
  --vpm_path "${VPM_PATH:-/path/to/backup/navit-siglip2/}" \
  --max_steps "${STAGE4_STEPS}" \
  --qwen3 \
  --cycle_limit 1 \
  --tune_vision \
  --tune_llm \
  --enhance_ocr \
  --aug_size \
  --loss_reduction_weight 0.5 \
  --model_checkpoint "${_STAGE4_CKPT}" \
  ${MODEL_EXTRA_ARGS} \
  "$@"
