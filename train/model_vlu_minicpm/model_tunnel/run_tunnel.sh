#!/bin/bash
set -e

REPO_ROOT=$(cd "$(dirname "$0")/../.." || exit 1; pwd)
cd "${REPO_ROOT}"

if [ -z "${PREFIX}" ]; then
  echo "错误: 必须通过环境变量 PREFIX 指定实验名称，例如: PREFIX=my_exp bash run_tunnel.sh"
  exit 1
fi

TUNNEL_USER=${TUNNEL_USER:-user}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_SLICE_NUMS=${MAX_SLICE_NUMS:-9}
LR_SCALE=${LR_SCALE:-1}
if [ "${LR_SCALE}" != "1" ]; then
  echo ">>> LR_SCALE=${LR_SCALE}: 所有阶段最大学习率乘以 ${LR_SCALE} <<<"
fi
STAGE_EXECUTED=0  # 追踪是否已有 stage 执行过（用于决定 SKIP_INSTALL）
_lr() { awk "BEGIN{printf \"%.2e\", $1 * ${LR_SCALE}}"; }

## ---- ENCODE_TYPE: uhd (默认, uhd_mlp_4_4) 或 whole (whole_mlp_4_4) ----
## 两种编码方式 max_slice_nums 保持一致, 但在 itembuilder.py 里走不同分支:
##   uhd_*  : slice_image_new, 出 thumbnail + 切片 (视觉 token 略多)
##   whole_*: 不切片, 整图 resize 后一次编码 (视觉 token 略少, 单 pack 能装更多样本)
## 把 max_len / total_max_length 按 1/ratio 对齐, 使两种方式同步数时过的
## 总样本数尽可能对齐.
ENCODE_TYPE=${ENCODE_TYPE:-uhd}
if [ "${ENCODE_TYPE}" = "whole" ]; then
  MODEL_TYPE_ARGS="--model_type whole_mlp_4_4"
  STAGE1_MAX_LEN=${STAGE1_MAX_LEN:-1408}
  STAGE2_MAX_LEN=${STAGE2_MAX_LEN:-6400}
  STAGE3_MAX_LEN=${STAGE3_MAX_LEN:-8704}
  STAGE4_MAX_LEN=${STAGE4_MAX_LEN:-3904}
elif [ "${ENCODE_TYPE}" = "uhd" ]; then
  MODEL_TYPE_ARGS="--model_type uhd_mlp_4_4"
  STAGE1_MAX_LEN=${STAGE1_MAX_LEN:-1700}
  STAGE2_MAX_LEN=${STAGE2_MAX_LEN:-8192}
  STAGE3_MAX_LEN=${STAGE3_MAX_LEN:-10240}
  STAGE4_MAX_LEN=${STAGE4_MAX_LEN:-4900}
else
  echo "错误: ENCODE_TYPE 必须是 'uhd' 或 'whole' (当前: ${ENCODE_TYPE})"
  exit 1
fi

LLM_PATH=/path/to/models/qwen3/llm_8b_init
VOCABS_PATH=/path/to/models/qwen3/llm_8b_init
MODEL_EXTRA_ARGS=""

CKPT_JOB_ID=${PREV_JOB_ID:-${JOB_ID:--1}}

command -v fuser >/dev/null 2>&1 || { echo ">>>  (psmisc)..."; apt-get update -qq && apt-get install -y -qq psmisc 2>/dev/null || yum install -y psmisc 2>/dev/null || true; }

STAGE1_FILE=${STAGE1_FILE:-/path/to/datasets/stage_1_rebalanced.json}
STAGE2_FILE=${STAGE2_FILE:-/path/to/datasets/stage_2_rebalanced.json}
STAGE3_FILE=${STAGE3_FILE:-/path/to/datasets/stage_3_rebalanced.json}
STAGE4_FILE=${STAGE4_FILE:-/path/to/datasets/stage_4_rebalanced.json}

if [ "${BACKUP}" = "False" ] || [ "${BACKUP}" = "false" ] || [ "${BACKUP}" = "0" ]; then
  BASE_CHECKPOINT_DIR=/user/${TUNNEL_USER}/checkpoints/model_tunnel/
  TENSORBOARD_DIR=/user/${TUNNEL_USER}/tensorboard/model_tunnel/
else
  BASE_CHECKPOINT_DIR=/backup/user/${TUNNEL_USER}/checkpoints/model_tunnel/
  TENSORBOARD_DIR=/backup/user/${TUNNEL_USER}/tensorboard/model_tunnel/
fi

_BACKUP_DIR=/backup/user/${TUNNEL_USER}/checkpoints/model_tunnel/
_LOCAL_DIR=/user/${TUNNEL_USER}/checkpoints/model_tunnel/
resolve_ckpt() {
  local ckpt_path="$1"
  if [ -f "$ckpt_path" ]; then
    echo "$ckpt_path"
  else
    # 尝试另一个目录
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
      echo "$ckpt_path"  # 都不存在，返回原路径让训练脚本报错
    fi
  fi
}

if [ "${DEBUG}" ]; then
  echo ">>> DEBUG mode: 每阶段仅训练 10 步 <<<"
  STAGE1_STEPS=10;   STAGE1_SAVE=10   STAGE1_T_INIT=10;   STAGE1_WARM=5
  STAGE2_STEPS=10;      STAGE2_SAVE=10;      STAGE2_T_INIT=10;      STAGE2_WARM=5
  STAGE3_STEPS=10;    STAGE3_SAVE=10;    STAGE3_T_INIT=10;    STAGE3_WARM=5
  STAGE4_STEPS=10;      STAGE4_SAVE=10;      STAGE4_T_INIT=10;      STAGE4_WARM=5
else
  STAGE1_STEPS=2000;   STAGE1_SAVE=1000;   STAGE1_T_INIT=2000;   STAGE1_WARM=1000
  STAGE2_STEPS=10000;     STAGE2_SAVE=5000;      STAGE2_T_INIT=10000;     STAGE2_WARM=1000
  STAGE3_STEPS=5000;    STAGE3_SAVE=2500;    STAGE3_T_INIT=5000;   STAGE3_WARM=500
  STAGE4_STEPS=10000;     STAGE4_SAVE=5000;     STAGE4_T_INIT=10000;     STAGE4_WARM=1000

  if [ -n "${STEP_SCALE}" ]; then
    _scale() { awk "BEGIN{v=int($1 * ${STEP_SCALE} + 0.5); print (v<1?1:v)}"; }
    STAGE1_STEPS=$(_scale $STAGE1_STEPS);  STAGE1_SAVE=$(_scale $STAGE1_SAVE);  STAGE1_T_INIT=$(_scale $STAGE1_T_INIT);  STAGE1_WARM=$(_scale $STAGE1_WARM)
    STAGE2_STEPS=$(_scale $STAGE2_STEPS);        STAGE2_SAVE=$(_scale $STAGE2_SAVE);        STAGE2_T_INIT=$(_scale $STAGE2_T_INIT);        STAGE2_WARM=$(_scale $STAGE2_WARM)
    STAGE3_STEPS=$(_scale $STAGE3_STEPS);    STAGE3_SAVE=$(_scale $STAGE3_SAVE);    STAGE3_T_INIT=$(_scale $STAGE3_T_INIT);    STAGE3_WARM=$(_scale $STAGE3_WARM)
    STAGE4_STEPS=$(_scale $STAGE4_STEPS);        STAGE4_SAVE=$(_scale $STAGE4_SAVE);        STAGE4_T_INIT=$(_scale $STAGE4_T_INIT);        STAGE4_WARM=$(_scale $STAGE4_WARM)
  fi
fi

_INIT_CKPT_USED=0

## Stage 1
STAGE1_LR=${STAGE1_LR:-1e-4}; STAGE1_LR_MIN=${STAGE1_LR_MIN:-5e-5}
if [ "${STAGE1_LARGE_LR}" ]; then
  STAGE1_LR=2e-4; STAGE1_LR_MIN=5e-5
  echo ">>> Stage 1 使用大学习率: lr=${STAGE1_LR}, lr_min=${STAGE1_LR_MIN} <<<"
fi
STAGE1_LR=$(_lr ${STAGE1_LR})
if [ "${SKIP_STAGE1}" ]; then
  echo ">>> 跳过 Stage 1 <<<"
else
  ## 如果传入了 INIT_CHECKPOINT 且尚未消费，则作为本 stage 的初始 checkpoint
  _STAGE1_CKPT_ARGS=""
  if [ -n "${INIT_CHECKPOINT}" ] && [ "${_INIT_CKPT_USED}" -eq 0 ]; then
    echo ">>> Stage 1 使用 INIT_CHECKPOINT: ${INIT_CHECKPOINT} <<<"
    _STAGE1_CKPT_ARGS="--model_checkpoint ${INIT_CHECKPOINT}"
    _INIT_CKPT_USED=1
  fi
  _VPM_CKPT_ARGS=""
  if [ -n "${VPM_CHECKPOINT}" ]; then
    echo ">>> Stage 1 使用 VPM_CHECKPOINT: ${VPM_CHECKPOINT} <<<"
    _VPM_CKPT_ARGS="--vpm_checkpoint ${VPM_CHECKPOINT}"
  fi
  echo ">>> 开始 Stage 1 <<<"
  ## packing 总长：uhd 1700 / whole 1408（按 dry-run 测得的 1.226 比例缩放，64 对齐）
  CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR} TENSORBOARD_DIR=${TENSORBOARD_DIR} bash ./model_vlu_minicpm/script/train.sh --train_file ${STAGE1_FILE} --max_len ${STAGE1_MAX_LEN} --total_max_length ${STAGE1_MAX_LEN} --query_num 64 --lr ${STAGE1_LR} --lr_min ${STAGE1_LR_MIN} --save_step ${STAGE1_SAVE} --use_datapipe --use_adaptive_slice --max_slice_nums 1 --slice_new --scale_resolution 448 --gradient_accumulation_steps ${GRAD_ACCUM} --vision_encoder huggingface/siglip-so400m-14-980-flash-attn2-navit --use_dynamic_batch --t_initial ${STAGE1_T_INIT} --epochs 4 --prefix ${PREFIX}/stage_1 --precision bf16 --sft --use_new_schema --warmup_lr_init 1e-8 --data_queue_size 100 --save_dataset --llm_path ${LLM_PATH} --vocabs_path ${VOCABS_PATH} --split_by_rank --warmup_t ${STAGE1_WARM} --gradient_checkpointing --vpm_path /path/to/backup/navit-siglip2/ --max_steps ${STAGE1_STEPS} --qwen3 ${_STAGE1_CKPT_ARGS} ${_VPM_CKPT_ARGS} ${MODEL_TYPE_ARGS} ${MODEL_EXTRA_ARGS} "$@"
  STAGE_EXECUTED=1; CKPT_JOB_ID=${JOB_ID:--1}
  wait_gpu_release
fi

## Stage 2
if [ "${SKIP_STAGE2}" ]; then
  echo ">>> 跳过 Stage 2 <<<"
else
  [ "${STAGE_EXECUTED}" -eq 1 ] && export SKIP_INSTALL=1
  if [ -n "${INIT_CHECKPOINT}" ] && [ "${_INIT_CKPT_USED}" -eq 0 ]; then
    _STAGE2_CKPT="${INIT_CHECKPOINT}"
    echo ">>> Stage 2 使用 INIT_CHECKPOINT: ${_STAGE2_CKPT} <<<"
    _INIT_CKPT_USED=1
  else
    _STAGE2_CKPT=$(resolve_ckpt ${BASE_CHECKPOINT_DIR}/minicpm-v/${PREFIX}/stage_1/job_${CKPT_JOB_ID}_ckpt_${STAGE1_STEPS}/minicpm-v_${STAGE1_STEPS}.pt)
  fi
  echo ">>> 开始 Stage 2 <<<"
  CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR} TENSORBOARD_DIR=${TENSORBOARD_DIR} bash ./model_vlu_minicpm/script/train.sh --train_file ${STAGE2_FILE} --max_len ${STAGE2_MAX_LEN} --total_max_length ${STAGE2_MAX_LEN} --query_num 64 --lr $(_lr 1e-5) --lr_min 5e-6 --save_step ${STAGE2_SAVE} --use_datapipe --use_adaptive_slice --max_slice_nums ${MAX_SLICE_NUMS} --slice_new --scale_resolution 448 --gradient_accumulation_steps ${GRAD_ACCUM} --vision_encoder huggingface/siglip-so400m-14-980-flash-attn2-navit --use_dynamic_batch --t_initial ${STAGE2_T_INIT} --epochs 2 --prefix ${PREFIX}/stage_2 --precision bf16 --sft --use_new_schema --warmup_lr_init 1e-8 --data_queue_size 100 --save_dataset --llm_path ${LLM_PATH} --vocabs_path ${VOCABS_PATH} --split_by_rank --warmup_t ${STAGE2_WARM} --gradient_checkpointing --vpm_path /path/to/backup/navit-siglip2/ --max_steps ${STAGE2_STEPS} --qwen3 --llm_gradient_checkpointing --tune_vision --tune_llm --vision_lr_scale 0.2 --model_checkpoint ${_STAGE2_CKPT} ${MODEL_TYPE_ARGS} ${MODEL_EXTRA_ARGS} "$@"
  STAGE_EXECUTED=1; CKPT_JOB_ID=${JOB_ID:--1}
  wait_gpu_release
fi

## Stage 3
if [ "${SKIP_STAGE3}" ]; then
  echo ">>> 跳过 Stage 3 <<<"
else
  [ "${STAGE_EXECUTED}" -eq 1 ] && export SKIP_INSTALL=1
  if [ -n "${INIT_CHECKPOINT}" ] && [ "${_INIT_CKPT_USED}" -eq 0 ]; then
    _STAGE3_CKPT="${INIT_CHECKPOINT}"
    echo ">>> Stage 3 使用 INIT_CHECKPOINT: ${_STAGE3_CKPT} <<<"
    _INIT_CKPT_USED=1
  else
    _STAGE3_CKPT=$(resolve_ckpt ${BASE_CHECKPOINT_DIR}/minicpm-v/${PREFIX}/stage_2/job_${CKPT_JOB_ID}_ckpt_${STAGE2_STEPS}/minicpm-v_${STAGE2_STEPS}.pt)
  fi
  echo ">>> 开始 Stage 3 <<<"
  CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR} TENSORBOARD_DIR=${TENSORBOARD_DIR} bash ./model_vlu_minicpm/script/train.sh --train_file ${STAGE3_FILE} --max_len ${STAGE3_MAX_LEN} --total_max_length ${STAGE3_MAX_LEN} --query_num 64 --lr $(_lr 5e-5) --lr_min 1e-5 --save_step ${STAGE3_SAVE} --use_datapipe --use_adaptive_slice --max_slice_nums ${MAX_SLICE_NUMS} --slice_new --scale_resolution 448 --gradient_accumulation_steps ${GRAD_ACCUM} --vision_encoder huggingface/siglip-so400m-14-980-flash-attn2-navit --use_dynamic_batch --t_initial ${STAGE3_T_INIT} --epochs 2 --prefix ${PREFIX}/stage_3 --precision bf16 --sft --use_new_schema --warmup_lr_init 1e-8 --data_queue_size 20 --save_dataset --llm_path ${LLM_PATH} --vocabs_path ${VOCABS_PATH} --split_by_rank --warmup_t ${STAGE3_WARM} --gradient_checkpointing --llm_gradient_checkpointing --vpm_path /path/to/backup/navit-siglip2/ --max_steps ${STAGE3_STEPS} --qwen3 --tune_vision --tune_llm --vision_lr_scale 0.2 --enhance_ocr --model_checkpoint ${_STAGE3_CKPT} ${MODEL_TYPE_ARGS} ${MODEL_EXTRA_ARGS} "$@"
  STAGE_EXECUTED=1; CKPT_JOB_ID=${JOB_ID:--1}
  wait_gpu_release
fi

## Stage 4
[ "${STAGE_EXECUTED}" -eq 1 ] && export SKIP_INSTALL=1
if [ -n "${INIT_CHECKPOINT}" ] && [ "${_INIT_CKPT_USED}" -eq 0 ]; then
  _STAGE4_CKPT="${INIT_CHECKPOINT}"
  echo ">>> Stage 4 使用 INIT_CHECKPOINT: ${_STAGE4_CKPT} <<<"
  _INIT_CKPT_USED=1
else
  _STAGE4_CKPT=$(resolve_ckpt ${BASE_CHECKPOINT_DIR}/minicpm-v/${PREFIX}/stage_3/job_${CKPT_JOB_ID}_ckpt_${STAGE3_STEPS}/minicpm-v_${STAGE3_STEPS}.pt)
fi
echo ">>> 开始 Stage 4 <<<"
CHECKPOINT_DIR=${BASE_CHECKPOINT_DIR} TENSORBOARD_DIR=${TENSORBOARD_DIR} bash ./model_vlu_minicpm/script/train.sh --train_file ${STAGE4_FILE} --max_len ${STAGE4_MAX_LEN} --total_max_length ${STAGE4_MAX_LEN} --query_num 64 --lr $(_lr 1e-5) --lr_min 1e-6 --save_step ${STAGE4_SAVE} --use_datapipe --use_adaptive_slice --max_slice_nums ${MAX_SLICE_NUMS} --slice_new --scale_resolution 448 --gradient_accumulation_steps ${GRAD_ACCUM} --vision_encoder huggingface/siglip-so400m-14-980-flash-attn2-navit --use_dynamic_batch --t_initial ${STAGE4_T_INIT} --epochs 3 --prefix ${PREFIX}/stage_4 --precision bf16 --sft --use_new_schema --warmup_lr_init 1e-8 --data_queue_size 20 --save_dataset --llm_path ${LLM_PATH} --vocabs_path ${VOCABS_PATH} --split_by_rank --warmup_t ${STAGE4_WARM} --gradient_checkpointing --llm_gradient_checkpointing --vpm_path /path/to/backup/navit-siglip2/ --max_steps ${STAGE4_STEPS} --qwen3 --tune_vision --tune_llm  --enhance_ocr --aug_size --loss_reduction_weight 0.5 --model_checkpoint ${_STAGE4_CKPT} ${MODEL_TYPE_ARGS} ${MODEL_EXTRA_ARGS} "$@"
