#!/bin/bash
set -e

export PATH=/usr/local/cuda/bin:$PATH

if [ "${SKIP_INSTALL:-}" ];then
	echo "Skip install"
else
  PIP_SOURCE="${PIP_SOURCE:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  pip config set global.index-url "$PIP_SOURCE"
  TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.51.0}"
  pip install "transformers==${TRANSFORMERS_VERSION}"
  pip install timm==0.9.10 opencc-python-reimplemented jieba imgaug decord opencv-python==4.6.0.66 diffusers PyMuPDF==1.24.10
  pip install torchdata==0.7.1
  pip install deepspeed==0.14.4
  pip install albumentations
  pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66 opencv-python-headless==4.6.0.66
  pip install func_timeout
  pip show transformers | awk '/Version:/{print $2}' | grep -q '^4\.44\.2$' && pip install huggingface-hub==0.25.2
fi

pip list

export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'

SELF_DIR=$(cd "$(dirname "$0")" || exit 1; pwd)
REPO_ROOT=$(cd "${SELF_DIR}/../.." || exit 1; pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LLM_PATH=${LLM_PATH:-/path/to/Qwen2_5-7B-Instruct-init/}
VOCABS_PATH=${VOCABS_PATH:-${VOCAB_PATH:-/path/to/Qwen2_5-7B-Instruct-init/}}


# --------------- 运行参数 ---------------
OPTS=""
OPTS+=" --self_dir ${SELF_DIR}"

# 模型 config
OPTS+=" --llm_path ${LLM_PATH}"
OPTS+=" --vocabs_path ${VOCABS_PATH}"

# 模型参数
OPTS+=" --query_num 64"
# use 27 layers siglip
#OPTS+=" --drop_vision_last_layer"

# dataset 配置
OPTS+=" --max_len 200"
OPTS+=" --batch_size 64"
OPTS+=" --save_step 1000"
OPTS+=" --epochs 2"
# training
# OPTS+=" --split_by_rank"
OPTS+=" --flash cuda"
OPTS+=" --tune_resampler"
OPTS+=" --vision_encoder vit_so400m_patch14_siglip_384.webli"

OPTS+=" --dynamic_deepspeed_config"
OPTS+=" --train_batch_size 15360"
OPTS+=" --lr 1e-4"
OPTS+=" --stage 2"
OPTS+=" --precision bf16"
OPTS+=" --log_step 5"

# scheduler
OPTS+=" --use_cosine_restart_scheduler"
OPTS+=" --t_initial 400"
OPTS+=" --lr_min 2e-6"
OPTS+=" --cycle_mul 1.5"
OPTS+=" --cycle_decay 0.9"
OPTS+=" --cycle_limit 10"
OPTS+=" --warmup_t 200"
OPTS+=" --warmup_lr_init 5e-7"

# caterpillar 8b sft 没有加 system prompt
OPTS+=" --no_system_prompt"

OPTS+=" --use_image_id"
OPTS+=" --use_new_schema"

OPTS+=" --use_datapipe"
OPTS+=" --use_adaptive_slice"

OPTS+=" $@"

# use torchrun
# 多机多卡使用这种方式
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# 支持单机多卡和多机多卡 DP 时会没有以下环境
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-12345}

echo "-------nvidia-smi------"
nvidia-smi

echo "-------pip list------"
pip list

# 某些调度环境需用 --master_addr=${MASTER_ENDPOINT} 而非 rdzv_endpoint，设 USE_MASTER_ENDPOINT=1 启用
if [ "${USE_MASTER_ENDPOINT:-}" == "1" ]; then
  CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} --master_addr=${MASTER_ENDPOINT} --master_port=${MASTER_PORT} model_vlu_minicpm/train_minicpmv.py  ${OPTS}"
else
  CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} model_vlu_minicpm/train_minicpmv.py  ${OPTS}"
fi


echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

$CMD