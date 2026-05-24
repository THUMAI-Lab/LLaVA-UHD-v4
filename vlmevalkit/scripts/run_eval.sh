export PATH=/usr/local/cuda/bin:$PATH

# Install tzdata for timezone files (failure is non-fatal)
if [ ! -f /usr/share/zoneinfo/Asia/Shanghai ]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tzdata 2>/dev/null || true
fi
if [ -f /usr/share/zoneinfo/Asia/Shanghai ]; then
  export TZ=Asia/Shanghai
  ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime 2>/dev/null || true
fi

script_start=$(date +%s)

if [ -n "${VENV_PATH:-}" ]; then
  source "${VENV_PATH}/bin/activate"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Active Python virtual environment: $VIRTUAL_ENV"
fi
# Optional pip mirror:
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# pip install --upgrade pip
# pip install -r requirements.txt
# pip install -U seaborn tabulate 
# pip install -U sty 
# pip install -U portalocker 
# pip install -U pycocoevalcap 
# pip install -U openpyxl 
# pip install -U xlsxwriter
# pip install -U validators 
# # pip uninstall -y opencv
# pip install opencv-python==4.8.0.76
# pip install numpy==1.26.4
# pip install ruff==0.5.5
# pip install tqdm==4.66.4
# pip install rich
# pip install pillow==9.4.0
# pip install peft
# # pip install timm==0.9.10
# Optional local wheel installs (replace with your own paths if needed):
# pip install /path/to/pkgs/starlette-0.42.0-py3-none-any.whl
# pip install /path/to/pkgs/torchvision-0.17.0+cu118-cp310-cp310-linux_x86_64.whl
# pip install /path/to/pkgs/torch-2.2.0+cu118-cp310-cp310-linux_x86_64.whl
# pip install /path/to/pkgs/flash_attn-2.5.9.post1+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# # 运行 3o 需要降低版本，运行 2.6 之前的版本，可以更新到 4.43 以上的新版本
# # pip install transformers==4.40.2
# # pip install transformers==4.44.2
# pip install transformers==4.51.0
pip list

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'
# export PYTHONPATH=$(dirname $SELF_DIR):$PYTHONPATH

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPU_NUM=${GPU_NUM:-8}
CUDA_DEVICES=$(seq -s',' 0 $((GPU_NUM-1)))
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

which torchrun
which python

export MODEL_PATH=$1
export FORCE_LOCAL=True

# fp16 17-18G
# int4 7-8G
MODELNAME=$2
DATALIST=$3
SAVE_PREFIX=$4

base_name=$(basename $MODEL_PATH .pt)

save_root="${SAVE_ROOT:-/data/checkpoints/vlmeval_kit}"
work_dir="$save_root/$SAVE_PREFIX/$base_name"

echo "work directory of $MODELNAME: $work_dir"

_fmt_elapsed() { local s=$1; printf '%dh %dm %ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

# --- Inference pass 1 ---
infer1_start=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting inference pass 1 with model $MODELNAME on datasets $DATALIST"
torchrun --master_port 29500 --nproc_per_node=$GPU_NUM run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --mode infer
infer1_exit=$?
infer1_elapsed=$(( $(date +%s) - infer1_start ))
[ $infer1_exit -ne 0 ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Inference pass 1 exited with code $infer1_exit"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inference pass 1 done. Elapsed: $(_fmt_elapsed $infer1_elapsed)"

# --- Inference pass 2 (retries items still marked FAIL_MSG) ---
infer2_start=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting inference pass 2 with model $MODELNAME on datasets $DATALIST"
torchrun --master_port 29501 --nproc_per_node=$GPU_NUM run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --mode infer
infer2_exit=$?
infer2_elapsed=$(( $(date +%s) - infer2_start ))
[ $infer2_exit -ne 0 ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Inference pass 2 exited with code $infer2_exit"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inference pass 2 done. Elapsed: $(_fmt_elapsed $infer2_elapsed)"

# --- Evaluation ---
eval_start=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation with model $MODELNAME on datasets $DATALIST"
python run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --nproc 16 --verbose
eval_elapsed=$(( $(date +%s) - eval_start ))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation done. Elapsed: $(_fmt_elapsed $eval_elapsed)"

script_end=$(date +%s)
total_elapsed=$(( script_end - script_start ))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Infer pass1: $(_fmt_elapsed $infer1_elapsed) | Infer pass2: $(_fmt_elapsed $infer2_elapsed) | Eval: $(_fmt_elapsed $eval_elapsed) | Total: $(_fmt_elapsed $total_elapsed)"