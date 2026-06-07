# coding=utf-8

import os
import gc
import time
import glob
import torch
import argparse
import torch.distributed
import numpy as np
import random
from datetime import datetime
from timm import create_model
from multimodal_common.utils import utils
from multimodal_common.utils.logger import init_logger

logger = init_logger(__name__, level="INFO")


def get_args():
    parser = argparse.ArgumentParser(
        'VLLM pre-training script', add_help=False)

    parser.add_argument('--self_dir', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--audio_data_json_path', type=str)
    parser.add_argument('--train_file_sstable', type=str)
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--test_file', type=str)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--split_by_rank', action='store_true',
                        default=False, help="split parquet by rank")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--save_step', default=100, type=int)
    parser.add_argument('--sft', action='store_true', help='is traing all parameter')
    parser.add_argument('--loss_reduction_weight', type=float, default=0,
                        help='sample loss reweight power r: w=1/L^r. 0=token-level, 0.5=sqrt, 1=sample-level')
    parser.add_argument('--tune_vision', action='store_true', help='is train vision parameter')
    parser.add_argument('--tune_resampler', action='store_true', help='is train resampler parameter')
    parser.add_argument('--tune_apm', action='store_true', help='is train audio pretrain model parameter')
    parser.add_argument('--tune_tts', action='store_true', help='is train tts model parameter')
    parser.add_argument('--tune_llm', action='store_true', help='is train llm parameter')
    parser.add_argument('--tune_llm_input', action='store_true', help='is train llm input parameter')
    parser.add_argument('--tune_llm_head', action='store_true', help='is train llm head parameter')
    parser.add_argument('--only_tune_special_head', action='store_true', help='is train llm special head parameter')
    parser.add_argument('--tune_patch_embedding', action='store_true', help='is train vit patch embedding')
    parser.add_argument('--train_last_8layer', action='store_true', help='is train llm last 8 layer')
    parser.add_argument('--delta_tuning', action='store_true', help='is use lora')
    parser.add_argument('--ncache', default=20, type=int)
    parser.add_argument('--log_train_ds_loss', action='store_true', help='is log_train_ds_loss')
    parser.add_argument('--time_stamp_train', action='store_true', help='is video frames interleaved by <x seconds>')

    # Model parameters
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--llm_path', default=None, help='Path to LLM model to use', type=str)
    parser.add_argument('--apm_path', default=None, help='Path to Whisper model to use', type=str) 
    parser.add_argument('--tts_path', default=None, help='Path to TTS model to use', type=str)   
    parser.add_argument('--vocabs_path', default=None, help='Path to vocabs to use', type=str)
    parser.add_argument('--model_checkpoint', default=None, help='Path to VLLM model to use', type=str)
    parser.add_argument('--llm_name', help='Path to LLM model to use',
                        choices=['clip', 'llama', 'cpm_bee_10b', 'cpm_bee_20b'], default='cpm_bee_10b', type=str)
    parser.add_argument('--llm_checkpoint', default=None, help='Path to LLM model to use', type=str)
    parser.add_argument('--data_state_dict_path', default=None, help='Path to dataset state dict', type=str)
    parser.add_argument(
        '--vpm_path', help='Path to VPM model to use', type=str)
    parser.add_argument('--vpm_checkpoint',
                        help='Path to VPM model to use', type=str)
    parser.add_argument('--sd_path', help='Path to SD model to use', type=str)
    parser.add_argument('--sd_checkpoint',
                        help='Path to SD model to use', type=str)
    parser.add_argument('--vision_encoder', default='clip', type=str)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--drop_vision_last_layer', action='store_true', help='is drop last layer')
    parser.add_argument('--prefix', default=None, help='Path prefix to save file', type=str)
    parser.add_argument('--fix_navit', action='store_true', help='(deprecated, always enabled)')
    parser.add_argument('--use_image_id', action='store_true', help='is use image_id')
    parser.add_argument('--use_new_schema', action='store_true', help='is use new_schema')
    parser.add_argument('--use_uhd_resampler', action='store_true', help='(deprecated, no longer used)')


    # vlu
    parser.add_argument('--img_aug', action='store_true', default=False, help="use image aug")
    parser.add_argument('--enhance_ocr', action='store_true', default=False, help="use image enhance ocr")
    parser.add_argument('--interleave', action='store_true', default=False, help="use image-text interleave mode")
    parser.add_argument('--skip_overlength', action='store_true', default=False, help="is skip over length data")
    parser.add_argument('--use_im_start_end', action='store_true', default=False, help="is use im_start_end")
    parser.add_argument('--skip_no_image', action='store_true', default=False, help="is skip no image data")
    parser.add_argument('--im_independent', action='store_true', default=False, help="is use im_independent mode")
    parser.add_argument("--flash", default="none", choices=["none", "1d", "triton", "cuda"])
    parser.add_argument('--packing_max_images', default=10000, type=int, help='max images for packing')
    parser.add_argument("--reverse_order", action='store_true', default=False)

    # vlg
    parser.add_argument('--clip_checkpoint', help='Path to CLIP model to use', type=str)
    parser.add_argument('--reference_image_path', help='Path to reference image for fid', type=str)
    parser.add_argument('--min_resolution', default=0, type=int, help='images input size for backbone')

    parser.add_argument('--train_task', default='caption', choices=['caption', 'text2img'], type=str,
                        help='training task')
    parser.add_argument('--training_mode', default='modify_only_kv',
                        choices=['modify_only_kv', 'modify_kv', 'add_mlp', 'only_mlp', 'all_unet'])
    parser.add_argument('--max_length', default=2048, type=int, help='max length of input')
    parser.add_argument('--total_max_length', default=2048, type=int, help='max length of input')

    parser.add_argument('--text_hidden_size', default=4096, type=int, help='max length of input')
    parser.add_argument('--prompt_text')
    parser.add_argument('--dataset_queue_size', type=int, default=5000)

    # ----- Training -----
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--exp_name', default='minicpm-v', type=str)
    parser.add_argument('--query_num', default=32, type=int,
                        help='query numbers')
    parser.add_argument('--max_len', default=96, type=int,
                        help='max len')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--eval', action='store_true', default=False,
                        help="Perform evaluation only")
    parser.add_argument('--eval_step', default=5000, type=int, help='evaluate step')
    parser.add_argument('--old_version', action='store_true', default=False,
                        help="version for cpmbee-10b")
    parser.add_argument('--save_dataset', action='store_true', default=False, help="is save dataset state dict")
    parser.add_argument('--ensure_break', action='store_true', default=False)
    parser.add_argument('--skip_files', default=0, type=int)
    parser.add_argument('--global_start_step', default=0, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--llm_gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--use_datapipe', action='store_true', default=False)
    parser.add_argument('--aug_size', action='store_true', default=False)
    parser.add_argument('--data_queue_size', default=500, type=int, help='data_queue size')
    parser.add_argument('--force_batch', action='store_true', default=False)
    parser.add_argument('--empty_cache_step', default=1000, type=int, help='empty gpu cache')
    parser.add_argument('--skip_steps', default=0, type=int)
    parser.add_argument('--max_steps', default=0, type=int)
    parser.add_argument('--num_ranks_for_sstable', default=0, type=int,
        help='number of ranks read sstable data, used for fused parquet and sstable dataloader')
    parser.add_argument('--num_ranks_for_audio', default=0, type=int)


    ## video
    parser.add_argument('--video_max_frame_nums', default=48, type=int)
    parser.add_argument('--video_max_slice_nums', default=2, type=int)
    parser.add_argument('--high_res_frame_nums', default=16, type=int)
    parser.add_argument('--use_low_res', action='store_true', default=False, help='(deprecated, no longer used)')
    parser.add_argument('--no_batch_vit', action='store_true', default=False,)
    parser.add_argument('--no_grad_vit', action='store_true', default=False,)
    parser.add_argument('--low_res_query_nums', default=1, type=int, help='(deprecated, no longer used)')
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--stack_frame_nums', default=1, type=int)
    parser.add_argument('--random_stack_ratio', default=0, type=float)

    # 
    parser.add_argument('--streaming_rate', default=0.5, type=float, help='streaming_rate')

    
    # 替代原有的 load_ckpt_dir,load_ckpt_tag
    parser.add_argument('--need_resume', action='store_true', default=False,
                        help="resume with deepspeed states")
    parser.add_argument('--need_resume_tag')
    parser.add_argument('--dataset_resume_dir', default=None, help="resume")
    parser.add_argument('--dataset_resume_dir_sstable', default=None, help="resume sstable dataset")
    parser.add_argument('--deepspeed_resume_dir', default=None, help="resume")
    parser.add_argument('--deepspeed_resume_tag', default=None, help="resume")
    ## deepspeed 参数
    parser.add_argument('--deepspeed_config', default=None, help='Path to deepspeed config to use', type=str)
    parser.add_argument('--save_deepspeed', action='store_true', default=False, help="is save deepspeed checkpoint")

    parser.add_argument('--dynamic_deepspeed_config', action='store_true', default=False)
    parser.add_argument('--train_micro_batch_size_per_gpu', default=16, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument('--train_batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--vision_lr', default=0, type=float)
    parser.add_argument('--vision_lr_scale', default=0, type=float)
    parser.add_argument('--mup_lr_scale', default=0, type=float)
    parser.add_argument('--stage', default=2, type=int)
    parser.add_argument('--offload', action='store_true', default=False)
    parser.add_argument('--offload_ratio', default=1.0, type=float)
    parser.add_argument('--precision', default='fp16', type=str)
    parser.add_argument('--overlap_comm', action='store_true', default=False)
    parser.add_argument('--adam_beta2', default=0.98, type=float)

    parser.add_argument('--zero3_tune', action='store_true', default=False,
                        help='Enable ZeRO-3 tuning preset (overlap_comm + bucket sizes + '
                             'stage3 persistence/live/prefetch + bf16 cleanup). 仅 --stage 3 生效.')
    parser.add_argument('--zero3_persistence_threshold', default=0, type=int,
                        help='preset=1e6 (常驻 LayerNorm γ/β、Merger bias、resampler query 等小参数)')
    parser.add_argument('--zero3_max_live_parameters', default=0, type=int,
                        help='preset=2e9')
    parser.add_argument('--zero3_prefetch_bucket_size', default=0, type=int,
                        help='preset=1e8 (显式优于让 DS 用 hidden² 推算, 多模态 hidden 不统一)')
    parser.add_argument('--zero3_reduce_bucket_size', default=0, type=int,
                        help='preset=5e8')
    parser.add_argument('--zero3_allgather_bucket_size', default=0, type=int,
                        help='preset=5e8')

    # LRScheduler
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--use_linear_decay_scheduler', action='store_true', default=False)

    # for CosineLRScheduler
    parser.add_argument('--use_cosine_restart_scheduler', action='store_true', default=False)
    parser.add_argument('--t_initial', default=500, type=int)
    parser.add_argument('--lr_min', default=5e-6, type=float)
    parser.add_argument('--cycle_mul', default=1.5, type=float)
    parser.add_argument('--cycle_decay', default=0.9, type=float)
    parser.add_argument('--cycle_limit', default=10, type=int)
    parser.add_argument('--warmup_t', default=200, type=int)
    parser.add_argument('--warmup_step,', default=200, type=int)
    parser.add_argument('--warmup_lr_init', default=1e-6, type=float)

    # for WSDLRScheduler (Warmup-Stable-Decay)
    parser.add_argument('--use_wsd_scheduler', action='store_true', default=False)
    parser.add_argument('--stable_t', default=300, type=int, help='Step at which the stable phase ends and decay begins')

    # for 自适应任意分辨率 + slice
    parser.add_argument('--use_adaptive_slice', action='store_true', default=False)
    parser.add_argument('--use_dynamic_batch', action='store_true', default=False)
    parser.add_argument('--max_slice_nums', default=9, type=int)
    parser.add_argument('--patch_size', default=14, type=int)
    parser.add_argument('--scale_resolution', default=448, type=float)
    parser.add_argument('--floating_ratio', default=0.33, type=float)
    parser.add_argument('--slice_new', action='store_true', default=False)

    parser.add_argument('--no_system_prompt', action='store_true', default=False)

    ## for qwen
    parser.add_argument('--qwen2', action='store_true', default=False)
    parser.add_argument('--qwen3', action='store_true', default=False)


    ## for evaluate
    parser.add_argument('--eval_common', action='store_true', default=False)
    parser.add_argument('--eval_few_shot', action='store_true', default=False)
    parser.add_argument('--shots', default=None, help="few shots", type=str)
    parser.add_argument("--eval_video", action="store_true", default=False)
    parser.add_argument('--eval_ckpt_path', default=None, help='hf path', type=str)
    parser.add_argument('--eval_datast_list', default=None, type=str)
    parser.add_argument('--use_subtitle', action='store_true', default=False)
    parser.add_argument('--eval_save_path', default=None, type=str)
    parser.add_argument('--eval_case_save_path', default=None, type=str)
    parser.add_argument('--merge_nums', default=5, type=int)
    parser.add_argument('--merge_subtitle', action='store_true', default=False)
    parser.add_argument('--packing_nums', default=None, type=int)

    # -----  distributed training parameters -----
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # perception encoder
    parser.add_argument('--use_ln_post', action='store_true', default=False)
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='resampler',
        choices=['resampler', 'uhd_mlp_insert_window_attention_ViTmlp_4_4'],
        help="Vision connector type. This minimal SFT repo supports resampler and uhd_mlp_insert_window_attention_ViTmlp_4_4."
    )

    # for ViT insertion
    parser.add_argument('--insert_layer_id', default=-1, type=int)

    # mixed 4x/16x downsampling: ratio of single-image data using 4x (0.0=disabled)
    parser.add_argument('--mixed_downsample', type=float, default=0.0)

    args = parser.parse_args()

    if args.mixed_downsample > 0 and args.model_type != 'uhd_mlp_insert_window_attention_ViTmlp_4_4':
        raise ValueError("--mixed_downsample requires uhd_mlp_insert_window_attention_ViTmlp_4_4")
    # 文件保存/导出路径（可用环境变量 CHECKPOINT_DIR / TENSORBOARD_DIR 覆盖）
    args.exp_ckpt_dir = os.getenv('CHECKPOINT_DIR', '/data/checkpoints/')
    if args.exp_name or args.prefix:
        prefix = '/'.join([i for i in [args.exp_name, args.prefix] if i])
        args.exp_ckpt_dir = os.path.join(args.exp_ckpt_dir, prefix)
    logger.info(f'export dir={args.exp_ckpt_dir}')

    tensorboard_base = os.getenv('TENSORBOARD_DIR', '/data/tensorboard/')
    job_id = os.getenv('JOB_ID', -1)
    args.tensorboard = '{base}/{export_model_name}-job_{job_id}-{timestamp}'.format(
        base=tensorboard_base,
        timestamp=datetime.now().strftime("%Y%m%d%H%M%S"), 
        job_id=job_id,
        export_model_name='/'.join([i for i in [args.exp_name, args.prefix] if i])
    )
    # ----- repo 内路径相关参数 -----
    # 模型 config，从基准模型的 config 复制而来
    if not args.llm_path:
        args.llm_path = _check_default_path(os.path.join(args.self_dir, 'config/config.json'))
    if not args.vocabs_path:
        args.vocabs_path = _check_default_path(os.path.join(args.self_dir, 'config/vocabs.txt'))
    if not args.deepspeed_config:
        args.deepspeed_config = _check_default_path(os.path.join(args.self_dir, 'config/deepspeed.json'))
    logger.info("get_args() done")
    return args


def _check_default_path(path: str):
    if os.path.exists(path):
        return path
    else:
        return None


def _extract_ckpt_path(base_dir: str):
    paths = glob.glob(base_dir + '/*.pt')
    if len(paths) > 0:
        return paths[0]
    else:
        logger.warning(f'.pt file not found in base_dir({base_dir})')
        return None


def train_data_validator(filepath: str) -> bool:
    if filepath.endswith("parquet"):
        return True
    return False


def setup(args):
    # init dist
    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    logger.info(f"rank={rank} init_distributed_mode done")

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # init dirs
    necessary_dirs = [args.exp_ckpt_dir, args.tensorboard]
    if utils.is_main_process():
        for necessary_dir in necessary_dirs:
            if not necessary_dir:
                continue
            os.makedirs(necessary_dir, exist_ok=True)
    logger.info(f"rank={rank} setup(args) done")


