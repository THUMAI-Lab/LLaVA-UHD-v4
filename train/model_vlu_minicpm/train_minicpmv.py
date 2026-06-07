import glob
import math
import os
import re
import time
import gc
from copy import deepcopy
import pandas as pd

from transformers import AutoModelForCausalLM

from multimodal_common.dataset.parquetdataset import get_node_info
from timm.scheduler import CosineLRScheduler


from multimodal_common.dataset.dataloader import UnpadParquetDataloader, UnpadCollater, CudaPrefetcher
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from multimodal_common import initializer, exporter
from multimodal_common.tokenizers import Qwen2TokenizerFastWrapper

from multimodal_common.utils import utils
import json
import torch
import datetime
import deepspeed
import timm
import torch.distributed
import torch.utils.data

from multimodal_common.dataset.itembuilder import Qwen2ChatBuilder
from multimodal_common.base_models.vlu.vlu_minicpm_navit import VLU_SmartCPM
from multimodal_common.scheduler.linear_decay_lr import LinearDecayLRScheduler
from multimodal_common.scheduler.wsd_lr import WSDLRScheduler
from deepspeed.utils import logger

from multimodal_common.utils.constants import bot_indicator

_original_load = torch.load
def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load


class SafeSummaryWriter:
    """SummaryWriter 的兜底封装：当底层存储不稳定（如 NFS/分布式存储出现
    OSError [Errno 5] Input/output error）时，吞掉异常并打印告警，避免训练
    主进程崩溃退出。出错后会重置内部 file_writer，下一次写入时尝试自动重建。

    用法与 torch.utils.tensorboard.SummaryWriter 一致。
    """

    # 日志限速：同一类异常每 _LOG_INTERVAL 秒最多打印一次告警
    _LOG_INTERVAL = 30.0

    def __init__(self, *args, **kwargs):
        from torch.utils.tensorboard import SummaryWriter
        self._SummaryWriterCls = SummaryWriter
        self._ctor_args = args
        self._ctor_kwargs = kwargs
        self._writer = None
        self._last_warn_ts = 0.0
        self._open()

    def _open(self):
        try:
            self._writer = self._SummaryWriterCls(
                *self._ctor_args, **self._ctor_kwargs
            )
        except Exception as e:
            self._writer = None
            self._warn(f"open SummaryWriter failed: {e!r}")

    def _warn(self, msg):
        now = time.time()
        if now - self._last_warn_ts >= self._LOG_INTERVAL:
            self._last_warn_ts = now
            try:
                logger.warning(f"[SafeSummaryWriter] {msg}")
            except Exception:
                pass

    def _reset_after_error(self):
        # tensorboard 的 EventFileWriter 后台线程一旦崩溃，后续每次 add_event
        # 都会重新抛出缓存的异常。这里强制丢弃损坏的 file_writer，下次写入时
        # SummaryWriter._get_file_writer 会惰性重建。
        w = self._writer
        if w is None:
            return
        try:
            fw = getattr(w, "file_writer", None)
            if fw is not None:
                try:
                    fw.close()
                except Exception:
                    pass
            w.file_writer = None
            w.all_writers = None
        except Exception:
            pass

    def __getattr__(self, name):
        # 仅在常规属性查找失败时触发；用于代理 add_scalar / add_image 等所有方法
        def safe_call(*args, **kwargs):
            if self._writer is None:
                self._open()
            if self._writer is None:
                return None
            try:
                return getattr(self._writer, name)(*args, **kwargs)
            except Exception as e:
                self._warn(f"{name} failed, skip this write: {e!r}")
                self._reset_after_error()
                return None
        return safe_call

    def close(self):
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception as e:
            self._warn(f"close failed: {e!r}")
        finally:
            self._writer = None

    def flush(self):
        if self._writer is None:
            return
        try:
            self._writer.flush()
        except Exception as e:
            self._warn(f"flush failed: {e!r}")
            self._reset_after_error()


class EMALossTracker:
    def __init__(self, min_beta=0.01, max_beta=0.99, warmup_steps=100):
        self.metrics = {}
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.warmup_steps = warmup_steps
        self.step = 0
        self.utterance = {}
        
    def update(self, ds_key, loss, mode="ema"):
        if mode == "ema":
            # 计算动态 alpha
            if self.step < self.warmup_steps:
                beta = self.min_beta + (self.max_beta - self.min_beta) * (self.step / self.warmup_steps)
            else:
                beta = self.max_beta
                
            if ds_key not in self.metrics:
                self.metrics[ds_key] = loss
            else:
                self.metrics[ds_key] = beta * self.metrics[ds_key] + (1 - beta) * loss
        elif mode == "sum":
            if ds_key not in self.metrics:
                self.metrics[ds_key] = loss
                self.utterance[ds_key] = 1
            else:
                self.metrics[ds_key] += loss
                self.utterance[ds_key] += 1

def get_transfrom(args, is_training=True):
    if args.vision_encoder.startswith('eva'):
        std_mode = 'OPENAI_CLIP'
    else:
        std_mode = 'IMAGENET_INCEPTION'
    transform = utils.build_transform(
        is_train=is_training, randaug=args.img_aug, input_size=args.img_size, std_mode=std_mode,
        use_adaptive_slice=args.use_adaptive_slice
    )
    return transform

def get_item_builder(tokenizer, args, is_training=True):
    transform = get_transfrom(args, is_training)

    if args.use_adaptive_slice:
        adapt_slice_config = {
            'patch_size': args.patch_size,
            'max_slice_nums': args.max_slice_nums,
            'scale_resolution': args.scale_resolution,
            'floating_ratio': args.floating_ratio,
            'slice_new': args.slice_new
        }
        logger.info(f'user adapt_slice_config={adapt_slice_config}')
    else:
        adapt_slice_config = None

    if args.use_dynamic_batch:
        dynamic_batch_config = {
            'patch_size': args.patch_size,
        }
        logger.info(f'user dynamic_batch_config={dynamic_batch_config}')
    else:
        dynamic_batch_config = None

    video_frame_config = {
        'max_frame_nums': args.video_max_frame_nums,
        'max_slice_nums': args.video_max_slice_nums,
        'high_res_frame_nums': args.high_res_frame_nums,
        'fps': args.fps,
        'stack_frame_nums': args.stack_frame_nums,
    }

    builder = Qwen2ChatBuilder(
        tokenizer=tokenizer,
        max_len=args.max_len,
        transform=transform,
        query_len=args.query_num,
        min_resolution=args.min_resolution,
        skip_overlength=args.skip_overlength,
        skip_no_image = args.skip_no_image,
        use_system_prompt=False if args.no_system_prompt else True,
        adapt_slice_config=adapt_slice_config,
        dynamic_batch_config=dynamic_batch_config,
        aug_size=args.aug_size,
        use_image_id=args.use_image_id,
        new_schema=args.use_new_schema,
        video_frame_config=video_frame_config,
        enhance_ocr=args.enhance_ocr,
        model_type=args.model_type,
        time_stamp_train=args.time_stamp_train,
        mixed_downsample=args.mixed_downsample
    )
    return builder

def get_dataloader2(tokenizer, args, is_training=True, resume_dir=None, rank=None, world_size=None):
    if rank is None:
        rank, world_size = get_node_info()
    state_dict_file = None
    if resume_dir is not None:
        dataset_state_dict_files = glob.glob(resume_dir + '/*.pkl')
        if len(dataset_state_dict_files) != world_size:
            logger.error(f"nums of dataset_state_dict_files={len(dataset_state_dict_files)} not equal world_size={world_size}")
        else:
            rand2file = {int(f.split('.')[-2].split('_')[-1]): f for f in dataset_state_dict_files}
            state_dict_file = rand2file[rank]
            logger.info(f"rank={rank} load dataset state_dict from {state_dict_file}")

    if is_training:
        file_path = args.train_file
    else:
        file_path = args.eval_file
    
    builder = get_item_builder(tokenizer, args, is_training)

    logger.info(f'use parquet dataset with file={args.train_file}')
    collater = UnpadCollater(
        tokenizer,
        args.total_max_length,
        ncache=args.ncache,
        max_images=args.packing_max_images,
        force_batch=args.force_batch,
        batch_size=args.batch_size
    )
    dataloader = UnpadParquetDataloader(
        file_path,
        builder,
        collater,
        num_workers=args.num_workers,
        data_queue_size=args.data_queue_size,
        split_by_rank=args.split_by_rank,
        skip_files=args.skip_files,
        state_dict_path=state_dict_file,
        rank=rank,
        world_size=world_size,
        reverse_order=args.reverse_order
    )

    return dataloader


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}


def load_llm_tokenizer(args):
    return Qwen2TokenizerFastWrapper.from_pretrained(args.vocabs_path)


def build_ds_config(args):
    if args.precision == 'fp16':
        enable_fp16 = True
        enable_bf16 = False
    elif args.precision == 'bf16':
        enable_fp16 = False
        enable_bf16 = True
    elif args.precision == 'fp32':
        enable_fp16 = False
        enable_bf16 = False
    else:
        raise ValueError(f"Invalid precision {args.precision}")

    device = "cpu" if args.offload else "none"
    zero_opt_dict = {
        "stage": args.stage,
        "overlap_comm": args.overlap_comm,
        # "allgather_partitions": True,
        # "allgather_bucket_size": 5e8,
        # "reduce_scatter": True,
        # "reduce_bucket_size": 5e8,
        # "contiguous_gradients": True
    }

    # ---- ZeRO-3 调优开关 (仅 stage==3 时生效) ----
    # 设计: 默认行为完全保持原状; 仅在显式传 --zero3_tune 时, 把一组经过验证的
    # 稳健默认值注入 zero_opt_dict; 每项还可单独 override (传 0 表示用 preset)。
    # 这样可以走"先整体打开 → 出问题时单点回退"的安全验证路径。
    zero3_tune = getattr(args, 'zero3_tune', False)
    if zero3_tune and args.stage == 3:
        # A 类 (高性价比, 通用): overlap_comm + 显式 bucket
        zero_opt_dict["overlap_comm"] = True
        zero_opt_dict["contiguous_gradients"] = True
        zero_opt_dict["reduce_bucket_size"] = int(args.zero3_reduce_bucket_size or 5e8)
        zero_opt_dict["allgather_bucket_size"] = int(args.zero3_allgather_bucket_size or 5e8)
        # B 类 (针对 dummy-path / 动态 batch / 多模态多小参数 场景的稳定性补丁):
        #   - persistence_threshold 调高 → 大量小参数常驻不分片, 反向 hook 触发次数显著
        #     下降, 间接降低 collective 序列漂移概率 (与本仓库 zero3 分支正在追的
        #     dummy-path mismatch 直接相关)。
        #   - max_live_parameters 调大 → prefetch 更稳, gather 次数减少。
        #   - prefetch_bucket_size 显式 → 多模态 hidden 不统一时, 让 DS 自动推算
        #     不准, 显式更可控。
        zero_opt_dict["stage3_param_persistence_threshold"] = int(args.zero3_persistence_threshold or 1e6)
        zero_opt_dict["stage3_max_live_parameters"] = int(args.zero3_max_live_parameters or 2e9)
        zero_opt_dict["stage3_prefetch_bucket_size"] = int(args.zero3_prefetch_bucket_size or 1e8)
        # 保存 ckpt 时仍在 rank0 聚合 16bit 权重 (与训练流程兼容; 关闭会导致
        # 现有 save_checkpoint 取不到完整权重)。
        zero_opt_dict["stage3_gather_16bit_weights_on_model_save"] = True
        logger.info(f'[zero3_tune] enabled: overlap_comm=True, '
                    f'persistence_threshold={zero_opt_dict["stage3_param_persistence_threshold"]}, '
                    f'max_live_parameters={zero_opt_dict["stage3_max_live_parameters"]}, '
                    f'prefetch_bucket_size={zero_opt_dict["stage3_prefetch_bucket_size"]}, '
                    f'reduce_bucket_size={zero_opt_dict["reduce_bucket_size"]}, '
                    f'allgather_bucket_size={zero_opt_dict["allgather_bucket_size"]}')
    elif zero3_tune and args.stage != 3:
        logger.warning(f'[zero3_tune] requested but --stage={args.stage} (not 3), skip preset injection.')

    if args.offload:
        zero_opt_dict['round_robin_gradients'] = True
        zero_opt_dict['offload_param'] = {
            "device": device,
            "pin_memory": True
        }
        zero_opt_dict["offload_optimizer"] = {
            "device": device,
            "pin_memory": True
        }
        zero_opt_dict["overlap_comm"] = args.overlap_comm
        zero_opt_dict["allgather_partitions"] = True
        zero_opt_dict["allgather_bucket_size"] = 5e8
        zero_opt_dict["reduce_scatter"] = True
        zero_opt_dict["reduce_bucket_size"] = 5e8
        zero_opt_dict["contiguous_gradients"] = True


    # if args.offload and args.offload_ratio < 1.0:
    #     zero_opt_dict['offload_optimizer']['ratio'] = args.offload_ratio

    world_size = utils.get_world_size()
    if args.use_datapipe and not args.force_batch:
        gradient_accumulation_steps = args.gradient_accumulation_steps
    else:
        gradient_accumulation_steps = args.train_batch_size // (args.batch_size * world_size)
    logger.info(f'using train_micro_batch_size_per_gpu={args.batch_size}')
    logger.info(f'using train_batch_size_per_step={args.batch_size * world_size}')
    logger.info(f'using gradient_accumulation_steps={gradient_accumulation_steps}')
    logger.info(f'using train_batch_size={args.train_batch_size}')

    # bf16 段清理: 仅在 --zero3_tune 下生效, 保持存量训练行为完全不变。
    # 动机: bf16 模式下 DeepSpeed BF16_Optimizer 不维护 loss scale,
    #   - "initial_scale_power" 是死代码;
    #   - "auto_cast" 会对部分 fp32 张量做隐式 cast, 在 dummy-path 这种边界
    #     数据 (例如 vpm 的 dummy 输出参与 *0 桥梁) 上可能产生意外类型转换。
    # fp16 路径下两者仍是必需的, 因此不动 fp16。
    if zero3_tune and enable_bf16:
        bf16_cfg = {"enabled": True}
    else:
        bf16_cfg = {
            "enabled": enable_bf16,
            "initial_scale_power": 10,
            "auto_cast": True,
        }

    output = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "fp16": {
            "enabled": enable_fp16,
            "initial_scale_power": 10,
            "auto_cast": True
        },
        "bf16": bf16_cfg,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        # "activation_checkpointing": {
        #     "partition_activations": False,
        #     "cpu_checkpointing": True,
        #     "contiguous_memory_optimization": False,
        #     "number_checkpoints": None,
        #     "synchronize_checkpoint_boundary": False,
        #     "profile": True
        # },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [
                    0.9,
                    args.adam_beta2
                ],
                "weight_decay": 0.01
            }
        },
        # "flops_profiler": {
        #     "enabled": True,
        #     "profile_step": args.log_step,
        #     "detailed": True,
        # }
    }

    if  args.lr_scheduler == 'WarmupLR':
        logger.info('use WarmupLR')
        output.update({"scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": args.warmup_lr_init,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_t
            }
        }}
        )
    elif args.lr_scheduler == 'WarmupDecayLR':
        logger.info('use WarmupDecayLR')
        output.update({"scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": args.warmup_lr_init,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_t,
                "total_num_steps": args.t_initial
            }
        }}
        )

    return output


def get_scheduler(args, optimizer):
    if args.use_linear_decay_scheduler:
        logger.info(f'use LinearDecayLRScheduler')
        scheduler = LinearDecayLRScheduler(
            optimizer,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            warmup_t=args.warmup_t,
            warmup_lr_init=args.warmup_lr_init,
            warmup_prefix=True,
            t_in_epochs=False
        )
        return scheduler
    if args.use_wsd_scheduler:
        logger.info(f'use WSDLRScheduler (Warmup-Stable-Decay)')
        scheduler = WSDLRScheduler(
            optimizer,
            t_initial=args.t_initial,
            stable_t=args.stable_t,
            lr_min=args.lr_min,
            warmup_t=args.warmup_t,
            warmup_lr_init=args.warmup_lr_init,
            t_in_epochs=False
        )
        return scheduler
    if args.use_cosine_restart_scheduler:
        logger.info(f'use CosineLRScheduler')
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_mul=args.cycle_mul,
            cycle_decay=args.cycle_decay,
            cycle_limit=args.cycle_limit,
            warmup_t=args.warmup_t,
            warmup_lr_init=args.warmup_lr_init,
            warmup_prefix=True,
            t_in_epochs=False
        )
        return scheduler
    else:
        return None


def create_optimizer(model, args) -> "torch.optim.Optimizer":
    if args.vision_lr == 0:
        vision_lr = args.lr
    else:
        vision_lr = args.vision_lr

    if args.mup_lr_scale > 0:
        mup_lr_scale = args.mup_lr_scale
    else:
        mup_lr_scale = 1

    if args.vision_lr_scale > 0:
        vision_lr_scale = args.vision_lr_scale
    else:
        vision_lr_scale = 1

    weight_decay = 0.01
    adam_beta1 = 0.9
    adam_beta2 = args.adam_beta2

    param_groups = [
        ## vision params
        {'params': [], 'lr': args.lr, 'weight_decay': weight_decay, 'lr_scale': vision_lr_scale},
        {'params': [], 'lr': args.lr, 'weight_decay': 0, 'lr_scale': vision_lr_scale},

        ## llm params 
        {'params': [], 'lr': args.lr, 'weight_decay': weight_decay},
        {'params': [], 'lr': args.lr, 'weight_decay': 0}, # 可能是空的

        {'params': [], 'lr': args.lr, 'weight_decay': weight_decay, 'lr_scale': mup_lr_scale},
        {'params': [], 'lr': args.lr, 'weight_decay': 0, 'lr_scale': mup_lr_scale}, # 可能是空的
    ]

    def scale_lr_cond(name):
        if name.endswith(".weight") and "layer_norm" not in name and "layernorm" not in name \
            and "embedding" not in name and "output_layer" not in name and "lm_head" not in name \
            and "embed_tokens" not in name and "rotary_emb" not in name:
            return True
        else:
            return False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith('llm'):
            if scale_lr_cond(name):
                # embedding、layernorm、output_layer 的 lr 更大
                if "bias" in name: # bias 不需要 weight_decay
                    param_groups[5]['params'].append(param)
                else:
                    param_groups[4]['params'].append(param)
            else:
                if "bias" in name:
                    param_groups[3]['params'].append(param)
                else:
                    param_groups[2]['params'].append(param)
        else:
            if "bias" in name:
                param_groups[1]['params'].append(param)
            else:
                param_groups[0]['params'].append(param)

    adamw_args = {
        'lr': args.lr,
        'weight_decay': weight_decay,
        'betas': (adam_beta1, adam_beta2),
    }

    for i, param in enumerate(param_groups):
        logger.info(f'param_groups[{i}] lr={param["lr"]}, params={len(param["params"])}')

    optimizer = torch.optim.AdamW(param_groups, **adamw_args)

    return optimizer


def get_grad_norm(vllm_engine):
    norm = vllm_engine.get_global_grad_norm()
    if norm is None:
        norm = 0
    return norm


def track_loss(loss_tracker,
               cu_seqlens,
               ds_names,
               loss_wo_reduction,
               valid_mask,
               stage="Train"):
    for i in range(len(ds_names)):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        cur_valid_loss = loss_wo_reduction[start:end]
        cur_valid_loss = cur_valid_loss.sum() / valid_mask[start:end].sum().clamp(min=1)

        ds_name = ds_names[i]
        if stage == "Train":
            loss_tracker.update(f"{stage}/Loss_{ds_name}", cur_valid_loss.item(), mode="ema")
        else:
            loss_tracker.update(f"{stage}/Loss_{ds_name}", cur_valid_loss.item(), mode="sum")
        loss_tracker.step += 1

def train(vllm_model, args):
    vllm_model.train()
    logger.info(f"train args tune_resampler={args.tune_resampler}, tune_vision={args.tune_vision}, tune_llm={args.tune_llm}")
    logger.info(f"train args tune_llm_input={args.tune_llm_input}, tune_llm_head={args.tune_llm_head}, only_tune_special_head={args.only_tune_special_head}")

    if not args.tune_vision:
        vllm_model.vpm.requires_grad_(False)
    if not args.tune_resampler:
        vllm_model.resampler.requires_grad_(False)
        if getattr(vllm_model, "vit_merger", None) is not None:
            vllm_model.vit_merger.requires_grad_(False)
    if not args.tune_llm:
        vllm_model.llm.requires_grad_(False)

    if args.train_last_8layer:
        for layer in vllm_model.llm.model.layers[-8:]:
            layer.requires_grad_(True)

    if hasattr(vllm_model.vpm, 'embeddings'):
        logger.info('use navit siglip, always train position_embedding')
        vllm_model.vpm.embeddings.position_embedding.requires_grad_(True)
        if args.tune_patch_embedding:
            logger.info('tune siglip position_embedding')
            vllm_model.vpm.embeddings.patch_embedding.requires_grad_(True)

    if args.tune_llm_input:
        vllm_model.llm.model.embed_tokens.requires_grad_(True)

    if args.tune_llm_head:
        vllm_model.llm.lm_head.requires_grad_(True)

    tokenizer = load_llm_tokenizer(args)
    if utils.get_rank() == 0:
        logger.info(f'tokenizer={tokenizer}')

    if args.only_tune_special_head:
        vllm_model.llm.lm_head.requires_grad_(True)
        tie_word_embeddings = vllm_model.llm.config.tie_word_embeddings
        if not tie_word_embeddings:
            vllm_model.llm.model.embed_tokens.requires_grad_(True)
            # ZeRO-3: modifier_rank=0 保证 GatheredParameters 退出时由 rank 0 broadcast,
            # 避免依赖 "各 rank 写入完全一致" 的隐式约束.
            if args.stage==3:
                with deepspeed.zero.GatheredParameters([vllm_model.llm.model.embed_tokens.weight], modifier_rank=0):
                    orig_input_embeds_params = vllm_model.llm.model.embed_tokens.weight.data.clone()
            else:
                orig_input_embeds_params = vllm_model.llm.model.embed_tokens.weight.data.clone()

        if args.stage==3:
            with deepspeed.zero.GatheredParameters([vllm_model.llm.lm_head.weight], modifier_rank=0):
                orig_head_embeds_params = vllm_model.llm.lm_head.weight.data.clone()
        else:
            orig_head_embeds_params = vllm_model.llm.lm_head.weight.data.clone()

        n_size = orig_head_embeds_params.shape[0]
        # 与 lm_head/embedding 同 device, 避免每步 D2H/H2D
        index_no_updates = torch.ones((n_size,), dtype=torch.bool, device=orig_head_embeds_params.device)

        sp_tokens = ['<image>', '</image>', '<ref>', '</ref>', '<box>', '</box>',
                     '<quad>', '</quad>', '<point>', '</point>', '<slice>', '</slice>',
                     '<image_id>', '</image_id>']
        for t in sp_tokens:
            if t in tokenizer.get_vocab():
                index_no_updates[tokenizer.convert_tokens_to_ids(t)] = False

        # 注册梯度 mask hook: backward 时把 non-special 行的 grad 清零.
        # 在 ZeRO-3 下, 用户 register_hook 在 autograd 把 grad 写入 param.grad 时触发,
        # 此时 param 已 gathered 为完整形状, 我们能直接按行 mask, mask 后的 grad
        # 才会被 DeepSpeed reduce-scatter 到分片. 这样 Adam 的 m / v 在 non-special
        # 位置永远保持 0, 避免 "权重被 restore 但 optimizer state 漂移" 的问题.
        def _make_grad_mask_hook(mask_no_update):
            def _hook(grad):
                if grad is not None and grad.dim() >= 1 and grad.shape[0] == mask_no_update.shape[0]:
                    grad = grad.clone()
                    grad[mask_no_update] = 0
                return grad
            return _hook

        vllm_model.llm.lm_head.weight.register_hook(_make_grad_mask_hook(index_no_updates))
        if not tie_word_embeddings:
            vllm_model.llm.model.embed_tokens.weight.register_hook(_make_grad_mask_hook(index_no_updates))

        if utils.get_rank() == 0:
            n_special = int((~index_no_updates).sum().item())
            logger.info(f'only_tune_special_head: registered grad mask hook, '
                        f'special tokens = {n_special} / {n_size}')

    # vision
    if args.drop_vision_last_layer:
        if hasattr(vllm_model.vpm, 'norm'): # timm
            vllm_model.vpm.norm.requires_grad_(True)
        if hasattr(vllm_model.vpm, 'post_layernorm'): # hf
            vllm_model.vpm.post_layernorm.requires_grad_(True)

    if args.dynamic_deepspeed_config:
        # 不通过配置文件构建 deepspeed config
        args.deepspeed_config = None
        ds_config = build_ds_config(args)
    else:
        ds_config = None

    parameter_info = get_parameter_number(vllm_model)
    logger.info(f'parameter_info={parameter_info}')

    optim = create_optimizer(vllm_model, args)

    vllm_engine, vllm_optim, _, _ = deepspeed.initialize(
        args=args, model=vllm_model, model_parameters=vllm_model.parameters(), config=ds_config, optimizer=optim
    )
    # torch.cuda.synchronize()
    logger.info(f'rank={utils.get_rank()} load model successful')

    gradient_accumulation_steps = vllm_engine.gradient_accumulation_steps()

    scheduler = get_scheduler(args, vllm_optim)
    
    
    dataloader_func = get_dataloader2

    if args.train_file:
        dataloader_train = dataloader_func(tokenizer, args, resume_dir=args.dataset_resume_dir)
    if args.eval:
        dataloader_eval = dataloader_func(tokenizer, args, is_training=False, resume_dir=args.dataset_resume_dir)
    else:
        dataloader_eval = None
    logger.info(f'rank={utils.get_rank()} load dataloader successful')

    steps = args.global_start_step
    vllm_engine.global_steps = steps
    if steps > 0 and scheduler is not None:
        scheduler.step_update(num_updates=steps)

    log_loss = 0
    total_trained_samples = 0
    last_trained_samples = 0
    total_trained_tokens = 0      # 累计有效 token 数（valid, 即参与 loss 计算的 token）
    total_seq_tokens = 0          # 累计总 token 数（含 prompt/padding，反映实际计算量）
    last_trained_tokens = 0       # 用于计算 token 吞吐
    last_seq_tokens = 0           # 用于计算总 token 吞吐
    total_4x_images = 0           # 累计 4x 下采样图片数
    total_16x_images = 0          # 累计 16x 下采样图片数
    last_4x_images = 0
    last_16x_images = 0
    throughput_time = time.time()
    if args.need_resume:
        start = time.time()
        load_path, client_state = vllm_engine.load_checkpoint(args.deepspeed_resume_dir, tag=args.deepspeed_resume_tag)
        logger.info(f'Load pre-trained checkpoint from {load_path}, states: {client_state}')
        last_steps = client_state['checkpoint_step']
        args.start_epoch = client_state.get('epoch', args.start_epoch)
        logger.info(f'rank={utils.get_rank()} load grad successful')


        ### 只恢复 optimizer state dict
        # if args.model_checkpoint:
        #     logger.info(f'load model_checkpoint from {args.model_checkpoint}')
        #     state_dict = torch.load(args.model_checkpoint, map_location='cpu')
        #     info = vllm_engine.module.load_state_dict(state_dict, strict=False)
        #     logger.info(f"load model checkpoint info={info}" )

        #     del state_dict
        #     gc.collect()

    # init tensorboard writer
    if args.tensorboard is not None and utils.is_main_process():
        writer = SafeSummaryWriter(log_dir=args.tensorboard)
    else:
        writer = None

    last_steps = vllm_engine.global_steps
    # loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=-100)

    # 改成 reduction='none' 为了统计不同数据集 loss
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)

    # beta参数根据需要进行修改，beta越接近1，loss越平滑且滞后，反之loss曲线越波动且敏感
    train_ds_loss_tracker = EMALossTracker(min_beta=0, max_beta=0, warmup_steps=20)
    # 这里beta参数实际没用到


    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f'start epoch={epoch}')
        time_monitor = {}
        utils.collect_statsd_metric("init", time_monitor)
        if hasattr(dataloader_train, 'set_epoch'):
            dataloader_train.set_epoch(epoch)

        dataloader = CudaPrefetcher(dataloader_train)
        for step, batch in enumerate(dataloader):
            if step < args.skip_steps:
                logger.info(f'rank={utils.get_rank()} skip step={step}')
                if os.getenv('SKIP_STEP_SYNC', False):
                    torch.cuda.synchronize()
                continue

            # 不需要 length 和 context
            del batch['length']
            del batch['context']
            total_trained_samples += len(batch['raw_data'])

            if args.mixed_downsample > 0 and 'use_4x_downsample' in batch:
                _flags = batch['use_4x_downsample']
                total_4x_images += sum(1 for f in _flags if f)
                total_16x_images += sum(1 for f in _flags if not f)

            # batch = utils.convert_data_to_cuda(batch)
            utils.collect_statsd_metric('dataload', time_monitor)
            vllm_engine.zero_grad()

            output = vllm_engine(data=batch)

            logits = output.logits.view(-1, output.logits.shape[-1]).contiguous()
            target = batch['target'].view(-1).type(torch.long).contiguous()

            loss_wo_reduction = loss_fct(logits, target)

            valid_mask = (target != -100).type_as(loss_wo_reduction)
            loss_wo_reduction = loss_wo_reduction * valid_mask

            # 累计 token 统计
            batch_valid_tokens = int(valid_mask.sum().item())
            batch_total_tokens = target.numel()
            total_trained_tokens += batch_valid_tokens
            total_seq_tokens += batch_total_tokens

            if args.loss_reduction_weight > 0:
                # loss_reduction_weight=r: per-token weight = 1/L^r, per-sample weight = L^(1-r)
                # r=0: token-level (else 分支), r=0.5: sqrt, r=1: hard sample-level
                cu_seqlens = batch['cu_seqlens']
                num_samples = len(batch['raw_data'])
                weighted_losses = []
                local_weight_sum = torch.tensor(0.0, device=logits.device)
                for i in range(num_samples):
                    s = cu_seqlens[i].item()
                    e = cu_seqlens[i + 1].item()
                    sample_valid_cnt = valid_mask[s:e].sum().clamp(min=1)
                    avg_loss = loss_wo_reduction[s:e].sum() / sample_valid_cnt
                    # per-token weight = 1/L^r → per-sample weight = L * (1/L^r) = L^(1-r)
                    w = sample_valid_cnt.item() ** (1.0 - args.loss_reduction_weight)
                    weighted_losses.append(avg_loss * w)
                    local_weight_sum += w

                local_weighted_sum = torch.stack(weighted_losses).sum() if weighted_losses else torch.tensor(0.0, device=logits.device)
                global_weight_sum = local_weight_sum.clone()
                torch.distributed.all_reduce(global_weight_sum, op=torch.distributed.ReduceOp.SUM)
                # 乘 world_size 抵消 DeepSpeed 梯度 all-reduce 的均分
                loss = local_weighted_sum / global_weight_sum.clamp(min=1) * torch.distributed.get_world_size()
            else:
                # loss_reduction_weight=0: token-level loss
                loss = loss_wo_reduction.sum() / valid_mask.sum().clamp(min=1)

            utils.collect_statsd_metric("forward", time_monitor)

            vllm_engine.backward(loss)
            utils.collect_statsd_metric("backward", time_monitor)

            grad_norm = get_grad_norm(vllm_engine)

            vllm_engine.step()
            utils.collect_statsd_metric("optim", time_monitor)

            if args.only_tune_special_head:
                # 梯度 mask hook 已经把 non-special 位置的 grad / Adam m,v 钉死为 0,
                # 此处的 restore 主要修正 AdamW weight_decay 带来的微小漂移
                # (weight -= lr * wd * weight, 即使 grad=0 仍会缩小).
                with torch.no_grad():
                    if args.stage == 3:
                        # ZeRO-3: modifier_rank=0 -> 只允许 rank 0 修改, 退出 context
                        # 时 DeepSpeed broadcast rank 0 的全权重再 re-partition.
                        is_rank0 = (torch.distributed.get_rank() == 0)
                        with deepspeed.zero.GatheredParameters([vllm_model.llm.lm_head.weight], modifier_rank=0):
                            if is_rank0:
                                vllm_model.llm.lm_head.weight.data[index_no_updates] = orig_head_embeds_params[index_no_updates]
                        if not tie_word_embeddings:
                            with deepspeed.zero.GatheredParameters([vllm_model.llm.model.embed_tokens.weight], modifier_rank=0):
                                if is_rank0:
                                    vllm_model.llm.model.embed_tokens.weight.data[index_no_updates] = orig_input_embeds_params[index_no_updates]

                    else:
                        vllm_model.llm.lm_head.weight.data[index_no_updates] = orig_head_embeds_params[index_no_updates]
                        if not tie_word_embeddings:
                            vllm_model.llm.model.embed_tokens.weight.data[index_no_updates] = orig_input_embeds_params[index_no_updates]

            cost_info = f'dataload cost: {(time_monitor["dataload"] - time_monitor["init"]): .2f} ' \
                + f'forward cost {(time_monitor["forward"] - time_monitor["dataload"]): .2f} ' \
                + f'backward cost {(time_monitor["backward"] - time_monitor["forward"]): .2f} ' \
                + f'optim cost {(time_monitor["optim"] - time_monitor["backward"]): .2f}'

            log_loss += loss.item()

            update = False
            cur_steps = vllm_engine.global_steps
            if cur_steps > last_steps:
                update = True
                last_steps = cur_steps

            if update and scheduler is not None:
                scheduler.step_update(num_updates=cur_steps)

            avg_loss = utils.mean(utils.all_gather(loss.item()))
            if args.tensorboard is not None and update and utils.is_main_process():
                writer.add_scalar("Loss/train", avg_loss, cur_steps)
                writer.add_scalar("GradNorm/train", grad_norm, cur_steps)
                # 第一个是 vit lr
                vpm_lr = optim.param_groups[0]['lr']
                writer.add_scalar('LearningRate/vpm', vpm_lr, cur_steps)

                llm_lr = optim.param_groups[-1]['lr']
                writer.add_scalar('LearningRate/llm', llm_lr, cur_steps)

            # 记录训练数据集loss
            if args.tensorboard is not None \
                and args.log_train_ds_loss \
                and update:
                cu_seqlens = batch['cu_seqlens']

                try:
                    ds_names = [k.split('###')[0] for k in batch['keys']]
                except:
                    ds_names = []
                track_loss(train_ds_loss_tracker, cu_seqlens, ds_names, loss_wo_reduction, valid_mask, stage="Train")
                gathered_metrics = utils.all_gather(train_ds_loss_tracker.metrics)

                if utils.is_main_process():
                    merged_metrics = {}
                    for metrics in gathered_metrics:
                        for k, v in metrics.items():
                            if k not in merged_metrics:
                                merged_metrics[k] = []
                            merged_metrics[k].append(v)

                    for k in merged_metrics:
                        merged_metrics[k] = sum(merged_metrics[k]) / len(merged_metrics[k])
                    
                    for k, v in merged_metrics.items():
                        writer.add_scalar(k, v, cur_steps)

            if update and cur_steps % args.log_step == 0:
                avg_log_loss = utils.mean(utils.all_gather(log_loss))
                samples = sum(utils.all_gather(total_trained_samples))
                new_trained_samples = samples - last_trained_samples

                # 聚合 token 统计
                global_valid_tokens_all = sum(utils.all_gather(total_trained_tokens))
                global_seq_tokens_all = sum(utils.all_gather(total_seq_tokens))
                new_valid_tokens = global_valid_tokens_all - last_trained_tokens
                new_seq_tokens = global_seq_tokens_all - last_seq_tokens

                cur_time = time.time()
                elapsed = max(cur_time - throughput_time, 1e-6)
                throughput_sample = new_trained_samples / elapsed
                throughput_valid_tok = new_valid_tokens / elapsed      # 有效 token/s
                throughput_seq_tok = new_seq_tokens / elapsed          # 总 token/s（含 prompt/padding）

                throughput_time = cur_time
                last_trained_samples = samples
                last_trained_tokens = global_valid_tokens_all
                last_seq_tokens = global_seq_tokens_all

                if args.mixed_downsample > 0:
                    global_4x = sum(utils.all_gather(total_4x_images))
                    global_16x = sum(utils.all_gather(total_16x_images))
                    new_4x = global_4x - last_4x_images
                    new_16x = global_16x - last_16x_images
                    last_4x_images = global_4x
                    last_16x_images = global_16x
                    interval_total = new_4x + new_16x
                    ratio_4x = new_4x / interval_total if interval_total > 0 else 0.0
                    cum_total = global_4x + global_16x
                    cum_ratio = global_4x / cum_total if cum_total > 0 else 0.0

                if utils.is_main_process():
                    avg_loss = avg_log_loss / args.log_step / gradient_accumulation_steps
                    def _human_tokens(n):
                        if n >= 1e12:
                            return f'{n/1e12:.2f}T'
                        elif n >= 1e9:
                            return f'{n/1e9:.2f}B'
                        elif n >= 1e6:
                            return f'{n/1e6:.2f}M'
                        elif n >= 1e3:
                            return f'{n/1e3:.1f}K'
                        return str(n)

                    logger.info(
                        f'Datetime: {datetime.datetime.now()} '
                        f'Step: {vllm_engine.global_steps - args.log_step:6d}-{vllm_engine.global_steps:6d} | '
                        f'loss: {avg_loss:.4f} | '
                        f'Samples: {samples} ({throughput_sample:.1f} samples/s) | '
                        f'Tokens: {_human_tokens(global_valid_tokens_all)} valid / {_human_tokens(global_seq_tokens_all)} total | '
                        f'Throughput: {throughput_valid_tok:.0f} valid_tok/s, {throughput_seq_tok:.0f} seq_tok/s')
                    logger.info(f'time cost info {cost_info}')

                    if args.mixed_downsample > 0:
                        logger.info(
                            f'MixedDS | 4x: {new_4x}, 16x: {new_16x}, '
                            f'4x_ratio: {ratio_4x:.2%} | '
                            f'cumulative 4x: {global_4x}, 16x: {global_16x}, '
                            f'4x_ratio: {cum_ratio:.2%}')

                    if args.tensorboard is not None and writer is not None:
                        writer.add_scalar('Throughput/samples_per_sec', throughput_sample, cur_steps)
                        writer.add_scalar('Throughput/valid_tokens_per_sec', throughput_valid_tok, cur_steps)
                        writer.add_scalar('Throughput/seq_tokens_per_sec', throughput_seq_tok, cur_steps)
                        writer.add_scalar('Data/total_valid_tokens', global_valid_tokens_all, cur_steps)
                        writer.add_scalar('Data/total_seq_tokens', global_seq_tokens_all, cur_steps)
                        writer.add_scalar('Data/total_samples', samples, cur_steps)
                        if args.mixed_downsample > 0:
                            writer.add_scalar('MixedDS/4x_ratio', ratio_4x, cur_steps)
                            writer.add_scalar('MixedDS/4x_images', global_4x, cur_steps)
                            writer.add_scalar('MixedDS/16x_images', global_16x, cur_steps)

                log_loss = 0

            if update and cur_steps % args.save_step == 0:
                exporter.export(vllm_engine, dataloader_train, cur_steps, epoch, args, tokenizer=tokenizer, final_save=False)
                gc.collect()
                torch.cuda.empty_cache()
                # statsd.gauges('gen_model_checkpoint', int(cur_steps), repeat=20)

            if update and cur_steps % args.empty_cache_step == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if update and args.max_steps != 0 and cur_steps == args.max_steps:
                # max_steps 不是 save_step 整数倍时，确保最终模型被保存
                if cur_steps % args.save_step != 0:
                    exporter.export(vllm_engine, dataloader_train, cur_steps, epoch, args, tokenizer=tokenizer, final_save=False)
                    gc.collect()
                    torch.cuda.empty_cache()
                logger.info(f'Reached max_steps={args.max_steps}, training finished.')
                return

            # end step
            utils.collect_statsd_metric('init', time_monitor)

            # if args.eval and update and cur_steps % args.eval_step == 0:
            #     evaluate(vllm_model, tokenizer, dataloader_eval, cur_steps, args)
            #     vllm_model.train()

        # exporter.export(vllm_engine, dataloader_train, last_steps, epoch, args, tokenizer=tokenizer, final_save=False)


    # 最终模型
    # exporter.export(vllm_engine, dataloader_train, step, args.epochs-1, args, final_save=True)
    # statsd.gauges('gen_final_model', int(vllm_engine.global_steps), repeat=20)


def load_llm(args, use_hf=False):
    if args.precision == 'fp16':
        torch_dtype = torch.float16
    elif args.precision == 'bf16':
        torch_dtype = torch.bfloat16
   
    if args.qwen3:
        from multimodal_common.base_models.qwen3.modeling_qwen3_flash import Qwen3ForCausalLM
        cpm_model = Qwen3ForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, _attn_implementation = 'flash_attention_2')
    else:
        raise ValueError("load_llm: 未指定受支持的 LLM 类型，请传入 --qwen3")

    if args.llm_checkpoint and not args.model_checkpoint:
        logger.info(f'load checkpoint from {args.llm_checkpoint}')
        state_dict = torch.load(args.llm_checkpoint)
        # ZeRO-3 下 from_pretrained 已经 partition, 必须用 GatheredParameters 写回
        if args.stage == 3:
            maybe_zero3_load_state_dict(cpm_model, state_dict)
        else:
            cpm_model.load_state_dict(state_dict)
        del state_dict
        gc.collect()

    if args.llm_gradient_checkpointing:
        # 与原仓库 (multimodal-copy-2) 一致: 不显式传 use_reentrant, 走 transformers
        # 默认值。注意几个相互冲突的兼容性约束 (任何方向显式覆盖都会换一种报错):
        #   - ZeRO-2 + reentrant=True + 多入口 module (vpm/resampler 等)
        #     → DeepSpeed `params_already_reduced` (stage_1_and_2 backward hook 重复触发)
        #   - ZeRO-2 + reentrant=False + flash_attn varlen 内核
        #     → `_flash_attn_varlen_backward → CUDA illegal memory access`
        #       (flash-attention#341, q/k/v view 在 SavedTensor hooks 下被提前释放)
        #   - ZeRO-3 + reentrant=False
        #     → check_recomputed_tensors_match 因 partition shape=[0] 不匹配而 CheckpointError
        # transformers 在不同版本里默认值不同 (4.49- 默认 True, 4.50+ 默认 False),
        # 只要保持与训练用 transformers 版本默认一致, 与原仓库训练行为相同, 通常都能跑。
        # 如果遇到上述任一报错, 通常需要从环境层面 (transformers / flash_attn 版本) 调整,
        # 而不是在这里硬编码 use_reentrant 单边覆盖。
        cpm_model.gradient_checkpointing_enable()

    return cpm_model


def load_vpm(args):
    if args.vision_encoder != 'huggingface/siglip-so400m-14-980-flash-attn2-navit':
        raise NotImplementedError(
            "This minimal SFT repo only supports huggingface/siglip-so400m-14-980-flash-attn2-navit"
        )

    if args.precision == 'fp16':
        torch_dtype = torch.float16
        _attn_implementation = 'flash_attention_2'
    elif args.precision == 'bf16':
        torch_dtype = torch.bfloat16
        _attn_implementation = 'flash_attention_2'
    else:
        torch_dtype = torch.float32
        _attn_implementation = 'eager'

    from multimodal_common.base_models.modeling_navit_siglip_fast import SiglipVisionTransformer

    vpm_path = args.vpm_path or '/path/to/checkpoints/navit-siglip'
    model = SiglipVisionTransformer.from_pretrained(
        vpm_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        _attn_implementation=_attn_implementation,
    )
    if args.gradient_checkpointing:
        # 与原仓库一致: 不显式传 use_reentrant, 走 transformers 默认。详细兼容性
        # 约束见 load_llm 中 cpm_model.gradient_checkpointing_enable 注释。
        model.gradient_checkpointing_enable()

    logger.info('load fix navit')

    if args.drop_vision_last_layer:
        model.encoder.layers = model.encoder.layers[:-1]

    if hasattr(model, 'embeddings'):
        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)
    elif not hasattr(model, 'embed_dim'):
        setattr(model, 'embed_dim', model.config.hidden_size)
        if not isinstance(getattr(model, 'patch_size', None), int):
            setattr(model, 'patch_size', model.config.patch_size)

    return model


def setup_model(args, use_hf=False):
    start = time.time()

    # 确定 ZeRO-3 下的目标 dtype, 必须与 ds_config 中 bf16/fp16 严格一致,
    # 否则 _create_fp16_partitions_with_defragmentation 会因 mixed dtype 报 assert.
    if args.precision == 'fp16':
        z3_dtype = torch.float16
    elif args.precision == 'bf16':
        z3_dtype = torch.bfloat16
    else:
        z3_dtype = torch.float32

    if args.stage == 3:
        ds_config = build_ds_config(args)
        if utils.get_rank() == 0:
            logger.info(f'ds_config={ds_config}')
        try:
            from transformers.integrations import HfDeepSpeedConfig
        except ImportError:
            from transformers.deepspeed import HfDeepSpeedConfig
        hfdsc = HfDeepSpeedConfig(ds_config)


    llm = load_llm(args, use_hf)

    if args.stage == 3:
        # 把 vpm + VLU_SmartCPM 一起放进 zero.Init(dtype=z3_dtype):
        # - HF 系列 vpm 由 HfDeepSpeedConfig 处理 (此处无副作用)
        # - timm/PE 系列 vpm 在该上下文中创建参数, 才会被 ZeRO-3 partition
        # - VLU_SmartCPM 新增的 resampler/vit_merger 等参数也在此 partition
        # 显式传 dtype=z3_dtype 至关重要: 否则新建的 nn.Linear/LayerNorm 默认 fp32,
        # 与 llm/vpm 的 bf16 混在同一 trainable_param_group, defragment 会 assert.
        # HfDeepSpeedConfig 的 dtype 传播在不同 deepspeed/transformers 版本行为不一,
        # 不能依赖, 这里明确指定.
        with deepspeed.zero.Init(dtype=z3_dtype):
            vpm = load_vpm(args)
            vision_dim = vpm.embed_dim
            model = VLU_SmartCPM(llm, vpm, vision_dim, args.query_num, args.use_adaptive_slice,
                                 batch_vit=not args.no_batch_vit, no_grad_vit=args.no_grad_vit,
                                 model_type=args.model_type, insert_layer_id=args.insert_layer_id,
                                 mixed_downsample=args.mixed_downsample)

    else:
        vpm = load_vpm(args)
        vision_dim = vpm.embed_dim
        model = VLU_SmartCPM(llm, vpm, vision_dim, args.query_num, args.use_adaptive_slice,
                            batch_vit=not args.no_batch_vit, no_grad_vit=args.no_grad_vit,
                            model_type=args.model_type, insert_layer_id=args.insert_layer_id,
                            mixed_downsample=args.mixed_downsample)
    
    if args.gradient_checkpointing and hasattr(model.resampler, 'set_grad_checkpointing'):
        model.resampler.set_grad_checkpointing(True)

    # vit_merger.foreach: 与原仓库 (multimodal-copy-2) 默认值一致 (foreach=True 逐图)。
    # ZeRO-3 切换到 packed 路径 (foreach=False) 仅触发一次 all_gather, 配合
    # modeling_navit_siglip_fast.py 里的 checkpoint 包裹使用。其他 stage 保持原仓库默认。
    if (
        args.stage == 3
        and getattr(model, 'vit_merger', None) is not None
        and hasattr(model.vit_merger, 'foreach')
    ):
        model.vit_merger.foreach = False
        logger.info('vit_merger.foreach=False (ZeRO-3 packed path)')

    if args.vpm_checkpoint and not args.model_checkpoint:
        logger.info(f'load vpm_checkpoint from {args.vpm_checkpoint}')
        state_dict = torch.load(args.vpm_checkpoint, map_location='cpu')
        has_vit_merger_keys = any(k.startswith('vit_merger.') for k in state_dict.keys())
        has_vpm_prefix_keys = any(k.startswith('vpm.') for k in state_dict.keys())

        if has_vit_merger_keys or has_vpm_prefix_keys:
            filtered_state_dict = {k: v for k, v in state_dict.items()
                                   if k.startswith('vpm.') or k.startswith('vit_merger.')}
            logger.info(f"filtered {len(state_dict)} keys -> {len(filtered_state_dict)} keys (vpm + vit_merger only)")
            # ZeRO-3 下走 GatheredParameters 路径; 否则保持原行为
            if args.stage == 3:
                maybe_zero3_load_state_dict(model, filtered_state_dict)
                logger.info("load vpm+vit_merger checkpoint via maybe_zero3_load_state_dict")
            else:
                info = model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"load vpm+vit_merger checkpoint into full model, info={info}")
        else:
            if args.stage == 3:
                maybe_zero3_load_state_dict(model.vpm, state_dict)
                logger.info("load vpm checkpoint via maybe_zero3_load_state_dict")
            else:
                info = model.vpm.load_state_dict(state_dict, strict=False)
                logger.info(f"load vpm checkpoint, info={info}")

        del state_dict
        gc.collect()

    if args.model_checkpoint:
        logger.info(f'load model_checkpoint from {args.model_checkpoint}')
        state_dict = torch.load(args.model_checkpoint, map_location='cpu')
        if args.stage == 3:
            maybe_zero3_load_state_dict(model, state_dict)
        else:
            info = model.load_state_dict(state_dict, strict=False)
            logger.info(f"load model checkpoint info={info}" )

        del state_dict
        gc.collect()

    # ZeRO-3 下 dtype 已由 zero.Init(dtype=z3_dtype) 在参数创建时锁定,
    # model.half()/bfloat16() 走 nn.Module._apply 路径, 对已经 partition 的参数
    # 只能更新 param.data (空 placeholder), 无法可靠地更新 param.ds_tensor (真正的 shard),
    # 跳过它避免造成 dtype 不一致的假象 (实际上 shard dtype 没变).
    if args.stage != 3:
        if args.precision == 'fp16':
            model.half()
        elif args.precision == 'bf16':
            model.bfloat16()

    # ZeRO-3: 参数 shard 由 DeepSpeed 管理 (尤其开 offload_param 时位置应在 CPU),
    # 强制 .cuda() 会破坏 offload 状态并可能导致显存爆掉.
    if args.stage != 3:
        model.cuda()
    torch.cuda.empty_cache()
    return model


def maybe_zero3_load_state_dict(module: nn.Module, state_dict):
    from typing import List, Union
    from collections import OrderedDict

    def check_zero3_optimization(model):
        for name, param in model.named_parameters():
            if hasattr(param, 'ds_id'):
                return True
        return False
    if check_zero3_optimization(module):
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module: nn.Module, local_state_dict, prefix=""):
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    module._load_from_state_dict(local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)

            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)
        
        load(module, state_dict)
        del load
        del check_zero3_optimization
    else:
        module.load_state_dict(state_dict)
        del check_zero3_optimization


def main():
    args = initializer.get_args()
    # setup file and device
    initializer.setup(args)
    # load model
    model = setup_model(args)
    # train
    train(model, args)

    # 安全退出: 同步所有 rank 并销毁分布式通信组，确保 torchrun 返回 exit code 0
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    logger.info('All done, process group destroyed, exiting.')
    # 强制退出，跳过 atexit handler 和 __del__ 析构器
    # 避免 parquetdataset.kill_children() 和 ParallelWithBufferDataPipe.__del__
    # 在清理时破坏 NCCL 导致 SIGABRT (exitcode: -6)
    os._exit(0)


if __name__ == '__main__':
    main()
