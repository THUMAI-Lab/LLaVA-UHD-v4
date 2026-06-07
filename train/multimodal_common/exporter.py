from collections import OrderedDict
import os
import shutil
from multimodal_common.utils import utils
import traceback
import torch
import torch.distributed
import deepspeed
from deepspeed.utils import logger
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def export(vllm_engine, dataloader_train, global_step, epoch, args, tokenizer=None, final_save=False):
    job_id = os.getenv('JOB_ID', -1)
    export_model_dir = os.path.join(args.exp_ckpt_dir,  f'job_{job_id}_ckpt_{global_step}')
    
    os.makedirs(export_model_dir, exist_ok=True)
    base_file_name = f'{args.exp_name}_{global_step}'

    if args.save_deepspeed:
        logger.info(f'start to deepspped ckpt, save_dir={export_model_dir}')
        vllm_engine.save_checkpoint(save_dir=args.exp_ckpt_dir, tag=f'global_step{global_step}', client_state={
            'checkpoint_step': global_step, 'epoch': epoch})

    rank = utils.get_rank()
    # dataset state_dict. todo（ysj）如果 ann 需要使用，可能得改格式
    if not final_save and args.save_dataset:
        try:
            if hasattr(dataloader_train, 'save'): # sstable dataloader
                dataloader_train.save(os.path.basename(export_model_dir))
                logger.info(f'save sstable dataset ckpt to {export_model_dir}')
            else: # parquet dataloader
                os.makedirs(os.path.join(export_model_dir, 'dataloader_states'), exist_ok=True)
                dataset_state_dict_path = os.path.join(export_model_dir, 'dataloader_states', f'{base_file_name}_data_state_{rank}.pkl')
                dataloader_train.save_state_dict(dataset_state_dict_path)
                # 固定位置存一份，方便 resume
                # dataloader_train.save_state_dict(args.exp_data_statedict)
                logger.info(f'save dataset_state_dict_path to {dataset_state_dict_path}')
        except Exception as e:
            tbk = traceback.format_exc()
            logger.warn(f'fail to save dataset_state_dict: {e}\n{tbk}')

    model_state_dict_path = os.path.join(export_model_dir, base_file_name + '.pt')
    
    if args.stage == 3:
        # NOTE: 必须传 vllm_engine.module 而非 vllm_engine 本身, 否则
        # named_parameters/named_buffers 的 key 会带 'module.' 前缀,
        # 与 stage 0/1/2 路径 (vllm_engine.module.state_dict()) 不一致,
        # 导致后续 model.load_state_dict(strict=False) 静默全部 missing.
        output_state_dict = collect_state_dict(vllm_engine.module)

        if utils.is_main_process():
            torch.save(output_state_dict, model_state_dict_path)
        del output_state_dict
    
    else:
        if utils.is_main_process():
            torch.save(vllm_engine.module.state_dict(), model_state_dict_path)

    # model files
    if utils.is_main_process():
        # config 和 vocabs 和模型文件一起存储
        model_cfg_path = os.path.join(export_model_dir, 'config.json')
        model_vocab_path = os.path.join(export_model_dir, 'vocabs.txt')
        paths = [model_state_dict_path, model_cfg_path]
            
        if os.path.isfile(args.llm_path):
            shutil.copy(args.llm_path, model_cfg_path)
        else: # 生成空文件满足导出条件
            open(model_cfg_path, "w")
        paths.append(model_cfg_path)

        if os.path.isfile(args.vocabs_path):
            shutil.copy(args.vocabs_path, model_vocab_path)
            paths.append(model_vocab_path)
        # huggingface tokenizer
        elif tokenizer is not None and hasattr(tokenizer, 'save_pretrained'):
            tokenizer_paths = tokenizer.save_pretrained(export_model_dir)
            paths.extend(list(tokenizer_paths))
            open(model_vocab_path, "w")
        else:
            open(model_vocab_path, "w")
            paths.append(model_vocab_path)

        # 复制 tensorboard 日志到 ckpt 目录
        if args.tensorboard and os.path.isdir(args.tensorboard):
            tb_dst = os.path.join(export_model_dir, 'tensorboard')
            try:
                shutil.copytree(args.tensorboard, tb_dst, dirs_exist_ok=True)
                logger.info(f'copy tensorboard logs to {tb_dst}')
            except Exception as e:
                logger.warn(f'fail to copy tensorboard: {e}')

        logger.info(f'Successfully save model files!  {paths}')
    torch.distributed.barrier()


def export_eval_file(df, global_step, args):
    export_dir = '/data/checkpoints/pretrain_eval/'
    os.makedirs(export_dir, exist_ok=True)
    base_file_name = f'{"_".join(args.model_checkpoint.split("/")[-2:])}_{args.vision_encoder.split("_")[0]}_{global_step}'

    if utils.is_main_process():
        eval_result_path = os.path.join(export_dir, base_file_name + '.csv')
        logger.info(f'save eval result file to {eval_result_path}')
        df.to_csv(eval_result_path, index=False)



def collect_state_dict(model, gather_batch_size=64):
    """ZeRO-3 下收集完整 state_dict (parameters + buffers).

    重要:
    - ``model`` 必须是底层 ``nn.Module`` (例如 ``vllm_engine.module``), 不是 DeepSpeedEngine
      本身, 否则 ``named_parameters``/``named_buffers`` 返回的 key 会带 ``module.`` 前缀,
      与非 stage 3 路径产出的 state_dict 不一致, 后续 ``load_state_dict(strict=False)``
      会静默全部 missing.
    - 函数内每次 ``GatheredParameters`` 都是 collective op, 必须所有 rank 同步进入.
    - 仅在 rank 0 持有最终 state_dict (CPU), 减少非 rank0 的内存浪费.

    Args:
        model: 底层 ``nn.Module``.
        gather_batch_size: 一次 ``GatheredParameters`` 同时 gather 的参数个数, 越大越快但
            峰值显存越高. 64 经验上比较平衡.
    """
    def _z3_params_to_fetch(param_list):
        return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]

    state_dict = OrderedDict()
    is_rank0 = utils.is_main_process()

    # 1) parameters: 按 batch gather 以减少 NCCL collective 次数.
    named_params = list(model.named_parameters())
    for i in range(0, len(named_params), gather_batch_size):
        chunk = named_params[i:i + gather_batch_size]
        z3_params = _z3_params_to_fetch([v for _, v in chunk])
        # 即使 z3_params 为空, 上下文也是 no-op, 所有 rank 安全进入.
        with deepspeed.zero.GatheredParameters(z3_params, enabled=True):
            if is_rank0:
                for k, v in chunk:
                    state_dict[k] = v.data.detach().cpu()

    # 2) buffers: 各 rank 一致, 不需要 gather, 但要包含进 state_dict 以与
    #    nn.Module.state_dict() 行为一致 (RoPE inv_freq, BN running stats 等).
    if is_rank0:
        for k, b in model.named_buffers():
            state_dict[k] = b.detach().cpu()

    return state_dict