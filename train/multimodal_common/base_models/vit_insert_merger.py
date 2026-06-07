import contextlib

import torch
from torch import nn
from typing import Iterable, List, Tuple
import torch.nn.functional as F
from transformers.activations import ACT2FN

from multimodal_common.base_models.modeling_navit_siglip_fast import SiglipAttention, SiglipFlashAttention2, SiglipMLP


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm 内部强制 fp32 计算, 输出 cast 回 input dtype.

    解决 packed-batch 场景下 (sum_tokens × hidden, hidden 高达 4×embed_dim) 在 bf16
    下做 mean/var reduction 时, 求和阶段精度坍塌 → 1/sqrt(var) 偶发 inf/NaN 的问题。

    问题根因 (commit 9326972a 引入):
      原 forward 把 batch 内每张图分别送进 `pre_norm/linear_1/linear_2` (per-image
      loop), bf16 mean/var 是在单图 token 数 × hidden 上 reduce, 量级可控;
      改造为 ZeRO-3 友好的 packed forward 后, mean/var 直接在 sum(N_img) × hidden
      上 reduce, 大数 + bf16 7-bit 尾数下偶发出现 var≈0 → 1/sqrt(var)→inf,
      下游 GEMM 即变成 NaN. 这与具体 batch 内容相关, 因此每次都在固定 step 复现.

    ZeRO-3 兼容性:
      仍然通过 `module.__call__` 路径调用 forward, DeepSpeed 的
      `_pre_forward_module_hook` 会在 forward 调用前 all_gather weight/bias 成
      full shape, forward 内部访问 self.weight 拿到的就是完整 bf16 张量,
      `.float()` 后变成完整 fp32 张量, 不会触发额外 collective.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        out = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.to(in_dtype)


def _gather_modules_ctx(modules: Iterable[nn.Module], modifier_rank=0):
    """ZeRO-3 友好: 聚合给定 modules 下所有 parameters, 让它们暂时回到完整 shape.

    - 若任意参数携带 ``ds_id``, 说明在 ZeRO-3 partition 状态, 用
      ``deepspeed.zero.GatheredParameters`` 包起来; ``modifier_rank=0`` 让
      rank 0 的写入在退出时被广播到全部 rank.
    - 否则返回 ``nullcontext``, stage 0/1/2 行为完全不变.

    需要这个的典型场景: 在 ``deepspeed.zero.Init()`` 内构造模块, 之后又要
    用 ``param.shape`` (此时是 1D partition 形状) 或读 ``param.data`` 的
    原始内容做权重拷贝/初始化.
    """
    params: List[torch.nn.Parameter] = []
    for m in modules:
        params.extend(p for p in m.parameters(recurse=True))
    if any(hasattr(p, 'ds_id') for p in params):
        import deepspeed  # 延迟 import: stage 0/1/2 不强依赖 deepspeed
        return deepspeed.zero.GatheredParameters(params, modifier_rank=modifier_rank)
    return contextlib.nullcontext()


def get_vit_insert_merger(model_type, hidden_size, intermediate_size, vpm, insert_layer_id):
    if model_type != "uhd_mlp_insert_window_attention_ViTmlp_4_4":
        raise NotImplementedError(f"Only uhd_mlp_insert_window_attention_ViTmlp_4_4 is supported, got {model_type}")
    return ViTWindowAttentionMerger(vpm, insert_layer_id, downsample='ViTmlp', foreach=True)

class ViTWindowAttentionMerger(nn.Module):
    def __init__(self, vpm, insert_layer_id, downsample=None, foreach=False, deepstack=False):
        super().__init__()
        self.window_kernel_size = (2, 2)
        assert downsample in ['average', 'ViTmlp', 'ViTmlp_only', 'self_attention_ViTmlp', 'mlp', None], f"Unknown downsample way: {downsample}"
        self.downsample = downsample
        self.foreach = foreach
        self.deepstack = deepstack
        if self.deepstack:
            assert self.downsample == 'ViTmlp'
            assert self.foreach == True

        self.embed_dim = vpm.config.hidden_size
        self._use_flash_attention_2 = vpm.config._attn_implementation == "flash_attention_2"
        self.self_attn = (
            SiglipAttention(vpm.config)
            if not self._use_flash_attention_2
            else SiglipFlashAttention2(vpm.config)
        )
        self.layer_norm1 = Fp32LayerNorm(self.embed_dim, eps=vpm.config.layer_norm_eps)
        self.mlp = SiglipMLP(vpm.config)
        self.layer_norm2 = Fp32LayerNorm(self.embed_dim, eps=vpm.config.layer_norm_eps)

        if self.downsample in ['ViTmlp', 'ViTmlp_only', 'self_attention_ViTmlp']:
            self.hidden_size = (
                self.embed_dim
                * self.window_kernel_size[0]
                * self.window_kernel_size[1]
            )

            self.intermediate_size = (
                vpm.config.intermediate_size
                * self.window_kernel_size[0]
                * self.window_kernel_size[1]
            )

            self.pre_norm = Fp32LayerNorm(self.hidden_size, eps=1e-6)
            self.linear_1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
            self.act = ACT2FN["gelu_pytorch_tanh"]
            self.linear_2 = nn.Linear(
                self.intermediate_size, self.embed_dim, bias=True
            )
        elif self.downsample in ['mlp']:
            self.hidden_size = (
                self.embed_dim
                * self.window_kernel_size[0]
                * self.window_kernel_size[1]
            )

            # Fp32LayerNorm: 同上, packed 大维度 LN, bf16 下精度风险
            self.pre_norm = Fp32LayerNorm(self.hidden_size, eps=1e-6)
            self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.act = nn.GELU()
            self.linear_2 = nn.Linear(
                self.hidden_size, self.embed_dim, bias=True
            )

            # average pooling初始化 (ZeRO-3 下需要 gather 才能 eye 拷贝)
            with _gather_modules_ctx([self.linear_1, self.linear_2, self.pre_norm]), torch.no_grad():
                self.linear_1.weight.data.copy_(torch.eye(self.hidden_size))
                self.linear_1.bias.data.zero_()
                custom_weight = torch.zeros(self.embed_dim, self.hidden_size)
                for i in range(self.embed_dim):
                    for j in range(4):
                        custom_weight[i, i + j * self.embed_dim] = 1.0 / 4.0
                self.linear_2.weight.data.copy_(custom_weight)
                self.linear_2.bias.data.zero_()

                self.pre_norm.bias.data.zero_()
                self.pre_norm.weight.data.fill_(1.0)

        self._init_weight(vpm, insert_layer_id)

    def _init_weight(self, vpm, insert_layer_id):
        copy_block = vpm.encoder.layers[insert_layer_id]

        gather_modules = [
            copy_block.self_attn, copy_block.layer_norm1,
            copy_block.layer_norm2, copy_block.mlp,
            self.self_attn, self.layer_norm1, self.layer_norm2, self.mlp,
        ]
        if self.downsample in ['ViTmlp', 'ViTmlp_only', 'self_attention_ViTmlp']:
            gather_modules.extend([self.linear_1, self.linear_2, self.pre_norm])

        with _gather_modules_ctx(gather_modules), torch.no_grad():
            # 拷贝 self-attention
            for target_module, src_module in [
                (self.self_attn, copy_block.self_attn),
                (self.layer_norm1, copy_block.layer_norm1),
                (self.layer_norm2, copy_block.layer_norm2),
                (self.mlp, copy_block.mlp),
            ]:
                target_state = target_module.state_dict()
                src_state = src_module.state_dict()
                for k, v in src_state.items():
                    if k in target_state and v.shape == target_state[k].shape:
                        target_state[k].copy_(v)
                target_module.load_state_dict(target_state)

            if self.downsample in ['ViTmlp', 'ViTmlp_only', 'self_attention_ViTmlp']:
                fc1_old = copy_block.mlp.fc1       # [intermediate, hidden]
                fc2_old = copy_block.mlp.fc2       # [hidden, intermediate]

                # -------------------------------
                # fc1_new: Linear(hidden*4, inter*4)
                # 形成 block diagonal (4个 fc1)
                # -------------------------------
                hidden = fc1_old.weight.shape[1]
                inter = fc1_old.weight.shape[0]

                fc1_blocks = [fc1_old.weight.data] * 4  # list of [inter, hidden]
                w_fc1_new = torch.zeros(inter * 4, hidden * 4, device=fc1_old.weight.device)
                for i in range(4):
                    w_fc1_new[i*inter:(i+1)*inter, i*hidden:(i+1)*hidden] = fc1_blocks[i]
                
                b_fc1_new = fc1_old.bias.data.repeat(4)  # [inter*4]

                self.linear_1.weight.copy_(w_fc1_new)
                self.linear_1.bias.copy_(b_fc1_new)

                # -------------------------------
                # fc2_new: Linear(inter*4, hidden)
                # 水平拼接 4 份 + 除以 4
                # -------------------------------
                w_fc2_new = torch.cat([fc2_old.weight.data] * 4, dim=1) / 4.0  # [hidden, inter*4]
                b_fc2_new = fc2_old.bias.data  # [hidden]

                self.linear_2.weight.copy_(w_fc2_new)
                self.linear_2.bias.copy_(b_fc2_new)

                # -------------------------------
                # pre_norm
                # -------------------------------
                self.pre_norm.weight.data.copy_(
                    copy_block.layer_norm2.weight.data.repeat(4)
                )
                self.pre_norm.bias.data.copy_(
                    copy_block.layer_norm2.bias.data.repeat(4)
                )

    def get_window_index(self, tgt_sizes):
        """
        tgt_sizes: list or tensor of (H, W)
        return:
            window_index: Tensor[total_tokens] -> 按 window 顺序排列的 token 索引
            cu_seqlens: Tensor[num_windows + 1]
        """
        window_h, window_w = self.window_kernel_size  # e.g. (2, 2)
        max_seqlens = window_h * window_w

        window_index_list = []
        cu_seqlens = [0]
        token_offset = 0  # 必须保持 python int, 否则下面 `index + token_offset` 会跨 device

        # ZeRO-3 / cuda 安全: tgt_sizes 在训练里是 cuda int32 张量, 直接 `for (H, W) in tgt_sizes`
        # 解出来的 H, W 是 cuda 0-d tensor, 第 1 轮 `token_offset += H * W` 之后
        # token_offset 就变成了 cuda 0-d tensor; 而 `index = torch.arange(H*W)` 默认在 CPU,
        # 第 2 轮 `index.reshape(-1) + token_offset` 就会报 "cuda:X and cpu" device mismatch。
        # get_window_index 只是做小整数 indexing, 全程放在 CPU 上算最省事, caller 再 .to(device)。
        for (H, W) in tgt_sizes:
            H = int(H)
            W = int(W)
            assert H % window_h == 0 and W % window_w == 0, \
                f"H={H}, W={W} must be divisible by window size ({window_h}, {window_w})"

            index = torch.arange(H * W).reshape(H, W)

            num_windows_h = H // window_h
            num_windows_w = W // window_w
            num_windows = num_windows_h * num_windows_w

            index = index.reshape(num_windows_h, window_h, num_windows_w, window_w)
            index = index.permute(0, 2, 1, 3).reshape(num_windows, window_h * window_w)

            index_flat = index.reshape(-1) + token_offset
            window_index_list.append(index_flat)

            window_token_count = window_h * window_w
            cu_this = torch.arange(1, num_windows + 1) * window_token_count + cu_seqlens[-1]
            cu_seqlens.extend(cu_this.tolist())

            token_offset += H * W      # 偏移 (python int)

        window_index = torch.cat(window_index_list)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        return window_index, cu_seqlens, max_seqlens

    def forward(
        self,
        hidden_states: torch.Tensor,
        tgt_sizes: torch.IntTensor, 
        attention_mask: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlens: torch.Tensor = None,
    ) -> Tuple[torch.FloatTensor]:
        
        if self.downsample == 'self_attention_ViTmlp':
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)

            hidden_states, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlens=max_seqlens,
                tgt_sizes=tgt_sizes,
            )
            hidden_states = residual + hidden_states
        elif self.downsample != 'ViTmlp_only':
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            device = hidden_states.device

            if self.foreach:
                # 逐图路径: 与原仓库 (minicpm-v-uhd-wcy-fix-linear-attn) 行为一致。
                # 每张图单独做 2×2 window self-attention, 不依赖全局 packed cu_seqlens,
                # 对 ZeRO-2 (use_reentrant=False) 和 ZeRO-3 都安全。
                # ZeRO-3 下每张图会触发一次 q/k/v/out_proj all_gather; 在动态分桶
                # (--use_dynamic_batch + --split_by_rank) 时各 rank batch_size
                # 可能不同 → 需 ZeRO-3 packed 路径。对 ZeRO-2 无此约束, 始终安全。
                all_pixel_values = []
                batch_size, _ = tgt_sizes.shape
                for batch_idx in range(batch_size):
                    hidden_state = hidden_states[0, cu_seqlens[batch_idx]:cu_seqlens[batch_idx+1], :].unsqueeze(0)
                    tgt_size = tgt_sizes[batch_idx].unsqueeze(0)

                    window_index, window_cu_seqlens, window_max_seqlens = self.get_window_index(tgt_size)
                    window_index = window_index.to(device)
                    hidden_state = hidden_state[:, window_index, :]

                    hidden_state, _ = self.self_attn(
                        hidden_states=hidden_state,
                        attention_mask=attention_mask,
                        cu_seqlens=window_cu_seqlens.to(device),
                        max_seqlens=window_max_seqlens,
                        tgt_sizes=tgt_size,
                    )
                    all_pixel_values.append(hidden_state[:, torch.argsort(window_index), :])

                hidden_states = torch.concat(all_pixel_values, dim=1)
                hidden_states = residual + hidden_states
            else:
                # packed 路径: 全图一次性做 window self-attention, 仅触发一次 all_gather。
                # 适用于 ZeRO-3 (use_reentrant=True) 场景。
                # 注意: ZeRO-2 下 use_reentrant=False 的 gradient checkpoint 会破坏
                # int32 tgt_sizes, 请勿在 ZeRO-2 下使用 foreach=False。
                window_index, window_cu_seqlens, window_max_seqlens = self.get_window_index(tgt_sizes)
                window_index = window_index.to(device)
                hidden_states = hidden_states[:, window_index, :]
                hidden_states, attn_weights = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    cu_seqlens=window_cu_seqlens.to(device),
                    max_seqlens=window_max_seqlens,
                    tgt_sizes=tgt_sizes,
                )
                hidden_states = hidden_states[:, torch.argsort(window_index), :]
                hidden_states = residual + hidden_states

        if self.downsample == 'average':
            batch_size, _ = tgt_sizes.shape
            all_pixel_values = []
            new_tgt_sizes = torch.zeros_like(tgt_sizes, dtype=tgt_sizes.dtype, device=tgt_sizes.device)

            m1, m2 = self.window_kernel_size
            for batch_idx in range(batch_size):
                h, w = tgt_sizes[batch_idx]
                assert h % 2 == 0 and w % 2 == 0, "patch尺寸不能被2整除, 无法拼接4个相邻patch"
                from einops import rearrange
                hidden_state = rearrange(hidden_states[0, cu_seqlens[batch_idx]:cu_seqlens[batch_idx+1], :].squeeze(0), "(h p1 w p2) d -> (h w) (p1 p2) d", h=h // m1, p1=m1, w=w // m2, p2=m2)

                hidden_state = hidden_state.mean(dim=1)  # (num_windows, D)
                
                all_pixel_values.append(hidden_state)
                new_tgt_sizes[batch_idx, :2] = torch.tensor([h // 2, w // 2], device=new_tgt_sizes.device, dtype=new_tgt_sizes.dtype)

            new_hidden_states = torch.concat(all_pixel_values, dim=0).unsqueeze(0)
            new_cu_seqlens = F.pad(torch.cumsum(new_tgt_sizes[:, 0] * new_tgt_sizes[:, 1], dim=0, dtype=torch.int32).cuda(), (1, 0))
            assert max_seqlens % 4 == 0
            new_max_seqlens = max_seqlens // 4

            return new_hidden_states, new_tgt_sizes, attention_mask, new_cu_seqlens, new_max_seqlens   # attention_mask=None
        elif self.downsample in ['ViTmlp', 'ViTmlp_only', 'self_attention_ViTmlp', 'mlp']:
            # ZeRO-3 友好改造: 原代码对 batch 内的每张图依次过
            # pre_norm/linear_1/act/linear_2, 每张图都会触发一次这些参数的
            # all_gather; 动态分桶下不同 rank 的图数不同 → collective
            # fingerprint mismatch。
            #
            # 解决: pre_norm/linear_1/linear_2 都是 token-wise 操作, 跨图
            # cat 一起送一次, 与逐图分别送出来的结果完全一致 (LayerNorm 在
            # 最后一维上做 normalize, Linear 也只看最后一维), 数学等价。
            # 这样这几个 Linear/LN 在一次 forward 内只触发 1 次 all_gather。
            from einops import rearrange
            batch_size, _ = tgt_sizes.shape
            new_tgt_sizes = torch.zeros_like(tgt_sizes, dtype=tgt_sizes.dtype, device=tgt_sizes.device)

            m1, m2 = self.window_kernel_size

            # ----- 第 1 步: 仅做无参数的 view/rearrange, 收集每张图的输入与 residual -----
            packed_in_list = []        # 送进 pre_norm/linear 的 token, 形状 [h*w/4, p1*p2*d]
            packed_residual_list = []  # 4 个相邻 patch 的均值 residual, 形状 [h*w/4, d]
            deepstack_local_list = []  # 仅 deepstack 时使用, 形状 [h*w/4, p1*p2, d]

            for batch_idx in range(batch_size):
                h, w = tgt_sizes[batch_idx]
                assert h % 2 == 0 and w % 2 == 0, "patch尺寸不能被2整除, 无法拼接4个相邻patch"

                token_slice = hidden_states[0, cu_seqlens[batch_idx]:cu_seqlens[batch_idx+1], :]  # [h*w, d]

                # 主路径输入: 4 个相邻 patch 拼成 (p1*p2*d) 的大向量
                packed_in_list.append(
                    rearrange(token_slice, "(h p1 w p2) d -> (h w) (p1 p2 d)",
                              h=h // m1, p1=m1, w=w // m2, p2=m2)
                )

                # residual: 把 4 个相邻 patch 在 d 维上保持, 在 patch 维上求均值
                local_p1p2 = rearrange(token_slice, "(h p1 w p2) d -> (h w) (p1 p2) d",
                                       h=h // m1, p1=m1, w=w // m2, p2=m2)  # [h*w/4, p1*p2, d]
                packed_residual_list.append(local_p1p2.mean(dim=1))         # [h*w/4, d]

                if self.deepstack:
                    deepstack_local_list.append(local_p1p2)

                new_tgt_sizes[batch_idx, :2] = torch.tensor([h // 2, w // 2],
                                                            device=new_tgt_sizes.device,
                                                            dtype=new_tgt_sizes.dtype)

            # ----- 第 2 步: 跨图 packed, 一次性过参数模块 (单次 all_gather) -----
            packed_in = torch.cat(packed_in_list, dim=0)            # [sum(h*w/4), p1*p2*d]
            packed_out = self.pre_norm(packed_in)
            packed_out = self.linear_1(packed_out)
            packed_out = self.act(packed_out)
            packed_out = self.linear_2(packed_out)                  # [sum(h*w/4), embed_dim]

            packed_residual = torch.cat(packed_residual_list, dim=0)  # [sum(h*w/4), embed_dim]
            new_hidden_states = (packed_out + packed_residual).unsqueeze(0)

            new_cu_seqlens = F.pad(torch.cumsum(new_tgt_sizes[:, 0] * new_tgt_sizes[:, 1], dim=0, dtype=torch.int32).cuda(), (1, 0))
            assert max_seqlens % 4 == 0
            new_max_seqlens = max_seqlens // 4

            if self.deepstack:
                # deepstack: 把每张图的 i-th 子 patch (i in 0..3) 跨图 cat 在一起,
                # 与原实现的 token 顺序完全一致 (按 batch_idx 顺序拼接)。
                deepstack_residual = []
                for i in range(4):
                    deepstack_residual.append(
                        torch.concat([t[:, i, :] for t in deepstack_local_list], dim=0).unsqueeze(0)
                    )
                return new_hidden_states, new_tgt_sizes, attention_mask, new_cu_seqlens, new_max_seqlens, deepstack_residual   # attention_mask=None
            else:
                return new_hidden_states, new_tgt_sizes, attention_mask, new_cu_seqlens, new_max_seqlens   # attention_mask=None
        else:    # no Downsample
            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states, tgt_sizes, attention_mask, cu_seqlens, max_seqlens   # attention_mask=None
