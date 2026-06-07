from functools import partial
import numpy as np

import torch
from torch import nn
from torch.nn.init import trunc_normal_
from typing import Tuple
from einops import rearrange

from multimodal_common.base_models.vit_insert_merger import Fp32LayerNorm


class DownsampleMLP(nn.Module):
    def __init__(self, hidden_size, llm_embed_dim, merge_kernel_size=(2, 2)):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size

        self.hidden_size = (
            hidden_size
            * self.merge_kernel_size[0]
            * self.merge_kernel_size[1]
        )

        self.pre_norm = Fp32LayerNorm(self.hidden_size, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, llm_embed_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.pre_norm(x).view(-1, self.hidden_size))
        return x


class Merger(nn.Module):
    def __init__(self, hidden_size, llm_embed_dim, merge_kernel_size=(2, 2), times=1):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.times = times
        self.mlp = nn.ModuleList([DownsampleMLP(hidden_size, llm_embed_dim if i==times-1 else hidden_size, merge_kernel_size) for i in range(times)])


    def forward(self, hidden_states: torch.Tensor, tgt_sizes: torch.IntTensor,) -> torch.Tensor:
        m1, m2 = self.merge_kernel_size

        # ----- 第 1 步: 收集每张图的 rearrange 输入 (无参数操作), 同时记录每图当前 h, w -----
        start = 0
        rearr_list = []
        h_list = []   # 每图过完 mlp[0] 后的 h (后续 rearrange 用)
        w_list = []   # 每图过完 mlp[0] 后的 w
        for batch_idx in range(len(tgt_sizes)):
            h = int(tgt_sizes[batch_idx][0])
            w = int(tgt_sizes[batch_idx][1])
            assert h % m1 == 0 and w % m2 == 0, \
                f"h={h}, w={w} 必须能被 merge_kernel_size={self.merge_kernel_size} 整除"
            num_patches = h * w

            _hidden_state = rearrange(
                hidden_states[0, start: start + num_patches, :],
                "(h p1 w p2) d -> (h w) (p1 p2 d)",
                h=h // m1, p1=m1, w=w // m2, p2=m2,
            )
            rearr_list.append(_hidden_state)
            h_list.append(h // m1)
            w_list.append(w // m2)
            start += num_patches

        # ----- 第 2 步: 跨图 packed, 一次过 mlp[0] (单次 all_gather) -----
        counts = [t.shape[0] for t in rearr_list]
        packed = torch.cat(rearr_list, dim=0)        # [sum(h*w/(m1*m2)), p1*p2*d]
        packed = self.mlp[0](packed)                  # [sum(h*w/(m1*m2)), out_dim]

        # ----- 第 3 步: 后续 mlp[i] (times > 1) -----
        # 后续每层之间需要"按图独立做 rearrange (无参数)" + "跨图 packed 过 mlp (有参数)"
        for i in range(1, self.times):
            chunks = list(torch.split(packed, counts, dim=0))
            new_chunks = []
            new_counts = []
            new_h_list = []
            new_w_list = []
            for j, chunk in enumerate(chunks):
                h = h_list[j]
                w = w_list[j]
                assert h % m1 == 0 and w % m2 == 0, \
                    f"层 {i}: h={h}, w={w} 必须能被 merge_kernel_size={self.merge_kernel_size} 整除"
                chunk = rearrange(
                    chunk,
                    "(h p1 w p2) d -> (h w) (p1 p2 d)",
                    h=h // m1, p1=m1, w=w // m2, p2=m2,
                )
                new_chunks.append(chunk)
                new_counts.append(chunk.shape[0])
                new_h_list.append(h // m1)
                new_w_list.append(w // m2)
            packed = torch.cat(new_chunks, dim=0)
            packed = self.mlp[i](packed)
            counts = new_counts
            h_list = new_h_list
            w_list = new_w_list

        return packed, counts