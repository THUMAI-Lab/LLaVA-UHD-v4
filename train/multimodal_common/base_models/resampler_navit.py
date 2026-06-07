from functools import partial
import numpy as np

import torch
from torch import nn
from torch.nn.init import trunc_normal_


def get_2d_sincos_pos_embed(embed_dim, image_size):
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size = image_size[0]
        grid_w_size = image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=-1)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    out = np.einsum("hw,d->hwd", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=-1)
    return emb


class Resampler(nn.Module):
    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)
        self.apply(self._init_weights)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)
        elif self.pos_embed.device != device:
            self.pos_embed = self.pos_embed.to(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, tgt_sizes=None):
        bs = tgt_sizes.shape[0]
        device = x.device
        dtype = x.dtype
        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)
        max_patch_len = torch.max(patch_len)

        if x.dim() == 3 and x.shape[0] == 1 and bs > 1:
            total_len = x.shape[1]
            expected_total = int(patch_len.sum().item())
            if expected_total != total_len:
                raise ValueError(f"Packed tokens length mismatch: expected {expected_total}, got {total_len}")
            seqs = []
            offset = 0
            for i in range(bs):
                n_i = int(patch_len[i].item())
                seqs.append(x[:, offset: offset + n_i, :].squeeze(0))
                offset += n_i
            x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
        elif x.dim() == 3 and x.shape[0] == bs and x.shape[1] != int(max_patch_len.item()):
            seqs = [x[i, : int(patch_len[i].item()), :] for i in range(bs)]
            x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)

        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(
                self.pos_embed[:tgt_h, :tgt_w, :]
                .reshape((tgt_h * tgt_w, -1))
                .to(device=device, dtype=dtype)
            )
            key_padding_mask[i, patch_len[i]:] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed,
            batch_first=True,
            padding_value=0.0,
        ).permute(1, 0, 2)

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)
        q = self.ln_q(self.query)

        out = self.attn(
            self._repeat(q, bs),
            x + pos_embed,
            x,
            key_padding_mask=key_padding_mask,
        )[0]

        x = out.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, num_repeats: int):
        return query.unsqueeze(1).repeat(1, num_repeats, 1)
