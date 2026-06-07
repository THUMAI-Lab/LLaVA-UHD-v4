#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2026 modelbest
#
# @date: 2026
#
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput

from multimodal_common.base_models.mlp_merger import Merger
from multimodal_common.base_models.resampler_navit import Resampler
from multimodal_common.base_models.vit_insert_merger import get_vit_insert_merger


MODEL_TYPE = "resampler"
UHD_MODEL_TYPE = "uhd_mlp_insert_window_attention_ViTmlp_4_4"
SUPPORTED_MODEL_TYPES = (MODEL_TYPE, UHD_MODEL_TYPE)


@dataclass
class CausalVLLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class VLU_SmartCPM(torch.nn.Module):
    def __init__(
        self,
        llm,
        vpm,
        vision_dim,
        query_num,
        adaptive=False,
        batch_vit=True,
        no_grad_vit=False,
        model_type=MODEL_TYPE,
        insert_layer_id=-1,
        mixed_downsample=False,
    ) -> None:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise NotImplementedError(f"Only {SUPPORTED_MODEL_TYPES} are supported, got {model_type}")
        if not batch_vit:
            raise NotImplementedError("This minimal SFT repo only keeps the batch_vit path")

        super().__init__()
        self.vpm = vpm
        self.llm = llm
        self.vision_dim = vision_dim
        self.query_num = query_num
        self.insert_layer_id = insert_layer_id
        self.mixed_downsample = mixed_downsample
        self.batch_vit = batch_vit
        self.no_grad_vit = no_grad_vit
        self.model_type = model_type

        self.vit_merger = None
        if vpm is not None:
            self.patch_size = self.vpm.patch_size
            embed_dim = self.llm.config.hidden_size
            if model_type == MODEL_TYPE:
                self.resampler = Resampler(
                    num_queries=query_num,
                    embed_dim=embed_dim,
                    num_heads=embed_dim // 128,
                    kv_dim=self.vpm.embed_dim,
                    adaptive=adaptive,
                )
            else:
                self.resampler = Merger(
                    hidden_size=vision_dim,
                    llm_embed_dim=embed_dim,
                    times=1,
                )
                self.vit_merger = get_vit_insert_merger(
                    model_type=model_type,
                    hidden_size=vision_dim,
                    intermediate_size=self.vpm.config.intermediate_size,
                    vpm=self.vpm,
                    insert_layer_id=self.insert_layer_id,
                )

    def unify_vision_embedding(self, vision_embedding):
        if hasattr(vision_embedding, "last_hidden_state"):
            vision_embedding = vision_embedding.last_hidden_state
        return vision_embedding

    def _merge_to_list(self, vision_embedding, tgt_sizes):
        # 与原仓库 (multimodal-copy-2) 一致: 直接调用 resampler。
        # - Resampler: 返回 [B, ...] tensor, 上层按 batch 拆成 list。
        # - Merger:    返回 (packed_tensor, counts_list), 由这里完成 split 还原成 list。
        # 不再额外套 torch.utils.checkpoint:
        #   ZeRO-2 + reentrant=True 嵌套 checkpoint 会让 DeepSpeed backward hook
        #   多次触发, 报 "params_already_reduced" assertion;
        #   ZeRO-2 + reentrant=False 又会触发 PyTorch 2.4+ 的
        #   check_recomputed_tensors_match 对 packed tensor storage 复用敏感的检查。
        #   resampler/Merger 参数量很小, 不套 checkpoint 对显存影响可忽略。
        if isinstance(self.resampler, Resampler):
            out = self.resampler(vision_embedding, tgt_sizes)
            return [out[i] for i in range(out.shape[0])]

        result = self.resampler(vision_embedding, tgt_sizes)
        if isinstance(result, tuple):
            packed, counts = result
            return list(torch.split(packed, counts, dim=0))
        return result

    def _vpm_forward_batch(self, pixel_values_batch, tgt_sizes, dtype, vit_merger):
        all_pv = torch.concat(pixel_values_batch, dim=-1).unsqueeze(0)
        cu_seqlens = F.pad(
            torch.cumsum(tgt_sizes[:, 0] * tgt_sizes[:, 1], dim=0, dtype=torch.int32).cuda(),
            (1, 0),
        )
        max_seqlen = int(torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item())

        if self.no_grad_vit:
            with torch.no_grad():
                vision_embedding, tgt_sizes = self.vpm(
                    all_pv.type(dtype),
                    tgt_sizes=tgt_sizes,
                    cu_seqlens=cu_seqlens,
                    max_seqlens=max_seqlen,
                    vit_merger=vit_merger,
                    insert_layer_id=self.insert_layer_id,
                )
        else:
            vision_embedding, tgt_sizes = self.vpm(
                all_pv.type(dtype),
                tgt_sizes=tgt_sizes,
                cu_seqlens=cu_seqlens,
                max_seqlens=max_seqlen,
                vit_merger=vit_merger,
                insert_layer_id=self.insert_layer_id,
            )

        vision_embedding = self.unify_vision_embedding(vision_embedding)
        return self._merge_to_list(vision_embedding, tgt_sizes)

    def _get_vpm_dtype_device(self):
        if hasattr(self.vpm, "embeddings"):
            pos_emb = self.vpm.embeddings.position_embedding
            pos_tensor = pos_emb.weight if hasattr(pos_emb, "weight") else pos_emb
            return pos_tensor.dtype, pos_tensor.device
        if hasattr(self.vpm, "pos_embed"):
            return self.vpm.pos_embed.weight.dtype, self.vpm.pos_embed.weight.device
        if hasattr(self.vpm, "conv1"):
            return self.vpm.conv1.weight.dtype, self.vpm.conv1.weight.device
        raise AttributeError("Cannot infer dtype/device from vpm")

    def _dummy_vit_merger_forward(self, target_embedding, dtype, device):
        dummy_hs = torch.zeros((1, 16, self.vision_dim), device=device, dtype=dtype)
        dummy_ts = torch.tensor([[4, 4]], dtype=torch.int32)
        dummy_cu = torch.tensor([0, 16], dtype=torch.int32, device=device)
        dummy_out = self.vit_merger(dummy_hs, dummy_ts, None, dummy_cu, 16)
        dummy_merged = dummy_out[0] if isinstance(dummy_out, tuple) else dummy_out
        return target_embedding + dummy_merged.mean() * 0

    def get_vision_embedding(self, pixel_values_list, tgt_sizes, dummy=True, use_4x_downsample=None):
        dtype, device = self._get_vpm_dtype_device()
        vision_hidden_states = []
        all_pixel_values = []

        for pixel_values in pixel_values_list:
            all_pixel_values.extend(pixel_values)

        if all_pixel_values:
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            if (
                self.mixed_downsample
                and self.vit_merger is not None
                and use_4x_downsample is not None
                and any(use_4x_downsample)
            ):
                n_images = len(all_pixel_values)
                assert len(use_4x_downsample) == n_images, (
                    f"use_4x_downsample length {len(use_4x_downsample)} != num images {n_images}"
                )

                idx_4x = [i for i in range(n_images) if use_4x_downsample[i]]
                idx_16x = [i for i in range(n_images) if not use_4x_downsample[i]]
                vision_embedding = [None] * n_images

                if idx_4x:
                    emb_4x = self._vpm_forward_batch(
                        [all_pixel_values[i] for i in idx_4x],
                        tgt_sizes[idx_4x],
                        dtype,
                        vit_merger=None,
                    )
                    for local_idx, global_idx in enumerate(idx_4x):
                        vision_embedding[global_idx] = emb_4x[local_idx]

                if idx_16x:
                    emb_16x = self._vpm_forward_batch(
                        [all_pixel_values[i] for i in idx_16x],
                        tgt_sizes[idx_16x],
                        dtype,
                        vit_merger=self.vit_merger,
                    )
                    for local_idx, global_idx in enumerate(idx_16x):
                        vision_embedding[global_idx] = emb_16x[local_idx]
                elif self.training:
                    # Keep vit_merger in the graph when the whole batch takes the 4x path.
                    vision_embedding[0] = self._dummy_vit_merger_forward(
                        vision_embedding[0],
                        dtype,
                        device,
                    )
            else:
                vision_embedding = self._vpm_forward_batch(
                    all_pixel_values,
                    tgt_sizes,
                    dtype,
                    vit_merger=self.vit_merger,
                )

            start = 0
            for pixel_values in pixel_values_list:
                img_cnt = len(pixel_values)
                if img_cnt > 0:
                    vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                    start += img_cnt
                else:
                    vision_hidden_states.append([])
        else:
            if self.training and dummy:
                dummy_image = torch.zeros(
                    (1, 3, self.patch_size * 8, self.patch_size * 8),
                    device=device,
                    dtype=dtype,
                )
                tgt_sizes = torch.tensor([[8, 8]], dtype=torch.int32)
                cu_seqlens = F.pad(
                    torch.cumsum(tgt_sizes[:, 0] * tgt_sizes[:, 1], dim=0, dtype=torch.int32).cuda(),
                    (1, 0),
                )
                max_seqlen = int(torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item())
                dummy_image_embedding, tgt_sizes = self.vpm(
                    dummy_image,
                    tgt_sizes=tgt_sizes,
                    cu_seqlens=cu_seqlens,
                    max_seqlens=max_seqlen,
                    vit_merger=self.vit_merger,
                    insert_layer_id=self.insert_layer_id,
                )
                dummy_image_embedding = self.unify_vision_embedding(dummy_image_embedding)
                dummy_feature = self._merge_to_list(dummy_image_embedding, tgt_sizes)
            else:
                dummy_feature = []
            for _ in range(len(pixel_values_list)):
                vision_hidden_states.append(dummy_feature)

        return vision_hidden_states

    def get_vllm_embedding(self, data):
        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])
        vllm_embedding = vllm_embedding.clone()

        if self.vpm is None:
            return vllm_embedding, None

        if "vision_hidden_states" not in data:
            vision_hidden_states = self.get_vision_embedding(
                data["pixel_values"],
                data["tgt_sizes"],
                dummy=True,
                use_4x_downsample=data.get("use_4x_downsample", None),
            )
        else:
            vision_hidden_states = data["vision_hidden_states"]

        vision_hidden_states = [
            item.type(vllm_embedding.dtype) if isinstance(item, torch.Tensor) else item
            for item in vision_hidden_states
        ]

        for i, cur_vs_hs in enumerate(vision_hidden_states):
            if len(cur_vs_hs) == 0:
                continue

            cur_vllm_emb = vllm_embedding[i]
            cur_image_bound = data["image_bound"][i]
            if len(cur_image_bound) > 0:
                for index, (bound_start, bound_end) in enumerate(cur_image_bound):
                    indices = torch.arange(
                        bound_start,
                        bound_end,
                        dtype=torch.long,
                        device=cur_vllm_emb.device,
                    )
                    cur_vllm_emb[indices] = cur_vs_hs[index]
            elif self.training:
                cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        sig = inspect.signature(self.llm.forward)
        parameters = list(sig.parameters.keys())
        if "cu_seqlens" in parameters and "max_seqlen" in parameters:
            output = self.llm(
                None,
                cu_seqlens=data["cu_seqlens"],
                max_seqlen=data["max_seqlen"],
                position_ids=position_ids,
                inputs_embeds=vllm_embedding,
                return_dict=True,
            )
        else:
            output = self.llm(
                None,
                position_ids=position_ids,
                inputs_embeds=vllm_embedding,
                return_dict=True,
            )

        return CausalVLLMOutput(
            logits=output.logits,
            hidden_states=output.hidden_states,
            vision_hidden_states=vision_hidden_states,
        )