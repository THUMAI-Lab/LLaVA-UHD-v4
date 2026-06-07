import io
import itertools
import json
import re
import os
from typing import Dict, Tuple, List, Any

import torch
import pandas as pd
import numpy as np
import math
from PIL import Image, PngImagePlugin

from multimodal_common.dataset.utils import detect_repetition_answer_with_source, extract_frame_default, extract_frame_high_refresh, enhance_image_for_ocr, extract_frame_stack, get_duration, pad, slice_image, slice_image_new, reshape_by_patch, ensure_divide
from multimodal_common.dataset.utils import extract_frame_default_cv, extract_frame_high_refresh_cv
from multimodal_common.dataset.utils import extract_frame_default_ffmpeg, extract_frame_stack_ffmpeg
import random
import imgaug.augmenters as iaa
import decord
from decord import VideoReader, cpu
import fitz

from multimodal_common.tokenizers import Qwen2TokenizerFastWrapper
from multimodal_common.utils.constants import usr_indicator, bot_indicator, tool_indicator, sys_indicator
from multimodal_common.utils.logger import init_logger
from multimodal_common.utils.prompts import caption_zh, caption_en, PROMPT_FOR_PPT
from multimodal_common.utils.utils import is_contain_chinese

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = None

logger = init_logger()


def unified_encode(tokenizer: Qwen2TokenizerFastWrapper, text: str):
    return tokenizer.encode(text)


def bytes2image(img_buffer):
    if 'PDF-1.7' in str(img_buffer):
        pdf_document = fitz.open(stream=img_buffer, filetype='pdf')
        assert len(pdf_document) == 1
        page = pdf_document.load_page(0)

        # 将页面转换为图像
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
    else:
        img_io = io.BytesIO(img_buffer)
        img_io.seek(0)
        image = Image.open(img_io).convert('RGB')
    return image


def maybe_select_text(raw_text):
    candidates = raw_text.split('<cap_sep>')
    return random.choice(candidates)


def zh_count(str):
    total = 0
    for s in str:
        if '\u4e00' <= s <= '\u9fef':
            total += 1
    return total


def maybe_parse_json(raw_text: str):
    # VG raw
    if raw_text.startswith('[{') and raw_text.endswith('}]'):
        try:
            data = json.loads(raw_text)
            text_list = [x['phrase'] for x in data if x['height'] > 160 and x['width'] > 160]
            if len(text_list) == 0:
                return max(data, key=lambda x: len(x['phrase'].split()))['phrase']
            else:
                return random.choice(text_list)
        except:
            return raw_text
    else:
        return raw_text


def clean_text(raw_text):
    text = raw_text.replace('<PERSON>', '')
    text = maybe_parse_json(maybe_select_text(text))
    return text


def check_text_valid(raw_text):
    if pd.isna(raw_text):
        return False
    if not is_contain_chinese(raw_text) and len(raw_text.split()) <= 3:
        return False
    if '<img' in raw_text or '<a href' in raw_text:
        return False
    return True


def get_image_placeholder(tokenizer, query_len, use_im_start_end=False):
    if use_im_start_end:
        return tokenizer.im_start + tokenizer.unk_token * query_len + tokenizer.im_end
    else:
        return tokenizer.unk_token * query_len


def get_slice_image_placeholder(tokenizer, query_len, use_im_start_end=False):
    if use_im_start_end:
        return tokenizer.slice_start + tokenizer.unk_token * query_len + tokenizer.slice_end
    else:
        return tokenizer.unk_token * query_len


def drop_message(text):
    if text.lower() == '<none>':
        return True
    if ' is is ' in text or ' you you ' in text or ' me me ' in text or ' the the ' in text:
        return True
    return False


class ItemBuilder():
    def __init__(self, transform=None):
        self.transform = transform

    def build_item(self, data):
        if self.transform is not None:
            return self.transform(data)
        return data


class Qwen2ChatBuilder(ItemBuilder):
    def __init__(self, tokenizer: Qwen2TokenizerFastWrapper, max_len, transform=None, query_len=64, min_resolution=0,
                 skip_overlength=False, skip_no_image=False, use_system_prompt=True, adapt_slice_config=None,
                 dynamic_batch_config=None, aug_size=False, use_image_id=False, new_schema=False,
                 video_frame_config={}, enhance_ocr=False, model_type="resampler", time_stamp_train=False, random_stack_ratio=0,
                 mixed_downsample=False):
        super().__init__(transform)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.skip_overlength = skip_overlength
        self.use_system_prompt = use_system_prompt
        self.query_len = query_len
        self.min_resolution = min_resolution
        self.image_placeholder = get_image_placeholder(self.tokenizer, self.query_len, use_im_start_end=True)
        self.slice_placeholder = get_slice_image_placeholder(self.tokenizer, self.query_len, use_im_start_end=True)
        self.skip_no_image = skip_no_image
        self.aug_size = aug_size
        self.enhance_ocr = enhance_ocr
        self.use_image_id = use_image_id
        self.new_schema = new_schema

        if model_type not in ["resampler", "uhd_mlp_insert_window_attention_ViTmlp_4_4"]:
            raise NotImplementedError(f"Only resampler and uhd_mlp_insert_window_attention_ViTmlp_4_4 are supported, got {model_type}")
        self.model_type = model_type
        self.time_stamp_train = time_stamp_train
        self.random_stack_ratio = random_stack_ratio
        self.mixed_downsample = mixed_downsample

        if adapt_slice_config is not None:
            assert isinstance(adapt_slice_config, Dict)
            assert 'patch_size' in adapt_slice_config
            assert 'max_slice_nums' in adapt_slice_config
            assert 'scale_resolution' in adapt_slice_config

        self.adapt_slice_config = adapt_slice_config
        self.dynamic_batch_config = dynamic_batch_config
        self.video_frame_config = video_frame_config

        logger.info(f'itembuilder with config: aug_size={aug_size}, use_image_id={use_image_id}, new_schema={new_schema}, adapt_slice_config={adapt_slice_config}, dynamic_batch_config={dynamic_batch_config}, video_frame_config={video_frame_config}')
        logger.info(f'itembuilder with config: random_stack_ratio={random_stack_ratio}')

    def get_slice_placeholder(self, grid, visual_tokens=None):
        if grid is None:
            return ''
        # 1 + cols * rows
        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                if visual_tokens == None:
                    lines.append(self.image_placeholder if not self.new_schema else self.slice_placeholder)
                else:
                    lines.append(get_image_placeholder(self.tokenizer, visual_tokens, use_im_start_end=True) \
                              if not self.new_schema else get_slice_image_placeholder(self.tokenizer, visual_tokens, use_im_start_end=True))
            slices.append(''.join(lines))
        
        if not self.new_schema:
            slice_placeholder = self.tokenizer.slice_start + '\n'.join(slices) + self.tokenizer.slice_end
        else:
            slice_placeholder = '\n'.join(slices)
        return slice_placeholder
    

    def get_image_id(self, idx, use_image_id=True):
        if self.use_image_id and use_image_id:
            return f'{self.tokenizer.im_id_start}{idx}{self.tokenizer.im_id_end}'
        else:
            return ''

    def adapt_slice(self, image, never_split=False, max_slice_nums=None, multiple_override=None):
        if max_slice_nums is None:
            max_slice_nums = self.adapt_slice_config['max_slice_nums']
        else:
            max_slice_nums = max_slice_nums
        patch_size = self.adapt_slice_config['patch_size']
        scale_resolution = self.adapt_slice_config['scale_resolution']
        floating_ratio = self.adapt_slice_config.get('floating_ratio', 0.33)

        slice_images =[]

        if multiple_override is not None:
            multiple = multiple_override
        else:
            multiple = 1 if self.model_type == "resampler" else 4

        if self.adapt_slice_config.get('slice_new', False):
            source_image, patches, best_grid = slice_image_new(image, max_slice_nums, scale_resolution, patch_size * multiple, floating_ratio, never_split)
        else:
            source_image, patches, best_grid = slice_image(image, max_slice_nums, scale_resolution, patch_size * multiple, floating_ratio)

        slice_images.append(self.transform(source_image))

        if self.model_type == "resampler":
            final_placeholder = self.image_placeholder
        else:
            W, H = source_image.size
            assert W % (patch_size * multiple) == 0 and H % (patch_size * multiple) == 0, f"{W}*{H} is wrong"
            visual_tokens = W * H // (patch_size * patch_size * multiple * multiple)
            final_placeholder = get_image_placeholder(self.tokenizer, visual_tokens, use_im_start_end=True)

        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(self.transform(patches[i][j]))

            if self.model_type == "resampler":
                final_placeholder += self.get_slice_placeholder(best_grid)
            else:
                W, H = patches[0][0].size
                assert W % (patch_size * multiple) == 0 and H % (patch_size * multiple) == 0, "{W}*{H} is wrong"
                visual_tokens = W * H // (patch_size * patch_size * multiple * multiple)
                final_placeholder += self.get_slice_placeholder(best_grid, visual_tokens)

        return slice_images, final_placeholder
    

    def _get_mixed_multiple(self, is_single_image):
        """mixed_downsample > 0 时，以该概率让单图数据走 4x (multiple=2)，否则保持 16x"""
        if self.mixed_downsample > 0 and is_single_image and random.random() < self.mixed_downsample:
            return 2
        return None

    def build_image_bound(self, res, images, temproal_ids=None, query_nums=0):
        return_res = []
        # 不再 torch.stack 支持任意分辨率
        # if isinstance(images, List) and len(images) > 0:
        #     images = torch.stack(images)
        for r in res:
            # r['input_ids'] (1, len)
            if self.new_schema:
                start_cond = (r['input_ids'][0] == self.tokenizer.im_start_id) | (r['input_ids'][0] == self.tokenizer.slice_start_id)
                end_cond = (r['input_ids'][0] == self.tokenizer.im_end_id) | (r['input_ids'][0] == self.tokenizer.slice_end_id)
            else:
                start_cond = r['input_ids'][0] == self.tokenizer.im_start_id
                end_cond = r['input_ids'][0] == self.tokenizer.im_end_id

            image_start_tokens = torch.where(start_cond)[0]
            # 跳过 im_start
            image_start_tokens += 1
            image_end_tokens = torch.where(end_cond)[0]


            if temproal_ids is None:
                if len(image_start_tokens) != len(image_end_tokens) or len(image_start_tokens) > len(images):
                    continue
                valid_images_cnt = len(image_start_tokens)
            else:
                if len(image_start_tokens) != len(image_end_tokens) or len(image_start_tokens) > len(temproal_ids):
                    continue
                valid_images_cnt = sum([len(t) for t in temproal_ids[:len(image_start_tokens)]])
                if valid_images_cnt > len(images):
                    continue

            image_bound = torch.hstack([image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)])
            
            if self.model_type == "resampler":
                if query_nums > 0 and torch.any((image_bound[:, 1] - image_bound[:, 0]) != query_nums):
                    continue
            else:
                invalid = False
                patch_size = self.adapt_slice_config['patch_size']
                for i, bound in enumerate(image_bound):
                    H, W = images[i].shape[1:]
                    multiple = 2 if self.mixed_downsample and i < len(self._per_image_use_4x) and self._per_image_use_4x[i] else 4
                    final_visual_tokens = W * H // ((patch_size * multiple) ** 2)
                    if bound[1] - bound[0] != final_visual_tokens:
                        invalid = True
                        logger.error(f"Wrong! bound[1] - bound[0] != final_visual_tokens, final_visual_tokens: {final_visual_tokens}, bound[1]: {bound[1]}, bound[0]: {bound[0]}")
                if invalid:
                    continue

            r['pixel_values'] = images[:valid_images_cnt]
            r['image_bound'] = image_bound
            r['temproal_ids'] = temproal_ids[: len(image_start_tokens)]
            return_res.append(r)

        return return_res

    # 新格式标注
    # clean_content + image_buffer_list
    def build_item(self, data):
        self._per_image_use_4x = []

        
        clean_content: Dict = json.loads(data['clean_content'])
        source = data.get('source', 'unk')

        assert isinstance(clean_content, Dict)
        if 'data_type' in clean_content:
            clean_content['task_type'] = clean_content['data_type']

        task_type = clean_content.get('task_type', 'single-turn-qa')
        image_buffer_map = {}
        image_buffer_list = data['image_buffer_list']
        if isinstance(image_buffer_list, dict):
            image_buffer_list = [image_buffer_list]
        
        for i in image_buffer_list:
            if 'buffer' in i:
                image_buffer_map[i['image_id']] = i['buffer']
            elif 'image_bytes' in i:
                image_buffer_map[i['image_id']] = i['image_bytes']

        images = []
        image_id_cnt = 0
        temproal_ids = [] #  -1 表示静态图像

       
        if task_type == 'video-caption' or task_type == 'video-qa':
            assert 'video_path' in clean_content
            max_frame_nums = self.video_frame_config.get('max_frame_nums', 48)
            max_slice_nums = self.video_frame_config.get('max_slice_nums', 2)
            max_merge_fps = self.video_frame_config.get('fps', 1)
            high_res_frame_nums = self.video_frame_config.get('high_res_frame_nums', 16)
            stack_frame_nums = self.video_frame_config.get('stack_frame_nums', 1)


            if max_merge_fps > 1:
                use_high_refresh = True
            else:
                use_high_refresh = False

            video_path = clean_content['video_path']

            duration = get_duration(video_path)
            if duration == 0:
                return None
            if duration > max_frame_nums * 2:
                stack_frame_nums = 4 ## 长视频走高刷 stack 分支
            if self.random_stack_ratio > 0 and random.random() < self.random_stack_ratio:
                stack_frame_nums = 4 ## 随机走 stack
            
            try:
                duration = None
                duration_type = None
                frame_ts_id_group = None
                choose_fps = None
                if 'duration' in clean_content:
                    duration = clean_content['duration']
                    duration_type = clean_content.get('duration_type', 'second')
                    assert duration[0] < duration[1]
                fix_fps = clean_content.get('fps', None)

                ## high_refresh 为 3d resampler 版本, 已弃用
                if use_high_refresh:
                    if os.getenv('USE_OPENCV', ''):
                        extract_func = extract_frame_high_refresh_cv
                        if random.random() > 0.5:
                            logger.info('use opencv to extract frame')
                    else:
                        extract_func = extract_frame_high_refresh
                    video_frames, max_slice_nums, frame_ts_id_group, choose_fps = extract_func(
                        video_path, max_frame_nums=max_frame_nums, max_slice_nums=max_slice_nums, duration=duration, duration_type=duration_type,
                        fps=max_merge_fps, time_scale=0.1, fix_fps=fix_fps
                    )
                ## stack 模式支持高刷
                elif stack_frame_nums > 1:
                    stack_func = extract_frame_stack_ffmpeg if os.getenv('USE_FFMPEG', '') else extract_frame_stack
                    if self.time_stamp_train:
                        video_frames, max_slice_nums, timestamps = stack_func(
                            video_path, max_frame_nums=max_frame_nums, max_slice_nums=max_slice_nums, duration=duration,
                            duration_type=duration_type, max_stack_frame_nums=stack_frame_nums, time_stamp_train=True
                        )
                    else:
                        video_frames, max_slice_nums = stack_func(
                            video_path, max_frame_nums=max_frame_nums, max_slice_nums=max_slice_nums, duration=duration, duration_type=duration_type, max_stack_frame_nums=stack_frame_nums
                        )
                else:
                    if os.getenv('USE_FFMPEG', ''):
                        extract_func = extract_frame_default_ffmpeg
                    elif os.getenv('USE_OPENCV', ''):
                        extract_func = extract_frame_default_cv
                    else:
                        extract_func = extract_frame_default
                    if self.time_stamp_train:
                        video_frames, max_slice_nums, timestamps = extract_func(
                            video_path, max_frame_nums=max_frame_nums, max_slice_nums=max_slice_nums, duration=duration,duration_type=duration_type, time_stamp_train=True
                        )
                    else:
                        video_frames, max_slice_nums = extract_func(
                            video_path, max_frame_nums=max_frame_nums, max_slice_nums=max_slice_nums, duration=duration, duration_type=duration_type
                        )

                assert video_frames is not None 
                               
            except:
                logger.error(f'read video={video_path} error')
                return None

            image_tokens = []
            if frame_ts_id_group is None: ## 非高刷情况, 直接按单图处理
                if isinstance(video_frames[0], list):
                    video_frames = list(itertools.chain(*video_frames))

                if self.time_stamp_train:
                    for idx in range(len(video_frames)):
                        frame = video_frames[idx]
                        slice_images, image_placeholder = self.adapt_slice(frame, max_slice_nums=max_slice_nums)
                        if timestamps[idx] is not None: # stack 帧不需要时间戳
                            ts_val = round(float(timestamps[idx]), 1)
                            image_placeholder = f"<{ts_val} seconds>" + image_placeholder
                        images.extend(slice_images)
                        self._per_image_use_4x.extend([False] * len(slice_images))
                        image_tokens.append(image_placeholder)
                        temproal_ids.extend([[-1]] * len(slice_images))
                else:
                    for frame in video_frames:
                        slice_images, image_placeholder = self.adapt_slice(frame, max_slice_nums=max_slice_nums)
                        images.extend(slice_images)
                        self._per_image_use_4x.extend([False] * len(slice_images))
                        image_tokens.append(image_placeholder)
                        temproal_ids.extend([[-1]] * len(slice_images))

                video_token = '\n'.join(image_tokens)

            else: # 高刷 / merge 情况
                for frame_group, frame_ts_id in zip(video_frames, frame_ts_id_group): # video_frames 分组
                    # 同一个 frame_group 需要过 3d resampler 压缩
                    slice_images_group = []
                    for frame in frame_group:
                        # # 视频分辨率一致，所以 image_placeholder 可以代表一个 group 的
                        slice_images, image_placeholder = self.adapt_slice(frame, max_slice_nums=max_slice_nums)
                        slice_images_group.append(slice_images)

                    group_cnt = len(slice_images_group[0])
                    for gidx in range(group_cnt):
                        group_images = [s[gidx] for s in  slice_images_group]
                        images.extend(group_images)
                        self._per_image_use_4x.extend([False] * len(group_images))
                        temproal_ids.append(frame_ts_id)

                    image_tokens.append(image_placeholder)
                video_token = '\n'.join(image_tokens)

            conversation = json.loads(clean_content['text'])
            messages = []
            for i in range(len(conversation)):
                role = usr_indicator if conversation[i]['from'] == 'human' else bot_indicator
                message = str(conversation[i]['value'])
                if i == 0:
                    assert role==usr_indicator
                    if '<video_placeholder>' in message:
                        message = message.replace('<video_placeholder>', video_token)
                    else:
                        message = video_token + '\n' + message

                messages.append((role, message))

            if not use_high_refresh and stack_frame_nums > 1:
                stack_prompt = 'You are a video model receiving a mixed frame sequence: some are standard full frames, others are composite (multi grids, left-to-right, top-to-bottom order). Parse all frames into a unified temporal sequence, then analyze the content.'
                messages = [(sys_indicator, stack_prompt)] + messages

            res = self.convert_conversation_data([messages], source=source)

        # 其他都为对话格式
        else:
            use_image_id = True
            max_slice_nums = None
        
            try:
                conversation = json.loads(clean_content['text'])
            except:
                logger.error(f'load conversation error: {clean_content["text"]}, source={source}')
                raise Exception(f'load conversation error: {clean_content["text"]}')
            tools = clean_content.get('tools')
            
            messages = []

            _all_conv_text = ''.join([str(c.get('value', '')) for c in conversation if c])
            _original_image_count = len(re.findall(r'<image>.+?</image>', _all_conv_text))

            if self.mixed_downsample:
                _all_conv_text = ''.join([str(c.get('value', '')) for c in conversation if c])
                _original_image_count = len(re.findall(r'<image>.+?</image>', _all_conv_text))
                _qa_is_single_image = (_original_image_count == 1) and not task_type.startswith('synth-video')
                _qa_mixed_mult = self._get_mixed_multiple(is_single_image=_qa_is_single_image)
            else:
                _qa_mixed_mult = None


            if _original_image_count > 1 and (max_slice_nums is None or max_slice_nums > 9):
                max_slice_nums = 9
                
            all_text = ''.join([str(c['value']) for c in conversation if c])

            for i in range(len(conversation)):
                if conversation[i]['from'] == 'system':
                    role = sys_indicator
                elif conversation[i]['from'] == 'human':# or conversation[i]['from'] == 'user':
                    role = usr_indicator
                elif conversation[i]['from'] == 'gpt': # or conversation[i]['from'] == 'assistant':
                    role = bot_indicator
                elif conversation[i]['from'] == 'tool':
                    role = tool_indicator
                else:
                    role = usr_indicator
                message = str(conversation[i]['value'])
                extra = {}
                for key in ('reasoning_content', 'tool_calls'):
                    if key in conversation[i]:
                        extra[key] = conversation[i][key]

                # 空消息过滤
                if message is None or (len(message.strip()) == 0 and 'tool_calls' not in extra and role != tool_indicator):
                    # logger.warn('filter message empty')
                    return None

                # message = refine_question(message)
                img_nums = len(re.findall(r'(<image>.+?</image>)', message))
                never_split = False
                # if img_nums > 5:
                #     never_split = True

                split_msg = re.split(r'(<image>.+?</image>)', message)
                # [
                #     'aaaaa',
                #     '<image>0f3556c7d1e91aa3d70a56daf132f4d3</image>',
                #     'bbbbb'
                # ]
                for j in range(len(split_msg)):
                    if split_msg[j].startswith('<image>') and split_msg[j].endswith('</image>'):
                        image_id = split_msg[j][7:-8]

                        if image_id not in image_buffer_map:
                            logger.info(f'source={source} image_id not in image_buffer_map')

                        img_buffer = image_buffer_map[image_id]
                        try:
                            image = bytes2image(img_buffer)
                            if max(image.size) < self.min_resolution:
                                # logger.warn('filter image size')
                                return None
                            if max(image.size) / min(image.size) > 10:
                                return None
                            
                            if self.aug_size:
                                # if not re.findall(r'<point>|<box>|<quad>', all_text):
                                image = aug_image_size(image)

                            if self.enhance_ocr and random.random() < 0.7:
                                image = enhance_image_for_ocr(image, source=source)

                            slice_images, image_placeholder = self.adapt_slice(image, never_split=never_split, max_slice_nums=max_slice_nums, multiple_override=_qa_mixed_mult)
                            images.extend(slice_images)
                            self._per_image_use_4x.extend([_qa_mixed_mult is not None] * len(slice_images))
                            image_placeholder = self.get_image_id(image_id_cnt) + image_placeholder
                            temproal_ids.extend([[-1]] * len(slice_images))

                            image_id_cnt += 1
                        except:
                            logger.warn(f'image encode error, source={source}')
                            return None
                        split_msg[j] = image_placeholder
                    else:
                        split_msg[j] = split_msg[j].strip() # 内容 strip, 图片文本之间用换行连接

                message = '\n'.join(split_msg)
                if len(extra) > 0:
                    messages.append((role, message, extra))
                else:
                    messages.append((role, message))

            res = self.convert_conversation_data([messages], source=source, tools=tools)

        self.build_image_bound(res, images=images, temproal_ids=temproal_ids, query_nums=self.query_len)

        for r in res:
            r['source'] = source
            if self.mixed_downsample:
                n_imgs = len(r.get('pixel_values', []))
                r['use_4x_downsample'] = self._per_image_use_4x[:n_imgs]

        if len(res) == 0 or 'pixel_values' not in res[0]:
            logger.warn("data is empty or pixel_values not in data")
            return None

        if self.skip_no_image and len(images) == 0:
            logger.warn("skip_no_image")
            return None

        if self.dynamic_batch_config:
            patch_size = self.dynamic_batch_config['patch_size']
            for r in res:
                cur_images = r['pixel_values']
                tgt_sizes = []

                reshape_images = []
                for image in cur_images:
                    H, W = image.shape[1:]
                    reshape_image = reshape_by_patch(image, patch_size)
                    reshape_images.append(reshape_image)
                    tgt_sizes.append([H//patch_size, W//patch_size])

                r['pixel_values'] = reshape_images
                r['tgt_sizes'] = tgt_sizes

        return res
    

    def check_id_in_image(self, input_ids, trunc_pos):
        input_id = input_ids[trunc_pos]
        if input_id in [self.tokenizer.im_start_id, self.tokenizer.im_end_id, self.tokenizer.unk_id,
                        self.tokenizer.slice_start_id, self.tokenizer.slice_end_id,
                        self.tokenizer.im_id_start_id, self.tokenizer.im_id_end_id]:
            return True
        
        if input_id == self.tokenizer.newline_id and input_ids[-1] == self.tokenizer.slice_end_id:
            return True
        
        if input_ids[trunc_pos-1] == self.tokenizer.im_id_start_id or (trunc_pos+1 < len(input_ids) and input_ids[trunc_pos+1] == self.tokenizer.im_id_end_id):
            return True
        
        return False
    

    def split_interleave_overlength(self, input_ids, context):
        split_input_ids = []
        split_context = []
        while len(input_ids) > min(800, self.max_len):
            reserve = False
            if len(input_ids) > self.max_len and self.check_id_in_image(input_ids, self.max_len):
                # 图中间拆分，往前找 image_start
                # 新 schema
                start_pos = np.where(input_ids[: self.max_len] == self.tokenizer.im_id_start_id)[0]
                if len(start_pos) > 1: # 至少还有一张图
                    reserve = True
                    trunc_pos = start_pos[-1]
                else:
                    trunc_pos = self.max_len
               
            else:
                start_pos = np.where(input_ids[: self.max_len] == self.tokenizer.im_id_start_id)[0]
                if len(start_pos) != 0:
                    reserve = True
                trunc_pos = self.max_len
                    
            if reserve:    
                split_input_ids.append(input_ids[:trunc_pos])
                split_context.append(context[:trunc_pos])
                
            input_ids = input_ids[trunc_pos:]
            context = context[trunc_pos:]
        return split_input_ids, split_context


    skip_overlength_source = ['OpenThoughts2', 'Nemotron_nonthinking', 'Nemotron_thinking']

    def convert_conversation_data_to_id(self, tokenizer, messages, use_system_prompt=False, tools=None, current_turn_only=False):
        # 转成标准的 chat 格式
        chat = []
        for msg in messages:
            if msg[0] == usr_indicator:
                chat_msg = {"role": "user", "content": msg[1]}
            elif msg[0] == sys_indicator:
                chat_msg = {"role": "system", "content": msg[1]}
            elif msg[0] == bot_indicator:
                chat_msg = {"role": "assistant", "content": msg[1]}
            elif msg[0] == tool_indicator:
                chat_msg = {"role": "tool", "content": msg[1]}
            else:
                continue
            if len(msg) > 2 and isinstance(msg[2], dict):
                chat_msg.update(msg[2])
            chat.append(chat_msg)

        assert set([i['role'] for i in chat]) & set(['assistant'])


        if '<think>' in chat[-1]['content'] and '</think>' in chat[-1]['content']:
            enable_thinking = True
        else:
            enable_thinking = False

        template_kwargs = {}
        if tools is not None:
            template_kwargs['tools'] = tools

        ret = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking, **template_kwargs
        ) # qwen3
        input_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False, enable_thinking=enable_thinking, **template_kwargs
        )
        # 兼容 transformers v5 (返回 BatchEncoding/dict) 和 v4 (返回 list)
        if isinstance(input_ids, dict) or hasattr(input_ids, 'input_ids'):
            input_ids = input_ids['input_ids']
        input_ids = np.array(input_ids)

        if current_turn_only:
            offset = 0
        elif '<think>\n\n</think>\n\n' in ret:
            offset = 4
        else:
            offset = 0

        start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_start|>'))[0]
        assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('assistant'))[0]
        end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_end|>'))[0]

        if current_turn_only:
            user_token_id = tokenizer.convert_tokens_to_ids('user')
            user_idxs = np.where(input_ids == user_token_id)[0]
            valid_user_idxs = [idx for idx in user_idxs if idx - 1 in set(start_idxs)]

            try:
                tool_response_id = tokenizer.convert_tokens_to_ids('<tool_response>')
            except (KeyError, TypeError):
                tool_response_id = None
            if tool_response_id is not None and tool_response_id != getattr(tokenizer, 'unk_token_id', None):
                real_user_idxs = [
                    idx for idx in valid_user_idxs
                    if idx + 2 >= len(input_ids) or input_ids[idx + 2] != tool_response_id
                ]
            else:
                real_user_idxs = valid_user_idxs

            last_user_start = real_user_idxs[-1] - 1 if real_user_idxs else 0
            assistant_idxs = assistant_idxs[assistant_idxs > last_user_start]

        context = np.ones_like(input_ids, dtype=np.int8)

        end_think_id = tokenizer.convert_tokens_to_ids('</think>') if offset else None
        for i, assistant_idx in enumerate(assistant_idxs):
            if assistant_idx-1 in set(start_idxs):
                st = assistant_idx + 2
                if i == len(assistant_idxs) - 1 and offset:
                    if st + 2 < len(input_ids) and input_ids[st + 2] == end_think_id:
                        st += offset
                for end_idx in end_idxs:
                    if end_idx > st:
                        context[st: end_idx + 1] = 0
                        break
        
        ids = np.hstack(input_ids)
        context = np.hstack(context)

        return ids, context, ret

    def convert_conversation_data(self, conversation_list: List[List], source=None, tools=None):
        res = []
        for conversation in conversation_list:
            try:
                input_ids, context, raw = self.convert_conversation_data_to_id(
                    self.tokenizer,
                    messages=conversation,
                    use_system_prompt=self.use_system_prompt,
                    tools=tools
                )
            except:
                print(f'source={source}, convert_conversation_data_to_id error')
                continue
            if len(input_ids) > self.max_len:
                if self.skip_overlength:
                    continue
                if source in self.skip_overlength_source:
                    continue

                if random.random() > 0.8:  # 20% print 一下
                    if random.random() > 0.99:
                        logger.warn(f"overlength={len(input_ids)}, raw_inp={conversation}, source={source}")
                    else:
                        logger.warn(f"overlength={len(input_ids)}, source={source}")

            input_ids = input_ids[: self.max_len]
            context = context[: self.max_len]

            if np.all(context): # 没有需要计算 loss 的部分
                continue

            res.append({
                'input_ids': torch.from_numpy(input_ids).unsqueeze(0),
                'context': torch.from_numpy(context).unsqueeze(0),
                'raw_data': raw,
            })
        return res


def aug_image_size(image):
    h,w = image.size
    ratio = (h*w/448/448)
    ratio_ceil = min(math.ceil(ratio), 10)
    cand = list(range(ratio_ceil, min(2*ratio_ceil+2, 9)))
    if ratio_ceil > 5:
        cand = [ratio_ceil-1] + cand
    if ratio_ceil > 8:
        cand = [ratio_ceil-2] + cand
                
    if random.random() < 0.5:
        # 保持不变
        return image
    else:
        r = math.sqrt(random.choice(cand)*448*448 / h / w)
        fix_h = int(h * r)
        fix_w = int(w * r)

        image = image.resize((fix_h, fix_w), Image.Resampling.BICUBIC)

        if random.random() < 0.2:
            if os.environ.get("REPRODUCIBLE", "false").lower() == "true":
                _iaa_seed = random.randint(0, 2**31 - 1)
            else:
                _iaa_seed = None
            aug = iaa.JpegCompression(compression=(75, 95), seed=_iaa_seed)
            image = Image.fromarray(aug(image=np.array(image)))
        return image



def refine_docstruct(text):
    """
        给问题回答都进行以下操作
    """
    def strip_specific_tags(text):
        """自动移除文本中的<ocr>, <doc>, <md>标签，保留标签内的内容。"""
        '''
        <ocr> Pirkanmaa Pirkanmaa Birkaland Tampere Western and Central Finland 
        Central Finland Keski-Suomi Mellersta Finland Jyväskylä Western and Central Finland 
        Satakunta Satakunta Satakunda Pori South-Western Finland </ocr>
        '''
        tags = ["ocr", "doc", "md"]
        for tag in tags:
            text = re.sub(f"<{tag}>", "", text)
            text = re.sub(f"</{tag}>", "", text)
        text = re.sub(r'\n\s+', '\n', text)
        return text.strip()

    def replace_bbox_with_box(text):
        """替换 bbox 标签为 box 标签，并格式化数字。"""
        '''<bbox>48,525,882,647</bbox>'''
        return re.sub(r"<bbox>(\d+),(\d+),(\d+),(\d+)</bbox>", r"<box>\1 \2 \3 \4</box>", text)

    def remove_extra_spaces(text):
        """移除行内多余的空格，保持换行符不变。"""
        # TODO: 也可以选择洗掉含有多余空格的数据
        '''
        school  and 
        beyond. 
        Governor    Doug    Ducey 
        … .…… </doc>
        '''
        return '\n'.join([re.sub(r'\s+', ' ', line).strip() for line in text.split('\n')])

    def contains_pipe_in_ocr(text):
        """如果要做，发生在去除标签之前，检查文本中是否在<ocr>标签内包含管道符号('|')。
            原因 ： ocr标签下出现 ｜ 基本就是识别错误了
        """
        """
        <ocr> ME 
        2|7a-8a CBS This Morning $70.00. 2.9| 12,200 
        M-F 
        3/8a-9a CBS This Morning $60.00 2.6] 11,100 
        MF 
        4/8a-9a CBS This Morning $30.00 2.6! 11,100 
        = \"""BQOKEND™* </ocr>
        """
        pattern = re.compile(r'<ocr>[^<]*\|[^<]*</ocr>', re.DOTALL)
        return bool(pattern.search(text))

    def check_repeated_words(text):
        """检查文本中是否有单词连续重复三次以上。 三个以上一直重复基本都是有问题，但也有柱状图会是这种情况， 只会发生在ocr和doc标签里"""
        '''
        workshop of decor with their own hands . <ocr> a a a a a a a a alamy alamy a a a alamy a a a a a a CO CO a a a alamy photo stock KT49NO a www.alamy.com </ocr>
        '''
        pattern = re.compile(r'\b(\w+)( \1\b){3,}')
        return bool(pattern.search(text))


    if contains_pipe_in_ocr(text) or check_repeated_words(text):
        return None
    else:
        text = strip_specific_tags(text)
        text = replace_bbox_with_box(text)
        text = remove_extra_spaces(text)
        return text


def uni_bbox(text):
    if re.findall(r'(<box>.+?</box>)', text):
        split_msg = re.split(r'(<box>.+?</box>)', text)
        for idx, msg in enumerate(split_msg):
            if msg.startswith('<box>') and msg.endswith('</box>'):
                if '(' not in msg:
                    continue
                msg = msg.replace('(', '').replace(')', '').replace(',', ' ')
                split_msg[idx] = msg
        final_content = ''.join(split_msg)
        final_content = final_content.replace(' <ref>', '<ref>').replace('</ref> ', '<ref>')
        final_content = final_content.replace(' <box>', '<box>').replace('</box> ', '</box>')
        return final_content
    else:
        return text

def uni_json(text):
    split_msg = re.split(r'(```json[\s\S]+?```)', text)
    for idx, msg in enumerate(split_msg):
        if msg.startswith('```json') and msg.endswith('```'):
            try:
                obj = json.loads(msg[7:-3])
                split_msg[idx] = f'```json{json.dumps(obj, ensure_ascii=False)}```'
            except:
                pass
    return ''.join(split_msg)

def refine_question(text):
    # text = """You are a clever and brilliant web automation agent. You can understand the meaning of the image and the text.
    # ## Your Task
    # what is the webicon?
    #
    # ## Next Action
    # """

    pattern = r"## Your Task\n(.*?)\n\n## Next Action"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        extracted_text = f'Please tell us how you plan to answer the question "{match.group(1)}"'
        return extracted_text
    else:
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3]
        # print("No match found")
        return text


def replace_point_with_box(text):
    """
    将文本中的<point>{x} {y}</point>替换为<box>{x} {y} {x} {y}</box>。
    """
    pattern = r'<point>(\d+)\s+(\d+)</point>'
    replacement = r'<box>\1 \2 \1 \2</box>'
    replaced_text = re.sub(pattern, replacement, text)
    return replaced_text


def remove_extra_newlines(text):
    return re.sub(r'\n{2,}', '\n', text)


def transform_actions_history(text):
    '''
        把历史动作多余的换行和多余的文本剔除掉
    '''
    pattern = re.compile(r'Actions History\n(.*?)\nYour Task', re.DOTALL)

    def repl(match):
        parts = match.group(1).strip().split('\n')
        steps = []
        has_infomation_section = False
        for part in parts:
            if part.startswith('step'):
                steps.append(part.split(': ')[1])
            elif part.strip().lower() == 'infomations':
                has_infomation_section = True
                break

        steps_transformed = ", ".join(steps)

        if has_infomation_section:
            infomations_index = match.group(1).lower().find('infomations')
            infomations_and_beyond = match.group(1)[infomations_index:]
            return f'Actions History\n{steps_transformed}\n{infomations_and_beyond}\nYour Task'
        else:
            return f'Actions History\n{steps_transformed}\nYour Task'

    transformed_text = pattern.sub(repl, text)
    if 'Infomations' in transformed_text:
        transformed_text = transformed_text.replace('Infomations', 'Information')
    return transformed_text


## fix patch for c4web
def fix_c4web_reconstruct(input_str):
    def should_remove_box(box_str):
        parts = box_str.split()
        if len(parts) != 4:
            return False
        parts = [int(i) for i in parts]
        # 坐标不对 过滤
        return parts[0] >= parts[2] or parts[1] >= parts[3]

    ref_pattern = "<ref>(.*?)</ref>"
    box_pattern = "<box>(.*?)</box>"

    ref_matches = re.findall(ref_pattern, input_str)
    box_matches = re.findall(box_pattern, input_str)

    filtered_pairs = [(ref, box) for ref, box in zip(ref_matches, box_matches) if not should_remove_box(box)]

    reconstructed_str = ''.join(
        f"<ref>{ref}</ref><box>{box}</box>" for ref, box in filtered_pairs
    )

    return reconstructed_str


def fix_idl_reconstruct(input_str):
    ref_pattern = "<ref>(.*?)</ref>"
    box_pattern = "<box>(.*?)</box>"

    ref_matches = re.findall(ref_pattern, input_str)
    box_matches = re.findall(box_pattern, input_str)

    filtered_pairs = []

    word_cnt = 0
    for ref, box in zip(ref_matches, box_matches):
        parts = box.split()
        parts[0] = int(int(parts[0]) * 672 / 1000)
        parts[1] = int(int(parts[1]) * 672 / 1000)
        parts[2] = int(int(parts[2]) * 1000 / 672)
        parts[3] = int(int(parts[3]) * 1000 / 672)

        word_cnt += len(ref.split())

        if parts[2] <= parts[0] or parts[3] <= parts[1]:
            return None

        box = ' '.join([str(i) for i in parts])
        filtered_pairs.append((ref, box))

    if word_cnt > 1000:
        return None

    reconstructed_str = ''.join(
        f"<ref>{ref}</ref><box>{box}</box>" for ref, box in filtered_pairs
    )

    return reconstructed_str


def remove_spaces_for_pdf(text):
    text = re.compile(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])').sub('', text)
    text = re.compile(r'(?<=[^\u4e00-\u9fa5])\s+(?=[^\u4e00-\u9fa5])').sub(' ', text)
    return text


def img2bytes(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='png')
    byte_data = img_buffer.getvalue()
    return byte_data