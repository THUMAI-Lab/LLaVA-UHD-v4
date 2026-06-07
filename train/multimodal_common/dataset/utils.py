import random
from func_timeout import func_set_timeout
import func_timeout
import pandas as pd
import pyarrow.parquet as pq
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from decord import VideoReader, cpu
import albumentations as A
from multimodal_common.utils.logger import init_logger
import re
from typing import List, Tuple
import os

DECORD_TIMEOUT = int(os.getenv('DECORD_TIMEOUT', 60))
logger = init_logger()


def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor

def slice_image(image, max_slice_nums=6, scale_resolution=448, patch_size=14, floating_ratio=0.33):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    multiple = original_width * original_height / (scale_resolution * scale_resolution)

    if multiple - int(multiple) > floating_ratio:
        split_grids_nums = int(multiple) + 1
    else:
        split_grids_nums = int(multiple)

    split_grids_nums = min(split_grids_nums, max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if split_grids_nums > 1:
        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)

        # find best grid

        candidate_grids = []
        m = 1
        while m <= split_grids_nums:
            # if split_grids_nums % m == 0:
            if max_slice_nums % m == 0: # bug for test
                candidate_grids.append([m, split_grids_nums // m])
            m += 1

        best_grid = [1, 1]
        min_error = float('inf')
        for grid in candidate_grids:
            # error = abs(log_ratio - math.log(grid[0] / grid[1]))

            # bug for test
            split_w = original_width / grid[0]
            split_h = original_height / grid[1]
            error = abs(log_ratio - math.log(split_w / split_h))

            if error < min_error:
                best_grid = grid
                min_error = error

        # slice by grid
        refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size)
        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    else:
        # dont need to slice
        if multiple < 1:
            # up-scaling image
            best_size = find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
        else:
            # 1 < multiple < 1.33
            best_size = (ensure_divide(original_width, patch_size), ensure_divide(original_height, patch_size))
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)

    return source_image, patches, best_grid


def slice_image_new(image, max_slice_nums=9, scale_resolution=448, patch_size=14, floating_ratio=0, never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split or max_slice_nums==1:
        # dont need to slice, upsample
        best_size = find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float('inf')
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        # slice by grid
        # if ratio < max_slice_nums:
        #     # just resize for grid and patch
        #     width, height = original_size
        #     grid_x, grid_y = best_grid
        #
        #     refine_width = width - width % (grid_x * patch_size)
        #     refine_height = height - height % (grid_y * patch_size)
        #     refine_size = (refine_width, refine_height)
        #
        # else: # need downsampling
        #     refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size)

        # alwayse near patches
        refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size, allow_upscale=True)

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid



def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(original_size, grid, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize((grid_width, grid_height), scale_resolution, patch_size, allow_upscale=allow_upscale)

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size

def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def display_patches(patches):  #
    m, n = len(patches), len(patches[0])
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            plt.imshow(patches[i][j])
            plt.title(f'i_{i * n + j + 1} {patches[i][j].size}')
            plt.axis('off')


def reshape_by_patch(image_tensor, patch_size=14):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor,
        (patch_size, patch_size),
        stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(image_tensor.size(0), patch_size, -1)
    return patches


def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]


from scipy.spatial import cKDTree

def map_to_nearest_scale(values, scale):
    """将值映射到最接近的刻度（高效大规模版本）"""
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


import os
_REPRODUCIBLE = os.environ.get("REPRODUCIBLE", "false").lower() == "true"

if _REPRODUCIBLE:
    def get_video_batch(video_path, frame_idx):
        vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        video = vr.get_batch(frame_idx).asnumpy()
        return video

    def get_duration(video_path):
        try:
            vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
            duration = len(vr) / vr.get_avg_fps()
            return duration
        except Exception as e:
            logger.error(f'get duration error: {video_path}')
            return 0
else:
    @func_set_timeout(DECORD_TIMEOUT)
    def get_video_batch(vr, frame_idx):
        video = vr.get_batch(frame_idx).asnumpy()
        return video

    @func_set_timeout(DECORD_TIMEOUT)
    def get_duration(video_path):
        try:
            vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
            duration = len(vr) / vr.get_avg_fps()
            return duration
        except Exception as e:
            logger.error(f'get duration error: {video_path}')
            return 0


def extract_frame_default(video_path, max_frame_nums=48, max_slice_nums=2, duration=None, duration_type=None, time_stamp_train=False):
    vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))

    # sample_fps = round(vr.get_avg_fps() / 1)  # FPS

    # if duration is not None:
    #     assert duration[0] < duration[1] and duration[1] <= len(vr)
    #     if duration_type == 'second':
    #         frame_idx = [i for i in range(0, len(vr), sample_fps)][round(duration[0]): round(duration[1])]
    #     else:
    #         frame_idx = [i for i in range(duration[0], duration[1], sample_fps)]
    # else:
    #     frame_idx = [i for i in range(0, len(vr), sample_fps)]

    avg_fps = vr.get_avg_fps()
    duration = len(vr) / avg_fps  # 总时长（秒）

    if duration > max_frame_nums:
        timestamps = [round(i * 0.1, 1) for i in range(int(duration / 0.1))]
        frame_idx = [min(int(ts * avg_fps), len(vr) - 1) for ts in timestamps]
        frame_idx = uniform_sample(frame_idx, max_frame_nums)
        timestamps = uniform_sample(timestamps, max_frame_nums)

    else:
        # 小于 MAX_NUM_FRAMES 按 1fps 抽取 frame
        # 训练时随机偏移起始时间（50%整秒，50%偏移 0.1~0.9s），提升泛化
        if time_stamp_train and random.random() > 0.5:
            offset = round(random.uniform(0.1, 0.9), 1)
        else:
            offset = 0.0
        n_frames = int(duration - offset)
        frame_idx = [min(int((i + offset) * avg_fps), len(vr) - 1) for i in range(n_frames)]
        timestamps = [round(float(i) + offset, 1) for i in range(n_frames)]

    first_frame = vr.get_batch([0]).asnumpy()
    h, w = first_frame.shape[1:3]
    assert frame_idx

    # 优先多取帧, 时长短取高清帧 slice=2
    slice_nums = 1
    if h*w >= 448 * 448 and len(frame_idx) < (max_frame_nums / 3):
        slice_nums = max_slice_nums

    try:
        video = get_video_batch(video_path if _REPRODUCIBLE else vr, frame_idx)
    except func_timeout.exceptions.FunctionTimedOut as e:
        logger.error(f'video read timeout: {video_path}')
        return None, None
    except Exception as e:
        logger.error(f'video read error: {video_path}')
        return None, None
    
    video_frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in video]
    assert video_frames

    vr.seek(0)
    del vr
    if time_stamp_train:
        return video_frames, slice_nums, timestamps
    return video_frames, slice_nums



def extract_frame_stack(video_path, max_frame_nums=48, max_slice_nums=2, duration=None, duration_type=None, max_stack_frame_nums=1, time_stamp_train=False):
    vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
    sample_fps = vr.get_avg_fps()
    video_duration = len(vr) / sample_fps # 秒

    # 1) 取可用帧索引范围（支持按秒或按帧区间）
    if duration is not None:
        assert duration[0] < duration[1] and duration[1] <= len(vr)
        if duration_type == 'second':
            assert int(duration[1] * sample_fps) < len(vr)
            start = round(duration[0] * sample_fps)
            end = round(duration[1] * sample_fps)
            frame_idx_full = [i for i in range(0, len(vr))][start:end]
        else:
            start = duration[0]
            end = duration[1]
            frame_idx_full = [i for i in range(start, end)]
    else:
        frame_idx_full = [i for i in range(0, len(vr))]

    time_stamps_full = [round(idx/sample_fps, 1) for idx in frame_idx_full]

    # groups 表示能拆成几个 1 + N
    groups = max_frame_nums // 2

    # stack_frame_nums 代表 1+N 的 N, 这里相当于随机抽取 1-N 个 stack 帧, 如果是 1 相当于没有 stack
    # 如果按最高 stack 方式，还是无法保障 1fps, 相当于尽可能多抽帧
    # 如果最高 stack 大于 1fps, 随机 fps
    if groups * (1 + max_stack_frame_nums) / video_duration < 1:
        stack_frame_nums = max_stack_frame_nums
    else:
        stack_frame_nums = random.randint(1, max_stack_frame_nums)

    max_available_frames = groups * (1 + stack_frame_nums)
    max_fps = min(10, max_available_frames / video_duration)
    max_fps = min(max_fps, sample_fps)

    if max_fps < 1: ## 长视频, 尽可能抽更多的帧
        target_raw_frames = max_available_frames
    else:
        target_raw_frames = random.randint(int(video_duration), min(max_available_frames, int(max_fps * video_duration)))

    if len(frame_idx_full) == 0:
        return None, None

    # 若可用帧多于目标，做均匀采样；否则使用全部帧
    if len(frame_idx_full) > target_raw_frames:
        frame_idx = np.array(uniform_sample(frame_idx_full, target_raw_frames))
        time_stamps = np.array(uniform_sample(time_stamps_full, target_raw_frames))
    else:
        frame_idx = np.array(frame_idx_full)
        time_stamps = np.array(time_stamps_full)
    if random.random() < 0.01:
        logger.info(f'video_duration: {video_duration}, sample_fps: {sample_fps}, stack_frame_nums: {stack_frame_nums}, total_frames: {len(frame_idx)}, actual_fps: {len(frame_idx) / video_duration}')

    # 3) 读取帧
    try:
        video = get_video_batch(video_path if _REPRODUCIBLE else vr, frame_idx)
    except func_timeout.exceptions.FunctionTimedOut as e:
        logger.error(f'video read timeout: {video_path}')
        return None, None
    except Exception as e:
        logger.error(f'video read error: {video_path}')
        return None, None

    video = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in video]
    assert len(video) > 0

    group_size = 1 + stack_frame_nums
    final_video = []
    final_stamps = []
    for i in range(groups):
        start = i * group_size
        end = start + group_size
        if start >= len(video):
            break
        key_frame = video[start]
        stack_frames = video[start + 1: min(end, len(video))]
        final_video.append(key_frame)
        final_stamps.append(round(float(time_stamps[start]), 1))
        if len(stack_frames) > 0:
            final_video.append(concat_images(stack_frames))
            final_stamps.append(None) ## stack 帧不需要时间戳

    vr.seek(0)
    del vr
    if time_stamp_train:
        return final_video, 1, final_stamps
    return final_video, 1


def _ffmpeg_probe(video_path):
    """用 ffprobe 获取视频元信息，返回 (width, height, avg_fps, total_frames)"""
    import subprocess, json
    video_path = str(video_path)
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', video_path
    ]
    probe = json.loads(subprocess.check_output(probe_cmd, timeout=30))
    vs = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    w, h = int(vs['width']), int(vs['height'])
    total = int(vs.get('nb_frames', 0))
    def _rate(s):
        if '/' in s:
            n, d = s.split('/')
            return float(n) / float(d) if float(d) else 0.0
        return float(s)
    fps = _rate(vs.get('avg_frame_rate', '0/0'))
    if fps <= 0:
        fps = _rate(vs.get('r_frame_rate', '25/1'))
    if total <= 0:
        dur = float(probe.get('format', {}).get('duration', 0))
        total = int(dur * fps)
    return w, h, fps, total


def _ffmpeg_decode_frames(video_path, frame_idx, w, h):
    """用 ffmpeg select filter 解码指定帧，返回 PIL Image 列表"""
    import subprocess
    video_path = str(video_path)
    select_expr = '+'.join(f'eq(n\\,{idx})' for idx in frame_idx)
    cmd = [
        'ffmpeg', '-v', 'error',
        '-i', video_path,
        '-vf', f'select={select_expr}',
        '-vsync', 'vfr',
        '-pix_fmt', 'rgb24',
        '-f', 'rawvideo',
        'pipe:1'
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=DECORD_TIMEOUT)
    raw = proc.stdout
    frame_size = w * h * 3
    n_frames = len(raw) // frame_size
    frames = []
    for i in range(n_frames):
        arr = np.frombuffer(raw[i * frame_size:(i + 1) * frame_size], dtype=np.uint8).reshape(h, w, 3)
        frames.append(Image.fromarray(arr))
    return frames


def extract_frame_stack_ffmpeg(video_path, max_frame_nums=48, max_slice_nums=2, duration=None, duration_type=None, max_stack_frame_nums=1, time_stamp_train=False):
    """
    使用 FFmpeg subprocess 实现，与 extract_frame_stack 完全对齐。
    采样策略、stack/concat 后处理逻辑完全一致，仅帧读取改用 ffmpeg。
    """
    video_path = str(video_path)
    w, h, sample_fps, total_frames = _ffmpeg_probe(video_path)
    video_duration = total_frames / sample_fps

    # 1) 取可用帧索引范围（支持按秒或按帧区间）
    if duration is not None:
        assert duration[0] < duration[1] and duration[1] <= total_frames
        if duration_type == 'second':
            assert int(duration[1] * sample_fps) < total_frames
            start = round(duration[0] * sample_fps)
            end = round(duration[1] * sample_fps)
            frame_idx_full = list(range(total_frames))[start:end]
        else:
            start = duration[0]
            end = duration[1]
            frame_idx_full = list(range(start, end))
    else:
        frame_idx_full = list(range(total_frames))

    time_stamps_full = [round(idx / sample_fps, 1) for idx in frame_idx_full]

    groups = max_frame_nums // 2

    if groups * (1 + max_stack_frame_nums) / video_duration < 1:
        stack_frame_nums = max_stack_frame_nums
    else:
        stack_frame_nums = random.randint(1, max_stack_frame_nums)

    max_available_frames = groups * (1 + stack_frame_nums)
    max_fps = min(10, max_available_frames / video_duration)
    max_fps = min(max_fps, sample_fps)

    if max_fps < 1:
        target_raw_frames = max_available_frames
    else:
        target_raw_frames = random.randint(int(video_duration), min(max_available_frames, int(max_fps * video_duration)))

    if len(frame_idx_full) == 0:
        return None, None

    if len(frame_idx_full) > target_raw_frames:
        frame_idx = np.array(uniform_sample(frame_idx_full, target_raw_frames))
        time_stamps = np.array(uniform_sample(time_stamps_full, target_raw_frames))
    else:
        frame_idx = np.array(frame_idx_full)
        time_stamps = np.array(time_stamps_full)

    if random.random() < 0.01:
        logger.info(f'[ffmpeg_stack] video_duration: {video_duration}, sample_fps: {sample_fps}, stack_frame_nums: {stack_frame_nums}, total_frames: {len(frame_idx)}, actual_fps: {len(frame_idx) / video_duration}')

    # 3) 用 ffmpeg 读取帧
    try:
        video = _ffmpeg_decode_frames(video_path, frame_idx.tolist(), w, h)
    except Exception as e:
        logger.error(f'ffmpeg stack read error: {video_path} - {e}')
        return None, None

    if len(video) == 0:
        return None, None

    # 4) stack 后处理（与 extract_frame_stack 完全一致）
    group_size = 1 + stack_frame_nums
    final_video = []
    final_stamps = []
    final_stack_infos = []
    for i in range(groups):
        start = i * group_size
        end = start + group_size
        if start >= len(video):
            break
        key_frame = video[start]
        stack_frames = video[start + 1: min(end, len(video))]
        final_video.append(key_frame)
        final_stamps.append(round(float(time_stamps[start]), 1))
        final_stack_infos.append(None)
        if len(stack_frames) > 0:
            grid_layout = get_concat_grid_layout(stack_frames)
            sub_ts = [round(float(time_stamps[start + 1 + j]), 1)
                      for j in range(len(stack_frames))
                      if start + 1 + j < len(time_stamps)]
            final_video.append(concat_images(stack_frames))
            final_stamps.append(None)
            final_stack_infos.append({
                'grid': grid_layout,
                'sub_timestamps': sub_ts,
            })

    if time_stamp_train:
        return final_video, 1, final_stamps
    return final_video, 1


def extract_frame_high_refresh(video_path, max_frame_nums=48, max_slice_nums=2, duration=None, duration_type=None, fps=1, time_scale=0.1, fix_fps=None):
    vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
    sample_fps = vr.get_avg_fps()
    video_duration = len(vr) / sample_fps # 秒

    fps = min(fps, round(sample_fps))
    assert sample_fps > 1 and fps >= 1
   
    if duration is not None:
        assert duration[0] < duration[1] and duration[1] <= len(vr)
        assert round(duration[0] * sample_fps) >= 0 and round(duration[1] * sample_fps) <= len(vr)
        assert round(duration[1] * sample_fps) <= len(vr) + 1
        if duration_type == 'second':
            assert int(duration[1] * sample_fps) < len(vr)
            offset = round(duration[0] * sample_fps)
            frame_idx = [i for i in range(0, len(vr))][round(duration[0] * sample_fps): round(duration[1] * sample_fps)]
            video_duration = duration[1] - duration[0]
        else:
            offset = duration[0]
            frame_idx = [i for i in range(duration[0], duration[1])]
            video_duration = (duration[1] - duration[0]) / sample_fps

    else:
        offset = 0
        frame_idx = [i for i in range(0, len(vr))]
    

    first_frame = vr.get_batch([0]).asnumpy()
    h, w = first_frame.shape[1:3]
    assert len(frame_idx) > 0
    

    # 优先多取帧, 时长短取高清帧
    slice_nums = 1
    if h*w >= 448 * 448 and video_duration < (max_frame_nums / 3):
        slice_nums = max_slice_nums


    # 小于 48s, 随机 fps * duration 抽帧, 并按 fps group
    # 大于 48s, 抽取 fps * 48 帧, 并按 fps group
    if fix_fps is not None and fix_fps <= fps:
        choose_fps = fix_fps
    else:
        choose_fps = random.randint(1, fps)
    
    frame_idx =  np.array(uniform_sample(frame_idx, round(choose_fps * min(max_frame_nums, video_duration))))

    try:
        video = get_video_batch(video_path if _REPRODUCIBLE else vr, frame_idx)
    except func_timeout.exceptions.FunctionTimedOut as e:
        logger.error(f'video read timeout: {video_path}')
        return None, None, None, None
    except Exception as e:
        logger.error(f'video read error: {video_path}')
        return None, None, None, None
    
    video_frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in video]

    ## frame_idx 映射到时间维度 pos_id
    frame_idx_ts = frame_idx / sample_fps
    if offset > 0:
        frame_idx_ts = frame_idx_ts - offset
    scale = np.arange(0, video_duration, time_scale)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / time_scale
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(video_frames) == len(frame_ts_id)

    video_frames_group = group_array(video_frames, choose_fps)
    frame_ts_id_group = group_array(frame_ts_id, choose_fps)

    vr.seek(0)
    del vr

    return video_frames_group, slice_nums, frame_ts_id_group, choose_fps


def enhance_image_for_ocr(image, source=None, output_path=None):
    """
    使用 Albumentations 对图像进行 OCR 前的增强处理
    
    参数:
        image: PIL.Image.Image 对象
        source: 图像来源，用于应用不同的增强配置
        output_path: 输出图像路径，如果为None则不保存
        
    返回:
        增强后的PIL图像对象
    """
    # 使用PIL打开图像
    
    # 将PIL图像转换为numpy数组(RGB)
    img_np = np.array(image)

    if os.environ.get("REPRODUCIBLE", "false").lower() == "true":
        _albu_seed = random.randint(0, 2**31 - 1)
    else:
        _albu_seed = None

    # 根据不同的图像来源应用不同的增强配置
    if source in ["idl", "docmatix_pdf", "idl-wo-bbox", 'web_ocr_a', 'web_ocr_b', 
                  'DocStruct4M', 'DocLocal', 'pdf_ocr_table', 'pdf_ocr_arxiv', 'pdf_ocr_pipeline', 'pdf_ocr_ccpdf', 
                  'ocr_pdf', 'publaynet', 'pubtable', 'c4web',
                  'TAT-DQA', 'pdfvqa', 'Docmatix', 'idl_qa', 'docvqa']:
        # IDL数据集的专用增强
        transform = A.Compose([
            A.GaussNoise(std_range=(0.1, 0.2)),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.1),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3)),
                A.ColorJitter(),
                A.Equalize(p=0.2),
            ], p=0.3),
            A.Sharpen(p=0.1),
            A.HueSaturationValue(p=0.2),
        ], strict=True, seed=_albu_seed)
    elif source in ['dots_ocr', 'synth_ocr_b', 'synth_ocr_a', 'html_annotation','thirdparty_ocr_a']:
        # 自然场景图像的专用增强
        transform = A.Compose([
            A.OneOf([
                A.Rotate(limit=(-90, -90), p=1.0),  # 固定 -90°
                A.Rotate(limit=(90, 90), p=1.0),    # 固定 90°
            ], p=0.2),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 5), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.1),
            A.OneOf([
                A.CLAHE(clip_limit=3),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3)),
                A.HueSaturationValue(p=0.3),
                A.ColorJitter(),
                A.RandomGamma(p=0.2),
            ], p=0.3),
            A.OneOf([
                A.RandomGamma(p=0.2),
                A.Equalize(p=0.2),
            ], p=0.1),
            A.Sharpen(p=0.1),
        ], strict=True, seed=_albu_seed)
    elif source in ['wukong', 'blip3', 'openimage_qa', 'textvqa']:
        # 自然场景图像的专用增强
        transform = A.Compose([
            A.GaussNoise(std_range=(0.05, 0.1)),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 5), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=3),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3)),
                A.HueSaturationValue(p=0.3),
                A.ColorJitter(),
                A.RandomGamma(p=0.2),
            ], p=0.5),
            A.OneOf([
                A.RandomGamma(p=0.2),
                A.Equalize(p=0.2),
            ], p=0.2),
            A.Sharpen(p=0.1),
        ], strict=True, seed=_albu_seed)
    else:
        return image
        # 自然场景图像的专用增强
        # transform = A.Compose([
        #     A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
        #     A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), p=0.2),
        #     A.Sharpen(p=0.1),
        # ], strict=True, seed=None)
    
    # 应用变换
    transformed = transform(image=img_np)
    enhanced_img_np = transformed["image"]
    
    # 转回PIL格式
    enhanced_pil_img = Image.fromarray(enhanced_img_np)
    
    # 如果需要保存
    if output_path:
        enhanced_pil_img.save(output_path)
    
    return enhanced_pil_img

def extract_frame_default_ffmpeg(video_path, max_frame_nums=48, max_slice_nums=2, duration=None, duration_type=None, time_stamp_train=False):
    """
    使用 FFmpeg subprocess + rawvideo pipe 实现，与 extract_frame_default 完全对齐。
    利用 ffmpeg 硬件级解码能力，对大分辨率 / 长视频有显著速度优势。
    """
    video_path = str(video_path)
    w, h, avg_fps, total_frames = _ffmpeg_probe(video_path)
    duration_sec = total_frames / avg_fps if avg_fps > 0 else 0

    if duration_sec > max_frame_nums:
        timestamps = [round(i * 0.1, 1) for i in range(int(duration_sec / 0.1))]
        frame_idx = [min(int(ts * avg_fps), total_frames - 1) for ts in timestamps]
        frame_idx = uniform_sample(frame_idx, max_frame_nums)
        timestamps = uniform_sample(timestamps, max_frame_nums)
    else:
        if time_stamp_train and random.random() > 0.5:
            offset = round(random.uniform(0.1, 0.9), 1)
        else:
            offset = 0.0
        n_frames = int(duration_sec - offset)
        frame_idx = [min(int((i + offset) * avg_fps), total_frames - 1) for i in range(n_frames)]
        timestamps = [round(float(i) + offset, 1) for i in range(n_frames)]

    if not frame_idx:
        return (None, None, []) if time_stamp_train else (None, None)

    slice_nums = 1
    if h * w >= 448 * 448 and len(frame_idx) < (max_frame_nums / 3):
        slice_nums = max_slice_nums

    try:
        video_frames = _ffmpeg_decode_frames(video_path, frame_idx, w, h)
    except Exception as e:
        logger.error(f'ffmpeg read error: {video_path} - {e}')
        return (None, None, []) if time_stamp_train else (None, None)

    if not video_frames:
        logger.error(f'ffmpeg decoded 0 frames: {video_path}')
        return (None, None, []) if time_stamp_train else (None, None)

    if time_stamp_train:
        return video_frames, slice_nums, timestamps[:len(video_frames)]
    return video_frames, slice_nums


def extract_frame_default_cv(video_path, max_frame_nums=48, max_slice_nums=2, duration=None, duration_type=None, time_stamp_train=False):
    """
    使用 OpenCV 实现，与 extract_frame_default 功能保持一致
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    avg_fps = fps
    duration_sec = total_frames / avg_fps if avg_fps > 0 else 0

    # 时间相关采样策略，与 extract_frame_default 保持一致
    if duration_sec > max_frame_nums:
        # 超过 max_frame_nums，采样更密
        timestamps = [round(i * 0.1, 1) for i in range(int(duration_sec / 0.1))]
        frame_idx = [min(int(ts * avg_fps), total_frames - 1) for ts in timestamps]
        frame_idx = uniform_sample(frame_idx, max_frame_nums)
        timestamps = uniform_sample(timestamps, max_frame_nums)
    else:
        # 低帧数，按 1fps 抽
        if time_stamp_train and random.random() > 0.5:
            offset = round(random.uniform(0.1, 0.9), 1)
        else:
            offset = 0.0
        n_frames = int(duration_sec - offset)
        frame_idx = [min(int((i + offset) * avg_fps), total_frames - 1) for i in range(n_frames)]
        timestamps = [round(float(i) + offset, 1) for i in range(n_frames)]

    # 获取第一帧大小
    # 读取第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("无法读取视频帧")
    h, w = first_frame.shape[:2]
    assert frame_idx

    # slice_nums 判断，和extract_frame_default一致
    slice_nums = 1
    if h*w >= 448 * 448 and len(frame_idx) < (max_frame_nums / 3):
        slice_nums = max_slice_nums

    # 读取所有指定帧
    video_frames = []
    for idx in frame_idx:
        # 设定抓取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 第一次帧用已经读好的 first_frame，避免重复读取
            if idx == 0 and len(video_frames) == 0:
                frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            video_frames.append(Image.fromarray(frame_rgb))

    cap.release()
    assert video_frames

    if time_stamp_train:
        return video_frames, slice_nums, timestamps
    return video_frames, slice_nums

    
def extract_frame_high_refresh_cv(video_path, max_frame_nums=48, max_slice_nums=2, duration=None, duration_type=None, fps=1, time_scale=0.1, fix_fps=None):
    """
    使用 OpenCV 实现的视频抽帧函数，功能与 extrat_frame_high_refresh 相同
    """
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / sample_fps  # 秒
    
    fps = min(fps, round(sample_fps))
    assert sample_fps > 1 and fps >= 1
    
    # 确定帧索引
    if duration is not None:
        assert duration[0] < duration[1] and duration[1] <= total_frames
        assert round(duration[0] * sample_fps) >= 0 and round(duration[1] * sample_fps) <= total_frames
        assert round(duration[1] * sample_fps) <= total_frames + 1
        if duration_type == 'second':
            assert int(duration[1] * sample_fps) < total_frames
            offset = round(duration[0] * sample_fps)
            frame_idx = [i for i in range(0, total_frames)][round(duration[0] * sample_fps): round(duration[1] * sample_fps)]
            video_duration = duration[1] - duration[0]
        else:
            offset = duration[0]
            frame_idx = [i for i in range(duration[0], duration[1])]
            video_duration = (duration[1] - duration[0]) / sample_fps
    else:
        offset = 0
        frame_idx = [i for i in range(0, total_frames)]
    
    # 读取第一帧获取尺寸
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频帧")
    h, w = first_frame.shape[:2]
    assert len(frame_idx) > 0
    
    # 优先多取帧, 时长短取高清帧
    slice_nums = 1
    if h*w >= 448 * 448 and video_duration < (max_frame_nums / 3):
        slice_nums = max_slice_nums
    
    # 小于 48s, 随机 fps * duration 抽帧, 并按 fps group
    # 大于 48s, 抽取 fps * 48 帧, 并按 fps group
    if fix_fps is not None and fix_fps <= fps:
        choose_fps = fix_fps
    else:
        choose_fps = random.randint(1, fps)
    
    frame_idx = np.array(uniform_sample(frame_idx, round(choose_fps * min(max_frame_nums, video_duration))))
    
    # 读取指定帧
    video_frames = []
    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV 读取的是 BGR 格式，需要转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(Image.fromarray(frame_rgb))
    
    ## frame_idx 映射到时间维度 pos_id
    frame_idx_ts = frame_idx / sample_fps
    if offset > 0:
        frame_idx_ts = frame_idx_ts - offset
    scale = np.arange(0, video_duration, time_scale)
    
    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / time_scale
    frame_ts_id = frame_ts_id.astype(np.int32)
    
    assert len(video_frames) == len(frame_ts_id)
    
    video_frames_group = group_array(video_frames, choose_fps)
    frame_ts_id_group = group_array(frame_ts_id, choose_fps)
    
    cap.release()
    
    return video_frames_group, slice_nums, frame_ts_id_group, choose_fps


@func_set_timeout(480)
def read_parquet_by_pyarrow(path, batch_size=30):
    chunks = []
    parquet_file = pq.ParquetFile(path)
    batch_size = min(parquet_file.metadata.num_rows, 30)
    if batch_size < 1:
        logger.error(f'parquet from {path} is empty') 
        return None

    for i in parquet_file.iter_batches(batch_size=batch_size):
        chunks.append(i.to_pandas())
    df = pd.concat(chunks, ignore_index=True)
    if 'index' in df.columns:
        df.drop('index', axis=1, inplace=True)

    df.reset_index(inplace=True)
    return df


@func_set_timeout(480)
def read_parquet_by_pandas(path):
    df = pd.read_parquet(path)
    if 'index' in df.columns:
        df.drop('index', axis=1, inplace=True)
    df.reset_index(inplace=True)
    return df


##### 重复检测和过滤
def detect_repetition_with_hash(text, window_size=10, max_reps=3):
    """
    Use hashing to efficiently detect repeated n-grams (split by space and underscore).
    Returns -1 if any specific n-gram repeats more than 6 times, otherwise 0.
    """
    # Split text by both space and underscore
    words = []
    for segment in text.split():
        words.extend(segment.split('_'))

    if len(words) <= window_size:
        return 0

    hash_counts = {}
    max_repetitions = 0

    for i in range(len(words) - window_size + 1):
        # Get window and its hash
        window = tuple(words[i:i+window_size])
        window_hash = hash(window)

        # Update count for this hash
        hash_counts[window_hash] = hash_counts.get(window_hash, 0) + 1

        # Update max repetitions and early exit if threshold crossed
        if hash_counts[window_hash] > max_repetitions:
            max_repetitions = hash_counts[window_hash]
            if max_repetitions >= max_reps:
                return -1
    return 0



def detect_repetition_with_hash_zh(text, window_size=10, max_reps=3):
    """
    使用哈希高效检测n-gram重复。适用于中英文混合文本，忽略标点符号。
    如果任何特定的n-gram重复次数超过max_reps，则返回1，否则返回0。
    - 中文按字切分
    - 英文按词切分 (以空格和下划线分隔)
    """
    # 将文本切分为一个token列表：单个汉字或英文单词，忽略标点。
    # 下划线被视为空格
    # 例如 "你好_world, 你好 world" -> ['你', '好', 'world', '你', '好', 'world']
    text = text.replace('_', ' ')
    tokens = re.findall(r'[\u4e00-\u9fa5]|\w+', text)

    if len(tokens) <= window_size:
        return 0

    hash_counts = {}
    max_repetitions = 0

    for i in range(len(tokens) - window_size + 1):
        # 获取窗口并计算其哈希值
        window = tuple(tokens[i:i+window_size])
        window_hash = hash(window)

        # 更新此哈希值的计数
        hash_counts[window_hash] = hash_counts.get(window_hash, 0) + 1

        # 更新最大重复次数，如果超过阈值则提前退出
        if hash_counts[window_hash] > max_repetitions:
            max_repetitions = hash_counts[window_hash]
            if max_repetitions >= max_reps:
                return 1

    return 0



def is_markdown_table_pattern(text: str, window: List[str]) -> bool:
    """
    判断一个重复窗口是否可能是 markdown 表格造成的。
    """
    # 如果全文中完全没有表格的标识符，排除
    if '|' not in text:
        return False
    # 检查重复窗口内容是否不含正常字符，全是 Markdown 表格常用符号
    allowed_chars = set("|:-. ")  
    for token in window:
        cleaned = ''.join(ch for ch in token if ch not in allowed_chars)
        if cleaned:  
            return False
    return True

def is_distributed_and_varied_repetition(locations, words, window_size, min_token_gap=15):
    """
    判断重复是否"分散且多样"
    Args:
        locations: 重复位置列表
        words: 单词列表
        window_size: n-gram窗口大小
        min_token_gap: 最小token间隔
        min_unique_gap_hashes: 最小唯一间隔哈希数
    Returns:
        bool: 是否"分散且多样"
    """
    if len(locations) < 3:
        return False
    gap_hashes = set()
    for i in range(1, len(locations)):
        prev_end = locations[i - 1] + window_size
        curr_start = locations[i]
        gap = curr_start - prev_end
        if gap < min_token_gap:
            return False
        gap_segment = tuple(words[prev_end:curr_start])
        gap_hashes.add(hash(gap_segment))
    # 检查所有间隔是否各不相同
    if len(gap_hashes) < len(locations) - 1:
        return False
    return True


def detect_repetition_advanced(text, window_size=10, hard_threshold=5, min_threshold=3):
    """
    检测文本中的重复模式
    Args:
        text: 输入文本
        window_size: 滑动窗口大小
        hard_threshold: 硬性重复阈值，超过此值直接返回-1
        min_threshold: 最小重复阈值，达到此值才进行进一步分析
    Returns:
        -1: 检测到恶性重复
        0: 未检测到恶性重复或重复可接受
    """
    words_with_indices = [] # 记录每个单词的位置
    current_pos = 0
    for segment in text.split(' '):
        if not segment:
            current_pos += 1
            continue
        sub_segments = segment.split('_')
        for i, sub in enumerate(sub_segments):
            words_with_indices.append({'text': sub, 'start': current_pos})
            current_pos += len(sub)
            if i < len(sub_segments) - 1:
                current_pos += 1
        current_pos += 1
    words = [item['text'] for item in words_with_indices]
    if len(words) < window_size * 2:  # 至少需要能形成2个窗口
        return 0
    ngram_counts = {}
    ngram_locations = {}
    for i in range(len(words) - window_size + 1):
        window = tuple(words[i:i+window_size])
        ngram_counts[window] = ngram_counts.get(window, 0) + 1
        if window not in ngram_locations:
            ngram_locations[window] = []
        ngram_locations[window].append(i)
    # 检查每个重复pattern
    for window_key, count in ngram_counts.items():
        # 1. 如果重复次数超过硬阈值，直接返回-1
        if count >= hard_threshold:
            return -1
        # 2. 对于次数较少的重复，进行进一步分析
        if count >= min_threshold:
            locations = ngram_locations[window_key]
            window_list = list(window_key)
            # 2.1 检查是否为markdown表格标记
            if is_markdown_table_pattern(text, window_list):
                continue
            # 2.2 检查是否为分散且多样的重复
            if is_distributed_and_varied_repetition(locations, words, window_size):
                continue
            # 如果任意一个window，重复次数超过min_threshold，且既不是表格标记也不是分散重复，返回-1
            return -1
    return 0


def detect_repetition_for_caption_en(text):
    text = str(text)
    if detect_repetition_with_hash(text, window_size=10, max_reps=3):
        return True
    if detect_repetition_with_hash(text, window_size=7, max_reps=5):
        return True
    return False


def detect_repetition_for_caption_zh(text):
    text = str(text)
    if detect_repetition_with_hash_zh(text, window_size=15, max_reps=3):
        return True
    if detect_repetition_with_hash_zh(text, window_size=10, max_reps=5):
        return True
    return False


def detect_repetition_for_textonly(text):
    text = str(text)
    if detect_repetition_with_hash(text, window_size=20, max_reps=3):
        return True
    if detect_repetition_with_hash(text, window_size=12, max_reps=5):
        return True
    return False


def detect_long_sentence_without_punctuation(text, word_threshold=30):
    """
    检测文本中是否存在连续超过N个单词而没有任何标点符号的情况。
    标点被定义为除字母、数字、下划线和空格之外的任何字符。
    """
    # 按标点分割文本
    chunks = re.split(r'[^\w\s_]+', str(text))
    for chunk in chunks:
        # 计算每个块中的单词数
        words = chunk.split()
        if len(words) > word_threshold:
            return True
    return False


def detect_repetition_answer_with_source(text, source):
    if source in ['arxiv_qa', 'inst_caption_a', 'pdfacc', 'cc12m-description-1m', 'laion-description-11k', 'allva', 'allava',
                  'doc_caption_a', 'synth_caption_a', 'synth_vqa_a']:
        return detect_repetition_for_caption_en(text)
    if source in ['minicpm3-text-only']:
        return detect_repetition_for_textonly(text)
    if source in ['ALLaVA-Evol-Instruct','Alpaca', 'Alpaca_zh','MathInstruct','MetaMathQA', 'Openorca_gpt4', 'UltraChat','UltraInteract',
                  'atlas-math', 'databricks-dolly','math', 'orca-math-word-problems','sharegpt','sharegpt_zh']:
        # /path/to/datasets/train_paruets/text-only/0623_split/
        return detect_repetition_for_textonly(text)
    if source in ['ultracot', 'Mulberry-SFT', 'R1-Onevision', 'open-r1-8k','ultra_mm',
                  'pixmo-cap', 'tigerlab_visual_web_instrcut', 'mammoth_ov_single_image',
                  'disciplinary_mm_a', 'disciplinary_grammar_a', 'synth_cot_a',
                  'align_single_a', 'llava-cot', 'Xkev-llava-cot', 'mathv_k12', 'R-CoT_geo170k',
                  'R-CoT_GeoMM', 'CoSyn_math', 'mathv360k', 'MMC-Instruction', 'Infinity-MM-Synth-part2',
                  'llava-instruct-eng-conversation-synth']:
        return detect_repetition_advanced(text)
    else:
        return False


def get_concat_grid_layout(images, line_width=6):
    """Return (rows, cols) that concat_images would use for the given images."""
    n = len(images)
    if n == 0:
        return (0, 0)
    if n == 4:
        return (2, 2)
    if n == 1:
        return (1, 1)

    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)

    if n == 3:
        candidates = [(1, 3), (3, 1)]
    elif n == 2:
        candidates = [(1, 2), (2, 1)]
    else:
        return (1, n)

    def _canvas_ratio(r, c):
        W = c * cell_w + (c - 1) * line_width
        H = r * cell_h + (r - 1) * line_width
        return W / max(1, H)

    ratios = [abs(_canvas_ratio(r, c) - 1.0) for (r, c) in candidates]
    if ratios[0] == ratios[1] and n == 2:
        avg_ar = np.mean([im.width / max(1, im.height) for im in images])
        return (1, 2) if avg_ar >= 1.0 else (2, 1)
    return candidates[int(np.argmin(ratios))]


def concat_images(images, bg_color=(255, 255, 255), cell_size=None,
                  line_color=(0, 0, 0), line_width=6):
    """
    images: List[PIL.Image.Image]
    规则：3 张 -> 1x3；4 张 -> 2x2；其余：1xN
    仅在拼接处画分界线（不画外框）。
    """
    n = len(images)
    if n == 0:
        raise ValueError("images is empty")

    if n == 4:
        rows, cols = 2, 2
    elif n == 3:
        # 动态选择 1x3 / 3x1 / 2x2，使最终更接近正方形
        # 先用原图最大宽高确定单元格尺寸（下方 letterbox 会自适应）
        if cell_size is None:
            cell_w = max(im.width for im in images)
            cell_h = max(im.height for im in images)
        else:
            cell_w, cell_h = cell_size

        candidates = [(1, 3), (3, 1)]
        def canvas_ratio(r, c):
            W = c * cell_w + (c - 1) * line_width
            H = r * cell_h + (r - 1) * line_width
            return W / max(1, H)
        ratios = [abs(canvas_ratio(r, c) - 1.0) for (r, c) in candidates]
        best_idx = int(np.argmin(ratios))
        rows, cols = candidates[best_idx]
    elif n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # 动态选择 1x2 / 2x1，使最终更接近正方形
        if cell_size is None:
            cell_w = max(im.width for im in images)
            cell_h = max(im.height for im in images)
        else:
            cell_w, cell_h = cell_size
        candidates = [(1, 2), (2, 1)]
        def canvas_ratio(r, c):
            W = c * cell_w + (c - 1) * line_width
            H = r * cell_h + (r - 1) * line_width
            return W / max(1, H)
        ratios = [abs(canvas_ratio(r, c) - 1.0) for (r, c) in candidates]
        # 如出现并列，依据平均宽高比进行决策：横向排列适合横图，纵向排列适合竖图
        if ratios[0] == ratios[1]:
            avg_ar = np.mean([im.width / max(1, im.height) for im in images])
            rows, cols = (1, 2) if avg_ar >= 1.0 else (2, 1)
        else:
            best_idx = int(np.argmin(ratios))
            rows, cols = candidates[best_idx]
    else:
        rows, cols = 1, n

    # 单元格尺寸
    if cell_size is None:
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
    else:
        cell_w, cell_h = cell_size

    # 保持纵横比缩放到单元格
    def letterbox(im, tw, th):
        im = im.convert("RGB")
        w, h = im.size
        s = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        try:
            im_r = im.resize((nw, nh), Image.Resampling.BICUBIC)
        except AttributeError:
            im_r = im.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (tw, th), bg_color)
        canvas.paste(im_r, ((tw - nw) // 2, (th - nh) // 2))
        return canvas

    # 仅在内部缝隙处留出 line_width 的带状区域作为分界线
    W = cols * cell_w + (cols - 1) * line_width
    H = rows * cell_h + (rows - 1) * line_width
    canvas = Image.new("RGB", (W, H), line_color)

    for i, im in enumerate(images[:rows * cols]):
        r, c = divmod(i, cols)
        cell = letterbox(im, cell_w, cell_h)
        x = c * (cell_w + line_width)
        y = r * (cell_h + line_width)
        canvas.paste(cell, (x, y))

    return canvas