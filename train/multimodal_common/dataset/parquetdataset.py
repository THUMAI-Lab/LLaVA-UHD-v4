import hashlib
import logging
import base64
import io
import os
import pickle
import random
import re
import time
import threading
import queue
import itertools
from typing import Dict
import traceback
from func_timeout import func_set_timeout
from opencc import OpenCC

import psutil
import atexit
import signal
import json
from torch import multiprocessing
from torch.multiprocessing import Process
from copy import deepcopy
from PIL import Image
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.utils.data as td
from torch import distributed as dist

from transformers import AutoTokenizer, AutoProcessor
from multimodal_common.dataset.itembuilder import ItemBuilder
from multimodal_common.dataset.utils import detect_repetition_for_caption_en, detect_repetition_for_caption_zh, detect_repetition_for_textonly
from multimodal_common.utils.prompts import detailed_instructions, detailed_instructions_zh
from multimodal_common.utils import utils
from multimodal_common.utils.utils import all_gather, is_dist_avail_and_initialized
from multimodal_common.utils.logger import init_logger

logger = init_logger(__name__)

STOP_SIGNAL = 'EOF'
STATE_DICT_KEY = 'state_dict'
STATE_FILE_PATH_KEY = 'file_path'
STATE_READ_IDX_KEY = 'read_idx'

torch.multiprocessing.set_sharing_strategy('file_system')

convert = OpenCC('t2s')

@atexit.register
def kill_children():
    ''' 确保子进程同步退出 '''
    logger.info("quitting, press Ctrl-C to force quit")
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        logger.info("Child pid is {}".format(child.pid))
        os.kill(child.pid, signal.SIGTERM)


def check_process_alive(process):
    return process.poll() is None


def terminate_process(process):
    process.terminate()
    process.wait()


def get_node_info(group=None):
    """
    获取 pytorch 的 node info, 兼容 standalone
    """
    rank, world_size = 0, 1
    # 优先通过环境变量获取
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            if dist.is_available() and dist.is_initialized():
                group = group or torch.dist.group.WORLD
                rank = dist.get_rank(group=group)
                world_size = dist.get_world_size(group=group)
        except:
            pass

    return rank, world_size


def bytes2image(img_buffer):
    if isinstance(img_buffer, str):
        img_buffer = base64.b64decode(img_buffer)
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image


def findMaxConsecutiveOnes(nums):
    maxCount = count = 0
    for i, num in enumerate(nums):
        if num == 1:
            count += 1
        else:
            maxCount = max(maxCount, count)
            count = 0
    maxCount = max(maxCount, count)
    return maxCount



def zh_count(str):
    total = 0
    for s in str:
        if '\u4e00' <= s <= '\u9fef':
            total += 1
    return total