#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2026 modelbest
#
# @date: 2026
#
import dataclasses
import json
import os
import time
import queue
import threading
import traceback
from copy import deepcopy

import func_timeout
import psutil
import atexit
import signal
import pandas as pd
import pyarrow.parquet as pq
import uuid
import torch
from typing import Optional, List, Callable, Any, final, Union
from dataclasses import dataclass

from torchdata.datapipes.iter import IterDataPipe

from multimodal_common.dataset.parquetdataset import get_node_info

from torch import multiprocessing
from torch.multiprocessing import Process, Lock
from torch.utils.data import functional_datapipe

from multimodal_common.dataset.utils import read_parquet_by_pandas, read_parquet_by_pyarrow
from multimodal_common.utils.logger import init_logger

logger = init_logger(__name__)

SLEEP_INTERVAL = 0.01
GET_TIMEOUT = 2
KEY_SEP = "###"
META_SPLIT_SEP = "###"

@atexit.register
def kill_children():
    logger.info("quitting, press Ctrl-C to force quit")
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        logger.info("Child pid is {}".format(child.pid))
        os.kill(child.pid, signal.SIGTERM)


@dataclass
class TraceItem(json.JSONEncoder):
    source: str
    file_idx: int
    file_path: str
    item_idx: int = 0

class Processor():
    """
        适配 ParallelWithBufferDataPipe 的 Processor, 一般用于聚合处理，对标 pytorch Dataloader 定义的 collate_fn
    """
    def __init__(self):
        super().__init__()

    def pop_left(self):
        raise NotImplementedError

    def __call__(self, key, sample):
        raise NotImplementedError

@functional_datapipe("parallel_process")
class ParallelWithBufferDataPipe(IterDataPipe):
    """
        支持多进程/多线程统一的 IterDataPipe
        可避免使用 torch Dataloader 的 num_workers 带来的无法细粒度并行处理问题，以及最小化多进程带来的资源消耗
        确保 num_threads > 0 or num_workers > 0
        如果是 io 密集型处理，请设置 num_threads, 如果是计算密集型处理，请设置 num_workers
        多线程模式需要注意 Processor 是否线程安全
    """
    def __init__(self, source_datapipe,
                 processor: Union[Processor, Callable],
                 buffer_size: int = 100,
                 num_threads: int = 0,
                 num_workers: int = 0,
                 ):
        self.source_datapipe = source_datapipe
        if buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be a positive integer.")
        if num_threads <= 0 and num_workers <= 0:
            raise ValueError(" please set num_thread or num_workers")

        self.buffer_size = buffer_size
        self.processor = processor
        self.num_threads = num_threads
        self.num_workers = num_workers
        if self.num_threads > 0:
            self.mode = 'thread'
        else:
            self.mode = 'process'

        self.manager = multiprocessing.Manager()
        self.source_queue = self.manager.Queue(maxsize=buffer_size)
        self.result_queue = self.manager.Queue(maxsize=buffer_size)

        self.thread_pool = []
        self.process_pool = []

        self.signal = self.manager.dict()
        self.signal['source_exhausted'] = False
        self.signal['result_exhausted_cnt'] = 0
        self.lock = Lock()

    @staticmethod
    def cache(datapipe, source_queue, signal):
        for idx, sample in enumerate(datapipe):
            while source_queue.full():
                time.sleep(SLEEP_INTERVAL)
            source_queue.put(sample)

        signal['source_exhausted'] = True
        logger.debug(f'source_exhausted, signal={signal}')

    @staticmethod
    def process(processor, source_queue, result_queue, signal, lock):
        torch.set_num_threads(1)
        def put_to_queue(data, queue):
            if data is not None:
                while queue.full():
                    time.sleep(SLEEP_INTERVAL)
                queue.put(data)

        while True:
            # exit
            if signal['source_exhausted'] and source_queue.empty():
                if isinstance(processor, Processor):  # 聚合类操作会有 buffer
                    res = processor.pop_left()
                    put_to_queue(res, result_queue)

                lock.acquire()
                signal['result_exhausted_cnt'] += 1
                lock.release()
                logger.debug(f'result_exhausted, signal={signal}')
                break

            # wait queue
            try:
                data = source_queue.get(block=True, timeout=GET_TIMEOUT)
            except queue.Empty as e:
                continue
            source_queue.task_done()

            try:
                res = processor(data)
            except:
                res = None
                logger.error(f'process_fn error with trace={traceback.format_exc()}')
            
            if res is None:
                continue

            if isinstance(res, list):
                for r in res:
                    put_to_queue(r, result_queue)
            else:
                put_to_queue(res, result_queue)

    def __iter__(self):
        self.clean()
        self.signal['source_exhausted'] = False
        self.signal['result_exhausted_cnt'] = 0

        processor_cnt = 0

        # 从上个 datapipe 缓存数据
        cache_thread = threading.Thread(
            target=ParallelWithBufferDataPipe.cache, args=(self.source_datapipe, self.source_queue, self.signal),
            daemon=True
        )
        cache_thread.start()
        self.cache_thread = cache_thread


        if self.mode == 'thread':
            processor_cnt = self.num_threads
            for i in range(self.num_threads):
                t = threading.Thread(
                    name=f'data_thread_{i}',
                    target=ParallelWithBufferDataPipe.process,
                    args=(self.processor, self.source_queue, self.result_queue, self.signal, self.lock),
                    daemon=True)
                t.start()
                self.thread_pool.append(t)

        elif self.mode == 'process':
            processor_cnt = self.num_workers
            for i in range(self.num_workers):
                p = Process(
                    name=f'data_process_{i}',
                    target=ParallelWithBufferDataPipe.process,
                    args=(self.processor, self.source_queue, self.result_queue, self.signal, self.lock)
                )
                p.daemon = True
                p.start()
                self.process_pool.append(p)

        while True:
            # exit
            if self.result_queue.empty() and self.signal['result_exhausted_cnt'] == processor_cnt:
                self.clean()
                return StopIteration

            try:
                data = self.result_queue.get(block=True, timeout=GET_TIMEOUT)
            except queue.Empty as e:
                continue
            self.result_queue.task_done()

            yield data

    def __del__(self):
        self.clean()

    @final
    def clean(self):
        while not self.source_queue.empty():
            self.source_queue.get_nowait()
        while not self.result_queue.empty():
            self.result_queue.get_nowait()

        if hasattr(self, "cache_thread") and self.cache_thread is not None:
            self.cache_thread = None
        for p in self.process_pool:
            p.terminate()

        self.thread_pool = []
        self.process_pool = []



@functional_datapipe("process")
class IterProcessDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, processor: Union[Processor, Callable]):
        self.source_datapipe = source_datapipe
        self.processor = processor
        
    def __iter__(self):
        for sample in self.source_datapipe:
            try:
                res = self.processor(sample)
            except:
                res = None
                logger.error(f'process_fn error with trace={traceback.format_exc()}')
                continue
            
            if isinstance(res, list):
                for r in res:
                    yield r
            else:
                yield res


@functional_datapipe("parquet_parse")
class ParquetParser(IterDataPipe):
    def __init__(self, source_dp, skip_files=0, state_dict=None, reverse_order=False):
        super().__init__()
        self.source_dp = source_dp
        self.skip_files = skip_files
        self.start_file_idx = 0
        self.start_file_path = None
        self.start_item_idx = 0
        self.rank, self.world_size = get_node_info() if get_node_info else (0, 1)
        self.reverse_order = reverse_order
        
        if state_dict is not None and len(state_dict) > 0:
            key = list(state_dict.keys())[-1]
            self.start_file_idx, self.start_file_path = key.split(KEY_SEP)
            self.start_file_idx = int(self.start_file_idx)
            self.start_item_idx = state_dict[key]
            logger.info(f"Locate to key={key}, start_item_idx={self.start_item_idx}")

        if skip_files > self.start_file_idx:
            self.start_file_idx = skip_files
            self.start_item_idx = 0

    def __iter__(self):
        for trace_item in self.source_dp:
            path = trace_item.file_path
            file_idx = trace_item.file_idx
            extra_meta_path = None
            if META_SPLIT_SEP in path:
                path, extra_meta_path = path.split(META_SPLIT_SEP)[:2]

            if file_idx < self.start_file_idx:
                logger.info(f'skip parquet from {path}, by {file_idx} < {self.start_file_idx}!')
                continue

            try:
                df = read_parquet_by_pyarrow(path, batch_size=30)
                assert df is not None
            except func_timeout.exceptions.FunctionTimedOut as e:
                logger.error(f'rank={self.rank} read_parquet_by_pyarrow from {path} timeout!')
                continue
            except:
                try:
                    df = read_parquet_by_pandas(path)
                except func_timeout.exceptions.FunctionTimedOut as e:
                    logger.error(f'rank={self.rank} read_parquet_by_pandas from {path} timeout!')
                    continue
                except:
                    logger.error(f'rank={self.rank} read parquet from {path} error!')
                    continue

            try:
                if self.reverse_order:
                    df = df[::-1]
                assert df is not None

                df.reset_index(inplace=True, drop=True)

                if self.start_file_idx == file_idx and self.start_file_path == path:
                    if self.start_item_idx + 1 < len(df):
                        df = df.iloc[self.start_item_idx + 1:]
                    else:
                        continue
            except func_timeout.exceptions.FunctionTimedOut as e:
                logger.error(f'rank={self.rank} parse_dataframe from {path} timeout!')
                continue
            except:
                logger.info(f'rank={self.rank} parse_dataframe from {path} error!')
                continue

            logger.info(f'rank={self.rank} read parquet from {path}')

            yield trace_item, df.iterrows()
        # new epoch reset
        self.start_file_idx = 0
        self.start_item_idx = 0


@functional_datapipe("item_fetch")
class ItemFetch(IterDataPipe):
    def __init__(self, source_dp):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        for trace_item, df_iter in self.source_dp:
            for item_idx, item in df_iter:
                cur_trace_item = deepcopy(trace_item)
                cur_trace_item.item_idx = item_idx
                data = item.to_dict()

                # fetch 后 item 确定了， json 序列化
                key = json.dumps(dataclasses.asdict(cur_trace_item))

                yield key, data


class UnpadCollectProcessor(Processor):
    def __init__(self, collater):
        super().__init__()
        self.collater = collater

    def pop_left(self):
        if self.collater.total_length > 0:
            return self.collater.pop()
        return None

    def __call__(self, sample):
        key, data = sample
        if self.collater.should_pop:
            pop_data = self.collater.pop()
            self.collater.append_cache()
            self.collater.put(key, data)
            return pop_data

        self.collater.put(key, data)
        return None
