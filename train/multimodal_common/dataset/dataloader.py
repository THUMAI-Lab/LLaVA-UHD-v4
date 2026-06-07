#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2026 modelbest
#
# @date: 2026
#
import json
import torch
import os
import pickle
import hashlib
from typing import Iterable, Dict, List, Any
import torch
import numpy as np
import collections
import random

from multimodal_common.dataset.parquetdataset import get_node_info

from multimodal_common.utils.logger import init_logger
try:
    torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available() # torch 2.3.0 bug
except:
    pass

from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.iter import SampleMultiplexer, Cycler

from multimodal_common.dataset.datapipe import ParquetParser, ItemFetch, UnpadCollectProcessor, \
    ParallelWithBufferDataPipe, KEY_SEP, TraceItem

logger = init_logger(__name__)

torch.set_num_threads(1)

def convert_data_to_cuda(data: Dict):
    def list_to_cuda(data_list: List):
        for i in range(len(data_list)):
            if isinstance(data_list[i], torch.Tensor):
                data_list[i] = data_list[i].cuda(non_blocking=True)
            elif isinstance(data_list[i], List):
                list_to_cuda(data_list[i])

    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = data[k].cuda(non_blocking=True)

        if isinstance(v, List):
            list_to_cuda(v)


class CudaPrefetcher(Iterable):
    """
    Wrap around a batch iterator for asynchornously copying data to gpu to shield memcpy latency.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data = next(self.loader)
        except StopIteration:
            self.data = None
            return
        with torch.cuda.stream(self.stream):
            convert_data_to_cuda(self.data)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        self.preload()
        if data is None:
            raise StopIteration
        return data

    def __iter__(self):
        return self


class UnpadCollater:
    def __init__(self, tokenizer, total_max_length, ncache=20, max_images=10000, force_batch=False, batch_size=None):
        self.tokenizer = tokenizer
        self.total_max_length = total_max_length
        assert ncache > 1
        self.ncache = ncache
        self.max_images = max_images
        self.force_batch = force_batch
        if self.force_batch:
            assert isinstance(batch_size, int)
            self.batch_size = batch_size
        self.buffer = collections.OrderedDict()

        self.clean()

    def clean(self):
        self.input_ids = []
        self.context = []
        self.pixel_values = []
        self.tgt_sizes = []
        self.image_bounds = []
        self.tgt_sizes = []
        self.raw_datas = []
        self.sources = []
        self.keys = []

        self.temproal_ids = []
        self.use_4x_downsample = []

        self.total_length = 0
        self.should_pop = False

    def clean_buffer(self):
        self.buffer = collections.OrderedDict()

    def append_cache(self):
        _append = False
        cur_image_cnt = len(self.pixel_values)
        for buf_key, buf_sample in self.buffer.items():
            # find best
            if len(buf_sample['input_ids'][0]) + self.total_length <= self.total_max_length and \
                    cur_image_cnt + len([i for i in buf_sample['pixel_values'] if len(i) > 0]) <= self.max_images:
                self.input_ids.append(buf_sample['input_ids'][0])
                self.context.append(buf_sample['context'][0])
                self.pixel_values.extend([i for i in buf_sample['pixel_values'] if len(i) > 0])
                self.raw_datas.append(buf_sample['raw_data'])
                self.sources.append(buf_sample.get('source', 'unk'))
                self.total_length += len(buf_sample['input_ids'][0])
                self.keys.append(buf_key)

                if 'image_bound' in buf_sample:
                    self.image_bounds.append(buf_sample['image_bound'])
                if 'tgt_sizes' in buf_sample:
                    self.tgt_sizes.append(buf_sample['tgt_sizes'])

                if 'temproal_ids' in buf_sample:
                    self.temproal_ids.extend(buf_sample['temproal_ids'])

                if 'use_4x_downsample' in buf_sample:
                    self.use_4x_downsample.extend(buf_sample['use_4x_downsample'])

                del self.buffer[buf_key]
                _append = True
                break
        return _append

    def put(self, key, sample):
        if sample is None:
            return
        if self.should_pop:
            print(f'Put sample failure, You Should pop first')
            return

        if len(self.buffer) < self.ncache:
            self.buffer[key] = sample

        _append = self.append_cache()

        if not _append and len(self.buffer) == self.ncache:
            self.should_pop = True

        elif self.force_batch and len(self.input_ids) == self.batch_size:
            self.should_pop = True
        else:
            self.should_pop = False

    def pop(self):
        f_input_ids = np.zeros(self.total_max_length, dtype=np.int32)
        f_context = np.zeros(self.total_max_length, dtype=np.int8)
        f_tgt = np.full((self.total_max_length), -100, dtype=np.int32)
        f_spans = np.zeros(self.total_max_length, dtype=np.int32)
        f_position_ids = np.zeros(self.total_max_length, dtype=np.int32)

        sample_cnt = len(self.input_ids)
        input_ids = np.concatenate(self.input_ids, axis=0)  # (instance_length, )
        context = np.concatenate(self.context, axis=0)  # (instance_length, )
        instance_length = input_ids.shape[0]

        f_input_ids[: instance_length] = input_ids
        f_context[: instance_length] = context

        _spans = list(np.cumsum([inp.shape[0] for inp in self.input_ids]))

        # cu_seqlens 和 max_seqlen 在 flash_attention cuda 时需要
        if _spans[-1] != self.total_max_length:
            cu_seqlens = np.array([0] + _spans + [self.total_max_length], dtype=np.int32)
        else:
            cu_seqlens = np.array([0] + _spans, dtype=np.int32)

        max_seqlen = int(np.max(cu_seqlens[1:] - cu_seqlens[:-1]))

        span_begin = 0
        for span_id, span_end in enumerate(_spans):
            f_spans[span_begin: span_end] = span_id
            f_position_ids[span_begin:span_end] = np.arange(span_end - span_begin)
            span_begin = span_end

        for j in range(instance_length):
            idx = input_ids[j]
            if j > 1:
                if context[j] == 0:
                    if idx != self.tokenizer.bos_id and input_ids[j - 1] != self.tokenizer.eos_id:
                        f_tgt[j - 1] = idx
                if context[j] == 1 and context[j - 1] == 0:
                    if idx != self.tokenizer.bos_id and input_ids[j - 1] != self.tokenizer.eos_id and idx != self.tokenizer.im_start_id and idx != self.tokenizer.im_id_start_id:
                        if hasattr(self.tokenizer, 'eot_id'):
                            f_tgt[j - 1] = self.tokenizer.eot_id # for llama3
                        else:
                            f_tgt[j - 1] = self.tokenizer.eos_id

        image_bounds = []
        tgt_sizes = []

        for i in range(sample_cnt):
            offset = _spans[i - 1] if i > 0 else 0
            if len(self.image_bounds[i]) > 0:
                image_bounds.append(self.image_bounds[i] + offset)
            if len(self.tgt_sizes[i]) > 0:
                tgt_sizes.append(torch.Tensor(self.tgt_sizes[i]).type(torch.int32))


        data = {}

        data['input_ids'] = torch.from_numpy(f_input_ids).unsqueeze(0)
        data['context'] = torch.from_numpy(f_context).unsqueeze(0) > 0
        data['length'] = torch.Tensor([instance_length]).type(torch.int32)
        data['spans'] = torch.from_numpy(f_spans).unsqueeze(0)
        data['cu_seqlens'] = torch.from_numpy(cu_seqlens)
        data['max_seqlen'] = max_seqlen
        data['position_ids'] = torch.from_numpy(f_position_ids).unsqueeze(0)
        data['pixel_values'] = [self.pixel_values]
        data['temproal_ids'] = [self.temproal_ids]
        data['target'] = torch.from_numpy(f_tgt).unsqueeze(0)
        data['raw_data'] = self.raw_datas
        data['source'] = self.sources
        data['keys'] = self.keys

        if image_bounds:
            data['image_bound'] = [torch.vstack(image_bounds)]
        else:
            data['image_bound'] = [[]]

        if tgt_sizes:
            data['tgt_sizes'] = [torch.vstack(tgt_sizes)]
        else:
            data['tgt_sizes'] = [[]]

        if self.use_4x_downsample:
            data['use_4x_downsample'] = self.use_4x_downsample

        self.clean()

        return data


class UnpadParquetDataloader(Iterable):
    def __init__(self,
                 path,
                 builder,
                 collater=None,
                 num_workers=2,
                 data_queue_size=1000,
                 split_by_rank=False,
                 skip_files=0,
                 state_dict_path=None,
                 rank=None,
                 world_size=None,
                 reverse_order=False,
                 ):
        self.path = path
        self.builder = builder
        self.collater = collater
        self.num_workers = num_workers
        self.data_queue_size = data_queue_size
        self.split_by_rank = split_by_rank
        self.skip_files = skip_files
        self.reverse_order = reverse_order
        self.dataset_config = self.get_dataset_config(self.path)
        if rank is None:
            self.rank, self.world_size = get_node_info() if get_node_info else (0, 1)
        else:
            self.rank = rank
            self.world_size = world_size
        self._state_dict = {}
        self._epoch = 0

        if state_dict_path:
            self.load_state_dict(state_dict_path)

    def get_dataset_config(self, path):
        dataset_config = []
        if path.endswith('.json'):
            # example
            # [
            #     {'path': '/path/to/train_files/train_eng.txt', 'weight': 0.5},
            #     {'path': '/path/to/train_files/train_zh.txt', 'weight': 0.5}
            # ]
            with open(path) as f:
                ds_list = json.load(f)
                logger.info(f'use multi datasets with config={ds_list}')
                for ds in ds_list:
                    skip_files = ds.get('skip_files', self.skip_files)
                    files = self.collect_file(ds['path'])
                    # if len(files) > skip_files:
                    #     files = files[skip_files:]
                    if os.environ.get("REPRODUCIBLE", "false").lower() == "true":
                        nw = 1
                    else:
                        nw = ds.get('num_workers', self.num_workers)

                    config = {
                        'path': ds['path'],
                        'files': files,
                        'skip_files': ds.get('skip_files', self.skip_files),
                        'weight': ds['weight'],
                        'split_by_rank': ds.get('split_by_rank', self.split_by_rank),
                        'num_workers': nw,
                    }
                    logger.info(f"head 5 files: {','.join(files[:5])}")

                    dataset_config.append(config)

        else:
            files = self.collect_file(path)
            # if len(files) > self.skip_files:
            #     files = files[self.skip_files:]
            config = {'path': path, 'files': files }
            logger.info(f"head 5 files: {','.join(files[:5])}")

            dataset_config.append(config)

        return dataset_config


    def collect_file(self, path):
        file_list = []
        if os.path.isfile(path):
            with open(path) as f:
                lines = [i.strip() for i in f.readlines()]
            file_list.extend(lines)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    if not name.endswith('.parquet'):
                        continue
                    file_list.append(os.path.join(root, name))
                for name in dirs:
                    if not name.endswith('.parquet'):
                        continue
                    file_list.append(os.path.join(root, name))

        if self.reverse_order:
            file_list = file_list[::-1]
        return file_list


    def warp_builder_to_fn(self):
        _none_count = [0]
        def itembuild_fn(x):
            trace_item, data = x
            if os.environ.get("REPRODUCIBLE", "false").lower() == "true":
                if isinstance(trace_item, str):
                    seed_key = trace_item
                else:
                    seed_key = (
                        f"{trace_item.source}{KEY_SEP}{trace_item.file_idx}"
                        f"{KEY_SEP}{trace_item.file_path}{KEY_SEP}{trace_item.item_idx}"
                    )
                if isinstance(data, dict):
                    data = dict(data)
                    data['_det_trace_key'] = seed_key
                seed = int.from_bytes(hashlib.md5(seed_key.encode("utf-8")).digest()[:4], "little")
                py_state = random.getstate()
                np_state = np.random.get_state()
                torch_state = torch.random.get_rng_state()
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                try:
                    res = self.builder.build_item(data)
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                    torch.random.set_rng_state(torch_state)
            else:
                res = self.builder.build_item(data)
            if res is None:
                _none_count[0] += 1
                if _none_count[0] % 100 == 1:
                    logger.warn(f'build_item res is None, trace_item={trace_item}, total_none={_none_count[0]}')

            if res is not None and isinstance(res, list) and len(res) > 0:
                res_list = []
                for r in res:
                    res_list.append((trace_item, r))
                return res_list
            return trace_item, res
        return itembuild_fn


    def build_single_item_datapipe(self, source, files, num_workers, split_by_rank, skip_files=0, state_dict=None):
        """
        构建 parquet datapipe
        通过返回值增加一个 key 来记录 datapipe 的数据读取进度，最终的 key 为 fileidx###filepath###itemidx
        1. IterableWrapper(self.files) -> (fileidx, filepath)
        2. ParquetParser(datapipe) -> (fileidx###filepath, dataframe iter)
        3. ItemFetch(datapipe) -> (fileidx###filepath###itemidx,  data dict)
        4. ParallelWithBufferDataPipe -> single item build -> (fileidx###filepath###itemidx, item result)
        """
        source_files = [TraceItem(source=source, file_idx=idx, file_path=path) for (idx, path) in enumerate(files)]
        logger.info(f"num of files: {len(source_files)}")

        datapipe = IterableWrapper(source_files)
        if split_by_rank:
            datapipe = datapipe.sharding_filter()
            torch.utils.data.graph_settings.apply_sharding(datapipe, self.world_size, self.rank)
            datapipe = datapipe.parquet_parse(skip_files, state_dict, self.reverse_order).prefetch(1)
            datapipe = datapipe.item_fetch()
        else:
            datapipe = datapipe.parquet_parse(skip_files, state_dict, self.reverse_order).prefetch(1)
            datapipe = datapipe.item_fetch()
            datapipe = datapipe.sharding_filter()
            torch.utils.data.graph_settings.apply_sharding(datapipe, self.world_size, self.rank)

        process_fn = self.warp_builder_to_fn()

        if num_workers > 1:
            datapipe = datapipe.parallel_process(
                buffer_size=self.data_queue_size, processor=process_fn, num_workers=num_workers)
        else:
            datapipe = datapipe.process(processor=process_fn)
        return datapipe


    def build_datapipe(self):
        """
            ParallelWithBufferDataPipe -> unpad collator  -> batch dict data
                filepath###idx save in data['keys']
        Returns:
            datapiple

        """

        if len(self.dataset_config) > 1:
            # multi dataset
            datapipe_weight = {}
            for ds_config in self.dataset_config:
                datapipe = self.build_single_item_datapipe(
                    ds_config['path'],
                    ds_config['files'],
                    ds_config['num_workers'],
                    ds_config['split_by_rank'],
                    ds_config['skip_files'],
                    self._state_dict.get(ds_config['path'], None)
                )
                datapipe = datapipe.cycle()

                datapipe_weight[datapipe] = ds_config['weight']

            datapipe = SampleMultiplexer(pipes_to_weights_dict=datapipe_weight, seed=0)

        else:
            if os.environ.get("REPRODUCIBLE", "false").lower() == "true":
                nw = 1
            else:
                nw = self.num_workers

            ds_config = self.dataset_config[0]
            datapipe = self.build_single_item_datapipe(
                ds_config['path'],
                ds_config['files'],
                nw,
                self.split_by_rank,
                self.skip_files,
                self._state_dict.get(ds_config['path'], None)
            )

        if self.collater is not None:
            unpad_processor = UnpadCollectProcessor(self.collater)
            datapipe = datapipe.parallel_process(buffer_size=10, processor=unpad_processor, num_workers=1)

        return datapipe


    def set_epoch(self, epoch):
        self._epoch = epoch
        if epoch > 0:
            self._state_dict = {}

    def update_state_dict(self, items):
        for item in items:
            source = item.source
            item_idx = item.item_idx
            file_key = f'{item.file_idx}{KEY_SEP}{item.file_path}'
            if source in self._state_dict:
                dataset_state_dict = self._state_dict[source]
                last_idx = dataset_state_dict.get(file_key, -1)
                if item_idx > last_idx:
                    dataset_state_dict[file_key] = item_idx
            else:
                self._state_dict[source] = collections.OrderedDict()
                self._state_dict[source][file_key] = item_idx

    def save_state_dict(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.get_state_dict(), f)

    def get_state_dict(self):
        return self._state_dict

    def load_state_dict(self, save_path):
        with open(save_path, 'rb') as f:
            state_dict = pickle.load(f)
        self._state_dict = state_dict

    def __iter__(self):
        datapipe = self.build_datapipe()
        for data in datapipe:
            if isinstance(data, dict) and 'keys' in data:
                keys = data['keys']
                items = [TraceItem(**json.loads(k)) for k in keys]
                self.update_state_dict(items)

                # 兼容语音分数据集统计 loss
                data['trace_keys'] = keys
                data['keys'] = data['source']
            yield  data
