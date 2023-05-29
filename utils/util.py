import json
import yaml
import io
import pickle
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import fnmatch
import os
from shutil import copytree, ignore_patterns


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_pkl(fname):
    fname = Path(fname)
    with fname.open('rb') as handle:
        return pickle.load(handle)

def write_pkl(content, fname):
    fname = Path(fname)
    with fname.open('wb') as handle:
        pickle.dump(content, handle)

def read_yaml(fname):
    with open(fname, 'r') as stream:
        return yaml.safe_load(stream)


def write_yaml(content, fname):
    with io.open(fname, 'w', encoding='utf8') as outfile:
        yaml.dump(content, outfile, default_flow_style=False, allow_unicode=True)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def copy_files(src_dir, dst_dir, *ignores):
    copytree(src_dir, dst_dir, ignore=ignore_patterns(*ignores))

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

class AverageMeterAutoKey:
    def __init__(self):
        self.total_dict = {}
        self.count_dict = {}
        self.reset()

    def reset(self):
        self.total_dict.clear()
        self.count_dict.clear()

    def update(self, key, value, n=1):
        if key in self.total_dict:
            self.total_dict[key] += value
            self.count_dict[key] += n
        else:
            self.total_dict[key] = value
            self.count_dict[key] = n

    def avg(self, key):
        if key in self.total_dict:
            return self.total_dict[key] / float(self.count_dict[key])
        else:
            return None

    def write_avg_to_board(self, tb_writer):
        for k, v in self.total_dict.items():
            tb_writer.add_scalar(k, self.avg(k))
