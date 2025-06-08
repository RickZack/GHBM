import copy
import functools
import logging
import math
import os
import pickle
import random
import signal
import sys
import time
from contextlib import contextmanager
from typing import Tuple, Union, List
from torch.utils.data import default_collate


import numpy as np
import torch
from tensorboardX import SummaryWriter

def seed_everything(seed: int, make_deterministic: bool = True):
    """Sets a seed for all the sources of randomness. Optionally enabled deterministic results

    Args:
        seed (int): the seed of choice
        make_deterministic (bool, optional): a flag indicating whether to enable deterministic resutls. Defaults to True.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = make_deterministic
        torch.backends.cudnn.benchmark = not make_deterministic

def set_debug_apis(state: bool = False):
    """Sets a state for torch debug APIs

    Args:
        state (bool, optional): whether to enable debug APIs. Defaults to False.
    """
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)

@contextmanager
def timer(name: str, logger: Union[logging.Logger, None] = None):
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    msg = f'[{name}] done in {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} (hh:mm:ss)'
    if logger:
        logger.info(msg)
    else:
        print(msg)


def save_pickle(obj, path: str, open_options: str = "wb"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, open_options) as f:
        pickle.dump(obj, f)
    f.close()

def exit_on_signal(sig, ret_code=0):
    signal.signal(sig, lambda *args: sys.exit(ret_code))


def shuffled_copy(x):
    x_copy = copy.copy(x)  # shallow copy
    np.random.shuffle(x_copy)
    return x_copy


def select_random_subset(x, portion: float):
    input_len = len(x)
    # drop at least one item but not all of them
    to_drop_num = max(1, min(input_len - 1, math.ceil(input_len * portion)))
    to_drop_indexes = np.random.randint(0, input_len, to_drop_num)
    return np.delete(x, to_drop_indexes)


class MeasureMeter:
    """
    Keeps track of predictions result to obtain some measures, e.g. accuracy
    """

    def __init__(self, num_classes: int):
        self.__num_classes = num_classes
        self.__tp = torch.zeros(num_classes)
        self.__tn = torch.zeros(num_classes)
        self.__fp = torch.zeros(num_classes)
        self.__fn = torch.zeros(num_classes)
        self.__total = torch.zeros(num_classes)  # helper, it is just tp+tn+fp+fn

    @property
    def num_classes(self):
        return self.__num_classes

    def reset(self):
        self.__tp.fill_(0)
        self.__tn.fill_(0)
        self.__fp.fill_(0)
        self.__fn.fill_(0)
        self.__total.fill_(0)

    @property
    def accuracy_overall(self) -> float:
        return 100. * torch.sum(self.__tp) / torch.sum(self.__total)

    @property
    def accuracy_per_class(self) -> torch.Tensor:
        return 100. * torch.divide(self.__tp, self.__total + torch.finfo().eps)

    def update(self, predicted_batch: torch.Tensor, label_batch: torch.Tensor):
        for predicted, label in zip(predicted_batch, label_batch.view_as(predicted_batch)):
            # implement only accuracy
            if predicted.item() == label.item():
                self.__tp[label.item()] += 1
            self.__total[label.item()] += 1


def move_tensor_list(tensor_l: List[torch.Tensor], device: str):
    for i in range(len(tensor_l)):
        tensor_l[i] = tensor_l[i].to(device)


def function_call_log(log: logging.Logger):
    def log_call(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            log.info(f"Calling {f.__name__}")
            ret = f(*args, **kwargs)
            log.info(f"{f.__name__} executed")
            return ret

        return wrapper

    return log_call


def store_tensor_list(tensor_l: List[torch.Tensor], storage: str, filename_suffix: str = ''):
    if storage.lower() == 'ram':
        move_tensor_list(tensor_l, "cpu")
    elif storage.lower() != 'gpu':
        move_tensor_list(tensor_l, "cpu")
        save_pickle(tensor_l, os.path.join(storage, filename_suffix))
        tensor_l.clear()


def load_pickle(path: str, open_options: str = "rb"):
    f = open(path, open_options)
    obj = pickle.load(f)
    f.close()
    return obj


def load_tensor_list(tensor_l: List[torch.Tensor], storage: str, device_dest: str, filename_suffix: str = ''):
    if storage.lower() == 'ram':
        move_tensor_list(tensor_l, device_dest)
    elif storage.lower() != 'gpu':
        saved_tensor_l = load_pickle(os.path.join(storage, filename_suffix))
        assert isinstance(saved_tensor_l, list)
        move_tensor_list(saved_tensor_l, device_dest)
        tensor_l.clear()  # just to be sure, tensor_l should be empty
        tensor_l.extend(saved_tensor_l)
        
def tensorlist_to_tensor(tl: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    shapes = [x.shape for x in tl]
    t = torch.cat([t.view(-1) for t in tl])
    return t, shapes
    
def tensor_to_tensorlist(t: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    shapes_flat = [s.numel() for s in shapes]
    tl = torch.split(t, shapes_flat)
    return [ti.reshape(s) for ti, s in zip(tl, shapes)]
