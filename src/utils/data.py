import logging
import json
from dataclasses import dataclass, field
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from src.datasets import *
from collections.abc import Iterator
from omegaconf import DictConfig
import numpy as np
from src.utils.dirichlet_non_iid import non_iid_partition_with_dirichlet_distribution
from enum import StrEnum, auto
from src.datasets import *
from src.utils.data_splitter import *

log = logging.getLogger(__name__)


def create_dataset(dataset_info: DictConfig):
    dataset_class = eval(dataset_info.classname)
    train_dataset = dataset_class(**dataset_info.args, train=True)
    test_dataset = dataset_class(**dataset_info.args, train=False)
    return train_dataset, test_dataset

def create_splitter(dataset_info: DictConfig) -> DataSplitter:
    splitter_info = dataset_info.data_splitter
    splitter_class = eval(splitter_info.classname)
    return splitter_class(**splitter_info.args)
