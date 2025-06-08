from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Callable, List, Optional, Self, Tuple, Union
import torch
import torchvision.transforms.v2 as T
import numpy as np
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from src.datasets.fed_dataset import DatasetFile, DownloadableDatasetFile, DownloadableFLDataset, Transform
from src.utils.data_splitter import DataSplitter
import logging

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class CIFARFile(DatasetFile):
    name: str
    md5: str
    
    def check_integrity(self, base_folder: str) -> bool:
        fpath = Path(base_folder).joinpath(self.name)
        return check_integrity(fpath, self.md5)
            
    def load(self, base_folder: str) -> Tuple[List[np.ndarray], List[int]]:
        data, targets = [], []
        file_path = Path(base_folder, self.name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            target_label = 'labels' if 'labels' in entry else 'fine_labels'
            data.append(entry["data"])
            targets.extend(entry[target_label])
        return data, targets

@dataclass(frozen=True)
class CIFARArchive(DownloadableDatasetFile):
    archive_name: str
    files: List[CIFARFile]
    files_subdir: str
    url: str
    md5: str
    
    def check_integrity(self, base_folder: str) -> bool:
        # Check that files contained in the archive exist
        folder = Path(base_folder).joinpath(self.files_subdir)
        for file in self.files:
            if not file.check_integrity(folder):
                return False
        return True

    def download(self, base_folder: str) -> None:
        if not self.check_integrity(base_folder):
            # If the archive exists (and its check for integrity succeeds)
            # it won't be downloaded again
            download_and_extract_archive(self.url, base_folder, md5=self.md5)
            
    def load(self, base_folder: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        folder = Path(base_folder).joinpath(self.files_subdir)
        data, targets = [], []
        for file in self.files:
            file_data, file_targets = file.load(folder)
            data.append(file_data)
            targets.extend(file_targets)
            
        return data, np.array(targets)

class CIFAR10(DownloadableFLDataset):
    
    base_folder = "cifar10"
    batches_dir = 'cifar-10-batches-py'

    train_list = CIFARArchive('cifar-10-python.tar.gz',
                [
                CIFARFile("data_batch_1", "c99cafc152244af753f735de768cd75f"),
                CIFARFile("data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
                CIFARFile("data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
                CIFARFile("data_batch_4", "634d18415352ddfa80567beed471001a"),
                CIFARFile("data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb")
                ],
                batches_dir, 
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'c58f30108f718f92721af3b95e74349a')
    
    test_list = CIFARArchive('cifar-10-python.tar.gz',
                [CIFARFile("test_batch", "40351d587109b95175f43aff81a1287e")],
                batches_dir,
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'c58f30108f718f92721af3b95e74349a')
    
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)
    
    train_transform = torch.nn.Sequential(
                        Transform(T.RandomCrop(32, 4)),
                        Transform(T.RandomHorizontalFlip()),
                        Transform(T.ConvertImageDtype(torch.float)),
                        Transform(T.Normalize(CIFAR_MEAN, CIFAR_STD))
                        )
    test_transform = torch.nn.Sequential(
                        Transform(T.ConvertImageDtype(torch.float)),
                        Transform(T.Normalize(CIFAR_MEAN, CIFAR_STD))
                        )
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        apply_transform_batchwise: bool = True,
        download: bool = False,
    ) -> None:
        samplewise_transform = batchwise_transform = None
        root = Path(root).joinpath(self.base_folder)
        if not transform:
            samplewise_transform = self.train_transform if train else self.test_transform
        if apply_transform_batchwise:
            batchwise_transform = samplewise_transform
            samplewise_transform = None
        super().__init__(root, train, samplewise_transform, batchwise_transform, download)
    
        data, targets = self._load_data()
        self._set_data(data, targets)

        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        file = self.train_list if self.train else self.test_list
        data, targets = file.load(self.root)

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        return data, targets
        
    def check_integrity(self) -> List[CIFARArchive]:
        files_failed = []
        for archive in (self.train_list, self.test_list):
            if not archive.check_integrity(self.root):
                files_failed.append(archive)
        return files_failed

    def download(self) -> None:
        if not self.check_integrity():
            log.info("Files already downloaded and verified")
            return
        self.train_list.download(self.root)
        
    @property
    def num_classes(self):
        return 10
        
    def __len__(self) -> int:
        return len(self._data)
        
    def _loaditem(self, index: int) -> Tuple[np.ndarray, int]:
        img = self._data[index].transpose(2, 0, 1)
        target = self._targets[index]

        return img, target
        
    def _set_data(self, data: Tuple[List[np.ndarray], List[np.ndarray]], 
                  targets: Tuple[List[np.ndarray], List[np.ndarray]]) -> Self:
        self._data, self._targets = data, targets
        return self

    def make_federated(self, num_splits: int, splitter: DataSplitter) -> List[Self]:
        from copy import copy
        clients_data = splitter.split((self._data, self._targets), num_splits)
        clients_ds = [copy(self)._set_data(*data) for data in clients_data]
        return clients_ds
    
    def _sorted_data_by_label(self):
        sorted_index = np.argsort(self._targets)
        samples = self._data[sorted_index]
        targets = self._targets[sorted_index]
        return samples, targets

    def _subset_indices(self, num_samples: int) -> List[int]:
        # CIFAR has the same number of samples for each class,
        # so the index of the first example of each class can
        # be easily calculated as num_ig_per_class * class_index
        num_img_per_class = len(self._data) // self.num_classes
        to_extract_per_class = num_samples // self.num_classes
        subset_indexes = []
        for class_index in range(self.num_classes):
            class_shard = num_img_per_class * class_index
            samples_index = np.random.choice(num_img_per_class, to_extract_per_class, replace=False) + class_shard
            subset_indexes += list(samples_index)
        return subset_indexes
    
    def extract_subset_eq_distr(self, num_samples: int) -> Self:
        from copy import copy
        assert num_samples >= self.num_classes, f"Cannot extract {num_samples} eq. distr. samples from {self.n_classes} classes"
        
        # Sort data by label and determine indexes of samples to extract
        samples, targets = self._sorted_data_by_label()
        subset_indexes = self._subset_indices(num_samples)

        # Extract samples and targets via indexes
        extracted_samples = samples[subset_indexes]
        extracted_targets = targets[subset_indexes]
        
        # Remove extracted samples from original dataset
        samples = np.delete(samples, subset_indexes, axis=0)
        targets = np.delete(targets, subset_indexes)
        self._set_data(samples, targets)
        
        return copy(self)._set_data(extracted_samples, extracted_targets)


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar100"
    batches_dir = 'cifar-100-python'

    train_list = CIFARArchive('cifar-100-python.tar.gz',
                [CIFARFile("train", "16019d7e3df5f24257cddd939b257f8d")],
                batches_dir,
                'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                'eb9058c3a382ffc7106e4002c42a8d85')
    
    test_list = CIFARArchive('cifar-100-python.tar.gz',
                [CIFARFile("test", "f0ef6b0ae62326f3e7ffdfab6717acfc")],
                batches_dir,
                'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                'eb9058c3a382ffc7106e4002c42a8d85')
    
    @property
    def num_classes(self):
        return 100