from dataclasses import dataclass
from enum import StrEnum, auto
from typing import List, Optional, Self, Tuple, Union
import json
import numpy as np
from pathlib import Path
from src.datasets.fed_dataset import DownloadableDatasetFile, DownloadableFLDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import logging

log = logging.getLogger(__name__)

class ShakespeareSplit(StrEnum):
    iid = auto()
    niid = auto()

@dataclass(frozen=True)
class ShakespeareFile(DownloadableDatasetFile):
    archive_name: str
    name: str
    url: str
    md5: Optional[str] = None
    
    def check_integrity(self, base_folder: str) -> bool:
        # Check that file contained in the archive exists
        fpath = Path(base_folder).joinpath(self.name)
        return check_integrity(fpath)

    def download(self, base_folder: str) -> None:
        if not self.check_integrity(base_folder):
            download_and_extract_archive(self.url, base_folder, filename=self.archive_name, md5=self.md5)
            
    def load(self, base_folder: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        with open(Path(base_folder).joinpath(self.name)) as f:
            data_dict = json.load(f)
            np_file_data = [np.array(s) for s in data_dict['x']]
            np_file_targets = [np.array(s).flatten() for s in data_dict['y']]
            return np_file_data, np_file_targets

class Shakespeare(DownloadableFLDataset):     
    base_folder = "shakespeare"
    train_files = {
                    ShakespeareSplit.iid: ShakespeareFile('train_sampled_iid.tar.gz', 'train_sampled_iid.json',
                          'https://drive.google.com/uc?export=download&id=1j0SSCmSFet_yPJ0Cmtx_5hbrL1YL_EgC'),
                    ShakespeareSplit.niid: ShakespeareFile('train_sampled_niid.tar.gz', 'train_sampled_niid.json',
                          'https://drive.google.com/uc?export=download&id=1iDuuMeyNudLJlAZFWSKQKXsuJXyrLBIa')
                }
    test_file = ShakespeareFile('test_sampled.tar.gz', 'test_sampled.json',
                          'https://drive.google.com/uc?export=download&id=1Y7RmcrPNw_Ldd5JG_jARuwaM0Y9RT8Jp')
    
    def __init__(self, root: Union[str, Path], train: bool = True, split: Optional[str] = None, download: bool = False) -> None:
        if train:
            assert split is not None, "During training the split must specified (iid, niid)"
            split = ShakespeareSplit(split)
        root = Path(root).joinpath(self.base_folder)
        super().__init__(root, train, download=download)
        
        self._split = split
        data, targets = self._load_data()
        self._set_data(data, targets)
    
    def _load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self.train:
            # The dataset is already federated, so data will be a list of 100 local datasets.
            # Here we keep this data structure, in the _loaditem we will take into account
            # this fact for the centralized
            return self.train_files[self._split].load(self.root)
        else:
            # The test dataset is not federated, so there will be a one-item list with all data
            samples, targets = self.test_file.load(self.root)         
            return [np.vstack(samples)], [np.vstack(targets).flatten()]

    def check_integrity(self) -> List[ShakespeareFile]:
        files_failed = []
        for file in (*self.train_files.values(), self.test_file):
            if not file.check_integrity(self.root):
                files_failed.append(file)
        return files_failed

    def download(self) -> None:
        files_to_download = self.check_integrity()
        if not files_to_download:
            return log.info("Files already downloaded and verified")
        for file in files_to_download:
            log.info(f"Downloading file {file.archive_name}")
            file.download(self.root)
        
    def _loaditem(self, index: int) -> Tuple[np.ndarray, int]:
        # Take into account that the Shakespeare dataset is naturally
        # divided among 100 clients, and self._data is a list of 100
        # local datasets
        local_dataset_index = np.argmax(self._local_datasets_sum > index) - 1
        sample_index = index - self._local_datasets_sum[local_dataset_index]
        sample = self._data[local_dataset_index][sample_index]
        target = self._targets[local_dataset_index][sample_index]        
        
        return sample, target
        
    def _set_data(self, data: List[np.ndarray], targets: List[np.ndarray]) -> Self:
        self._data, self._targets = data, targets
        self._local_datasets_sum = np.insert(np.cumsum([len(d) for d in self._data]), 0, 0)

        return self
    
    def make_federated(self, num_splits: int = 100, _ = None) -> List[Self]:
        from copy import copy
        assert num_splits == 100, "The Shakespeare dataset is divided among 100 clients"
        log.info("Producing the 100 splits of Shakespeare dataset")
        clients_ds = [copy(self)._set_data([d], [t]) for d,t in zip(self._data, self._targets)]
        return clients_ds
    
    def __len__(self):
        return self._local_datasets_sum[-1]
    
    @property
    def num_classes(self):
        return 80
    
    @property
    def split_name(self):
        return super().split_name + " - " + self._split