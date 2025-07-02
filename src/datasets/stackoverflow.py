from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Self, Tuple, Union
import numpy as np
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from tqdm import tqdm
from src.datasets.fed_dataset import DownloadableDatasetFile, DownloadableFLDataset
import logging

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class StackOverflowFile(DownloadableDatasetFile):
    archive_name: str
    files: List[str]
    url: str
    md5: Optional[str] = None
    
    def check_integrity(self, base_folder: str) -> bool:
        # Check that files contained in the archive exist
        for fname in self.files:
            fpath = Path(base_folder).joinpath(fname)
            if not check_integrity(fpath):
                return False
        return True

    def download(self, base_folder: str) -> None:
        if not self.check_integrity(base_folder):
            download_and_extract_archive(self.url, base_folder, filename=self.archive_name, md5=self.md5)
            
    def load(self, base_folder: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        data, targets = [], []
        for filename in tqdm(self.files, desc='Loading files', leave=False):
            with open(Path(base_folder).joinpath(filename)) as f:
                data_dict = json.load(f)
                data += data_dict['x']
                targets += data_dict['y']
        data = [np.array(d) for d in data]
        targets = [np.array(d) for d in targets]
        return data, targets


class StackOverflow(DownloadableFLDataset):
    
    base_folder = "stackoverflow"
    train_file = StackOverflowFile('train.tar.gz', ['train_0.json', 'train_1.json', 'train_2.json', 'train_3.json'],
                          'https://drive.google.com/uc?export=download&id=1MAM7DBf66WruABh8t2ucvllu7oMM5ffW')
    test_file = StackOverflowFile('test.tar.gz', ['test_mix_0.json', 'test_mix_1.json', 'test_mix_2.json', 'test_mix_3.json'],
                          'https://drive.google.com/uc?export=download&id=1kQxLdvRZM5z9FfpN6XyoqdU7VELHDGTS')
    
    def __init__(self, root: Union[str, Path], subsample_test_dim: int = 515815, train: bool = True, download: bool = False) -> None:
        root = Path(root).joinpath(self.base_folder)
        super().__init__(root, train, download=download)
        self.subsample_test_dim = subsample_test_dim
        
        data, targets = self._load_data()
        self._set_data(data, targets)
        
    def _load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self.train:
            # The dataset is already federated, so data will be a list of 40K local datasets.
            # Here we keep this data structure, in the _loaditem we will take into account
            # this fact for the centralized
            return self.train_file.load(self.root)
        else:
            # The test dataset is not federated, so there will be a one-item list with all data
            samples, targets = self.test_file.load(self.root)
            # subsample
            chosen = np.random.choice(len(samples), self.subsample_test_dim, False)            
            return [np.array(samples)[chosen]], [np.array(targets)[chosen]]
    
    def check_integrity(self) -> List[StackOverflowFile]:
        files_failed = []
        for file in (self.train_file, self.test_file):
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
        # Take into account that the StackOverflow dataset is naturally
        # divided among 40K clients, and self._data is a list of 40K
        # local datasets
        local_dataset_index = np.argmax(self._local_datasets_sum > index) - 1
        sample_index = index - self._local_datasets_sum[local_dataset_index]
        sample = self._data[local_dataset_index][sample_index]
        target = self._targets[local_dataset_index][sample_index]        
        
        return sample, target
        
    def _set_data(self, data: Tuple[List[np.ndarray], List[np.ndarray]], 
                  targets: Tuple[List[np.ndarray], List[np.ndarray]]) -> Self:
        self._data, self._targets = data, targets
        self._local_datasets_sum = np.insert(np.cumsum([len(d) for d in self._data]), 0, 0)

        return self
    
    def __len__(self):
        return self._local_datasets_sum[-1]
    
    @property
    def num_classes(self):
        return 10004

    def make_federated(self, num_splits: int = 40_000, _ = None) -> List[Self]:
        from copy import copy
        assert num_splits == 40_000, "The StackOverflow dataset is divided among 40.000 clients"
        log.info("Producing the 40.000 splits of StackOverflow dataset")
        clients_ds = [copy(self)._set_data([d], [t]) for d,t in zip(self._data, self._targets)]
        return clients_ds
