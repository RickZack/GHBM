from typing import Any, Tuple, Union, Optional, Callable, List, Self
from pathlib import Path
import torch
from abc import ABC, abstractmethod
import numpy as np
from src.utils.data_splitter import DataSplitter

_identity = lambda x: x
_joint_identity = lambda x,y: (x,y)

class Transform(torch.nn.Module):
    """Wrapper for torch transformations, encapsulates the logic for applying transformations
    on inputs, targets and jointly on both (e.g. CutMix, Mixup)
    
    """
    def __init__(self, input_transform: Optional[Callable] = None, 
                       target_transform: Optional[Callable] = None,
                       joint_transform: Optional[Callable] = None) -> None:
        is_valid = bool(input_transform or target_transform) ^ bool(joint_transform)
        assert is_valid, "Either (transform or target_transform) or (exclusive) joint_trasform must be specified"
        super().__init__()
        self.input_transform = input_transform or _identity
        self.target_transform = target_transform or _identity
        self.joint_transform = joint_transform or _joint_identity

    def __call__(self, item: Tuple[Any, Any]) -> Tuple[Any, Any]:
        input, target = item
        input = self.input_transform(input)
        target = self.target_transform(target)
        input, target = self.joint_transform(input, target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = []
        if self.input_transform is not _identity:
            body += self._format_transform_repr(self.input_transform, "Input Transform: ")
        if self.target_transform is not _identity:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")
        if self.joint_transform is not _joint_identity:
            body += self._format_transform_repr(self.joint_transform, "Joint transform: ")

        return "\n".join(body)

class FLDataset(torch.utils.data.Dataset, ABC):
    """Abstract class for a dataset to  be used in FL simulations"""

    _repr_indent = 4
    
    def __init__(self, root: Union[str, Path], train: bool = True,
                 samplewise_trasform: Optional[Callable] = None,
                 batchwise_trasform: Optional[Callable] = None) -> None:
        super().__init__()
        self._root = Path(root).absolute()
        self._train = train
        self._samplewise_transform = samplewise_trasform or _identity
        self._batchwise_transform = batchwise_trasform or _identity
        
    @property
    def root(self):
        return self._root
    
    @property
    def train(self):
        return self._train
    
    @property
    def samplewise_trasform(self):
        return self._samplewise_transform
        
    @property
    def batchwise_transform(self):
        return self._batchwise_transform
    
    @property
    def samplewise_transform_repr(self) -> str:
        return repr(self._samplewise_transform)
    
    @property
    def batchwise_transform_repr(self):
        return repr(self._batchwise_transform)
    
    @property
    @abstractmethod
    def num_classes(self):
        ...
        
    @property
    def split_name(self) -> str:
        return "train" if self.train is True else "test"
    
    def __repr__(self) -> str:
        head = '\n'.join((f"Dataset: {self.__class__.__name__}", f"Split: {self.split_name}"))
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if self._samplewise_transform is not _identity:
            body += self._format_transform_repr(self._samplewise_transform, "Samplewise Transforms: ")
        if self._batchwise_transform is not _identity:
            body += self._format_transform_repr(self._batchwise_transform, "Batchwise Transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ''
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._loaditem(index)
        return self._samplewise_transform(item)
    
    @abstractmethod
    def _loaditem(self, index: int) -> Tuple[Any, Any]:
        """Loads one item, without applying any transformation

        Args:
            index (int): the index of the item to load

        Returns:
            Tuple[Any, Any]: a tuple consisting of the sample and its annotation (e.g. class label)
        """
        ...
        
    @abstractmethod
    def make_federated(self, num_splits: int, splitter: DataSplitter) -> List[Self]:
        """Splits the whole dataset in disjoiny subsets to be assigned to FL clients.
        
        The method does not change the FLDataset object it is applied to

        Args:
            num_splits (int): the number of splits to produce
            splitter (DataSplitter): a DataSplitter object representing the strategy to adopt in sharding

        Returns:
            List[Self]: a list of FLDatasets produced by splitting the current dataset
        """
        ...
        
    def extract_subset_eq_distr(self, num_samples: int) -> Self:
        """Extracts a FLDataset with equally distributed samples.
        
        This method modifed the original FLDataset, by removing the samples
        used to build the new FLDataset

        Args:
            num_samples (int): total number of samples to extract

        Raises:
            NotImplementedError: if the method is called but not implemented in subclasses

        Returns:
            Self: a new FLDataset with equally distributed samples
        """
        raise NotImplementedError()

class DatasetFile:
    """Abstract class for a file to be read to load dataset data (e.g. annotation file)"""
    
    @abstractmethod
    def load(self, base_folder: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads the DatasetFile from disk.

        Args:
            base_folder (str): the path of the DatasetFile

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple of numpy arrays for samples and targets
        """
        ...

class DownloadableDatasetFile(DatasetFile):
    """Abstract class for a DatasetFile that supports download from an external source"""
    
    @abstractmethod
    def download(self, base_folder: str) -> None:
        """Downloads and extracts (if applicable) the DatasetFile

        Args:
            base_folder (str): the folder to download the file onto
        """
        ...
    
    @abstractmethod
    def check_integrity(self, base_folder: str) -> bool:
        """Checks that the DatasetFile is correctly stored onto disk.
        
        In case the dataset file is an archive, it just checks the integrity of
        files the archive is supposed to contain. It does not check the integrity
        of the archive itself (see DownloadableDatasetFile.download())
        
        In case of regular files, it checks the integrity of the file itself.

        Args:
            base_folder (str): the folder the file has been stored onto

        Returns:
            bool: True if the file (or the archive's contents) are intact, False otherwise.
        """
        ...

    
class DownloadableFLDataset(FLDataset, ABC):
    """Abstract class for a FLDataset that supports download from an external source"""
    def __init__(self, root: str | Path, train: bool = True, 
                 samplewise_trasform: Optional[Callable[..., Any]] = None, 
                 batchwise_trasform: Optional[Callable[..., Any]] = None,
                 download: bool = False) -> None:
        super().__init__(root, train, samplewise_trasform, batchwise_trasform)
        if download:
            self.download()
        elif self.check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
    
    @abstractmethod
    def download(self) -> None:
        """Downloads the files the dataset is made of"""
        ...
    
    @abstractmethod
    def check_integrity(self) -> List[DownloadableDatasetFile]:
        """Checks that the FLDataset is correctly stored onto disk.
        
        The semantic of return value is different from DownloadableDatasetFile.check_integrity().
        Indeed, this method returns a list of DownloadableDatasetFile for which check_integrity()
        returned False. When the returned list is empty, it means that all the files are correctly
        stored, hence that the check succeeded.

        Returns:
            List[DownloadableDatasetFile]: a list of dataset files that are not correctly stored onto disk
        """
        ...
        
