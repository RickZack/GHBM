import os
import pickle
import signal
from abc import ABC, abstractmethod
from typing import Any, Mapping, Set
import torch.nn
from omegaconf import DictConfig
from src.optim import *
from torch.utils.data import DataLoader
from torch import nn
from src.losses import *
from src.utils import *
from src.utils.analyzers import AnalyzerController
import logging

from src.utils.state import Device

log = logging.getLogger(__name__)


class Algo(ABC):
    """ Base (abstract) class for any algorithm """

    def __init__(self, params, device: str, dataset: DictConfig,
                 output_suffix: str, savedir: str, writer=None, state_keys: Set[str] = frozenset()):
        loss_info = params.loss
        self._loss_fn: torch.nn.Module = eval(loss_info.classname)(**loss_info.params)
        self._analyzer = AnalyzerController(params.analyze_container, writer)
        self._device = device
        self.savedir = savedir
        self._iteration: int = 0
        
        # get the proper dataset
        train_ds, test_ds = create_dataset(dataset)
        test_len, excluded_len, self._excluded_from_test = len(test_ds), dataset.num_exemplars, None
        if dataset.num_exemplars > 0:
            self._excluded_from_test = test_ds.extract_subset_eq_distr(dataset.num_exemplars)
            test_len_reduced = len(test_ds)
            log.info(f"Len of total test set = {test_len}")
            log.info(f"Len of reduced test set = {test_len_reduced}, {100 * test_len_reduced / test_len}% of total test set")
            log.info(f"Len of extracted examples from test set = {excluded_len}")
        
        self.train_ds, self.test_ds = train_ds, test_ds
        
        exit_on_signal(signal.SIGTERM)

    @property
    def result(self) -> Mapping[str, Any]:
        return self._analyzer.result

    @abstractmethod
    def train_step(self) -> None:
        pass

    def fit(self, iterations: int) -> None:
        """
        Trains the algorithm.

        Resets the results if a previous fit completed successfully, evaluates the starting model and
        wraps the algorithm-specific fit procedure to ensure results saving after graceful or erroneous
        termination. Not intended to be overridden, see _fit

        Parameters:
        ----------
        iterations
            the number of iterations to train the algorithm for

        """
        assert iterations > self._iteration, "Num of rounds to perform must be greater of equal to the current round"
        try:
            self._fit(iterations)
            self.save_result()
        except SystemExit as e:
            log.warning(f"Training stopped at iteration {self._iteration}: {e}")

    @abstractmethod
    def _fit(self, iterations: int) -> None:
        """
        Defines the main training loop of any algorithm. Must be overridden to describe the algorithm-specific procedure

        Parameters
        ----------
        iterations
            the number of iterations to train the algorithm for
        """
        pass

    def load_from_checkpoint(self, checkpoint_path: str):
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data: dict = pickle.load(f)
                self.load_state_dict(checkpoint_data)
                log.info(f'Reloaded checkpoint from round {checkpoint_data["iteration"]}')
        except BaseException as err:
            log.warning(f"Unable to load from checkpoint, starting from scratch: {err}")
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Assigns buffers from state_dict into this client's state. If strict is True, then the keys of state_dict 
        must exactly match the keys returned by the algorithm's state state_dict() function.

        Args:
            state_dict (Mapping[str, Any]): a dict-like object containing the client's state buffers
            strict (bool, optional): whether to strictly enforce that the keys in state_dict match the 
            keys returned by the client's state state_dict() function. Defaults to True.
        """
        self._analyzer.load_state_dict(state_dict['analyzer_state'])
        self._iteration = state_dict['iteration']
        
    def state_dict(self, device: str = 'cpu') -> Mapping[str, Any]:
        """Returns a shallow copy of the current state of the algorithm

        Args:
            device (str, optional): the device (e.g. folder) to store states detatched from the checkpoint file. Defaults to 'cpu'.

        Returns:
            Mapping[str, Any]: a read-only reference to the dict object containing a shallow copy of system's state
        """
        return self._state_dict(Device(device))

    def _state_dict(self, device: Device) -> dict:
        data = {'iteration': self._iteration, 'analyzer_state': self._analyzer.state_dict()}
        return data

    @staticmethod
    def train(model: nn.Module, device: str, optimizer: nn.Module, loss_fn: nn.Module, data: DataLoader) -> None:
        model.train()
        for img, target in data:
            img, target = img.to(device), target.to(device)
            img, target = data.dataset.batchwise_transform((img, target))
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, target)

            loss.backward()
            optimizer.step()

    @staticmethod
    @torch.no_grad
    def test(model: nn.Module, meter: MeasureMeter, device: str, loss_fn, data: DataLoader) -> float:
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for img, target in data:
                img, target = img.to(device), target.to(device)
                img, target = data.dataset.batchwise_transform((img, target))
                # target = target.to(device)
                logits = model(img)
                test_loss += loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                meter.update(pred, target)
        test_loss = test_loss / len(data)
        return test_loss

    def save_result(self) -> None:
        """
        Saves the results of the training process
        """
        results_path = os.path.join(self.savedir, f"result.pkl")
        log.info(f"Saving results in {results_path}")
        save_pickle(self.result, results_path)
