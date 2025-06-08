import copy
from typing import Any, List, Mapping, Tuple, Optional
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import torch
from src.algo.center_server.types import BaseServerState, BaseServerStateType
from src.optim import *
from torch.utils.data import DataLoader
from src.algo.fed_clients.base_client import Client
from src.models import Model
from src.utils import MeasureMeter
from src.utils.analyzers import Analyzer, ChainedAnalyzer
from src.utils.state import Device
import logging

log = logging.getLogger(__name__)


class CenterServer(ABC):
    """ Base (abstract) class for a center server in any FL algorithm """

    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig, lr_scheduler: dict,
                 state: Optional[BaseServerStateType] = None, analyzer: Optional[Analyzer] = None):
        self._state = state or BaseServerState()
        self._state._device = Device(device)
        self._model = model.to(device)
        self._dataloader = dataloader
        self._analyzer = analyzer or ChainedAnalyzer.empty()
        optimizer_class = eval(optim.classname)
        optimizer_args = optim.args
        self._opt: FederatedOptimizer = optimizer_class(self.model.parameters(), **optimizer_args)
        
        scheduler_class: type = eval(lr_scheduler.classname)
        scheduler_args = lr_scheduler.args
        self._lr_scheduler: torch.optim.lr_scheduler._LRScheduler = scheduler_class(self._opt, **scheduler_args)
        
        self._state.model_state = self._model.state_dict()
        self._state.opt_state = self._opt.state_dict()
        self._state.lr_scheduler_state = self._lr_scheduler.state_dict()
        
        
    @property
    def device(self) -> Device:
        return self._state._device

    @device.setter
    def device(self, device: Device):
        self._state.to(device)
        self._model.to(device.name)

    @property
    def model(self):
        return self._model
    
    def trigger_validation(self, s_round: int):
        """Triggers a validation event for the proper analyzer, forcing the validation

        Args:
            s_round (int): the current round of Federated training
        """
        self._analyzer('validation', model=self._model, loss_fn=torch.nn.CrossEntropyLoss(), s_round=s_round,
                device=self.device.name, dataloader=self._dataloader, force=True)
        
    
    @abstractmethod
    def _aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        pass

    @torch.no_grad
    def aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        """
        Aggregate the client's data according to their weights

        Parameters
        ----------
        clients
            the clients whose data have to be aggregated
        aggregation_weights
            the weights corresponding to clients
        s_round
            the current round of the server
        """
        self._aggregation(clients, aggregation_weights, s_round)
        self._analyzer('validation', model=self._model, loss_fn=torch.nn.CrossEntropyLoss(), s_round=s_round,
                       device=self.device.name, dataloader=self._dataloader)

    def send_data(self, *args, **kwargs) -> dict:
        """
        Sends out the current data of the central server. Wrapper for protected method _send_data.

        Parameters
        ----------
        args
            positional arguments specific to server subclass
        kwargs
            keyword arguments specific to server subclass

        Returns
        -------
        a dictionary containing the current data of the center server
        """
        data = self._send_data(*args, data={}, **kwargs)
        self._analyzer('send_data', data=data, server=self)
        return data

    def _send_data(self, *args, data: dict, **kwargs) -> dict:
        """
        Sends out the current data of the central server. To be used to send current round data to FL clients.
        For any specific FL algorithm, (CenterServer) _send_data must output the data needed by the specific
        client receive_data

        Returns
        -------
        a dictionary containing the current data of the center server
        """
        to_send = {"model": copy.deepcopy(self._model)}
        to_send.update(data)
        return to_send

    @abstractmethod
    def validation(self, loss_fn) -> Tuple[float, MeasureMeter]:
        """
        Validates the center server model

        Parameters
        ----------
        loss_fn the loss function to be used

        Returns
        -------
        a tuple containing the value of the loss function and a reference to the center server MeasureMeter object
        """
        pass

    def state_dict(self) -> Mapping[str, Any]:     
        """Returns a shallow copy of the current state of the server

        Returns:
            Mapping[str, Any]: a read-only reference to the dict object containing a shallow copy of server's state
        """
        return self._state.state_dict()

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True, tentative: bool = False) -> None:
        """Assigns buffers from state_dict into this server's state. If strict is True, then the keys of state_dict 
        must exactly match the keys returned by the server's state state_dict() function.

        Args:
            state (Mapping[str, Any]): a dict-like object containing the server's state buffers
            strict (bool, optional): whether to strictly enforce that the keys in state_dict match the 
            keys returned by the server's state state_dict() function. Defaults to True.
            tentative (bool, optional): whether to only try to load the state dict, used when not only this 
            server's state must be loaded without failure. Defaults to False.
        """
        self._state.load_state_dict(state, strict, tentative)
        self._model.load_state_dict(state["model_state"], strict=strict)
        self._opt.load_state_dict(state["opt_state"])
        self._lr_scheduler.load_state_dict(state['lr_scheduler_state'])
        # Ok, at this point it is safe to reload the state
        self._state.load_state_dict(state, strict)
        # The following step is necessary because when loading a state_dict, the parameters from the state_dict 
        # are copied inside the model. Therefore, after loading the state_dict, parameters in model are decoupled 
        # from the ones in state_dict, hence our state should not point to the parameters in state_dict, 
        # but to the same parameters in model.
        self._state.model_state = self._model.state_dict()
        self._state.opt_state = self._opt.state_dict()

