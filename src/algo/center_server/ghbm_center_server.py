from dataclasses import dataclass, field
from typing import List, Optional, Set
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from src.algo.center_server.types import BaseServerState
from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients import Client
from src.models import Model
from src.utils.analyzers import Analyzer
import copy

from src.utils.state import Device, move_to

@dataclass   
class GHBMServerState(BaseServerState):
    past_models_queue: List[List[torch.Tensor]] = field(default_factory=list)
    momentum: Optional[torch.Tensor] = None
    
@dataclass   
class FedHBMServerState(BaseServerState):
    init_model: List[torch.Tensor] = None
    clients_set: Set[int] = field(default_factory=set)


class GHBMCenterServer(FedAvgCenterServer):
    """ Implements the center server in GHBM algorithm, as proposed in Zaccone et al., 
    Communication-Efficient Federated Learning with Generalized Heavy-Ball Momentum """
    
    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig, lr_scheduler: DictConfig,
                 tau: int, analyzer: Optional[Analyzer] = None):
        state = GHBMServerState(Device(device))
        super().__init__(model, dataloader, device, optim, lr_scheduler, state, analyzer)
        self.__tau = tau

    @torch.no_grad
    def _aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        state: GHBMServerState = self._state
        models_queue = state.past_models_queue
        if len(models_queue) >= self.__tau:
            models_queue.pop(0)
        models_queue.append([p.to('cpu', copy=True) for p in self.model.params_with_grad()])
        super()._aggregation(clients, aggregation_weights, s_round)
        state.momentum = self.__calculate_momentum(models_queue)

        layers_momentum = {n: m for (n, param), m in zip(self.model.named_parameters(), state.momentum)
                            if len(param.shape)>=2 and param.requires_grad}
        self._analyzer('checkrank', tensors=layers_momentum, s_round=s_round, prefix='momentum')

    def _send_data(self, *args, **kwargs) -> dict:
        state: GHBMServerState = self._state
        data = {'momentum': state.momentum}
        return super()._send_data(data=data)
    
    def __calculate_momentum(self, models_queue: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        momentum = []
        if len(models_queue) >= self.__tau:
            # deepcopy because each client could be on a different device
            old_model = move_to(models_queue[0].copy(), self.device)
            last_model = self.model.params_with_grad()
            momentum = [(old - last).div_(self.__tau) for old, last in zip(old_model, last_model)]
        return momentum


class FedHBMCenterServer(FedAvgCenterServer):

    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig, lr_scheduler: DictConfig,
                 pseudo_grad_compressor: DictConfig, analyzer: Optional[Analyzer] = None):
        state = FedHBMServerState(Device(device), init_model=[p.to(device, copy=True) for p in model.parameters()])
        super().__init__(model, dataloader, device, optim, lr_scheduler, pseudo_grad_compressor, state, analyzer)

    def _send_data(self, client_id: int, *args, **kwargs) -> dict:
        data = {}
        data.update(self.__get_init_model_dict(client_id))
        return super()._send_data(data=data)

    def __get_init_model_dict(self, client_id: int) -> dict:
        state: FedHBMServerState = self._state
        data = {}
        if client_id not in state.clients_set:
            # deepcopy because each client could be on a different device
            data['anchor_model'] = copy.deepcopy(state.init_model)
            state.clients_set.add(client_id)
        return data
    