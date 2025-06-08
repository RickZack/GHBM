import copy
from dataclasses import dataclass
from typing import List, Optional
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from src.algo.center_server.types import BaseServerState
from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients import Client
from src.models import Model
from src.utils.analyzers import Analyzer
from src.utils.state import Device

@dataclass   
class AdaBestServerState(BaseServerState):
    h: Optional[List[torch.Tensor]] = None
    last_avg_model: Optional[List[torch.Tensor]] = None


class AdaBestCenterServer(FedAvgCenterServer):
    """ Implements the center server in AdaBest algorithm, as proposed in Varno et al, AdaBest: Minimizing Client Drift 
    in Federated Learning via Adaptive Bias Estimation, ECCV 2022"""

    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig, lr_scheduler, 
                 beta: float, analyzer: Optional[Analyzer] = None):
        assert beta < 1, "Beta parameter of AdaBest must be < 1"
        h = [torch.zeros_like(p, device=device) for p in model.parameters()]
        first_model_p = [p.to(device, copy=True) for p in model.parameters()]
        state = AdaBestServerState(Device(device), h=h, last_avg_model=first_model_p)
        super().__init__(model, dataloader, device, optim, lr_scheduler, state, analyzer)
        self.__beta = beta

    @torch.no_grad
    def _aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        state: AdaBestServerState = self._state
        clients_data = [c.send_data() for c in clients]
        pseudo_grads = self._get_pseudo_grads([data['delta_params'] for data in clients_data], aggregation_weights)

        # avg model
        avg_model = [p.sub(pg) for p, pg in zip(self.model.params_with_grad(), pseudo_grads, strict=True)] 
        last_avg_model = state.last_avg_model
        
        h = [self.__beta * (old - new) for old, new in zip(last_avg_model, avg_model)]

        for p, avg, h in zip(self._model.parameters(), avg_model, h):
            p.copy_(avg - h)
            
        state.h = h
        state.last_avg_model = avg_model