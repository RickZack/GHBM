from dataclasses import dataclass
import torch
from typing import List, Optional
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from src.algo.center_server.types import BaseServerState
from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients import Client
from src.models import Model
from src.utils.analyzers import Analyzer
from src.utils.state import Device

@dataclass   
class FedDynServerState(BaseServerState):
    h: Optional[List[torch.Tensor]] = None


class FedDynCenterServer(FedAvgCenterServer):
    """ Implements the center server in FedDyn algorithm, as proposed in Acar et al, Federated Learning
    Based on Dynamic Regularization, ICLR 2021 """

    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig, lr_scheduler: DictConfig,
                 alpha: float, num_clients: int, analyzer: Optional[Analyzer] = None):
        h = [torch.zeros_like(p, device=device) for p in model.parameters()]
        super().__init__(model, dataloader, device, optim, lr_scheduler,
                         FedDynServerState(Device(device), h=h), analyzer)
        self.__alpha = alpha
        self.__num_clients = num_clients
        
    @torch.no_grad
    def _aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        # compute the sum of all the model parameters of the clients involved in training
        state: FedDynServerState = self._state
        clients_data = [c.send_data() for c in clients]
        delta_theta = self._get_pseudo_grads([data['delta_params'] for data in clients_data], aggregation_weights)
        # update the h parameter
        for h_p, theta in zip(state.h, delta_theta):
            h_p.add_(theta, alpha=-(self.__alpha / self.__num_clients))
        # update the server model
        self._update_model(delta_theta)
        for p, h, in zip(self.model.params_with_grad(), state.h, strict=True):
            p.sub_(1. / self.__alpha * h)