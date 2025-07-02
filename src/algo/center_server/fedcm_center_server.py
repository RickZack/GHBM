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
class FedCMServerState(BaseServerState):
    momentum: Optional[List[torch.Tensor]] = None


class FedCMCenterServer(FedAvgCenterServer):
    """ Implements the center server in FedCM algorithm, as proposed in Xu et al., 
    FedCM: Federated Learning with Client-level Momentum, arXiv 2021 """

    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig,
                 lr_scheduler: DictConfig, analyzer: Optional[Analyzer] = None):
        momentum = [torch.zeros_like(p, device=device) for p in model.parameters() if p.requires_grad]
        super().__init__(model, dataloader, device, optim, lr_scheduler,
                         FedCMServerState(Device(device), momentum=momentum), analyzer)

    def _send_data(self, *args, **kwargs) -> dict:
        # deepcopy because each client could be on a different device
        state: FedCMServerState = self._state
        data = {"momentum": torch.clone(p) for p in state.momentum}
        return super()._send_data(data=data)

    def _aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        state: FedCMServerState = self._state
        clients_data = [c.send_data() for c in clients]
        # obtain (pseudo) gradients
        pseudo_grads = self._get_pseudo_grads([data['delta_params'] for data in clients_data], aggregation_weights)

        # use pseudo grads to update the model
        self._update_model(pseudo_grads)
        # client side it needs to be divided by (lr * num_steps)
        state.momentum = pseudo_grads
