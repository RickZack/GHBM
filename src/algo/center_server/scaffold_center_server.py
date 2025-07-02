import copy
from dataclasses import dataclass
from typing import List, Optional
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from src.algo.center_server.types import BaseServerState
from src.algo.center_server.fedavg_center_server import FedAvgCenterServer
from src.algo.fed_clients import SCAFFOLDClient
from src.models import Model
from src.utils.analyzers import Analyzer
from src.utils.state import Device

@dataclass   
class SCAFFOLDServerState(BaseServerState):
    controls: Optional[List[torch.Tensor]] = None


class SCAFFOLDCenterServer(FedAvgCenterServer):
    """ Implements the center server in SCAFFOLD algorithm, as proposed in Karimireddy et al., SCAFFOLD: Stochastic Controlled
    Averaging for Federated Learning """

    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig, lr_scheduler: DictConfig, 
                 analyzer: Optional[Analyzer] = None):
        controls = [torch.zeros_like(p, device=device) for p in model.parameters() if p.requires_grad]
        state = SCAFFOLDServerState(Device(device), controls=controls)
        super().__init__(model, dataloader, device, optim, lr_scheduler, state, analyzer)

    def _aggregation(self, clients: List[SCAFFOLDClient], aggregation_weights: List[float], s_round: int):
        state: SCAFFOLDServerState = self._state
        clients_data = [c.send_data() for c in clients]
        # obtain (pseudo) gradients
        pseudo_grads = self._get_pseudo_grads([data['delta_params'] for data in clients_data], aggregation_weights)
        # use pseudo grads to update the model
        self._update_model(pseudo_grads)

        controls = state.controls
        for data in clients_data:
            delta_c = data["delta_controls"]
            for sc, d in zip(controls, delta_c):
                sc.add_(d)
        
    def _send_data(self, *args, **kwargs) -> dict:
        state: SCAFFOLDServerState = self._state
        # deepcopy because each client could be on a different device
        to_send = {'server_controls': copy.deepcopy(state.controls)}
        return super()._send_data(data=to_send)
