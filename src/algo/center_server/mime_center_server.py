import copy
from dataclasses import dataclass
from typing import List, Optional
import torch
from omegaconf import DictConfig
from torch.optim import *
from torch.utils.data import DataLoader
from src.algo.center_server.types import BaseServerState
from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients import MimeClient
from src.models import Model
from src.utils.analyzers import Analyzer
from src.utils.state import Device

@dataclass   
class MimeServerState(BaseServerState):
    controls: Optional[List[torch.Tensor]] = None

class MimeCenterServer(FedAvgCenterServer):
    """ Implements the center server in Mime algorithm family, as proposed in Karimireddy et al., 
    Breaking the centralized barrier for cross-device federated learning, NeurIPS 2021 """

    def __init__(self, model: Model, dataloader: DataLoader, device: str, optim: DictConfig, lr_scheduler: DictConfig, 
                 analyzer: Optional[Analyzer] = None):
        super().__init__(model, dataloader, device, optim, lr_scheduler,
                         MimeServerState(Device(device)), analyzer)
        self.lr = optim.args.lr

    def _send_data(self, send_model_and_state: bool = False, send_controls: bool = False, *args, **kwargs) -> dict:
        assert send_model_and_state or send_controls, f'{self.__class__.__name__} is not sending anything'
        state: MimeServerState = self._state
        data = {}
        if send_model_and_state:
            data.update({'opt_state': copy.deepcopy(self._opt.partial_state_dict()),
                         'model': copy.deepcopy(self._model)})
        if send_controls:
            # deepcopy because each client could be on a different device
            data.update({'server_controls': copy.deepcopy(state.controls)})
        return data

    def __average_full_grads(self, clients_data: List[dict]) -> List[torch.Tensor]:
        total_weight = len(clients_data)
        aggregation_weights = [1. / total_weight for _ in range(len(clients_data))]

        # average full grads
        full_grads_avg = [torch.zeros_like(p) for p in self.model.parameters() if p.requires_grad]
        for data, w in zip(clients_data, aggregation_weights):
            grads = data['full_grad']
            [c.add_(g, alpha=w) for c, g in zip(full_grads_avg, grads)]
        return full_grads_avg

    torch.no_grad
    def aggregate_fullgrads(self, clients: List[MimeClient]):
        state: MimeServerState = self._state
        clients_data = [c.send_data(send_model=False, send_fullgrads=True) for c in clients]
        state.controls = self.__average_full_grads(clients_data)

    @torch.no_grad
    def _aggregation(self, clients: List[MimeClient], aggregation_weights: List[float], s_round: int):
        state: MimeServerState = self._state
        clients_data = [c.send_data() for c in clients]

        # obtain gradients
        pseudo_grads = self._get_pseudo_grads([data['delta_params'] for data in clients_data], aggregation_weights)

        # optimizer state update
        full_grads_avg = state.controls
        # set gradients to model to make optimizer update its state
        # but avoid changing the model
        for p, g in zip(self.model.params_with_grad(), full_grads_avg):
            p.grad = g
        self._opt.step(update_model=False)
        self._opt.zero_grad(set_to_none=True)

        # global model update
        for p, g in zip(self.model.params_with_grad(), pseudo_grads):
            p.add_(g, alpha=-self.lr)