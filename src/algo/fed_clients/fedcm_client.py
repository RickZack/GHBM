import torch
from dataclasses import dataclass
from typing import List, Optional
from src.models import Model
from src.algo.fed_clients.fedavg_client import FedAvgClient
from src.algo.fed_clients.types import BaseClientTempState, BaseClientUpdateState, IterationInfo, IterationNum

@dataclass
class FedCMClientTempState(BaseClientTempState):
    momentum: Optional[List[torch.Tensor]] = None
    
@dataclass
class FedCMClientUpdateState(BaseClientUpdateState):
    momentum: List[torch.Tensor]

class FedCMClient(FedAvgClient):
    """ Implements a client in FedCM algorithm, as proposed in Xu et al., FedCM: Federated Learning with
    Client-level Momentum, arXiv 2021 """

    def __init__(self, *args, alpha: float, **kwargs):
        super().__init__(*args, temp_state=FedCMClientTempState(), **kwargs)
        assert 0 < alpha < 1, "FedCM alpha must be between 0 and 1"
        self.__alpha = alpha
        
    def _get_client_update_state(self, optimizer: type, optimizer_args, loss_fn: torch.nn.Module, s_round: int,
                                 local_iterations: IterationNum) -> FedCMClientUpdateState:
        base_state = super()._get_client_update_state(optimizer, optimizer_args, loss_fn, s_round, local_iterations)
        temp_state: FedCMClientTempState = self._temp_state
        momentum = temp_state.momentum
        temp_state.momentum = None
        return FedCMClientUpdateState(**vars(base_state), momentum=momentum)
    
    def _before_opt_step(self, model: Model, current_iteration: IterationInfo, state: FedCMClientUpdateState):
            lr = state.optimizer.state_dict()['param_groups'][0]['lr']
            factor = 1 / (lr * current_iteration.client.total_steps)
            for g, m in zip(model.grads(), state.momentum):
                g.mul_(self.__alpha).add_(m, alpha=(1 - self.__alpha) * factor)
