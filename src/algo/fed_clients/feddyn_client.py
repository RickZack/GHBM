import torch
from src.models import Model
from dataclasses import dataclass
from typing import List, Optional
from torch.nn import functional as F
from src.algo.fed_clients.types import IterationInfo, BaseClientUpdateState, BaseClientState, IterationNum, Loss
from src.algo.fed_clients import FedAvgClient

@dataclass
class FedDynUpdateState(BaseClientUpdateState):
    prev_params: List[torch.Tensor]
    current_params: List[torch.Tensor]
    prev_grads: List[torch.Tensor]

@dataclass
class FedDynClientState(BaseClientState):
    prev_grads: Optional[List[torch.Tensor]] = None
    
@dataclass(frozen=True)
class FedDynLoss(Loss):
    linear_penalty: torch.Tensor
    quadratic_penalty: torch.Tensor
    
class FedDynClient(FedAvgClient):
    """ Implements a client in FedDyn algorithm, as proposed in Acar et al, Federated Learning
    Based on Dynamic Regularization, ICLR 2021"""

    def __init__(self, *args, alpha: float, **kwargs):
        super().__init__(*args, state=FedDynClientState(), **kwargs)
        self.__alpha = alpha
        
    def _get_client_update_state(self, optimizer: type, optimizer_args, loss_fn: torch.nn.Module, s_round: int,
                                 local_iterations: IterationNum) -> FedDynUpdateState:
        base_state = super()._get_client_update_state(optimizer, optimizer_args, loss_fn, s_round, local_iterations)
        current_params = list(base_state.model.params_with_grad())
        prev_params = list(base_state.model.params_with_grad(copy=True))
        prev_grads = self._state.prev_grads or  [torch.zeros_like(p, device=self.device.name) for p in current_params]

        return FedDynUpdateState(**vars(base_state), prev_params=prev_params, current_params=current_params, prev_grads=prev_grads)

    def _post_client_forward(self, model: Model, batch: tuple, loss: Loss, state: FedDynUpdateState, 
                             current_iteration: IterationInfo) -> FedDynLoss:
        linear_p = 0
        for param, grad in zip(state.current_params, state.prev_grads):
            linear_p += torch.sum(param * grad)

        quadratic_p = 0
        for cur_param, prev_param in zip(state.current_params, state.prev_params):
            quadratic_p += F.mse_loss(cur_param, prev_param, reduction='sum')

        task_loss = loss.loss_task
        quadratic_p = self.__alpha / 2. * quadratic_p

        return FedDynLoss(task_loss - linear_p + quadratic_p, task_loss, linear_p, quadratic_p)
    
    @torch.no_grad
    def _post_client_update(self, model: Model, state: FedDynUpdateState):
        super()._post_client_update(model, state)
        client_state: FedDynClientState = self._state
        for prev_g, new_p, prev_p in zip(state.prev_grads, state.current_params, state.prev_params):
            prev_g.add_(new_p - prev_p, alpha=-self.__alpha)
            
        client_state.prev_grads = state.prev_grads