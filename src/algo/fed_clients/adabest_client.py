import torch
from src.models import Model
from dataclasses import dataclass
from typing import List, Optional
from src.algo.fed_clients.fedavg_client import FedAvgClient
from src.algo.fed_clients.types import BaseClientState, BaseClientUpdateState, BaseClientUpdateStateType, IterationNum, Loss, IterationInfo


@dataclass
class AdaBestClientState(BaseClientState):
    prev_grads: Optional[List[torch.Tensor]] = None
    last_round: int = 0
    
@dataclass
class AdaBestClientUpdateState(BaseClientUpdateState):
    prev_params: List[torch.Tensor]
    current_params: List[torch.Tensor]
    prev_grads: List[torch.Tensor]
    last_round: int
    
@dataclass(frozen=True)
class AdaBestLoss(Loss):
    linear_penalty: torch.Tensor

class AdaBestClient(FedAvgClient):
    """ Implements a client in AdaBest algorithm, as proposed in Varno et al., AdaBest: Minimizing Client Drift
    in Federated Learning via Adaptive Bias Estimation, ECCV 2022 """

    def __init__(self, *args, mu: float, **kwargs):
        super().__init__(*args, state=AdaBestClientState(), **kwargs)
        self.__mu = mu
    
    def _get_client_update_state(self, optimizer: type, optimizer_args, loss_fn: torch.nn.Module, s_round: int,
                                 local_iterations: IterationNum) -> BaseClientUpdateStateType:                
        state: AdaBestClientState = self._state
        up_state: BaseClientUpdateState = super()._get_client_update_state(optimizer, optimizer_args, loss_fn, s_round, local_iterations)

        model = self._temp_state.model
        current_params = list(model.params_with_grad())
        prev_params = list(model.params_with_grad(copy=True))
        prev_grads = state.prev_grads or [torch.zeros_like(p, device=self.device.name) for p in current_params]
        last_round = state.last_round

        return AdaBestClientUpdateState(**vars(up_state), prev_params=prev_params, current_params=current_params, 
                                        prev_grads=prev_grads, last_round=last_round)

    def _post_client_forward(self, model: Model, batch: tuple, loss: Loss, state: AdaBestClientUpdateState, 
                             current_iteration: IterationInfo) -> AdaBestLoss:
        linear_p = 0
        for param, grad in zip(state.current_params, state.prev_grads):
            linear_p += torch.sum(param * grad)

        task_loss = loss.loss_task
        return AdaBestLoss(task_loss - linear_p, task_loss, linear_p)
    
    @torch.no_grad
    def _post_client_update(self, model: Model, state: AdaBestClientUpdateState):
        super()._post_client_update(model, state)
        c_state: AdaBestClientState = self._state
        rounds_elapsed = (state.s_round - state.last_round)
        for prev_g, new_p, prev_p in zip(state.prev_grads, state.current_params, state.prev_params):
            prev_g.div_(rounds_elapsed).add_(new_p - prev_p, alpha=-self.__mu)
            
        c_state.prev_grads = state.prev_grads
        c_state.last_round = state.last_round
