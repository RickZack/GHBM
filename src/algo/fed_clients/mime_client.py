import copy
import torch
from dataclasses import dataclass
from typing import List, Optional, TypeVar
from src.optim.fed_optimizer import FederatedOptimizer
from src.models import Model
from src.algo.fed_clients.fedavg_client import FedAvgClient
from src.algo.fed_clients.types import BaseClientForwardState, BaseClientUpdateState, Loss, BaseClientTempState, \
                                       ClientDataLoaderIterator, IterationInfo, IterationNum, IterationType


@dataclass(frozen=True)
class LossMime(Loss):
    loss_s: Optional[torch.Tensor]
    
    @staticmethod
    def fromLoss(loss_client: Loss, loss_server: Loss):
        lc = loss_client.loss_task
        ls = loss_server.loss_task
        return LossMime(lc, lc, ls)
    
    def lossServer(self) -> Loss:
        return Loss(self.loss_s, self.loss_s)
    
    def lossClient(self) -> Loss:
        return Loss(self.backward_obj, self.loss_task)

@dataclass
class MimeClientTempState(BaseClientTempState):
    server_controls: Optional[List[torch.Tensor]] = None
    full_grad: Optional[List[torch.Tensor]] = None
    opt_state: Optional[List[torch.Tensor]] = None
    
@dataclass
class MimeClientUpdateState(BaseClientUpdateState):
    server_controls: List[torch.Tensor]
    server_model: Model
    opt_state: dict

class MimeClient(FedAvgClient):
    """ Implements a client in Mime algorithm, as proposed in Karimireddy et al., Breaking the centralized barrier
    for cross-device federated learning, NeurIPS 2021 """

    def __init__(self, *args, mime_lite: bool, loc_mime: bool = False, **kwargs):
        super().__init__(*args, **kwargs, temp_state=MimeClientTempState())
        self.__mime_lite = mime_lite
        self.__loc_mime = loc_mime

    def _get_client_update_state(self, optimizer: type, optimizer_args, loss_fn: torch.nn.Module, s_round: int,
                                 local_iterations: IterationNum) -> MimeClientUpdateState:
        temp_state: MimeClientTempState = self._temp_state
        state: BaseClientUpdateState = super()._get_client_update_state(optimizer, optimizer_args, loss_fn, s_round, local_iterations)
        model = temp_state.model
        server_model = copy.deepcopy(model)
        server_controls = temp_state.server_controls
        opt_state = temp_state.opt_state
        optimizer = optimizer(model.parameters(), **optimizer_args)
        temp_state.opt_state = temp_state.server_controls = None
        return MimeClientUpdateState(**vars(state), server_controls=server_controls,
                                     server_model=server_model, opt_state=opt_state)

    def receive_data(self, **kwargs):
        model, opt_state, controls = kwargs.get('model'), kwargs.get('opt_state'), kwargs.get('server_controls')
        assert (model and opt_state) or controls, f'{self.__class__.__name__} must receive at least (x,s) or c'
        if controls:
            assert not self.__mime_lite, "MimeLite does not receive controls"
        super().receive_data(**kwargs)
        
    def calculate_full_grad(self, loss_fn: torch.nn.Module, s_round: int):
        temp_state: MimeClientTempState = self._temp_state
        fullbatch_iterations = IterationNum(IterationType.epoch, 1)
        model: Model = temp_state.model
        state = BaseClientForwardState(model, s_round, loss_fn)
        for batch, iteration_info in ClientDataLoaderIterator(self.dataloader, fullbatch_iterations, s_round):
            batch = self._preprocess_batch(batch)
            loss = super()._client_forward(model, batch, state, iteration_info)
            super()._client_backward(model, loss, state, iteration_info, iteration_info.client.total_steps)
            
        # gradients are already scaled by the number of minibatches in a full batch. No need to do a clone:
        # model's gradients are reset by setting them to None, since we do not need gradients until next round
        temp_state.full_grad = list(model.grads())
        model.zero_grad(set_to_none=True)

    @torch.no_grad
    def _pre_client_update(self, model: Model, state: MimeClientUpdateState):
        # take layers hyperparameters from initialization, e.g. the one sent by server, not the one used by server,
        # as they can be different.
        opt_state = state.opt_state
        optimizer = state.optimizer
        for opt_state_p_group, current_p_group in zip(opt_state['param_groups'], optimizer.state_dict()['param_groups']):
            opt_state_p_group.update(current_p_group)
        optimizer.load_state_dict(opt_state)

    def _client_forward(self, model: Model, batch: tuple, state: MimeClientUpdateState,
                        current_iteration: IterationInfo) -> LossMime:
        c_loss: Loss = super()._client_forward(model, batch, state, current_iteration)
        s_loss: Loss = Loss(torch.tensor(0), torch.tensor(0))
        if not self.__mime_lite:
            s_loss: Loss = super()._client_forward(state.server_model, batch, state, current_iteration)
        return LossMime.fromLoss(c_loss, s_loss)

    def _client_backward(self, model: Model, loss: LossMime, state: MimeClientUpdateState,
                         current_iteration: IterationInfo, num_batches_accumulate: int):
        super()._client_backward(model, loss.lossClient(), state, current_iteration, num_batches_accumulate)
        if not self.__mime_lite:
            super()._client_backward(state.server_model, loss.lossServer(), state, current_iteration, num_batches_accumulate)
    
    @torch.no_grad        
    def _before_opt_step(self, model: Model, current_iteration: IterationInfo, state: MimeClientUpdateState):
        if not self.__mime_lite:
            s_model = state.server_model
            s_controls = state.server_controls
            for p, ci, c in zip(model.params_with_grad(), s_model.grads(), s_controls):
                p.grad.add_(c - ci)
            s_model.zero_grad()
    
    @torch.no_grad        
    def _opt_step(self, optimizer: FederatedOptimizer, current_iteration: IterationInfo, state: BaseClientUpdateState) -> None:
        optimizer.step(update_opt_state=self.__loc_mime)
        optimizer.zero_grad()

