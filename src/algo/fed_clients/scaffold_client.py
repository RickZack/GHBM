import torch
from dataclasses import dataclass
from typing import List, Optional
from src.models import Model
from src.algo.fed_clients.types import BaseClientState, BaseClientTempState, BaseClientUpdateState
from src.algo.fed_clients import FedAvgClient
from src.algo.fed_clients.types import ClientDataLoaderIterator, IterationInfo, IterationNum, IterationType


@dataclass
class SCAFFOLDClientState(BaseClientState):
    client_controls: Optional[List[torch.Tensor]] = None
    
@dataclass
class SCAFFOLDClientTempState(BaseClientTempState):
    server_controls: Optional[List[torch.Tensor]] = None
    delta_controls: Optional[List[torch.Tensor]] = None
    
@dataclass
class SCAFFOLDClientUpdateState(BaseClientUpdateState):
    current_params: List[torch.Tensor]
    client_controls: List[torch.Tensor]
    server_controls: List[torch.Tensor]
    new_client_controls: Optional[List[torch.Tensor]]

class SCAFFOLDClient(FedAvgClient):
    """ Implements a client in SCAFFOLD algorithm, as proposed in Karimireddy et al., SCAFFOLD: Stochastic Controlled
    Averaging for Federated Learning, ICML 2020 """

    def __init__(self, *args, fullbatch_controls: bool, **kwargs):
        super().__init__(*args, state=SCAFFOLDClientState(), temp_state=SCAFFOLDClientTempState(), **kwargs)
        self.__fullbatch_controls = fullbatch_controls
        
    def _get_client_update_state(self, optimizer: type, optimizer_args, loss_fn: torch.nn.Module, s_round: int,
                                 local_iterations: IterationNum) -> SCAFFOLDClientUpdateState:
        base_state = super()._get_client_update_state(optimizer, optimizer_args, loss_fn, s_round, local_iterations)
        state: SCAFFOLDClientState = self._state
        temp_state: SCAFFOLDClientTempState = self._temp_state
        params = list(base_state.model.params_with_grad())
        c_controls = state.client_controls
        if c_controls is None:
            c_controls = [torch.zeros_like(p, device=self.device.name) for p in params]
        s_controls = temp_state.server_controls
        new_c_controls = None
        if not self.__fullbatch_controls:
            # set new controls' buffer to be filled during client update
            new_c_controls = [torch.zeros_like(p, device=self.device.name) for p in params]
        temp_state.clear()
            
        return SCAFFOLDClientUpdateState(**vars(base_state), current_params=params, client_controls=c_controls, 
                                         server_controls=s_controls, new_client_controls=new_c_controls)

    def _pre_client_update(self, model: Model, state: SCAFFOLDClientUpdateState):
        """Calculates the fullbatch gradients to be used as future client controls in following rounds,
        if implementing option I of the algorithm. Option II is implemented by accumulating gradients at
        client's parameters in _client_step() method.

        Args:
            model (Model): the model involved
            state (SCAFFOLDClientUpdateState): the current client update state
        """
        if self.__fullbatch_controls:
            fullbatch_iterations = IterationNum(IterationType.epoch, 1)
            for batch, iteration_info in ClientDataLoaderIterator(self.dataloader, fullbatch_iterations):
                batch = self._preprocess_batch(batch)
                loss = self._client_forward(model, batch, state, iteration_info)
                self._client_backward(model, loss, state, iteration_info, iteration_info.total_steps)

            # gradients are already scaled by the number of minibatches in a full batch, see FedAvgClient._client_backward()
            state.new_client_controls = list(model.grads(copy=True))
            
    def _before_opt_step(self, model: Model, current_iteration: IterationInfo, state: SCAFFOLDClientUpdateState):
        if not self.__fullbatch_controls:
            # accumulate gradients for updated client controls
            for cc, g in zip(state.new_client_controls, model.grads()):
                cc.add_(g, alpha=1 / current_iteration.client.total_steps)

        # apply correction to grads
        for g, c, ci in zip(model.grads(), state.server_controls, state.client_controls):
            g.add_(c - ci)

    @torch.no_grad
    def _post_client_update(self, model: Model, state: SCAFFOLDClientUpdateState):
        """Calculates the delta of this client's controls and fills the client's temp state

        Args:
            model (Model): the model involved
            state (SCAFFOLDClientUpdateState): the current client update state
        """
        super()._post_client_update(model, state)
        c_state: SCAFFOLDClientState = self._state
        temp_state: SCAFFOLDClientTempState = self._temp_state
        c_state.client_controls = state.new_client_controls

        delta = [new - old for new, old in zip(state.new_client_controls, state.client_controls)] 
        temp_state.delta_controls = delta