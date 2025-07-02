from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import List, Optional
import torch
from src.models import Model
from src.algo.fed_clients.types import BaseClientState, BaseClientTempState, BaseClientUpdateState, IterationInfo, IterationNum
from src.algo.fed_clients.fedavg_client import FedAvgClient
   
@dataclass
class GHBMClientUpdateState(BaseClientUpdateState):
    momentum: Optional[List[torch.Tensor]] = None
            
class GHBMBaseClient(FedAvgClient, ABC):
    """ Implements a client in GHBM algorithm, as proposed in Zaccone et al., 
    Communication-Efficient Federated Learning with Generalized Heavy-Ball Momentum """
    
    def __init__(self, *args, beta: float, nesterov: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.__beta = beta
        self.__nesterov = nesterov
        
    def _get_client_update_state(self, optimizer: type, optimizer_args, loss_fn: torch.nn.Module, s_round: int,
                                 local_iterations: IterationNum) -> GHBMClientUpdateState:
        state: BaseClientUpdateState = super()._get_client_update_state(optimizer, optimizer_args, loss_fn, s_round, local_iterations)
        state = GHBMClientUpdateState(**vars(state))
        self._set_momentum(state.model, state)
        return state
    
    @abstractmethod
    def _set_momentum(self, model: Model, state: GHBMClientUpdateState) -> None:
        ...

    @torch.no_grad
    def _pre_client_forward(self, model: Model, batch: tuple, state: GHBMClientUpdateState, current_iteration: IterationInfo):
        if self._is_batch_complete(current_iteration, self._num_batches_accumulate):
            self._set_momentum(model, state)
            if self.__nesterov:
                factor = self.__beta / (current_iteration.client.total_steps)
                for p, c in zip(model.params_with_grad(), state.momentum):
                    p.add_(c, alpha=-factor)
            
    def _before_opt_step(self, model: Model, current_iteration: IterationInfo, state: GHBMClientUpdateState):
        if not self.__nesterov:
            lr = state.optimizer.state_dict()['param_groups'][0]['lr']
            factor = self.__beta / (lr * current_iteration.client.total_steps)
            for p, c in zip(model.grads(), state.momentum):
                p.add_(c, alpha=factor) 
    
@dataclass
class GHBMClientTempState(BaseClientTempState):
    momentum: Optional[List[torch.Tensor]] = None

class GHBMClient(GHBMBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, temp_state=GHBMClientTempState(), **kwargs)
        
    def _set_momentum(self, model: Model, state: GHBMClientUpdateState) -> None:
        momentum = state.momentum
        if momentum is None:
            # This branch is executed only the before the first step,
            # then the momentum will be cached into state.momentum
            temp_state: GHBMClientTempState = self._temp_state
            state.momentum = temp_state.momentum
            if temp_state.momentum is None:
                state.momentum = [torch.zeros_like(p) for p in model.params_with_grad()]

@dataclass
class LocalGHBMClientState(BaseClientState):
    anchor_model: Optional[List[torch.Tensor]] = None
    
@dataclass
class LocalGHBMClientTempState(BaseClientTempState):
    anchor_model: Optional[List[torch.Tensor]] = None
    
class LocalGHBMClient(GHBMBaseClient):
    """ Implements a client in LocalGHBM algorithm, as proposed in Zaccone et al., 
    Communication-Efficient Federated Learning with Generalized Heavy-Ball Momentum """
    
    def __init__(self, *args, C: float, variant: str = 'standard', **kwargs):
        super().__init__(*args, state=LocalGHBMClientState(), temp_state=LocalGHBMClientTempState(), **kwargs)
        self._participation_ratio = C
        self._mode = GHBVariant(variant)
        self._current_mode = self._mode
        
    def _get_anchor_model(self, model: Model) -> Optional[List[torch.Tensor]]:
        # Check that server sent all the necessary but no more
        c_state: LocalGHBMClientState = self._state
        temp_state: LocalGHBMClientTempState = self._temp_state     
        if self._current_mode is GHBVariant.random_init:
            assert temp_state.anchor_model is None, "when using random init server must not send the anchor model"
            anchor_model = Model(model.model_info, model.num_classes).to(self.device.name)
            anchor_model = list(anchor_model.params_with_grad())
        elif self._current_mode is GHBVariant.shared_init:
            assert temp_state.anchor_model is not None, "server did not provide anchor model"
            anchor_model = temp_state.anchor_model
            temp_state.anchor_model = None
        else:
            assert temp_state.anchor_model is None, "in standard mode server must not send the anchor model"
            anchor_model = c_state.anchor_model
            # Set the new anchor model for the next round to be the model just received from server
            c_state.anchor_model = list(model.params_with_grad(copy=True))
        self._current_mode = GHBVariant.standard
        return anchor_model
            
    def _set_momentum(self, model: Model, state: GHBMClientUpdateState) -> None:
        if state.momentum is None:
            # This branch is executed only the before the first step,
            # then the momentum will be cached into state.momentum
            anchor_model = self._get_anchor_model(model)
            end_model = list(model.params_with_grad())
            if anchor_model is None:
                state.momentum = [torch.zeros_like(p) for p in model.params_with_grad()]
            else:
                state.momentum = [self._participation_ratio * (a-c) for a, c in zip(anchor_model, end_model)]

class GHBType(StrEnum):
    ghb = auto()
    local_ghb = auto()
    fedhbm = auto()

class GHBVariant(StrEnum):
    standard = auto()
    shared_init = auto()
    random_init = auto()

@dataclass
class GHBInstance:
    type: GHBType
    variant: GHBVariant

    def __post_init__(self):
        if self.type is GHBType.ghb:
            assert self.variant is GHBVariant.standard, f"Invalid GHB instance {self.type}-{self.variant}"
            
class FedHBMClient(LocalGHBMClient): 
    """ Implements a client in FedHBM algorithm, as proposed in Zaccone et al., 
    Communication-Efficient Federated Learning with Generalized Heavy-Ball Momentum """
    
    def _get_anchor_model(self, model: Model) -> Optional[List[torch.Tensor]]:
        # Check that server sent all the necessary but no more
        c_state: LocalGHBMClientState = self._state
        temp_state: LocalGHBMClientTempState = self._temp_state     
        if self._current_mode is GHBVariant.random_init:
            assert temp_state.anchor_model is None, "when using random init server must not send the anchor model"
            anchor_model = Model(model.model_info, model.num_classes).to(self.device.name)
            anchor_model = list(anchor_model.params_with_grad())
        elif self._current_mode is GHBVariant.shared_init:
            assert temp_state.anchor_model is not None, "server did not provide anchor model"
            anchor_model = temp_state.anchor_model
            temp_state.anchor_model = None
        else:
            assert temp_state.anchor_model is None, "in standard mode server must not send the anchor model"
            anchor_model = c_state.anchor_model
        self._current_mode = GHBVariant.standard
        return anchor_model
    
    def _set_momentum(self, model: Model, state: GHBMClientUpdateState) -> None:  
        # Momentum is different at each optimization step
        anchor_model = self._get_anchor_model(model)
        end_model = list(model.params_with_grad()) # current model
        if anchor_model is None:
            state.momentum = [torch.zeros_like(p) for p in model.params_with_grad()]
        else:
            state.momentum = [self._participation_ratio * (a-c) for a, c in zip(anchor_model, end_model)]
    
    def _post_client_update(self, model: Model, state: GHBMClientUpdateState):
        super()._post_client_update(model, state)
        c_state: LocalGHBMClientState = self._state
        c_state.anchor_model = list(model.params_with_grad(copy=True))
