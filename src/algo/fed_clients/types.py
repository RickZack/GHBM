from collections.abc import Iterator
from typing import Any, Tuple, TypeVar, List, Optional
from dataclasses import dataclass, field
import torch
from src.optim.fed_optimizer import FederatedOptimizer
from src.models import Model
from src.utils import State
from enum import StrEnum, auto

class IterationType(StrEnum):
    """Enumeration defining the type of iteration in local training (steps or epochs)"""
    step = auto()
    epoch = auto()

@dataclass(frozen=True)
class IterationNum:
    """Dataclass defining the type and the number of local training iterations"""
    type: IterationType
    num: int
    
    def to_steps(self, steps_in_epoch: int) -> int:
        steps = self.num
        if self.type is IterationType.epoch:
            steps *= steps_in_epoch
        return steps

@dataclass(frozen=True)
class LocalIterationInfo:
    step: int
    total_steps: int
    epoch: int = field(init=False)
    step_in_epoch: int = field(init=False) 
    
    def __post_init__(self):
        epoch, step_in_epoch = divmod(self.step - 1, self.total_steps)
        object.__setattr__(self, 'epoch', epoch + 1) # make epoch start from 1 instead of 0, as step is 1 the first time
        object.__setattr__(self, 'step_in_epoch', step_in_epoch + 1)
        
@dataclass(init=False, frozen=True)
class GlobalIterationInfo:
    round: int
    step: int
    
    def __init__(self, round: int, steps_in_epoch: int, local_step: int) -> None:
        object.__setattr__(self, 'round', round)
        object.__setattr__(self, 'step', (round - 1) * steps_in_epoch + local_step)
            
@dataclass(frozen=True)
class IterationInfo:
    client: LocalIterationInfo
    server: GlobalIterationInfo
    
    
class IterationInfoIterator(Iterator[IterationInfo]):
    """Iterator over the steps of local training"""
    def __init__(self, iterations: IterationNum, steps_in_epoch: int, round: int,
                 num_batches_accumulate: int = 1) -> None:
        self.__round = round
        iterations = IterationNum(iterations.type, iterations.num * num_batches_accumulate)
        self.__total_steps = iterations.to_steps(steps_in_epoch)
        self.__current_step = 0
    
    def __next__(self):
        if self.__current_step < self.__total_steps:
            self.__current_step += 1
            local_info = LocalIterationInfo(self.__current_step, self.__total_steps)
            global_info = GlobalIterationInfo(self.__round, self.__total_steps, local_info.step)
            return IterationInfo(local_info, global_info)
        raise StopIteration
    
class ClientDataLoaderIterator(Iterator[Tuple[Any, IterationInfo]]):
    """Iterator over a DataLoader for clients local training.
    
    Iterates over a DataLoader for a given number of iterations (steps or epochs), independently on the
    size of the local dataset. At each steps returns the batch retrieved from the original DataLoader
    iterator and a IterationInfo object defining the current step in training.
    """
    def __init__(self, dataloader: torch.utils.data.DataLoader, iterations: IterationNum, s_round: int,
                 num_batches_accumulate: int = 1) -> None:
        self.__dataloader = dataloader
        self.__iterator = iter(dataloader)
        self.__itinfo_iterator = IterationInfoIterator(iterations, len(dataloader), s_round, num_batches_accumulate)
    
    def __next__(self):
        iteration = next(self.__itinfo_iterator)
        try:
            data = next(self.__iterator)
        except StopIteration:
            self.__iterator = iter(self.__dataloader)
            data = next(self.__iterator)
        return data, iteration
    
# Loss types

@dataclass(frozen=True)
class Loss:
    backward_obj: torch.Tensor
    loss_task: torch.Tensor
    
ForwardLossType = TypeVar('ForwardLossType', bound=Loss)
AfterForwardLossType = TypeVar('AfterForwardLossType', bound=Loss)


# ClientUpdateState

@dataclass
class BaseClientState:
    model: Model
    s_round: int
    
@dataclass
class BaseClientForwardState(BaseClientState):
    loss_fn: torch.nn.Module
    
@dataclass
class BaseClientUpdateState(BaseClientForwardState):
    init_params: List[torch.Tensor]
    optimizer: FederatedOptimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

BaseClientForwardStateType = TypeVar('BaseClientForwardStateType', bound=BaseClientForwardState)
BaseClientUpdateStateType = TypeVar('BaseClientUpdateStateType', bound=BaseClientUpdateState)
    
# State types

@dataclass
class BaseClientState(State):
    ...
    
@dataclass
class BaseClientTempState(State):
    model: Optional[Model] = None
    delta_params: Optional[List[torch.Tensor]] = None
    
BaseClientStateType = TypeVar('BaseClientStateType', bound=BaseClientState)
BaseClientTempStateType = TypeVar('BaseClientTempStateType', bound=BaseClientTempState)
