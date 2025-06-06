from dataclasses import dataclass
from typing import Optional, TypeVar
from src.utils.state import State


@dataclass
class BaseServerState(State):
    model_state: Optional[dict] = None
    opt_state: Optional[dict] = None
    lr_scheduler_state: Optional[dict] = None
    
BaseServerStateType = TypeVar('BaseServerStateType', bound=BaseServerState)
