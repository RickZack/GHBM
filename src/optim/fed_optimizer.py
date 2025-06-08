from typing import Optional, Callable, List
from abc import ABC, abstractmethod

from torch import Tensor
from torch.optim import Optimizer
import torch


class FederatedOptimizer(Optimizer, ABC):

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None, *,
             update_opt_state: bool = True, update_model: bool = True) -> Optional[float]:
        assert update_opt_state or update_model, "You must at least update the model or the optimizer state!"
        return self._step(closure, update_opt_state, update_model)

    @abstractmethod
    def _step(self, closure: Optional[Callable[[], float]] = None, update_opt_state: bool = True,
              update_model: bool = True) -> Optional[float]:
        pass

    def partial_state_dict(self) -> dict:
        return self.state_dict()

    @staticmethod
    def get_params_grads(params: List[Tensor], *, weight_decay: float) -> List[Tensor]:
        grads = []
        for param in params:
            grad = param.grad
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)
            grads.append(grad)
        return grads
