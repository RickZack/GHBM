from typing import Callable, List, Optional
import torch
from torch import Tensor
from src.optim.fed_optimizer import FederatedOptimizer


class FedSGD(FederatedOptimizer):
    r"""Implements stochastic gradient descent (optionally with momentum), with model update and
        optimizer state update stages decoupled, to be used in frameworks like Mime.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @staticmethod
    def sgd_state_update(d_p_list: List[Tensor],
                         momentum_buffer_list: List[Optional[Tensor]],
                         *,
                         momentum: float,
                         dampening: float):
        for i, (d_p, buf) in enumerate(zip(d_p_list, momentum_buffer_list)):
            if momentum != 0:
                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

    @staticmethod
    def sgd_model_update(params: List[Tensor],
                         d_p_list: List[Tensor],
                         momentum_buffer_list: List[Optional[Tensor]],
                         *,
                         momentum: float,
                         lr: float,
                         dampening: float,
                         nesterov: bool):
        for i, param in enumerate(params):

            d_p = d_p_list[i]

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                else:
                    # here difference with pytorch implementation:
                    # we don't modify the original momentum tensor
                    buf = buf.mul(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            param.add_(d_p, alpha=-lr)

    @torch.no_grad()
    def _step(self, closure: Optional[Callable[[], float]], update_opt_state: bool, update_model: bool):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)

                    state = self.state[p]
                    momentum_buffer = state.get('momentum_buffer')
                    momentum_buffer_list.append(momentum_buffer)

            d_p_list = FederatedOptimizer.get_params_grads(params_with_grad, weight_decay=weight_decay)

            if update_model:
                FedSGD.sgd_model_update(params_with_grad,
                                        d_p_list,
                                        momentum_buffer_list,
                                        momentum=momentum,
                                        lr=lr,
                                        dampening=dampening,
                                        nesterov=nesterov)

            # update momentum_buffers in state
            if update_opt_state:
                FedSGD.sgd_state_update(d_p_list, momentum_buffer_list, momentum=momentum, dampening=dampening)
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer

        return loss
