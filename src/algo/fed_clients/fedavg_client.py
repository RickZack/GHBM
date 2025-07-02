from typing import Optional
import torch
from src.optim.fed_optimizer import FederatedOptimizer
from src.models import Model
from src.algo.fed_clients.types import BaseClientForwardState, BaseClientUpdateState, BaseClientUpdateStateType, Loss, IterationInfo
from src.algo.fed_clients.base_client import Client


class FedAvgClient(Client):
    """ Implements a client in FedAvg algorithm, as proposed in McMahan et al., Communication-efficient learning of
    deep networks from decentralized data, AISTATS 2017"""
    
    def _client_forward(self, model: Model, batch: tuple, state: BaseClientForwardState,
                        current_iteration: IterationInfo) -> Loss:
        data, target = batch
        logits = model(data)
        loss = state.loss_fn(logits, target)
        return Loss(loss, loss)
    
    def _client_backward(self, model: Model, loss: Loss, state: BaseClientForwardState,
                         current_iteration: IterationInfo, num_batches_accumulate: int):
        loss.backward_obj.backward()
        if num_batches_accumulate > 1:
            if self._is_batch_complete(current_iteration, num_batches_accumulate):
                model.grads_div_(num_batches_accumulate)
                    
    def _before_opt_step(self, model: Model, current_iteration: IterationInfo, state: BaseClientUpdateStateType):
        """Executes some operations, e.g. gradient modifications, right before a local optimization step. 
        This method is meant to be for subclasses to define gradient mangling operations

        Args:
            model (Model): the model being trained
            current_iteration (IterationInfo): information about the current iteration
            state (BaseClientUpdateStateType): the state to keep thoughout the whole client_update
        """
        pass
    
    @torch.no_grad
    def _opt_step(self, optimizer: FederatedOptimizer, current_iteration: IterationInfo, state: BaseClientUpdateState) -> None:
        """Executes the step on the given optimizer

        Args:
            optimizer (FederatedOptimizer): the optimizer to step on
        """
        optimizer.step()
        optimizer.zero_grad()
    
    @torch.no_grad            
    def _client_step(self, model: Model, optimizer: FederatedOptimizer, current_iteration: IterationInfo, state: BaseClientUpdateState,
                     num_batches_accumulate: int, lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
        if self._is_batch_complete(current_iteration, num_batches_accumulate):
            self._before_opt_step(model, current_iteration, state)
            model.clip_grad_norm(self._clipping_norm)
            self._opt_step(optimizer, current_iteration, state)
            lr_step_epoch = current_iteration.server.step // self._lr_step_period
            lr_scheduler.step(epoch=lr_step_epoch)

    
