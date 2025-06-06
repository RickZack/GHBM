from pathlib import Path
from collections.abc import Iterator, Iterable
from typing import List, Self

import torch.nn as nn
from src.models import *
import torch
import logging
import torchvision
log = logging.getLogger(__name__)


def tensors_norm(tensors: Iterable[torch.Tensor], norm_type: int) -> float:
    """Calculates the norm of tensors

    Args:
        tensors (Iterable[torch.Tensor]): iterable to tensor whose norms should be calculated
        norm_type (float): type of the used p-norm.

    Returns:
        float: the norm of gradients
    """
    return torch.stack([p.norm(norm_type) for p in tensors]).norm(norm_type)

class Model(nn.Module):

    def __init__(self, model_info, num_classes: int) -> None:
        super().__init__()
        self._model_info = model_info
        self._num_classes = num_classes
        self._model = self.get_model(model_info, num_classes)

        if model_info.pretraining.weights:
            self.load_from_pretraining(model_info.pretraining)
        if model_info.pretraining.freeze_layers:
            self.feature_extraction(model_info.pretraining.freeze_layers)

    def load_from_pretraining(self, pretrain_info):
        weights = pretrain_info.weights
        assert Path(weights).is_file(), f"{weights} is not a valid path for a file"

        with open(weights, 'rb') as f:
            state = torch.load(f)
            load_layers = pretrain_info.load_layers
            if load_layers:  # if list is not empty, select subset of the layers
                state = {f'_model.{layer}': state[f'_model.{layer}'] for layer in load_layers}
            ret = self.load_state_dict(state, strict=False)
            message = f"Starting from pretained model in file: {weights}"
            if ret.missing_keys:
                message += f" Not loading layers {ret.missing_keys}"
            log.info(message)
            if ret.unexpected_keys:
                log.warning(f' Unexpected layers loading the model: {ret.unexpected_keys}')

    def feature_extraction(self, layers):
        for name, p in self._model.named_parameters():
            if name in layers:
                p.requires_grad = False

    @property
    def model_info(self):
        return self._model_info

    @property
    def num_classes(self):
        return self._num_classes

    def same_setup(self, other):
        return self.model_info == other.model_info and self.num_classes == other.num_classes

    def __str__(self):
        return str(self.model_info)

    @staticmethod
    def get_model(model_info, num_classes: int) -> torch.nn.Module:
        m = eval(model_info.classname)(num_classes, **model_info.args)
        return m

    @staticmethod
    def set_parameter_requires_grad(target_model, feature_extracting):
        if feature_extracting:
            for param in target_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self._model.forward(x)

    def has_batchnorm(self):
        for m in self._model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                return True
        return False

    def params_norm(self, norm_type: int) -> float:
        return tensors_norm(self._model.parameters(), norm_type)
    
    def grad_norm(self, norm_type: int = 2) -> float:
        """Calculates the norm of parameters' gradients

        Args:
            norm_type (float, optional): type of the used p-norm. Defaults to 2.

        Returns:
            float: the norm of gradients
        """
        return tensors_norm(self.grads(), norm_type)
    
    def params_with_grad(self, copy: bool = False) -> Iterator[torch.Tensor]:
        """Returns an iterator to model's parameters that require gradient calculation, optionally copying them

        Args:
            copy (bool, optional): whether to return a shallow or deep copy of parameters. Defaults to False.

        Yields:
            Iterator[torch.Tensor]: an iterator to model's parameters, or their copy
        """
        if copy:
            return (torch.clone(p) for p in self._model.parameters() if p.requires_grad)
        return (p for p in self._model.parameters() if p.requires_grad)
        
    def clip_grad_norm(self, max_norm: float, norm_type: float = 2, error_if_nonfinite: bool = True):
        """Perform gradient clipping on the model

        Args:
            max_norm (float): max norm of gradients. A value of 0 means not applying any clipping at all.
            norm_type (float, optional): type of the used p-norm. Defaults to 2.
            error_if_nonfinite (bool, optional): if True, an error is thrown if the total
            norm of the gradients from parameters is nan, inf, or -inf. Defaults to True.
        """
        if max_norm > 0:
            p_with_grad = self.params_with_grad()
            torch.nn.utils.clip_grad_norm_(p_with_grad, max_norm, norm_type, error_if_nonfinite)
            
    def grads(self, copy: bool = False) -> Iterator[torch.Tensor]:
        """Returns an iterator to model's gradients, optionally copying them

        Args:
            copy (bool, optional): whether to return a shallow or deep copy of gradients. Defaults to False.

        Yields:
            Iterator[torch.Tensor]: an iterator to model's gradients, or their copy
        """
        if copy:
            return (torch.clone(p.grad) for p in self.params_with_grad())
        return (p.grad for p in self.params_with_grad())
    
    def grads_div(self, divisor: float) -> List[torch.Tensor]:
        """Copies and scales the gradients of the model

        Args:
            divisor (float): the factor to divide the gradients to

        Returns:
            List[torch.Tensor]: a copy of the gradients scaled by divisor
        """
        grads_copy = list(self.grads(copy=True))
        for g in grads_copy:
            g.div_(divisor)
        return grads_copy
    
    def grads_div_(self, divisor: float) -> None:
        """Scales in-place the gradients of the model

        Args:
            divisor (float): the factor to divide the gradients to
        """
        for g in self.grads():
            g.div_(divisor)
    
    @torch.no_grad    
    def init_from_model(self, other: Self) -> Self:
        """Initializes the weights of this model with those of another

        Args:
            other (Model): the model to copy the weights from
        """
        for p, op in zip(self._model.parameters(), other.parameters()):
            p.copy_(op)
        return self
            
    @torch.no_grad    
    def init_from_weights(self, weights: List[torch.Tensor]) -> Self:
        """Initializes the weights of this model with provided ones

        Args:
            weights (List[torch.Tensor]): the weights to load
        """
        
        for p, w in zip(self._model.parameters(), weights, strict=True):
            p.copy_(w)
        return self