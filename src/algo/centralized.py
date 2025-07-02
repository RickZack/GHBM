import logging
from typing import Any, Mapping
import torch.optim.lr_scheduler
from omegaconf import DictConfig
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from src.models import Model
from torch.utils.data import DataLoader
from src.algo import Algo
from src.utils.analyzers import Analyzer
from src.optim import *
from src.utils.state import Device

log = logging.getLogger(__name__)


class Centralized(Algo):
    def __init__(self, model_info, params, device: str, dataset: DictConfig,
                 output_suffix: str, savedir: str, writer=None):
        assert params.loss.type == "crossentropy", "Loss function for centralized algorithm must be crossentropy"
        super().__init__(params, device, dataset, output_suffix, savedir, writer, {'model', 'scheduler_state', 'opt_state'})
        self._optimizer: type = eval(params.optim.classname)
        self._optimizer_args: dict = params.optim.args

        self._batch_size = params.batch_size

        self.test_loader = DataLoader(self.test_ds, num_workers=params.num_workers, batch_size=self._batch_size, 
                                      pin_memory=True, shuffle=False)
        
        self._train_loader = DataLoader(self.train_ds, num_workers=params.num_workers, batch_size=self._batch_size, 
                                        pin_memory=True, shuffle=True)
        self._model = Model(model_info, self.train_ds.num_classes).to(self._device[0])

        # setup optimizer and lr scheduler
        optim: torch.optim.Optimizer = self._optimizer(self._model.parameters(), **self._optimizer_args)
        warmup_scheduler = eval(params.warmup.lr_scheduler.classname)(optim, **params.warmup.lr_scheduler.args)
        training_scheduler = eval(params.training.lr_scheduler.classname)(optim, **params.training.lr_scheduler.args)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warmup_scheduler, training_scheduler], [params.warmup.lr_scheduler.args.total_iters])
        
        self._optim = optim
        self._scheduler = scheduler


    def _fit(self, iterations):
        model = self._model.to(self._device[0])
        loss_fn = self._loss_fn.to(self._device[0])
        server_analyzer: Analyzer = self._analyzer.module_analyzer('server')
        server_analyzer('validation', model=model, loss_fn=loss_fn, s_round=self._iteration, device=self._device[0], dataloader=self.test_loader)
        
        with logging_redirect_tqdm():
            for self._iteration in trange(self._iteration+1, iterations+1, initial=self._iteration, total=iterations):
                self.train_step()        
                server_analyzer('validation', model=model, loss_fn=loss_fn, s_round=self._iteration,
                                device=self._device[0], dataloader=self.test_loader)
                server_analyzer('checkpoint', s_round=self._iteration, state_dict_fn=self.state_dict)
        
    def _state_dict(self, device: Device) -> dict:
        def copy_and_move(obj, device):
            import copy
            from src.utils.state import move_to
            cpy = copy.deepcopy(obj)
            return move_to(cpy, device)
        
        data = super()._state_dict(device) 
        data.update({'model': copy_and_move(self._model.state_dict(), device), 
                     'scheduler_state': copy_and_move(self._scheduler.state_dict(), device), 
                     'opt_state': copy_and_move(self._optim.state_dict(), device)
                    })
        return data
        
    def train_step(self):
        model = self._model
        Algo.train(model, self._device[0], self._optim, self._loss_fn, self._train_loader)
        self._scheduler.step()
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        super().load_state_dict(state_dict, strict)
        self._model.load_state_dict(state_dict['model'], strict)
        self._optim.load_state_dict(state_dict['opt_state'])
        self._scheduler.load_state_dict(state_dict['scheduler_state'])
