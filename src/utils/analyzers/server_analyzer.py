import torch.nn
from torch.utils.data import DataLoader
import logging
from src.models import Model
from src.utils import MeasureMeter
from src.utils.analyzers.analyzer import Analyzer

log = logging.getLogger(__name__)

__all__ = ['ServerAnalyzer']


class ServerAnalyzer(Analyzer):
    """
    Analyzer for a center server
    """

    def __init__(self, val_period: int, val_always_last_rounds: int, total_rounds: int, *args, **kwargs):
        super().__init__({'model': Model, 'loss_fn': torch.nn.Module, 's_round': int, 'device': str,
                          'dataloader': DataLoader}, *args, **kwargs)
        self._val_period = val_period
        self._val_always_last_rounds = val_always_last_rounds
        self._total_rounds = total_rounds
        
    def _do_validate(self, s_round, force):
        return force  or \
               (s_round % self._val_period == 0) or \
               (s_round > self._total_rounds - self._val_always_last_rounds)

    def _analyze(self, event, model: Model, loss_fn: torch.nn.Module, s_round: int, device: str, 
                 dataloader: DataLoader, other_scalars: dict = {}, force: bool = False, **kwargs) -> None:
        from src.algo import Algo
        if self._do_validate(s_round, force):
            mt = MeasureMeter(model.num_classes)
            loss = Algo.test(model, mt, device, loss_fn, dataloader)
            p_norm = model.params_norm(2)
            scalars = other_scalars.copy()
            scalars.update({'param_norm': p_norm})
            self._log(s_round, loss, mt, scalars)
            data = {'loss': loss, 'accuracy': mt.accuracy_overall, 'accuracy_class': mt.accuracy_per_class}
            data.update(scalars)
            self._result.update({s_round: data})

    def _log(self, s_round, loss, mt, other_scalars: dict):
        if self._verbose:
            log.info(
                    f"[Round: {s_round: 05}] Test set: Average loss: {loss:.4f}, Accuracy: {mt.accuracy_overall:.2f}%"
            )
        if self._writer is not None:
            data = {f'{self.__class__.__name__}':
                        {'val': {'loss': loss, 'accuracy': mt.accuracy_overall, 'round': s_round}}}
            for tag, scalar in other_scalars.items():
                data[f'{self.__class__.__name__}'].update({tag: scalar, 'round': s_round})
            self._writer.log(data, s_round)



        

