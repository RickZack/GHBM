from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import wandb

from tensorboardX import SummaryWriter


class LoggingSystem(ABC):
    @abstractmethod
    def log(self, data: Dict[str, Any], step: int) -> None:
        pass

    def finish(self):
        pass


class TensorboardLogger(LoggingSystem):
    def __init__(self, config, *, _=None, **kwargs):
        self.__writer = SummaryWriter(**kwargs)
        self.__writer.add_text("Parameters", str(config), 0)

    def __log(self, tags: Tuple[str, ...], scalar_value, step: int):
        tag = '/'.join(tags)
        self.__writer.add_scalar(tag, scalar_value, global_step=step)

    def log(self, data: Dict[str, Any], step: int, tags: Tuple[str, ...] = ()) -> None:
        for key, val in data.items():
            new_tags = tags + (key,)
            if isinstance(val, dict):
                self.log(val, step, new_tags)
            else:
                self.__log(new_tags, val, step)

    def finish(self):
        self.__writer.close()


class WandbLogger(LoggingSystem):
    def __init__(self, config, *, _=None, **kwargs):
        kwargs['config'] = config
        wandb.init(**kwargs)

    def log(self, data: Dict[str, Any], step: int, *args) -> None:
        wandb.log(data)

    def finish(self):
        wandb.finish()
