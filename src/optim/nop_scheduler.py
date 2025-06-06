from torch.optim.lr_scheduler import LRScheduler


class NopScheduler(LRScheduler):
    """ Implements a learning rate scheduler that does nothing, e.g. NOP operation
    """

    def __init__(self, optimizer):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return None

    def step(self, epoch=None):
        pass