import logging
from omegaconf import DictConfig
from src.algo.fedbase import FedBase
from src.utils import select_random_subset
from src.utils.state import Device

log = logging.getLogger(__name__)


def setup_and_train(c, dev, s, local_iterations, loss_fn, r):
    c.receive_data(**s.send_data(c.client_id))
    c.device = Device(dev)
    # c.setup()
    c.client_update(local_iterations, loss_fn, r)


class FedAvg(FedBase):
    """
    FedAvg algorithm as proposed in McMahan et al., Communication-efficient learning of deep networks from
    decentralized data, AISTATS 2017.
    """

    def __init__(self, model_info, params, device: str, dataset: DictConfig,
                 output_suffix: str, savedir: str, writer=None):
        assert 0 <= params.clients_dropout < 1, f"Dropout rate d must be 0 <= d < 1, got {params.clients_dropout}"
        super(FedAvg, self).__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        self.__clients_dropout = params.clients_dropout
        self.__dropping = (lambda x: select_random_subset(x, self.__clients_dropout)) if self.__clients_dropout > 0 \
            else (lambda x: x)

    def train_step(self):
        from itertools import cycle
        self._select_clients(self._clients, self.__dropping)

        dev = self._device[0]
        for c in self._selected_clients:
            call_args = (c, dev, self._center_server,
                            self._local_iterations, self._loss_fn, self._iteration)
            setup_and_train(*call_args)