import logging

from omegaconf import DictConfig
from src.utils import select_random_subset
from typing import cast, List
from src.algo import FedAvg
from src.algo.fed_clients import MimeClient
from src.algo.center_server import MimeCenterServer
from src.utils.state import Device

log = logging.getLogger(__name__)


class Mime(FedAvg):
    """ Implements the controller in Mime algorithm family, as proposed in Karimireddy et al., 
    Breaking the centralized barrier for cross-device federated learning, NeurIPS 2021 """

    def __init__(self, model_info, params, device: str, dataset: DictConfig,
                 output_suffix: str, savedir: str, writer=None):
        assert 0 <= params.clients_dropout < 1, f"Dropout rate d must be 0 <= d < 1, got {params.clients_dropout}"
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        self.__clients_dropout = params.clients_dropout
        self.__dropping = (lambda x: select_random_subset(x, self.__clients_dropout)) if self.__clients_dropout > 0 \
            else (lambda x: x)
        self.__mime_lite = params.mime_lite

    def train_step(self):
        self._select_clients(self._clients, self.__dropping)

        # Communication round to calculate the controls
        center_server = cast(MimeCenterServer, self._center_server)
        clients = cast(List[MimeClient], self._selected_clients)

        for c in clients:
            c.receive_data(**center_server.send_data(send_model_and_state=True, send_controls=False))
            c.device = Device(self._device[0])
            c.calculate_full_grad(self._loss_fn, self._iteration)
        center_server.aggregate_fullgrads(clients)

        # Regular training round
        for c in clients:
            # MimeLite does not need any additional data to be sent, so only when using Mime we send controls
            if not self.__mime_lite:
                c.receive_data(**center_server.send_data(send_controls=True, send_model_and_state=False))
            c.setup()
            c.client_update(self._local_iterations, self._loss_fn, self._iteration)
