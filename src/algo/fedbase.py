import os
from typing import Any, List, Mapping

from abc import abstractmethod
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from src.algo.center_server import *
from src.algo import Algo
from src.algo.fed_clients import *
from src.algo.fed_clients.types import IterationNum, IterationType
from src.models import Model
import logging

from src.utils import *
from src.utils.client_sampler import *
from src.utils.state import Device
from src.utils.analyzers.analyzer import Analyzer

log = logging.getLogger(__name__)



class FedBase(Algo):
    """
    Base (abstract) class for any Federated Learning algorithm
    """

    def __init__(self, model_info, params, device: str, dataset: DictConfig,
                 output_suffix: str, savedir: str, writer=None, *args, **kwargs):
        common = params.common
        C, K, B, E = common.C, common.K, common.B, common.E
        assert 0 < C <= 1, f"Illegal value, expected 0 < C <= 1, given {C}"
        super().__init__(params, device, dataset, output_suffix, savedir, writer)
        self._num_clients = K
        self._batch_size = B
        self._fraction = C
        self._local_iterations = IterationNum(IterationType(params.iteration_type), E)
        self._aggregation_policy = params.aggregation_policy

        ds_splitter = create_splitter(dataset)
        local_ds = self.train_ds.make_federated(self._num_clients, ds_splitter)

        model = Model(model_info, self.train_ds.num_classes)
        model_has_batchnorm = model.has_batchnorm()
        clients_device = [Device(self._device[0]) for _ in range(self._num_clients)]        

        local_dataloaders = [DataLoader(data, batch_size=self._batch_size, shuffle=True,
                             drop_last=model_has_batchnorm, num_workers=params.num_workers)
                             for data in local_ds]
        
        client_analyzer = self._analyzer.module_analyzer('client')
        self._clients: List[Client] = [eval(params.client.classname)(k, data,  self.train_ds.num_classes, dev,
                                                                     analyzer=client_analyzer, **params.client.args)
                                       for k, (data, dev) in enumerate(zip(local_dataloaders, clients_device))]
        self._selected_clients: List[Client] = []

        test_dataloader = DataLoader(self.test_ds, batch_size=self._batch_size, num_workers=params.num_workers)
        self._center_server: CenterServer = eval(params.center_server.classname)(model, test_dataloader, self._device[0],
                                                                   analyzer=self._analyzer.module_analyzer('server'),
                                                                   **params.center_server.args)
        self.__client_sampler: ClientSampler = eval(params.client_sampler.classname)(**params.client_sampler.args)

    @abstractmethod
    def train_step(self):
        pass

    def _fit(self, iterations: int):
        self._center_server.device = Device(self._device[0])
        controller_analyzer: Analyzer = self._analyzer.module_analyzer('controller')
        # always validate at the fist round
        self._center_server.trigger_validation(self._iteration)
        
        with logging_redirect_tqdm():
            for self._iteration in trange(self._iteration+1, iterations+1, initial=self._iteration, total=iterations):
                self.train_step()
                self._aggregate()
                self._cleanup_clients()
                controller_analyzer('checkpoint', s_round=self._iteration, state_dict_fn=self.state_dict)

    def _select_clients(self, clients_pool: List[Client], dropping_fn=lambda x: x) -> None:
        """
        Selects the C portion of clients that will participate in the current round

        Parameters
        ----------
        clients_pool : list
            the pool of clients to choose among
        dropping_fn
            function that determines how the selected clients will drop the current round
        """
        n_sample = max(int(self._fraction * len(clients_pool)), 1)
        sampled_clients = self.__client_sampler.sample(clients_pool, n_sample)
        self._selected_clients = dropping_fn(sampled_clients)

    def _setup_clients(self) -> None:
        """
        Sends the data of the center server to the clients involved in the current round.
        """
        for client in self._selected_clients:
            client.receive_data(**self._center_server.send_data(client.client_id))
            client.setup()

    def _cleanup_clients(self):
        """
        Perform clients cleanup for the clients involved in the current round, i.e. to save computational resources
        """
        for client in self._selected_clients:
            client.cleanup()
        self._selected_clients.clear()

    def _aggregate(self) -> None:
        """
        Issues the aggregation to the center server, with the aggregation weights according to the aggregation policy
        """
        clients = self._selected_clients
        if self._aggregation_policy == "weighted":
            total_weight = sum([len(c) for c in clients])
            weights = [len(c) / total_weight for c in clients]
        else:  # uniform
            total_weight = len(clients)
            weights = [1. / total_weight for _ in range(len(clients))]
        self._center_server.aggregation(clients, weights, self._iteration)       
           
    def _state_dict(self, device: Device):
        data = super()._state_dict(device)
        clients_state = {}
        for c in self._clients:
            client_device = device.add_part(f'client_{c.client_id}.pkl')
            clients_state[c.client_id] = c.state_dict(client_device)
        data.update({'server_state': self._center_server.state_dict(), 'clients_state': clients_state}) 
        
        return data 
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        assert len(self._clients) == len(state_dict['clients_state']), "Mismatch in number of clients' states"
        # tentative loading: if it goes well, we proceed with actual loading
        for c_id, state in state_dict['clients_state'].items():
            self._clients[c_id].load_state_dict(state, tentative=True)
        self._center_server.load_state_dict(state_dict['server_state'], strict, tentative=True)
        super().load_state_dict(state_dict, strict)

        # Ok there should be no errors, we can safely load the state_dict
        for c_id, state in state_dict['clients_state'].items():
            self._clients[c_id].load_state_dict(state)
        self._center_server.load_state_dict(state_dict['server_state'], strict)
