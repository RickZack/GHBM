from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.algo.fed_clients import Client


class ClientSampler(ABC):
    """Abstract class for components that implement the strategy to sample clients at each FL round"""

    @abstractmethod
    def sample(self, clients: List[Client], num_samples: int) -> List[Client]:
        """Samples a subset of clients for a FL round

        Args:
            clients (List[Client]): the list of all available clients
            num_samples (int): the number of clients to select

        Returns:
            List[Client]: a subset of clients selected for training
        """
        ...


class UniformClientSampler(ClientSampler):
    """ClientSampler that select clients with uniform probability"""

    def sample(self, clients: List[Client], num_samples: int) -> List[Client]:
        indexes = np.random.choice(range(len(clients)), num_samples, replace=False)
        return [clients[k] for k in iter(indexes)]


class DirichletClientSampler(ClientSampler):
    """ClientSampler that uses a Dirichlet distribution as prior for clients' sampling probabilities"""
    def __init__(self, num_clients: int, gamma: float):
        assert gamma > 0, f"Concentration parameter of Dirichlet distribution must be > 0, got {gamma}"
        self.__distribution = np.random.dirichlet(np.repeat(gamma, num_clients))

    def sample(self, clients: List[Client], num_samples: int) -> List[Client]:
        indexes = np.random.choice(range(len(clients)), num_samples, replace=False, p=self.__distribution)
        return [clients[k] for k in iter(indexes)]

class CycleCPClientSampler(ClientSampler):
    """ClientSampler that selects clients in a cyclic order
    
    This class implements sampling according to the CycleCP framework as introduced in Cho et. al.,
    On the Convergence of Federated Averaging with Cyclic Client Participation, ICML 2023

    """
    def __init__(self, num_clients: int, num_groups: int) -> None:
        clients_idx = np.random.permutation(num_clients)
        self.__client_groups = np.array_split(clients_idx, num_groups)
        self.__cur_group_id = 0
        
    def sample(self, clients: List[Client], num_samples: int) -> List[Client]:
        current_group = self.__client_groups[self.__cur_group_id]
        indexes = np.random.choice(current_group, num_samples, replace=False)
        self.__cur_group_id = (self.__cur_group_id + 1) % len(self.__client_groups)
        return [clients[k] for k in iter(indexes)]
        