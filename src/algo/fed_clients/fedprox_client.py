import torch
from src.models import Model
from src.algo.fed_clients.types import BaseClientUpdateStateType, IterationInfo
from src.algo.fed_clients import FedAvgClient

class FedProxClient(FedAvgClient):
    """ Implements a client in FedProx algorithm, as proposed in Li et al., Federated Optimization in Heterogeneous Networks, MLSys 2020 """

    def __init__(self, *args,  mu: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mu = mu
    
    def _before_opt_step(self, model: Model, current_iteration: IterationInfo, state: BaseClientUpdateStateType):
        for g, p, pp in zip(model.grads(), model.params_with_grad(), state.init_params):
            g.add_(p - pp, alpha=self.__mu/2)