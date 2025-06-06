from typing import List, Tuple
import torch
from src.algo import Algo
from src.algo.center_server.center_server import CenterServer
from src.algo.fed_clients import Client
from src.utils import MeasureMeter


class FedAvgCenterServer(CenterServer):
    """ Implements the center server in FedAvg algorithm, as proposed in McMahan et al., Communication-Efficient
    learning of deep networks from Decentralized Data, AISTATS 2017"""

    def _update_model(self, pseudo_grads: List[torch.Tensor]):
        # use pseudo grads to update the model
        for p, g in zip(self.model.params_with_grad(), pseudo_grads, strict=True):
            p.grad = g
        self._opt.step()
        self._lr_scheduler.step()

    def _get_pseudo_grads(self, clients_delta: List[List[torch.Tensor]], aggregation_weights: List[float]):
        pseudo_grads = [torch.zeros_like(p) for p in self.model.params_with_grad()]
        for client_delta, w in zip(clients_delta, aggregation_weights, strict=True):
            for pg, g in zip(pseudo_grads, client_delta):
                pg.add_(g, alpha=w)
        return pseudo_grads

    def _aggregation(self, clients: List[Client], aggregation_weights: List[float], s_round: int):
        clients_data = [c.send_data() for c in clients]
        # obtain (pseudo) gradients
        pseudo_grads = self._get_pseudo_grads([data['delta_params'] for data in clients_data], aggregation_weights)

        # use pseudo grads to update the model
        self._update_model(pseudo_grads)
        self.__analyze_pseudo_grads(clients_data, pseudo_grads, s_round)
            
    def __analyze_pseudo_grads(self, clients_data: dict, pseudo_grads: List[torch.Tensor], s_round: int):
        layers_pgrad = {n: pg for (n, param), pg in zip(self.model.named_parameters(), pseudo_grads) 
                        if len(param.shape)>=2}
        self._analyzer('checkrank', tensors=layers_pgrad, s_round=s_round, prefix='avg_pseudo_grad')
        for i, cdata in enumerate(clients_data):
            prefix = f"pseudo_grad_{i}"
            layers_pgrad = {n: pg for (n, param), pg in zip(self.model.named_parameters(), cdata['delta_params']) 
                            if len(param.shape)>=2}
            self._analyzer('checkrank', tensors=layers_pgrad, s_round=s_round, prefix=prefix)
                

    def validation(self, loss_fn) -> Tuple[float, MeasureMeter]:
        self._model.to(self._device)
        loss_fn.to(self._device)
        mt = MeasureMeter(self._model.num_classes)
        loss = Algo.test(self._model, mt, self._device, loss_fn, self._dataloader)
        return loss, mt
