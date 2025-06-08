from abc import ABC, abstractmethod
from torch.nn.modules import Module

from torch.optim.optimizer import Optimizer as Optimizer
from src.models import Model
from src.models import Model
from typing import Any, Mapping, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils import MeasureMeter
from src.utils.analyzers import Analyzer, ChainedAnalyzer
from src.algo.fed_clients.types import ClientDataLoaderIterator, GlobalIterationInfo, IterationInfo, IterationNum, IterationType
from src.algo.fed_clients.types import BaseClientForwardStateType, BaseClientState, BaseClientStateType, BaseClientTempState, BaseClientTempStateType, \
                                       ForwardLossType, AfterForwardLossType, BaseClientUpdateState, BaseClientUpdateStateType
from src.utils.state import Device
from src.optim import *

class Client(ABC):
    """
    Base (abstract) class for a client in any FL algorithm

    The algorithm specific client class is loosely coupled with its corresponding center server class: the return data
    of (CenterServer) send_data method must correspond to (Client) receive_data.
    """
    def __init__(self, client_id: int, dataloader: DataLoader, num_classes: int, device: Device, temp_store_device: str, optim: dict, 
                 lr_scheduler: dict, state: BaseClientStateType = None, temp_state: BaseClientTempStateType = None, 
                 analyzer: Optional[Analyzer] = None, large_batch_size: Optional[int] = None, clipping_norm: float = 50, lr_step_period: int = 1):
        # Batch accumlation configurations
        large_batch_size = large_batch_size or dataloader.batch_size 
        minibatches, modulo = divmod(large_batch_size, dataloader.batch_size)
        assert minibatches >= 1 and modulo == 0, f"Invalid batch accumulation configuration, got large_batch={large_batch_size}, \
                                                   minibatch={dataloader.batch_size}"
        self._num_batches_accumulate = minibatches
        
        # State and analyzer configuration
        self._state = state or BaseClientState()
        self._temp_state = temp_state or BaseClientTempState()
        self.__temp_store_device = Device(temp_store_device)
        if self.__temp_store_device.is_dir():
            self.__temp_store_device.set_file(f'state_client{client_id}.pkl')
        self._analyzer = analyzer or ChainedAnalyzer.empty()

        # Basic client attributes
        self.__client_id = client_id
        self.__dataloader = dataloader
        self.__device = device
        self.__num_classes = num_classes
        
        # Optimization related stuff
        self._optim_class: type = eval(optim.classname)
        self._optim_args = optim.args
        self._scheduler_class: type = eval(lr_scheduler.classname)
        self._scheduler_args = lr_scheduler.args
        self._lr_step_period = lr_step_period
        
        self._clipping_norm = clipping_norm

    @property
    def client_id(self):
        return self.__client_id

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device: Device):
        self._state.to(device)
        self._temp_state.to(device)
        self.__device = device

    @property
    def dataloader(self) -> DataLoader:
        return self.__dataloader

    @dataloader.setter
    def dataloader(self, dataloader: DataLoader):
        assert isinstance(dataloader, DataLoader), "Client's dataloader is not an instance of torch DataLoader"
        self.__dataloader = dataloader

    @property
    def num_classes(self):
        return self.__num_classes

    def send_data(self, *args, **kwargs) -> dict:
        """
        Sends data to the server

        Parameters
        ----------
        args
            positional arguments specific of the client subclass
        kwargs
            keyword arguments specific of the client subclass

        Returns
        -------
        a dictionary containing the data that the client must send to its server
        """
        data = self._temp_state.data()
        self._analyzer('send_data', data=data, client=self)
        return data

    def receive_data(self, **kwargs):
        """
        Receives the data to use in the next client_update

        Parameters
        ----------
        kwargs
            arguments relative to client subclass
        """
        self._temp_state.update(kwargs)

    def setup(self):
        """
        Performs some setup before the start of client_update
        """
        self._state.to(self.device)
        self._temp_state.to(self.device)

    def cleanup(self):
        """
        Performs some cleanup, especially for computational resources, after the end of the train_step, i.e. after the
        server has aggregated the clients' data
        """
        self._state.to(self.__temp_store_device)
        self._temp_state.clear()
        
    def state_dict(self, device: Device) -> Mapping[str, Any]:
        """Returns a shallow copy of the current state of the client
        
        Returning the state_dict can be an expensive operation, because we need to move all the tensors
        to the store device, that could be a file. Please note that sometimes this is inevitable, as stateful 
        algorithms can occupy more memory than available when there are a lot of clients, hence we cannot have
        a single large file containing the state of all clients (see FedBase.state_dict())

        Returns:
            Mapping[str, Any]: a read-only reference to the dict object containing a shallow copy of client's state
        """
        self._state.to(device)
        return self._state.state_dict()
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, tentative: bool = False):
        """Assigns buffers from state_dict into this client's state. If strict is True, then the keys of state_dict 
        must exactly match the keys returned by the client's state state_dict() function.

        Args:
            state_dict (Mapping[str, Any]): a dict-like object containing the client's state buffers
            strict (bool, optional): whether to strictly enforce that the keys in state_dict match the 
            keys returned by the client's state state_dict() function. Defaults to True.
            tentative (bool, optional): whether to only try to load the state dict, used when not only this 
            client's state must be loaded without failure. Defaults to False.
        """
        self._state.load_state_dict(state_dict, strict, tentative)
        
    def _get_client_update_state(self, optimizer: type, optimizer_args, loss_fn: torch.nn.Module, s_round: int,
                                 local_iterations: IterationNum) -> BaseClientUpdateStateType:
        """Costruct the client update state to be used and updated throught the local update

        Args:
            optimizer (type): the class of the optimizer
            optimizer_args (_type_): the args to the optimizer constructor
            loss_fn (torch.nn.Module): the loss function
            s_round (int): the current round of Federated Training
            local_iterations (IterationNum): the number of local iterations to perform

        Returns:
            BaseClientUpdateStateType: the state to keep thoughout the whole client_update
        """
        model = self._temp_state.model
        optimizer = optimizer(model.parameters(), **optimizer_args)

        lr_scheduler = self._scheduler_class(optimizer, **self._scheduler_args)
        global_info = GlobalIterationInfo(s_round, local_iterations.to_steps(len(self.dataloader)), 0)
        # Retrieve the correct lr for the current round
        lr_scheduler.step(epoch=(global_info.step // self._lr_step_period))
        init_params = list(model.params_with_grad(copy=True))
        return BaseClientUpdateState(model, s_round, loss_fn, init_params, optimizer, lr_scheduler)
    
    def _preprocess_batch(self, batch: tuple) -> tuple:
        """Moves the batch to the current client device and applies the transformations

        Args:
            batch (tuple): a tuple consisting of the data and the target

        Returns:
            tuple: a transformed copy of the input batch
        """
        data, target = batch
        data, target = data.to(self.device.name), target.to(self.device.name)
        data, target = self.dataloader.dataset.batchwise_transform((data, target))
        return data, target
            
    def _is_batch_complete(self, cur_it: IterationInfo, num_batches_accumulate: int) -> bool:
        """Helper function to determine if the current step completes a batch accumulation

        Args:
            cur_it (IterationInfo): information about the current iteration
            num_batches_accumulate (int): number of steps that complete a batch accumulation

        Returns:
            bool: True if the current step completes a batch accumulation, False otherwise
        """
        return cur_it.client.step % num_batches_accumulate == 0 or cur_it.client.step == cur_it.client.total_steps
  
    def _pre_client_update(self, model: Model, state: BaseClientUpdateStateType):
        """Executes operations prior to initiation the local training loop

        Args:
            model (Model): the model involved
            state (BaseClientUpdateStateType): the state to keep thoughout the whole client_update
        """
        pass

    def _post_client_update(self, model: Model, state: BaseClientUpdateStateType):
        """Executes operations after the last step of the local training loop, e.g. calculating the the delta params

        Args:
            model (Model): the model involved
            state (BaseClientUpdateStateType): the state to keep thoughout the whole client_update
        """
        temp_state: BaseClientTempState = self._temp_state
        delta_params = [old - new for old, new in zip(state.init_params, model.params_with_grad())]
        temp_state.delta_params = delta_params
        temp_state.model = None

    def _pre_client_forward(self, model: Model, batch: tuple, state: BaseClientForwardStateType, current_iteration: IterationInfo):
        """Executes operations before each forward step of the local training loop

        Args:
            model (Model): the model involved
            batch (tuple): the batch of data that will be forwarded in the curren step
            state (BaseClientForwardStateType): the state to keep thoughout the whole client_update
            current_iteration (IterationInfo): information about the current iteration
        """
        pass

    @abstractmethod
    def _client_forward(self, model: Model, batch: tuple, state: BaseClientForwardStateType,
                        current_iteration: IterationInfo) -> ForwardLossType:
        """Forwards the batch of data through the model and calculates the loss

        Args:
            model (Model): the model involved
            batch (tuple): the batch of data to forward at in the current step
            state (BaseClientForwardStateType): the state to keep thoughout the whole client_update
            current_iteration (IterationInfo): information about the current iteration

        Returns:
            ForwardLossType: the value calculated by the loss function on the current batch of data
        """
        pass

    def _post_client_forward(self, model: Model, batch: tuple, loss: ForwardLossType, state: BaseClientForwardStateType, 
                             current_iteration: IterationInfo) -> AfterForwardLossType:
        """Executes operations after each forward step of the local training loop, potentially modifying the loss

        Args:
            model (Model): the model involved
            batch (tuple): the batch of data forwarded at in the current step
            loss (ForwardLossType): the loss value calculated by _client_forward()
            state (BaseClientForwardStateType): the state to keep thoughout the whole client_update
            current_iteration (IterationInfo): information about the current iteration

        Returns:
            AfterForwardLossType: a transformed version of the received loss
        """
        return loss

    @abstractmethod
    def _client_backward(self, model: Model, loss: AfterForwardLossType, state: BaseClientForwardStateType,
                         current_iteration: IterationInfo, num_batches_accumulate: int):
        """Performs the backward step on the loss value

        Args:
            model (Model): the model involved
            loss (AfterForwardLossType): the loss value calculated by _post_client_forward
            state (BaseClientForwardStateType): the state to keep thoughout the whole client_update
            current_iteration (IterationInfo): information about the current iteration
            num_batches_accumulate (int): number of steps in a batch accumulation
        """
        pass

    @abstractmethod
    def _client_step(self, model: Model, optimizer: FederatedOptimizer, current_iteration: IterationInfo, state: BaseClientUpdateState,
                     num_batches_accumulate: int, lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
        """Applies the local optimization step, given by the optimizer's policy

        Args:
            model (Model): the model involved
            optimizer (torch.optim.Optimizer): the optimizer adopted in local optimization
            current_iteration (IterationInfo): information about the current iteration
            state (BaseClientUpdateStateType): the state to keep thoughout the whole client_update
            num_batches_accumulate (int): number of steps in a batch accumulation
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): the learning rate scheduler to step
        """
        pass
    
    def _post_client_step(self, model: Model, optimizer: torch.optim.Optimizer, current_iteration: IterationInfo, state: BaseClientUpdateStateType):
        """Executes operations after each local step

        Args:
            model (Model): the model involved
            optimizer (torch.optim.Optimizer): the optimizer adopted in local optimization
            current_iteration (IterationInfo): information about the current iteration
            state (BaseClientUpdateStateType): the state to keep thoughout the whole client_update
        """
        pass

    def client_update(self, local_iterations: IterationNum, loss_fn: torch.nn.Module, s_round: int):
        state: BaseClientUpdateStateType = self._get_client_update_state(self._optim_class, self._optim_args, 
                                                                         loss_fn, s_round, local_iterations)
        state.model.train()
        loss_fn.to(self.device.name)

        self._pre_client_update(state.model, state)
        state.model.zero_grad()

        for batch, cur_it in ClientDataLoaderIterator(self.dataloader, local_iterations, s_round, self._num_batches_accumulate):
            batch = self._preprocess_batch(batch)
            self._pre_client_forward(state.model, batch, state, cur_it)
            loss = self._client_forward(state.model, batch, state, cur_it)
            loss = self._post_client_forward(state.model, batch, loss, state, cur_it)

            self._client_backward(state.model, loss, state, cur_it, self._num_batches_accumulate)
            self._client_step(state.model, state.optimizer, cur_it, state, self._num_batches_accumulate, state.lr_scheduler)
            self._post_client_step(state.model, state.optimizer, cur_it, state)

        self._post_client_update(state.model, state)

    def client_evaluate(self, loss_fn, test_data: DataLoader) -> Tuple[float, MeasureMeter]:
        """
        Performs an evaluation step of the current client's model using a given test set

        Parameters
        ----------
        loss_fn
            the loss function to use
        test_data
            the dataset to test the model with

        Returns
        -------
        a tuple containing the loss value and a MeasureMeter object
        """
        from src.algo import Algo

        model = self._temp_state.model
        model.to(self.__device.name)
        loss_fn.to(self.__device.name)
        mt = MeasureMeter(self.num_classes)
        loss = Algo.test(model, mt, self.__device.name, loss_fn, test_data)
        return loss, mt

    def __len__(self) -> int:
        """Returns the number of samples composing the client's dataset

        Returns:
            int: the number of samples of the local dataset
        """
        return len(self.dataloader.dataset)

    def num_ex_per_class(self) -> np.array:
        """
        Returns the data distribution of the client

        Returns
        -------
        a numpy array containing the number of examples for each class in the client's local dataset
        """
        num_ex_per_class = np.zeros(self.num_classes)
        for _, batch in self.dataloader:
            for target in batch.numpy():
                num_ex_per_class[target] += 1
        return num_ex_per_class
    