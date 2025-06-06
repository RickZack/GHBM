from pathlib import Path
from typing import Any, Mapping
import torch
from src.utils import save_pickle
from src.utils.utils import load_pickle
from dataclasses import dataclass, field
import shutil

@dataclass
class Device:
    name: str = "cpu"
    
    def is_cuda(self):
        return 'cuda' in self.name

    def is_cpu(self):
        return self.name == 'cpu'

    def is_cuda_or_cpu(self):
        return self.is_cuda() or self.is_cpu()
    
    def is_dir(self):
        if self.is_cuda_or_cpu():
            return False
        Path(self.name).mkdir(exist_ok=True)
        return True
    
    def is_file(self):
        return Path(self.name).is_file()
    
    def set_file(self, filename: str):
        assert self.is_dir()
        self.name = str(Path(self.name).joinpath(filename))
        return self
        
    def change_storage(self, filename: str):
        self.name = str(Path(filename))
        assert self.is_file()
        
    def add_part(self, part: str):
        d = Device(self.name)
        if not d.is_cuda_or_cpu():
            Path(d.name).mkdir(exist_ok=True)
            d.name = str(Path(d.name).joinpath(part))
        return d

# Helper functions for data transfer in State.to()     
def is_device_to_device(src: Device, dst: Device) -> bool:
    return src.is_cuda_or_cpu() and dst.is_cuda_or_cpu()

def is_file_to_file(src: Device, dst: Device) -> bool:
    return not src.is_cuda_or_cpu() and  not dst.is_cuda_or_cpu()

def is_device_to_file(src: Device, dst: Device) -> bool:
    return src.is_cuda_or_cpu() and not dst.is_cuda_or_cpu()

def is_file_to_device(src: Device, dst: Device) -> bool:
    return not src.is_cuda_or_cpu() and dst.is_cuda_or_cpu()
    
    
def move_to(val, device: Device):
    if isinstance(val, list):
        for i in range(len(val)):
            val[i] = move_to(val[i], device)
    elif isinstance(val, dict):
        for key in val:
            val[key] = move_to(val[key], device)
        return val
    elif isinstance(val, (torch.nn.Module, torch.Tensor)):
        val = val.to(device.name)
    return val

    
@dataclass
class State:
    _device: Device = field(default_factory=Device)
    
    def replace_store_path(self, new_store_path: Device):
        if not self._device.is_cuda_or_cpu():
            self._device.change_storage(new_store_path.name)
    
    def update(self, new: Mapping[str, Any]):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def _relevant_keys(self):
        data = vars(self)
        if len(data) == 1:
            return set()
        return set(data.keys())       
                
    def state_dict(self) -> Mapping[str, Any]:
        data = vars(self)
        relevant_keys = self._relevant_keys()
        data = {k: data[k] for k in relevant_keys}
        if '_device' in data:
            data['_device'] = self._device.name
        return data

    def data(self) -> Mapping[str, Any]:
        return {key: val for key, val in vars(self).items() if key != '_device' and val is not None}
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, tentative: bool = False,
                        replace_store_device: Device = None):
        if tentative:
            assert strict, "Tentative only makes sense if strict is True"
        if strict:
            if not self._device.is_cuda_or_cpu(): # then it should be a file 
                device = replace_store_device or self._device # we eant to consider the actual device
                assert device.is_file(), f"file {device.name} is not valid"
            keys = self._relevant_keys() # set(vars(self).keys()) # - {'_device'}
            missing_keys = [key for key in keys if key not in state_dict]
            spurious_keys = [key for key in state_dict if key not in keys]
            assert not missing_keys, f"Found missing keys: {missing_keys}"
            assert not spurious_keys, f"new_state contains the following spurious keys: {spurious_keys}"
        if not tentative:
            # state_dict contains _device as a string, we need to convert to Device object
            if '_device' not in state_dict: return # it means that the state is empty
            device = replace_store_device or Device(state_dict['_device'])
            self.update({**state_dict, **{'_device': device}})
        
    def clear(self):
        new_state = {key: None for key in vars(self)}
        del new_state['_device']
        self.update(new_state)
        
    def to(self, device: Device):  # device can be: 'ram', 'cuda:{}' or file path
        # if device.is_cpu():
        #     import pdb; pdb.set_trace()
        state = self.__dict__ #vars(self)
        if state:
            if is_device_to_device(self._device, device):
                move_to(state, device)
            elif is_file_to_file(self._device, device): # used when trasferring from temp_state to store
                assert self._device.is_file(), f"source file does not exist: {self._device.name}"
                shutil.copy(self._device.name, device.name)
            elif is_device_to_file(self._device, device):
                move_to(state, Device("cpu"))
                save_pickle(self.state_dict(), device.name)
                self.clear()
            elif is_file_to_device(self._device, device):
                assert self._device.is_file(), f"source file does not exist: {self._device.name}"
                saved_state = load_pickle(self._device.name)
                move_to(saved_state, device)
                self.update(saved_state)            
            else:
                raise ValueError(f"device is {device}, current location is {self._device}")
        self._device = device
        return self
