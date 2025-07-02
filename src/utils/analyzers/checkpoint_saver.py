import itertools
from pathlib import Path
from typing import Callable, Optional, List
import logging
from src.utils import save_pickle
from src.utils.analyzers.analyzer import Analyzer
from src.utils.state import Device

log = logging.getLogger(__name__)

__all__ = ['CheckpointSaver']

def rm(obj: Path):
    if obj.is_file():
        obj.unlink()
    elif obj.is_dir():
        for item in obj.iterdir():
            rm(item)
        obj.rmdir()


class CheckpointSaver(Analyzer):
    """
    Utility to save checkpoints during training. Allows to specify a save period, custom milestones and optionally
    keeping only the last n most recent checkpoints. Always saves a checkpoint at the last training iteration
    """

    def __init__(self, savedir: str, filename_format: str, store_device: str, last_round: int, keep_last_n: int = 1, 
                 save_period: Optional[str] = None, save_milestones: Optional[List[int]] = None, *args, **kwargs):
        """Initializes a CheckpointSaver instance

        Args:
            savedir (str): folder to save checkpoints to
            filename_format (str): a string specifying the format for a checkpoint name, e.g. 'checkpoint_round{}'
            store_device (str): a string specifying the path and format of the store folder, e.g. 'dir1/state_round{}'
            last_round (int): the number of total rounds
            keep_last_n (int, optional): the number of most recent checkpoints to keep, zero means keep all. Defaults to 1.
            save_period (Optional[str], optional): the number of rounds between one checkpoint and the next. Defaults to None.
            save_milestones (Optional[List[int]], optional): a list of round in which a checkpoint must be saved. Defaults to None.
        """
        super().__init__({'s_round': int}, *args, **kwargs)
        assert bool(save_period) ^ bool(save_milestones), "Cannot specify both save_period and save_milestones"
        assert CheckpointSaver._is_filename_format_valid(filename_format), "Invalid filename format"
        if not Device(store_device).is_cuda_or_cpu():
            assert CheckpointSaver._is_filename_format_valid(store_device), "Invalid dirname format"
        self._savedir = savedir
        self._file_format = filename_format
        self._store_device = store_device
        self._last_round = last_round
        self._keep_last_n = keep_last_n
        self._save_period = save_period
        self._save_milestones = save_period

        if save_period is not None:
            self._save_checkpoint = self._check_period
        else:
            self._save_checkpoint = self._check_milestone
    
    @staticmethod
    def _is_filename_format_valid(f: str) -> bool:
        return f.count('{') == f.count('}') == 1
            
    def _check_period(self, s_round: int) -> bool:
        return s_round % self._save_period == 0 or s_round == self._last_round
    
    def _check_milestone(self, s_round: int) -> bool:
        return s_round in self._save_milestones or s_round == self._last_round
    
    def _old_checkpoint_files(self, cur_checkpoint_file: str) -> List[Path]:
        filename_pattern = self._file_format.replace('{}', '*')
        checkpoint_files = [f for f in Path(self._savedir).glob(filename_pattern) if f.is_file() and str(f) < cur_checkpoint_file]
        checkpoint_files.sort(reverse=True)
        return checkpoint_files
    
    def _old_checkpoint_dir(self, cur_checkpoint_dir: str) -> List[Path]:
        dirname_pattern = Path(self._store_device.replace('{}', '*'))
        checkpoint_dirs = [f for f in dirname_pattern.parent.glob(dirname_pattern.name) if f.is_dir() and str(f) < cur_checkpoint_dir]
        checkpoint_dirs.sort(reverse=True)
        return checkpoint_dirs      
    
    def _delete_old_checkpoints(self, cur_checkpoint_file: str, cur_checkpoint_dir: str):
        """ Deletes the oldest checkpoints, except the n+1 most recent ones, i.e. the last checkpoint is never removed """
        if self._keep_last_n > 0:
            files = self._old_checkpoint_files(cur_checkpoint_file)
            dirs = self._old_checkpoint_dir(cur_checkpoint_dir)
            for f in itertools.chain(files[self._keep_last_n:], dirs[self._keep_last_n:]):
                rm(f)
                if self._verbose:
                    log.info(f"Deleted: {str(f)}")
    
    @staticmethod
    def _get_formatted_path(base_path: str, file_format: str, s_round: int, last_round: int) -> str:
        zero_filled_round = str(s_round).zfill(len(str(last_round)))
        filename = file_format.format(zero_filled_round)
        checkpoint_path = Path(base_path).joinpath(filename)
        return str(checkpoint_path)

    def _analyze(self, event, state_dict_fn: Callable[[str], dict], s_round: int, **kwargs) -> None:
        if self._save_checkpoint(s_round):
            checkpoint_path = CheckpointSaver._get_formatted_path(self._savedir, self._file_format, s_round, self._last_round)
            store_path = CheckpointSaver._get_formatted_path('', self._store_device, s_round, self._last_round)
            state_dict = state_dict_fn(store_path)
            self._delete_old_checkpoints(checkpoint_path, store_path)
            save_pickle(state_dict, checkpoint_path)
            if self._verbose:
                log.info(f"Saved checkpoint at iteration {s_round} in {checkpoint_path}, {store_path}")