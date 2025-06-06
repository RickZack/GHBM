from typing import Dict, Optional, List
import logging
from src.utils import LoggingSystem
from .analyzer import AnalyzerContainer, Analyzer
from .checkpoint_saver import CheckpointSaver
from .server_analyzer import ServerAnalyzer
log = logging.getLogger(__name__)



class ChainedAnalyzer(AnalyzerContainer, Analyzer):
    """
    Container of analyzers that applies each analyzer sequentially
    """

    def __init__(self, analyzers: List[dict], verbose: bool = False,
                 writer: Optional[LoggingSystem] = None):
        # check format
        assert all(['classname' in a and 'args' in a for a in analyzers]), "Error in format for analyzers"
        analyzers: Dict[str, Analyzer] = {a['classname']: eval(a['classname'])(**a['args'], writer=writer)
                                               for a in analyzers}
        AnalyzerContainer.__init__(self, analyzers)
        Analyzer.__init__(self, {}, "", verbose, writer)

    @staticmethod
    def empty():
        return ChainedAnalyzer([{'classname': 'EmptyAnalyzer', 'args': {}}])

    def _analyze(self, event, **kwargs) -> None:
        for name, analyzer in self._analyzers.items():
            if analyzer.listen_to_event(event):
                analyzer(event, **kwargs)
                self._result.update({name: analyzer.result})

    def reset(self) -> None:
        AnalyzerContainer.reset(self)
        Analyzer.reset(self)

    def listen_to_event(self, event) -> bool:
        listeners = [a.listen_to_event(event) for a in self._analyzers.values()]
        return any(listeners)