import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging
from src.utils import LoggingSystem
log = logging.getLogger(__name__)

class Analyzer(ABC):
    """ Base (abstract) for any analyzer """

    def __init__(self, call_args: Dict[str, type], event: str, verbose: bool = False,
                 writer: Optional[LoggingSystem] = None):
        """

        Parameters
        ----------
        call_args mandatory arguments for a call to analyze()
        event describes what request the analyzer responds to
        verbose outputs additional information about the analyzer
        writer LoggingSystem instance to log results to
        """
        self._event = event
        self.__args = call_args
        self._verbose = verbose
        self._writer = writer
        self._result = {}

    @property
    def result(self) -> dict:
        return self._result

    def reset(self) -> None:
        self._result.clear()

    def __call__(self, event: str, *args, **kwargs) -> None:
        assert len(args) == 0, "Only named parameters accepted"
        self.__verify_args(kwargs)
        if self.listen_to_event(event):
            return self._analyze(event, **kwargs)

    def __verify_args(self, kwargs):
        assert all([p in kwargs for p in self.__args]), \
            f"{self.__class__.__name__}: missing parameters: given {kwargs.keys()}, required {self.__args.keys()}"
        for arg_name, arg_type in self.__args.items():
            assert isinstance(kwargs[arg_name], arg_type), f"Parameter {arg_name} expected to be of type {arg_type}, " \
                                                           f"instead is of type {kwargs[arg_name]}"

    @abstractmethod
    def _analyze(self, event, **kwargs) -> None:
        """
        Perform analysis on the given keyword arguments in response to an event

        Parameters
        ----------
        event describe the request made to the analyzer
        kwargs arguments to use to perform analysis
        """
        pass

    def state_dict(self) -> dict:
        return {'classname': self.__class__.__name__, 'result': self._result}

    def load_state_dict(self, state: dict) -> None:
        assert 'classname' in state and 'result' in state, "Incomplete state for analyzer"
        if state['classname'] != self.__class__.__name__:
            log.warning(f'Reloading results from different analyzer class, expected {self.__class__.__name__}, '
                        f'given {state["classname"]}')
        self._result = copy.deepcopy(state['result'])

    def listen_to_event(self, event) -> bool:
        return event == self._event


class EmptyAnalyzer(Analyzer):
    """ Dummy analyzer that does nothing"""

    def __init__(self, **kwargs):
        super().__init__({}, "")

    def _analyze(self, event, **kwargs):
        if self._verbose:
            log.info("Empty analyzer called")


class AnalyzerContainer(ABC):
    """
    Base (abstract) class for a container of analyzers
    """

    def __init__(self, analyzers):
        self._analyzers: Dict[str, Analyzer] = analyzers
        self._old_state_dict = {}

    def add_analyzer(self, analyzer: Analyzer):
        self._analyzers.update({analyzer.__class__.__name__: analyzer})

    def contains_analyzer(self, classname) -> bool:
        return classname in self._analyzers

    def state_dict(self) -> dict:
        state = copy.deepcopy(self._old_state_dict)
        for name, analyzer in self._analyzers.items():
            state[name] = analyzer.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        if not all([name in state for name in self._analyzers]):
            log.warning("Missing states for some analyzers")
        if any([name not in self._analyzers for name in state]):
            log.warning("Found analyzers in previous run not instantiated for this run")
        for name, analyzer_state in state.items():
            if name not in self._analyzers:
                self._old_state_dict[name] = analyzer_state
            else:
                self._analyzers[name].load_state_dict(analyzer_state)

    def reset(self) -> None:
        for analyzer in self._analyzers.values():
            analyzer.reset()





