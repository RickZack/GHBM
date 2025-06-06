from typing import Optional
import logging
from src.utils import LoggingSystem
from .analyzer import AnalyzerContainer, EmptyAnalyzer, Analyzer
from .chained_analyzer import ChainedAnalyzer
log = logging.getLogger(__name__)

class AnalyzerController(AnalyzerContainer):
    """
    Simple container that associates an analyzer container to each module
    """

    def __init__(self, analyzers: dict, writer: Optional[LoggingSystem] = None):
        verbose = analyzers['verbose']
        modules = {component_name: ChainedAnalyzer(component_analyzer, verbose, writer)
                   for component_name, component_analyzer in analyzers['modules'].items()}
        super().__init__(modules)
        if verbose:
            log.info(f'Components of AnalyzerContainer:\n{analyzers["modules"]}')

    @property
    def result(self) -> dict:
        result = {}
        for module, chained_analyzer in self._analyzers.items():
            previous_result = result.get(module, {})
            new_result = chained_analyzer.result
            previous_result.update(new_result)
            result.update({module: previous_result})
        return result

    def module_analyzer(self, module: str) -> ChainedAnalyzer:
        return self._analyzers.get(module, EmptyAnalyzer())

    def add_analyzer(self, analyzer: Analyzer):
        assert isinstance(analyzer, ChainedAnalyzer), "Modules must have a ChainedAnalyzer, not plain analyzer"
        super(AnalyzerController, self).add_analyzer(analyzer)