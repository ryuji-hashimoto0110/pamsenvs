import ot
import pathlib
from pathlib import Path

class ParamsSearcher:
    """parameter searcher class.

    ParamsSearcher find appropriate parameters in artificial market configuration
    by calculating optimal transport distance between artificial dataset and real dataset.
    """
    def __init__(
        self,
        base_config_path: Path,
        search_params_config_path: Path
    ) -> None:
        """initialization."""