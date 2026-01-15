from dataclasses import dataclass
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configuration object to hold settings for the application.
    """
    def __init__(self):
        # Paths
        try:
            self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
        except NameError:
            self.ROOT_DIR = Path.cwd()
        logger.info("Root dir: " + str(self.ROOT_DIR))

        self.RUN_PIPELINE_CONFIG_PATH = self.ROOT_DIR / "configs" / "run_pipeline_config.json"
        logger.info("run_pipeline config path: " + str(self.RUN_PIPELINE_CONFIG_PATH))

        # Data path configurations
        self.fred_path: str|None|Path = None
        self.prices_path: str|None|Path = None
        self.outputs_path: str|None|Path = None


        # Load json config to attributes of Config class
        self._load_run_pipeline_config()

    def _load_run_pipeline_config(self)->None:
        """
        Load run_pipeline_config.json file
        :return:
        """
        with open(self.ROOT_DIR / "configs" / "run_pipeline_config.json" , "r") as f:
            config: dict = json.load(f)

            if config.get("PATHS").get("FRED_PATH") is None:
                self.fred_path = self.ROOT_DIR / "data" / "macro" / "FRED-MD-2024-12.csv"
            else:
                self.fred_path = config.get("PATHS").get("FRED_PATH")

            if config.get("PATHS").get("PRICES_PATH") is None:
                self.prices_path = self.ROOT_DIR / "data" / "market" / "prices.parquet"
            else:
                self.prices_path = config.get("PATHS").get("PRICES_PATH")

            if config.get("PATHS").get("OUTPUTS_PATH") is None:
                self.outputs_path = self.ROOT_DIR / "outputs"
            else:
                self.outputs_path = config.get("PATHS").get("OUTPUTS_PATH")
