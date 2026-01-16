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

        # AWS profile
        self.aws_profile: str|None = None

        # Data path configurations
        self.fred_path: str|None|Path = None
        self.prices_path: str|None|Path = None
        self.outputs_path: str|None|Path = None
        self.s3_path: str|None|Path = None

        # Files ext
        self.macro_ext: str|None = None
        self.prices_ext: str|None = None

        # FMP
        self.decay: int|float|None = None
        self.macro_var_name: str|None = None

        # Load json config to attributes of Config class
        self._load_run_pipeline_config()

    def _load_run_pipeline_config(self)->None:
        """
        Load run_pipeline_config.json file
        :return:
        """
        with open(self.ROOT_DIR / "configs" / "run_pipeline_config.json" , "r") as f:
            config: dict = json.load(f)

            # AWS
            if config.get("AWS").get("PROFILE") is not None:
                self.aws_profile = config.get("AWS").get("PROFILE")

            # Paths
            if config.get("PATHS").get("S3_FRED_PATH") is not None:
                self.fred_path = config.get("PATHS").get("S3_FRED_PATH")

            if config.get("PATHS").get("S3_PRICES_PATH") is not None:
                self.prices_path = config.get("PATHS").get("S3_PRICES_PATH")

            if config.get("PATHS").get("S3_OUTPUTS_PATH") is not None:
                self.outputs_path = config.get("PATHS").get("S3_OUTPUTS_PATH")

            if config.get("PATHS").get("S3_PATH") is not None:
                self.s3_path = config.get("PATHS").get("S3_PATH")

            # Files ext
            self.macro_ext = config.get("FILES_EXT").get("MACRO")
            self.prices_ext = config.get("FILES_EXT").get("PRICES")

            # FMP
            self.decay = config.get("FMP").get("DECAY")
            self.macro_var_name = config.get("FMP").get("MACRO_VAR_NAME")
