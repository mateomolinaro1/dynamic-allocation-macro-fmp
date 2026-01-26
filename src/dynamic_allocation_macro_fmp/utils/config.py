from dataclasses import dataclass
from pathlib import Path
import logging
import json
from typing import List, Dict, Type, Tuple
from dynamic_allocation_macro_fmp.forecasting.models import (
    Model, Lasso, WLSExponentialDecay, OLS, RidgeModel, ElasticNetModel, RandomForestModel,
    GradientBoostingModel, SVRModel, NeuralNetModel, LightGBMModel, XGBoostModel
)

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
        self.fmp_min_nb_periods_required: int|None = None
        self.percentiles_winsorization: Tuple[int, int]|None = None
        self.percentiles_portfolios: Tuple[int, int]|None = None
        self.rebal_periods: int|None = None
        self.portfolio_type_positive: str|None = None
        self.portfolio_type_negative: str|None = None
        self.transaction_costs: float|int|None = None
        self.strategy_name: str|None = None

        # Feature engineering
        self.start_date: str|None = None
        self.end_date: str|None = None
        self.lags: List[int]|list|None = None

        # Forecasting
        self.forecast_horizon: int|None = None
        self.validation_window: int|None = None
        self.min_nb_periods_required: int|None = None
        self.models: Dict[str, Type[Model]]|None = None
        self.hyperparams_grid: Dict[str, dict]|None = None
        self.with_pca: bool|None = None
        self.nb_pca_components: int|None = None

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
            self.fmp_min_nb_periods_required = config.get("FMP").get("MIN_NB_PERIODS_REQUIRED")
            self.percentiles_winsorization = tuple(config.get("FMP").get("PERCENTILES_WINSORIZATION"))
            self.percentiles_portfolios = tuple(config.get("FMP").get("PERCENTILES_PORTFOLIOS"))
            self.rebal_periods = config.get("FMP").get("REBAL_PERIODS")
            self.portfolio_type_positive = config.get("FMP").get("PORTFOLIO_TYPE_POSITIVE")
            self.portfolio_type_negative = config.get("FMP").get("PORTFOLIO_TYPE_NEGATIVE")
            self.transaction_costs = config.get("FMP").get("TRANSACTION_COSTS_BPS")
            self.strategy_name = config.get("FMP").get("STRATEGY_NAME")

            # Feature engineering
            self.start_date = config.get("FEATURE_ENGINEERING").get("START_DATE")
            self.end_date = config.get("FEATURE_ENGINEERING").get("END_DATE")
            self.lags = config.get("FEATURE_ENGINEERING").get("LAGS")

            # Forcasting
            self.forecast_horizon = config.get("FORECASTING").get("FORECAST_HORIZON")
            self.validation_window = config.get("FORECASTING").get("VALIDATION_WINDOW")
            self.min_nb_periods_required = config.get("FORECASTING").get("MIN_NB_PERIODS_REQUIRED")
            models = {
                "wls": WLSExponentialDecay,
                "lasso": Lasso,
                "ols": OLS,
                "ridge": RidgeModel,
                "elastic_net": ElasticNetModel,
                "random_forest": RandomForestModel,
                "gradient_boosting": GradientBoostingModel,
                "svr": SVRModel,
                "neural_net": NeuralNetModel,
                "lightgbm": LightGBMModel,
                "xgboost": XGBoostModel,
            }
            for model in config.get("FORECASTING").get("MODELS"):
                if model not in models.keys():
                    raise ValueError(f"Model {model} not implemented.")
            self.models = {model: models[model] for model in config.get("FORECASTING").get("MODELS")}
            self.with_pca = config.get("FORECASTING").get("WITH_PCA")
            self.nb_pca_components = config.get("FORECASTING").get("NB_PCA_COMPONENTS")
            if self.with_pca:
                self.models.update({model+"_pca": models[model] for model in config.get("FORECASTING").get("MODELS")})
            self.hyperparams_grid = config.get("FORECASTING").get("HYPERPARAMS_GRID")
            if self.with_pca:
                for model_name, model in self.models.items():
                    if model_name.endswith("_pca"):
                        base_model_name = model_name[:-4]
                        self.hyperparams_grid[model_name] = self.hyperparams_grid.get(base_model_name)
