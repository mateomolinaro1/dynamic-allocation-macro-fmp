from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Type, Iterable, List
from itertools import product
from dynamic_allocation_macro_fmp.forecasting.models import Model
from dynamic_allocation_macro_fmp.utils.config import Config


class EstimationScheme(ABC):
    """
    Abstract base class for estimation schemes
    (expanding, rolling, static).
    """

    def __init__(
        self,
        config: Config,
        x: pd.DataFrame,
        y: pd.DataFrame,
        forecast_horizon: int,
        validation_window: int,
        min_nb_periods_required: int,
    ):
        self.config = config
        self.x = x
        self.y = y
        if x.index != y.index:
            raise ValueError("X and Y must have the same index")
        self.data: pd.DataFrame = pd.concat([x, y], axis=1)

        self.forecast_horizon = forecast_horizon
        self.validation_window = validation_window
        self.min_nb_periods_required = min_nb_periods_required

        self.date_range = self.x.index

        # Storage (generic)
        self.oos_predictions: Dict[str, pd.DataFrame|pd.Series] = {}
        self.oos_true : pd.DataFrame = pd.DataFrame(
            index=self.date_range,
            columns=[self.config.macro_var_name],
        )
        self.best_score_all_models_overtime : pd.DataFrame = pd.DataFrame(
            index=self.date_range,
            columns=[],
        )
        self.hyperparams_all_models_overtime : Dict[str, pd.DataFrame] = {}
        self.best_params_all_models_overtime : Dict[str, pd.DataFrame] = {}

    # -----------------------------
    # CORE API
    # -----------------------------
    @abstractmethod
    def run(
        self,
        models: Dict[str, Type[Model]],
        hyperparams_grid: Dict[str, Iterable[dict]]
    ) -> None:
        """Main entry point"""

    # -----------------------------
    # SPLITTING LOGIC
    # -----------------------------
    @abstractmethod
    def get_train_validation_split(self, t: int):
        """
        Return (train_data, val_data, val_end_date)
        """

    # -----------------------------
    # UTILITIES
    # -----------------------------
    def _split_xy(self, df: pd.DataFrame):
        x = df.drop(columns=self.config.macro_var_name)
        y = df[self.config.macro_var_name]
        return x, y

    @staticmethod
    def build_hyperparams_combinations(
            hyperparameters_grid: Dict[str, Dict[str, list]]
    ) -> Dict[str, List[dict]]:

        hyperparams_all_combinations = {}

        for model_name, params in hyperparameters_grid.items():
            keys = params.keys()
            values = params.values()

            hyperparams_all_combinations[model_name] = [
                dict(zip(keys, combination))
                for combination in product(*values)
            ]

        return hyperparams_all_combinations