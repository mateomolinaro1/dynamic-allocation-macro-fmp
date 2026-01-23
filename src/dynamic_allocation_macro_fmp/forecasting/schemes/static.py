import pandas as pd
import numpy as np
from typing import Dict, Type, Iterable
import pandas as pd
from sklearn.metrics import mean_squared_error
from dynamic_allocation_macro_fmp.forecasting.models import Model
from dynamic_allocation_macro_fmp.utils.config import Config
from .base import EstimationScheme


class StaticScheme(EstimationScheme):
    """
    Fit once on an initial sample, then predict for all future dates.
    """

    def __init__(
        self,
        config: Config,
        x: pd.DataFrame,
        y: pd.DataFrame,
        forecast_horizon: int,
        validation_window: int,
        test_window: int,
        training_periods: int,
    ):
        super().__init__(config, x, y, forecast_horizon, validation_window, training_periods)
        self.test_window = test_window

        # Découpage fixe
        self.train_end = self.training_periods
        self.val_end = self.train_end + self.validation_window
        self.test_end = len(self.data) if self.test_window is None else self.val_end + self.test_window

        # Slices fixes
        self.train_data = self.data.iloc[:self.train_end]
        self.val_data = self.data.iloc[self.train_end:self.val_end]
        self.test_data = self.data.iloc[self.val_end:self.test_end]

        # Stockage des résultats
        self.oos_predictions = {}
        self.oos_true_val = self.val_data[[self.config.macro_var_name]]
        self.oos_true_test = self.test_data[[self.config.macro_var_name]]

    def _get_prediction_index(self, X, y) -> pd.Index:
        return X.index[X.index > self.train_end]

    def _get_training_index(self, t, X, y) -> pd.Index:
        return X.index[X.index <= self.train_end]

    def _should_refit(self, t) -> bool:
        if not self._fitted:
            self._fitted = True
            return True
        return False
    
    def get_train_validation_split(self, t=None):
        
        return self.train_data, self.val_data, self.val_data.index[-1]

    def run(self, models: Dict[str, Type[Model]], hyperparams_grid: Dict[str, Iterable[dict]]):
        
        hyperparams_all_combinations = self.build_hyperparams_combinations(hyperparams_grid)

        for model_name, ModelClass in models.items():
            best_score = float("inf")
            best_params = None
            best_model = None

            for params in hyperparams_all_combinations[model_name]:
                model = ModelClass(**params)
                x_train, y_train = self._split_xy(self.train_data)
                x_val, y_val = self._split_xy(self.val_data)

                model.fit(x_train, y_train)
                y_pred = model.predict(x_val)

                score = np.sqrt(mean_squared_error(y_val, y_pred))
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = model

            self.oos_predictions[model_name] = {
                "val": pd.Series(
                best_model.predict(self._split_xy(self.val_data)[0]).to_numpy().ravel(),
                index=self.val_data.index
                                ),
                "test": pd.Series(
                best_model.predict(self._split_xy(self.test_data)[0]).to_numpy().ravel(),
                index=self.test_data.index
                                )
            }

            if not hasattr(self, "best_params_all_models_overtime"):
                self.best_params_all_models_overtime = {}
            self.best_params_all_models_overtime[model_name] = best_params

            if not hasattr(self, "best_score_all_models_overtime"):
                self.best_score_all_models_overtime = {}
            self.best_score_all_models_overtime[model_name] = best_score
