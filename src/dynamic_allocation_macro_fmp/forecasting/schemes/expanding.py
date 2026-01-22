from __future__ import annotations
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from typing import Dict, Iterable, Type
from .base import EstimationScheme
import logging
from dynamic_allocation_macro_fmp.forecasting.models import Model

logger = logging.getLogger(__name__)


class ExpandingWindowScheme(EstimationScheme):

    def run(
        self,
        models: Dict[str, Type[Model]],
        hyperparams_grid: Dict[str, Dict[str, list]]
    ) -> None:

        # -----------------------------
        # Hyperparams combinations
        # -----------------------------
        hyperparams_all_combinations = self.build_hyperparams_combinations(
            hyperparameters_grid=hyperparams_grid
        )

        # -----------------------------
        # STORE BETAS OVER TIME (linear only)
        # -----------------------------
        linear_models = []
        for model_name, model in models.items():
            if model_name in ["ridge","lasso","elastic_net","ols"]:
                linear_models.append(model_name)

        self.best_params_all_models_overtime = {
            m: (
                pd.DataFrame(
                    index=self.date_range,
                    columns=["intercept"]
                            + list(
                        self.data.drop(columns=[self.config.macro_var_name]).columns
                    ),
                    dtype=float,
                )
                if m in linear_models
                else None
            )
            for m in models
        }

        # -----------------------------
        # STORAGE
        # -----------------------------
        self.oos_predictions = {m: pd.DataFrame(
            index=self.date_range,
            data=np.nan,
            columns=self.config.macro_var_name
        ) for m in models}
        self.best_score_all_models_overtime.columns = models.keys()
        self.hyperparams_all_models_overtime = {
            m: pd.DataFrame(
                index=self.date_range,
                columns=list(hyperparams_all_combinations[m][0].keys()),
            )
            for m in models
        }

        # -----------------------------
        # WALK-FORWARD LOOP
        # -----------------------------
        start_idx = (
            self.min_nb_periods_required
            + self.validation_window
            + self.forecast_horizon
        )

        for t in range(start_idx, len(self.date_range) - self.forecast_horizon):
            date_t = self.date_range[t]
            logger.info(
                f"Training models for date {date_t} "
                f"({t}/{len(self.date_range) - self.forecast_horizon - 1})"
            )

            t0 = time.time()

            train_data, val_data, val_end = self._get_train_validation_split(t)
            X_train, y_train = self._split_xy(train_data)

            for model_name, ModelClass in models.items():
                logger.info(f"Model: {model_name}")

                best_score = -np.inf
                best_hyperparams = None

                # -----------------------------
                # HYPERPARAMETER SEARCH
                # -----------------------------
                def evaluate(hyperparams):
                    model = ModelClass(**hyperparams)
                    model.fit(X_train, y_train)

                    ICs = []
                    for d in np.sort(val_data.index.unique()):
                        val_d = val_data[val_data.index == d]
                        X_val, y_val = self._split_xy(val_d)

                        if len(y_val) <= 1:
                            continue

                        y_hat = model.predict(X_val)
                        ic = np.sqrt(mean_squared_error(y_val, y_hat))
                        if not np.isnan(ic):
                            ICs.append(ic)

                    if not ICs:
                        return None

                    return np.mean(ICs), hyperparams

                # =========================
                # PARALLEL GRID SEARCH
                # =========================
                results = Parallel(n_jobs=-1)(
                    delayed(evaluate)(hp)
                    for hp in hyperparams_all_combinations[model_name]
                )

                for res in results:
                    if res is None:
                        continue
                    score, hp = res
                    if score > best_score:
                        best_score = score
                        best_hyperparams = hp

                # -----------------------------
                # FINAL TRAIN
                # -----------------------------
                full_train = self.data[self.data.index <= val_end]
                X_full, y_full = self._split_xy(full_train)

                model_final = ModelClass(**best_hyperparams)
                model_final.fit(X_full, y_full)

                test_date = self.date_range[t]
                test_df = self.data[self.data.index == test_date]
                X_test, y_test = self._split_xy(test_df)

                y_hat = model_final.predict(X_test)

                self.oos_predictions[model_name].loc[test_date,self.config.macro_var_name] = y_hat
                self.oos_true.loc[test_date,self.config.macro_var_name] = y_test

                logger.info(
                    f"{model_name} done in "
                    f"{round((time.time() - t0) / 60, 3)} min"
                )

    # -----------------------------
    # SPLITS
    # -----------------------------
    def _get_train_validation_split(self, t: int):
        train_end = self.date_range[t - self.validation_window - self.forecast_horizon]
        val_end = self.date_range[t - self.forecast_horizon]

        train_data = self.data[self.data.index <= train_end]
        val_data = self.data[
            (self.data[self.data.index] > train_end)
            & (self.data[self.data.index] <= val_end)
        ]

        return train_data, val_data, val_end

