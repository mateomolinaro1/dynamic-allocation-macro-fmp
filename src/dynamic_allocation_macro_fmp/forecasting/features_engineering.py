from __future__ import annotations
import pandas as pd
import numpy as np
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.utils.config import Config, logger
from sklearn.preprocessing import StandardScaler 

class FeaturesEngineering:
    def __init__(
            self,
            config: Config,
            data: DataManager
    ):
        self.config = config
        self.data = data

        self.transformed_fred_data = None
        self.y = None
        self.x = None

    @staticmethod
    def _preprocess_var(var: pd.DataFrame, code_transfo: int | float):
        var = var.astype(float)

        if code_transfo == 1.0:
            return var
        elif code_transfo == 2.0:
            return var.diff()
        elif code_transfo == 3.0:
            return var.diff().diff()
        elif code_transfo == 4.0:
            return np.log(var)
        elif code_transfo == 5.0:
            return np.log(var).diff()
        elif code_transfo == 6.0:
            return np.log(var).diff().diff()
        elif code_transfo == 7.0:
            return var.pct_change()
        else:
            raise ValueError(f"Unknown transformation code: {code_transfo}")

    def _transform_fred_date(self):
        if self.transformed_fred_data is None:
            transformed_data = {}
            for col in self.data.fred_data.columns:
                logger.info(f"Transforming FRED variable: {col}")
                transformed_data[col] = self._preprocess_var(
                    var=self.data.fred_data[col],
                    code_transfo=self.data.code_transfo[col]
                )
            df = pd.DataFrame(transformed_data)
            df.index = self.data.fred_data.index
            self.transformed_fred_data = df

    def _build_lagged_features(self):
        lagged_vars = {}
        for col in self.transformed_fred_data.columns:
            for lag in self.config.lags:
                logger.info(f"Building lagged feature: {col}_lag{lag}")
                lagged_vars[f"{col}_lag{lag}"] = self.transformed_fred_data[col].shift(lag)
        lagged_df = pd.DataFrame(lagged_vars)
        lagged_df.index = self.transformed_fred_data.index
        self.transformed_fred_data = pd.concat([self.transformed_fred_data, lagged_df], axis=1)

    def _handle_missing_values(self):
        self.transformed_fred_data = self.transformed_fred_data.dropna()

    def _crop_date_range(self):
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        if not pd.isna(start_date) and not pd.isna(end_date):
            self.transformed_fred_data = self.transformed_fred_data.loc[
                (self.transformed_fred_data.index >= start_date) &
                (self.transformed_fred_data.index <= end_date)
            ,:]
        if not pd.isna(start_date):
            self.transformed_fred_data = self.transformed_fred_data.loc[
                self.transformed_fred_data.index >= start_date
            ,:]
        if not pd.isna(end_date):
            self.transformed_fred_data = self.transformed_fred_data.loc[
                self.transformed_fred_data.index <= end_date
            ,:]

    def _split_y_x(self):
        self.y = self.transformed_fred_data[[self.config.macro_var_name]].copy().shift(-self.config.forecast_horizon)
        self.x = self.transformed_fred_data.drop(columns=[self.config.macro_var_name]).copy()
        # Avoid overloading memory
        self.transformed_fred_data = None

    def get_features(self):
        self._transform_fred_date()
        self._build_lagged_features()
        self._crop_date_range()
        # Missing values will be handled later in the pipeline, more precisely when
        # passing the features to the model because we'll use expanding/rolling windows.
        self._split_y_x()


class StandardScaler():
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, x: pd.DataFrame):
        x_scaled = self.scaler.fit_transform(x)
        return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_scaled = self.scaler.transform(x)
        return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)




