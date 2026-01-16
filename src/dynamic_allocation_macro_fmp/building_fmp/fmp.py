from __future__ import annotations
import pandas as pd
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.forecasting.models import Models
from dynamic_allocation_macro_fmp.forecasting.features_engineering import FeaturesEngineering

class FactorMimickingPortfolio:
    def __init__(
            self,
            config: Config,
            data: DataManager,
            market_returns: pd.DataFrame|None,
            rf: pd.DataFrame|None
    ):
        self.config = config
        self.data = data
        self.asset_returns = self.data.returns_data
        self.macro_var = self.data.fred_data[[self.config.macro_var_name]]
        self.market_returns = market_returns
        self.rf = rf

        self.dates = self.asset_returns.index

    def fit_wls(self):
        X = self._get_x()
        y = self._get_y()
        res = Models.wls_exponential_decay(X,y,decay=self.config.decay)
        return res

    def _get_x(self):
        self._get_market_premium()
        self._get_macro_var_change()

    def _get_y(self):
        ...

    def _get_market_premium(self)->pd.DataFrame:
        if self.market_returns is None:
            # We consider the market as the EW of all assets
            self.market_returns = self.asset_returns.mean(axis=1)
        else:
            ...

        if self.rf is None:
            # We consider rf=0
            market_premium = self.market_returns
        else:
            ...

        return market_premium

    def _get_macro_var_change(self):
        macro_var_transformed = FeaturesEngineering._preprocess_var(
            var=self.macro_var,
            code_transfo=self.data.code_transfo[self.config.macro_var_name]
        )
        return macro_var_transformed





