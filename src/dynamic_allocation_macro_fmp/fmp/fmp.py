from __future__ import annotations
import numpy as np
import pandas as pd
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.utils.config import Config, logger
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

        # storing results
        self.betas_macro = self._empty_like()
        self.betas_mkt = self._empty_like()
        self.white_var_betas = self._empty_like()
        self.newey_west_var_betas = self._empty_like()
        self.default_pvalue = self._empty_like()
        self.newey_west_pvalue = self._empty_like()
        self.adjusted_rsquared = self._empty_like()

    def fit_wls(self)->None:
        # Get X and ys (ys are y for each asset)
        X = self._get_x()
        X = X.sort_index()
        ys = self._get_ys()
        ys = ys.sort_index()

        # For each asset we run a regression
        for i,col in enumerate(ys.columns):
            logger.info(f"Running WLS ({i+1}/{len(ys.columns)})")
            y = ys.loc[:,col]

            # Align X and y
            xy = pd.merge_asof(left=y, right=X, left_index=True, right_index=True, direction="backward")
            xy = xy.dropna(axis=0, how="any")

            # Retrieve x and y
            x = xy.loc[:,X.columns]
            y = xy.loc[:,col]

            # Model
            min_obs = X.shape[1] + 1  # parameters incl. constant
            if len(y) <= min_obs:
                logger.info(f"Not enough data for asset {col}")
                continue

            res = Models.wls_exponential_decay(X=x,y=y,decay=self.config.decay)
            # Store
            date = y.index[-1]
            self.betas_macro.loc[date,col] = res["results"].params.loc[self.config.macro_var_name]
            self.betas_mkt.loc[date, col] = res["results"].params.loc["market_premium"]

            self.white_var_betas.loc[date, col] = (res["results"].HC0_se**2).loc[self.config.macro_var_name]
            self.newey_west_var_betas.loc[date, col] = (res["hac_bse"]**2).loc[self.config.macro_var_name]

            self.default_pvalue.loc[date, col] = res["results"].pvalues.loc[self.config.macro_var_name]
            self.newey_west_pvalue.loc[date, col] = res["hac_pvalues"].loc[self.config.macro_var_name]

            self.adjusted_rsquared.loc[date, col] = res["results"].rsquared_adj

        return

    def _get_x(self):
        mkt_premium = self._get_market_premium()
        macro_var_transformed = self._get_macro_var_change()
        x = pd.merge_asof(left=mkt_premium, right=macro_var_transformed, left_index=True, right_index=True, direction="backward")
        return x

    def _get_ys(self):
        if self.rf is None:
            return self.asset_returns
        raise NotImplementedError("Excess returns not implemented yet")

    def _get_market_premium(self)->pd.DataFrame:
        if self.market_returns is None:
            # We consider the market as the EW of all assets
            self.market_returns = self.asset_returns.mean(axis=1)
        else:
            raise NotImplementedError("Not implemented yet")

        if self.rf is None:
            # We consider rf=0
            market_premium = self.market_returns
        else:
            raise NotImplementedError("Not implemented yet")

        return pd.DataFrame(data=market_premium, columns=["market_premium"])

    def _get_macro_var_change(self)->pd.DataFrame:
        macro_var_transformed = FeaturesEngineering._preprocess_var(
            var=self.macro_var,
            code_transfo=self.data.code_transfo[self.config.macro_var_name]
        )
        return macro_var_transformed

    def _empty_like(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.nan,
            index=self.asset_returns.index,
            columns=self.asset_returns.columns,
        )






