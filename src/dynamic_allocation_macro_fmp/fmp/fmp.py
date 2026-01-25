from __future__ import annotations
import numpy as np
import pandas as pd
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.utils.config import Config, logger
from dynamic_allocation_macro_fmp.forecasting.models import WLSExponentialDecay
from dynamic_allocation_macro_fmp.forecasting.features_engineering import FeaturesEngineering
from dynamic_allocation_macro_fmp.utils.s3_utils import s3Utils
from dynamic_allocation_macro_fmp.backtesting.portfolio import EqualWeightingScheme
from dynamic_allocation_macro_fmp.backtesting.strategies import CrossSectionalPercentiles, BuyAndHold
from dynamic_allocation_macro_fmp.backtesting.backtest_pandas import Backtest

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
        # Macro exposure
        self.betas_macro = self._empty_like()
        self.betas_mkt = self._empty_like()
        self.white_var_betas = self._empty_like()
        self.newey_west_var_betas = self._empty_like()
        self.default_pvalue = self._empty_like()
        self.newey_west_pvalue = self._empty_like()
        self.adjusted_rsquared = self._empty_like()
        self.bayesian_betas = self._empty_like()
        # Portfolio returns
        self.positive_betas_fmp_returns = None
        self.negative_betas_fmp_returns = None
        self.benchmark_returns = None

    def build_macro_portfolios(self):
        if self.bayesian_betas.isna().all().all():
            try:
                # Download from s3
                self.bayesian_betas = s3Utils.pull_file_from_s3(
                    path=self.config.s3_path + "/outputs/fmp/fmp_bayesian_betas.parquet",
                    file_type="parquet"
                )
            except Exception as e:
                logger.info("Could not download bayesian_betas from s3, computing them locally.")
                self._get_betas()

        # Create portfolios
        strategy = CrossSectionalPercentiles(
            returns=self.asset_returns,
            signal_function=None,
            signal_function_inputs=None,
            signal_values=self.bayesian_betas,
            percentiles_winsorization=self.config.percentiles_winsorization
        )
        strategy.compute_signals_values()
        strategy.compute_signals(
            percentiles_portfolios=self.config.percentiles_portfolios,
            industry_segmentation=None
        )

        # Positive betas portfolio
        positive_ptf = EqualWeightingScheme(
            returns=self.asset_returns,
            signals=strategy.signals,
            rebal_periods=self.config.rebal_periods,
            portfolio_type=self.config.portfolio_type_positive
        )
        positive_ptf.compute_weights()
        positive_ptf.rebalance_portfolio()
        positive_backtester = Backtest(
            returns=self.asset_returns.shift(-1),
            weights=positive_ptf.rebalanced_weights,
            turnover=positive_ptf.turnover,
            transaction_costs=self.config.transaction_costs,
            strategy_name="POSITIVE_" + self.config.strategy_name
        )
        positive_backtester.run_backtest()
        self.positive_betas_fmp_returns = positive_backtester.cropped_portfolio_net_returns

        # Negative betas portfolio
        negative_ptf = EqualWeightingScheme(
            returns=self.asset_returns,
            signals=strategy.signals,
            rebal_periods=self.config.rebal_periods,
            portfolio_type=self.config.portfolio_type_negative
        )
        negative_ptf.compute_weights()
        negative_ptf.rebalance_portfolio()
        negative_backtester = Backtest(
            returns=self.asset_returns.shift(-1),
            weights=negative_ptf.rebalanced_weights,
            turnover=negative_ptf.turnover,
            transaction_costs=self.config.transaction_costs,
            strategy_name="NEGATIVE_" + self.config.strategy_name
        )
        negative_backtester.run_backtest()
        self.negative_betas_fmp_returns = negative_backtester.cropped_portfolio_net_returns

        # Benchmark
        bench_strategy = BuyAndHold(
            returns=self.asset_returns
        )
        bench_strategy.compute_signals_values()
        bench_strategy.compute_signals()
        bench_ptf = EqualWeightingScheme(
            returns=self.asset_returns,
            signals=bench_strategy.signals,
            rebal_periods=self.config.rebal_periods,
            portfolio_type="long_only"
        )
        bench_ptf.compute_weights()
        bench_ptf.rebalance_portfolio()
        bench_backtester = Backtest(
            returns=self.asset_returns.shift(-1),
            weights=bench_ptf.rebalanced_weights,
            turnover=bench_ptf.turnover,
            transaction_costs=self.config.transaction_costs,
            strategy_name="BENCHMARK_LO_EW"
        )
        bench_backtester.run_backtest()
        self.benchmark_returns = bench_backtester.cropped_portfolio_net_returns


    def _get_betas(self):
        self._fit_wls()
        self.bayesian_betas = self._get_bayesian_betas()

    def _fit_wls(self)->None:
        # Get X and ys (ys are y for each asset)
        X = self._get_x()
        X = X.sort_index()
        ys = self._get_ys()
        ys = ys.sort_index()

        # Expanding scheme
        for idx,date in enumerate(X.index[self.config.fmp_min_nb_periods_required:]):
            logger.info(f"Fitting WLS for date {date.date()} ({idx+1}/{len(X.index[self.config.fmp_min_nb_periods_required:])})")
            X_subset = X.loc[:date,:]
            ys_subset = ys.loc[:date,:]

            # For each asset we run a regression
            for i,col in enumerate(ys.columns):
                # logger.info(f"Running WLS ({i+1}/{len(ys_subset.columns)})")
                y = ys_subset.loc[:,col]

                # Align X and y
                xy = pd.merge_asof(left=y, right=X_subset, left_index=True, right_index=True, direction="backward")
                xy = xy.dropna(axis=0, how="any")

                # Retrieve x and y
                x = xy.loc[:,X_subset.columns]
                y = xy.loc[:,col]

                # Model
                min_obs = X_subset.shape[1] + 1  # parameters incl. constant
                if len(y) <= min_obs:
                    logger.info(f"Not enough data for asset {col}")
                    continue

                hyperparams = {"decay":self.config.decay}
                wls = WLSExponentialDecay(**hyperparams)
                wls.fit(x=x, y=y)
                # Store
                date = y.index[-1]
                self.betas_macro.loc[date,col] = wls.results.params.loc[self.config.macro_var_name]
                self.betas_mkt.loc[date, col] = wls.results.params.loc["market_premium"]

                self.white_var_betas.loc[date, col] = (wls.results.HC0_se**2).loc[self.config.macro_var_name]
                self.newey_west_var_betas.loc[date, col] = (wls.hac_bse**2).loc[self.config.macro_var_name]

                self.default_pvalue.loc[date, col] = wls.results.pvalues.loc[self.config.macro_var_name]
                self.newey_west_pvalue.loc[date, col] = wls.hac_pvalues.loc[self.config.macro_var_name]

                self.adjusted_rsquared.loc[date, col] = wls.results.rsquared_adj

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

    def _get_bayesian_betas(self):

        if self.betas_macro.isna().all().all():
            raise ValueError("fit_wls first.")

        # Cross-sectional prior
        prior_betas = self.betas_macro.mean(axis=1)

        # Time-series variance (Newey–West), aggregated across assets
        ts_var = self.newey_west_var_betas
        ts_var_agg = ts_var.mean(axis=1)

        # Cross-sectional variance
        cs_var = self.betas_macro.var(axis=1, ddof=0)

        # Shrinkage intensity
        denom = ts_var_agg + cs_var
        s = ts_var_agg / denom.replace(0, np.nan)

        # Bayesian shrinkage
        bayesian_betas = (
                prior_betas.values[:, None] * s.values[:, None]
                + self.betas_macro * (1 - s.values)[:, None]
        )

        return bayesian_betas







