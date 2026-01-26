from __future__ import annotations
import logging
import pandas as pd
from typing import Dict
from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.backtesting.portfolio import EqualWeightingScheme
from dynamic_allocation_macro_fmp.backtesting.backtest_pandas import Backtest

logger = logging.getLogger(__name__)

class DynamicAllocation:
    """
    Class to handle dynamic allocation based on macro predictions and FMPs.
    """
    def __init__(
            self,
            config: Config,
            predictions: Dict[str, pd.DataFrame],
            long_leg_fmp: pd.DataFrame,
            short_leg_fmp: pd.DataFrame,
            benchmark_ptf: pd.DataFrame
    ):
        self.config = config
        self.predictions = predictions
        self.long_leg_fmp = long_leg_fmp
        self.short_leg_fmp = short_leg_fmp
        self.benchmark_ptf = benchmark_ptf
        self.date_range = self.long_leg_fmp.index
        self.asset_returns = None
        self.net_returns = None
        self.benchmark_ew_fmp_net_returns = None

    def run_backtest(self) -> None:
        """
        Compute allocation signals based on predictions.
        """
        merged = self._merge_dfs()
        signals = self._compute_signals(merged)
        self._backtest_pipeline(signals)
        return

    def _merge_dfs(self)->dict[str, pd.DataFrame]:
        end_ret_ptf = (
                self.predictions[list(self.predictions.keys())[0]].index
                + pd.DateOffset(months=2)
                + pd.DateOffset(days=1)
        )
        tmp_predictions = self.predictions.copy()
        for k,v in tmp_predictions.items():
            v.index = end_ret_ptf

        # Getting a merged df of predictions and ptf returns for each model
        merged = {}
        for model_name, oos_predictions in tmp_predictions.items():
            merged[model_name] = pd.merge(
                oos_predictions,
                self.long_leg_fmp,
                left_index=True,
                right_index=True,
                how="inner"
            )
            merged[model_name] = pd.merge(
                merged[model_name],
                self.short_leg_fmp,
                left_index=True,
                right_index=True,
                how="inner"
            )
            merged[model_name] = pd.merge(
                merged[model_name],
                self.benchmark_ptf,
                left_index=True,
                right_index=True,
                how="inner"
            )
            merged[model_name].index = (
                merged[model_name].index
                - pd.DateOffset(months=1)
                - pd.DateOffset(days=1)
            )
            merged[model_name] = merged[model_name].dropna()

        self.asset_returns = merged[model_name].drop(columns=[self.config.macro_var_name])
        self.benchmark_ew_fmp_net_returns = self.asset_returns[["BENCHMARK_LO_EW"]]
        self.asset_returns["NEGATIVE_EW_MACRO_FMP"] = -self.asset_returns["NEGATIVE_EW_MACRO_FMP"]
        return merged


    def _compute_signals(self, merged:dict)->dict[str, pd.DataFrame]:
        signals = {}
        for model_name, df in merged.items():
            cols = list(df.columns.drop(self.config.macro_var_name))
            signals[model_name] = pd.DataFrame(
                data=0.0,
                index = df.index,
                columns=cols,
            )
            signals[model_name].loc[df[self.config.macro_var_name] > 0, "POSITIVE_EW_MACRO_FMP"] = 1.0
            signals[model_name].loc[df[self.config.macro_var_name] < 0, "NEGATIVE_EW_MACRO_FMP"] = 1.0
            signals[model_name].loc[df[self.config.macro_var_name].isna() | (df[self.config.macro_var_name] == 0), "BENCHMARK_LO_EW"] = 1.0
        return signals

    def _backtest_pipeline(self, signals:dict[str, pd.DataFrame])->None:
        net_returns = {}

        for model_name, signal_df in signals.items():
            ptf = EqualWeightingScheme(
                returns=self.asset_returns,
                signals=signal_df,
                rebal_periods=self.config.dynamic_allocation_rebal_periods,
                portfolio_type="long_only"
            )
            ptf.compute_weights()
            ptf.rebalance_portfolio()

            backtester = Backtest(
                returns=self.asset_returns,
                weights=ptf.rebalanced_weights,
                turnover=ptf.turnover,
                transaction_costs=self.config.dynamic_allocation_tc,
                strategy_name=f"DYNAMIC_ALLOCATION_{model_name}"
            )
            backtester.run_backtest()
            net_returns[model_name] = backtester.cropped_portfolio_net_returns

        # Save
        self.net_returns = net_returns

        # Same logic for benchmark, which is an EW between long and short FMP portfolios
        asset_returns_bench = self.asset_returns.copy()
        asset_returns_bench = asset_returns_bench.drop(columns=["BENCHMARK_LO_EW"])
        signal_df = pd.DataFrame(
            data=1.0,
            index=asset_returns_bench.index,
            columns=asset_returns_bench.columns,
        )

        bench_ptf = EqualWeightingScheme(
            returns=asset_returns_bench,
            signals=signal_df,
            rebal_periods=self.config.dynamic_allocation_rebal_periods,
            portfolio_type="long_only"
        )
        bench_ptf.compute_weights()
        bench_ptf.rebalance_portfolio()

        bench_backtester = Backtest(
            returns=asset_returns_bench,
            weights=bench_ptf.rebalanced_weights,
            turnover=bench_ptf.turnover,
            transaction_costs=self.config.dynamic_allocation_tc,
            strategy_name=f"BENCHMARK_EW_FMP"
        )
        bench_backtester.run_backtest()
        self.benchmark_ew_fmp_net_returns = bench_backtester.cropped_portfolio_net_returns

        return
