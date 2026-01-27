from pathlib import Path
import pandas as pd
import numpy as np
import dataframe_image as dfi
import matplotlib

from dynamic_allocation_macro_fmp.dynamic_allocation.dynamic_allocation import DynamicAllocation

matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import logging
from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.fmp.fmp import FactorMimickingPortfolio
from dynamic_allocation_macro_fmp.forecasting.schemes.expanding import ExpandingWindowScheme
from dynamic_allocation_macro_fmp.utils.vizu import Vizu
from typing import Type

logger = logging.getLogger(__name__)

class AnalyticsFMP:
    def __init__(self, config:Config, fmp=Type[FactorMimickingPortfolio]):
        """
        Analytics for Factor Mimicking Portfolio (FMP)
        :param config:
        :param fmp:
        """
        self.config = config
        self.fmp = fmp

    def get_analytics(self) -> None:
        """
        Get analytics for FMP
        :return: None
        """
        self._get_bayesian_betas_distribution()
        self._get_bayesian_betas_overtime()
        self._get_table_comparison_between_bayesian_and_non_bayesian_betas()
        self._get_summary_statistics_rsquared()
        self._get_cross_sectional_statistics_rsquared()
        self._plot_significant_proportion_bayesian_betas_overtime()
        self._plot_equity_curves_fmp()
        self._export_fmp_performance_table()

    def _get_bayesian_betas_distribution(self):
        betas = self.fmp.bayesian_betas.stack().dropna()

        start_date = betas.dropna(how='all').index[0][0]
        end_date = betas.index[-1][0]

        # Clip extreme tails only for visualization
        lo, hi = betas.quantile([0.01, 0.99])
        betas_clip = betas.clip(lo, hi)

        plt.figure(figsize=(10, 6))
        plt.hist(betas_clip, bins=40, density=True, alpha=0.6, color="steelblue")
        betas_clip.plot(kind="kde", color="darkred", linewidth=2)

        plt.title(
            f"Bayesian Betas Distribution (clipped 1–99%)\n{start_date.date()} – {end_date.date()}"
        )
        plt.xlabel("Bayesian Beta")
        plt.ylabel("Density")
        plt.tight_layout()

        plt.savefig(self.config.ROOT_DIR / "outputs/figures/bayesian_betas_distribution.png")
        plt.close()
        logger.info("Saved Bayesian betas distribution plot.")

    def _get_bayesian_betas_overtime(self):
        betas = self.fmp.bayesian_betas

        avg = betas.mean(axis=1)
        p10 = betas.quantile(0.10, axis=1)
        p90 = betas.quantile(0.90, axis=1)

        start_date, end_date = avg.dropna().index[[0, -1]]

        plt.figure(figsize=(12, 6))
        plt.plot(avg, label="Cross-sectional mean", linewidth=2)
        plt.fill_between(p10.index, p10, p90, alpha=0.3, label="10–90 percentile band")

        # Robust y-limits
        ylo, yhi = np.nanpercentile(betas.values, [2, 98])
        plt.ylim(ylo, yhi)

        plt.title(f"Bayesian Betas Cross-Sectional Dynamics\n{start_date.date()} – {end_date.date()}")
        plt.ylabel("Beta")
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.config.ROOT_DIR / "outputs/figures/bayesian_betas_overtime.png")
        plt.close()
        logger.info("Saved Bayesian betas over time plot.")

    def _get_cross_sectional_statistics_rsquared(self):
        rsq = self.fmp.adjusted_rsquared

        avg = rsq.mean(axis=1)
        p10 = rsq.quantile(0.10, axis=1)
        p90 = rsq.quantile(0.90, axis=1)

        start_date, end_date = avg.dropna().index[[0, -1]]

        plt.figure(figsize=(12, 6))
        plt.plot(avg, label="Mean adjusted $R^2$", linewidth=2)
        plt.fill_between(p10.index, p10, p90, alpha=0.3)

        plt.ylim(0, 1)
        plt.title(f"Adjusted $R^2$ Cross-Sectional Dynamics\n{start_date.date()} – {end_date.date()}")
        plt.ylabel("Adjusted $R^2$")
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.config.ROOT_DIR / "outputs/figures/rsquared_overtime.png")
        plt.close()
        logger.info("Saved R-squared over time plot.")

    def _plot_significant_proportion_bayesian_betas_overtime(self, significance_level=0.05):
        p_values = self.fmp.newey_west_pvalue

        significant = p_values < significance_level
        total = p_values.notnull().sum(axis=1)
        proportion = significant.sum(axis=1) / total

        start_date, end_date = proportion.dropna().index[[0, -1]]

        plt.figure(figsize=(12, 6))
        plt.plot(proportion, linewidth=2)
        plt.ylim(0, 1)

        plt.title(
            f"Proportion of Significant Bayesian Betas (α={significance_level})\n"
            f"{start_date.date()} – {end_date.date()}"
        )
        plt.ylabel("Proportion")
        plt.tight_layout()

        plt.savefig(
            self.config.ROOT_DIR
            / "outputs/figures/proportion_significant_bayesian_betas_overtime.png"
        )
        plt.close()
        logger.info("Saved proportion of significant Bayesian betas over time plot.")

    def _get_table_comparison_between_bayesian_and_non_bayesian_betas(self):
        """
        Create a dataframe comparing Bayesian and non-Bayesian betas summary statistics
        """
        bayesian_betas = self.fmp.bayesian_betas.stack().dropna()
        non_bayesian_betas = self.fmp.betas_macro.stack().dropna()

        stats = ["Mean", "Median", "Std Dev", "Min", "Max"]

        summary_df = pd.DataFrame(
            {
                "Bayesian Betas": [
                    bayesian_betas.mean(),
                    bayesian_betas.median(),
                    bayesian_betas.std(),
                    bayesian_betas.min(),
                    bayesian_betas.max(),
                ],
                "Non-Bayesian Betas": [
                    non_bayesian_betas.mean(),
                    non_bayesian_betas.median(),
                    non_bayesian_betas.std(),
                    non_bayesian_betas.min(),
                    non_bayesian_betas.max(),
                ],
            },
            index=stats,
        )

        # Nice formatting for export
        summary_df = summary_df.round(2)

        output_path = (
                self.config.ROOT_DIR
                / "outputs"
                / "figures"
                / "bayesian_vs_non_bayesian_betas_summary.png"
        )
        dfi.export(summary_df, output_path, table_conversion="matplotlib")

        logger.info("Saved Bayesian vs Non-Bayesian Betas summary statistics table.")

    def _get_summary_statistics_rsquared(self):
        """
        Create a dataframe summarizing adjusted R-squared statistics
        """
        rsquared = self.fmp.adjusted_rsquared.stack().dropna()

        stats = ["Mean", "Median", "Std Dev", "Min", "Max"]

        summary_df = pd.DataFrame(
            {
                "Adjusted R-squared": [
                    rsquared.mean(),
                    rsquared.median(),
                    rsquared.std(),
                    rsquared.min(),
                    rsquared.max(),
                ]
            },
            index=stats,
        )

        summary_df = summary_df.round(2)

        output_path = (
                self.config.ROOT_DIR
                / "outputs"
                / "figures"
                / "rsquared_summary.png"
        )
        dfi.export(summary_df, output_path, table_conversion="matplotlib")

        logger.info("Saved R-squared summary statistics table.")

    def _plot_equity_curves_fmp(self):
        returns = pd.concat(
            [
                self.fmp.positive_betas_fmp_returns,
                self.fmp.negative_betas_fmp_returns,
                self.fmp.benchmark_returns,
            ],
            axis=1,
        ).dropna(how="all")

        start_date = returns.index[0]

        equity = returns.loc[start_date:].cumsum()

        Vizu.plot_time_series(
            data=equity,
            title="LO / SO Factor-Mimicking Portfolios – Equity Curves",
            ylabel="Equity Curve",
            xlabel="Date",
            save_path=self.config.ROOT_DIR / "outputs/figures/fmp_equity_curves.png",
            show=False,
            block=False,
        )
        logger.info("Saved FMP equity curves plot.")

    @staticmethod
    def _compute_performance_metrics(
            returns: pd.Series | pd.DataFrame,
            portfolio_type: str = "long_only",
            freq: int = 12,
            risk_free_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Compute performance, risk and risk-adjusted metrics.

        Metrics:
        - Annualized Return
        - Annualized Volatility
        - Sharpe Ratio
        - Max Drawdown

        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Periodic portfolio returns
        portfolio_type : {"long_only", "short_only", "long_short"}
        freq : int
            Annualization factor (252 for daily, 12 for monthly)
        risk_free_rate : float
            Annualized risk-free rate

        Returns
        -------
        pd.DataFrame
            Metrics table
        """

        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()

        if not isinstance(returns, pd.Series):
            raise ValueError("returns must be a pd.Series or single-column DataFrame")

        # CRITICAL FIX: short-only economics
        if portfolio_type == "short_only":
            returns = -returns

        returns = returns.dropna()

        # === Annualized return (log-safe version) ===
        ann_return = (1 + returns).prod() ** (freq / len(returns)) - 1

        # === Annualized volatility ===
        ann_vol = returns.std() * np.sqrt(freq)

        # === Sharpe ratio ===
        sharpe = np.nan
        if ann_vol > 0:
            sharpe = (ann_return - risk_free_rate) / ann_vol

        # === Max drawdown ===
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1
        max_dd = drawdown.min()

        metrics = pd.DataFrame(
            {
                "Annualized Return": ann_return,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd,
            },
            index=["Portfolio"],
        )

        return metrics.round(2)

    @staticmethod
    def _export_fmp_performance_table_util(
            fmp,
            save_path: Path,
            freq: int = 12,
            risk_free_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Compute and export performance metrics for FMP portfolios.

        Portfolios:
        - Long-only FMP
        - Short-only FMP (sign-corrected)
        - Benchmark

        Metrics:
        - Annualized Return
        - Annualized Volatility
        - Sharpe Ratio
        - Max Drawdown

        Parameters
        ----------
        fmp : FactorMimickingPortfolio
            FMP object with computed returns
        save_path : Path
            Path to save the exported table (png)
        freq : int
            Annualization factor
        risk_free_rate : float
            Annualized risk-free rate

        Returns
        -------
        pd.DataFrame
            Performance metrics table
        """

        perf_pos = AnalyticsFMP._compute_performance_metrics(
            fmp.positive_betas_fmp_returns,
            portfolio_type="long_only",
            freq=freq,
            risk_free_rate=risk_free_rate
        )

        perf_neg = AnalyticsFMP._compute_performance_metrics(
            fmp.negative_betas_fmp_returns,
            portfolio_type="short_only",  # critical
            freq=freq,
            risk_free_rate=risk_free_rate
        )

        perf_bench = AnalyticsFMP._compute_performance_metrics(
            fmp.benchmark_returns,
            portfolio_type="long_only",
            freq=freq,
            risk_free_rate=risk_free_rate
        )

        performance_table = pd.concat(
            [perf_pos, perf_neg, perf_bench],
            axis=0
        )

        performance_table.index = ["LO FMP", "SO FMP", "Benchmark"]

        # Export as image
        dfi.export(performance_table, save_path)

        return performance_table

    def _export_fmp_performance_table(self):
        save_path = (
                self.config.ROOT_DIR
                / "outputs"
                / "figures"
                / "fmp_performance_summary.png"
        )

        AnalyticsFMP._export_fmp_performance_table_util(
            fmp=self.fmp,
            save_path=save_path
        )

        logger.info("Saved FMP performance summary table.")

class AnalyticsForecasting:
    def __init__(self, config: Config, exp_window: ExpandingWindowScheme):
        self.config = config
        self.exp_window = exp_window
        self.objs = None
        self.rmse_df = None
        self.linear_models = ["ols", "ols_pca", "lasso", "lasso_pca", "ridge", "ridge_pca", "elastic_net", "elastic_net_pca"]
        self.linear_models_without_pca = ["ols", "lasso", "ridge", "elastic_net"]
        self.non_numeric_hyperparams = ["neural_net", "neural_net_pca", "ols", "ols_pca", "svr", "svr_pca"]

    def get_analytics(self) -> None:
        """Run all forecasting analytics"""
        self._load_objects()
        self._export_features_table()
        self._plot_best_score_overtime()
        self._plot_best_hyperparams_overtime()
        self._plot_model_parameters_overtime()
        self._export_selected_features_proportion()
        self._export_mean_parameters()
        self._plot_oos_rmse_overtime()
        self._export_oos_rmse_table()
        self._compute_oos_sign_accuracy()
        logger.info("Completed forecasting analytics.")

    def _load_objects(self):
        self.objs = {k: v for k, v in vars(self.exp_window).items()}

    def _export_features_table(self):
        dfi.export(
            self.objs["x"],
            self.config.ROOT_DIR / "outputs" / "figures" / "features_df_short.png",
            max_rows=10,
            max_cols=10
        )

    def _plot_best_score_overtime(self, y_lim:float=0.015):
        df = self.objs["best_score_all_models_overtime"]

        plt.figure(figsize=(10, 6))
        plt.plot(df)
        plt.title("Best Validation RMSE overtime per model")
        plt.ylabel("RMSE")
        plt.xlabel("Date")
        plt.ylim(0, y_lim)
        plt.legend(df.columns)
        plt.grid(True)

        plt.savefig(self.config.ROOT_DIR / "outputs" / "figures" / "best_val_score_all_models_overtime.png")
        plt.close()

    def _plot_best_hyperparams_overtime(self):
        hyperparams = self.objs["best_hyperparams_all_models_overtime"]
        mdl_names = list(hyperparams.keys())

        fig, axes = plt.subplots(5, 4, figsize=(14, 12))
        axes = axes.flatten()

        for ax, mdl in zip(axes, mdl_names):
            if mdl in self.non_numeric_hyperparams:
                ax.axis("off")
                continue

            df = hyperparams[mdl]
            ax.plot(df)
            ax.set_title(mdl)
            ax.grid(True)
            ax.legend(df.columns, fontsize=8)

        plt.tight_layout()
        plt.savefig(
            self.config.ROOT_DIR / "outputs" / "figures" / "best_hyperparams_all_models_overtime.png",
            dpi=300
        )
        plt.close()

    def _plot_model_parameters_overtime(self):
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models}
        n_cols = 4
        n_rows = 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True)
        axes = axes.flatten()

        for ax, (model, df) in zip(axes, params.items()):
            ax.plot(df)

            vals = df.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals):
                q1, q99 = np.percentile(vals, [1, 99])
                ax.set_ylim(q1, q99)

            ax.set_title(model)
            ax.grid(True)
            # ax.legend(df.columns, fontsize=8, frameon=False)

        plt.tight_layout()
        plt.savefig(
            self.config.ROOT_DIR / "outputs" / "figures" / "best_parameters_all_models_overtime.png"
        )
        plt.close()

    def _export_selected_features_proportion(self, max_rows:int=25):
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models_without_pca}
        features = next(iter(params.values())).columns

        df = pd.DataFrame(index=features, columns=list(params.keys()))

        for k, v in params.items():
            valid = v.dropna(how="all")
            df[k] = (abs(valid) > 0.001).sum() / valid.shape[0] * 100

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False)
        df.insert(0, "rank", range(1, len(df) + 1))

        dfi.export(df.round(2),
                   self.config.ROOT_DIR / "outputs" / "figures" / "proportion_selected_features.png",
                   max_rows=max_rows)

    def _export_mean_parameters(self, n:int=10, max_rows:int=25):
        params = self.objs["best_params_all_models_overtime"]
        params = {k: v for k, v in params.items() if k in self.linear_models_without_pca}
        features = next(iter(params.values())).columns

        df = pd.DataFrame(index=features, columns=list(params.keys()))

        for k, v in params.items():
            df[k] = v.mean(axis=0)

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False)
        df.insert(0, "rank", range(1, len(df) + 1))

        # df = pd.concat([df.head(n), df.tail(n)])
        dfi.export(df.round(2),
                   self.config.ROOT_DIR / "outputs" / "figures" / "mean_parameters.png",
                   max_rows=max_rows)

    def _plot_oos_rmse_overtime(self, y_lim:float=0.01, window:int=12, freq:str="m") -> None:
        """
        Plot OOS RMSE (here: absolute error) over time for each model.

        Assumes:
        - oos_true is a Tx1 DataFrame
        - oos_predictions[model] is a Tx1 DataFrame
        - indices are already aligned
        """
        y_true = self.objs["oos_true"]  # Tx1 DF
        models = list(self.objs["oos_predictions"].keys())

        rmse_df = pd.DataFrame(index=y_true.index, columns=models, dtype=float)

        for model, y_pred in self.objs["oos_predictions"].items():
            # Absolute error = RMSE with 1 observation per date
            rmse_df[model] = (y_pred.iloc[:, 0] - y_true.iloc[:, 0]).abs()

        # ---- Plot ----
        plt.figure(figsize=(10, 6))
        rolling_rmse = rmse_df.rolling(window, min_periods=1).mean()
        plt.plot(rolling_rmse)
        plt.title(f"Rolling {window}{freq} OOS RMSE overtime per model")
        plt.xlabel("Date")
        plt.ylabel("OOS RMSE")
        plt.ylim(0, y_lim)
        plt.grid(True)
        plt.legend(rmse_df.columns)

        plt.savefig(
            self.config.ROOT_DIR / "outputs" / "figures" / "oos_rmse_all_models_overtime.png"
        )
        plt.close()

        self.rmse_df = rmse_df

    def _export_oos_rmse_table(self):
        mean_rmse = self.rmse_df.mean().sort_values()
        df = mean_rmse.to_frame("mean_rmse")
        df.insert(0, "rank", range(1, len(df) + 1))

        dfi.export(df.round(2),
                   self.config.ROOT_DIR / "outputs" / "figures" / "oos_rmse_all_models.png")

    def _compute_oos_sign_accuracy(self) -> None:
        """
        Compute the proportion of times the model predicts the correct sign
        of the target.

        Returns
        -------
        pd.DataFrame
            Index: model name
            Column: sign_accuracy
        """
        y_true = self.objs["oos_true"].iloc[:, 0]

        sign_acc = {}

        for model, y_pred_df in self.objs["oos_predictions"].items():
            y_pred = y_pred_df.iloc[:, 0]

            # Align and drop missing values
            df = pd.concat([y_true, y_pred], axis=1, join="inner").dropna()
            df.columns = ["y_true", "y_pred"]

            if len(df) == 0:
                sign_acc[model] = np.nan
                continue

            correct_sign = (
                    np.sign(df["y_true"]) == np.sign(df["y_pred"])
            )

            # Exclude zero predictions or targets
            non_zero = (df["y_true"] != 0) & (df["y_pred"] != 0)

            sign_acc[model] = correct_sign[non_zero].mean()

        res = pd.DataFrame.from_dict(
            sign_acc,
            orient="index",
            columns=["sign_accuracy"]
        )
        # Sort by sign accuracy and add a column rank in first position
        res = res.sort_values("sign_accuracy", ascending=False)
        res.insert(0, "rank", range(1, len(res) + 1))
        dfi.export(
            res.round(2),
            self.config.ROOT_DIR / "outputs" / "figures" / "oos_sign_accuracy_all_models.png"
        )

class AnalyticsDynamicAllocation:
    def __init__(self, config: Config, dynamic_alloc: DynamicAllocation, fmp: FactorMimickingPortfolio):
        self.config = config
        self.dynamic_alloc = dynamic_alloc
        self.fmp = fmp

    def get_analytics(self) -> None:
        """Run all dynamic_alloc analytics"""
        returns_df = self._build_aligned_returns_df()
        cum_rets = self._build_cumulative_returns_dict(returns_df)
        self._plot_dynamic_allocation_cum_returns(
            cum_rets,
            save_path=self.config.ROOT_DIR
                      / "outputs"
                      / "figures"
                      / "dynamic_allocation_cum_returns.png"
        )
        perf_table = self._build_performance_table(returns_df, periods_per_year=12)
        self._export_performance_table(
            perf_table,
            save_path=self.config.ROOT_DIR
                      / "outputs"
                      / "figures"
                      / "performance_table.png"
        )

        logger.info("Completed dynamic allocation analytics.")

    def _build_aligned_returns_df(
            self
    ) -> pd.DataFrame:
        """
        Build a DataFrame of aligned non-cumulated returns:
        - Dynamic allocation strategies
        - Benchmark LO EW stocks
        - Benchmark EW FMP
        """

        # Strategy returns (dict[str, Tx1 DF])
        strat_rets = self.dynamic_alloc.net_returns

        # Concatenate strategy returns
        returns_df = pd.concat(
            {k: v.iloc[:, 0] for k, v in strat_rets.items()},
            axis=1
        )

        # ---- Benchmark LO EW stocks ----
        bench = self.fmp.benchmark_returns.copy()

        bench.index = (
                bench.index
                - pd.DateOffset(months=1)
                - pd.DateOffset(days=1)
        )

        returns_df["Bench LO EW stocks"] = bench.iloc[:, 0]

        # ---- Benchmark EW FMP ----
        returns_df["Bench EW FMP"] = (
            self.dynamic_alloc.benchmark_ew_fmp_net_returns.iloc[:, 0]
        )

        return returns_df.dropna(how="any")

    @staticmethod
    def _build_cumulative_returns_dict(
            returns_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Convert a returns DataFrame into a dict of cumulative return DataFrames.
        """

        cum_rets = {}

        for col in returns_df.columns:
            cum_rets[col] = (1 + returns_df[[col]]).cumprod() - 1

        return cum_rets

    @staticmethod
    def _plot_dynamic_allocation_cum_returns(
            cum_rets: dict[str, pd.DataFrame],
            save_path: str | Path
    ) -> None:
        """
        Plot cumulative returns for dynamic allocation strategies and benchmarks.
        """

        Vizu.plot_timeseries_dict(
            data=cum_rets,
            save_path=save_path,
            title="Dynamic Allocation Strategy Cumulative Returns",
            y_label="Cumulative Returns",
            dashed_black_keys=[
                "Bench LO EW stocks",
                "Bench EW FMP"
            ]
        )

    @staticmethod
    def _compute_performance_metrics(
            returns: pd.Series,
            periods_per_year: int = 12
    ) -> pd.Series:
        """
        Compute performance, risk and risk-adjusted metrics from a return series.

        Metrics:
        - Annualized return
        - Annualized volatility
        - Sharpe ratio (rf = 0)
        - Max drawdown
        """

        returns = returns.dropna()

        # ---- Annualized return ----
        ann_ret = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1

        # ---- Annualized volatility ----
        ann_vol = returns.std() * np.sqrt(periods_per_year)

        # ---- Sharpe ratio ----
        sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan

        # ---- Max drawdown ----
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        max_dd = drawdown.min()

        return pd.Series(
            {
                "Ann. Return": ann_ret,
                "Ann. Vol": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": max_dd,
            }
        )

    def _build_performance_table(
            self,
            returns_df: pd.DataFrame,
            periods_per_year: int = 12
    ) -> pd.DataFrame:
        """
        Build a performance table for all strategies/benchmarks.
        """

        perf_table = pd.DataFrame(
            {
                col: self._compute_performance_metrics(
                    returns_df[col],
                    periods_per_year=periods_per_year
                )
                for col in returns_df.columns
            }
        ).T

        return perf_table

    @staticmethod
    def _export_performance_table(
            perf_table: pd.DataFrame,
            save_path: str | Path
    ) -> None:
        """
        Export performance table as an image using dataframe_image.
        """

        perf_table_fmt = perf_table.copy()
        perf_table_fmt[["Ann. Return", "Ann. Vol", "Max Drawdown"]] *= 100
        perf_table_fmt = perf_table_fmt.round(2)
        # Sort by sharpe ratio descending and add a colum rank in first position
        perf_table_fmt = perf_table_fmt.sort_values("Sharpe", ascending=False)
        perf_table_fmt.insert(
            0,
            "Rank",
            range(1, len(perf_table_fmt) + 1)
        )

        dfi.export(
            perf_table_fmt,
            save_path
        )






