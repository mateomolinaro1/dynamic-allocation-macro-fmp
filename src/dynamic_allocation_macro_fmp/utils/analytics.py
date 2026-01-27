from pathlib import Path
import pandas as pd
import numpy as np
import dataframe_image as dfi
import matplotlib
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
    def __init__(self, config: Config, exp_window: Type[ExpandingWindowScheme]):
        self.config = config
        self.exp_window = exp_window
        self.objs = None

    def get_analytics(self) -> None:
        """Run all forecasting analytics"""
        self._load_objects()
        self._export_features_tables()
        self._plot_best_score_overtime()
        self._plot_best_hyperparams_overtime()
        self._plot_model_parameters_overtime()
        self._export_selected_features_proportion()
        self._export_mean_parameters()
        self._plot_oos_rmse_overtime()
        self._export_oos_rmse_table()
        logger.info("Completed forecasting analytics.")

    def _load_objects(self):
        self.objs = {k: v for k, v in vars(self.exp_window).items()}

    def _export_features_tables(self):
        dfi.export(
            self.objs["x"],
            self.config.ROOT_DIR / "outputs" / "figures" / "features_df.png",
            max_rows=10
        )
        dfi.export(
            self.objs["x"],
            self.config.ROOT_DIR / "outputs" / "figures" / "features_df_short.png",
            max_rows=10,
            max_cols=10
        )

    def _plot_best_score_overtime(self):
        df = self.objs["best_score_all_models_overtime"]

        plt.figure(figsize=(10, 6))
        plt.plot(df)
        plt.title("Best validation score overtime per model")
        plt.ylabel("RMSE")
        plt.xlabel("Date")
        plt.ylim(0, 2)
        plt.legend(df.columns)
        plt.grid(True)

        plt.savefig(self.config.ROOT_DIR / "outputs" / "figures" / "best_score_all_models_overtime.png")
        plt.close()

    def _plot_best_hyperparams_overtime(self):
        hyperparams = self.objs["best_hyperparams_all_models_overtime"]
        mdl_names = list(hyperparams.keys())

        fig, axes = plt.subplots(5, 4, figsize=(14, 12))
        axes = axes.flatten()

        for ax, mdl in zip(axes, mdl_names):
            if mdl in ["neural_net", "neural_net_pca", "ols", "ols_pca"]:
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
        n_cols = 4

        fig, axes = plt.subplots(2, n_cols, figsize=(20, 12), sharex=True)
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
            ax.legend(df.columns, fontsize=8, frameon=False)

        plt.tight_layout()
        plt.savefig(
            self.config.ROOT_DIR / "outputs" / "figures" / "best_parameters_all_models_overtime.png"
        )
        plt.close()

    def _export_selected_features_proportion(self):
        params = self.objs["best_params_all_models_overtime"]
        features = next(iter(params.values())).columns

        df = pd.DataFrame(index=features, columns=params.keys())

        for k, v in params.items():
            valid = v.dropna(how="all")
            df[k] = (abs(valid) > 0.01).sum() / valid.shape[0] * 100

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False)
        df.insert(0, "rank", range(1, len(df) + 1))

        dfi.export(df.round(2),
                   self.config.ROOT_DIR / "outputs" / "figures" / "proportion_selected_features.png")

    def _export_mean_parameters(self):
        params = self.objs["best_params_all_models_overtime"]
        features = next(iter(params.values())).columns

        df = pd.DataFrame(index=features, columns=params.keys())

        for k, v in params.items():
            df[k] = v.mean(axis=0)

        df["mean_models"] = df.mean(axis=1)
        df = df.sort_values("mean_models", ascending=False)
        df.insert(0, "rank", range(1, len(df) + 1))

        dfi.export(df.round(2),
                   self.config.ROOT_DIR / "outputs" / "figures" / "mean_parameters.png")

    def _plot_oos_rmse_overtime(self):
        dates = sorted(self.objs["OOS_TRUE"].keys())
        models = self.objs["OOS_PRED"].keys()

        rmse_df = pd.DataFrame(index=dates, columns=models)

        for model, preds in self.objs["OOS_PRED"].items():
            for date, y_pred in preds.items():
                if date not in self.objs["OOS_TRUE"]:
                    continue
                y_true = self.objs["OOS_TRUE"][date]
                df = pd.concat([y_true, y_pred], axis=1).dropna()
                if len(df):
                    rmse_df.loc[date, model] = np.sqrt(((df.iloc[:, 0] - df.iloc[:, 1]) ** 2).mean())

        plt.figure(figsize=(10, 6))
        plt.plot(rmse_df)
        plt.ylim(0, 2)
        plt.title("OOS RMSE overtime per model")
        plt.grid(True)
        plt.legend(rmse_df.columns)

        plt.savefig(self.config.ROOT_DIR / "outputs" / "figures" / "oos_rmse_all_models_overtime.png")
        plt.close()

        self.rmse_df = rmse_df

    def _export_oos_rmse_table(self):
        mean_rmse = self.rmse_df.mean().sort_values()
        df = mean_rmse.to_frame("mean_rmse")
        df.insert(0, "rank", range(1, len(df) + 1))

        dfi.export(df.round(2),
                   self.config.ROOT_DIR / "outputs" / "figures" / "oos_rmse_all_models.png")
