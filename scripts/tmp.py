import pandas as pd
from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.fmp.fmp import FactorMimickingPortfolio
from dynamic_allocation_macro_fmp.forecasting.features_engineering import FeaturesEngineering
from dynamic_allocation_macro_fmp.utils.s3_utils import s3Utils
from dynamic_allocation_macro_fmp.utils.vizu import Vizu
from dynamic_allocation_macro_fmp.forecasting.schemes.expanding import ExpandingWindowScheme
from dynamic_allocation_macro_fmp.forecasting.models import Lasso, WLSExponentialDecay
from dynamic_allocation_macro_fmp.dynamic_allocation.dynamic_allocation import DynamicAllocation
from dynamic_allocation_macro_fmp.utils.utils import Utils
from dynamic_allocation_macro_fmp.utils.analytics import AnalyticsFMP
import sys
import logging

logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
config = Config()
data_manager = DataManager(config=config)
fmp = FactorMimickingPortfolio(
    config=config,
    data=data_manager,
    market_returns=None,
    rf=None
)
fmp.build_macro_portfolios()
analytics_fmp = AnalyticsFMP(
    config=config,
    fmp=fmp
)
analytics_fmp.get_analytics()


fe = FeaturesEngineering(config=config, data=data_manager)
fe.get_features()

exp_window = ExpandingWindowScheme(
    config=config,
    x=fe.x,
    y=fe.y,
    forecast_horizon=config.forecast_horizon,
    validation_window=config.validation_window,
    min_nb_periods_required=config.min_nb_periods_required
)
exp_window.run(
    models=config.models,
    hyperparams_grid=config.hyperparams_grid
)

oos_predictions_d = s3Utils.pull_file_from_s3(
    path="s3://dynamic-allocation-macro-fmp/outputs/forecasting/oos_predictions.pkl",
    file_type="pkl"
)

dynamic_alloc = DynamicAllocation(
    config=config,
    predictions=oos_predictions_d,
    long_leg_fmp=fmp.positive_betas_fmp_returns,
    short_leg_fmp=fmp.negative_betas_fmp_returns,
    benchmark_ptf=fmp.benchmark_returns
)
dynamic_alloc.run_backtest()

cum_rets = Utils.compute_cumulative_returns_for_dict_of_df(dynamic_alloc.net_returns)
bench_cum_ret = fmp.benchmark_returns
# Dates alignment for bench
bench_idx_aligned = (
        bench_cum_ret.index
        - pd.DateOffset(months=1)
        - pd.DateOffset(days=1)
)
bench_cum_ret.index = bench_idx_aligned
bench_cum_ret_aligned = pd.merge(
    left=bench_cum_ret,
    right=cum_rets[list(cum_rets.keys())[0]],
    left_index=True,
    right_index=True,
    how="inner"
)
bench_cum_ret_aligned = bench_cum_ret_aligned.drop(columns=cum_rets[list(cum_rets.keys())[0]].columns)
bench_cum_ret_aligned = (1+bench_cum_ret_aligned).cumprod()-1

cum_rets["Bench LO EW stocks"] = bench_cum_ret_aligned
cum_rets["Bench EW FMP"] = (1+dynamic_alloc.benchmark_ew_fmp_net_returns).cumprod()-1

Vizu.plot_timeseries_dict(
    data=cum_rets,
    save_path=r"C:\Users\mateo\Code\ENSAE\MacroML\DynamicAllocationMacroFMP\outputs\figures\dynamic_allocation_cum_returns.png",
    title="Dynamic Allocation Strategy Cumulative Returns",
    y_label="Cumulative Returns",
    dashed_black_keys=["Bench LO EW stocks", "Bench EW FMP"]
)

# from pathlib import Path
# import pickle
# path=r"C:\Users\mateo\Code\ENSAE\MacroML\DynamicAllocationMacroFMP\outputs\results\forecasting"
# path = Path(path)
# for attr_name, attr_value in vars(exp_window).items():
#
#     # Skip empty attributes
#     if attr_value is None:
#         continue
#
#     attr_path = path / attr_name
#
#     # -------------------------
#     # DataFrame → parquet OR pickle fallback
#     # -------------------------
#     if isinstance(attr_value, pd.DataFrame):
#         try:
#             attr_value.to_parquet(attr_path.with_suffix(".parquet"))
#         except Exception as e:
#             print(
#                 f"Parquet failed for {attr_name}, fallback to pickle: {e}"
#             )
#             with open(attr_path.with_suffix(".pkl"), "wb") as f:
#                 pickle.dump(attr_value, f)
#
#     # -------------------------
#     # Dict → pickle
#     # -------------------------
#     elif isinstance(attr_value, dict):
#         with open(attr_path.with_suffix(".pkl"), "wb") as f:
#             pickle.dump(attr_value, f)
#
#     else:
#         # silently skip unsupported types
#         continue



# bayesian_betas = (
#                 prior_betas.values[:, None] * s.values[:, None]
#                 + fmp.betas_macro * (1 - s.values)[:, None]
#         )

