import pandas as pd

from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.fmp.fmp import FactorMimickingPortfolio
from dynamic_allocation_macro_fmp.forecasting.features_engineering import FeaturesEngineering
from dynamic_allocation_macro_fmp.utils.vizu import Vizu
from dynamic_allocation_macro_fmp.forecasting.schemes.expanding import ExpandingWindowScheme
from dynamic_allocation_macro_fmp.forecasting.models import Lasso, WLSExponentialDecay
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
# fmp.build_macro_portfolios()
#
# fmp_returns = pd.concat([fmp.positive_betas_fmp_returns, fmp.negative_betas_fmp_returns, fmp.benchmark_returns], axis=1)
# from dynamic_allocation_macro_fmp.utils.vizu import Vizu
# Vizu.plot_time_series(
#     data=fmp_returns.loc["2010-01-02":,:].cumsum(),
#     title="LO EW and SH EW FMP Returns over Time",
#     ylabel="Cumulative Returns",
#     xlabel="Date",
#     save_path=r"C:\Users\mateo\Code\ENSAE\MacroML\DynamicAllocationMacroFMP\outputs\figures\fmp_returns.png",
#     show=False,
#     block=False
# )
#
# # LO
# Vizu.plot_time_series(
#     data=(1+fmp.positive_betas_fmp_returns.loc["2010-01-02":,:]).cumprod()-1,
#     title="LO EW FMP Returns over Time",
#     ylabel="Cumulative Returns",
#     xlabel="Date",
#     save_path=r"C:\Users\mateo\Code\ENSAE\MacroML\DynamicAllocationMacroFMP\outputs\figures\positive_fmp_returns.png",
#     show=False,
#     block=False
# )
# # SO
# Vizu.plot_time_series(
#     data=fmp.negative_betas_fmp_returns.loc["2010-01-02":,:].cumsum(),
#     title="SO EW FMP Returns over Time",
#     ylabel="Cumulative Returns",
#     xlabel="Date",
#     save_path=r"C:\Users\mateo\Code\ENSAE\MacroML\DynamicAllocationMacroFMP\outputs\figures\negative_fmp_returns.png",
#     show=False,
#     block=False
# )

# from dynamic_allocation_macro_fmp.forecasting.features_selection import PCAFactorExtractor
# pca_extractor = PCAFactorExtractor(
#     n_factors=5
# )
# factors = pca_extractor.fit_transform(fe.x)

# Vizu.plot_time_series(
#     data=fmp.betas_macro,
#     title="Stock-level Macro Betas over Time",
# )


# fmp.betas_macro.mean(axis=1)
# fmp.betas_macro.mean(axis=1).mean()
#
# fmp.betas_mkt.mean(axis=1)
# fmp.betas_mkt.mean(axis=1).mean()

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

# models = {
#     "wls": WLSExponentialDecay,
#     "lasso": Lasso
# }
# hyperparams_grid = {
#     "wls": {
#         "decay": [0.9, 0.99]
#     },
#     "lasso": {
#         "alpha": [0.01, 10.0]
#     }
# }
exp_window.run(
    models=config.models,
    hyperparams_grid=config.hyperparams_grid
)
print(exp_window.best_score_all_models_overtime.mean())

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

