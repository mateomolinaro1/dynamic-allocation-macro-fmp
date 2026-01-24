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
fmp.get_betas()

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



# bayesian_betas = (
#                 prior_betas.values[:, None] * s.values[:, None]
#                 + fmp.betas_macro * (1 - s.values)[:, None]
#         )

