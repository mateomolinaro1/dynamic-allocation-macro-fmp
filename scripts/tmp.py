from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.fmp.fmp import FactorMimickingPortfolio
from dynamic_allocation_macro_fmp.forecasting.features_engineering import FeaturesEngineering
from dynamic_allocation_macro_fmp.utils.vizu import Vizu
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
fmp.fit_wls()
b=fmp._get_bayesian_betas()

Vizu.plot_time_series(
    data=fmp.betas_macro,
    title="Stock-level Macro Betas over Time",
)


# fmp.betas_macro.mean(axis=1)
# fmp.betas_macro.mean(axis=1).mean()
#
# fmp.betas_mkt.mean(axis=1)
# fmp.betas_mkt.mean(axis=1).mean()

fe = FeaturesEngineering(config=config, data=data_manager)
fe.get_features()


bayesian_betas = (
                prior_betas.values[:, None] * s.values[:, None]
                + fmp.betas_macro * (1 - s.values)[:, None]
        )
