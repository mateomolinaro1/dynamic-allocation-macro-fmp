from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.fmp.fmp import FactorMimickingPortfolio
from dynamic_allocation_macro_fmp.forecasting.features_engineering import FeaturesEngineering
from dynamic_allocation_macro_fmp.forecasting.schemes.expanding import ExpandingWindowScheme
from dynamic_allocation_macro_fmp.dynamic_allocation.dynamic_allocation import DynamicAllocation
from dynamic_allocation_macro_fmp.analytics.analytics import AnalyticsFMP, AnalyticsForecasting, \
    AnalyticsDynamicAllocation
import sys
import logging

logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
config = Config()

# Data
data_manager = DataManager(config=config)

# FMP
fmp = FactorMimickingPortfolio(
    config=config,
    data=data_manager,
    market_returns=None,
    rf=None
)
fmp.build_macro_portfolios()

# Analytics FMP
analytics_fmp = AnalyticsFMP(
    config=config,
    fmp=fmp
)
analytics_fmp.get_analytics()

# Feature Engineering
fe = FeaturesEngineering(config=config, data=data_manager)
fe.get_features()
# Expanding Window Scheme
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

# Analytics Forecasting
analytics_forecasting = AnalyticsForecasting(
    config=config,
    exp_window=exp_window
)
analytics_forecasting.get_analytics()

# Dynamic Allocation
dynamic_alloc = DynamicAllocation(
    config=config,
    predictions=exp_window.oos_predictions,
    long_leg_fmp=fmp.positive_betas_fmp_returns,
    short_leg_fmp=fmp.negative_betas_fmp_returns,
    benchmark_ptf=fmp.benchmark_returns
)
dynamic_alloc.run_backtest()

# Analytics Dynamic Allocation
analytics_dynamic_alloc = AnalyticsDynamicAllocation(
    config=config,
    dynamic_alloc=dynamic_alloc,
    fmp=fmp
)
analytics_dynamic_alloc.get_analytics()
