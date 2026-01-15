from abc import ABC, abstractmethod
from typing import Union, Tuple
import pandas as pd
import numpy as np
import src.alpha_in_analysts.backtester.utilities as utilities

class Strategy(ABC):
    """Abstract class to define the interface for the strategy"""
    def __init__(self, prices: pd.DataFrame, returns:pd.DataFrame):
        self.prices = prices
        self.returns = returns
        self.signals_values = None
        self.signals = None

    @abstractmethod
    def compute_signals_values(self):
        """Compute the signals values for the strategy"""
        pass

    @abstractmethod
    def compute_signals(self, *args, **kwargs):
        """Compute the signals for the strategy"""
        pass

class BuyAndHold(Strategy):
    """Class to implement the buy and hold strategy"""
    def compute_signals_values(self):
        """Compute the signals for the buy and hold strategy"""
        self.signals_values = (~np.isnan(self.returns)).astype(int)
        return self.signals_values

    def compute_signals(self):
        """Compute the signals for the buy and hold strategy"""
        self.signals = (~np.isnan(self.returns)).astype(int)
        return self.signals

class CrossSectionalPercentiles(Strategy):
    """ Class to implement the cross-sectional percentiles strategy"""
    def __init__(self,
                 prices:pd.DataFrame,
                 returns:pd.DataFrame,
                 signal_function,
                 signal_function_inputs:dict=None,
                 percentiles_winsorization: Tuple[int, int] = (1, 99)
                 ):
        """
        Initializes the CrossSectionalPercentiles strategy.

        Parameters:
        - prices: pd.DataFrame, price data for the assets.
        - returns: pd.DataFrame, return data for the assets.
        - signal_function: callable, the function to compute the signals.
        - signal_function_inputs: dict, arguments to be passed to signal_function. Keywords of the dict must match
        argument names of the function.
        - percentiles_portfolios: Tuple[int, int], percentiles to apply to signal values.
        - percentiles_winsorization: Tuple[int, int], percentiles to apply to signal values for winsorization.
        - industry_segmentation: Union[None, pd.DataFrame], optional, industry segmentation data for the assets. This df must have the same
        shape, indices and columns as the returns dataframe. It must contain the industry segmentation for each asset. If this dataframe
        is provided, the signals will be computed within each industry segment.
        """
        super().__init__(prices, returns)
        if not callable(signal_function):
            raise ValueError("signal_function must be a callable function.")
        self.signal_function = signal_function
        self.signal_function_inputs = signal_function_inputs if signal_function_inputs is not None else {}
        if not isinstance(percentiles_winsorization, tuple) and len(percentiles_winsorization) == 2 and all(
                isinstance(pct, int) for pct in percentiles_winsorization):
            raise ValueError("percentiles must be a tuple of two int. (1,99) for example.")
        self.percentiles_winsorization = percentiles_winsorization

    def compute_signals_values(self):
        """ Compute the signals for the cross-sectional percentiles strategy"""
        # Compute signal values
        self.signals_values = self.signal_function(**self.signal_function_inputs)
        # Clean the signal values
        self.signals_values = utilities.clean_dataframe(self.signals_values)
        # Compute zscores
        self.signals_values = utilities.compute_zscores(self.signals_values, axis=1)
        # Winsorize the signal values
        self.signals_values = utilities.winsorize_dataframe(df=self.signals_values, percentiles=self.percentiles_winsorization, axis=1)
        return self.signals_values

    def compute_signals(self,
                        percentiles_portfolios:Tuple[int, int] = (10,90),
                        industry_segmentation:Union[None, pd.DataFrame] = None):
        # Input checks
        if not isinstance(percentiles_portfolios, tuple) and len(percentiles_portfolios) == 2 and all(isinstance(pct, int) for pct in percentiles_portfolios):
            raise ValueError("percentiles must be a tuple of two int. (5,95) for example.")
        if not (isinstance(industry_segmentation, pd.DataFrame) or industry_segmentation is None):
            raise ValueError("industry_segmentation must be a pandas dataframe or None.")
        if isinstance(industry_segmentation, pd.DataFrame):
            if industry_segmentation.shape != self.returns.shape:
                raise ValueError("industry_segmentation must have the same shape as the returns dataframe.")
            if not all(industry_segmentation.index == self.returns.index):
                raise ValueError("industry_segmentation must have the same indices as the returns dataframe.")
            if not all(industry_segmentation.columns == self.returns.columns):
                raise ValueError("industry_segmentation must have the same columns as the returns dataframe.")
        industry_segmentation = industry_segmentation

        if industry_segmentation is not None:
            industries = pd.unique(industry_segmentation.values.ravel())
            all_signals = pd.DataFrame(data=0.0, index=self.signals_values.index, columns=self.signals_values.columns)
            for industry in industries:
                mask = industry_segmentation == industry
                signals_industry = utilities.compute_percentiles(self.signals_values[mask], percentiles_portfolios)['signals']
                # Grouping the signals of all industries
                all_signals = all_signals + signals_industry.fillna(0.0)
            self.signals = all_signals

        else:
            self.signals = utilities.compute_percentiles(self.signals_values, percentiles_portfolios)['signals']
        return self.signals


