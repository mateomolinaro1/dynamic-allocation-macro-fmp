import pandas as pd
from .base import EstimationScheme


class RollingScheme(EstimationScheme):
    """
    Rolling window estimation.
    """

    def __init__(self, window: int):
        """
        Parameters
        ----------
        window : int
            Number of observations in the rolling window
        """
        self.window = window

    def _get_prediction_index(self, X, y) -> pd.Index:
        return X.index[self.window:]

    def _get_training_index(self, t, X, y) -> pd.Index:
        t_loc = X.index.get_loc(t)
        start = t_loc - self.window
        return X.index[start:t_loc]

    def _should_refit(self, t) -> bool:
        return True
