import pandas as pd
from .base import EstimationScheme


class ExpandingScheme(EstimationScheme):
    """
    Expanding window estimation.
    """

    def __init__(self, start):
        """
        Parameters
        ----------
        start : timestamp
            First date used for training
        """
        self.start = start

    def _get_prediction_index(self, X, y) -> pd.Index:
        return X.index[X.index > self.start]

    def _get_training_index(self, t, X, y) -> pd.Index:
        return X.index[(X.index >= self.start) & (X.index < t)]

    def _should_refit(self, t) -> bool:
        return True
