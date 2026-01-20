import pandas as pd
from .base import EstimationScheme


class StaticScheme(EstimationScheme):
    """
    Fit once on an initial sample, then predict for all future dates.
    """

    def __init__(self, train_end):
        """
        Parameters
        ----------
        train_end : timestamp
            Last date used for training
        """
        self.train_end = train_end
        self._fitted = False

    def _get_prediction_index(self, X, y) -> pd.Index:
        return X.index[X.index > self.train_end]

    def _get_training_index(self, t, X, y) -> pd.Index:
        return X.index[X.index <= self.train_end]

    def _should_refit(self, t) -> bool:
        if not self._fitted:
            self._fitted = True
            return True
        return False
