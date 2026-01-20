import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple
from statsmodels.regression.linear_model import RegressionResults
from abc import ABC, abstractmethod
from sklearn.linear_model import Lasso as SklearnLasso

# class Models:
#
#     @staticmethod
#     def wls_exponential_decay(
#             X: pd.DataFrame,
#             y: pd.DataFrame,
#             decay: float|int =60
#     ) -> dict:
#         """
#         Weighted Least Squares with exponential time decay.
#
#         Parameters
#         ----------
#         X : pd.DataFrame
#             Design matrix
#         y : pd.DataFrame
#             Target vector
#         decay : float|int
#             Decay parameter (e.g. 60 for 5Y if monthly data)
#
#         Returns
#         -------
#         RegressionResults
#         """
#         T = len(y)
#
#         # exponential half-life weights
#         age = np.abs((T - 1) - np.arange(T))
#         weights = np.exp(-np.log(2) * age / decay)
#
#         X = sm.add_constant(X, has_constant="add")
#
#         model = sm.WLS(y, X, weights=weights)
#         results = model.fit()
#         # Get HAC standard errors (Newey-West)
#         results_hac = results.get_robustcov_results(
#             cov_type='HAC',
#             maxlags=None # Automatic lag selection
#         )
#         hac_bse = pd.Series(
#             results_hac.bse,
#             index=results.model.exog_names,
#             name="HAC SE"
#         )
#         hac_pvalues = pd.Series(
#             results_hac.pvalues,
#             index=results.model.exog_names,
#             name="HAC pvalues"
#         )
#
#         return {
#             "results":results,
#             "results_hac":results_hac,
#             "hac_bse":hac_bse,
#             "hac_pvalues":hac_pvalues
#         }

class Model(ABC):
    """
    Abstract base class for forecasting models.
    """
    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        x : pd.DataFrame
            Input features for training.
        y : pd.DataFrame
            Target variable for training.
        """
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the fitted model.

        Parameters
        ----------
        x : pd.DataFrame
            Input features for prediction.

        Returns
        -------
        pd.DataFrame
            Predictions from the model.
        """
        pass

class WLSExponentialDecay(Model):

    def __init__(self, decay: float | int = 60):
        self.decay = decay
        self.results = None
        self.results_hac = None
        self.hac_bse = None
        self.hac_pvalues = None
        self.exog_names = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        T = len(y)

        age = np.abs((T - 1) - np.arange(T))
        weights = np.exp(-np.log(2) * age / self.decay)

        X = sm.add_constant(x, has_constant="add")
        self.exog_names = X.columns

        model = sm.WLS(y, X, weights=weights)
        results = model.fit()

        results_hac = results.get_robustcov_results(
            cov_type="HAC",
            maxlags=None
        )

        self.results = results
        self.results_hac = results_hac
        self.hac_bse = pd.Series(
            results_hac.bse,
            index=self.exog_names,
            name="HAC SE"
        )
        self.hac_pvalues = pd.Series(
            results_hac.pvalues,
            index=self.exog_names,
            name="HAC pvalues"
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Model is not fitted yet.")

        X = sm.add_constant(x, has_constant="add")
        y_hat = self.results.predict(X)

        return pd.DataFrame(y_hat, index=x.index, columns=["y_hat"])

class Lasso(Model):
    """
    Lasso regression model.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Regularization strength.
        """
        self.alpha = alpha
        self.model = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the Lasso model to the data.

        Parameters
        ----------
        x : pd.DataFrame
            Input features for training.
        y : pd.DataFrame
            Target variable for training.
        """
        self.model = SklearnLasso(alpha=self.alpha)
        self.model.fit(x, y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the fitted Lasso model.

        Parameters
        ----------
        x : pd.DataFrame
            Input features for prediction.

        Returns
        -------
        pd.DataFrame
            Predictions from the model.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        predictions = self.model.predict(x)
        return pd.DataFrame(predictions, index=x.index, columns=["y_hat"])
