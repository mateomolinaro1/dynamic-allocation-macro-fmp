import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from abc import ABC, abstractmethod
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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

    def __init__(self, **kwargs):
        self.hyperparams = dict(kwargs)
        self.decay = self.hyperparams.get("decay", 60)
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
    def __init__(self,**kwargs):
        """
        Parameters
        ----------
        alpha : float
            Regularization strength.
        """
        self.hyperparams = dict(kwargs)
        self.alpha = self.hyperparams.get("alpha", 1.0)
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

"""Série de modèles usuels provenant de de scikit-learn
sur le modèle abstrait Model"""

class OLS(Model):

    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y.values.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        y_pred = self.model.predict(x)
        return pd.DataFrame(y_pred, index=x.index, columns=["y_hat"])

class RidgeModel(Model):

    def __init__(self, **kwargs):
        self.model = Ridge(**kwargs)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y.values.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )

class ElasticNetModel(Model):

    def __init__(self, **kwargs):
        self.model = ElasticNet(
            **kwargs
        )

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y.values.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )

class RandomForestModel(Model):

    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            **kwargs
        )

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y.values.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )

class GradientBoostingModel(Model):

    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(
            **kwargs)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y.values.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )

class SVRModel(Model):

    def __init__(self, **kwargs):
        self.model = SVR(**kwargs)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y.values.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )

class NeuralNetModel(Model):

    def __init__(
        self,
        **kwargs):

        self.model = MLPRegressor(
            **kwargs)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(x, y.values.ravel())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )

class DynamicFactorModel(Model):

    def __init__(
        self,
        target_name,
        standardize: bool = True,
        **kwargs
    ):
        """
        kwargs are passed directly to statsmodels DynamicFactor
        (e.g. k_factors, factor_order, error_order, trend, ...)
        """
        self.standardize = standardize
        self.target_name = target_name
        self.df_kwargs = kwargs

        self.model = None
        self.results = None
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:

        data = pd.concat([y, x], axis=1)
        self.columns_ = data.columns

        if self.standardize:
            self.mean_ = data.mean()
            self.std_ = data.std()
            data = (data - self.mean_) / self.std_

        self.model = DynamicFactor(
            data,
            **self.df_kwargs
        )

        self.results = self.model.fit(
            disp=False
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:

        h = len(x)

        forecast = self.results.forecast(h)

        if self.standardize:
            forecast = forecast * self.std_ + self.mean_

        return pd.DataFrame(
            forecast[self.target_name].values,
            index=x.index,
            columns=["y_hat"]
        )

class LightGBMModel(Model):

    def __init__(self, **kwargs):

        self.model = LGBMRegressor(**kwargs)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(
            x,
            y.values.ravel()
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )

class XGBoostModel(Model):

    def __init__(self, **kwargs):
        """
        kwargs are passed directly to XGBRegressor
        """
        self.model = XGBRegressor(**kwargs)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(
            x,
            y.values.ravel()
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.model.predict(x),
            index=x.index,
            columns=["y_hat"]
        )
