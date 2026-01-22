import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.linear_model import Lars
import statsmodels.api as sm

class FeatureSelector(ABC):
    """
    Abstract base class for feature selection methods.
    """

    @abstractmethod
    def _fit(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        pass

    def return_features(self, df: pd.DataFrame, target: str, n_feats: int) -> list:
        pass

class LARSSelector(FeatureSelector):

    def _fit(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        data = df.dropna().copy()

        X = data.drop(columns=[target])
        y = data[target].values

        lars = Lars(n_nonzero_coefs=X.shape[1], fit_intercept=True)
        lars.fit(X.values, y)

        coef_path = lars.coef_path_
        variables = X.columns

        # Nombre de zéros avant activation
        zero_counts = (coef_path == 0).sum(axis=1)

        order = (
            pd.DataFrame({
                "variable": variables,
                "pecking_order": zero_counts
            })
            .sort_values("pecking_order")
            .reset_index(drop=True)
        )

        return order
    
    def return_features(self, df: pd.DataFrame, target: str, n_feats: int) -> list:
        order = self._fit(df, target)
        return list(order.head(n_feats)["variable"])

class TStatSelector(FeatureSelector):

    def _fit(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        data = df.dropna().copy()

        # Création des lags
        data["L1_target"] = data[target].shift(1)
        data["L2_target"] = data[target].shift(2)
        data = data.dropna()

        predictors = [c for c in df.columns if c != target]

        results = []

        for var in predictors:
            X = data[[var, "L1_target", "L2_target"]]
            X = sm.add_constant(X)
            y = data[target]

            model = sm.OLS(y, X).fit()

            tstat = abs(model.tvalues[var])
            results.append((var, tstat))

        order = (
            pd.DataFrame(results, columns=["variable", "tstat"])
            .sort_values("tstat", ascending=False)
            .reset_index(drop=True)
        )

        return order
    
    def return_features(self, df: pd.DataFrame, target: str, n_feats: int) -> list:
        order = self._fit(df, target)
        return list(order.head(n_feats)["variable"])

class CorrelationSelector(FeatureSelector):

    def _fit(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        data = df.dropna().copy()

        corrs = {}

        for col in data.columns:
            if col != target:
                corrs[col] = abs(
                    data[col].corr(data[target], method="pearson")
                )

        order = (
            pd.DataFrame.from_dict(corrs, orient="index", columns=["corr"])
            .sort_values("corr", ascending=False)
            .reset_index()
            .rename(columns={"index": "variable"})
        )

        return order
    
    def return_features(self, df: pd.DataFrame, target: str, n_feats: int) -> list:
        order = self._fit(df, target)
        return list(order.head(n_feats)["variable"])
