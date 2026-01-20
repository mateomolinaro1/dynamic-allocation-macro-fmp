from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class EstimationScheme(ABC):
    """
    Abstract base class for estimation schemes
    (static, expanding, rolling, etc.).
    """

    def run(
        self,
        model,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run the estimation scheme.

        Parameters
        ----------
        model : Model
            A model implementing fit() and predict()
        X : pd.DataFrame
            Feature matrix indexed by time
        y : pd.DataFrame or pd.Series
            Target indexed by time

        Returns
        -------
        dict
            Standardized results (predictions, diagnostics, etc.)
        """
        self._validate_inputs(X, y)

        prediction_index = self._get_prediction_index(X, y)

        predictions = []

        for t in prediction_index:
            train_idx = self._get_training_index(t, X, y)

            if self._should_refit(t):
                model.fit(X.loc[train_idx], y.loc[train_idx])
                self._on_fit(model, t)

            y_hat = model.predict(X.loc[[t]])
            self._on_predict(y_hat, t)

            predictions.append(y_hat)

        predictions = pd.concat(predictions)

        return {
            "predictions": predictions
        }

    # ---------- required hooks ----------

    @abstractmethod
    def _get_prediction_index(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> pd.Index:
        pass

    @abstractmethod
    def _get_training_index(
        self,
        t,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> pd.Index:
        pass

    @abstractmethod
    def _should_refit(self, t) -> bool:
        pass

    # ---------- optional hooks ----------

    def _on_fit(self, model, t) -> None:
        pass

    def _on_predict(self, prediction: pd.DataFrame, t) -> None:
        pass

    # ---------- validation ----------

    def _validate_inputs(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> None:
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the same index.")

        if not X.index.is_monotonic_increasing:
            raise ValueError("Time index must be increasing.")