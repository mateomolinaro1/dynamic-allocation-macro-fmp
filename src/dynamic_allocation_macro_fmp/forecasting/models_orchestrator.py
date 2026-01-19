from __future__ import annotations
import pandas as pd
import numpy as np
from dynamic_allocation_macro_fmp.data.data import DataManager
from dynamic_allocation_macro_fmp.utils.config import Config, logger
from dynamic_allocation_macro_fmp.forecasting.features_engineering import FeaturesEngineering
from dynamic_allocation_macro_fmp.forecasting.models import Model
from abc import ABC, abstractmethod

class ModelOrchestratorTemplate(ABC):
    @abstractmethod
    def __init__(
            self,
            features_engineering: FeaturesEngineering,
            model: Model,
            framework: str
    ):
        pass

    @abstractmethod
    def orchestrate(self) -> pd.DataFrame:
        """
        Orchestrate the model training and prediction process.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predictions.
        """
        pass

    def static_framework(self) -> pd.DataFrame:
        """
        Implement the static framework where the model is trained once on the entire training set.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predictions.
        """
        # Generate features
        x_train, y_train = self.features_engineering.generate_features(train=True)
        x_test, _ = self.features_engineering.generate_features(train=False)

        # Fit the model
        self.model.fit(x_train, y_train)

        # Predict
        predictions = self.model.predict(x_test)

        return predictions


class ModelOrchestrator:
    def __init__(
            self,
            features_engineering: FeaturesEngineering,
            model: Model,
            framework: str
    ):
        """
        Initialize the ModelsOrchestrator.
        Parameters
        ----------
        framework : str
            The framework to be used for orchestration, i.e. "static" (model just trained once),
            "expanding" (model retrained on an expanding window) or "rolling" (model retrained on a rolling window).
        features_engineering : FeaturesEngineering
            Instance of the FeaturesEngineering class for feature generation.
        model : Model
            Instance of the Model class for training and prediction.
        """
        self.framework = framework
        self.features_engineering = features_engineering
        self.model = model