import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults

class Models:

    @staticmethod
    def wls_exponential_decay(X, y, decay=60)->RegressionResults:
        """
        Weighted Least Squares with exponential time decay.

        Parameters
        ----------
        X : ndarray (T, p)
            Design matrix
        y : ndarray (T,)
            Target vector
        decay : float
            Decay parameter (e.g. 60 for 5Y if monthly data)

        Returns
        -------
        RegressionResults
        """
        T = len(y)

        # exponential half-life weights
        age = np.abs((T - 1) - np.arange(T))
        weights = np.exp(-np.log(2) * age / decay)

        X = sm.add_constant(X)

        model = sm.WLS(y, X, weights=weights)
        results = model.fit()

        return results
