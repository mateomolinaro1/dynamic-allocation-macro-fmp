import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple
from statsmodels.regression.linear_model import RegressionResults

class Models:

    @staticmethod
    def wls_exponential_decay(
            X: pd.DataFrame,
            y: pd.DataFrame,
            decay: float|int =60
    ) -> dict:
        """
        Weighted Least Squares with exponential time decay.

        Parameters
        ----------
        X : pd.DataFrame
            Design matrix
        y : pd.DataFrame
            Target vector
        decay : float|int
            Decay parameter (e.g. 60 for 5Y if monthly data)

        Returns
        -------
        RegressionResults
        """
        T = len(y)

        # exponential half-life weights
        age = np.abs((T - 1) - np.arange(T))
        weights = np.exp(-np.log(2) * age / decay)

        X = sm.add_constant(X, has_constant="add")

        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
        # Get HAC standard errors (Newey-West)
        results_hac = results.get_robustcov_results(
            cov_type='HAC',
            maxlags=None # Automatic lag selection
        )
        hac_bse = pd.Series(
            results_hac.bse,
            index=results.model.exog_names,
            name="HAC SE"
        )
        hac_pvalues = pd.Series(
            results_hac.pvalues,
            index=results.model.exog_names,
            name="HAC pvalues"
        )

        return {
            "results":results,
            "results_hac":results_hac,
            "hac_bse":hac_bse,
            "hac_pvalues":hac_pvalues
        }
