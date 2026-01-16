from __future__ import annotations
import pandas as pd
import numpy as np
from dynamic_allocation_macro_fmp.data.data import DataManager

class FeaturesEngineering:
    def __init__(
            self,
            data: DataManager
    ):
        self.data = data

    @staticmethod
    def _preprocess_var(var: pd.DataFrame, code_transfo: int | float):
        var = var.astype(float)

        if code_transfo == 6.0:
            # log second difference
            return (
                np.log(var)
                .diff()
                .diff()
            )

        raise ValueError(f"Unknown transformation code: {code_transfo}")
