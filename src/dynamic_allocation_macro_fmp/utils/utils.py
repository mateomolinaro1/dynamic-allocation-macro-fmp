from pathlib import Path
import pickle
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Utils:
    @staticmethod
    def save_object_attributes(path: str | Path, obj) -> None:
        """
        Save each attribute of an object depending on its type:
        - pd.DataFrame -> parquet
        - dict -> pickle

        Parameters
        ----------
        path : str | Path
            Base directory where attributes will be saved.
        obj : Any
            Object whose attributes will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for attr_name, attr_value in vars(obj).items():

            # Skip empty attributes
            if attr_value is None:
                continue

            attr_path = path / attr_name

            # -------------------------
            # DataFrame → parquet OR pickle fallback
            # -------------------------
            if isinstance(attr_value, pd.DataFrame):
                try:
                    attr_value.to_parquet(attr_path.with_suffix(".parquet"))
                except Exception as e:
                    logger.warning(
                        f"Parquet failed for {attr_name}, fallback to pickle: {e}"
                    )
                    with open(attr_path.with_suffix(".pkl"), "wb") as f:
                        pickle.dump(attr_value, f)

            # -------------------------
            # Dict → pickle
            # -------------------------
            elif isinstance(attr_value, dict):
                with open(attr_path.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(attr_value, f)

            else:
                # silently skip unsupported types
                continue
