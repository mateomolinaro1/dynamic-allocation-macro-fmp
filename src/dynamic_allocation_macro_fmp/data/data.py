from __future__ import annotations
import pandas as pd
from typing import Dict
from dynamic_allocation_macro_fmp.utils.s3_utils import s3Utils
from dynamic_allocation_macro_fmp.utils.config import Config


class DataManager:
    def __init__(self, config: Config):
        self.config = config

        self.fred_data: pd.DataFrame | None = None
        self.returns_data: pd.DataFrame | None = None
        self.code_transfo: Dict[str, int|float] | None = None

        self.load()

    # ---------- Public API ---------- #
    def load(self) -> None:
        raw = self._fetch_from_s3()
        self._process(raw)

    # ---------- Internal helpers ---------- #
    def _fetch_from_s3(self) -> dict[str, pd.DataFrame]:
        return {
            "fred": s3Utils.pull_file_from_s3(
                path=self.config.fred_path,
                file_type=self.config.macro_ext,
                index_col=0,
            ),
            "prices": s3Utils.pull_file_from_s3(
                path=self.config.prices_path,
                file_type=self.config.prices_ext,
            ),
        }

    def _process(self, raw: dict[str, pd.DataFrame]) -> None:
        self.code_transfo = self._extract_fred_transform_codes(raw["fred"])
        self.fred_data = self._clean_fred(raw["fred"])
        self.returns_data = raw["prices"]

    # ---------- FRED-specific logic ---------- #
    @staticmethod
    def _extract_fred_transform_codes(fred: pd.DataFrame) -> dict[str, float]:
        res = dict(
            zip(fred.columns, fred.loc["Transform:", :])
        )
        res = dict(sorted(res.items()))
        return res

    @staticmethod
    def _clean_fred(fred: pd.DataFrame) -> pd.DataFrame:
        fred = fred.iloc[1:].copy()
        fred.index = pd.to_datetime(fred.index, format="%m/%d/%Y")
        fred = fred.sort_index(axis=1)
        return fred
