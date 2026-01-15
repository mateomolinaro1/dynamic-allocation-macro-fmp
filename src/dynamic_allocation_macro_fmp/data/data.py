import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract class to define the interface for the data source"""

    @abstractmethod
    def _fetch_data(self):
        """Retrieve gross data from the data source"""
        pass

class CSVDataSource(DataSource):
    """Class to fetch data from a CSV file"""
    def __init__(
            self,
            file_path:str|None=None,
            index_col:int=0,
            date_column=None
    ):
        self.file_path = file_path
        self.index_col = index_col
        self.date_column = date_column
        self.data = None
        self._fetch_data()

    def _fetch_data(self):
        """Retrieve gross data from csv file"""
        data = pd.read_csv(self.file_path, index_col=self.index_col)
        if self.date_column:
            data.index = pd.to_datetime(data.index)
        if self.data is None:
            self.data = data