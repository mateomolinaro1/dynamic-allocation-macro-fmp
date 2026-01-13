import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from botocore.client import BaseClient
import io
import pickle
from typing import Union, Dict
import boto3
import os

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract class to define the interface for the data source"""
    @abstractmethod
    def fetch_data(self):
        """Retrieve gross data from the data source"""
        pass


class ExcelDataSource(DataSource):
    """Class to fetch data from an Excel file"""
    def __init__(self,
                 file_path:Union[str, Path],
                 sheet_name:str="data",
                 index_col:int=0):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.index_col = index_col

    def fetch_data(self):
        """Retrieve gross data from excel file"""
        data = pd.read_excel(self.file_path, sheet_name=self.sheet_name, index_col=self.index_col)
        # Convert excel date to datetime
        # data.index = pd.to_datetime("1899-12-30") + pd.to_timedelta(data.index, unit="D")
        return data


class CSVDataSource(DataSource):
    """Class to fetch data from a CSV file"""
    def __init__(self, file_path:str="//data//data.csv", index_col:int=0, date_column=None):
        self.file_path = file_path
        self.index_col = index_col
        self.date_column = date_column

    def fetch_data(self):
        """Retrieve gross data from csv file"""
        data = pd.read_csv(self.file_path, index_col=self.index_col)
        if self.date_column:
            data.index = pd.to_datetime(data.index)
        return data

class AmazonS3(DataSource):
    """ class to fetch data from AWS S3"""
    def __init__(self,
                 bucket_name:str,
                 s3_object_name:str):
        """
        :param self.bucket_name: S3 bucket name.
        :param self.s3_object_name: S3 object name (file key).
        """
        self.s3 = None
        self.bucket_name = bucket_name
        self.s3_object_name = s3_object_name
        self.aws_credentials = None


    def connect_aws_s3(self) -> None:
        """
        Connect to AWS S3 using environment-based credentials.
        Credentials are automatically resolved by boto3 from:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_SESSION_TOKEN (optional)
        - AWS_DEFAULT_REGION
        """
        try:
            self.s3 = boto3.client("s3")
            logger.info("Successfully connected to AWS S3.")
        except Exception as e:
            logger.error(f"Failed to connect to AWS S3: {e}")
            raise RuntimeError("Could not connect to AWS S3. Check AWS credentials and region.")

    def fetch_data(self):
        """
        Get a file from S3 and return its content as originally stored format.
        :return: file content as original object (.parquet and .pkl supported for now).
        """
        if self.s3 is None:
            self.connect_aws_s3()

        response = self.s3.get_object(
            Bucket=self.bucket_name,
            Key=self.s3_object_name
        )
        file_content = response['Body'].read()
        if 'pkl' in self.s3_object_name.split('.')[-1]:
            file_content = pickle.loads(file_content)
        elif 'parquet' in self.s3_object_name.split('.')[-1]:
            file_content = pd.read_parquet(io.BytesIO(file_content))
        else:
            logger.error(f"Unsupported file extension for {self.s3_object_name}.")
            raise ValueError(f"Unsupported file extension for {self.s3_object_name}.")

        return file_content


class DataManager:
    """Class to manage, clean and preprocess data"""
    def __init__(self, data_source:DataSource,
                 max_consecutive_nan:int=5,
                 rebase_prices:bool=False, n_implementation_lags:int=1):
        """

        :param data_source:
        :param max_consecutive_nan:
        :param rebase_prices:
        :param n_implementation_lags:
        """
        # Data loading
        self.data_source = data_source
        # Data cleaning
        self.max_consecutive_nan = max_consecutive_nan
        self.rebase_prices = rebase_prices
        self.n_implementation_lags = n_implementation_lags
        self.raw_data = None
        self.cleaned_data = None
        self.returns = None
        # Aligned data
        self.aligned_prices = None
        self.aligned_returns = None

    def load_data(self):
        """Load data from the data source"""
        self.raw_data = self.data_source.fetch_data()
        return self.raw_data

    def clean_data(self,crop_lookback_period:int=0):
        """Clean the data by filling missing values"""
        if self.raw_data is None:
            self.load_data()

        if self.rebase_prices:
            df_filled = self.raw_data.copy()
            df_filled = (df_filled / df_filled.iloc[0,:])*100
        else:
            df_filled = self.raw_data.copy()

        for col in self.raw_data.columns:
            series = self.raw_data[col]
            first_valid_index = series.first_valid_index()

            if first_valid_index is None:
                continue

            is_nan = series.isna()
            counter = 0

            for i in series.index:
                if i < first_valid_index:
                    continue

                if is_nan[i]:
                    counter += 1
                    if counter <= self.max_consecutive_nan:
                        i_idx = series.index.get_loc(i)
                        df_filled.iloc[i_idx, self.raw_data.columns.get_loc(col)] = df_filled.iloc[
                            i_idx - 1, self.raw_data.columns.get_loc(col)]

                else:
                    counter = 0

        # Crop data if specified
        if crop_lookback_period!=0:
            if not crop_lookback_period<=df_filled.shape[0]:
                raise ValueError(f"Cannot lookback this further. Max allowed: {df_filled.shape[0]}")
            start_idx = df_filled.shape[0]-crop_lookback_period
            self.cleaned_data = df_filled.iloc[start_idx:,:]
        else:
            self.cleaned_data = df_filled
        return self.cleaned_data

    def compute_returns(self):
        """Compute returns from the cleaned data"""
        if self.cleaned_data is None:
            self.clean_data()

        self.returns = self.cleaned_data.pct_change(fill_method=None)
        self.returns.fillna(0.0)
        return self.returns

    def account_implementation_lags(self):
        if self.aligned_returns is None:
            self.aligned_returns = self.returns.shift(-self.n_implementation_lags)

        else:
            pass

    def get_data(
            self,
            format_date:str="%Y-%m-%d",
            crop_lookback_period:int=0,
            return_bool:bool=False
    )->Union[None, dict]:
        """Get all data prepared"""
        if self.cleaned_data is None:
            self.clean_data(crop_lookback_period=crop_lookback_period)
        if self.returns is None:
            self.compute_returns()
        if self.aligned_returns is None:
            self.account_implementation_lags()

        # Ensure index is dates
        for df in [self.raw_data, self.cleaned_data, self.returns, self.aligned_returns]:
            df.index = pd.to_datetime(df.index, format=format_date)

        if return_bool:
            return {'raw_data' : self.raw_data,
                    'cleaned_data' : self.cleaned_data,
                    'returns' : self.returns,
                    'aligned_returns': self.aligned_returns
                    }