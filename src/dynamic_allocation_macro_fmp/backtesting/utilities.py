import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from sklearn.linear_model import LinearRegression
import warnings

def compute_percentiles(df: pd.DataFrame, percentiles: Tuple[int, int]):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas df.")
    if not (isinstance(percentiles, tuple) and len(percentiles) == 2 and all(isinstance(x, int) for x in percentiles)):
        raise ValueError("percentiles must be a tuple of exactly two elements, containing only int.")

    # Calcul des bornes supérieures et inférieures
    upper_bound = df.apply(
        lambda row: np.nanpercentile(row, q=percentiles[1]) if not row.dropna().empty else np.nan, axis=1)
    lower_bound = df.apply(
        lambda row: np.nanpercentile(row, q=percentiles[0]) if not row.dropna().empty else np.nan, axis=1)

    # Formatage pour comparaison
    upper_bound = pd.DataFrame(data=np.tile(upper_bound.values[:, None], (1, df.shape[1])), index=df.index,
                               columns=df.columns)
    lower_bound = pd.DataFrame(data=np.tile(lower_bound.values[:, None], (1, df.shape[1])), index=df.index,
                               columns=df.columns)

    # Calcul des signaux
    signals = pd.DataFrame(data=np.nan, index=df.index, columns=df.columns)
    signals[df >= upper_bound] = 1.0
    signals[df <= lower_bound] = -1.0

    # Calcul de tous les percentiles 
    all_percentiles = {f"p{q}": df.apply(
        lambda row: np.nanpercentile(row, q=q) if not row.dropna().empty else np.nan, axis=1)
        for q in range(0, 101, 10)} # Calcul des percentiles de 0 à 100 par pas de 10

    # Formatage des percentiles
    for key, series in all_percentiles.items():
        all_percentiles[key] = pd.DataFrame(data=np.tile(series.values[:, None], (1, df.shape[1])), index=df.index,
                                            columns=df.columns)

    # Retourne les bornes, les signaux et tous les percentiles
    return {
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'signals': signals,
        'all_percentiles': all_percentiles  # Tous les percentiles calculés
    }


def clean_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by replacing -inf or inf values by nan.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: A cleaned DataFrame with NaN rows and columns removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas df.")

    # Replace -inf and inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def compute_zscores(df:pd.DataFrame, axis:int=1) -> pd.DataFrame:
    """
    Computes the z-scores of a DataFrame along the specified axis.

    Args:
        df (pd.DataFrame): The DataFrame to compute z-scores for.
        axis (int): The axis along which to compute z-scores. 0 for rows, 1 for columns.

    Returns:
        pd.DataFrame: A DataFrame containing the z-scores.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas df.")
    if axis not in [0, 1]:
        raise ValueError("axis must be either 0 (rows) or 1 (columns).")

    mean = df.mean(axis=axis, skipna=True)
    std = df.std(axis=axis, skipna=True)

    zscores = (df.values - mean.values[:,None]) / std.values[:,None]
    zscores = pd.DataFrame(data=zscores, index=df.index, columns=df.columns)
    return zscores

def winsorize_dataframe(df:pd.DataFrame, percentiles:Tuple[int, int]=(1,99), axis:int=1) -> pd.DataFrame:
    """
    Winsorizes the DataFrame by replacing extreme values with the specified percentiles row-wise or column-wise.

    Args:
        df (pd.DataFrame): The DataFrame to winsorize.
        percentiles (Tuple[int, int]): The lower and upper percentiles to use for winsorization.
        axis (int): The axis along which to apply winsorization.
                    0 for column-wise (apply on each column),
                    1 for row-wise (apply on each row).
    Returns:
        pd.DataFrame: A winsorized DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if not (
            isinstance(percentiles, tuple)
            and len(percentiles) == 2
            and all(isinstance(x, int) for x in percentiles)
    ):
        raise ValueError("percentiles must be a tuple of two integers.")
    if axis not in [0, 1]:
        raise ValueError("axis must be either 0 (rows) or 1 (columns).")

    def winsorize_row_or_col(row_or_col):
        if row_or_col.isna().all():
            return row_or_col  # we keep the row or colum empty
        lower = np.nanpercentile(row_or_col, percentiles[0])
        upper = np.nanpercentile(row_or_col, percentiles[1])
        return row_or_col.clip(lower=lower, upper=upper)

    return df.apply(winsorize_row_or_col, axis=axis)

def compute_sharpe_ratio(df_returns:pd.DataFrame,
                         risk_free_rate:Union[float, pd.DataFrame]=0.0,
                         frequency:str='daily') -> Union[pd.DataFrame,float]:
    """ Computes the Sharpe ratio of a DataFrame of returns."""
    # Input checks
    if not isinstance(df_returns, pd.DataFrame):
        raise ValueError("df_returns must be a pandas DataFrame.")
    if not isinstance(risk_free_rate, (float, pd.DataFrame)):
        raise ValueError("risk_free_rate must be a float or a pandas DataFrame.")
    if not isinstance(frequency, str) and frequency not in ['daily', 'weekly', 'monthly', 'yearly']:
        raise ValueError("frequency must be either 'daily', 'weekly', 'monthly' or 'yearly.")

    # Frequency conversion
    if frequency == 'daily':
        freq = 252
    elif frequency == 'weekly':
        freq = 52
    elif frequency == 'monthly':
        freq = 12
    elif frequency == 'yearly':
        freq = 1

    # Compute excess returns
    if isinstance(risk_free_rate, pd.DataFrame):
        excess_returns = df_returns.values - risk_free_rate.values
        excess_returns = pd.DataFrame(data=excess_returns, index=df_returns.index, columns=df_returns.columns)
    else:
        excess_returns = df_returns - risk_free_rate

    # Compute mean and standard deviation nd avoiding warnings in the presence of nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_excess_returns = excess_returns.mean(axis=0, skipna=True)
        std_excess_returns = excess_returns.std(axis=0, skipna=True)

    # Compute Sharpe ratio
    sharpe_ratio = mean_excess_returns / std_excess_returns
    # Annualize the Sharpe ratio
    sharpe_ratio = sharpe_ratio * np.sqrt(freq)

    return pd.DataFrame(sharpe_ratio).T

def rolling_sharpe_ratio(df_returns:pd.DataFrame,
                         rolling_window:int,
                         risk_free_rate:Union[float, pd.DataFrame]=0.0,
                         frequency:str='daily') -> Union[pd.DataFrame,float]:

    if not isinstance(rolling_window, int):
        raise ValueError("rolling_window must be an int.")
    if rolling_window <= 0:
        raise ValueError("rolling_window must be greater than 0.")
    if df_returns.shape[0] < rolling_window:
        raise ValueError("rolling_window must be less than the number of rows in df_returns.")

    df_sharpe_ratios = pd.DataFrame(data=np.nan, index=df_returns.index, columns=df_returns.columns)
    for i_end in range(rolling_window, df_returns.shape[0]):
        df_returns_local = df_returns.iloc[i_end-rolling_window:i_end,:]
        sharpe_ratio = compute_sharpe_ratio(df_returns=df_returns_local,
                                            risk_free_rate=risk_free_rate,
                                            frequency=frequency)
        df_sharpe_ratios.iloc[i_end,:] = sharpe_ratio.values

    return df_sharpe_ratios

def compute_idiosyncratic_returns(df_assets:pd.DataFrame, df_factors:pd.DataFrame, window_regression:int):
    """
    Computes the idiosyncratic returns of the assets using a rolling regression.

    Args:
        df_assets (pd.DataFrame): DataFrame of asset returns.
        df_factors (pd.DataFrame): DataFrame of factor returns.
        window_regression (int): Window size for the rolling regression.
    """
    # Check inputs
    if not isinstance(df_assets, pd.DataFrame):
        raise TypeError("The `df_assets` parameter must be a pandas DataFrame.")
    if not isinstance(df_factors, pd.DataFrame):
        raise TypeError("The `df_factors` parameter must be a pandas DataFrame.")
    if not df_assets.shape[0] == df_factors.shape[0]:
        raise ValueError("The number of rows in df_assets and df_factors must be equal.")
    if not df_assets.index.equals(df_factors.index):
        raise ValueError("The indices of df_assets and df_factors must be equal.")
    if window_regression > df_assets.shape[0]:
        raise ValueError("window_regression cannot be greater than the nb of rows in df.")

    # Perform rolling regression
    residuals = pd.DataFrame(data=np.nan, index=df_assets.index, columns=df_assets.columns)
    for col_idx,col in enumerate(df_assets.columns):
        print(f"Working on column {col} ({col_idx+1}/{df_assets.shape[1]})")
        for i in range(window_regression, df_assets.shape[0]):
            print(f"Working on row {i} ({i+1}/{df_assets.shape[0]})")
            # Get the current window of data
            y = df_assets.iloc[i-window_regression:i,col_idx]
            x = df_factors.iloc[i-window_regression:i]

            # Check presence of NaN values
            merged_yx = pd.merge(y, x, left_index=True, right_index=True, how='inner')
            merged_yx = merged_yx.dropna()
            # Check if we have enough data points for regression
            # If not, skip this iteration
            if merged_yx.shape[0] < 2:
                continue
            y_cleaned = merged_yx.iloc[:,0].values.reshape(-1,1)
            x_cleaned = merged_yx.iloc[:,1:].values

            # Perform regression
            model = LinearRegression()
            model.fit(x_cleaned, y_cleaned)
            # Get residuals
            res = y_cleaned - model.predict(x_cleaned)
            # Store residuals
            residuals.iloc[i, col_idx] = res[-1][0]

    return residuals

def plot_dataframe(df:pd.DataFrame,
                   bench:pd.DataFrame=None,
                   save_path:str=None,
                   title:str="Cumulative Log Performance overtime",
                   xlabel:str="Date",
                   ylabel:str="Cumulative Return",
                   legend:bool=True,
                   figsize:Tuple[int, int]=(20, 15),
                   bbox_to_anchor:Tuple[float, float]=(0.5, -0.15),
                   ncol:int=6,
                   fontsize:int=7,
                   show:bool=False,
                   blocking:bool=False) -> None:

    plt.figure(figsize=figsize)
    plt.plot(df)
    if bench is not None:
        plt.plot(bench, color='black', label='Benchmark', linestyle="--", linewidth=4)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(
            df.columns,
            loc='upper center',
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            fontsize=fontsize,
            frameon=False
        )

    plt.grid()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show(block=blocking)
    plt.close()
