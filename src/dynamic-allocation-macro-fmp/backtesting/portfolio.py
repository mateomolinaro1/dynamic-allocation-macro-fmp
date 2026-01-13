import polars as pl
from typing import Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union

class CreatePortfolio:
    """Class to compute portfolio level returns given assets' weights and returns"""

    def __init__(self,
                 weights: pl.DataFrame,
                 returns: pl.DataFrame,
                 rebal_periods: int = 0
                 ):
        self.weights = weights
        self.returns = returns
        self.rebal_periods = rebal_periods

        self.turnover = None
        self.rebalanced_weights = None


    def rebalance_portfolio(
            self,
            return_bool: bool = False
    ) -> Union[None, pl.DataFrame]:
        """
        Rebalance the portfolio and compute weights evolution between rebalancing dates.
        Returns rebalanced weights.

        Parameters:
        - return_bool: return the df is set to true otherwise just stored

        Returns:
        - None or df: rebalanced_weights
        """
        # If no rebalancing period specified or rebalancing at every period, return original weights
        if self.rebal_periods is None or self.rebal_periods == 0:
            if return_bool:
                return self.weights

        if not self.weights.schema["date"] in (pl.Date, pl.Datetime):
            raise ValueError("weights dataframe must have a 'date' column of type Date or Datetime")

        # If rebalancing period specified, we must compute weights accounting for drift
        # Step 0 Initialize the rebalanced_weights df
        self.rebalanced_weights = self.weights.clone()
        self.turnover = self.weights.select(
            "date",
            pl.lit(None).cast(pl.Float64).alias("turnover")
        )

        # Step 1 define the rebalancing dates range
        weights_only = pl.all().exclude("date")
        dates = (
            self.weights
            .with_columns(
                pl.sum_horizontal(weights_only).alias("row_sum")
            )
            .filter(pl.col("row_sum").is_not_null())
            .select("date")
            .to_series()
            .to_list()
        )

        start_date = dates[0]
        rebal_dates = dates[::self.rebal_periods]
        rebal_dates.pop(0)

        # Step 2: loop on all the dates
        for date in self.rebalanced_weights.select("date").to_series().to_list():
            if date < start_date:
                # we do nothing as it is already set to nan
                continue
            elif ((date > start_date) and (date in rebal_dates)) or date == start_date:
                # we do nothing because being at a rebalancing date means that we "reset" the weights
                # to EW and self.computes weights does that to every dates by default

                # Compute turnover
                num_idx = (
                    self.rebalanced_weights
                    .with_row_index()
                    .filter(pl.col("date") == date)
                    .select("index")
                    .item()
                )

                weights_only = pl.all().exclude("date")

                curr = (
                    self.rebalanced_weights
                    .slice(num_idx, 1)
                    .select(weights_only)
                )

                prev = (
                    self.rebalanced_weights
                    .slice(num_idx - 1, 1)
                    .select(weights_only)
                )

                turnover = (
                    (curr.fill_null(0.0) - prev.fill_null(0.0))
                    .select(pl.sum_horizontal(pl.all().abs()))
                    .item()
                )

                self.turnover = self.turnover.with_columns(
                    pl.when(pl.col("date") == date)
                    .then(turnover)
                    .otherwise(pl.col("turnover"))
                    .alias("turnover")
                )


            elif (date > start_date) and (date not in rebal_dates):
                # If we are not at a rebalancing date, we must derive the weights in between rebal dates
                weights_cols = pl.all().exclude("date")

                num_idx = (
                    self.rebalanced_weights
                    .with_row_index()
                    .filter(pl.col("date") == date)
                    .select("index")
                    .item()
                )

                prev_weights = (
                    self.rebalanced_weights
                    .slice(num_idx - 1, 1)
                    .select(weights_cols)
                )

                aligned_ret = (
                    self.returns
                    .slice(num_idx, 1)
                    .select(weights_cols)
                )

                drifted_w = (
                        prev_weights
                        * (1 + aligned_ret)
                )

                drifted_w = drifted_w / drifted_w.sum_horizontal()

                # write back
                self.rebalanced_weights = self.rebalanced_weights.with_columns([
                    pl.when(pl.col("date") == date)
                    .then(drifted_w[col])
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in self.rebalanced_weights.columns
                    if col != "date"
                ])

            else:
                continue

    def rebalance_portfolio_irregular(self, max_drift: int = 12)->None:
        """
        Vectorized irregular-weight backtest (no asset loop).
        """
        weights_pd = self.weights.to_pandas()
        weights_pd.index = weights_pd["date"]
        weights_pd = weights_pd.drop(columns=["date"])
        returns_pd = self.returns.to_pandas()
        returns_pd.index = returns_pd["date"]
        returns_pd = returns_pd.drop(columns=["date"])

        assert weights_pd.index.equals(returns_pd.index)
        assert (weights_pd.columns == returns_pd.columns).all()

        dates = weights_pd.index
        assets = weights_pd.columns

        # Storage
        self.rebalanced_weights = pd.DataFrame(index=dates, columns=assets, dtype=float)
        self.turnover = pd.DataFrame(index=dates, columns=["turnover"], dtype=float)

        # Months since last update
        months_since_update = pd.DataFrame(
            0, index=dates, columns=assets, dtype=int
        )

        # --- Initialization ---
        w0 = weights_pd.iloc[0]
        gross0 = w0.abs().sum()

        w0_final = w0 / gross0 if gross0 > 0 else np.nan
        self.rebalanced_weights.iloc[0] = w0_final

        # Initial turnover = build portfolio from zero
        self.turnover.iloc[0] = w0_final.abs().sum() if gross0>0 else 0.0

        # --- Time loop only ---
        for t in range(1, len(dates)):
            w_prev = self.rebalanced_weights.iloc[t - 1]
            r_t = returns_pd.iloc[t]
            w_obs = weights_pd.iloc[t]

            # Identify updates
            has_update = w_obs.notna()

            # Update months-since-update
            months_since_update.iloc[t] = (
                    months_since_update.iloc[t - 1] + 1
            )
            months_since_update.loc[dates[t],has_update] = 0

            # Drifted weights
            w_drift = w_prev * (1 + r_t)

            # Raw weights construction
            w_raw = pd.Series(index=assets, dtype=float)

            # Case 1: observed weights
            w_raw[has_update] = w_obs[has_update]

            # Case 2: drift allowed
            drift_ok = (~has_update) & (months_since_update.iloc[t] <= max_drift)
            w_raw[drift_ok] = w_drift[drift_ok]

            # Case 3: drift expired → NaN (already NaN by construction)

            # Normalize (gross exposure)
            gross = w_raw.abs().sum()
            if gross > 0:
                w_final = w_raw / gross
                self.rebalanced_weights.iloc[t] = w_final

                # --- Turnover ---
                self.turnover.iloc[t] = (
                    w_final.sub(w_prev, fill_value=0.0)
                    .abs()
                    .sum()
                )
            else:
                w_final = 0.0
                self.rebalanced_weights.iloc[t] = np.nan

                # --- Turnover ---
                # When moving to zero position, turnover is the sum of absolute previous weights
                self.turnover.iloc[t] = w_prev.abs().sum() if isinstance(w_prev, pd.Series) else 0.0

        # Convert back to pl.DataFrame
        self.rebalanced_weights = pl.from_pandas(
            self.rebalanced_weights.reset_index().rename(columns={"index":"date"})
        )
        # cast date to pl.Date
        self.rebalanced_weights = self.rebalanced_weights.with_columns(
            pl.col("date").cast(pl.Date)
        )
        self.turnover = pl.from_pandas(
            self.turnover.reset_index().rename(columns={"index":"date"})
        )
        self.turnover = self.turnover.with_columns(
            pl.col("date").cast(pl.Date)
        )

class WeightingScheme(ABC):
    """Abstract class to define the interface for the weighting scheme"""

    def __init__(self, returns: pd.DataFrame, aligned_returns:pd.DataFrame, signals: pd.DataFrame, rebal_periods: Union[int, None],
                 portfolio_type: str = "long_only"):
        self.returns = returns
        self.aligned_returns = aligned_returns
        self.signals = signals
        if not isinstance(rebal_periods, (int, type(None))):
            raise ValueError("rebal_periods must be an int or None.")
        self.rebal_periods = rebal_periods
        self.portfolio_type = portfolio_type
        self.weights = None
        self.rebalanced_weights = None
        self.returns_after_fees = None
        self.turnover = None

    @abstractmethod
    def compute_weights(self):
        """Compute the weights for the strategy"""
        pass

    @abstractmethod
    def rebalance_portfolio(self, transaction_cost_bp: float = 10):
        """
        Rebalance the portfolio at the specified frequency and computes transaction costs

        Parameters:
        - transaction_cost_bp (float): Transaction costs in basis points (default: 10bp)

        Returns:
        - tuple: (rebalanced_weights, net_returns)
        """
        pass

class EqualWeightingScheme(WeightingScheme):
    """Class to implement the equal weighting scheme"""

    def __init__(self, returns: pd.DataFrame, aligned_returns:pd.DataFrame, signals: pd.DataFrame, rebal_periods: int = 0,
                 portfolio_type: str = "long_only"):
        super().__init__(returns, aligned_returns, signals, rebal_periods, portfolio_type)

    def compute_weights(
            self,
            return_bool:bool=False
    )->Union[None, pd.DataFrame]:
        """Compute the weights for the equal weighting scheme"""
        number_of_positive_signals = (self.signals == 1).sum(axis=1)
        number_of_negative_signals = (self.signals == -1).sum(axis=1)

        weights = pd.DataFrame(data=np.nan, index=self.signals.index, columns=self.signals.columns)

        if self.portfolio_type == "long_only":
            # Check that there is at least one positive signal at each date
            if (number_of_positive_signals > 0).sum() == number_of_positive_signals.shape[0]:
                weights[self.signals == 1] = self.signals[self.signals == 1].div(number_of_positive_signals, axis=0)
            else:
                # Compute weights where there is at least one positive signal on the row
                mask_valid = number_of_positive_signals > 0
                weights[self.signals == 1] = self.signals[self.signals == 1].div(
                    number_of_positive_signals.where(mask_valid), axis=0)

                # Setting weights to nan at rows where all signals are missing
                weights.loc[~mask_valid, :] = np.nan  # Optional as in the creation of weights, default data set to 0
                print("For some dates, there is no signals. Weights set to nan by default.")

        elif self.portfolio_type == "short_only":
            # Check that there is at least one negative signal at each date
            if (number_of_negative_signals > 0).sum() == number_of_negative_signals.shape[0]:
                weights[self.signals == -1] = self.signals[self.signals == -1].div(number_of_negative_signals, axis=0)
            else:
                # Compute weights where there is at least one negative signal on the row
                mask_valid = number_of_negative_signals > 0
                weights[self.signals == -1] = self.signals[self.signals == -1].div(
                    number_of_negative_signals.where(mask_valid), axis=0)

                # Setting weights to 0 at rows where all signals are missing
                weights.loc[~mask_valid, :] = np.nan  # Optional as in the creation of weights, default data set to 0
                print("For some dates, there is no signals. Weights set to nan by default.")

        elif self.portfolio_type == "long_short":
            # Check that there is at least one positive and one negative signal at each date
            if ((number_of_positive_signals > 0).sum() == number_of_positive_signals.shape[0] and
                    (number_of_negative_signals > 0).sum() == number_of_negative_signals.shape[0]):
                weights[self.signals == 1] = self.signals[self.signals == 1].div(number_of_positive_signals, axis=0)
                weights[self.signals == -1] = self.signals[self.signals == -1].div(number_of_negative_signals, axis=0)
            else:
                raise ValueError("Error division by 0. There is no positive or negative signal for some dates.")

        else:
            raise ValueError("portfolio_type not supported")

        self.weights=weights
        if return_bool:
            return self.weights

    def rebalance_portfolio(
            self,
            return_bool:bool=False
    )->Union[None, pd.DataFrame]:
        """
        Rebalance the portfolio and compute weights evolution between rebalancing dates.
        Returns rebalanced weights and net returns (after transaction costs).

        Parameters:
        - return_bool: return the df is set to true otherwise just stored

        Returns:
        - None or df: rebalanced_weights
        """
        # First compute weights "not yet with drifts" if not already done
        if self.weights is None:
            self.weights = self.compute_weights()

        # If no rebalancing period specified or rebalancing at every period, return original weights
        if self.rebal_periods is None or self.rebal_periods == 0:
            if return_bool:
                return self.weights

        if not isinstance(self.weights.index, pd.DatetimeIndex):
            raise ValueError("The index must be of type pd.DatetimeIndex")

        # If rebalancing period specified, we must compute weights accounting for drift
        # Step 0 Initialize the rebalanced_weights df
        self.rebalanced_weights = self.weights.copy()
        self.turnover = pd.DataFrame(data=np.nan, index=self.weights.index, columns=["turnover"])

        # Step 1 define the rebalancing dates range
        flg = self.weights.sum(min_count=1, axis=1)
        flg = pd.notna(flg)
        dates = list(self.weights.index[flg])
        start_date = dates[0]
        rebal_dates = dates[::self.rebal_periods]
        rebal_dates.pop(0)

        # Step 2: loop on all the dates
        for date in self.rebalanced_weights.index:
            if date < start_date:
                # we do nothing as it is already set to nan
                continue
            elif ((date>start_date) and (date in rebal_dates)) or date==start_date:
                # we do nothing because being at a rebalancing date means that we "reset" the weights
                # to EW and self.computes weights does that to every dates by default

                # Compute turnover
                num_idx = self.rebalanced_weights.index.get_loc(date)
                turnover = (
                    (self.rebalanced_weights.iloc[num_idx, :].sub(
                        self.rebalanced_weights.iloc[num_idx - 1, :],
                        fill_value=0.0)
                    )
                    .abs()
                    .sum()
                )
                self.turnover.loc[date] = turnover

            elif (date>start_date) and (date not in rebal_dates):
                # If we are not at a rebalancing date, we must derive the weights in between rebal dates
                num_idx = self.rebalanced_weights.index.get_loc(date)
                prev_weights = self.rebalanced_weights.iloc[num_idx-1,:]
                aligned_ret = self.returns.iloc[num_idx,:]
                drifted_w = prev_weights*(1+aligned_ret)
                drifted_w = drifted_w/(drifted_w.sum())
                self.rebalanced_weights.loc[date,:] = drifted_w

            else:
                continue
