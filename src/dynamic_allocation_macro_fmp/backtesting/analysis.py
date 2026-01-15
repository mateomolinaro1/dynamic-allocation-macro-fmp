import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class PerformanceAnalyser:
    """Class to analyse the performance of a strategy"""
    def __init__(self, portfolio_returns:pd.DataFrame,
                 freq:str, zscores:Union[None, pd.DataFrame]=None,
                 bench_returns: pd.DataFrame = None,
                 forward_returns:Union[None, pd.DataFrame]=None,
                 percentiles:str="",
                 industries:str="",
                 rebal_freq:str=""):
        self.portfolio_returns = portfolio_returns
        self.bench_returns = bench_returns
        self.bench_cumulative_perf = None
        self.cumulative_performance = None
        self.equity_curve = None
        self.metrics = None
        if freq not in ['d', 'w', 'm', 'y']:
            raise ValueError("freq must be in ['d', 'w', 'm', 'y'].")
        self.freq = freq

        if (zscores is not None) and isinstance(zscores, pd.DataFrame):
            self.zscores = zscores
        elif (zscores is not None) and not isinstance(zscores, pd.DataFrame):
            raise ValueError("zscores must be a pandas dataframe.")
        else:
            pass

        if (forward_returns is not None) and isinstance(forward_returns, pd.DataFrame):
            self.forward_returns = forward_returns
        elif (forward_returns is not None) and not isinstance(forward_returns, pd.DataFrame):
            raise ValueError("forward_returns must be a pandas dataframe.")
        else:
            pass
        self.percentiles = percentiles
        self.industries = industries
        self.rebal_freq = rebal_freq

    def compute_cumulative_performance(self, compound_type: str = "geometric"):
        """Compute the cumulative performance of the strategy"""

        if self.portfolio_returns is not None:
            if compound_type == "geometric":
                self.cumulative_performance = (1 + self.portfolio_returns).cumprod() - 1
            elif compound_type == "arithmetic":
                self.cumulative_performance = self.portfolio_returns.cumsum()
            else:
                raise ValueError("Compound type not supported")

        if self.bench_returns is not None:
            if compound_type == "geometric":
                self.bench_cumulative_perf = (1 + self.bench_returns).cumprod() - 1
            elif compound_type == "arithmetic":
                self.bench_cumulative_perf = self.bench_returns.cumsum()

        return {
            "cumulative_perf": self.cumulative_performance,
            "bench_cumulative_perf": self.bench_cumulative_perf,
        }

    def compute_equity_curve(self):
        """Compute the equity curve of the strategy"""
        self.equity_curve = self.compute_cumulative_performance(compound_type="arithmetic")["cumulative_perf"]
        return self.equity_curve

    def compute_information_coefficient(self, ic_type:str='not_ranked',
                                        percentiles:Union[None, tuple]=None):
        """
        This functions returns a time-series of the information coefficient ('gross', i.e. not a percentage)
        :param ic_type: 'not_ranked' returns the ic of the values themselves whereas 'ranked' uses the ranks of the values
        :param percentiles: if set to None, uses all the data, if set to a tuple of int, e.g. (10,90) skip the "middle"
        part and only computes the ic of the extremities.
        :return: a time-series of the ic
        """
        if self.zscores is None or not isinstance(self.zscores, pd.DataFrame):
            raise ValueError("To compute the information coefficient, you must create PerformanceAnalyser object with zscores as a pandas dataframe. ")
        if self.forward_returns is None or not isinstance(self.forward_returns, pd.DataFrame):
            raise ValueError("To compute the information coefficient, you must create PerformanceAnalyser object with forward_returns as a pandas dataframe. ")
        if ic_type not in ['not_ranked', 'ranked']:
            raise ValueError("ic_type must be either 'not_ranked' or 'ranked' (string).")
        if percentiles is None:
            pass
        elif percentiles is not None:
            if not (isinstance(percentiles, tuple) and len(percentiles) == 2 and all(
                    isinstance(x, (int, float)) for x in percentiles)):
                raise ValueError("percentiles must be a tuple of exactly two elements, containing only int and float.")

        ic = pd.DataFrame(data=np.nan, index=self.zscores.index, columns=['information_coefficient'])
        first_date = self.zscores.first_valid_index()
        zs_truncated = self.zscores.loc[first_date:,:]
        for date in zs_truncated.index:
            if zs_truncated.loc[date, :].to_frame().T.isna().all().all():
                continue
            if ic_type == 'not_ranked':
                if percentiles is not None:
                    rank_temp = zs_truncated.loc[date, :].to_frame().T # we do not take the rank (this is just a variable name)
                    fwd_ret_temp = self.forward_returns.loc[date, :]

                    # Only evaluates the IC for the percentiles
                    upper_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[1]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    upper_bound = pd.DataFrame(data=np.tile(upper_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    lower_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[0]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    lower_bound = pd.DataFrame(data=np.tile(lower_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    # Percentiles selection for the ic computation
                    mask = (rank_temp >= upper_bound) | (rank_temp <= lower_bound)
                    cols_to_keep = mask.columns[mask.iloc[0, :]]
                    pct_rank = rank_temp[cols_to_keep].values.T
                    pct_fwd_ret = fwd_ret_temp[cols_to_keep].values[:, None]

                    # Store
                    corr_temp = np.corrcoef(pct_rank.ravel(), pct_fwd_ret.ravel())[0][1]
                    ic.loc[date, :] = corr_temp
                else:
                    corr_temp = np.corrcoef( zs_truncated.loc[date,:] , self.forward_returns.loc[date,:] )[0][1]
                    ic.loc[date,:] = corr_temp

            elif ic_type == 'ranked':
                if percentiles is not None:
                    rank_temp = zs_truncated.loc[date, :].rank(ascending=True).to_frame().T
                    fwd_ret_temp = self.forward_returns.loc[date, :]

                    # Only evaluates the IC for the percentiles
                    upper_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[1]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    upper_bound = pd.DataFrame(data=np.tile(upper_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    lower_bound = np.nanpercentile(rank_temp,
                                                   q=percentiles[0]) if not rank_temp.dropna().empty else np.nan
                    # Format to ease comparison after
                    lower_bound = pd.DataFrame(data=np.tile(lower_bound, (1, rank_temp.shape[1])),
                                               index=rank_temp.index,
                                               columns=rank_temp.columns)

                    # Percentiles selection for the ic computation
                    mask = (rank_temp >= upper_bound) | (rank_temp <= lower_bound)
                    cols_to_keep = mask.columns[mask.iloc[0,:]]
                    pct_rank = rank_temp[cols_to_keep].values.T
                    pct_fwd_ret = fwd_ret_temp[cols_to_keep].values[:,None]

                    # Store
                    corr_temp = np.corrcoef(pct_rank.ravel(), pct_fwd_ret.ravel())[0][1]
                    ic.loc[date, :] = corr_temp
                else:
                    rank = zs_truncated.loc[date,:].rank(ascending=True)
                    corr_temp = np.corrcoef( rank, self.forward_returns.loc[date,:] )[0][1]
                    ic.loc[date,:] = corr_temp
            else:
                raise ValueError("ic_type must be either 'not_ranked' or 'ranked' (string).")

        return ic

    @staticmethod
    def compute_max_drawdown(ret:pd.DataFrame) -> pd.Series:
        """
        Computes the maximum drawdown for each column in a returns DataFrame.

        Parameters:
        - ret (pd.DataFrame): A DataFrame of returns (each column = an asset or a portfolio).

        Returns:
        - pd.Series: Maximum drawdown for each column (negative values).
        """
        # Compute cumulative returns
        cumulative_returns = (1 + ret).cumprod()

        # Compute running maximum
        running_max = cumulative_returns.cummax()

        # Compute drawdowns
        drawdowns = (cumulative_returns / running_max) - 1

        # Compute maximum drawdown for each column
        max_drawdown = drawdowns.min()

        return max_drawdown


    def compute_metrics(self):
        """Compute the performance metrics of the strategy"""
        freq_mapping = {'d': 252, 'w': 52, 'm': 12, 'y': 1}

        if self.freq in freq_mapping:
            freq_num = freq_mapping[self.freq]
        else:
            raise ValueError(f"Invalid frequency '{self.freq}'. Expected one of {list(freq_mapping.keys())}.")

        if self.cumulative_performance is None:
            self.compute_cumulative_performance()

        # Compute basic performance metrics
        total_return = self.cumulative_performance.iloc[-1, 0]
        annualized_return = (1 + total_return) ** (freq_num / len(self.portfolio_returns)) - 1
        volatility = self.portfolio_returns.std() * np.sqrt(freq_num)
        sharpe_ratio = annualized_return / volatility

        # Maximum drawdown
        max_drawdown = self.compute_max_drawdown(self.portfolio_returns)

        # Metrics for bench
        if self.bench_returns is not None:
            total_return_bench = self.bench_cumulative_perf.iloc[-1, 0]
            annualized_return_bench = (1 + total_return_bench) ** (freq_num / len(self.bench_returns)) - 1
            volatility_bench = self.bench_returns.std() * np.sqrt(freq_num)
            sharpe_ratio_bench = annualized_return_bench / volatility_bench
            max_drawdown_bench = self.compute_max_drawdown(self.bench_returns)
        else:
            total_return_bench=None
            annualized_return_bench=None
            volatility_bench=None
            sharpe_ratio_bench=None
            max_drawdown_bench=None

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility.values[0],
            "annualized_sharpe_ratio": sharpe_ratio.values[0],
            "max_drawdown": max_drawdown.values[0],
            "total_return_bench": total_return_bench,
            "annualized_return_bench": annualized_return_bench,
            "annualized_volatility_bench": volatility_bench.values[0],
            "annualized_sharpe_ratio_bench": sharpe_ratio_bench.values[0],
            "max_drawdown_bench": max_drawdown_bench.values[0],
        }

    def plot_cumulative_performance(self,
                                    saving_path:str=None,
                                    show:bool=False,
                                    blocking:bool=True):
        """Plot the cumulative performance of the strategy"""
        if self.cumulative_performance is None:
            self.compute_cumulative_performance()
        if self.metrics is None:
            self.metrics = self.compute_metrics()

        plt.figure(figsize=(12, 6))
        if self.bench_returns is None:
            plt.plot(self.cumulative_performance.index, self.cumulative_performance, label=f"{self.portfolio_returns.columns[0]}_returns")
            plt.title(f"Cumulative Performance of strategy:{self.portfolio_returns.columns[0]} - {self.percentiles} - {self.industries} - {self.rebal_freq} \n"
                      f"Performance Metrics: ann. ret={self.metrics['annualized_return']:.2%}; ann. vol={self.metrics['annualized_volatility']:.2%}; ann. SR={self.metrics['annualized_sharpe_ratio']:.2f}; max. dd={self.metrics['max_drawdown']:.2%}",
                      fontsize=10)
        else:
            plt.plot(self.cumulative_performance.index, self.cumulative_performance,
                     label=f"{self.portfolio_returns.columns[0]}")
            plt.plot(self.bench_cumulative_perf.index, self.bench_cumulative_perf,
                     label=f"{self.bench_returns.columns[0]}")
            plt.title(
                f"Cumulative Performance of strategy:{self.portfolio_returns.columns[0]} - {self.percentiles} - {self.industries} - {self.rebal_freq} \n"
                f"Performance Metrics: ann. ret={self.metrics['annualized_return']:.2%}; ann. vol={self.metrics['annualized_volatility']:.2%}; ann. SR={self.metrics['annualized_sharpe_ratio']:.2f}; max. dd={self.metrics['max_drawdown']:.2%} \n"
                f"Cumulative Performance of bench:{self.bench_returns.columns[0]} \n"
                f"Performance Metrics Bench: ann. ret={self.metrics['annualized_return_bench']:.2%}; ann. vol={self.metrics['annualized_volatility_bench']:.2%}; ann. SR={self.metrics['annualized_sharpe_ratio_bench']:.2f}; max. dd={self.metrics['max_drawdown_bench']:.2%}",
                fontsize=10)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Performance")
        plt.legend()
        plt.grid()

        if saving_path is not None:
            plt.savefig(saving_path, bbox_inches='tight')
        if show:
            plt.show(block=blocking)

        plt.close()
