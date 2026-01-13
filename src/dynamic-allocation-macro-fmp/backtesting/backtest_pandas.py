import pandas as pd
import numpy as np

class Backtest:
    """Class to backtest a strategy"""
    def __init__(self,
                 returns:pd.DataFrame,
                 weights:pd.DataFrame,
                 turnover:pd.DataFrame,
                 transaction_costs:int|float=10,
                 strategy_name:str=""):
        self.returns = returns
        self.weights = weights
        self.portfolio_gross_returns = None
        self.portfolio_net_returns = None
        self.cropped_portfolio_gross_returns = None
        self.cropped_portfolio_net_returns = None
        self.transaction_costs = transaction_costs
        self.turnover = turnover
        self.start_date = None
        self.strategy_name = strategy_name

    def run_backtest(self)->None:
        """Run the backtest"""
        self.portfolio_gross_returns = (pd.DataFrame(
            (self.returns * self.weights).sum(axis=1),
        columns=[f"{self.strategy_name}"])
        )

        self.portfolio_net_returns = pd.DataFrame(data=np.nan,
                                                  index=self.portfolio_gross_returns.index,
                                                  columns=self.portfolio_gross_returns.columns)
        tc = (self.transaction_costs * (self.turnover / 10000)).fillna(0.0)
        self.portfolio_net_returns.loc[:,:] = self.portfolio_gross_returns.values - tc.values

        flg = self.weights.notna().any(axis=1)
        dates = self.weights.index[flg]
        self.start_date = dates.min()

        # Crop
        self.cropped_portfolio_gross_returns = self.portfolio_gross_returns.loc[self.start_date:,:]
        self.cropped_portfolio_net_returns = self.portfolio_net_returns.loc[self.start_date:, :]
        return

    def get_results(self)->None:
        """Get the backtest results"""
        if self.portfolio_gross_returns is None or self.portfolio_net_returns is None:
            self.run_backtest()
        return