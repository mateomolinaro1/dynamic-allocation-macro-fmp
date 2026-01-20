import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, List, Optional
from pathlib import Path

class Vizu:
    @staticmethod
    def plot_time_series(
            data: Union[pd.Series, pd.DataFrame, List[pd.Series]],
            title: Optional[str] = None,
            ylabel: Optional[str] = None,
            xlabel: str = "Date",
            save_path: Optional[str]|Optional[Path] = None,
            show: bool = True,
            block: bool = True,
            figsize: tuple = (10, 5),
    ):
        """
        Plot one or multiple time series with a DateTime index.

        Parameters
        ----------
        data : pd.Series, pd.DataFrame, or list of pd.Series
            Time series data to plot. Index must be dates.
        title : str, optional
            Plot title.
        ylabel : str, optional
            Y-axis label.
        xlabel : str
            X-axis label.
        save_path : str, Path, optional
            Path to save the figure (e.g. 'figures/my_plot.png').
        show : bool
            Whether to display the plot.
        block : bool
            Whether plt.show() should block execution (important for PyCharm).
        figsize : tuple
            Figure size.
        """

        # --------------------
        # Normalize input
        # --------------------
        if isinstance(data, pd.Series):
            df = data.to_frame()
        elif isinstance(data, list):
            df = pd.concat(data, axis=1)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("data must be a Series, DataFrame, or list of Series")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex")

        # --------------------
        # Plot
        # --------------------
        fig, ax = plt.subplots(figsize=figsize)

        df.plot(ax=ax)

        ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        ax.grid(True)
        ax.legend()

        # --------------------
        # Save
        # --------------------
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        # --------------------
        # Show / close
        # --------------------
        if show:
            plt.show(block=block)
        else:
            plt.close(fig)