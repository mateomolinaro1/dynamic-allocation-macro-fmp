import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, List, Optional, Dict, Iterable
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
            figsize: tuple = (10, 5)
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

    @staticmethod
    def plot_timeseries_dict(
            data: Dict[str, pd.DataFrame],
            save_path: str,
            title: str = "Time Series Plot",
            y_label: str = "Cumulative Returns",
            figsize: tuple = (12, 6),
            linewidth: float = 1.5,
            dashed_black_keys: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Plot all time series from a dict of DataFrames on a single figure and save it.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Dictionary of DataFrames with DatetimeIndex
        save_path : str
            Path where the plot will be saved
        title : str, optional
            Plot title
        y_label : str, optional
            Y-axis label
        figsize : tuple, optional
            Figure size
        linewidth : float, optional
            Line width
        dashed_black_keys : Iterable[str], optional
            Keys in the data dict for which the lines should be dashed black
        """
        dashed_black_keys = list(dashed_black_keys or [])
        black_linestyles = ["--", "-.", ":", (0, (5, 1))]

        plt.figure(figsize=figsize)

        for name, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"Index of DataFrame '{name}' must be a DatetimeIndex")

            is_highlighted = name in dashed_black_keys

            if is_highlighted:
                style_idx = dashed_black_keys.index(name) % len(black_linestyles)
                linestyle = black_linestyles[style_idx]
                color = "black"
                lw = linewidth + 0.8
                alpha = 1.0
                zorder = 3
            else:
                linestyle = "-"
                color = None
                lw = linewidth
                alpha = 0.8
                zorder = 2

            for col in df.columns:
                plt.plot(
                    df.index,
                    df[col],
                    label=f"{col}",
                    linestyle=linestyle,
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=zorder,
                )

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()
