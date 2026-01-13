import matplotlib.pyplot as plt


class Visualizer:
    """Visualize results of the backtest"""

    def __init__(self, performance):
        self.performance = performance

    def plot_cumulative_performance(self, title="Cumulative Performance", figsize=(10, 6)):
        """Display the cumulative performance"""
        plt.figure(figsize=figsize)
        plt.plot(self.performance.cumulative_performance, label="Strategy Returns")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show(block=True)

    def plot_equity_curve(self, title="Equity Curve", figsize=(10, 6)):
        """Display the equity curve"""
        plt.figure(figsize=figsize)
        plt.plot(self.performance.equity_curve, label="Equity Curve")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show(block=True)

    def plot_drawdowns(self, title="Drawdowns", figsize=(10, 6)):
        """Display the drawdowns"""
        if self.performance.cumulative_performance is None:
            self.performance.compute_cumulative_performance()

        rolling_max = self.performance.cumulative_performance.cummax()
        drawdown = (self.performance.cumulative_performance / rolling_max) - 1

        plt.figure(figsize=figsize)
        plt.plot(drawdown, label="Drawdowns")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.fill_between(drawdown.index, drawdown.iloc[:, 0], 0, color='red', alpha=0.3)
        plt.show(block=True)