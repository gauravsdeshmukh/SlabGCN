"""Utilites for plotting."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

import matplotlib
matplotlib.use('Agg')

class Canvas:
    """Wrapper class for a matplotlib figure."""

    def __init__(self, n_rows, n_cols, dpi=500, scale=[3, 3], facecolor="white"):
        """Initialize a canvas.

        Parameters
        ----------
        n_rows: int
            Number of rows in figure
        n_cols: int
            Number of columns in figure
        dpi: int
            Quality of figure (dots-per-inch)
        scale: list
            Scaling factor to determine figure size
        facecolor: str
            Facecolor for the figure
        """
        # Create figure and axes
        fig_size = np.array(np.array(scale) * np.array([n_cols, n_rows])).astype(int)
        self.fig, self.ax = plt.subplots(
            n_rows, n_cols, figsize=(fig_size), dpi=dpi, facecolor=facecolor
        )

    def save(self, save_path):
        """Save a figure to the given path.

        Parameters
        ----------
        save_path: str
            Path to save the file at (must include valid extension).
        """
        self.fig.tight_layout()
        self.fig.savefig(save_path)

    def show(self):
        """Show the figure."""
        self.fig.show()


# Utility functions
# Utility functions
def make_parity_plot(y_true, y_pred, save_path):
    """Make a parity plot of the true targets against the predictions.

    Parameters
    ----------
    y_true: list or np.ndarray or torch.Tensor
        True values (targets)
    y_pred: list or np.ndarray or torch.Tensor
        Predicted values (predictions)
    save_path: str
        Path to save the parity plot at (must include valid extension).
    """
    ## Create canvas
    canvas = Canvas(1, 1)

    ## Make plot
    canvas.ax.scatter(
        y_pred, y_true, color="mediumseagreen", marker="o", edgecolors="black"
    )

    ## Make parity line
    # Set proper limits
    min_lim = np.amin(np.concatenate((y_true, y_pred)))
    max_lim = np.amax(np.concatenate((y_true, y_pred)))
    min_lim = min_lim * (1 - np.sign(min_lim) * 0.15)
    max_lim = max_lim * (1 + np.sign(max_lim) * 0.15)
    canvas.ax.set_xlim([min_lim, max_lim])
    canvas.ax.set_ylim([min_lim, max_lim])

    # Plot the line
    canvas.ax.plot(
        [min_lim, max_lim], [min_lim, max_lim], color="black", linestyle="--"
    )
    canvas.ax.margins(x=0, y=0)

    ## Set labels
    canvas.ax.set_xlabel("Predictions")
    canvas.ax.set_ylabel("Targets")
    canvas.ax.set_title("Parity plot")

    # Save
    canvas.save(save_path)

# Utility functions
def make_loss_plot(train_losses, val_losses, save_path):
    """Make a parity plot of the true targets against the predictions.

    Parameters
    ----------
    train_losses: list or np.ndarray or torch.Tensor
        Training losses
    val_losses: list or np.ndarray or torch.Tensor
        Validation losses
    save_path: str
        Path to save the parity plot at (must include valid extension).
    """
    # Make epochs array
    epochs = np.arange(1, len(train_losses) + 1)

    ## Create canvas
    canvas = Canvas(1, 1)

    ## Make plot
    canvas.ax.plot(
        epochs,
        train_losses,
        color="black",
        linestyle="-",
        label="Training loss"
    )
    canvas.ax.plot(
        epochs,
        val_losses,
        color="royalblue",
        linestyle="-",
        label="Validation loss"
    )

    ## Set labels
    canvas.ax.set_xlabel("Epochs")
    canvas.ax.set_ylabel("Loss")
    canvas.ax.set_title("Loss plot")
    canvas.ax.legend()

    # Save
    canvas.save(save_path)

def make_phase_diagram(coverage_grid, T, P_ratio, max_coverage, save_path):
    """Make a (T, P, coverage) phase diagram based on the given coverage grid.
    
    Parameters
    ----------
    coverage_grid: np.ndarray
        A 2D matrix containing coverages for each T (X-axis) and P_ratio (Y-axis)
    T: np.ndarray
        An array containing temperatures (X-axis)
    P_ratio: np.ndarray
        An array containing pressure ratios (Y-axis)
    max_coverage: float
        Maximum coverage (for colorbar)
    save_path: str or Path
        Path to save the plot to
    
    """
    # Create Canvas
    canvas = Canvas(1, 1, scale=[5, 3])

    # Make imshow plot
    _ = canvas.ax.pcolormesh(T, P_ratio, coverage_grid, cmap="inferno_r",
                             vmin=0, vmax=max_coverage)

    # Set log scale
    canvas.ax.set_yscale("log")

    # Set labels
    canvas.ax.set_xlabel("Temperature (K)")
    canvas.ax.set_ylabel("$P_{H_2S} / P_{H_2}$")

    # Add colorbar
    canvas.fig.colorbar(_)
    
    # Save figure
    canvas.save(save_path)
