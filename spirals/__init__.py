import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

__version__ = '0.1.0'


def cycle_bin_counts(sample, n_bins, samples_per_cycle):
    """Count the active samples that fall in each of `n_bins` equal bins of the cycle."""
    sample = np.asarray(sample)
    # Position within the cycle in [0, samples_per_cycle); computed from sample
    # indices rather than angles so bin edges are not affected by rounding.
    position = np.mod(np.arange(len(sample)), samples_per_cycle)
    bin_index = np.floor(position * n_bins / samples_per_cycle).astype(int)
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    return np.bincount(bin_index[sample == 1], minlength=n_bins)


def histogram_entropy(counts):
    """Shannon entropy (bits) of a histogram and its value normalised to [0, 1].

    The normalised entropy is 1 when counts are uniform across bins and 0 when
    all counts fall in a single bin.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0 or len(counts) < 2:
        return 0.0, 0.0
    p = counts[counts > 0] / total
    entropy = float(np.sum(p * np.log2(1 / p)))
    return entropy, entropy / np.log2(len(counts))


def plot_temporal_spiral(sample, n_bins, samples_per_cycle, ax=None):
    """Plot a binary time series on an outward spiral with an outer histogram.

    Each turn of the spiral contains `samples_per_cycle` consecutive samples.
    The outer histogram counts the active samples in `n_bins` equal bins of
    the cycle.
    """
    sample = np.asarray(sample)
    n_points = len(sample)
    theta = 2 * np.pi * np.arange(n_points) / samples_per_cycle
    radius = np.linspace(0.5, 10, n_points)

    x_spiral = radius * np.cos(theta)
    y_spiral = radius * np.sin(theta)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure

    inactive = sample == 0
    ax.scatter(
        x_spiral[inactive],
        y_spiral[inactive],
        s=18,
        color='#bdbdbd',
        alpha=0.35,
        label='inactive'
    )
    ax.scatter(
        x_spiral[sample == 1],
        y_spiral[sample == 1],
        s=28,
        color='#f28e2b',
        alpha=0.9,
        label='active'
    )

    ax.plot(x_spiral, y_spiral, color='#555555', alpha=0.2, linewidth=0.8)

    # Outer histogram: count active samples in each angular (temporal) bin of the cycle.
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_counts = cycle_bin_counts(sample, n_bins, samples_per_cycle)

    hist_base = radius.max() + 0.6
    hist_max_height = 3.0
    bar_heights = hist_max_height * bin_counts / max(bin_counts.max(), 1)

    for start, end, height, count in zip(bin_edges[:-1], bin_edges[1:], bar_heights, bin_counts):
        ax.add_patch(Wedge(
            (0, 0),
            hist_base + height,
            np.degrees(start),
            np.degrees(end),
            width=height,
            facecolor='#f28e2b',
            edgecolor='white',
            alpha=0.8
        ))
        mid = (start + end) / 2
        label_radius = hist_base + height + 0.4
        ax.text(label_radius * np.cos(mid), label_radius * np.sin(mid), str(count),
                ha='center', va='center', fontsize=8, color='#555555')

    ax.add_patch(plt.Circle((0, 0), hist_base, fill=False, color='#555555', alpha=0.4, linewidth=0.8))
    plot_limit = hist_base + hist_max_height + 1.0
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    n_cycles = n_points / samples_per_cycle
    ax.set(
        title=f'Activations on a temporal spiral\n ({samples_per_cycle} samples per cycle, '
              f'{n_cycles:g} cycles)',
        xlabel='Spiral x-coordinate',
        ylabel='Spiral y-coordinate'
    )
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(alpha=0.2)
    return fig, ax



def plot_cycle_histogram(sample, n_bins, samples_per_cycle, ax=None):
    """Plot the circular histogram of `plot_temporal_spiral` unrolled as a bar chart.

    The x-axis is the position within the cycle (in samples), split into
    `n_bins` equal bins, and each bar counts the active samples in that bin.
    The Shannon entropy of the histogram is shown in the title.

    Returns the figure, the axis, and the normalised entropy in [0, 1].
    """
    bin_counts = cycle_bin_counts(sample, n_bins, samples_per_cycle)
    entropy, normalised_entropy = histogram_entropy(bin_counts)
    bin_edges = np.linspace(0, samples_per_cycle, n_bins + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    ax.bar(
        bin_edges[:-1],
        bin_counts,
        width=np.diff(bin_edges),
        align='edge',
        color='#f28e2b',
        edgecolor='white',
        alpha=0.8
    )
    ax.set_xlim(0, samples_per_cycle)
    ax.set(
        title=f'Active samples per cycle bin\n'
              f'entropy = {entropy:.3f} bits (normalised {normalised_entropy:.3f})',
        xlabel='Position in cycle (samples)',
        ylabel='Count of active samples'
    )
    ax.grid(axis='y', alpha=0.2)
    return fig, ax, normalised_entropy


def plot_entropy_by_cycle_length(sample, n_bins, samples_per_cycle_list, ax=None):
    """Plot the histogram entropy for several candidate cycle lengths.

    For each value in `samples_per_cycle_list`, the active samples are binned
    into `n_bins` equal bins of the cycle and the normalised entropy of that
    histogram is computed. Low entropy means the activations concentrate in a
    few parts of the cycle, suggesting that cycle length matches a periodicity
    in the data.

    Returns the figure, the axis, and the array of normalised entropies.
    """
    samples_per_cycle_list = np.asarray(samples_per_cycle_list)
    entropies = np.array([
        histogram_entropy(cycle_bin_counts(sample, n_bins, samples_per_cycle))[1]
        for samples_per_cycle in samples_per_cycle_list
    ])

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    else:
        fig = ax.figure

    ax.plot(samples_per_cycle_list, entropies, marker='o', markersize=4, color='#f28e2b')

    ax.set(
        title=f'Histogram entropy by cycle length ({n_bins} bins)',
        xlabel='Samples per cycle',
        ylabel='Normalised entropy'
    )
    ax.grid(alpha=0.2)
    return fig, ax, entropies
