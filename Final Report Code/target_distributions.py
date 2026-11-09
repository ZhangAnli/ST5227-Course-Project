"""Independent distribution plots for an editable registry of target transforms."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.stats import gaussian_kde


# Add a mode with a callable, an axis label, and its theoretical support.
# Callables receive the original grade array, not another mode's output.
TARGET_MODES = {
    'Original': dict(transform=lambda y: y, xlabel='Grade (percentage points)', support=(0, 100)),
    'Logit': dict(transform=lambda y: logit((y + .5) / 101.),
                  xlabel='Adjusted logit: logit((grade + 0.5) / 101)',
                  support=(logit(.5 / 101.), logit(100.5 / 101.))),
    'Square root': dict(transform=np.sqrt, xlabel='Square root of grade', support=(0, 10)),
    'Log1p': dict(transform=np.log1p, xlabel='log(1 + grade)', support=(0, np.log1p(100))),
}


def plot_target_distributions(grades, modes=None, bins=30):
    """Show one histogram/KDE per mode; return a descriptive summary table.

    Use the same students and bin count, but independent axes. Optional support
    sets the plot domain; omit it for a custom mode to use its observed range.
    Invalid transformations fail explicitly rather than dropping observations.
    """
    values = np.asarray(grades, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Grades must be a nonempty, finite, one-dimensional array.')
    if ((values < 0) | (values > 100)).any():
        raise ValueError('Expected original grades between 0 and 100.')
    modes = TARGET_MODES if modes is None else modes
    rows = []
    for name, mode in modes.items():
        transformed = np.asarray(mode['transform'](values.copy()), dtype=float)
        if transformed.shape != values.shape or not np.isfinite(transformed).all():
            raise ValueError(f'{name}: transform must return one finite value per student.')
        domain = mode.get('support', (transformed.min(), transformed.max()))
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        ax.hist(transformed, bins=bins, density=True, alpha=.4,
                edgecolor='white', label='Histogram')
        if len(transformed) > 1 and np.ptp(transformed) > 0:
            grid = np.linspace(*domain, 600)
            ax.plot(grid, gaussian_kde(transformed)(grid), linewidth=2, label='KDE')
        if domain[0] < domain[1]:
            ax.set_xlim(*domain)
        ax.set(xlabel=mode['xlabel'], ylabel='Density in transformed units',
               title=f'{name} target distribution · {len(values):,} students')
        ax.legend()
        plt.show()
        plt.close(fig)
        series = pd.Series(transformed)
        rows.append(dict(mode=name, students=len(series), minimum=series.min(),
                         median=series.median(), maximum=series.max(),
                         mean=series.mean(), SD=series.std(), skewness=series.skew()))
    return pd.DataFrame(rows).set_index('mode')
