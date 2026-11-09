"""Plot prior versus current grades for the experiment's complete-case cohort."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

root = Path(__file__).resolve().parent
df = (pd.read_csv(root / 'student_grade_ses_dataset.csv')
      .drop_duplicates().set_index('student_id').dropna())
x, y = df['previous_year_grade_pct'], df['final_grade_pct']
r = pearsonr(x, y).statistic
rho = spearmanr(x, y).statistic
low = df['grade_class'].eq('F')
fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
ax.scatter(x[~low], y[~low], s=12, alpha=.15, color='steelblue', rasterized=True, label='Other grades')
ax.scatter(x[low], y[low], s=22, alpha=.6, color='tomato', rasterized=True, label='F band (current grade)')
grid = np.linspace(x.min(), x.max(), 200)
slope, intercept = np.polyfit(x, y, 1)
ax.plot(grid, intercept + slope*grid, color='darkblue', lw=2, label='Linear fit (all students)')
ax.plot([0,100], [0,100], '--', color='grey', lw=1.2, label='Same grade in both years')
ax.set(xlabel='Previous-year grade (%)', ylabel='Current target grade (%)', xlim=(0,100), ylim=(0,100),
       title=f'Previous-year versus current grade — {len(df):,} students\nPearson r = {r:.3f} | Spearman ρ = {rho:.3f}')
ax.legend(loc='upper left')
ax.grid(alpha=.15)
output = root / 'figures' / 'previous-year-vs-target.png'
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=170)
plt.close(fig)
print(f'n={len(df)}, Pearson r={r:.6f}, Spearman rho={rho:.6f}')
print(f'F-band students: {low.sum()}, previous grade range: {x[low].min():.2f}–{x[low].max():.2f}')
print(output)
