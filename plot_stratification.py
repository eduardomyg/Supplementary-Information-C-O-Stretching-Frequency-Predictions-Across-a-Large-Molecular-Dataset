import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('CO_dataset_final_ULTIMO.csv')
y = data['nu_CO_exp'].to_numpy(dtype=float)

strat_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
_, _, y_train, y_test = train_test_split(
    y, y, test_size=0.2, random_state=42, stratify=strat_bins
)

# ── CDF ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for arr, label, color, ls in [
    (y,       'Full dataset', 'black',   '-'),
    (y_train, 'Train (80%)',  '#2980b9', '--'),
    (y_test,  'Test (20%)',   '#e74c3c', ':'),
]:
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(arr) + 1) / len(arr)
    ax.plot(sorted_arr, cdf, label=f'{label}  (n={len(arr)})',
            color=color, linestyle=ls, linewidth=2)
ax.set_xlabel('ν(C=O) experimental (cm⁻¹)', fontsize=12)
ax.set_ylabel('Cumulative probability', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('strat_cdf.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: strat_cdf.png")

# ── Histogram ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
bins = np.linspace(y.min(), y.max(), 25)
ax.hist(y_train, bins=bins, density=True, alpha=0.55,
        color='#2980b9', label=f'Train (n={len(y_train)})')
ax.hist(y_test,  bins=bins, density=True, alpha=0.55,
        color='#e74c3c', label=f'Test  (n={len(y_test)})')
ax.set_xlabel('ν(C=O) experimental (cm⁻¹)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.97, 'Stratified by ν(C=O) quintiles, seed = 42',
        transform=ax.transAxes, fontsize=8.5, color='gray',
        ha='right', va='top')
plt.tight_layout()
plt.savefig('strat_hist.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: strat_hist.png")

# ── Estadísticas por split ──────────────────────────────────────────────────
print(f"\n{'':20s} {'Full':>10} {'Train':>10} {'Test':>10}")
print("-" * 52)
for stat, fn in [('Media',  np.mean), ('Mediana', np.median),
                 ('Std',    np.std),  ('Min',     np.min), ('Max', np.max)]:
    print(f"{stat:20s} {fn(y):10.2f} {fn(y_train):10.2f} {fn(y_test):10.2f}")
print(f"{'N':20s} {len(y):10d} {len(y_train):10d} {len(y_test):10d}")
