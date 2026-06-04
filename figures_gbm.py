import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from matplotlib.lines import Line2D

# ── Data and split (identical to gbm.py) ─────────────────────────────────────
data = pd.read_csv('CO_dataset_final_ULTIMO.csv')

FEATURES = [
    'bond_CO', 'sp_ratio', 'angle_O_C_X', 'HOMO', 'lowdin_O',
    'charge_C', 'nu_CO_DFT', 'lowdin_C', 'HOMO_1', 'dipole',
]

X = data[FEATURES].to_numpy(dtype=float)
y = data['nu_CO_exp'].to_numpy(dtype=float)

strat_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat_bins
)

# Best hyperparameters from Optuna — update with your run's values
BEST_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=3, max_features=0.7,
)

best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gbm', GradientBoostingRegressor(**BEST_PARAMS, random_state=42))
])

strat_bins_train = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_preds = np.zeros(len(y_train))
fold_r2  = []
for tr, val in skf.split(X_train, strat_bins_train):
    m = clone(best_pipe)
    m.fit(X_train[tr], y_train[tr])
    cv_preds[val] = m.predict(X_train[val])
    fold_r2.append(r2_score(y_train[val], cv_preds[val]))

best_pipe.fit(X_train, y_train)
y_pred_test = best_pipe.predict(X_test)

r2_cv     = r2_score(y_train, cv_preds)
rmse_cv   = np.sqrt(mean_squared_error(y_train, cv_preds))
r2_test   = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"CV   R²={r2_cv:.4f}  RMSE={rmse_cv:.2f} cm⁻¹")
print(f"Test R²={r2_test:.4f}  RMSE={rmse_test:.2f} cm⁻¹")

def class_color(nu):
    return '#e74c3c' if nu > 2000 else '#2980b9'

c_train = [class_color(v) for v in y_train]
c_test  = [class_color(v) for v in y_test]

nu_min = min(y.min(), cv_preds.min(), y_pred_test.min()) - 20
nu_max = max(y.max(), cv_preds.max(), y_pred_test.max()) + 20

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9',
           markersize=9, label='Carbonyl (ν < 2000 cm⁻¹)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=9, label='Isocyanate (ν > 2000 cm⁻¹)'),
]

res_cv   = cv_preds - y_train
res_test = y_pred_test - y_test

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1a — Pred vs Exp (CV train)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_train, cv_preds, c=c_train, alpha=0.7, s=40,
           edgecolors='white', lw=0.4)
ax.plot([nu_min, nu_max], [nu_min, nu_max], 'k--', lw=1.2)
ax.set_xlim(nu_min, nu_max)
ax.set_ylim(nu_min, nu_max)
ax.set_xlabel('ν(C=O) experimental (cm⁻¹)', fontsize=12)
ax.set_ylabel('ν(C=O) predicted (cm⁻¹)', fontsize=12)
ax.text(0.05, 0.93, f'R² = {r2_cv:.4f}\nRMSE = {rmse_cv:.1f} cm⁻¹',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('gbm_pred_cv.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1b — Pred vs Exp (test)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_test, y_pred_test, c=c_test, alpha=0.7, s=40,
           edgecolors='white', lw=0.4)
ax.plot([nu_min, nu_max], [nu_min, nu_max], 'k--', lw=1.2)
ax.set_xlim(nu_min, nu_max)
ax.set_ylim(nu_min, nu_max)
ax.set_xlabel('ν(C=O) experimental (cm⁻¹)', fontsize=12)
ax.set_ylabel('ν(C=O) predicted (cm⁻¹)', fontsize=12)
ax.text(0.05, 0.93, f'R² = {r2_test:.4f}\nRMSE = {rmse_test:.1f} cm⁻¹',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('gbm_pred_test.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2a — Residuals vs Prediction
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(cv_preds, res_cv, c=c_train, alpha=0.65, s=35,
           edgecolors='white', lw=0.3)
ax.axhline(0, color='k', lw=1.2, ls='--')
ax.set_xlabel('ν(C=O) predicted (cm⁻¹)', fontsize=12)
ax.set_ylabel('Residual (pred − exp) (cm⁻¹)', fontsize=12)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('gbm_residuals.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2b — Residual distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
all_res = np.concatenate([res_cv, res_test])
ax.hist(res_cv,   bins=25, alpha=0.6, color='#2980b9', density=True, label='CV train')
ax.hist(res_test, bins=15, alpha=0.6, color='#e74c3c', density=True, label='Test')
x_r = np.linspace(all_res.min(), all_res.max(), 200)
mu, sigma = np.mean(res_cv), np.std(res_cv)
ax.plot(x_r, stats.norm.pdf(x_r, mu, sigma), 'k-', lw=1.5, label='Normal fit (CV)')
ax.set_xlabel('Residual (cm⁻¹)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('gbm_residuals_hist.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2c — Q-Q plot
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
(osm, osr), (slope, intercept, r) = stats.probplot(res_cv, dist='norm')
ax.scatter(osm, osr, color='#2980b9', alpha=0.65, s=35,
           edgecolors='white', lw=0.3)
x_qq = np.array([osm.min(), osm.max()])
ax.plot(x_qq, slope * x_qq + intercept, 'k--', lw=1.5)
ax.set_xlabel('Theoretical quantiles', fontsize=12)
ax.set_ylabel('Sample quantiles', fontsize=12)
ax.text(0.05, 0.93, f'r = {r:.3f}', transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('gbm_qq.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
scaler    = StandardScaler()
X_tr_sc   = scaler.fit_transform(X_train)
gbm_final = GradientBoostingRegressor(**BEST_PARAMS, random_state=42)
gbm_final.fit(X_tr_sc, y_train)

explainer        = shap.TreeExplainer(gbm_final)
shap_values      = explainer.shap_values(X_tr_sc)
mean_shap_signed = shap_values.mean(axis=0)

order   = np.argsort(mean_shap_signed)
abs_max = np.abs(mean_shap_signed).max()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(np.array(FEATURES)[order], mean_shap_signed[order],
               color=['#e74c3c' if mean_shap_signed[i] > 0 else '#2980b9' for i in order])
ax.set_xlabel('Mean SHAP value (cm⁻¹)', fontsize=12)
ax.set_xlim(-(abs_max * 1.30), abs_max * 1.30)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.grid(True, axis='x', alpha=0.3)
for bar, val in zip(bars, mean_shap_signed[order]):
    offset = abs_max * 0.02
    xpos = val + offset if val >= 0 else val - offset
    ha = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}', va='center', ha=ha, fontsize=9)
plt.tight_layout()
plt.savefig('gbm_shap.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — R² per fold
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
colors_fold = ['#e74c3c' if v < 0.7 else '#2980b9' for v in fold_r2]
bars = ax.bar([f'Fold {i+1}' for i in range(5)], fold_r2,
              color=colors_fold, edgecolor='white', linewidth=0.8)
ax.axhline(np.mean(fold_r2), color='k', ls='--', lw=1.5,
           label=f'Mean = {np.mean(fold_r2):.4f}')
ax.set_ylim(0, 1.05)
ax.set_ylabel('R²', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, fold_r2):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f'{val:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('gbm_cv_folds.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nSaved figures:")
for f in ['gbm_pred_cv.png', 'gbm_pred_test.png', 'gbm_residuals.png',
          'gbm_residuals_hist.png', 'gbm_qq.png', 'gbm_shap.png', 'gbm_cv_folds.png']:
    print(f"  {f}")
