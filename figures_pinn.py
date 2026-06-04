import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from matplotlib.lines import Line2D

# ── Data and split (identical to pinn.py) ────────────────────────────────────
data = pd.read_csv('CO_dataset_final_ULTIMO.csv')
data['dihedral_real'] = data['dihedral_real'].fillna(90.0)

NN_FEATURES = [
    'bond_CO', 'sp_ratio', 'angle_O_C_X', 'HOMO', 'lowdin_O',
    'charge_C', 'nu_CO_DFT', 'lowdin_C', 'HOMO_1', 'dipole',
]
D_CO = 0.68

def build_Xp(df):
    r   = df['bond_CO'].values.astype(np.float32)
    bo  = df['mayer_CO'].values.astype(np.float32)
    qC  = df['charge_C'].values.astype(np.float32)
    qO  = df['charge_O'].values.astype(np.float32)
    dih = df['dihedral_real'].values.astype(np.float32)
    pis = np.abs(df['pi_star'].values.astype(np.float32))
    Ef  = df['Epar_lowdin_MV_per_cm'].values.astype(np.float32)
    return np.column_stack([
        (r - D_CO) ** (-3),
        np.cos(np.radians(df['angle_O_C_X'].values)),
        np.cos(np.radians(dih)) ** 2,
        np.clip(bo, 0, None) ** 0.75,
        1.0 / (pis + 1e-3),
        (qO - qC) / (r ** 2),
        Ef,
    ]).astype(np.float32)

X_nn = data[NN_FEATURES].to_numpy(dtype=np.float32)
Xp   = build_Xp(data)
y    = data['nu_CO_exp'].to_numpy(dtype=np.float32)

strat_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
idx = np.arange(len(y))
idx_train, idx_test = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=strat_bins
)

X_nn_train, X_nn_test = X_nn[idx_train], X_nn[idx_test]
Xp_train,   Xp_test   = Xp[idx_train],  Xp[idx_test]
y_train,    y_test     = y[idx_train],   y[idx_test]

strat_bins_train = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
a_init = np.array([0.1, -0.1, 0.05, 0.1, 0.05, 0.0, 0.0], dtype=np.float32)

def init_logC(Xp_arr, y_mean):
    k_avg = float(np.mean(np.abs((Xp_arr * a_init).sum(axis=1))))
    return float(np.log(y_mean / (np.sqrt(k_avg) + 1e-8)))

class PINN(nn.Module):
    def __init__(self, n_in, y_mean, y_std, logC_init):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 16), nn.Tanh(), nn.Dropout(0.3),
            nn.Linear(16, 8),    nn.Tanh(), nn.Dropout(0.3),
            nn.Linear(8, 1),
        )
        self.a    = nn.Parameter(torch.tensor([0.1, -0.1, 0.05, 0.1, 0.05, 0.0, 0.0]))
        self.logC = nn.Parameter(torch.tensor(logC_init))
        self.register_buffer('y_mean', torch.tensor(float(y_mean)))
        self.register_buffer('y_std',  torch.tensor(float(y_std)))

    def physics(self, xp):
        k = (self.a * xp).sum(dim=1)
        return (self.logC.exp() * k.abs().sqrt() - self.y_mean) / self.y_std

    def forward(self, x, xp):
        return self.net(x).squeeze(), self.physics(xp)


def train_model(X_t, y_n, Xp_t, y_mean, y_std, logC_init, epochs):
    m   = PINN(X_t.shape[1], y_mean, y_std, logC_init)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()
    for _ in range(epochs):
        m.train()
        p, ph = m(X_t, Xp_t)
        mono = torch.relu(-m.a[0]) + torch.relu(-m.a[3])
        loss = mse(p, y_n) + 0.01 * mse(p, ph) + 0.1 * mono
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m

# ── CV predictions ────────────────────────────────────────────────────────────
cv_preds = np.zeros(len(y_train))
fold_r2  = []

for fold, (tr, val) in enumerate(skf.split(X_nn_train, strat_bins_train)):
    sc      = StandardScaler()
    Xf_tr   = torch.from_numpy(sc.fit_transform(X_nn_train[tr]).astype(np.float32))
    Xf_val  = torch.from_numpy(sc.transform(X_nn_train[val]).astype(np.float32))
    ym = float(y_train[tr].mean())
    ys = float(y_train[tr].std())
    yn = torch.tensor((y_train[tr] - ym) / ys, dtype=torch.float32)
    lC = init_logC(Xp_train[tr], ym)
    torch.manual_seed(42)
    m = train_model(Xf_tr, yn, torch.from_numpy(Xp_train[tr]),
                    ym, ys, lC, epochs=3000)
    m.eval()
    with torch.no_grad():
        pv, _ = m(Xf_val, torch.from_numpy(Xp_train[val]))
    cv_preds[val] = pv.numpy() * ys + ym
    fold_r2.append(r2_score(y_train[val], cv_preds[val]))
    print(f"Fold {fold+1}: R²={fold_r2[-1]:.4f}")

# ── Final model ───────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_tr_sc = torch.from_numpy(scaler.fit_transform(X_nn_train).astype(np.float32))
X_te_sc = torch.from_numpy(scaler.transform(X_nn_test).astype(np.float32))
ym_f    = float(y_train.mean())
ys_f    = float(y_train.std())
yn_f    = torch.tensor((y_train - ym_f) / ys_f, dtype=torch.float32)

torch.manual_seed(42)
model = train_model(X_tr_sc, yn_f, torch.from_numpy(Xp_train),
                    ym_f, ys_f, init_logC(Xp_train, ym_f), epochs=5000)
model.eval()
with torch.no_grad():
    p_te, _ = model(X_te_sc, torch.from_numpy(Xp_test))

y_te_pred = p_te.numpy() * ys_f + ym_f
cv_r2     = r2_score(y_train, cv_preds)
cv_rmse   = np.sqrt(mean_squared_error(y_train, cv_preds))
te_r2     = r2_score(y_test, y_te_pred)
te_rmse   = np.sqrt(mean_squared_error(y_test, y_te_pred))

print(f"\nCV R²={cv_r2:.4f}  RMSE={cv_rmse:.2f} cm⁻¹")
print(f"Test R²={te_r2:.4f}  RMSE={te_rmse:.2f} cm⁻¹")

def cc(nu): return '#e74c3c' if nu > 2000 else '#2980b9'
c_tr = [cc(v) for v in y_train]
c_te = [cc(v) for v in y_test]
lim  = (min(y.min(), cv_preds.min(), y_te_pred.min()) - 20,
        max(y.max(), cv_preds.max(), y_te_pred.max()) + 20)

legend_el = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#2980b9', markersize=9,
           label='Carbonyl (ν < 2000 cm⁻¹)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=9,
           label='Isocyanate (ν > 2000 cm⁻¹)'),
]

res_cv = cv_preds - y_train
res_te = y_te_pred - y_test

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1a — Pred vs Exp (CV train)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_train, cv_preds, c=c_tr, alpha=0.7, s=40,
           edgecolors='white', lw=0.4)
ax.plot(lim, lim, 'k--', lw=1.2)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel('ν(C=O) experimental (cm⁻¹)', fontsize=12)
ax.set_ylabel('ν(C=O) predicted (cm⁻¹)', fontsize=12)
ax.text(0.05, 0.93, f'R² = {cv_r2:.4f}\nRMSE = {cv_rmse:.1f} cm⁻¹',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(handles=legend_el, fontsize=10, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('pinn_pred_cv.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1b — Pred vs Exp (test)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_test, y_te_pred, c=c_te, alpha=0.7, s=40,
           edgecolors='white', lw=0.4)
ax.plot(lim, lim, 'k--', lw=1.2)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel('ν(C=O) experimental (cm⁻¹)', fontsize=12)
ax.set_ylabel('ν(C=O) predicted (cm⁻¹)', fontsize=12)
ax.text(0.05, 0.93, f'R² = {te_r2:.4f}\nRMSE = {te_rmse:.1f} cm⁻¹',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(handles=legend_el, fontsize=10, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('pinn_pred_test.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2a — Residuals vs Prediction
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(cv_preds, res_cv, c=c_tr, alpha=0.65, s=35,
           edgecolors='white', lw=0.3)
ax.axhline(0, color='k', lw=1.2, ls='--')
ax.set_xlabel('ν(C=O) predicted (cm⁻¹)', fontsize=12)
ax.set_ylabel('Residual (pred − exp) (cm⁻¹)', fontsize=12)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('pinn_residuals.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2b — Residual distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
xr = np.linspace(min(res_cv.min(), res_te.min()),
                 max(res_cv.max(), res_te.max()), 200)
ax.hist(res_cv, bins=25, alpha=0.6, color='#2980b9', density=True, label='CV train')
ax.hist(res_te, bins=15, alpha=0.6, color='#e74c3c', density=True, label='Test')
mu, sig = np.mean(res_cv), np.std(res_cv)
ax.plot(xr, stats.norm.pdf(xr, mu, sig), 'k-', lw=1.5, label='Normal fit (CV)')
ax.set_xlabel('Residual (cm⁻¹)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('pinn_residuals_hist.png', dpi=200, bbox_inches='tight')
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
plt.savefig('pinn_qq.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Learned physical coefficients
# ══════════════════════════════════════════════════════════════════════════════
labels_phys = [
    '(r−d)⁻³   Badger',
    'cos(θ)    Bent',
    'cos²(φ)   Dihedral',
    'BO⁰·⁷⁵    Gordy',
    '1/π*      π* antibonding',
    'Δq/r²     Local Stark',
    'Epar      Field Stark',
]
coefs    = model.a.detach().numpy()
colors_c = ['#27ae60' if c > 0 else '#e74c3c' for c in coefs]
abs_max  = np.abs(coefs).max()

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(labels_phys, coefs, color=colors_c, edgecolor='white', linewidth=0.6)
ax.axvline(0, color='k', lw=1.0)
ax.set_xlabel('Learned coefficient  aᵢ', fontsize=12)
ax.set_xlim(-(abs_max * 1.30), abs_max * 1.30)
ax.grid(True, axis='x', alpha=0.3)
for bar, val in zip(bars, coefs):
    offset = abs_max * 0.02
    xpos = val + offset if val >= 0 else val - offset
    ha   = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{val:+.4f}', va='center', ha=ha, fontsize=9)
plt.tight_layout()
plt.savefig('pinn_physics_coefs.png', dpi=200, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — R² per fold
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
cols_f = ['#e74c3c' if v < 0.7 else '#2980b9' for v in fold_r2]
bars = ax.bar([f'Fold {i+1}' for i in range(5)], fold_r2,
              color=cols_f, edgecolor='white', linewidth=0.8)
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
plt.savefig('pinn_cv_folds.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nSaved figures:")
for f in ['pinn_pred_cv.png', 'pinn_pred_test.png', 'pinn_residuals.png',
          'pinn_residuals_hist.png', 'pinn_qq.png',
          'pinn_physics_coefs.png', 'pinn_cv_folds.png']:
    print(f"  {f}")
