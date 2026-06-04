import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.lines import Line2D

FEATURES = [
    'bond_CO', 'sp_ratio', 'angle_O_C_X', 'HOMO', 'lowdin_O',
    'charge_C', 'nu_CO_DFT', 'lowdin_C', 'HOMO_1', 'dipole',
]

FEATURE_LABELS = [
    'bond$_{CO}$', 'sp ratio', 'angle$_{OCX}$', 'HOMO', 'Löwdin$_O$',
    'charge$_C$', 'ν$_{CO}^{DFT}$', 'Löwdin$_C$', 'HOMO−1', 'dipole',
]

MODEL_COLORS = {
    'RF':   '#2980b9',
    'GBM':  '#27ae60',
    'PINN': '#e67e22',
}

D_CO = 0.68

# ── Data ──────────────────────────────────────────────────────────────────────
data = pd.read_csv('CO_dataset_final_ULTIMO.csv')
data['dihedral_real'] = data['dihedral_real'].fillna(90.0)

X = data[FEATURES].to_numpy(dtype=float)
y = data['nu_CO_exp'].to_numpy(dtype=float)

strat_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat_bins
)

# ── Load predictions (from predictions_all.csv) ───────────────────────────────
preds    = pd.read_csv('predictions_all.csv')
train_df = preds[preds['split'] == 'train'].reset_index(drop=True)
test_df  = preds[preds['split'] == 'test'].reset_index(drop=True)

MODEL_COLS = {'RF': 'RF_pred', 'GBM': 'GBM_pred', 'PINN': 'PINN_pred'}
MODELS     = ['RF', 'GBM', 'PINN']

# ── Performance metrics ───────────────────────────────────────────────────────
metrics = {}
for m, col in MODEL_COLS.items():
    r2_cv   = r2_score(train_df['nu_CO_exp'], train_df[col])
    rmse_cv = np.sqrt(mean_squared_error(train_df['nu_CO_exp'], train_df[col]))
    r2_t    = r2_score(test_df['nu_CO_exp'],  test_df[col])
    rmse_t  = np.sqrt(mean_squared_error(test_df['nu_CO_exp'],  test_df[col]))
    metrics[m] = dict(r2_cv=r2_cv, rmse_cv=rmse_cv, r2_test=r2_t, rmse_test=rmse_t)

# ── LaTeX table ───────────────────────────────────────────────────────────────
print("\n% ── Performance table ──────────────────────────────────────────────")
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{lcccc}")
print("\\hline")
print("Model & $R^2_{\\mathrm{CV}}$ & $\\mathrm{RMSE_{CV}}$ (cm$^{-1}$)"
      " & $R^2_{\\mathrm{test}}$ & $\\mathrm{RMSE_{test}}$ (cm$^{-1}$) \\\\")
print("\\hline")
for m in MODELS:
    v = metrics[m]
    print(f"{m} & {v['r2_cv']:.4f} & {v['rmse_cv']:.1f}"
          f" & {v['r2_test']:.4f} & {v['rmse_test']:.1f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{Cross-validation and test-set performance for all models.}")
print("\\label{tab:performance}")
print("\\end{table}\n")

# ── Colour helpers ────────────────────────────────────────────────────────────
def class_color(nu):
    return '#e74c3c' if nu > 2000 else '#2980b9'

legend_class = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9',
           markersize=8, label='Carbonyl (ν < 2000 cm⁻¹)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=8, label='Isocyanate (ν > 2000 cm⁻¹)'),
]

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — 1×3 scatter: CV train + test, all models
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

model_titles = {'RF': 'Random Forest', 'GBM': 'Gradient Boosting', 'PINN': 'PINN'}

for ax, m in zip(axes, MODELS):
    col  = MODEL_COLS[m]
    y_tr = train_df['nu_CO_exp'].values
    yp_tr = train_df[col].values
    y_te  = test_df['nu_CO_exp'].values
    yp_te = test_df[col].values

    nu_min = min(y_tr.min(), yp_tr.min(), y_te.min(), yp_te.min()) - 20
    nu_max = max(y_tr.max(), yp_tr.max(), y_te.max(), yp_te.max()) + 20

    c_tr = [class_color(v) for v in y_tr]
    c_te = [class_color(v) for v in y_te]

    v = metrics[m]
    ax.scatter(y_tr, yp_tr, c=c_tr, alpha=0.40, s=22, edgecolors='none')
    ax.scatter(y_te, yp_te, c=c_te, alpha=0.90, s=35,
               edgecolors='k', linewidths=0.5)
    ax.plot([nu_min, nu_max], [nu_min, nu_max], 'k--', lw=1.2)
    ax.set_xlim(nu_min, nu_max)
    ax.set_ylim(nu_min, nu_max)
    ax.set_xlabel('ν(C=O) experimental (cm⁻¹)', fontsize=11)
    ax.set_ylabel('ν(C=O) predicted (cm⁻¹)', fontsize=11)
    ax.set_title(model_titles[m], fontsize=12, fontweight='bold')
    ax.text(0.05, 0.93,
            f"CV:   R²={v['r2_cv']:.3f}  RMSE={v['rmse_cv']:.1f}\n"
            f"Test: R²={v['r2_test']:.3f}  RMSE={v['rmse_test']:.1f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.grid(True, alpha=0.25)

legend_scatter = legend_class + [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, alpha=0.4, label='CV train'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, markeredgecolor='k', markeredgewidth=0.8, label='Test'),
]
fig.legend(handles=legend_scatter, loc='lower center', ncol=4,
           fontsize=10, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
plt.savefig('comparison_scatter.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: comparison_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# Train models for SHAP (best hyperparams, no Optuna)
# ══════════════════════════════════════════════════════════════════════════════
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_tr_f32 = X_tr_sc.astype(np.float32)

# RF
RF_PARAMS = dict(n_estimators=342, max_depth=17, max_features=0.5554,
                 min_samples_leaf=1, min_samples_split=3, bootstrap=False)
rf = RandomForestRegressor(**RF_PARAMS, random_state=42, n_jobs=-1)
rf.fit(X_tr_sc, y_train)
shap_rf = np.abs(shap.TreeExplainer(rf).shap_values(X_tr_sc)).mean(axis=0)
print("RF SHAP done.")

# GBM
GBM_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=3, max_features=0.7)
gbm = GradientBoostingRegressor(**GBM_PARAMS, random_state=42)
gbm.fit(X_tr_sc, y_train)
shap_gbm = np.abs(shap.TreeExplainer(gbm).shap_values(X_tr_sc)).mean(axis=0)
print("GBM SHAP done.")

# PINN — DeepExplainer on the NN component
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

data_train = data.iloc[
    train_test_split(np.arange(len(data)), test_size=0.2,
                     random_state=42,
                     stratify=pd.qcut(y, q=5, labels=False,
                                      duplicates='drop'))[0]
]
Xp_train = build_Xp(data_train)

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

ym = float(y_train.mean())
ys = float(y_train.std())
a0 = np.array([0.1, -0.1, 0.05, 0.1, 0.05, 0.0, 0.0], dtype=np.float32)
k_avg = float(np.mean(np.abs((Xp_train * a0).sum(axis=1))))
lC = float(np.log(ym / (np.sqrt(k_avg) + 1e-8)))

X_t  = torch.from_numpy(X_tr_f32)
Xp_t = torch.from_numpy(Xp_train)
yn   = torch.tensor((y_train - ym) / ys, dtype=torch.float32)

pinn = PINN(X_t.shape[1], ym, ys, lC)
opt  = torch.optim.Adam(pinn.parameters(), lr=1e-3, weight_decay=1e-3)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
mse  = nn.MSELoss()
torch.manual_seed(42)
for _ in range(5000):
    pinn.train()
    p, ph = pinn(X_t, Xp_t)
    mono  = torch.relu(-pinn.a[0]) + torch.relu(-pinn.a[3])
    loss  = mse(p, yn) + 0.01 * mse(p, ph) + 0.1 * mono
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
    opt.step(); sch.step()
pinn.eval()
print("PINN training done.")

class NNWrapper(nn.Module):
    def __init__(self, model, xp_bg):
        super().__init__()
        self.model = model
        self.register_buffer('xp_bg', xp_bg)
    def forward(self, x):
        xp = self.xp_bg[:x.shape[0]]
        pred, _ = self.model(x, xp)
        return pred.unsqueeze(1)

wrapper    = NNWrapper(pinn, Xp_t)
wrapper.eval()
background = X_t[:50]
sv_pinn    = shap.DeepExplainer(wrapper, background).shap_values(X_t)
if isinstance(sv_pinn, list):
    sv_pinn = sv_pinn[0]
sv_pinn  = np.array(sv_pinn)
if sv_pinn.ndim == 3:
    sv_pinn = sv_pinn[:, :, 0]
shap_pinn = np.abs(sv_pinn).mean(axis=0)
print("PINN SHAP done.")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Grouped SHAP bar chart
# ══════════════════════════════════════════════════════════════════════════════
shap_rf_norm   = shap_rf   / shap_rf.sum()   * 100
shap_gbm_norm  = shap_gbm  / shap_gbm.sum()  * 100
shap_pinn_norm = shap_pinn / shap_pinn.sum()  * 100

shap_data = np.column_stack([shap_rf_norm, shap_gbm_norm, shap_pinn_norm])
order     = np.argsort(shap_data.mean(axis=1))

labels_ord    = np.array(FEATURE_LABELS)[order]
rf_ord        = shap_rf_norm[order]
gbm_ord       = shap_gbm_norm[order]
pinn_ord      = shap_pinn_norm[order]

n   = len(FEATURES)
y_  = np.arange(n)
h   = 0.26

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y_ + h,   rf_ord,   h, color=MODEL_COLORS['RF'],   label='RF',   alpha=0.90)
ax.barh(y_,       gbm_ord,  h, color=MODEL_COLORS['GBM'],  label='GBM',  alpha=0.90)
ax.barh(y_ - h,   pinn_ord, h, color=MODEL_COLORS['PINN'], label='PINN', alpha=0.90)

ax.set_yticks(y_)
ax.set_yticklabels(labels_ord, fontsize=11)
ax.set_xlabel('Relative SHAP importance (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, shap_data.max() * 1.20)
plt.tight_layout()
plt.savefig('comparison_shap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: comparison_shap.png")

print("\nDone. Output files:")
print("  comparison_scatter.png")
print("  comparison_shap.png")
print("  (LaTeX table printed above)")
