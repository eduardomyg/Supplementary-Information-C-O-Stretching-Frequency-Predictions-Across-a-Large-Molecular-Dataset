import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Data and model (identical to figures_pinn.py) ────────────────────────────
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
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=strat_bins
)

X_nn_train = X_nn[idx_train]
Xp_train   = Xp[idx_train]
y_train    = y[idx_train]

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_nn_train).astype(np.float32)

a_init = np.array([0.1, -0.1, 0.05, 0.1, 0.05, 0.0, 0.0], dtype=np.float32)

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

ym = float(y_train.mean())
ys = float(y_train.std())
k_avg = float(np.mean(np.abs((Xp_train * a_init).sum(axis=1))))
lC = float(np.log(ym / (np.sqrt(k_avg) + 1e-8)))

torch.manual_seed(42)
model = train_model(
    torch.from_numpy(X_tr_sc),
    torch.tensor((y_train - ym) / ys),
    torch.from_numpy(Xp_train),
    ym, ys, lC, epochs=5000
)
model.eval()

# ── NN-only wrapper for SHAP (Xp fixed as buffer) ────────────────────────────
# DeepExplainer requires a module that receives a single tensor
Xp_tr_t = torch.from_numpy(Xp_train)

class NNWrapper(nn.Module):
    """Exposes only the NN output (no physics module) for SHAP."""
    def __init__(self, pinn, xp_background):
        super().__init__()
        self.pinn = pinn
        self.register_buffer('xp_bg', xp_background)

    def forward(self, x):
        # Use Xp from the batch if sizes match, otherwise use background
        xp = self.xp_bg[:x.shape[0]]
        pred, _ = self.pinn(x, xp)
        return pred.unsqueeze(1)

wrapper = NNWrapper(model, Xp_tr_t)
wrapper.eval()

X_tr_t  = torch.from_numpy(X_tr_sc)
background = X_tr_t[:50]   # fondo para DeepExplainer

explainer   = shap.DeepExplainer(wrapper, background)
shap_values = explainer.shap_values(X_tr_t)

# shap_values puede ser lista (clasificación) o array (regresión)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({'feature': NN_FEATURES, 'shap': mean_abs_shap})
imp_df = imp_df.sort_values('shap', ascending=False)

print("SHAP — PINN neural network (mean |SHAP| importance):")
print(imp_df.to_string(index=False))

# ── Beeswarm PINN ─────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_tr_sc, feature_names=NN_FEATURES,
                  show=False, max_display=10)
plt.tight_layout()
plt.savefig('pinn_shap_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nSaved figure: pinn_shap_beeswarm.png")
