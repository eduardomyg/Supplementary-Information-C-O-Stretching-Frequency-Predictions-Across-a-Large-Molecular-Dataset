import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Datos ──────────────────────────────────────────────────────────────────────
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
y_train,     y_test    = y[idx_train],   y[idx_test]

strat_bins_train = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

a_init = np.array([0.1, -0.1, 0.05, 0.1, 0.05, 0.0, 0.0], dtype=np.float32)

def init_logC(Xp_arr, y_mean):
    k_avg = float(np.mean(np.abs((Xp_arr * a_init).sum(axis=1))))
    return float(np.log(y_mean / (np.sqrt(k_avg) + 1e-8)))


# ── Arquitectura PINN parametrizada ───────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, n_in, y_mean, y_std, logC_init, hidden, dropout):
        super().__init__()
        layers = []
        in_dim = n_in
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net  = nn.Sequential(*layers)
        self.a    = nn.Parameter(torch.tensor([0.1, -0.1, 0.05, 0.1, 0.05, 0.0, 0.0]))
        self.logC = nn.Parameter(torch.tensor(logC_init))
        self.register_buffer('y_mean', torch.tensor(float(y_mean)))
        self.register_buffer('y_std',  torch.tensor(float(y_std)))

    def physics(self, xp):
        k = (self.a * xp).sum(dim=1)
        return (self.logC.exp() * k.abs().sqrt() - self.y_mean) / self.y_std

    def forward(self, x, xp):
        return self.net(x).squeeze(), self.physics(xp)


def train_model(X_t, y_n, Xp_t, y_mean, y_std, logC_init,
                hidden, dropout, lr, weight_decay, lam_phys, lam_mono, epochs):
    m   = PINN(X_t.shape[1], y_mean, y_std, logC_init, hidden, dropout)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()
    for _ in range(epochs):
        m.train()
        p, ph = m(X_t, Xp_t)
        mono = torch.relu(-m.a[0]) + torch.relu(-m.a[3])
        loss = mse(p, y_n) + lam_phys * mse(p, ph) + lam_mono * mono
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m


def cv_r2_pinn(hidden, dropout, lr, weight_decay, lam_phys, lam_mono, epochs):
    scores = []
    for tr, val in skf.split(X_nn_train, strat_bins_train):
        sc      = StandardScaler()
        Xf_tr   = torch.from_numpy(sc.fit_transform(X_nn_train[tr]).astype(np.float32))
        Xf_val  = torch.from_numpy(sc.transform(X_nn_train[val]).astype(np.float32))
        Xpf_tr  = torch.from_numpy(Xp_train[tr])
        Xpf_val = torch.from_numpy(Xp_train[val])
        ym = float(y_train[tr].mean())
        ys = float(y_train[tr].std())
        yn = torch.tensor((y_train[tr] - ym) / ys)
        lC = init_logC(Xp_train[tr], ym)
        torch.manual_seed(42)
        m = train_model(Xf_tr, yn, Xpf_tr, ym, ys, lC,
                        hidden, dropout, lr, weight_decay,
                        lam_phys, lam_mono, epochs)
        m.eval()
        with torch.no_grad():
            pv, _ = m(Xf_val, Xpf_val)
        scores.append(r2_score(y_train[val], pv.numpy() * ys + ym))
    return float(np.mean(scores))


# ── Optuna ────────────────────────────────────────────────────────────────────
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden   = [trial.suggest_int(f'h{i}', 8, 64) for i in range(n_layers)]
    dropout  = trial.suggest_float('dropout', 0.1, 0.5)
    lr       = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd       = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    lp       = trial.suggest_float('lam_phys', 1e-3, 0.1, log=True)
    lm       = trial.suggest_float('lam_mono', 0.01, 1.0, log=True)
    return cv_r2_pinn(hidden, dropout, lr, wd, lp, lm, epochs=1500)


study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

bp = study.best_params
best_hidden = [bp[f'h{i}'] for i in range(bp['n_layers'])]

print(f"\nMejor CV R² (Optuna, 1500 ep): {study.best_value:.4f}")
print(f"Arquitectura: {best_hidden}  dropout={bp['dropout']:.3f}")
print(f"lr={bp['lr']:.5f}  weight_decay={bp['weight_decay']:.5f}")
print(f"λ_phys={bp['lam_phys']:.4f}  λ_mono={bp['lam_mono']:.4f}")

# ── CV final con parámetros óptimos y más épocas ──────────────────────────────
cv_preds = np.zeros(len(y_train))

for fold, (tr, val) in enumerate(skf.split(X_nn_train, strat_bins_train)):
    sc      = StandardScaler()
    Xf_tr   = torch.from_numpy(sc.fit_transform(X_nn_train[tr]).astype(np.float32))
    Xf_val  = torch.from_numpy(sc.transform(X_nn_train[val]).astype(np.float32))
    Xpf_tr  = torch.from_numpy(Xp_train[tr])
    Xpf_val = torch.from_numpy(Xp_train[val])
    ym = float(y_train[tr].mean())
    ys = float(y_train[tr].std())
    yn = torch.tensor((y_train[tr] - ym) / ys)
    lC = init_logC(Xp_train[tr], ym)
    torch.manual_seed(42)
    m = train_model(Xf_tr, yn, Xpf_tr, ym, ys, lC,
                    best_hidden, bp['dropout'], bp['lr'], bp['weight_decay'],
                    bp['lam_phys'], bp['lam_mono'], epochs=3000)
    m.eval()
    with torch.no_grad():
        pv, _ = m(Xf_val, Xpf_val)
    cv_preds[val] = pv.numpy() * ys + ym
    print(f"  Fold {fold+1}: R²={r2_score(y_train[val], cv_preds[val]):.4f}")

# ── Modelo final sobre todo el training ───────────────────────────────────────
scaler  = StandardScaler()
X_tr_sc = torch.from_numpy(scaler.fit_transform(X_nn_train).astype(np.float32))
X_te_sc = torch.from_numpy(scaler.transform(X_nn_test).astype(np.float32))
ym_full = float(y_train.mean())
ys_full = float(y_train.std())
yn_full = torch.tensor((y_train - ym_full) / ys_full)
lC_full = init_logC(Xp_train, ym_full)

torch.manual_seed(42)
model_final = train_model(
    X_tr_sc, yn_full, torch.from_numpy(Xp_train),
    ym_full, ys_full, lC_full,
    best_hidden, bp['dropout'], bp['lr'], bp['weight_decay'],
    bp['lam_phys'], bp['lam_mono'], epochs=5000
)

model_final.eval()
with torch.no_grad():
    p_te, _ = model_final(X_te_sc, torch.from_numpy(Xp_test))

y_te_pred = p_te.numpy() * ys_full + ym_full

cv_r2   = r2_score(y_train, cv_preds)
cv_rmse = np.sqrt(mean_squared_error(y_train, cv_preds))
te_r2   = r2_score(y_test, y_te_pred)
te_rmse = np.sqrt(mean_squared_error(y_test, y_te_pred))

print()
print("── Resultados finales ───────────────────────────────────")
print(f"  CV R²   (5-fold): {cv_r2:.4f}  RMSE: {cv_rmse:.2f} cm⁻¹")
print(f"  Test R² (20%):    {te_r2:.4f}  RMSE: {te_rmse:.2f} cm⁻¹")
print()
print("  Coeficientes físicos aprendidos:")
labels = ['(r−d)⁻³  Badger', 'cos(θ)   Bent  ', 'cos²(φ)  Diedro',
          'BO⁰·⁷⁵   Gordy ', '1/π*     π*    ',
          'Δq/r²    Stark L', 'Epar     Stark F']
for lbl, ai in zip(labels, model_final.a.tolist()):
    print(f"    {lbl}: {ai:+.4f}")
print()
print("── Comparación ──────────────────────────────────────────")
print(f"  {'Modelo':<18} {'CV R²':>8} {'Test R²':>8} {'Test RMSE':>12}")
print(f"  {'RF (Optuna)':<18} {'0.8357':>8} {'0.9826':>8} {'16.5 cm⁻¹':>12}")
print(f"  {'PINN (Optuna)':<18} {cv_r2:>8.4f} {te_r2:>8.4f} {te_rmse:>11.2f} cm⁻¹")
