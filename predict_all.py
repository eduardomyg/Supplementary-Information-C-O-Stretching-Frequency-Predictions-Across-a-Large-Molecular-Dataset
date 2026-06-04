import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS_RF   = 100
N_TRIALS_GBM  = 100
N_TRIALS_PINN = 40

# ── Datos ──────────────────────────────────────────────────────────────────────
data = pd.read_csv('CO_dataset_final_ULTIMO.csv')
data['dihedral_real'] = data['dihedral_real'].fillna(90.0)

FEATURES = [
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
        (qO - qC) / r**2,
        Ef,
    ]).astype(np.float32)

X    = data[FEATURES].to_numpy(dtype=float)
X_nn = data[FEATURES].to_numpy(dtype=np.float32)
Xp   = build_Xp(data)
y    = data['nu_CO_exp'].to_numpy(dtype=float)

# ── Split identico al de todos los scripts ────────────────────────────────────
strat_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
idx_all    = np.arange(len(y))
idx_train, idx_test = train_test_split(
    idx_all, test_size=0.2, random_state=42, stratify=strat_bins
)

X_train,    X_test    = X[idx_train],    X[idx_test]
X_nn_train, X_nn_test = X_nn[idx_train], X_nn[idx_test]
Xp_train,   Xp_test   = Xp[idx_train],  Xp[idx_test]
y_train,    y_test    = y[idx_train],    y[idx_test]

strat_bins_train = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# arrays de predicciones finales (orden del CSV original)
pred_rf   = np.full(len(y), np.nan)
pred_gbm  = np.full(len(y), np.nan)
pred_pinn = np.full(len(y), np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("── Random Forest ────────────────────────────────────────")

def rf_cv(params):
    pipe = Pipeline([('sc', StandardScaler()),
                     ('rf', RandomForestRegressor(**params, random_state=42, n_jobs=-1))])
    scores = []
    for tr, val in skf.split(X_train, strat_bins_train):
        m = clone(pipe); m.fit(X_train[tr], y_train[tr])
        scores.append(r2_score(y_train[val], m.predict(X_train[val])))
    return float(np.mean(scores))

def rf_objective(trial):
    return rf_cv({
        'n_estimators':      trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':         trial.suggest_int('max_depth', 3, 20),
        'max_features':      trial.suggest_float('max_features', 0.2, 1.0),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'bootstrap':         trial.suggest_categorical('bootstrap', [True, False]),
    })

rf_study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(rf_objective, n_trials=N_TRIALS_RF, show_progress_bar=True)
print(f"  Mejor CV R²: {rf_study.best_value:.4f}  params: {rf_study.best_params}")

best_rf = Pipeline([('sc', StandardScaler()),
                    ('rf', RandomForestRegressor(**rf_study.best_params,
                                                  random_state=42, n_jobs=-1))])

# predicciones out-of-fold para train
oof_rf = np.zeros(len(y_train))
for tr, val in skf.split(X_train, strat_bins_train):
    m = clone(best_rf); m.fit(X_train[tr], y_train[tr])
    oof_rf[val] = m.predict(X_train[val])

# modelo final para test
best_rf.fit(X_train, y_train)
pred_rf[idx_train] = oof_rf
pred_rf[idx_test]  = best_rf.predict(X_test)

print(f"  OOF R²={r2_score(y_train, oof_rf):.4f}  "
      f"Test R²={r2_score(y_test, pred_rf[idx_test]):.4f}  "
      f"Test RMSE={np.sqrt(mean_squared_error(y_test, pred_rf[idx_test])):.2f} cm⁻¹")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GBM
# ══════════════════════════════════════════════════════════════════════════════
print("── GBM ──────────────────────────────────────────────────")

def gbm_cv(params):
    pipe = Pipeline([('sc', StandardScaler()),
                     ('gbm', GradientBoostingRegressor(**params, random_state=42))])
    scores = []
    for tr, val in skf.split(X_train, strat_bins_train):
        m = clone(pipe); m.fit(X_train[tr], y_train[tr])
        scores.append(r2_score(y_train[val], m.predict(X_train[val])))
    return float(np.mean(scores))

def gbm_objective(trial):
    return gbm_cv({
        'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
        'max_depth':        trial.suggest_int('max_depth', 2, 6),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':     trial.suggest_float('max_features', 0.3, 1.0),
    })

gbm_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
gbm_study.optimize(gbm_objective, n_trials=N_TRIALS_GBM, show_progress_bar=True)
print(f"  Mejor CV R²: {gbm_study.best_value:.4f}  params: {gbm_study.best_params}")

best_gbm = Pipeline([('sc', StandardScaler()),
                     ('gbm', GradientBoostingRegressor(**gbm_study.best_params,
                                                        random_state=42))])

oof_gbm = np.zeros(len(y_train))
for tr, val in skf.split(X_train, strat_bins_train):
    m = clone(best_gbm); m.fit(X_train[tr], y_train[tr])
    oof_gbm[val] = m.predict(X_train[val])

best_gbm.fit(X_train, y_train)
pred_gbm[idx_train] = oof_gbm
pred_gbm[idx_test]  = best_gbm.predict(X_test)

print(f"  OOF R²={r2_score(y_train, oof_gbm):.4f}  "
      f"Test R²={r2_score(y_test, pred_gbm[idx_test]):.4f}  "
      f"Test RMSE={np.sqrt(mean_squared_error(y_test, pred_gbm[idx_test])):.2f} cm⁻¹")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PINN
# ══════════════════════════════════════════════════════════════════════════════
print("── PINN ─────────────────────────────────────────────────")

a_init = np.array([0.1, -0.1, 0.05, 0.1, 0.05, 0.0, 0.0], dtype=np.float32)

def init_logC(Xp_arr, y_mean):
    k_avg = float(np.mean(np.abs((Xp_arr * a_init).sum(axis=1))))
    return float(np.log(y_mean / (np.sqrt(k_avg) + 1e-8)))


class PINN(nn.Module):
    def __init__(self, n_in, y_mean, y_std, logC_init, hidden, dropout):
        super().__init__()
        layers, in_dim = [], n_in
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


def train_pinn(X_t, y_n, Xp_t, y_mean, y_std, logC_init,
               hidden, dropout, lr, weight_decay, lam_phys, lam_mono, epochs):
    m   = PINN(X_t.shape[1], y_mean, y_std, logC_init, hidden, dropout)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()
    for _ in range(epochs):
        m.train()
        p, ph = m(X_t, Xp_t)
        mono  = torch.relu(-m.a[0]) + torch.relu(-m.a[3])
        loss  = mse(p, y_n) + lam_phys * mse(p, ph) + lam_mono * mono
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m


def pinn_cv_r2(hidden, dropout, lr, weight_decay, lam_phys, lam_mono, epochs):
    scores = []
    for tr, val in skf.split(X_nn_train, strat_bins_train):
        sc      = StandardScaler()
        Xf_tr   = torch.from_numpy(sc.fit_transform(X_nn_train[tr]).astype(np.float32))
        Xf_val  = torch.from_numpy(sc.transform(X_nn_train[val]).astype(np.float32))
        Xpf_tr  = torch.from_numpy(Xp_train[tr])
        Xpf_val = torch.from_numpy(Xp_train[val])
        ym = float(y_train[tr].mean())
        ys = float(y_train[tr].std())
        yn = torch.tensor((y_train[tr] - ym) / ys, dtype=torch.float32)
        lC = init_logC(Xp_train[tr], ym)
        torch.manual_seed(42)
        m = train_pinn(Xf_tr, yn, Xpf_tr, ym, ys, lC,
                       hidden, dropout, lr, weight_decay, lam_phys, lam_mono, epochs)
        m.eval()
        with torch.no_grad():
            pv, _ = m(Xf_val, Xpf_val)
        scores.append(r2_score(y_train[val], pv.numpy() * ys + ym))
    return float(np.mean(scores))


def pinn_objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden   = [trial.suggest_int(f'h{i}', 8, 64) for i in range(n_layers)]
    dropout  = trial.suggest_float('dropout', 0.1, 0.5)
    lr       = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd       = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    lp       = trial.suggest_float('lam_phys', 1e-3, 0.1, log=True)
    lm       = trial.suggest_float('lam_mono', 0.01, 1.0, log=True)
    return pinn_cv_r2(hidden, dropout, lr, wd, lp, lm, epochs=1500)


pinn_study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
pinn_study.optimize(pinn_objective, n_trials=N_TRIALS_PINN, show_progress_bar=True)

bp = pinn_study.best_params
best_hidden = [bp[f'h{i}'] for i in range(bp['n_layers'])]
print(f"  Mejor CV R² (Optuna): {pinn_study.best_value:.4f}")
print(f"  Arquitectura: {best_hidden}  dropout={bp['dropout']:.3f}")

# CV final con mas epocas para train predictions
oof_pinn = np.zeros(len(y_train))
for fold, (tr, val) in enumerate(skf.split(X_nn_train, strat_bins_train)):
    sc      = StandardScaler()
    Xf_tr   = torch.from_numpy(sc.fit_transform(X_nn_train[tr]).astype(np.float32))
    Xf_val  = torch.from_numpy(sc.transform(X_nn_train[val]).astype(np.float32))
    Xpf_tr  = torch.from_numpy(Xp_train[tr])
    Xpf_val = torch.from_numpy(Xp_train[val])
    ym = float(y_train[tr].mean())
    ys = float(y_train[tr].std())
    yn = torch.tensor((y_train[tr] - ym) / ys, dtype=torch.float32)
    lC = init_logC(Xp_train[tr], ym)
    torch.manual_seed(42)
    m = train_pinn(Xf_tr, yn, Xpf_tr, ym, ys, lC,
                   best_hidden, bp['dropout'], bp['lr'], bp['weight_decay'],
                   bp['lam_phys'], bp['lam_mono'], epochs=3000)
    m.eval()
    with torch.no_grad():
        pv, _ = m(Xf_val, Xpf_val)
    oof_pinn[val] = pv.numpy() * ys + ym
    print(f"    Fold {fold+1}: R²={r2_score(y_train[val], oof_pinn[val]):.4f}")

# modelo final sobre todo el training para predecir test
scaler_pinn  = StandardScaler()
X_tr_sc = torch.from_numpy(scaler_pinn.fit_transform(X_nn_train).astype(np.float32))
X_te_sc = torch.from_numpy(scaler_pinn.transform(X_nn_test).astype(np.float32))
ym_full = float(y_train.mean())
ys_full = float(y_train.std())
yn_full = torch.tensor((y_train - ym_full) / ys_full, dtype=torch.float32)
lC_full = init_logC(Xp_train, ym_full)

torch.manual_seed(42)
model_final = train_pinn(
    X_tr_sc, yn_full, torch.from_numpy(Xp_train),
    ym_full, ys_full, lC_full,
    best_hidden, bp['dropout'], bp['lr'], bp['weight_decay'],
    bp['lam_phys'], bp['lam_mono'], epochs=5000
)

model_final.eval()
with torch.no_grad():
    p_te, _ = model_final(X_te_sc, torch.from_numpy(Xp_test))

pinn_test_preds = p_te.numpy() * ys_full + ym_full

pred_pinn[idx_train] = oof_pinn
pred_pinn[idx_test]  = pinn_test_preds

print(f"  OOF R²={r2_score(y_train, oof_pinn):.4f}  "
      f"Test R²={r2_score(y_test, pinn_test_preds):.4f}  "
      f"Test RMSE={np.sqrt(mean_squared_error(y_test, pinn_test_preds)):.2f} cm⁻¹")


# ══════════════════════════════════════════════════════════════════════════════
# Guardar CSV
# ══════════════════════════════════════════════════════════════════════════════
split_col = np.where(np.isin(idx_all, idx_test), 'test', 'train')

out = pd.DataFrame({
    'Molecule':   data['Molecule'].values,
    'nu_CO_exp':  y,
    'split':      split_col,
    'RF_pred':    np.round(pred_rf,   2),
    'GBM_pred':   np.round(pred_gbm,  2),
    'PINN_pred':  np.round(pred_pinn, 2),
})

out.to_csv('predictions_all.csv', index=False, float_format='%.4f')
print("\n── Guardado: predictions_all.csv ────────────────────────")
print(out.head(10).to_string(index=False))
