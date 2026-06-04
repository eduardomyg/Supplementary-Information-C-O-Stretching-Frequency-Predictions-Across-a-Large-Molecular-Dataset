import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

strat_bins_train = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def cv_r2(params):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(**params, random_state=42, n_jobs=-1))
    ])
    scores = []
    for tr, val in skf.split(X_train, strat_bins_train):
        m = clone(pipe)
        m.fit(X_train[tr], y_train[tr])
        scores.append(r2_score(y_train[val], m.predict(X_train[val])))
    return float(np.mean(scores))


def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':        trial.suggest_int('max_depth', 3, 20),
        'max_features':     trial.suggest_float('max_features', 0.2, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split':trial.suggest_int('min_samples_split', 2, 10),
        'bootstrap':        trial.suggest_categorical('bootstrap', [True, False]),
    }
    return cv_r2(params)


study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nMejor CV R²:          {study.best_value:.4f}")
print("Mejores hiperparámetros:")
for k, v in study.best_params.items():
    print(f"  {k:<22} {v}")

# ── Entrenar modelo final con los mejores hiperparámetros ─────────────────────
best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1))
])

cv_scores = []
for tr, val in skf.split(X_train, strat_bins_train):
    m = clone(best_pipe)
    m.fit(X_train[tr], y_train[tr])
    cv_scores.append(r2_score(y_train[val], m.predict(X_train[val])))
cv_scores = np.array(cv_scores)

best_pipe.fit(X_train, y_train)
y_pred    = best_pipe.predict(X_test)
r2_test   = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n── Resultados finales ───────────────────────────────────")
print(f"5-Fold CV R²  (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"CV R² por fold:        {np.round(cv_scores, 4)}")
print(f"Test R²       (20%):   {r2_test:.4f}")
print(f"Test RMSE     (20%):   {rmse_test:.2f} cm⁻¹")
