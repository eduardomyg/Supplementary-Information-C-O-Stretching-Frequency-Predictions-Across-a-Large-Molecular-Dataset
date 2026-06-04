import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('CO_dataset_final_ULTIMO.csv')
data['dihedral_real'] = data['dihedral_real'].fillna(data['dihedral_real'].median())

DROP = {'Molecule', 'C_index', 'O_index', 'nu_CO_exp'}

feature_cols = [c for c in data.columns if c not in DROP]
X = data[feature_cols].to_numpy(dtype=float)
y = data['nu_CO_exp'].to_numpy(dtype=float)

strat_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat_bins
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=500, max_features=0.5,
                            min_samples_leaf=2, random_state=42)
rf.fit(X_train_sc, y_train)

explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_sc)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("SHAP — importancia media absoluta (entrenamiento):")
print(importance_df.to_string(index=False))

# Figura 1: bar plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e74c3c' if importance_df.iloc[i]['mean_abs_shap'] > 20
          else '#3498db' for i in range(len(importance_df))]
ax.barh(importance_df['feature'][::-1], importance_df['mean_abs_shap'][::-1],
        color=colors[::-1])
ax.set_xlabel('Mean |SHAP value| (cm⁻¹)')
ax.set_title('Feature importance — Random Forest (SHAP)')
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150)
plt.close()

# Figura 2: beeswarm (distribución de SHAP por feature)
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_train_sc, feature_names=feature_cols,
                  show=False, max_display=26)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150)
plt.close()

print("\nFiguras guardadas: shap_bar.png, shap_beeswarm.png")
