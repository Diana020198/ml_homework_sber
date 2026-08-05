import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)
n_samples, n_features = 1000, 10

# Генерация коррелированных признаков
X_base = np.random.randn(n_samples, n_features)
corr_matrix = np.array([[1 if i==j else 0.5 for j in range(n_features)] for i in range(n_features)])
L = np.linalg.cholesky(corr_matrix)
X = X_base @ L.T 

coefficients = np.random.randn(n_features)
noise = np.random.normal(0, 2, n_samples)
y = X @ coefficients + noise

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение модели
base_lr = LinearRegression()
bagging_model = BaggingRegressor(estimator=base_lr, n_estimators=50, random_state=42, n_jobs=-1)
bagging_model.fit(X_train, y_train)

# Оценка качества
y_pred = bagging_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Регрессия - MSE: {mse:.4f}, R^2: {r2:.4f}")

# Анализ мультиколлинеарности (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = [f"x_{i}" for i in range(n_features)]
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print("\nМультиколлинеарность (VIF):")
print(vif_data.head())

# Анализ остатков
residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(residuals, kde=True, ax=axes[0])
axes[0].set_title('Распределение остатков')
axes[0].set_xlabel('Ошибка')
axes[1].scatter(y_pred, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_title('Остатки vs Предсказанные значения')
axes[1].set_xlabel('Предсказание')
axes[1].set_ylabel('Остаток')
plt.tight_layout()
plt.savefig('regression_residuals.png')
plt.show()

# Важность признаков (коэффициенты базового LR внутри бэггинга)
coefs = np.array([est.coef_ for est in bagging_model.estimators_])
mean_coefs = np.mean(np.abs(coefs), axis=0)
imp_df = pd.DataFrame({'Feature': vif_data['feature'], 'Importance': mean_coefs}).sort_values('Importance', ascending=False)
print("\nВажность признаков:")
print(imp_df.head().to_markdown(index=False))