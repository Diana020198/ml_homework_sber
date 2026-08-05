import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
n_obs = 1000
time_index = pd.date_range(start="2020-01-01", periods=n_obs, freq='D')

# Генерация временного ряда
trend = 0.05 * np.arange(n_obs)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n_obs) / 365)
noise = np.random.normal(0, 1, n_obs)
ts_values = trend + seasonality + noise
ts = pd.Series(ts_values, index=time_index)

# Разделение
train_ts = ts.iloc[:int(0.8 * n_obs)]
test_ts = ts.iloc[int(0.8 * n_obs):]

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts.values, label='Полный ряд', alpha=0.6)
plt.axvline(train_ts.index[-1], color='black', linestyle='--', label='Точка отсечения теста')
plt.title('Синтетический временной ряд')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('timeseries_full_plot.png')
plt.show()

# Подготовка для AR (стационаризация путем взятия разности)
train_diff = train_ts.diff().dropna()

# Модель 1: Автокорреляционная (AR) - Исправлено под новые версии библиотеки
ar_lag = 30 
ar_model = AutoReg(endog=train_diff, lags=ar_lag).fit() # Убрали old_names=False

# Модель 2: Экспоненциальное сглаживание (ETS/Holt-Winters)
ets_model = ExponentialSmoothing(
    train_ts, 
    seasonal='add', 
    seasonal_periods=365, 
    trend='add'
).fit()

# Функция для инверсии разностей
def inverse_difference(last_ob, forecast):
    return last_ob + forecast

# Прогнозы
start_idx = len(train_diff)
end_idx = len(train_diff) + len(test_ts) - 1

# # Прогнозируем сразу все изменения (разности), а потом превращаем их обратно в уровни
forecasted_diffs = ar_model.predict(start=len(train_diff), end=len(train_diff) + len(test_ts) - 1)
ar_predictions = [inverse_difference(train_ts.iloc[-1], diff) for diff in forecasted_diffs]

# ETS делает прямой прогноз
ets_forecast = ets_model.forecast(steps=len(test_ts))

# Оценка качества
mae_ar = mean_absolute_error(test_ts, ar_predictions)
rmse_ar = np.sqrt(mean_squared_error(test_ts, ar_predictions))

mae_ets = mean_absolute_error(test_ts, ets_forecast)
rmse_ets = np.sqrt(mean_squared_error(test_ts, ets_forecast))

print(f"AR - MAE: {mae_ar:.4f}, RMSE: {rmse_ar:.4f}")
print(f"ETS - MAE: {mae_ets:.4f}, RMSE: {rmse_ets:.4f}")

# Визуализация прогнозов
plt.figure(figsize=(14, 7))
plt.plot(ts.index, ts.values, label='Истинный ряд', linewidth=2)
plt.plot(test_ts.index, ar_predictions, label='Прогноз AR', marker='o', markersize=3)
plt.plot(test_ts.index, ets_forecast, label='Прогноз ETS', marker='x', markersize=3)
plt.title('Сравнение прогнозов на тестовой выборке')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('timeseries_forecast_comparison.png')
plt.show()