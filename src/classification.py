import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

print("--- Шаг 1: Генерируем синтетические данные ---")
# Делаем облако точек. random_state=42 нужен, чтобы каждый раз получались одни и те же случайные точки
X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=3.8, center_box=(-6.0, 6.0), random_state=42)

# Разделяем на тренировку (70%) и тест (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Рисуем картинку, чтобы увидеть наши точки глазами
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Класс 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Класс 1')
plt.title('Наши исходные данные')
plt.savefig('blob_visual.png') # Сохраняем картинку в папку ml_project
plt.show()

print("\n--- Шаг 2: Масштабируем данные ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Словарь наших моделей и того, какие винтики (параметры) мы хотим в них крутить
models_to_try = {
    'Логистическая регрессия': (
        LogisticRegression(max_iter=1000),
        {'C': [0.1, 1, 10]}
    ),
    'SVM': (
        SVC(probability=True),
        {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
    ),
    'Дерево решений': (
        DecisionTreeClassifier(),
        {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}
    ),
    'Случайный лес': (
        RandomForestClassifier(),
        {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]}
    )
}

results_list = []
for name, (model, params) in models_to_try.items():
    print(f"\nОбучаем {name}...")
    
    # GridSearchCV берет модель, перебирает ВСЕ комбинации параметров и делает перекрестную проверку
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    predictions_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    auc = roc_auc_score(y_test, predictions_proba)
    gini = 2 * auc - 1
    
    results_list.append({
        'Модель': name,
        'Лучшие параметры': grid_search.best_params_,
        'Джини': round(gini, 4)
    })
    print(f"Найден лучший Gini: {round(gini, 4)}")

final_df = pd.DataFrame(results_list)
print("\n=== ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
print(final_df.to_markdown(index=False))