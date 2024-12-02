import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv(r'D:\Programming\programms\train.csv')
label_encoder = LabelEncoder()
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        train_df[column] = label_encoder.fit_transform(train_df[column])

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['LeaveOrNot'], axis=1), train_df['LeaveOrNot'], test_size=0.23, random_state=42)

def train_gradient_boosting(X_train, y_train):
    # Определяем параметры для поиска
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5],
        'n_estimators': [10, 50, 60, 70, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 7, 9, 10],
        'min_samples_leaf': [1, 5, 7, 9, 10],
        'max_features': [None, 'log2', 'sqrt'],
        'subsample': [0.5, 0.6, 0.75, 1.0, 1.5],
        'criterion': ['friedman_mse', 'squared_error'],
        'min_weight_fraction_leaf': [0.0, 0.01, 0.05]
    }

    # Создаем модель и GridSearchCV
    model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Запускаем поиск лучших параметров
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

# Обучаем модель с оптимизированными параметрами
best_model = train_gradient_boosting(X_train, y_train)

# Делаем предсказания на тестовом наборе
y_pred = best_model.predict(X_test)

# Оцениваем модель
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Лучшие параметры:", best_model.get_params())
print(f"Точность модели: {accuracy:.3f}")
print("Отчет о классификации:")
print(report)