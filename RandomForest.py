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

def train_random_forest(X_train, y_train):
    # Определяем параметры для поиска
    param_grid = {
        'n_estimators': [10, 30, 50, 100, 125, 150, 175, 200],
        'max_depth': [None] + list(range(3, 11)),
        'min_samples_split': [2, 5, 7, 8, 9, 10],
        'min_samples_leaf': [1, 3, 5, 7, 8, 9, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }

    # Создаем модель и GridSearchCV
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Запускаем поиск лучших параметров
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

# Обучаем модель с оптимизированными параметрами
best_model = train_random_forest(X_train, y_train)

# Делаем предсказания на тестовом наборе
y_pred = best_model.predict(X_test)

# Оцениваем модель
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Лучшие параметры: {best_model.get_params()}")
print(f"Точность модели: {accuracy:.3f}")
print("Отчет о классификации:")
print(report)