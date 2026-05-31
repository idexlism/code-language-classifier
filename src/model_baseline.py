"""
Базовая модель для классификации языков программирования.

Использует TF-IDF признаки + классификаторы:
- SVM (Support Vector Machine)
- Random Forest
- Logistic Regression
- Multinomial Naive Bayes
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

from src.utils import MODELS_DIR, save_object, save_results


class BaselineModel:
    """Базовая модель классификации на основе TF-IDF + классификаторы."""
    
    def __init__(self):
        self.models = {
            "SVM": SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, 
                max_depth=20, 
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000, 
                C=1.0, 
                solver='lbfgs',
                random_state=42
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            "NaiveBayes": MultinomialNB(alpha=1.0)
        }
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def train(self, X_train, y_train, classes: np.ndarray):
        """
        Обучает все модели.
        
        Параметры:
            X_train: TF-IDF матрица признаков
            y_train: метки классов
            classes: массив названий классов
        """
        print("=" * 50)
        print("ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\nОбучение {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Оценка на training set
            train_acc = model.score(X_train, y_train)
            print(f"  Train accuracy: {train_acc:.4f}")
        
        return self
    
    def evaluate(self, X_test, y_test, classes: np.ndarray) -> dict:
        """
        Оценивает все модели на тестовых данных.
        
        Возвращает словарь с результатами.
        """
        print("\n" + "=" * 50)
        print("ОЦЕНКА МОДЕЛЕЙ")
        print("=" * 50)
        
        for name, model in self.trained_models.items():
            print(f"\n--- {name} ---")
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  F1 macro:  {f1_macro:.4f}")
            print(f"  F1 weighted: {f1_weighted:.4f}")
            
            self.results[name] = {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "y_pred": y_pred,
                "classification_report": classification_report(
                    y_test, y_pred, 
                    target_names=classes,
                    zero_division=0
                )
            }
        
        # Находим лучшую модель
        best_name = max(self.results, key=lambda k: self.results[k]["accuracy"])
        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]
        print(f"\nЛучшая модель: {best_name} (accuracy={self.results[best_name]['accuracy']:.4f})")
        
        return self.results
    
    def cross_validate(self, X_train, y_train, cv=5) -> dict:
        """Кросс-валидация для всех моделей."""
        print("\n" + "=" * 50)
        print("КРОСС-ВАЛИДАЦИЯ")
        print("=" * 50)
        
        cv_results = {}
        for name, model in self.models.items():
            print(f"\nКросс-валидация {name}...")
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_results[name] = {
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
                "scores": scores
            }
            print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return cv_results
    
    def predict(self, X) -> np.ndarray:
        """Предсказывает класс для новых данных."""
        if self.best_model is None:
            raise ValueError("Сначала обучите модель!")
        return self.best_model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Предсказывает вероятности классов."""
        if self.best_model is None:
            raise ValueError("Сначала обучите модель!")
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            # Для моделей без predict_proba используем decision_function
            from sklearn.preprocessing import label_binarize
            return np.abs(self.best_model.decision_function(X))
    
    def get_best_model_name(self) -> str:
        """Возвращает название лучшей модели."""
        return self.best_model_name
    
    def save_model(self, filepath: str = None):
        """Сохраняет лучшую модель."""
        if filepath is None:
            filepath = MODELS_DIR / f"baseline_{self.best_model_name}.pkl"
        save_object(self.best_model, filepath)
        print(f"Модель сохранена в {filepath}")
    
    def get_results_summary(self) -> pd.DataFrame:
        """Возвращает сводку результатов."""
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                "model": name,
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_weighted": result["f1_weighted"]
            })
        return pd.DataFrame(summary_data).sort_values("accuracy", ascending=False)


def train_baseline_model(X_train, X_test, y_train, y_test, classes: np.ndarray):
    """
    Полная функция обучения базовой модели.
    
    Возвращает:
        baseline: обученный BaselineModel
        results: результаты оценки
        cv_results: результаты кросс-валидации
    """
    # Создаём и обучаем модель
    baseline = BaselineModel()
    baseline.train(X_train, y_train, classes)
    
    # Оцениваем
    results = baseline.evaluate(X_test, y_test, classes)
    
    # Кросс-валидация
    cv_results = baseline.cross_validate(X_train, y_train, cv=5)
    
    return baseline, results, cv_results


if __name__ == "__main__":
    from src.preprocessing import preprocess_data
    from src.data_collection import collect_data
    
    # Сбор и предобработка данных
    df = collect_data(source="synthetic")
    X_train, X_test, y_train, y_test, classes = preprocess_data(df, save_results=False)
    
    # Обучение базовой модели
    baseline, results, cv_results = train_baseline_model(
        X_train, X_test, y_train, y_test, classes
    )
    
    # Вывод результатов
    print("\n" + "=" * 50)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 50)
    summary = baseline.get_results_summary()
    print(summary.to_string(index=False))
    
    # Сохраняем лучшую модель
    baseline.save_model()
