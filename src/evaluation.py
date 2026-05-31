"""
Модуль оценки и визуализации результатов.

Включает:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Анализ ошибок и "путаницы" между языками
- Визуализация результатов
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.utils import RESULTS_DIR


def evaluate_model(model, X_test, y_test, classes: np.ndarray, model_name: str = "Model") -> dict:
    """
    Оценивает модель и возвращает результаты.
    
    Параметры:
        model: обученная модель с методом predict
        X_test: тестовые признаки
        y_test: тестовые метки
        classes: массив названий классов
        model_name: название модели для отчёта
    
    Возвращает:
        словарь с результатами оценки
    """
    print(f"\n{'='*50}")
    print(f"ОЦЕНКА МОДЕЛИ: {model_name}")
    print(f"{'='*50}")
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:         {acc:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1 (macro):        {f1_macro:.4f}")
    print(f"F1 (weighted):     {f1_weighted:.4f}")
    
    # Отчёт по классам
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    print(f"\nClassification Report:\n{report}")
    
    # Матрица путаницы
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_true": y_test
    }
    
    return results


def analyze_confusion_matrix(confusion_matrix: np.ndarray, classes: np.ndarray, save_path: str = None):
    """
    Анализирует и визуализирует матрицу путаницы.
    
    Параметры:
        confusion_matrix: матрица путаницы
        classes: массив названий классов
        save_path: путь для сохранения графика
    """
    print(f"\n{'='*50}")
    print("АНАЛИЗ МАТРИЦЫ ПУТАНИЦЫ")
    print(f"{'='*50}")
    
    # Нормализуем матрицу
    cm_normalized = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Обычная матрица
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Нормализованная матрица
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='YlOrRd',
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён в {save_path}")
    
    plt.show()
    
    # Анализ наиболее частых ошибок
    print("\nНаиболее частые ошибки (путаница между языками):")
    
    # Находим недиагональные элементы
    errors = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and confusion_matrix[i, j] > 0:
                errors.append({
                    "actual": classes[i],
                    "predicted": classes[j],
                    "count": confusion_matrix[i, j],
                    "percentage": confusion_matrix[i, j] / confusion_matrix[i].sum() * 100
                })
    
    # Сортируем по количеству ошибок
    errors.sort(key=lambda x: x["count"], reverse=True)
    
    for err in errors[:15]:
        print(f"  {err['actual']} -> {err['predicted']}: {err['count']} ({err['percentage']:.1f}%)")
    
    return errors


def analyze_language_confusion(results_list: list, classes: np.ndarray):
    """
    Анализирует путаницу между похожими языками across multiple models.
    
    Параметры:
        results_list: список результатов оценки от разных моделей
        classes: массив названий классов
    """
    print(f"\n{'='*50}")
    print("АНАЛИЗ ПУТАНИЦЫ МЕЖДУ ЯЗЫКАМИ")
    print(f"{'='*50}")
    
    confusion_data = []
    
    for result in results_list:
        model_name = result["model_name"]
        cm = result["confusion_matrix"]
        
        # Нормализуем
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        # Находим пары с максимальной путаницей
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm_norm[i, j] > 0.1:  # Более 10% путаницы
                    confusion_data.append({
                        "model": model_name,
                        "actual": classes[i],
                        "predicted": classes[j],
                        "percentage": cm_norm[i, j] * 100
                    })
    
    if confusion_data:
        df_confusion = pd.DataFrame(confusion_data)
        print("\nЗначительная путаница (>10%):")
        print(df_confusion.to_string(index=False))
        
        # Анализ конкретных пар
        print("\n--- Java vs C++ ---")
        java_cpp = df_confusion[
            (df_confusion["actual"].isin(["java", "cpp"])) & 
            (df_confusion["predicted"].isin(["java", "cpp"])) &
            (df_confusion["actual"] != df_confusion["predicted"])
        ]
        if not java_cpp.empty:
            print(java_cpp.to_string(index=False))
        else:
            print("Путаница между Java и C++ отсутствует или минимальна")
    else:
        print("Значительной путаницы между языками не обнаружено.")


def plot_model_comparison(results_list: list, save_path: str = None):
    """
    Визуализирует сравнение моделей.
    
    Параметры:
        results_list: список результатов оценки
        save_path: путь для сохранения графика
    """
    print(f"\n{'='*50}")
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'='*50}")
    
    # Собираем данные
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            "model": result["model_name"],
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "f1_weighted": result["f1_weighted"]
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("accuracy", ascending=False)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df))
    width = 0.25
    
    bars1 = ax.bar([i - width for i in x], df["accuracy"], width, label='Accuracy')
    bars2 = ax.bar(x, df["f1_macro"], width, label='F1 Macro')
    bars3 = ax.bar([i + width for i in x], df["f1_weighted"], width, label='F1 Weighted')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён в {save_path}")
    
    plt.show()
    
    return df


def manual_check(model, test_samples: dict, classes: np.ndarray, model_name: str = "Model",
                 tfidf_vectorizer=None):
    """
    Ручная проверка модели на примерах.
    
    Параметры:
        model: обученная модель
        test_samples: словарь {язык: [список примеров кода]}
        classes: массив названий классов
        model_name: название модели
        tfidf_vectorizer: векторизатор TF-IDF (для sklearn моделей)
    """
    print(f"\n{'='*50}")
    print(f"РУЧНАЯ ПРОВЕРКА: {model_name}")
    print(f"{'='*50}")
    
    all_samples = []
    true_langs = []
    
    for lang, samples in test_samples.items():
        for sample in samples:
            all_samples.append(sample)
            true_langs.append(lang)
    
    # Векторизация если нужно
    if tfidf_vectorizer is not None:
        X_samples = tfidf_vectorizer.transform(all_samples)
    else:
        X_samples = all_samples
    
    # Предсказания
    predictions = model.predict(X_samples)
    predicted_langs = [classes[p] for p in predictions]
    
    print("\nРезультаты ручной проверки:")
    print("-" * 60)
    
    correct = 0
    for i, (sample, true_lang, pred_lang) in enumerate(
        zip(all_samples, true_langs, predicted_langs)
    ):
        is_correct = true_lang == pred_lang
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        sample_preview = sample[:80].replace('\n', ' ')
        print(f"{status} True: {true_lang:12} | Pred: {pred_lang:12} | {sample_preview}")
    
    accuracy = correct / len(all_samples) * 100
    print(f"\nРучная проверка accuracy: {accuracy:.1f}% ({correct}/{len(all_samples)})")
    
    return accuracy


def save_all_results(results_list: list, save_dir: str = None):
    """Сохраняет все результаты в файлы."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    
    save_dir = save_dir if isinstance(save_dir, str) else str(save_dir)
    
    # Сохраняем сводку
    summary_data = []
    for result in results_list:
        summary_data.append({
            "model": result["model_name"],
            "accuracy": result["accuracy"],
            "precision_macro": result.get("precision_macro", 0),
            "recall_macro": result.get("recall_macro", 0),
            "f1_macro": result["f1_macro"],
            "f1_weighted": result["f1_weighted"]
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{save_dir}/model_comparison.csv", index=False)
    print(f"Сводка сохранена в {save_dir}/model_comparison.csv")
    
    # Сохраняем отчёты
    import json
    for result in results_list:
        report_data = {
            "model": result["model_name"],
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "f1_weighted": result["f1_weighted"],
            "classification_report": result["classification_report"]
        }
        with open(f"{save_dir}/{result['model_name']}_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
    
    print("Отчёты сохранены.")


if __name__ == "__main__":
    # Пример использования
    print("Модуль оценки готов к использованию.")
    print("Импортируйте функции в основной скрипт.")
