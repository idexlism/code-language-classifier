"""
Главный скрипт для запуска полного пайплайна проекта.

Автоматизирует:
1. Сбор данных
2. Предобработку
3. Обучение базовой модели
4. Оценка модели
5. Визуализация результатов
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import LANGUAGES, RESULTS_DIR, PROCESSED_DATA_DIR
from src.data_collection import collect_data
from src.preprocessing import preprocess_data, CodePreprocessor
from src.model_baseline import BaselineModel, train_baseline_model
from src.evaluation import (
    evaluate_model,
    analyze_confusion_matrix,
    analyze_language_confusion,
    plot_model_comparison,
    manual_check,
    save_all_results
)


def main():
    """Главная функция запуска проекта."""
    print("=" * 60)
    print("ПРОЕКТ: Классификация языков программирования")
    print("=" * 60)
    
    # 1. Сбор данных
    print("\n[ШАГ 1/5] Сбор данных...")
    df = collect_data(source="synthetic")
    print(f"Собрано {len(df)} примеров")
    print(f"Распределение:\n{df['language'].value_counts()}")
    
    # 2. Предобработка
    print("\n[ШАГ 2/5] Предобработка данных...")
    X_train, X_test, y_train, y_test, classes = preprocess_data(df, save_results=True)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Классы: {classes}")
    
    # 3. Обучение базовой модели
    print("\n[ШАГ 3/5] Обучение базовых моделей...")
    baseline = BaselineModel()
    baseline.train(X_train, y_train, classes)
    
    # 4. Оценка моделей
    print("\n[ШАГ 4/5] Оценка моделей...")
    results_list = []
    
    for name, model in baseline.trained_models.items():
        result = evaluate_model(model, X_test, y_test, classes, model_name=name)
        results_list.append(result)
    
    # 5. Анализ результатов
    print("\n[ШАГ 5/5] Анализ результатов...")
    
    # Матрица путаницы для лучшей модели
    best_result = min(results_list, key=lambda x: -x["accuracy"])
    analyze_confusion_matrix(
        best_result["confusion_matrix"], 
        classes,
        save_path=str(RESULTS_DIR / "confusion_matrix.png")
    )
    
    # Анализ путаницы между языками
    analyze_language_confusion(results_list, classes)
    
    # Сравнение моделей
    comparison_df = plot_model_comparison(
        results_list,
        save_path=str(RESULTS_DIR / "model_comparison.png")
    )
    print("\nСравнение моделей:")
    print(comparison_df.to_string(index=False))
    
    # Ручная проверка
    print("\n--- Ручная проверка ---")
    test_samples = {
        "python": [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nprint(df.head())",
        ],
        "javascript": [
            "function greet(name) {\n    return 'Hello, ' + name;\n}\nconsole.log(greet('World'));",
            "const arr = [1, 2, 3, 4, 5];\nconst doubled = arr.map(x => x * 2);",
        ],
        "java": [
            "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}",
            "List<String> list = new ArrayList<>();\nlist.add(\"test\");",
        ],
        "cpp": [
            '#include <iostream>\nusing namespace std;\nint main() {\n    cout << "Hello" << endl;\n    return 0;\n}',
            "#include <vector>\nvector<int> v = {1, 2, 3};",
        ],
        "go": [
            'package main\nimport "fmt"\nfunc main() {\n    fmt.Println("Hello")\n}',
            "func add(a int, b int) int {\n    return a + b\n}",
        ],
        "rust": [
            "fn main() {\n    println!(\"Hello\");\n}",
            "let mut vec = vec![1, 2, 3];\nvec.push(4);",
        ],
        "sql": [
            "SELECT * FROM users WHERE age > 18 ORDER BY name;",
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com');",
        ],
        "htmlcss": [
            "<!DOCTYPE html>\n<html>\n<head><title>Test</title></head>\n<body><h1>Hello</h1></body>\n</html>",
            ".container {\n    display: flex;\n    justify-content: center;\n}",
        ],
    }
    
    # Проверяем лучшей моделью
    best_model_name = best_result["model_name"]
    best_model = baseline.trained_models[best_model_name]
    
    # Загружаем векторизатор для ручной проверки
    from src.preprocessing import CodePreprocessor
    preprocessor = CodePreprocessor.load(str(PROCESSED_DATA_DIR / "preprocessor.pkl"))
    tfidf_vec = preprocessor.tfidf_vectorizer
    
    manual_check(best_model, test_samples, classes, model_name=best_model_name,
                 tfidf_vectorizer=tfidf_vec)
    
    # Сохраняем результаты
    save_all_results(results_list)
    
    # Сохраняем лучшую модель
    baseline.save_model()
    
    print("\n" + "=" * 60)
    print("ПРОЕКТ ЗАВЕРШЁН!")
    print("=" * 60)
    print(f"\nРезультаты сохранены в: {RESULTS_DIR}")
    print(f"Модели сохранены в: models/")
    
    return baseline, results_list


if __name__ == "__main__":
    baseline, results = main()
