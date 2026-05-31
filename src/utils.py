"""
Вспомогательные функции для проекта.
"""

import os
import pickle
from pathlib import Path

# Каталог проекта
PROJECT_ROOT = Path(__file__).parent.parent

# Каталог для данных
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Каталог для моделей
MODELS_DIR = PROJECT_ROOT / "models"

# Каталог для результатов
RESULTS_DIR = PROJECT_ROOT / "results"

# Языки программирования для классификации
LANGUAGES = [
    "python",
    "javascript",
    "java",
    "cpp",
    "go",
    "rust",
    "sql",
    "htmlcss"
]


def ensure_directories():
    """Создаёт необходимые директории проекта."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_object(obj, filename: str):
    """Сохраняет объект в файл с помощью pickle."""
    ensure_directories()
    filepath = MODELS_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")


def load_object(filename: str):
    """Загружает объект из файла с помощью pickle."""
    filepath = MODELS_DIR / filename
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_results(results: dict, filename: str):
    """Сохраняет результаты оценки в JSON."""
    import json
    ensure_directories()
    filepath = RESULTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filename: str) -> dict:
    """Загружает результаты из JSON."""
    import json
    filepath = RESULTS_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
