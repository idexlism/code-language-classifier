"""
Предобработка данных для классификации языков программирования.

Включает:
- Очистку кода от комментариев и пустых строк
- Токенизацию
- Балансировку датасета
- Векторизацию (TF-IDF)
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.utils import PROCESSED_DATA_DIR, LANGUAGES


class CodePreprocessor:
    """Предобработка сниппетов кода."""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2
        )
    
    def clean_code(self, code: str) -> str:
        """
        Очищает сниппет кода:
        - Удаляет многострочные комментарии
        - Удаляет однострочные комментарии
        - Удаляет лишние пробелы
        """
        if not isinstance(code, str):
            return ""
        
        # Удаляем многострочные комментарии (/* ... */)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Удаляем однострочные комментарии (// ...)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Удаляем комментарии в Python (# ...)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Удаляем HTML комментарии (<!-- ... -->)
        code = re.sub(r'<!--.*?-->', '', code, flags=re.DOTALL)
        
        # Удаляем строковые литералы (оставляем пустые кавычки для токенизации)
        code = re.sub(r'""".*?"""', '""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''", code, flags=re.DOTALL)
        code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
        code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
        
        # Убираем лишние пустые строки (более 2 подряд)
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        # Удаляем ведущие/ведущие пробелы в каждой строке
        lines = [line.strip() for line in code.split('\n')]
        code = '\n'.join(lines)
        
        return code.strip()
    
    def tokenize(self, code: str) -> list:
        """
        Токенизирует код на отдельные элементы.
        Использует regex для разделения на токены.
        """
        # Разделяем по символам-разделителям, сохраняя разделители
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\\s\\w]', code)
        return tokens
    
    def tokenize_column(self, df: pd.DataFrame, column: str = "code") -> pd.Series:
        """Токенизирует все сниппеты в колонке."""
        print("Токенизация кода...")
        df = df.copy()
        df[f"{column}_tokens"] = tqdm(df[column], desc="Токенизация")
        df[f"{column}_tokens"] = df[column].apply(self.tokenize)
        return df
    
    def balance_dataset(self, df: pd.DataFrame, n_samples_per_class: int = None) -> pd.DataFrame:
        """
        Балансирует датасет, уравнивая количество примеров в каждом классе.
        """
        print("Балансировка датасета...")
        
        # Определяем максимальное количество样本 на класс
        class_counts = df["language"].value_counts()
        
        if n_samples_per_class is None:
            n_samples_per_class = min(class_counts)
        else:
            n_samples_per_class = min(n_samples_per_class, min(class_counts))
        
        print(f"Каждый класс будет содержать {n_samples_per_class} примеров")
        
        balanced_dfs = []
        for lang in df["language"].unique():
            lang_df = df[df["language"] == lang]
            if len(lang_df) > n_samples_per_class:
                sampled = lang_df.sample(n=n_samples_per_class, random_state=42)
            else:
                sampled = lang_df
            balanced_dfs.append(sampled)
        
        result = pd.concat(balanced_dfs, ignore_index=True)
        print(f"Сбалансированный датасет: {len(result)} примеров")
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Выполняет полную предобработку:
        1. Очистка кода
        2. Кодирование меток
        3. TF-IDF векторизация
        4. Разделение на train/test
        
        Возвращает:
            X_train, X_test, y_train, y_test, classes
        """
        print("=" * 50)
        print("НАЧАЛО ПРЕДОБРАБОТКИ")
        print("=" * 50)
        
        # 1. Очистка
        print("\n1. Очистка кода...")
        df["clean_code"] = df["code"].apply(self.clean_code)
        
        # Удаляем пустые примеры
        df = df[df["clean_code"].str.len() > 0]
        print(f"После очистки: {len(df)} примеров")
        
        # 2. Фильтрация по известным языкам
        df = df[df["language"].isin(LANGUAGES)]
        print(f"После фильтрации: {len(df)} примеров")
        
        # 3. Балансировка
        df = self.balance_dataset(df, n_samples_per_class=400)
        
        # 4. Кодирование меток
        print("\n2. Кодирование меток...")
        y = self.label_encoder.fit_transform(df["language"])
        classes = self.label_encoder.classes_
        print(f"Классы: {classes}")
        
        # 5. TF-IDF векторизация
        print("\n3. TF-IDF векторизация...")
        X = self.tfidf_vectorizer.fit_transform(df["clean_code"])
        print(f"Векторы: {X.shape}")
        
        # 6. Разделение на train/test
        print("\n4. Разделение на train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test, classes
    
    def transform(self, texts: list) -> np.ndarray:
        """Векторизует новые тексты."""
        return self.tfidf_vectorizer.transform(texts)
    
    def save(self, filepath: str):
        """Сохраняет предобработчик."""
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump({
                "label_encoder": self.label_encoder,
                "tfidf_vectorizer": self.tfidf_vectorizer
            }, f)
        print(f"Предобработчик сохранён в {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "CodePreprocessor":
        """Загружает предобработчик."""
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        # Создаём новый объект и присваиваем атрибуты напрямую
        preprocessor = object.__new__(cls)
        preprocessor.label_encoder = data["label_encoder"]
        preprocessor.tfidf_vectorizer = data["tfidf_vectorizer"]
        print(f"Предобработчик загружен из {filepath}")
        return preprocessor


def preprocess_data(df: pd.DataFrame, save_results: bool = False) -> tuple:
    """
    Функция для полной предобработки данных.
    
    Возвращает:
        X_train, X_test, y_train, y_test, classes
    """
    preprocessor = CodePreprocessor()
    X_train, X_test, y_train, y_test, classes = preprocessor.fit_transform(df)
    
    if save_results:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        preprocessor.save(PROCESSED_DATA_DIR / "preprocessor.pkl")
    
    return X_train, X_test, y_train, y_test, classes


if __name__ == "__main__":
    from src.data_collection import collect_data
    
    # Сбор данных
    df = collect_data(source="synthetic")
    
    # Предобработка
    X_train, X_test, y_train, y_test, classes = preprocess_data(df, save_results=True)
    
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Classes: {classes}")
    print(f"y_train distribution: {np.bincount(y_train)}")
