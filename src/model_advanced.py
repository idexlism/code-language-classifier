"""
Продвинутая модель для классификации языков программирования.

Использует предобученную модель CodeBERT для извлечения признаков
и классификатора на их основе.

CodeBERT - это специализированная модель для понимания кода,
предобученная на больших объёмах данных с GitHub.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.utils import MODELS_DIR, RESULTS_DIR


class CodeBERTClassifier:
    """Классификатор на основе CodeBERT."""
    
    def __init__(self, num_classes: int, model_name: str = "microsoft/CodeBERT-base-v1"):
        """
        Параметры:
            num_classes: количество классов (языков)
            model_name: название предобученной модели
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
    
    def load_model(self):
        """Загружает предобученную модель CodeBERT."""
        print(f"Загрузка модели {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Используем модель для классификации
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        self.model.to(self.device)
        print("Модель загружена!")
    
    def tokenize_data(self, texts: list, max_length: int = 512) -> dict:
        """
        Токенизирует тексты для подачи в модель.
        
        Параметры:
            texts: список строк с кодом
            max_length: максимальная длина последовательности
        
        Возвращает:
            словарь с input_ids, attention_mask, token_type_ids
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encodings
    
    def prepare_dataset(self, texts: list, labels: list = None, max_length: int = 512):
        """Подготавливает PyTorch Dataset."""
        from torch.utils.data import Dataset
        
        class CodeDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                if self.labels is not None:
                    item["labels"] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        encodings = self.tokenize_data(texts, max_length)
        
        if labels is not None:
            labels_encoded = self.label_encoder.transform(labels) if isinstance(labels[0], str) else labels
            return CodeDataset(encodings, labels_encoded)
        return CodeDataset(encodings, None)
    
    def compute_metrics(self, eval_pred):
        """Вычисляет метрики для Trainer."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        acc = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        
        return {
            "accuracy": acc,
            "f1_macro": f1_macro
        }
    
    def train(self, train_texts: list, train_labels: list, 
              eval_texts: list = None, eval_labels: list = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """
        Обучает модель CodeBERT.
        
        Параметры:
            train_texts: список тренировочных текстов
            train_labels: список тренировочных меток
            eval_texts: список оценочных текстов (опционально)
            eval_labels: список оценочных меток (опционально)
            epochs: количество эпох
            batch_size: размер батча
            learning_rate: скорость обучения
        """
        if self.model is None:
            self.load_model()
        
        # Кодирование меток
        if isinstance(train_labels[0], str):
            self.label_encoder.fit(train_labels)
            train_labels_encoded = self.label_encoder.transform(train_labels)
        else:
            train_labels_encoded = np.array(train_labels)
        
        # Подготовка датасетов
        train_dataset = self.prepare_dataset(train_texts, train_labels_encoded)
        
        if eval_texts is not None and eval_labels is not None:
            if isinstance(eval_labels[0], str):
                eval_labels_encoded = self.label_encoder.transform(eval_labels)
            else:
                eval_labels_encoded = np.array(eval_labels)
            eval_dataset = self.prepare_dataset(eval_texts, eval_labels_encoded)
        else:
            eval_dataset = None
        
        # Настройка обучения
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none"
        )
        
        # Создание Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Обучение
        print("Начало обучения...")
        trainer.train()
        
        print("Обучение завершено!")
        return trainer
    
    def evaluate(self, test_texts: list, test_labels: list, classes: np.ndarray) -> dict:
        """
        Оценивает модель на тестовых данных.
        
        Возвращает:
            словарь с результатами
        """
        if self.model is None:
            raise ValueError("Сначала обучите модель!")
        
        # Предсказания
        predictions = []
        true_labels = []
        
        batch_size = 32
        for i in tqdm(range(0, len(test_texts), batch_size), desc="Оценка"):
            batch_texts = test_texts[i:i+batch_size]
            encodings = self.tokenize_data(batch_texts)
            
            with torch.no_grad():
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(test_labels[i:i+batch_size])
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Метрики
        acc = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        report = classification_report(
            true_labels, predictions,
            target_names=classes,
            zero_division=0
        )
        
        results = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report,
            "y_pred": predictions,
            "y_true": true_labels
        }
        
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print(f"F1 weighted: {f1_weighted:.4f}")
        
        return results
    
    def predict(self, texts: list) -> np.ndarray:
        """Предсказывает класс для новых текстов."""
        if self.model is None:
            raise ValueError("Сначала обучите модель!")
        
        encodings = self.tokenize_data(texts)
        
        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
        
        return preds.cpu().numpy()
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """Предсказывает вероятности классов."""
        if self.model is None:
            raise ValueError("Сначала обучите модель!")
        
        encodings = self.tokenize_data(texts)
        
        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        return probs.cpu().numpy()
    
    def save(self, filepath: str = None):
        """Сохраняет модель."""
        if filepath is None:
            filepath = MODELS_DIR / "codebert_classifier"
        
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        # Сохраняем label encoder
        import pickle
        with open(f"{filepath}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Модель сохранена в {filepath}")
    
    @classmethod
    def load(cls, filepath: str, num_classes: int = None) -> "CodeBERTClassifier":
        """Загружает модель."""
        if num_classes is None:
            num_classes = 8  # По умолчанию
        
        classifier = cls(num_classes=num_classes, model_name=filepath)
        classifier.load_model()
        
        # Загружаем label encoder
        import pickle
        encoder_path = f"{filepath}/label_encoder.pkl"
        with open(encoder_path, "rb") as f:
            classifier.label_encoder = pickle.load(f)
        
        print(f"Модель загружена из {filepath}")
        return classifier


def train_codebert_model(train_texts: list, train_labels: list,
                         eval_texts: list = None, eval_labels: list = None,
                         test_texts: list = None, test_labels: list = None,
                         classes: np.ndarray = None) -> CodeBERTClassifier:
    """
    Полная функция обучения CodeBERT модели.
    
    Возвращает:
        classifier: обученный классификатор
        results: результаты оценки (если есть тестовые данные)
    """
    num_classes = len(set(train_labels)) if isinstance(train_labels[0], str) else max(train_labels) + 1
    
    classifier = CodeBERTClassifier(num_classes=num_classes)
    
    # Обучение
    classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        epochs=3,
        batch_size=8,
        learning_rate=2e-5
    )
    
    # Оценка
    results = None
    if test_texts is not None and test_labels is not None and classes is not None:
        results = classifier.evaluate(test_texts, test_labels, classes)
    
    return classifier, results


if __name__ == "__main__":
    from src.preprocessing import preprocess_data
    from src.data_collection import collect_data
    
    # Сбор и предобработка данных
    df = collect_data(source="synthetic")
    
    # Получаем очищенные данные
    from src.preprocessing import CodePreprocessor
    preprocessor = CodePreprocessor()
    df["clean_code"] = df["code"].apply(preprocessor.clean_code)
    df = df[df["clean_code"].str.len() > 0]
    df = df[df["language"].isin(LANGUAGES)]
    
    # Разделение данных
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["language"])
    
    train_texts = train_df["clean_code"].tolist()
    train_labels = train_df["language"].tolist()
    test_texts = test_df["clean_code"].tolist()
    test_labels = test_df["language"].tolist()
    classes = np.array(sorted(df["language"].unique()))
    
    # Обучение CodeBERT
    classifier, results = train_codebert_model(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        classes=classes
    )
    
    # Сохраняем модель
    classifier.save()
