"""
Скрипт для загрузки датасета с Kaggle.

Использование:
1. Установите Kaggle CLI: pip install kaggle
2. Получите API ключ на https://www.kaggle.com/account
3. Поместите kaggle.json в ~/.kaggle/
4. Запустите: python download_kaggle_data.py
"""

import os
import json
import zipfile
import shutil
from pathlib import Path

# Датасеты с кодом и языками
DATASETS = {
    "programming_language": {
        "id": "concreteio/programming-languages",
        "description": "Датасет с примерами кода на разных языках",
    },
    "code-dataset": {
        "id": "crawford/code",
        "description": "Большой датасет с примерами кода",
    },
}


def install_kaggle():
    """Устанавливает Kaggle CLI."""
    import subprocess
    print("Установка Kaggle CLI...")
    subprocess.run(["pip", "install", "kaggle", "-q"], check=True)


def setup_kaggle_credentials():
    """Проверяет наличие kaggle.json."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if not kaggle_file.exists():
        print("\n" + "="*60)
        print("Kaggle API ключ не найден!")
        print("="*60)
        print("\nДля получения ключа:")
        print("1. Зарегистрируйтесь на https://www.kaggle.com")
        print("2. Перейдите в Account → API")
        print("3. Нажмите 'Create New Token'")
        print("4. Создайте директорию и поместите файл:")
        print("   mkdir -p ~/.kaggle")
        print("   cp kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_dataset(dataset_id, output_dir="data/raw/kaggle"):
    """Скачивает датасет с Kaggle."""
    import subprocess
    
    print(f"\nСкачивание: {dataset_id}")
    print("-" * 40)
    
    cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", output_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Скачано успешно!")
        
        # Распаковка ZIP
        zip_files = list(Path(output_dir).glob("*.zip"))
        for zip_file in zip_files:
            print(f"Распаковка: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(output_dir)
        
        return True
    else:
        print(f"✗ Ошибка: {result.stderr}")
        return False


def convert_to_format(input_dir, output_file="data/raw/code_snippets_kaggle.csv"):
    """Конвертирует данные в нужный формат (code, language)."""
    import pandas as pd
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ищем CSV файлы
    csv_files = list(Path(input_dir).rglob("*.csv"))
    
    if csv_files:
        # Читаем первый найденный CSV
        df = pd.read_csv(csv_files[0])
        print(f"\nНайден CSV: {csv_files[0]}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"Строк: {len(df)}")
        
        # Сохраняем
        df.to_csv(output_path, index=False)
        print(f"Сохранено в: {output_path}")
        return df
    else:
        print("CSV файлы не найдены")
        return None


def main():
    """Основная функция."""
    from src.utils import ensure_directories
    ensure_directories()
    
    print("="*60)
    print("Kaggle Data Downloader")
    print("="*60)
    
    # Устанавливаем Kaggle CLI
    try:
        install_kaggle()
    except Exception as e:
        print(f"Ошибка установки: {e}")
        return
    
    # Проверяем credentials
    if not setup_kaggle_credentials():
        print("\nПожалуйста, настройте API ключ и попробуйте снова.")
        return
    
    # Скачиваем датасеты
    output_dir = "data/raw/kaggle"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, info in DATASETS.items():
        if download_dataset(info["id"], output_dir):
            df = convert_to_format(output_dir)
            if df is not None:
                print(f"\n✓ Датасет '{name}' успешно загружен!")
                print(f"Колонки: {df.columns.tolist()}")
                print(f"Первые 5 строк:")
                print(df.head())
                break
        else:
            print(f"\nПропуск: {info['id']}")


if __name__ == "__main__":
    main()
