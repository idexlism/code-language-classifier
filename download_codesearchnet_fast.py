#!/usr/bin/env python3
"""
Скрипт для скачивания датасета CodeSearchNet.

Данные скачиваются из AWS S3 и сохраняются в папку data/raw/codesearchnet/.

Языки: python, javascript, ruby, go, java, php

Использование:
    python download_codesearchnet_fast.py python        # Только Python
    python download_codesearchnet_fast.py all            # Все языки
"""

import sys
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# URL для скачивания данных из S3 (официальный от GitHub)
S3_BASE_URL = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{lang}.zip"

# Все доступные языки
ALL_LANGUAGES = ["python", "javascript", "ruby", "go", "java", "php"]

# Языки для нашего проекта (8 классов)
PROJECT_LANGUAGES = ["python", "javascript", "go", "java"]

# Папка для сохранения данных
OUTPUT_DIR = Path("data/raw/codesearchnet")


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Скачивает файл с прогресс-баром."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"  Ошибка: {e}")
        return False


def extract_zip(zip_path: Path):
    """Распаковывает zip файл."""
    print(f"Распаковка {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=OUTPUT_DIR)
        print(f"  ✓ Распаковано")
        return True
    except Exception as e:
        print(f"  Ошибка распаковки: {e}")
        return False


def download_language(lang: str):
    """Скачивает и распаковывает данные для одного языка."""
    url = S3_BASE_URL.format(lang=lang)
    output_path = OUTPUT_DIR / f"codesearchnet-{lang}.zip"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if (OUTPUT_DIR / lang).exists() and any((OUTPUT_DIR / lang).iterdir()):
        print(f"{lang}: уже распаковано")
        return True
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"{lang}: файл уже существует ({size_mb:.1f} MB)")
    else:
        print(f"Скачивание {lang}...")
        if not download_file(url, output_path):
            return False
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  ✓ Скачано ({size_mb:.1f} MB)")
    
    # Распаковываем
    if not extract_zip(output_path):
        return False
    
    # Переименовываем папку если нужно
    extract_dir = OUTPUT_DIR / lang
    # Проверяем структуру zip - обычно внутри папка с тем же именем
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir() and item.name.startswith('CodeSearchNet') or item.name == lang:
            if not (OUTPUT_DIR / lang).exists():
                item.rename(extract_dir)
            break
    
    return True


def main():
    """Главная функция."""
    if len(sys.argv) > 1:
        lang_arg = sys.argv[1].lower()
        
        if lang_arg == "all":
            languages = PROJECT_LANGUAGES
            print("Скачивание для проекта (Python, JavaScript, Go, Java)...")
        elif lang_arg in ALL_LANGUAGES:
            languages = [lang_arg]
            print(f"Скачивание: {lang_arg}")
        else:
            print(f"Неизвестный язык: {lang_arg}")
            print(f"Доступные: {', '.join(ALL_LANGUAGES)}")
            print(f"Для проекта используйте: {', '.join(PROJECT_LANGUAGES)}")
            return
    else:
        languages = PROJECT_LANGUAGES
        print("По умолчанию скачиваются языки для проекта: Python, JavaScript, Go, Java")
    
    print(f"\nЯзыки: {', '.join(languages)}")
    print(f"Папка: {OUTPUT_DIR.absolute()}\n")
    
    success = 0
    for lang in languages:
        if download_language(lang):
            success += 1
        print()
    
    print(f"\n{'='*50}")
    print(f"Готово: {success}/{len(languages)} языков скачано")
    print(f"Данные в: {OUTPUT_DIR.absolute()}")
    print(f"\nДля запуска проекта:")
    print(f"  python3 main.py  # использует синтетические данные")
    print(f"  # Или измените в main.py на source='codesearchnet'")


if __name__ == "__main__":
    main()
