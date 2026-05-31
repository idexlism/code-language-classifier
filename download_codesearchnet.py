#!/usr/bin/env python3
"""
Скрипт для скачивания датасета CodeSearchNet.

Данные скачиваются из AWS S3 и сохраняются в папку data/raw/.

Языки: python, javascript, ruby, go, java, php
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# URL для скачивания данных из S3
# Формат: https://github-com.s3-us-west-2.amazonaws.com/...
S3_BASE_URL = "https://github-com.s3-us-west-2.amazonaws.com/codesearchnet-data-{lang}.tar.gz"

# Языки, доступные в CodeSearchNet
LANGUAGES = ["python", "javascript", "ruby", "go", "java", "php"]

# Папка для сохранения данных
OUTPUT_DIR = Path("data/raw/codesearchnet")


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Скачивает файл с прогресс-баром."""
    response = requests.get(url, stream=True)
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


def download_dataset():
    """Скачивает все языки CodeSearchNet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for lang in LANGUAGES:
        url = S3_BASE_URL.format(lang=lang)
        output_path = OUTPUT_DIR / f"codesearchnet-data-{lang}.tar.gz"
        
        if output_path.exists():
            print(f"Файл {output_path.name} уже существует, пропускаем.")
            continue
        
        print(f"Скачивание {lang}...")
        try:
            download_file(url, output_path)
            print(f"  ✓ {lang} скачан ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Ошибка скачивания {lang}: {e}")
    
    print(f"\nДанные сохранены в {OUTPUT_DIR}")
    print("Для распаковки выполните:")
    print(f"  tar -xzf {OUTPUT_DIR}/codesearchnet-data-python.tar.gz")
    print(f"  tar -xzf {OUTPUT_DIR}/codesearchnet-data-javascript.tar.gz")
    # ... и т.д.


if __name__ == "__main__":
    download_dataset()
