"""
Скрипт для подготовки датасета из Kaggle данных.

Объединяет:
1. data_python.csv - Python решения
2. data_cpp.csv - C++ решения
3. index.csv - Hello World на разных языках

Результат: code_snippets_kaggle.csv с колонками 'code' и 'language'
"""

import pandas as pd
from pathlib import Path


def prepare_kaggle_dataset(output_path="data/raw/code_snippets_kaggle.csv"):
    """Подготавливает датасет из Kaggle данных."""
    
    all_data = []
    
    # 1. Python данные
    print("Загрузка Python данных...")
    try:
        df_py = pd.read_csv("data/raw/kaggle/data_python.csv")
        for _, row in df_py.iterrows():
            code = str(row['python_solutions'])
            title = str(row.get('problem_title', ''))
            if code and len(code) > 10:  # Фильтруем короткие/пустые
                all_data.append({'code': code, 'language': 'python'})
        print(f"  Python: {len(df_py)} примеров")
    except Exception as e:
        print(f"  Ошибка Python: {e}")
    
    # 2. C++ данные
    print("Загрузка C++ данных...")
    try:
        df_cpp = pd.read_csv("data/raw/kaggle/data_cpp.csv")
        for _, row in df_cpp.iterrows():
            code = str(row['Answer'])
            if code and len(code) > 10:
                all_data.append({'code': code, 'language': 'cpp'})
        print(f"  C++: {len(df_cpp)} примеров")
    except Exception as e:
        print(f"  Ошибка C++: {e}")
    
    # 3. Hello World на разных языках
    print("Загрузка Hello World данных...")
    try:
        df_hw = pd.read_csv("data/raw/kaggle/index.csv")
        target_langs = {
            'JavaScript': 'javascript',
            'Java': 'java',
            'Go': 'go',
            'Rust': 'rust',
            'SQL': 'sql',
            'HTML': 'html',
            'C#': 'csharp',
        }
        
        for lang, lang_code in target_langs.items():
            row = df_hw[df_hw['language_name'] == lang]
            if len(row) > 0:
                code = str(row.iloc[0]['program'])
                all_data.append({'code': code, 'language': lang_code})
                print(f"  {lang}: 1 пример")
    except Exception as e:
        print(f"  Ошибка Hello World: {e}")
    
    # Создаём DataFrame
    df = pd.DataFrame(all_data)
    
    # Сохраняем
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nИтого: {len(df)} примеров")
    print(f"Языки: {df['language'].value_counts().to_dict()}")
    print(f"Сохранено в: {output_path}")
    
    return df


if __name__ == "__main__":
    df = prepare_kaggle_dataset()
    print("\nПервые 5 примеров:")
    for i, row in df.head(5).iterrows():
        print(f"\n[{row['language']}] {row['code'][:100]}...")
