"""
Сбор данных для проекта классификации языков программирования.

Использует публичный датасет CodeSearchNet, который содержит
сниппеты кода с метками языков программирования.

Альтернативные источники:
- CodeSearchNet: https://github.com/github/CodeSearchNet
- Kaggle: https://www.kaggle.com/datasets
- HuggingFace Datasets: https://huggingface.co/datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from string import Template
warnings.filterwarnings("ignore")

from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, LANGUAGES, ensure_directories


# Маппинг тегов StackOverflow -> наши классы
TAG_TO_LANGUAGE = {
    "python": "python",
    "javascript": "javascript",
    "java": "java",
    "c++": "cpp",
    "cpp": "cpp",
    "c-plus-plus": "cpp",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "sql": "sql",
    "html": "htmlcss",
    "css": "htmlcss",
    "html-css": "htmlcss",
}


def generate_synthetic_data(n_samples_per_class: int = 500) -> pd.DataFrame:
    """
    Генерирует синтетические сниппеты кода для каждого языка.
    Используется как fallback, если нет доступа к реальному датасету.
    """
    print("Генерация синтетических данных...")
    
    # Шаблоны кода для каждого языка
    # Используем $func, $Class, $param для string.Template (без конфликтов с {})
    code_templates = {
        "python": [
            "def $func($param):\n    return $param + 1\n\nresult = $func(10)",
            "import pandas as pd\ndf = pd.DataFrame({'col': [1, 2, 3]})\nprint(df)",
            "class $Class:\n    def __init__(self):\n        self.value = 0\n    \n    def get_value(self):\n        return self.value",
            "with open('file.txt', 'r') as f:\n    content = f.read()\nprint(content)",
            "import re\npattern = r'\\d+'\ntext = 'abc123def'\nresult = re.findall(pattern, text)",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "nums = [1, 2, 3, 4, 5]\nsquared = [x**2 for x in nums]",
            "import json\ndata = {'name': 'John', 'age': 30}\njson_str = json.dumps(data)",
            "def greet(name):\n    '''Return greeting message'''\n    return f'Hello, {name}!'\n\nprint(greet('World'))",
            "class Animal:\n    def speak(self):\n        raise NotImplementedError\n\nclass Dog(Animal):\n    def speak(self):\n        return 'Woof!'",
        ],
        "javascript": [
            "function $func($param) {\n    return $param + 1;\n}\n\nconst result = $func(10);",
            "const arr = [1, 2, 3];\nconst doubled = arr.map(x => x * 2);",
            "class $Class {\n    constructor() {\n        this.value = 0;\n    }\n    getValue() {\n        return this.value;\n    }\n}",
            "const fs = require('fs');\nconst data = fs.readFileSync('file.txt', 'utf8');",
            "const regex = /\\d+/g;\nconst text = 'abc123def';\nconst matches = text.match(regex);",
            "const promise = new Promise((resolve, reject) => {\n    resolve('Done!');\n});",
            "const obj = {name: 'John', age: 30};\nconst {name, age} = obj;",
            "document.getElementById('myId').innerHTML = 'Hello';",
            "async function fetchData() {\n    const response = await fetch('/api/data');\n    return response.json();\n}",
            "const filter = arr.filter(x => x > 3);\nconst sort = arr.sort((a, b) => a - b);",
        ],
        "java": [
            "public class $Class {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}",
            "List<String> list = new ArrayList<>();\nlist.add(\"item\");\nfor (String s : list) {\n    System.out.println(s);\n}",
            "Map<String, Integer> map = new HashMap<>();\nmap.put(\"key\", 1);\nint value = map.get(\"key\");",
            "try {\n    FileReader fr = new FileReader(\"file.txt\");\n} catch (IOException e) {\n    e.printStackTrace();\n}",
            "public interface $Interface {\n    void method();\n}\n\nclass Impl implements $Interface {\n    public void method() {}\n}",
            "String[] parts = \"a,b,c\".split(\",\");\nString joined = String.join(\"-\", parts);",
            "int sum = IntStream.of(1, 2, 3).sum();\nOptional<Integer> opt = Optional.of(10);",
            "class Person {\n    private String name;\n    public String getName() { return name; }\n}",
            "Set<String> set = new HashSet<>();\nset.add(\"element\");\nboolean has = set.contains(\"element\");",
            "CompletableFuture.runAsync(() -> {\n    System.out.println(\"Async task\");\n});",
        ],
        "cpp": [
            "#include <iostream>\nint main() {\n    std::cout << \"Hello\" << std::endl;\n    return 0;\n}",
            "#include <vector>\nstd::vector<int> v = {1, 2, 3};\nfor (int x : v) {\n    std::cout << x << std::endl;\n}",
            "#include <string>\nstd::string s = \"hello\";\nstd::cout << s.length() << std::endl;",
            "class $Class {\npublic:\n    int value;\n    $Class() : value(0) {}\n};",
            "#include <algorithm>\nstd::sort(v.begin(), v.end());\nstd::reverse(v.begin(), v.end());",
            "int* arr = new int[10];\ndelete[] arr;",
            "#include <fstream>\nstd::ifstream file(\"data.txt\");\nstd::string line;\nstd::getline(file, line);",
            "std::unique_ptr<int> ptr = std::make_unique<int>(42);",
            "template<typename T>\nT max(T a, T b) { return (a > b) ? a : b; }",
            "#include <map>\nstd::map<std::string, int> m;\nm[\"key\"] = 1;",
        ],
        "go": [
            "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello\")\n}",
            "nums := []int{1, 2, 3}\nfor _, num := range nums {\n    fmt.Println(num)\n}",
            "type $Struct struct {\n    Name string\n    Value int\n}",
            "result, err := someFunction()\nif err != nil {\n    log.Fatal(err)\n}",
            "ch := make(chan int)\n go func() {\n    ch <- 42\n }()\n<-ch",
            "import \"strings\"\nvar s = strings.Join([]string{\"a\", \"b\"}, \",\")",
            "var ptr *int\nif ptr != nil {\n    fmt.Println(*ptr)\n}",
            "type Reader interface {\n    Read(p []byte) (n int, err error)\n}",
            "data := map[string]interface{}{\n    \"key\": \"value\",\n}",
            "func add(a, b int) int {\n    return a + b\n}",
        ],
        "rust": [
            "fn main() {\n    println!(\"Hello\");\n}",
            "let v = vec![1, 2, 3];\nfor item in &v {\n    println!(\"{}\", item);\n}",
            "struct $Struct {\n    name: String,\n    value: i32,\n}",
            "match result {\n    Ok(val) => println!(\"{}\", val),\n    Err(e) => println!(\"Error: {}\", e),\n}",
            "let mut map = std::collections::HashMap::new();\nmap.insert(\"key\", 1);",
            "let filtered: Vec<i32> = nums.iter()\n    .filter(|&&x| x > 0)\n    .cloned()\n    .collect();",
            "let handle = std::thread::spawn(|| {\n    println!(\"From thread\");\n});\nhandle.join().unwrap();",
            "match option {\n    Some(x) => x,\n    None => 0,\n}",
            "trait Speak {\n    fn speak(&self);\n}\n\nimpl Speak for Dog {\n    fn speak(&self) {}\n}",
            "let result = (1..100)\n    .filter(|x| x % 2 == 0)\n    .map(|x| x * 3)\n    .collect::<Vec<_>>();",
        ],
        "sql": [
            "SELECT * FROM users WHERE age > 18;",
            "SELECT name, COUNT(*) as count FROM orders GROUP BY name HAVING count > 5;",
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com');",
            "UPDATE users SET age = 30 WHERE name = 'John';",
            "DELETE FROM sessions WHERE expired = true;",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id;",
            "SELECT * FROM users ORDER BY created_at DESC LIMIT 10;",
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(255));",
            "SELECT AVG(salary) FROM employees WHERE department = 'Engineering';",
            "SELECT name FROM users WHERE email LIKE '%@example.com';",
        ],
        "htmlcss": [
            "<!DOCTYPE html>\n<html>\n<head><title>Page</title></head>\n<body>\n    <h1>Hello</h1>\n</body>\n</html>",
            "<div class=\"container\">\n    <p class=\"text\">Content</p>\n</div>",
            "<form action=\"/submit\" method=\"POST\">\n    <input type=\"text\" name=\"username\">\n    <button type=\"submit\">Submit</button>\n</form>",
            "<ul>\n    <li>Item 1</li>\n    <li>Item 2</li>\n</ul>",
            "body {\n    font-family: Arial, sans-serif;\n    margin: 0;\n    padding: 20px;\n}",
            ".container {\n    display: flex;\n    justify-content: center;\n    align-items: center;\n}",
            "#header {\n    background-color: #333;\n    color: white;\n    padding: 10px;\n}",
            "@media (max-width: 768px) {\n    .container {\n        flex-direction: column;\n    }\n}",
            "<a href=\"https://example.com\" target=\"_blank\">Link</a>",
            "<table>\n    <thead><tr><th>Name</th></tr></thead>\n    <tbody><tr><td>John</td></tr></tbody>\n</table>",
        ],
    }
    
    func_names = ["greet", "process", "calculate", "transform", "handle"]
    class_names = ["User", "Product", "Order", "Item", "Data"]
    param_names = ["x", "value", "data", "input", "item"]
    
    data = []
    
    for lang, lang_templates in tqdm(code_templates.items(), desc="Generating data"):
        for i in range(n_samples_per_class):
            template_str = lang_templates[i % len(lang_templates)]
            t = Template(template_str)
            try:
                code = t.substitute(
                    func=func_names[i % len(func_names)],
                    Class=class_names[i % len(class_names)],
                    param=param_names[i % len(param_names)],
                    Struct=class_names[i % len(class_names)],
                    Interface=class_names[i % len(class_names)],
                    Struct2=class_names[(i + 1) % len(class_names)],
                )
                data.append({"code": code, "language": lang})
            except Exception:
                pass
    
    return pd.DataFrame(data)


def load_huggingface_data() -> pd.DataFrame:
    """
    Загружает данные из HuggingFace Datasets.
    
    Возвращает DataFrame с колонками 'code' и 'language'.
    Если данные недоступны, возвращает пустой DataFrame.
    """
    print("Попытка загрузки данных из HuggingFace...")
    
    try:
        from datasets import load_dataset
        
        # Пробуем CodeSearchNet
        try:
            ds = load_dataset("CodeSearchNet", "default", split="train")
            if len(ds) > 0:
                df = pd.DataFrame(ds)
                if 'code' in df.columns and 'language' in df.columns:
                    return df[['code', 'language']]
        except Exception:
            pass
        
        # Если CodeSearchNet не доступен, возвращаем пустой DataFrame
        print("CodeSearchNet недоступен")
        return pd.DataFrame(columns=['code', 'language'])
        
    except ImportError:
        print("Библиотека datasets не установлена")
        return pd.DataFrame(columns=['code', 'language'])


def load_codesearchnet_data(data_dir: str = "data/raw/codesearchnet") -> pd.DataFrame:
    """
    Загружает данные из CodeSearchNet датасета.
    
    Пытается загрузить из локальных файлов или HuggingFace.
    """
    print("Попытка загрузки CodeSearchNet данных...")
    
    # Сначала пробуем HuggingFace
    hf_df = load_huggingface_data()
    if len(hf_df) > 0:
        return hf_df
    
    # Затем пробуем локальные файлы
    data_path = Path(data_dir)
    if data_path.exists():
        jsonl_files = list(data_path.rglob("*.jsonl"))
        if jsonl_files:
            data = []
            for f in tqdm(jsonl_files, desc="Reading files"):
                df = pd.read_json(f, lines=True)
                if 'code' in df.columns and 'language' in df.columns:
                    data.append(df[['code', 'language']])
            
            if data:
                return pd.concat(data, ignore_index=True)
    
    print("CodeSearchNet данные не найдены")
    return pd.DataFrame(columns=['code', 'language'])


def load_kaggle_data(dataset_path: str = None) -> pd.DataFrame:
    """
    Загружает данные из Kaggle datasets.
    
    Поддерживаемые форматы:
    - code_snippets_kaggle.csv: колонки 'code', 'language'
    - data_python.csv + data_cpp.csv: Python и C++ решения
    - code_data.csv: колонки 'code', 'language'
    
    Если Kaggle данные недоступны, дополняет синтетическими данными.
    """
    # Пробуем несколько путей
    paths_to_try = [
        RAW_DATA_DIR / "code_snippets_kaggle.csv",
        RAW_DATA_DIR / "code_data.csv",
    ]
    
    if dataset_path:
        paths_to_try.insert(0, Path(dataset_path))
    
    kaggle_df = None
    for path in paths_to_try:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if 'code' in df.columns and 'language' in df.columns:
                    kaggle_df = df
                    print(f"Загружено из {path}: {len(df)} примеров")
                    print(f"Языки: {df['language'].value_counts().to_dict()}")
                    break
            except Exception as e:
                print(f"Ошибка загрузки {path}: {e}")
    
    if kaggle_df is not None:
        # Нормализуем названия языков
        lang_map = {
            'cpp': 'cpp',
            'c++': 'cpp',
            'python': 'python',
            'javascript': 'javascript',
            'js': 'javascript',
            'java': 'java',
            'go': 'go',
            'golang': 'go',
            'rust': 'rust',
            'sql': 'sql',
            'html': 'htmlcss',
            'css': 'htmlcss',
            'htmlcss': 'htmlcss',
        }
        
        kaggle_df['language'] = kaggle_df['language'].str.lower().map(lambda x: lang_map.get(x, x))
        
        # Оставляем только нужные языки
        valid_langs = set(LANGUAGES)
        kaggle_df = kaggle_df[kaggle_df['language'].isin(valid_langs)]
        
        # Проверяем, какие языки отсутствуют
        existing_langs = set(kaggle_df['language'].unique())
        missing_langs = valid_langs - existing_langs
        
        if missing_langs:
            print(f"\nДополняем синтетическими данными для языков: {missing_langs}")
            synthetic_df = generate_synthetic_data(n_samples_per_class=500)
            synthetic_df = synthetic_df[synthetic_df['language'].isin(missing_langs)]
            kaggle_df = pd.concat([kaggle_df, synthetic_df], ignore_index=True)
        
        print(f"\nИтого: {len(kaggle_df)} примеров")
        print(f"Распределение по языкам:")
        print(kaggle_df['language'].value_counts())
        
        return kaggle_df
    else:
        print("Kaggle данные не найдены. Генерируются синтетические данные.")
        return generate_synthetic_data(n_samples_per_class=500)


def collect_data(source: str = "synthetic") -> pd.DataFrame:
    """
    Основной метод для сбора данных.
    
    Параметры:
        source: источник данных ('synthetic', 'codesearchnet', 'kaggle')
    """
    ensure_directories()
    
    if source == "synthetic":
        df = generate_synthetic_data(n_samples_per_class=500)
    elif source == "codesearchnet":
        df = load_codesearchnet_data()
    elif source == "huggingface":
        df = load_huggingface_data()
    elif source == "kaggle":
        df = load_kaggle_data()
    else:
        raise ValueError(f"Неизвестный источник: {source}")
    
    # Сохраняем сырые данные
    raw_path = RAW_DATA_DIR / "code_snippets.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Данные сохранены в {raw_path}")
    
    return df


if __name__ == "__main__":
    df = collect_data(source="synthetic")
    print(f"\nРазмер датасета: {len(df)} примеров")
    print(f"\nРаспределение по языкам:")
    print(df["language"].value_counts())
    print(f"\nПримеры кода:")
    for lang in LANGUAGES[:3]:
        sample = df[df["language"] == lang].iloc[0]["code"]
        print(f"\n[{lang}]:")
        print(sample[:100] + "..." if len(sample) > 100 else sample)
