import os
import re
import pandas as pd
from typing import Tuple

BASE_DIR = "/content/text_autocomplete"
RAW_PATH_DEFAULT = os.path.join(BASE_DIR, "data", "tweets.txt")
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "data")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_from_txt(raw_path: str = RAW_PATH_DEFAULT,
                     data_dir: str = DATA_DIR_DEFAULT,
                     train_frac: float = 0.8,
                     val_frac: float = 0.1) -> Tuple[str, str, str]:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Не найден файл: {raw_path}")

    os.makedirs(data_dir, exist_ok=True)

    # 1) читаем исходный txt
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # 2) raw → CSV
    raw_csv_path = os.path.join(data_dir, "raw_dataset.csv")
    pd.DataFrame({"text": lines}).to_csv(raw_csv_path, index=False)

    # 3) чистка
    cleaned = [clean_text(t) for t in lines if len(t) > 1]
    df = pd.DataFrame({"text": cleaned})
    proc_path = os.path.join(data_dir, "dataset_processed.csv")
    df.to_csv(proc_path, index=False)

    # 4) сплиты
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    train_path = os.path.join(data_dir, "train.csv")
    val_path   = os.path.join(data_dir, "val.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"✔ raw → {raw_csv_path}")
    print(f"✔ processed → {proc_path}")
    print(f"✔ splits → {train_path}, {val_path}, {test_path}")
    print(f"Размеры: train={len(train)}, val={len(val)}, test={len(test)}")
    return train_path, val_path, test_path

if __name__ == "__main__":
    prepare_from_txt(RAW_PATH_DEFAULT, DATA_DIR_DEFAULT)
