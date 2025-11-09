import re
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple

# ===== токенизация =====
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(str(s).lower())

# зарезервированные токены
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

# ===== словарь =====
def build_vocab(train_csv: str, min_freq: int = 2, out_dir: str = "artifacts"
               ) -> Tuple[dict, dict, int, int, int, int]:
    """
    Создаёт словарь по train.csv (колонка 'text').
    Возвращает: stoi, itos, pad_id, unk_id, bos_id, eos_id
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(train_csv)
    texts = df["text"].astype(str).tolist()

    counter = Counter()
    for s in texts:
        counter.update(tokenize(s))

    vocab = [PAD, UNK, BOS, EOS] + [t for t, c in counter.items() if c >= min_freq]
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}

    with open(f"{out_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}},
            f, ensure_ascii=False, indent=2
        )

    return stoi, itos, stoi[PAD], stoi[UNK], stoi[BOS], stoi[EOS]

# ===== кодирование =====
def encode(tokens: List[str], stoi: dict, unk_id: int) -> List[int]:
    return [stoi.get(t, unk_id) for t in tokens]

# ===== разбиение на пары (как в Ячейке 7) =====
def make_pairs_from_stream(
    text_list: List[str],
    stoi: dict,
    bos_id: int,
    eos_id: int,
    unk_id: int,
    max_len: int = 32
) -> List[Tuple[List[int], List[int]]]:
    """
    Собирает один длинный поток id из всех текстов и режет на блоки длиной <= max_len.
    Каждый блок даёт (x, y) со сдвигом на 1.
    """
    ids: List[int] = []
    for s in text_list:
        toks = tokenize(s)
        seq  = [bos_id] + encode(toks, stoi, unk_id) + [eos_id]
        ids.extend(seq)

    pairs: List[Tuple[List[int], List[int]]] = []
    for i in range(0, len(ids) - 1, max_len):
        x = ids[i : i + max_len]
        y = ids[i + 1 : i + 1 + max_len]
        if len(x) == len(y):
            pairs.append((x, y))
    return pairs

# ===== Dataset / collate / DataLoader =====
class BlockDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        x, y = self.pairs[i]
        return torch.tensor(x), torch.tensor(y)

def collate_pad(batch, pad_id: int):
    xs, ys = list(zip(*batch))
    T = max(x.size(0) for x in xs)
    xpad = torch.full((len(xs), T), pad_id)
    ypad = torch.full((len(xs), T), pad_id)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xpad[i, : x.size(0)] = x
        ypad[i, : y.size(0)] = y
    return xpad.long(), ypad.long()

def make_loader(
    pairs: List[Tuple[List[int], List[int]]],
    batch_size: int,
    pad_id: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int
) -> DataLoader:
    return DataLoader(
        BlockDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )

# ===== утилита для быстрой загрузки текстов из csv =====
def load_texts(csv_path: str) -> List[str]:
    return pd.read_csv(csv_path)["text"].astype(str).tolist()

# ===== локальный тест (необязательно) =====
if __name__ == "__main__":
    # пример самопроверки — поменяй пути под себя при желании
    DATA_DIR = "/content/text_autocomplete/data"
    train_csv = f"{DATA_DIR}/train.csv"

    stoi, itos, pad_id, unk_id, bos_id, eos_id = build_vocab(train_csv, min_freq=2, out_dir="artifacts")
    texts = load_texts(train_csv)
    pairs = make_pairs_from_stream(texts, stoi, bos_id, eos_id, unk_id, max_len=32)

    PIN = torch.cuda.is_available()
    loader = make_loader(pairs, batch_size=128, pad_id=pad_id, shuffle=True,
                         pin_memory=PIN, num_workers=0)

    xb, yb = next(iter(loader))
    print("shapes:", xb.shape, yb.shape, "| steps/epoch:", len(loader))
