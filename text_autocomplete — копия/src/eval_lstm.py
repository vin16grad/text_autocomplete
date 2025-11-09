import os, sys, json, torch
from collections import Counter
import matplotlib.pyplot as plt

# --- универсальная настройка путей (скрипт/ноутбук) ---
try:
    HERE = os.path.dirname(os.path.abspath(__file__))   # .../text_autocomplete/src
    BASE = os.path.abspath(os.path.join(HERE, ".."))    # .../text_autocomplete
except NameError:
    BASE = "/content/text_autocomplete"
SRC  = os.path.join(BASE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA_DIR    = os.path.join(BASE, "data")
ART_DIR     = os.path.join(BASE, "artifacts")
MODEL_DIR   = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")

from next_token_dataset import build_vocab, load_texts, make_pairs_from_stream, make_loader
from lstm_model import LSTMLM

UNK = "<unk>"

def ids_to_text(ids, itos: dict, pad_id: int):
    return " ".join(itos.get(i, UNK) for i in ids if i != pad_id)

def _ngrams(seq, n):
    return [" ".join(seq[i:i+n]) for i in range(len(seq)-n+1)] if len(seq) >= n else []

def rouge_f1(pred_tokens, ref_tokens, n):
    p_ngr, r_ngr = Counter(_ngrams(pred_tokens, n)), Counter(_ngrams(ref_tokens, n))
    overlap = sum((p_ngr & r_ngr).values())
    pred_cnt, ref_cnt = max(1, sum(p_ngr.values())), max(1, sum(r_ngr.values()))
    prec = overlap / pred_cnt; rec = overlap / ref_cnt
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

@torch.no_grad()
def eval_rouge_on_loader(model, loader, itos, pad_id, eos_id, device, take_ratio=0.75, max_batches=None):
    model.eval()
    r1s, r2s, seen = [], [], 0
    for xb, _ in loader:
        if max_batches is not None and seen >= max_batches:
            break
        seen += 1
        xb = xb.to(device)
        seq = xb[0].tolist()
        L = len([t for t in seq if t != pad_id])
        k = max(1, int(L * take_ratio))
        prefix, ref = seq[:k], seq[k:L]
        gen = model.generate(prefix, max_new=len(ref), eos=eos_id, device=device)
        pred = gen[k:L]
        r1s.append(rouge_f1([itos.get(i, UNK) for i in pred], [itos.get(i, UNK) for i in ref], 1))
        r2s.append(rouge_f1([itos.get(i, UNK) for i in pred], [itos.get(i, UNK) for i in ref], 2))
    n = max(1, len(r1s))
    return float(sum(r1s)/n), float(sum(r2s)/n)

def main():
    # файлы
    train_csv = os.path.join(DATA_DIR, "train.csv")
    val_csv   = os.path.join(DATA_DIR, "val.csv")
    test_csv  = os.path.join(DATA_DIR, "test.csv")
    ckpt = os.path.join(MODEL_DIR, "lstm.pt")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Нет чекпоинта: {ckpt}. Сначала обучи модель (lstm_train.py).")

    # словарь и data loaders
    stoi, itos, pad_id, unk_id, bos_id, eos_id = build_vocab(train_csv, min_freq=2, out_dir=ART_DIR)
    MAX_LEN = 32; BS = 128; PIN = torch.cuda.is_available()

    val_pairs   = make_pairs_from_stream(load_texts(val_csv),   stoi, bos_id, eos_id, unk_id, max_len=MAX_LEN)
    test_pairs  = make_pairs_from_stream(load_texts(test_csv),  stoi, bos_id, eos_id, unk_id, max_len=MAX_LEN)
    val_loader  = make_loader(val_pairs,  BS, pad_id, False, PIN, 0)
    test_loader = make_loader(test_pairs, BS, pad_id, False, PIN, 0)

    # модель + веса
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMLM(vocab_size=len(stoi), emb=256, hidden=512, num_layers=2, drop=0.1, pad_id=pad_id).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # метрики
    r1_val, r2_val   = eval_rouge_on_loader(model, val_loader,  itos, pad_id, eos_id, device, take_ratio=0.75, max_batches=None)
    r1_test, r2_test = eval_rouge_on_loader(model, test_loader, itos, pad_id, eos_id, device, take_ratio=0.75, max_batches=None)

    print(f"[VAL]  ROUGE-1={r1_val:.4f} | ROUGE-2={r2_val:.4f}")
    print(f"[TEST] ROUGE-1={r1_test:.4f} | ROUGE-2={r2_test:.4f}")

    # пример генерации
    xb, _ = next(iter(test_loader))
    seq = xb[0].tolist()
    L = len([t for t in seq if t != pad_id]); k = max(1, int(L*0.75))
    prefix, ref = seq[:k], seq[k:L]
    pred = model.generate(prefix, max_new=len(ref), eos=eos_id, device=device)
    print("\nПРИМЕР ГЕНЕРАЦИИ (test):")
    print("  Вход (3/4):  ", ids_to_text(prefix, itos, pad_id))
    print("  Таргет (1/4):", ids_to_text(ref,    itos, pad_id))
    print("  Модель (1/4):", ids_to_text(pred[k:L], itos, pad_id))

    # сохранить метрики в JSON
    out_json = os.path.join(RESULTS_DIR, "lstm_metrics_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"val":{"rouge1_f1":r1_val,"rouge2_f1":r2_val},
                   "test":{"rouge1_f1":r1_test,"rouge2_f1":r2_test}}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Метрики сохранены в {out_json}")

    # мини-график ROUGE (бар-чарт) — чтобы было «визуально»
    plt.figure(figsize=(5,4))
    x = ["Val R1","Val R2","Test R1","Test R2"]
    y = [r1_val, r2_val, r1_test, r2_test]
    plt.bar(x, y)
    plt.ylim(0, 1)
    plt.title("ROUGE (F1)")
    plt.grid(axis="y", alpha=0.3)
    rouge_png = os.path.join(RESULTS_DIR, "rouge_eval.png")
    plt.savefig(rouge_png, bbox_inches="tight"); plt.close()
    print(f"📊 График сохранён: {rouge_png}")

    # показать в ноутбуке (если выполняется из ipykernel)
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display
        display(Image(filename=rouge_png))

if __name__ == "__main__":
    main()
