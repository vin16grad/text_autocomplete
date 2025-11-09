# src/lstm_train.py
import os, sys, json, math
import torch, torch.nn as nn
from tqdm.auto import tqdm
import argparse
from collections import Counter
import matplotlib.pyplot as plt

# === Гиперпараметры, как в Ячейке 7 ===
MAX_LEN = 32        # можно 24–48
BATCH_SIZE = 128    # если GPU тянет — 160/192/256
PIN = torch.cuda.is_available()
NUM_WORKERS = 0     # в Colab часто стабильнее 0

# --- корректная настройка путей (работает и в .py, и в ноутбуке) ---
try:
    HERE = os.path.dirname(os.path.abspath(__file__))   # .../text_autocomplete/src
    BASE = os.path.abspath(os.path.join(HERE, ".."))    # .../text_autocomplete
except NameError:
    BASE = "/content/text_autocomplete"
    HERE = os.path.join(BASE, "src")
SRC  = os.path.join(BASE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA_DIR    = os.path.join(BASE, "data")
ART_DIR     = os.path.join(BASE, "artifacts")
MODEL_DIR   = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")

from data_utils import prepare_from_txt
from next_token_dataset import build_vocab, load_texts, make_pairs_from_stream, make_loader
from lstm_model import LSTMLM

UNK = "<unk>"

# ---------------- helpers ----------------
def ids_to_text(ids, itos: dict, pad_id: int):
    return " ".join(itos.get(i, UNK) for i in ids if i != pad_id)

def _ngrams(seq, n):
    return [" ".join(seq[i:i+n]) for i in range(len(seq)-n+1)] if len(seq) >= n else []

def rouge_f1(pred_tokens, ref_tokens, n):
    p_ngr, r_ngr = Counter(_ngrams(pred_tokens, n)), Counter(_ngrams(ref_tokens, n))
    overlap = sum((p_ngr & r_ngr).values())
    pred_cnt, ref_cnt = max(1, sum(p_ngr.values())), max(1, sum(r_ngr.values()))
    prec = overlap / pred_cnt
    rec  = overlap / ref_cnt
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

def run_epoch_bar(model, loader, criterion, optimizer, scaler, device, pad_id, train=True, desc=""):
    model.train(train)
    total_loss, total_tok = 0.0, 0
    pbar = tqdm(loader, desc=desc)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            tokens = int((yb != pad_id).sum().item())
            total_loss += loss.item() * tokens; total_tok += tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, total_tok)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)         # 1–3 ок
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--raw_txt", default=os.path.join(DATA_DIR, "tweets.txt"))
    args = ap.parse_args([]) if "ipykernel" in sys.modules else ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Данные
    train_csv = os.path.join(DATA_DIR, "train.csv")
    val_csv   = os.path.join(DATA_DIR, "val.csv")
    test_csv  = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(train_csv):
        if not os.path.exists(args.raw_txt):
            raise FileNotFoundError(f"Нет исходника: {args.raw_txt}")
        prepare_from_txt(args.raw_txt, DATA_DIR)

    # 2) Словарь
    stoi, itos, pad_id, unk_id, bos_id, eos_id = build_vocab(train_csv, min_freq=args.min_freq, out_dir=ART_DIR)
    vocab_size = len(stoi)

    # 3) Пары/лоадеры
    train_texts = load_texts(train_csv)
    val_texts   = load_texts(val_csv)
    test_texts  = load_texts(test_csv)

    train_pairs = make_pairs_from_stream(train_texts, stoi, bos_id, eos_id, unk_id, max_len=MAX_LEN)
    val_pairs   = make_pairs_from_stream(val_texts,   stoi, bos_id, eos_id, unk_id, max_len=MAX_LEN)
    test_pairs  = make_pairs_from_stream(test_texts,  stoi, bos_id, eos_id, unk_id, max_len=MAX_LEN)

    train_loader = make_loader(train_pairs, BATCH_SIZE, pad_id, True,  PIN, NUM_WORKERS)
    val_loader   = make_loader(val_pairs,   BATCH_SIZE, pad_id, False, PIN, NUM_WORKERS)
    test_loader  = make_loader(test_pairs,  BATCH_SIZE, pad_id, False, PIN, NUM_WORKERS)

    # 4) Модель
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMLM(vocab_size, emb=256, hidden=512, num_layers=2, drop=0.1, pad_id=pad_id).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # ——— логгеры для графиков ———
    train_losses, val_losses, ppls = [], [], []
    best_val = float("inf")
    ckpt = os.path.join(MODEL_DIR, "lstm.pt")

    for ep in range(1, args.epochs + 1):
        tr = run_epoch_bar(model, train_loader, criterion, optimizer, scaler, device, pad_id, True,  f"Epoch {ep}/{args.epochs} [Train]")
        va = run_epoch_bar(model, val_loader,   criterion, optimizer, scaler, device, pad_id, False, f"Epoch {ep}/{args.epochs} [Val]  ")
        r1, r2 = eval_rouge_on_loader(model, val_loader, itos, pad_id, eos_id, device, take_ratio=0.75, max_batches=200)
        ppl = math.exp(va) if va < 20 else float("inf")

        train_losses.append(tr); val_losses.append(va); ppls.append(ppl)

        xb, _ = next(iter(val_loader))
        seq = xb[0].tolist(); L = len([t for t in seq if t != pad_id]); k = max(1, int(L*0.75))
        prefix, ref = seq[:k], seq[k:L]
        pred = model.generate(prefix, max_new=len(ref), eos=eos_id, device=device)

        print(f"\nEpoch {ep}: TrainLoss={tr:.4f} | ValLoss={va:.4f} | ValPPL={ppl:.2f} | ROUGE-1={r1:.4f} | ROUGE-2={r2:.4f}")
        print("  Вход (3/4):  ", ids_to_text(prefix, itos, pad_id))
        print("  Таргет (1/4):", ids_to_text(ref,    itos, pad_id))
        print("  Модель (1/4):", ids_to_text(pred[k:L], itos, pad_id))

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), ckpt)
            print(f"✅ Saved best to {ckpt}")

    # 5) Финальные метрики (полный val/test) + сохранение
    r1_val, r2_val = eval_rouge_on_loader(model, val_loader,  itos, pad_id, eos_id, device, take_ratio=0.75, max_batches=None)
    r1_test, r2_test = eval_rouge_on_loader(model, test_loader, itos, pad_id, eos_id, device, take_ratio=0.75, max_batches=None)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "lstm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val":{"rouge1_f1":r1_val,"rouge2_f1":r2_val},
                   "test":{"rouge1_f1":r1_test,"rouge2_f1":r2_test}}, f, ensure_ascii=False, indent=2)
    print(f"[VAL]  ROUGE-1={r1_val:.4f} | ROUGE-2={r2_val:.4f}")
    print(f"[TEST] ROUGE-1={r1_test:.4f} | ROUGE-2={r2_test:.4f}")
    print(f"✅ Метрики сохранены в {os.path.join(RESULTS_DIR, 'lstm_metrics.json')}")

    # 6) Графики (loss и perplexity)
    #   а) Loss per epoch
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, '-o', label='train')
    plt.plot(val_losses,   '-o', label='val')
    plt.title('Loss per epoch'); plt.xlabel('epoch'); plt.ylabel('loss')
    plt.grid(True); plt.legend()
    loss_png = os.path.join(RESULTS_DIR, "loss.png")
    plt.savefig(loss_png, bbox_inches='tight'); plt.close()

    #   б) Val Perplexity
    plt.figure(figsize=(6,4))
    plt.plot(ppls, '-o', label='val PPL')
    plt.title('Validation Perplexity'); plt.xlabel('epoch'); plt.ylabel('PPL')
    plt.grid(True); plt.legend()
    ppl_png = os.path.join(RESULTS_DIR, "ppl.png")
    plt.savefig(ppl_png, bbox_inches='tight'); plt.close()

    print(f"📈 Графики сохранены: {loss_png} и {ppl_png}")

    # показывать в ноутбуке (если запущено внутри ipykernel)
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display
        display(Image(filename=loss_png))
        display(Image(filename=ppl_png))

if __name__ == "__main__":
    main()

