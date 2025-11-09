import os, sys, json, random
import torch
import pandas as pd
import matplotlib.pyplot as plt

# --- пути проекта (универсально для .py и для ноутбука) ---
try:
    HERE = os.path.dirname(os.path.abspath(__file__))   # .../text_autocomplete/src
    BASE = os.path.abspath(os.path.join(HERE, ".."))    # .../text_autocomplete
except NameError:
    BASE = "/content/text_autocomplete"
SRC  = os.path.join(BASE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA_DIR    = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- зависимости HF ---
# предполагается, что transformers и rouge-score уже установлены из requirements
from transformers import pipeline, set_seed
from rouge_score import rouge_scorer

def load_val_texts(val_csv_path: str, sample_size: int = 100, seed: int = 42):
    df = pd.read_csv(val_csv_path)
    if "text" not in df.columns:
        # берём первый столбец как текст
        df = df.rename(columns={df.columns[0]: "text"})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)
    return df["text"].tolist()

def build_generator(model_name="distilgpt2", seed=42):
    device = 0 if torch.cuda.is_available() else -1
    set_seed(seed)
    gen = pipeline(
        task="text-generation",
        model=model_name,
        device=device
    )
    return gen

def complete_text(gen, prompt, max_new_tokens=30):
    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        pad_token_id=gen.model.config.eos_token_id,  # для gpt2 важно указать pad=eos
    )[0]["generated_text"]
    # вырезаем только продолжение после префикса
    if out.startswith(prompt):
        cont = out[len(prompt):]
    else:
        cont = out
    return cont

def main():
    # --- конфиг ---
    VAL_CSV = os.path.join(DATA_DIR, "val.csv")
    MODEL_NAME = "distilgpt2"
    SAMPLE_SIZE = 100   # можно уменьшить, если нужно быстрее
    SEED = 42
    CUTOFF_RATIO = 0.75
    MAX_NEW_TOKENS = 30

    # --- данные ---
    if not os.path.exists(VAL_CSV):
        raise FileNotFoundError(f"Не найден {VAL_CSV}. Сначала подготовь сплиты (train/val/test.csv).")
    texts = load_val_texts(VAL_CSV, sample_size=SAMPLE_SIZE, seed=SEED)

    # --- генератор ---
    generator = build_generator(MODEL_NAME, seed=SEED)

    # --- подготовка ROUGE ---
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    r1s, r2s = [], []
    samples = []

    # --- цикл по примерам ---
    random.seed(SEED)
    for text in texts:
        text = text.strip()
        if not text:
            continue
        cutoff = max(1, int(len(text) * CUTOFF_RATIO))
        prefix = text[:cutoff]
        target = text[cutoff:]
        pred_cont = complete_text(generator, prefix, max_new_tokens=MAX_NEW_TOKENS)

        # ROUGE по токенам строки (scorer сам токенизирует/стеммит)
        scores = scorer.score(target, pred_cont)
        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)

        # сохраним несколько примеров для печати
        if len(samples) < 3:
            samples.append({"prefix": prefix, "target": target, "pred": pred_cont})

    # --- средние метрики ---
    r1_mean = float(sum(r1s) / max(1, len(r1s)))
    r2_mean = float(sum(r2s) / max(1, len(r2s)))

    print(f"distilgpt2 on val ({len(r1s)} samples)")
    print(f"ROUGE-1 F1 = {r1_mean:.4f} | ROUGE-2 F1 = {r2_mean:.4f}")

    # --- примеры ---
    for i, s in enumerate(samples, 1):
        print(f"\nПример {i}:")
        print("  Вход (3/4): ", s["prefix"])
        print("  Таргет (1/4):", s["target"])
        print("  Модель (1/4):", s["pred"])

    # --- сохранить метрики в JSON ---
    out_json = os.path.join(RESULTS_DIR, "transformer_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rouge1_f1": r1_mean, "rouge2_f1": r2_mean}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Метрики сохранены в {out_json}")

    # --- мини-график ROUGE ---
    plt.figure(figsize=(5,4))
    x = ["ROUGE-1", "ROUGE-2"]
    y = [r1_mean, r2_mean]
    plt.bar(x, y)
    plt.ylim(0, 1)
    plt.title("distilgpt2 on val")
    plt.grid(axis="y", alpha=0.3)
    out_png = os.path.join(RESULTS_DIR, "transformer_rouge.png")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()
    print(f"📊 График сохранён: {out_png}")

    # показать в ноутбуке
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display
        display(Image(filename=out_png))

if __name__ == "__main__":
    main()
