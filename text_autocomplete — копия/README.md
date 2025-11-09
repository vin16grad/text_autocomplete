
ВАЖНО!!!!
ПРИ ЗАГРУЗКЕ ВОЗНИКЛИ ПРОБЛЕМЫ С ТЕМ ЧТО Я НЕ СМОГЛА ЗАГРУЗИТЬ ИЗ ПАПКИ MODELS/LSTM.PT - ИЗ-ЗА БОЛЬШОГО РАЗМЕРА
АНАЛОГИЧНО С CSV
НО ЕСЛИ ЗАПУСТИТЬ КОД ТО ОНИ СОЗДАДУТСЯ
ЛИБО МОГУ ОТДЕЛЬНО КАК-ТО ИХ ДОГРУЗИТЬ ЕСЛИ НЕОБХОДИМО, НЕ ЗНАЮ ЧТО С НИМИ УЖЕ ДЕЛАТЬ 

Text Autocomplete Project

Учебный проект из курса по нейросетям.  
**Автодополнение текста** с помощью:
1. Собственной модели на основе **LSTM** (обученной на пользовательском датасете);
2. Предобученного трансформера **DistilGPT-2** из библиотеки `transformers`.

Обе модели сравниваются по метрикам **ROUGE-1 / ROUGE-2** и качеству генерации.

Структура проекта

text_autocomplete/
├── data/ # Датасеты
│ ├── tweets.txt # исходные данные
│ ├── raw_dataset.csv
│ ├── dataset_processed.csv
│ ├── train.csv / val.csv / test.csv
│ └── README.md
│
├── src/ # Исходный код
│ ├── data_utils.py # загрузка и очистка данных
│ ├── next_token_dataset.py # подготовка датасета и DataLoader
│ ├── lstm_model.py # архитектура LSTM LM
│ ├── lstm_train.py # обучение модели
│ ├── eval_lstm.py # оценка обученной LSTM
│ └── eval_transformer_pipeline.py # оценка DistilGPT2
│
├── models/ # сохранённые веса модели (.pt)
│ └── lstm.pt
│
├── results/ # результаты обучения и метрики
│ ├── loss.png / ppl.png
│ ├── lstm_metrics.json
│ ├── lstm_metrics_eval.json
│ ├── transformer_metrics.json
│ └── transformer_rouge.png
│
├── solution.ipynb # основной ноутбук с выполнением этапов
├── requirements_sprint_2_project.txt
└── README.md # (этот файл)


Подготовка данных через
python src/data_utils.py
Создаёт файлы train.csv, val.csv, test.csv из исходного tweets.txt

Обучение LSTM
python src/lstm_train.py --epochs 2
Результаты:
сохранённые веса в models/lstm.pt
графики loss.png и ppl.png
метрики lstm_metrics.json

Оценка LSTM
python src/eval_lstm.py

Предобученный трансформер (DistilGPT2)
python src/eval_transformer_pipeline.py

Генерирует тексты с помощью transformers.pipeline, считает метрики и сохраняет:
results/transformer_metrics.json
results/transformer_rouge.png

