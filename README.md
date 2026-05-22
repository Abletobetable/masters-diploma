# Методы выравнивания LLM для рекомендательных систем

Код к магистерской диссертации (СПбГУ, 2026).

## Структура

```
configs/           параметры экспериментов (гл. 2–5)
rec_rlp/           промпты, таргеты, метрики, инференс
scripts/           исполняемые шаги пайплайна
outputs/           чекпоинты и предсказания
```


## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для обучения нужна GPU (A100 80GB в работе). Для инференса — vLLM или режим `hf` в конфиге.

## Пайплайн

### 1. Промпт-датасет (глава 2)

Язык, тип таргета, размер выборки — как в экспериментах диплома:

```bash
python scripts/prepare_dataset.py --lang en --target-mode combined --n-users 300000
python scripts/prepare_dataset.py --lang ru --target-mode combined --n-users 300000
python scripts/prepare_dataset.py --lang en --target-mode wishlist --n-users 300000
```

Быстрая проверка на примере:

```bash
python scripts/prepare_dataset.py --raw data/example/users.jsonl
```

### 2. Обучение (глава 3)

**Decoder-only + QLoRA (Unsloth):** LLaMA-1B, Qwen2-0.5B, OPT-125M — параметры в `configs/train_qlora.yaml`.

```bash
python scripts/train_lora.py --model unsloth/Qwen2.5-0.5B-Instruct --dataset data/processed/train_en_combined.jsonl
python scripts/train_lora.py --model unsloth/Llama-3.2-1B-Instruct --output-dir outputs/llama
python scripts/train_lora.py --model facebook/opt-125m --output-dir outputs/opt
```

**P5 (T5-base):**

```bash
python scripts/train_p5.py
```

### 3. Инференс (главы 4–5)

```bash
python scripts/infer.py --model-path outputs/qlora --backend vllm --num-chains 10
```

Несколько коротких цепочек (`num_chains` 4–10) — приём из гл. 5: быстрее длинной одной генерации, списки склеиваются с дедупликацией.

### 4. Метрики (глава 4.1)

Топ-200: HitRate, Precision, Recall, Revenue, Recall by category.

```bash
python scripts/eval_metrics.py --predictions outputs/predictions.jsonl --k 200
```

Сравнение с P5 (таблица Hit@10):

```bash
python scripts/eval_metrics.py --k 10
```

### 5. Throughput (глава 5)

Секунды на пользователя (цель работы ~0,0018 с при масштабировании):

```bash
python scripts/benchmark_infer.py --n-users 1000
```

Меняйте в `configs/infer.yaml`: `backend`, `batch_size`, `num_chains`, `model_path`.

### Все эксперименты из матрицы

```bash
python scripts/run_experiments.py --stage prepare
python scripts/run_experiments.py --stage train
python scripts/run_experiments.py --stage infer
python scripts/run_experiments.py --stage eval
```

Список прогонов — `configs/experiments.yaml`.


## Конфиги

- `configs/data.yaml` — язык, таргет, размеры 125k–5M
- `configs/train_qlora.yaml` — LoRA r=16, 4-bit, batch 128, lr 2e-5
- `configs/train_p5.yaml` — T5-base, batch 128, grad accum 2
- `configs/eval.yaml` — k=200 и k=10 для P5
- `configs/infer.yaml` — vLLM, bf16, batch, цепочки
- `configs/experiments.yaml` — полная матрица прогонов

## Автор

Локис Александр Владимирович, СПбГУ, ОП «Технологии ИИ и Big Data», 2026.
