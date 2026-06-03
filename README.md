## Описание проекта

Проект посвящён задаче суммаризации диалогов с использованием современных LLM и LoRA fine-tuning.

В проекте реализованы:

- null baseline
- zero-shot inference
- LoRA fine-tuning
- автоматическая оценка (ROUGE/BLEU)
- semantic evaluation через LLM-as-a-Judge

---

# Структура проекта

```text
dialog_llm/
├── configs/
├── notebooks/
├── results/
├── scripts/
├── src/
├── requirements_lora.txt
├── requirements_vllm.txt
├── .gitignore
└── README.md

## Используемые модели

### Основная модель

- Qwen/Qwen2.5-0.5B-Instruct

### Judge модели

- llama-3.1-8b-instant
- llama-3.3-70b-versatile

---

# Реализованные пайплайны

## 1. Null Baseline

Простейший baseline:
- первая реплика диалога
- последняя реплика диалога

### Запуск

```bash
python scripts/null_baseline.py --config configs/null.yaml
```

---

## 2. Zero-Shot Summarization

Инференс через Qwen2.5-0.5B-Instruct с использованием vLLM.

### Особенности

- deterministic decoding
- chat template prompting
- batch generation

### Запуск

```bash
python scripts/zero_shot_pipeline.py --config configs/zero_shot.yaml
```

---

## 3. LoRA Fine-Tuning

LoRA обучение выполнено с помощью:

- Unsloth
- PEFT
- TRL SFTTrainer

### LoRA применялась к

- q_proj
- k_proj
- v_proj
- o_proj

### Запуск

```bash
python scripts/lora_train.py --config configs/lora_train.yaml
```

---

## 4. LoRA Inference

Инференс с использованием:

- базовой модели Qwen
- обученного LoRA adapter

### Запуск

```bash
python scripts/lora_inference.py --config configs/lora_inference.yaml
```

---

## 5. Automatic Metrics

### Используемые метрики

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU

### Запуск

```bash
python scripts/evaluate_metrics.py --config configs/metrics.yaml
```

---

## 6. LLM-as-a-Judge

Semantic evaluation через внешние judge модели с использованием Groq API.

### Критерии оценки

- factual correctness
- coverage
- conciseness
- fluency

### Запуск

```bash
python scripts/judge_pipeline.py --config configs/judge_8b.yaml
```

---

# Основные результаты

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---:|---:|---:|---:|
| Null baseline | 0.179 | 0.056 | 0.147 | 0.044 |
| Zero-shot | 0.328 | 0.104 | 0.261 | 0.106 |
| LoRA | 0.358 | 0.104 | 0.283 | 0.146 |

LoRA улучшила overlap-based метрики относительно zero-shot baseline.

---

# LLM-as-a-Judge результаты

Judge модели показали разные результаты.

## llama-3.3-70b-versatile

Чаще предпочитала zero-shot summaries.

## llama-3.1-8b-instant

Чаще выдавала Tie между моделями.

Это показывает зависимость semantic evaluation от выбора judge модели.

---

# Основные выводы

- LoRA улучшила lexical overlap метрики
- summaries стали короче и ближе к reference summaries
- semantic evaluation не всегда совпадает с ROUGE/BLEU
- более сильные judge модели лучше замечают потерю информации

---

# Установка

## vLLM окружение

```bash
pip install -r requirements_vllm.txt
```

## LoRA окружение

```bash
pip install -r requirements_lora.txt
```

---

# .env

Создать `.env` в корне проекта:

```text
GROQ_API_KEY=your_api_key_here
```

---

# Используемые технологии

- Python
- PyTorch
- Transformers
- vLLM
- Unsloth
- PEFT
- TRL
- HuggingFace Datasets
- Evaluate
- Groq API

---

# Примечания

- данные и веса моделей не коммитятся в git
- обучение проводилось на RTX 4060 Laptop GPU (8 GB VRAM)
- LoRA обучалась в 4-bit режиме
