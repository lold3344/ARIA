# **This is a research tool. The author is not responsible for misuse, training data, or generated content.**

> **Legal notice:** I have nothing to do with anyone who uses this tool for illegal purposes. If you train my AI model and use it for hacking, criminal activity, or any other unlawful actions, that is entirely your own responsibility and problem.


# ARIA Atom 3.5.2

![LOGO](screenshots/ARIA-Logo.png)

**ARIA Atom 3.5.2** — локальная языковая модель на основе GPT-style Transformer с обучением на диалоговых данных, написанная на Rust. Вычисления выполняются на NVIDIA GPU через CUDA/cuBLAS с кастомными PTX ядрами. Версия 3.5.2 переходит на **GGUF** как единственный формат хранения — модель, токенизатор и состояние оптимизатора в одном файле.

> **Важно:** ARIA поддерживает только видеокарты NVIDIA. Работа на AMD, Intel и других GPU не гарантируется.

| Версия | Кодовое имя | Архитектура | Параметры | VRAM | Статус |
|---|---|---|---|---|---|
| 3.2.0 | Wotan | LSTM (1 слой) | ~44.5M | 6GB | Архив |
| 3.3.0 | Atom | Transformer (12 слоёв) | ~124M | 8GB | Архив |
| 3.4.0 | Atom | Transformer + warmup/clip | ~40M | 4GB | Архив |
| 3.5.0 | Efkolos (стандартная) | Transformer + LoRA | 250M | ~4GB | Архив |
| 3.5.1 | Efkolos (улучшенная) | Transformer + LoRA + INT4 | 250M | ~3GB | Архив |
| **3.5.2** | **Efkolos** | **Transformer + GGUF + Q4_0** | **250M** | **~3GB** | **Stable** |
| 3.5.0 | Varys (тяжелая) | Transformer + LoRA | 1B | ~8GB | Планируется |

## Что нового в 3.5.2

### v3.5.2 (Stable)
- **GGUF как единственный формат** — модель, токенизатор и Adam моменты в одном `.gguf` файле
- **Q4_0 квантизация** — экспорт инференс-модели в 4-bit (файл в ~2× меньше чекпоинта)
- Потоковая генерация кэша последовательностей — RAM не зависит от размера датасета
- Удалены форматы JSON и бинарный ARIA v2 — только GGUF
- `export_gguf` бинарь для конвертации чекпоинта в Q4_0 инференс-файл

### v3.5.1
- INT4 quantization базовой модели
- Gradient checkpointing
- LoRA Backward Pass

### v3.5.0
- LoRA (Low-Rank Adaptation)
- 250M параметров, контекст 2048 токенов

## Архитектура (3.5.0 Efkolos)

### Efkolos (250M параметров)

```
Тип:           GPT-style decoder-only Transformer + LoRA
Слои:          20
d_model:       896
Головы:        14  (head_dim = 64)
FFN dim:       3584  (4 × d_model)
Макс. контекст: 2048 токенов
Словарь:       32 000 (BPE)
Параметры:     250M (база) + 5M (LoRA адаптер)
Точность:      FP16 (база) + FP16 (адаптер) + FP32 моменты Adam
LoRA rank:     8
```

#### VRAM (batch=1, seq=2048, текущая реализация)

| Компонент | Размер |
|---|---|
| Модель FP16 | 250 МБ |
| LoRA адаптер | 60 МБ |
| Adam моменты | 2000 МБ |
| Активации | 1100 МБ |
| Attention scores | 110 МБ |
| Буферы | 100 МБ |
| **Итого** | **~3620 МБ** |

Поместится в RTX 4060 (8GB) с буфером 4GB.

### Varys (1B параметров) — *планируется*

```
Тип:           GPT-style decoder-only Transformer + LoRA
Параметры:     1B (база) + 10M (LoRA адаптер)
Требуемый VRAM: ~8–10 ГБ (с INT4 + gradient checkpointing)
Целевой GPU:   RTX 4080S (20GB)
```

## Требования

### Обязательное ПО

| Компонент | Версия | Скачать |
|---|---|---|
| Rust + Cargo | stable (2021 edition) | https://rustup.rs |
| Visual Studio Build Tools | 2017 или новее | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |
| NVIDIA CUDA Toolkit | 12.x | https://developer.nvidia.com/cuda-downloads |
| Драйвер NVIDIA | актуальный | https://www.nvidia.com/drivers |

### Установка Build Tools (обязательно для Windows)

1. Скачай **Build Tools for Visual Studio** по ссылке выше.
2. В установщике выбери компонент **"Desktop development with C++"**.
3. Установи (~3-5 ГБ).
4. Перезапусти терминал.

> VS Code — это другой продукт и для компиляции **не подходит**.

## Запуск

```bash
# Клонировать репозиторий
git clone https://github.com/USER/ARIA.git
cd ARIA

# Собрать (release)
cargo build --release

# Запустить чат
.\target\release\aria.exe
```

При первом запуске ARIA автоматически:
1. Создаёт папку `data base/` с пустыми файлами датасетов.
2. Строит BPE-словарь (32 000 токенов) из диалоговых данных.
3. Инициализирует Transformer модель (~124M параметров).
4. Проводит предобучение на данных из `data base/`.
5. Сохраняет чекпоинт в `aria json/aria_checkpoint.json` и токенизатор в `aria json/aria_tokenizer.json`.

При последующих запусках модель загружается из сохранённых файлов.

### Обучение с нуля

```powershell
.\target\release\train_fresh.exe
```

Строит словарь и обучает модель с нуля на файле `data base/DataBase_roles.jsonl`.

### Дообучение (SFT) от чекпоинта

```powershell
.\target\release\sft_train.exe
```

### Продолжение обучения через main

```powershell
$env:ARIA_CONTINUE_TRAIN="1"
.\target\release\aria.exe
```

### Тестирование модели

```powershell
# Жадная генерация по набору промптов
.\target\release\greedy_test.exe

# Top-K и Top-P тесты с разными температурами
.\target\release\sample_test.exe

# Полный набор тестов с сохранением отчёта
.\target\release\test_suite.exe

# Инференс по одному промпту
.\target\release\inference.exe "привет"

# Отладка логитов
.\target\release\debug_logits.exe
```

## Обучение

### Что нужно для обучения

- Видеокарта **NVIDIA** (обязательно, вычисления идут через CUDA/cuBLAS).
- VRAM: рекомендуется **8 ГБ** и более.
- RAM: от **16 ГБ**.
- Диалоговый датасет в формате JSONL в папке `data base/`.

### Файлы датасетов

```
data base/
  DataBase_roles.jsonl   - основной диалоговый датасет (формат: {"text": "..."})
  DataBase.txt           - дополнительные тексты (опционально)
  Words.txt              - словарный запас (опционально)
```

Каждая строка `DataBase_roles.jsonl` — отдельный диалог в формате:
```json
{"text": "Пользователь: привет\nАссистент: привет, чем могу помочь?"}
```

При обучении токенизатор автоматически вставляет ролевые токены `<USER>` / `<ASSISTANT>` и обучается только на токенах ассистента (маскирование потерь).

### Параметры обучения (переменные окружения)

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `ARIA_LR` | Learning rate (max после warmup) | 0.0003 |
| `ARIA_WARMUP` | Шагов линейного прогрева LR | 1000 |
| `ARIA_MAX_SEQS` | Максимум последовательностей на эпоху | 500 000 |
| `ARIA_EPOCHS` | Количество эпох | 5 |
| `ARIA_VOCAB_LINES` | Строк для построения словаря | 500 000 |
| `ARIA_CONTINUE_TRAIN` | Продолжить предобучение | — |

Gradient clipping всегда включён (norm=1.0), это часть тренировочного цикла и не отключается.

**LR schedule:** первые `ARIA_WARMUP` шагов lr растёт линейно от 0 до `ARIA_LR`. Затем cosine decay до 0.3 × `ARIA_LR` к концу обучения.

Пример — тестовый запуск на 100k последовательностей:

```powershell
$env:ARIA_MAX_SEQS="200000"
$env:ARIA_EPOCHS="1"
.\target\release\train_fresh.exe
```

Полное обучение:

```powershell
$env:ARIA_MAX_SEQS="500000"
$env:ARIA_EPOCHS="5"
.\target\release\train_fresh.exe > train_transformer.log
```

Ожидаемое время: ~6–7 часов на эпоху на RTX 4060 (скорость ~450–550 seq/s).

## Команды в диалоге

| Команда | Описание |
|---|---|
| `stats` | Показать статистику сессии |
| `settings` | Показать текущие настройки генерации |
| `mode greedy` | Жадная генерация (детерминированная) |
| `mode topk` | Top-K сэмплинг (k=20, по умолчанию) |
| `mode topp` | Nucleus (top-p) сэмплинг (p=0.9) |
| `temp <0.1-2.0>` | Установить температуру генерации |
| `topk <n>` | Установить значение K |
| `topp <0.0-1.0>` | Установить значение P |
| `exit` | Выйти |

## Файлы, создаваемые при работе

| Файл | Описание |
|---|---|
| `aria json/aria_checkpoint.gguf` | Веса модели + токенизатор + Adam моменты (GGUF формат) |
| `aria json/aria_inference.gguf` | Q4_0 инференс-модель без Adam моментов (создаётся через `export_gguf`) |
| `aria json/aria_dialogs.json` | Зашифрованная история диалогов |
| `data base/sequences_cache_transformer_v*.bin` | Кеш токенизированных последовательностей |
| `data base/sequences_cache_transformer_v*.bin.idx` | Индекс кэша для потокового чтения |
| `logs/validation_log.txt` | Отчёт после `test_suite` |

## Возможные проблемы

**`error: linker link.exe not found`**
Установи Visual Studio Build Tools с компонентом C++ (см. раздел выше).

**Модель не видит GPU**
Убедись, что установлен драйвер NVIDIA и CUDA Toolkit 12.x. CUDA должна быть доступна через `nvcc`.

**Несовместимый чекпоинт**
3.5.2 использует только GGUF. Старые `.json` и бинарные ARIA чекпоинты не поддерживаются — удали `aria json/aria_checkpoint.json` и запусти `train_fresh.exe` заново.

**Экспорт инференс-модели**
```powershell
.\target\release\export_gguf.exe "aria json/aria_checkpoint.gguf" "aria json/aria_inference.gguf"
```

**Медленное обучение**
Используй только `cargo build --release` и запускай `.exe` напрямую. Debug-сборка в 10-20 раз медленнее.

**Низкое качество генерации**
1. Проверь чистоту и формат данных в `data base/`.
2. Убедись, что диалоги имеют структуру `Пользователь: ...\nАссистент: ...`.
3. Проверь `logs/validation_log.txt` после `test_suite`.
4. Увеличь количество эпох или объём данных.
