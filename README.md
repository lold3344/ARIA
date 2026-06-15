# **This is a research tool. The author is not responsible for misuse, training data, or generated content.**

> **Legal notice:** I have nothing to do with anyone who uses this tool for illegal purposes. If you train my AI model and use it for hacking, criminal activity, or any other unlawful actions, that is entirely your own responsibility and problem.


# ARIA Wotan 3.1

**ARIA Wotan 3.1** - локальная языковая модель на основе LSTM с обучением через обратную связь (RLHF-like), написанная на Rust. Вычисления выполняются на NVIDIA GPU через CUDA/cuBLAS с кастомными PTX ядрами.

> **Важно:** ARIA поддерживает только видеокарты NVIDIA. Работа на AMD, Intel и других GPU не гарантируется.

> Текущая версия: **3.1.0**. Следующая планируемая версия: **ARIA Wotan 3.5**.

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

> VS Code - это другой продукт и для компиляции **не подходит**.

## Запуск

```bash
# Клонировать репозиторий
git clone https://github.com/USER/ARIA.git
cd ARIA

# Собрать и запустить (debug)
cargo run

# Собрать и запустить оптимизированную сборку
cargo run --release
```

При первом запуске ARIA автоматически:
1. Создаёт папку `data base/` с пустыми файлами датасетов.
2. Строит словарь из текстовых файлов в `data base/`.
3. Проводит предобучение (pre-training) на этих данных.
4. Сохраняет чекпоинт модели в `aria_checkpoint.json` и токенизатор в `aria_tokenizer.json`.

При последующих запусках модель загружается из сохранённых файлов.

### Продолжение обучения

Чтобы продолжить предобучение с последнего чекпоинта:

```bash
set ARIA_CONTINUE_TRAIN=1
cargo run --release
```

### Тестирование модели

```bash
# Базовый инференс по промпту
cargo run --release --bin inference "привет"

# Полный набор тестов и сохранение отчёта
cargo run --release --bin test_suite
```

## Обучение

### Что нужно для обучения

- Видеокарта **NVIDIA** (обязательно, вычисления идут через CUDA/cuBLAS).
- VRAM: рекомендуется от **8 ГБ** (модель: embed=1024, hidden=2048, ~44.5M параметров).
- RAM: от **16 ГБ**.
- Текстовые датасеты в папке `data base/`.

### Файлы датасетов

Положи свои тексты в папку `data base/` (создаётся автоматически):

```
data base/
  DataBase.txt     - основной датасет (книги, диалоги, новости, статьи и т.д.)
  Words.txt        - словарный запас
```

Чем больше чистого и разнообразного текста, тем лучше качество. Рекомендуемый минимум: **50-100 МБ** суммарно.

### Параметры обучения (переменные окружения)

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `ARIA_LR` | Learning rate | 0.0003 |
| `ARIA_CLIP` | Gradient clipping | 5.0 |
| `ARIA_LSTM_OPT` | Оптимизировать LSTM часть | 1 |
| `ARIA_ASM_OPT` | Оптимизировать Adaptive Softmax | 1 |
| `ARIA_CONTINUE_TRAIN` | Продолжить предобучение | 0 |

Пример:

```bash
set ARIA_LSTM_OPT=1
set ARIA_ASM_OPT=1
set ARIA_CLIP=5.0
set ARIA_LR=0.0003
cargo run --release
```

### Процесс обучения

ARIA обучается в двух режимах:

**1. Предобучение (pre-training)** - происходит автоматически при первом запуске на основе файлов из `data base/`. Модель учится предсказывать следующий токен.

**2. Дообучение через фидбек (RLHF)** - происходит в реальном времени во время диалога. После каждого ответа ARIA спрашивает оценку:

```
Rate (I like / I dont like / skip):
```

- `I like` - положительный сигнал, модель усиливает этот паттерн.
- `I dont like` - отрицательный сигнал, модель подавляет этот паттерн.
- `skip` (или Enter) - пропустить без обновления.

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
| `aria_checkpoint.json` | Веса модели, конфигурация и состояние оптимизатора |
| `aria_tokenizer.json` | Словарь токенизатора (BPE) |
| `aria_dialogs.json` | Зашифрованная история диалогов |
| `data base/` | Папка с обучающими текстами |

## Возможные проблемы

**`error: linker link.exe not found`**
Установи Visual Studio Build Tools с компонентом C++ (см. раздел выше).

**Модель не видит GPU**
Убедись, что установлен драйвер NVIDIA и CUDA Toolkit. CUDA должна быть доступна через `nvcc`.

**Медленное предобучение**
Используй `cargo build --release` и запускай `.exe` напрямую. Для ускорения можно уменьшить `MAX_TOKENS_PER_SEQ` в `src/model_cuda.rs`, но это повлияет на контекст.

**Низкое качество генерации**
1. Проверь чистоту данных в `data base/`.
2. Убедись, что в тексте достаточно диалогов и вопросов.
3. Оценивай ответы через `I like / I dont like` для RLHF.
4. Проверь `validation_log.txt` после `test_suite`.
