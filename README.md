# Проект рекомендации книг (ODS.AI Academy Hackathon)

## Обзор
Репозиторий содержит пайплайн подготовки данных, генерации кандидатов, обучения модели ранжирования (CatBoostRanker) и формирования сабмита. Задача: для каждого пользователя сформировать ранжированный список из 20 изданий (edition_id) из пула 200 кандидатов на следующий месяц, с балансом релевантности и жанрового разнообразия.

Ключевые файлы:
- `ETL.py` — Extract/Transform/Load, генерация train с негативами и метками `0/1/3`.
- `agregation.py` — фичи, агрегаты, ранги, кластеры, близость user-author.
- `create_candidates.py` — кандидаты по месяцам (1–5 для валидации, 6 — submit).
- `main.ipynb` — обучение CatBoostRanker, rerank с diversity, расчёт метрики, сабмит.

## Структура данных
Ожидаемые файлы (относительно корня проекта):

```
data/
  data/
    interactions.csv
    editions.csv
    users.csv
    book_genres.csv
  submit/
    candidates.csv
```

Выходные артефакты:

```
data/
  after_transform_csv/
    dataset1.csv ... dataset5.csv
  candidates/
    dataset1_candidates.csv ... dataset5_candidates.csv
    submit_candidates.csv
submission.csv
```

## Лейблы и негативы
- `event_type = 2 (read)` → `label = 3`
- `event_type = 1 (wishlist)` → `label = 1`
- отсутствие взаимодействия → `label = 0`

Негативы сэмплируются случайно:
- часть книг берётся из пула, встречавшегося в train-окне,
- часть — из остальных книг.

Соотношение негативов задаётся случайно **от 5 до 10 на один позитив**. Это задаётся параметрами ETL.

## Временной сплит
Разбиение по месяцам фиксировано в `ETL.py`:

```
date_split = [
  2024-11-14,
  2024-12-14,
  2025-01-14,
  2025-02-14,
  2025-03-14,
  None
]
```

Принцип:
- train-окно: все события раньше `date_split[k]`.
- валидация: события между `date_split[k]` и `date_split[k+1]`.
- сабмит: candidates из `data/submit/candidates.csv`.

## Быстрый старт

### 1) Сгенерировать train (с негативами)

```python
from ETL import ETL_function

ETL_function(
    cyclecally_by_month=True,
    add_negatives=True,
    neg_min=5,
    neg_max=10,
    neg_train_share=0.5,
    random_state=42,
    rank_lambda=0.5,
    sim_noise_scale=0.01,
)
```

### 2) Сгенерировать кандидатов

```python
from create_candidates import create_candidates

create_candidates(
    total_candidates=100,
    pos_limit=50,
    neg_train_share=0.5,
    random_state=42,
    rank_lambda=0.5,
    sim_noise_scale=0.01,
)
```

### 3) Обучение и сабмит
Открыть `main.ipynb` и последовательно выполнить ячейки:
- загрузка датасетов
- обучение CatBoostRanker
- rerank top-30 → top-20 с diversity
- сохранение `submission.csv`

## Фичи
Основные типы фичей:
- **Пользовательские**: возраст, возрастной ранг, мульти-язычность, количество жанров.
- **Книжные**: год публикации + ранг, язык, жанровая популярность.
- **Авторские/книжные ранги**: `author_rank`, `book_rank`.
- **История**: `user_author_hits`, `user_genre_cnt`.
- **Кластеры**: `user_cluster`, `author_cluster`.
- **Сходство**: `user_author_sim`.

### Категориальные признаки (CatBoost)
Рекомендуемые категориальные:
```
main_genre, language_id, author_cluster, user_cluster, publication_year
```

Важно: `author_cluster` и `user_cluster` приводятся к `int64` и заполняются `-1`, чтобы CatBoost принимал их как категориальные.

## Rerank с diversity
В `main.ipynb` реализован rerank:
1. Модель отдаёт топ‑30 по score.
2. Жадный отбор формирует топ‑20 с учётом diversity по жанрам:
   - **Coverage**: новые жанры
   - **ILD**: Jaccard distance между жанрами

Параметры:
- `alpha=0.7` — вес релевантности
- `beta=0.5` — баланс coverage/ILD

## Метрика
Итоговый score:

```
Score = 0.7 * NDCG@20 + 0.3 * Diversity@20
```

Релевантность:
- read → 3
- wishlist → 1

Diversity:
- coverage по новым жанрам
- ILD (Jaccard distance)

Функция `compute_score(...)` реализована в `main.ipynb`.

## Частые проблемы
1. **NaN в признаках** — модель не обучится. Проверь `df.isna().sum()`.
2. **Категориальные float** — CatBoost падает. Все cat‑признаки должны быть `int` или `string`.
3. **Появление `Unnamed: 0`** — это индекс, записанный при `to_csv`. Используй `index=False`.

## Примечания
- `create_candidates.py` формирует candidates 1–5 с `label`, а `submit_candidates.csv` строится из `data/submit/candidates.csv`.
- Валидация строится по следующему месяцу относительно train‑окна.
- Для ускорения можно уменьшить `iterations`, `depth` или `top_n` в rerank.

## Контакты/заметки
Если нужно расширить логику (например, более умный negative sampling или другие признаки) — лучше делать это в `agregation.py` и затем пересобрать датасеты.
