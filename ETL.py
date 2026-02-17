import numpy as np
import pandas as pd
from pandas import DataFrame
from collections.abc import Iterable

from agregation import prosessing_NaN, aggr_by_publ_year, aggr_by_number_years_users,\
                       aggr_by_authors, aggr_by_lang, aggr_by_books, aggr_by_genre,\
                       aggr_user_genre, aggr_user_author_cnt, log1p_feature,\
                       total_rank, add_clusters_and_dis


date_split = [pd.Timestamp("2024-11-14"), pd.Timestamp("2024-12-14"), pd.Timestamp("2025-01-14"),\
              pd.Timestamp("2025-02-14"), pd.Timestamp("2025-03-14"), None]

def _add_labels(interactions):
    label_map = {1: 1, 2: 2}
    interactions = interactions.copy()
    interactions["label"] = (
        interactions["event_type"].replace(label_map).fillna(0).astype("int64")
    )
    return interactions

def _dedupe_positives(interactions):
    interactions = interactions.sort_values("event_ts")
    return (
        interactions.groupby(["user_id", "edition_id"], as_index=False)
        .agg(
            event_type=("event_type", "max"),
            rating=("rating", "max"),
            event_ts=("event_ts", "max"),
            label=("label", "max"),
            genre_id=("genre_id", "first")
        )
    )

def _sample_negatives(
    positives,
    all_editions: Iterable[int],
    editions: DataFrame,
    book_genres: DataFrame,
    neg_min: int,
    neg_max: int,
    random_state: int | None,
    popular_share: float = 0.3,
    genre_share: float = 0.2,
):
    rng = np.random.default_rng(random_state)
    pool_all = set(all_editions)

    def _flatten_genres(series):
        genres = set()
        for item in series:
            if isinstance(item, list):
                genres.update([g for g in item if pd.notna(g)])
        return genres

    popular_list = (
        positives.groupby("edition_id").size().sort_values(ascending=False).index.tolist()
    )
    top_count = int(len(popular_list) * popular_share)
    popular_pool = set(popular_list[:top_count]) if top_count > 0 else set()

    edition_genres = (
        editions[["edition_id", "book_id"]]
        .merge(book_genres, on="book_id", how="left")
        .groupby("edition_id")["genre_id"]
        .apply(lambda x: set(x.dropna().tolist()))
        .to_dict()
    )

    genre_to_editions = {}
    for eid, genres in edition_genres.items():
        for g in genres:
            genre_to_editions.setdefault(g, set()).add(eid)

    user_genres = positives.groupby("user_id")["genre_id"].apply(_flatten_genres).to_dict()

    min_ts = positives["event_ts"].min()
    max_ts = positives["event_ts"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        min_ts = max_ts = pd.Timestamp.utcnow()
    min_val = min_ts.value
    max_val = max_ts.value
    neg_rows = []

    positives_by_user = positives.groupby("user_id")["edition_id"].apply(set)
    for user_id, pos_items in positives_by_user.items():
        pos_count = len(pos_items)
        if pos_count == 0:
            continue

        neg_total = int(rng.integers(neg_min, neg_max + 1, size=pos_count).sum())
        pop_target = int(round(neg_total * popular_share))
        genre_target = int(round(neg_total * genre_share))
        rand_target = max(0, neg_total - pop_target - genre_target)

        neg_items = set()
        excluded = set(pos_items)

        if pop_target > 0 and popular_pool:
            pop_candidates = list(popular_pool - excluded)
            k = min(pop_target, len(pop_candidates))
            if k > 0:
                neg_items.update(rng.choice(pop_candidates, size=k, replace=False).tolist())
                excluded.update(neg_items)

        if genre_target > 0:
            user_genre_set = user_genres.get(user_id, set())
            if user_genre_set:
                genre_candidates = set()
                for g in user_genre_set:
                    genre_candidates.update(genre_to_editions.get(g, set()))
                if popular_pool:
                    genre_candidates = genre_candidates & popular_pool
                genre_candidates = list(genre_candidates - excluded)
                k = min(genre_target, len(genre_candidates))
                if k > 0:
                    neg_items.update(rng.choice(genre_candidates, size=k, replace=False).tolist())
                    excluded.update(neg_items)

        if rand_target > 0:
            rand_candidates = list(pool_all - excluded)
            k = min(rand_target, len(rand_candidates))
            if k > 0:
                neg_items.update(rng.choice(rand_candidates, size=k, replace=False).tolist())

        if len(neg_items) < neg_total:
            remaining = list(pool_all - pos_items - neg_items)
            if remaining:
                k = min(neg_total - len(neg_items), len(remaining))
                neg_items.update(rng.choice(remaining, size=k, replace=False).tolist())

        for edition_id in neg_items:
            if min_val < max_val:
                ts_val = rng.integers(min_val, max_val + 1)
                event_ts = pd.to_datetime(ts_val, unit="ns")
            else:
                event_ts = max_ts
            neg_rows.append(
                {
                    "user_id": user_id,
                    "edition_id": edition_id,
                    "event_type": 0,
                    "rating": np.nan,
                    "event_ts": event_ts,
                    "label": 0,
                }
            )

    return pd.DataFrame(neg_rows)

def _fill_rank_exponential(df: DataFrame, column: str, rng: np.random.Generator, rank_lambda: float) -> None:
    if column not in df.columns:
        return
    mask = df[column].isna()
    missing_count = int(mask.sum())
    if missing_count == 0:
        return
    ranks = np.arange(1, 11)
    probs = np.exp(-rank_lambda * (ranks - 1))
    probs = probs / probs.sum()
    df.loc[mask, column] = rng.choice(ranks, size=missing_count, p=probs)
    df[column] = df[column].astype("int64")

def _impute_user_author_sim(df: DataFrame, rng: np.random.Generator, noise_scale: float) -> None:
    if "user_author_sim" not in df.columns:
        return
    mask = df["user_author_sim"].isna()
    missing_count = int(mask.sum())
    if missing_count == 0:
        return
    if "label" in df.columns:
        median_sim = df.loc[df["label"] > 0, "user_author_sim"].median()
    else:
        median_sim = df["user_author_sim"].median()
    if pd.isna(median_sim):
        median_sim = df["user_author_sim"].median()
    if pd.isna(median_sim):
        median_sim = 0.5
    noise = rng.normal(0.0, noise_scale, size=missing_count)
    df.loc[mask, "user_author_sim"] = np.clip(median_sim + noise, 0.0, 1.0)

def _fill_clusters(df: DataFrame) -> None:
    if "author_cluster" in df.columns:
        df["author_cluster"] = df["author_cluster"].fillna(-1).astype("int64")
    if "user_cluster" in df.columns:
        df["user_cluster"] = df["user_cluster"].fillna(-1).astype("int64")

def Extract(
    path_dir_data: str,
    num_months: int | None,
    add_negatives: bool = False,
    neg_min: int = 10,
    neg_max: int = 20,
    random_state: int | None = 42,
) -> DataFrame:

    #Считывание данных
    book_genres = pd.read_csv(path_dir_data + "/book_genres.csv")
    editions = pd.read_csv(path_dir_data + "/editions.csv")
    interactions = pd.read_csv(path_dir_data + "/interactions.csv")
    users = pd.read_csv(path_dir_data + "/users.csv")

    #Ограничение по времени
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])
    if num_months is not None and date_split[num_months - 1] is not None:
        interactions = interactions[interactions["event_ts"] < date_split[num_months - 1]]

    #Составление позитива с жанрами
    interactions = _add_labels(interactions)
    genres_mas = (book_genres.groupby("book_id")["genre_id"].apply(list).reset_index())
    df_ = interactions.merge(editions, on="edition_id", how="left")
    df_ = df_.merge(users, on="user_id", how="left")
    df_ = df_.merge(genres_mas, on="book_id", how="left")
    df_ = df_[["user_id", "edition_id", "event_type", "rating", "event_ts", "genre_id", "label"]]

    if add_negatives:
        positives = pd.DataFrame(_dedupe_positives(df_))
        negatives = pd.DataFrame(
            _sample_negatives(
                positives=positives,
                all_editions=editions["edition_id"],
                editions=editions,
                book_genres=book_genres,
                neg_min=neg_min,
                neg_max=neg_max,
                random_state=random_state,
            )
        )
        interactions = pd.concat([positives, negatives], ignore_index=True)

    if "genre_id" in interactions.columns:
        interactions = interactions.drop(columns=["genre_id"])

    genres_mas = (book_genres.groupby("book_id")["genre_id"].apply(list).reset_index())

    df = interactions.merge(editions, on="edition_id", how="left")
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(genres_mas, on="book_id", how="left")

    return df

def Transform(df: DataFrame, kol_in_group_publ_year: int, kol_in_group_users_year: int,
              start_val_authors: int, start_val_books: int,
              kol_user_clusters: int, kol_author_clusters: int,
              random_state: int | None = None,
              rank_lambda: float = 0.5,
              sim_noise_scale: float = 0.01,
              stats_df: DataFrame | None = None) -> DataFrame:
    #Обработка NaN
    df = prosessing_NaN(df)
    rng = np.random.default_rng(random_state)

    stats_df_base = stats_df
    if stats_df_base is None and "label" in df.columns:
        stats_df_base = df[df["label"] > 0].copy()
        if stats_df_base.empty:
            stats_df_base = None
    if stats_df_base is not None:
        stats_df_base = prosessing_NaN(stats_df_base)

    #Удаляем лишние столбцы
    df.drop(columns=["description", "title", "age_restriction", "event_type"], inplace=True)

    #Агрегация по годам публикации
    df = aggr_by_publ_year(df, kol_in_group_publ_year, stats_df=stats_df_base)

    #Агрегация по количеству лет пользователей(угруппы пользователей по 5 лет)
    df = aggr_by_number_years_users(df, kol_in_group_users_year, stats_df=stats_df_base)

    #Агрегация по авторам
    df = aggr_by_authors(df, start_val_authors, stats_df=stats_df_base)

    #Агрегации по языкам(основной ли в книге язык и на рзных ли языках книги пользователь читал)
    df = aggr_by_lang(df, stats_df=stats_df_base)

    #Агрегации по книгам(даю ранги книгам в зависимости от частоты их использования)
    df = aggr_by_books(df, start_val_books, stats_df=stats_df_base)

    _fill_rank_exponential(df, "author_rank", rng, rank_lambda)
    _fill_rank_exponential(df, "book_rank", rng, rank_lambda)
    _fill_rank_exponential(df, "year_rank", rng, rank_lambda)

    #Агрегации по жанрам
    df = aggr_by_genre(df, stats_df=stats_df_base)

    #Агрегация польователь\жанры(сколько различных жанров прочитал пользователь)
    df = aggr_user_genre(df, stats_df=stats_df_base)

    #Агрегация сколько раз пользователь взаимодействовал с данным автором
    df = aggr_user_author_cnt(df, stats_df=stats_df_base)

    #Логарифмирования больших величин
    df = log1p_feature(df)

    #Присваивание общего ранга
    df = total_rank(df)

    #Добавление векторного представление пользователя и автора + их близость
    stats_df_features = None
    if stats_df_base is not None:
        stats_df_features = stats_df_base.copy()
        stats_df_features = aggr_by_publ_year(stats_df_features, kol_in_group_publ_year, stats_df=stats_df_base)
        stats_df_features = aggr_by_number_years_users(stats_df_features, kol_in_group_users_year, stats_df=stats_df_base)
        stats_df_features = aggr_by_authors(stats_df_features, start_val_authors, stats_df=stats_df_base)
        stats_df_features = aggr_by_lang(stats_df_features, stats_df=stats_df_base)
        stats_df_features = aggr_by_books(stats_df_features, start_val_books, stats_df=stats_df_base)
        stats_df_features = aggr_by_genre(stats_df_features, stats_df=stats_df_base)
        stats_df_features = aggr_user_genre(stats_df_features, stats_df=stats_df_base)
        stats_df_features = aggr_user_author_cnt(stats_df_features, stats_df=stats_df_base)
        stats_df_features = log1p_feature(stats_df_features)
        stats_df_features = total_rank(stats_df_features)

    df = add_clusters_and_dis(df, kol_user_clusters, kol_author_clusters, stats_df=stats_df_features)

    _impute_user_author_sim(df, rng, sim_noise_scale)
    _fill_clusters(df)

    #Последние удаление лишних фич
    df.drop(columns=["publisher_id", "author_id", "book_id", "event_ts",\
                     "rating", "genre_id"], inplace=True)

    return df

def Load(df: DataFrame, path_save: str, expansion: str) -> None:
    if expansion == "csv":
        df.to_csv(path_save, index=False)
    elif expansion == "xlsx":
        df.to_excel(path_save, index=False)
    else:
        print(f"Ваш формат сохранения: {expansion}")
        raise ValueError("Формат сохранения не поддерживается, поддерживается только: csv, xlsx")

def ETL_function(path_dir_data: str =r"./data/data", num_months: int|None =None,
        kol_in_group_publ_year: int =2, kol_in_group_users_year: int =5,
        start_val_authors: int =10, start_val_books: int=10,
        kol_user_clusters: int =75, kol_author_clusters: int=45,
        path_save: str =r"data/after_transform_csv/dataset.csv",
        cyclecally_by_month: bool =False,
        add_negatives: bool =True,
        neg_min: int =10,
        neg_max: int =20,
        random_state: int | None =42,
        rank_lambda: float =0.5,
        sim_noise_scale: float =0.01):

    if cyclecally_by_month:
        for ind in range(len(date_split) - 1):
            print(f"Формируем датафрейм длиной логов в {ind + 1} месяц")
            print("Запуск Extract")
            df = Extract(
                path_dir_data,
                num_months=ind + 1,
                add_negatives=add_negatives,
                neg_min=neg_min,
                neg_max=neg_max,
                random_state=random_state,
            )
            print("Датафрейм сформировался", "\n")

            print("Запуск Transform")
            df = Transform(df, kol_in_group_publ_year, kol_in_group_users_year,
                           start_val_authors, start_val_books,
                           kol_user_clusters, kol_author_clusters,
                           random_state=random_state,
                           rank_lambda=rank_lambda,
                           sim_noise_scale=sim_noise_scale)
            print("Трансформации выполнены", "\n")

            print("Запуск Load")
            path_save_ = path_save.split(".")[0] + f"{ind + 1}" + "." + path_save.split(".")[1]
            expansion = path_save.split(".")[1]
            Load(df, path_save_, expansion)
            print(f"Датафрейм сохранен по пути: {path_save_}", "\n\n")
    else:
        print("Запуск Extract")
        df = Extract(
            path_dir_data,
            num_months,
            add_negatives=add_negatives,
            neg_min=neg_min,
            neg_max=neg_max,
            random_state=random_state,
        )
        print("Датафрейм сформировался", "\n")

        print("Запуск Transform")
        df = Transform(df, kol_in_group_publ_year, kol_in_group_users_year,
                       start_val_authors, start_val_books,
                       kol_user_clusters, kol_author_clusters,
                       random_state=random_state,
                       rank_lambda=rank_lambda,
                       sim_noise_scale=sim_noise_scale)
        print("Трансформации выполнены", "\n")

        print("Запуск Load")
        expansion = path_save.split(".")[1]
        Load(df, path_save, expansion)
        print(f"Датафрейм сохранен по пути: {path_save}")
