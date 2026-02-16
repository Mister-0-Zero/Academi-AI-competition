import os
import numpy as np
import pandas as pd
from pandas import DataFrame

from ETL import Transform, date_split


def _add_labels(interactions: DataFrame) -> DataFrame:
    label_map = {1: 1, 2: 3}
    interactions = interactions.copy()
    interactions["label"] = interactions["event_type"].replace(label_map).fillna(0).astype("int64")
    return interactions


def _filter_window(interactions: DataFrame, start_ts, end_ts) -> DataFrame:
    if start_ts is None and end_ts is None:
        return interactions
    if start_ts is None:
        return interactions[interactions["event_ts"] < end_ts]
    if end_ts is None:
        return interactions[interactions["event_ts"] >= start_ts]
    return interactions[(interactions["event_ts"] >= start_ts) & (interactions["event_ts"] < end_ts)]


def _dedupe_positives(interactions: DataFrame) -> DataFrame:
    interactions = interactions.sort_values("event_ts")
    return (
        interactions.groupby(["user_id", "edition_id"], as_index=False)
        .agg(
            event_type=("event_type", "max"),
            event_ts=("event_ts", "max"),
            label=("label", "max"),
        )
    )


def _limit_positives(positives: DataFrame, limit: int) -> DataFrame:
    positives = positives.sort_values("event_ts", ascending=False)
    return positives.groupby("user_id", as_index=False).head(limit)


def _sample_negatives_for_user(
    user_id: int,
    pos_all: set,
    train_user_items: dict,
    pool_train: set,
    pool_other: set,
    total_needed: int,
    rng: np.random.Generator,
    neg_train_share: float,
) -> list[int]:
    if total_needed <= 0:
        return []
    excluded = set(train_user_items.get(user_id, set())) | set(pos_all)
    candidates_train = list(pool_train - excluded)
    candidates_other = list(pool_other - excluded)

    train_target = int(round(total_needed * neg_train_share))
    other_target = total_needed - train_target
    neg_items = set()

    if candidates_train and train_target > 0:
        k = min(train_target, len(candidates_train))
        neg_items.update(rng.choice(candidates_train, size=k, replace=False).tolist())

    if candidates_other and other_target > 0:
        k = min(other_target, len(candidates_other))
        neg_items.update(rng.choice(candidates_other, size=k, replace=False).tolist())

    if len(neg_items) < total_needed:
        remaining = list((pool_train | pool_other) - excluded - neg_items)
        if remaining:
            k = min(total_needed - len(neg_items), len(remaining))
            neg_items.update(rng.choice(remaining, size=k, replace=False).tolist())

    return list(neg_items)


def _build_stats_df(
    interactions: DataFrame,
    editions: DataFrame,
    users: DataFrame,
    genres_mas: DataFrame,
    train_end,
) -> DataFrame:
    train_df = _filter_window(interactions, None, train_end)
    train_df = train_df[train_df["event_type"].isin([1, 2])].copy()
    train_df = _add_labels(train_df)
    train_df = train_df.merge(editions, on="edition_id", how="left")
    train_df = train_df.merge(users, on="user_id", how="left")
    train_df = train_df.merge(genres_mas, on="book_id", how="left")
    return train_df


def _merge_base(
    candidates: DataFrame,
    editions: DataFrame,
    users: DataFrame,
    genres_mas: DataFrame,
    event_ts,
) -> DataFrame:
    df = candidates.merge(editions, on="edition_id", how="left")
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(genres_mas, on="book_id", how="left")
    if "label" in df.columns:
        df["event_type"] = df["label"].map({1: 1, 3: 2}).fillna(0).astype("int64")
    else:
        df["event_type"] = 0
    df["rating"] = np.nan
    df["event_ts"] = event_ts
    return df


def _build_candidates_for_month(
    month_index: int,
    interactions: DataFrame,
    editions: DataFrame,
    pos_limit: int,
    total_candidates: int,
    neg_train_share: float,
    rng: np.random.Generator,
) -> DataFrame:
    train_end = date_split[month_index - 1]
    val_start = date_split[month_index - 1]
    val_end = date_split[month_index] if month_index < len(date_split) else None

    train_df = _filter_window(interactions, None, train_end)
    train_df = train_df[train_df["event_type"].isin([1, 2])].copy()
    train_users = train_df["user_id"].unique().tolist()

    val_df = _filter_window(interactions, val_start, val_end)
    val_df = val_df[val_df["event_type"].isin([1, 2])].copy()
    val_df = _add_labels(val_df)
    val_pos_all = _dedupe_positives(val_df)
    val_pos_all = val_pos_all[val_pos_all["user_id"].isin(train_users)]
    val_pos_included = _limit_positives(val_pos_all, pos_limit)

    train_user_items = train_df.groupby("user_id")["edition_id"].apply(set).to_dict()
    pos_all_by_user = val_pos_all.groupby("user_id")["edition_id"].apply(set).to_dict()

    pool_all = set(editions["edition_id"].tolist())
    pool_train = set(train_df["edition_id"].tolist())
    pool_other = pool_all - pool_train

    rows = []
    pos_by_user = val_pos_included.groupby("user_id")

    for user_id in train_users:
        pos_all = pos_all_by_user.get(user_id, set())
        pos_rows = pos_by_user.get_group(user_id) if user_id in pos_by_user.groups else pd.DataFrame()
        pos_count = len(pos_rows)
        target_neg = max(0, total_candidates - pos_count)

        neg_items = _sample_negatives_for_user(
            user_id=user_id,
            pos_all=pos_all,
            train_user_items=train_user_items,
            pool_train=pool_train,
            pool_other=pool_other,
            total_needed=target_neg,
            rng=rng,
            neg_train_share=neg_train_share,
        )

        for _, row in pos_rows.iterrows():
            rows.append(
                {
                    "user_id": user_id,
                    "edition_id": int(row["edition_id"]),
                    "label": int(row["label"]),
                }
            )

        for edition_id in neg_items:
            rows.append(
                {
                    "user_id": user_id,
                    "edition_id": int(edition_id),
                    "label": 0,
                }
            )

    return pd.DataFrame(rows)


def create_candidates(
    path_dir_data: str = r"./data/data",
    path_dir_submit: str = r"./data/submit",
    path_save_dir: str = r"data/candidates",
    total_candidates: int = 100,
    pos_limit: int = 50,
    neg_train_share: float = 0.5,
    random_state: int | None = 42,
    kol_in_group_publ_year: int = 2,
    kol_in_group_users_year: int = 5,
    start_val_authors: int = 10,
    start_val_books: int = 10,
    kol_user_clusters: int = 75,
    kol_author_clusters: int = 45,
    rank_lambda: float = 0.5,
    sim_noise_scale: float = 0.01,
    cyclecally_by_month: bool = True,
) -> None:
    book_genres = pd.read_csv(path_dir_data + "/book_genres.csv")
    editions = pd.read_csv(path_dir_data + "/editions.csv")
    interactions = pd.read_csv(path_dir_data + "/interactions.csv")
    users = pd.read_csv(path_dir_data + "/users.csv")
    submit_candidates = pd.read_csv(path_dir_submit + "/candidates.csv")

    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])

    genres_mas = book_genres.groupby("book_id")["genre_id"].apply(list).reset_index()

    os.makedirs(path_save_dir, exist_ok=True)

    rng = np.random.default_rng(random_state)

    if cyclecally_by_month:
        for month_index in range(1, 6):
            print(f"Формируем кандидатов для месяца {month_index}")
            candidates = _build_candidates_for_month(
                month_index=month_index,
                interactions=interactions,
                editions=editions,
                pos_limit=pos_limit,
                total_candidates=total_candidates,
                neg_train_share=neg_train_share,
                rng=rng,
            )

            train_end = date_split[month_index - 1]
            stats_df = _build_stats_df(interactions, editions, users, genres_mas, train_end)

            event_ts = date_split[month_index] if month_index < len(date_split) else interactions["event_ts"].max()
            if event_ts is None:
                event_ts = interactions["event_ts"].max()
            candidates_base = _merge_base(candidates, editions, users, genres_mas, event_ts)

            candidates_features = Transform(
                candidates_base,
                kol_in_group_publ_year,
                kol_in_group_users_year,
                start_val_authors,
                start_val_books,
                kol_user_clusters,
                kol_author_clusters,
                random_state=random_state,
                rank_lambda=rank_lambda,
                sim_noise_scale=sim_noise_scale,
                stats_df=stats_df,
            )

            path_save = os.path.join(path_save_dir, f"dataset{month_index}_candidates.csv")
            candidates_features.to_csv(path_save, index=False)
            print(f"Сохранено: {path_save}")

    print("Формируем кандидатов для месяца 6")
    stats_df = _build_stats_df(interactions, editions, users, genres_mas, None)
    submit_base = submit_candidates.copy()
    event_ts = interactions["event_ts"].max()
    submit_base = _merge_base(submit_base, editions, users, genres_mas, event_ts)

    submit_features = Transform(
        submit_base,
        kol_in_group_publ_year,
        kol_in_group_users_year,
        start_val_authors,
        start_val_books,
        kol_user_clusters,
        kol_author_clusters,
        random_state=random_state,
        rank_lambda=rank_lambda,
        sim_noise_scale=sim_noise_scale,
        stats_df=stats_df,
    )

    path_save = os.path.join(path_save_dir, "dataset6_candidates.csv")
    submit_features.to_csv(path_save, index=False)
    print(f"Сохранено: {path_save}")


if __name__ == "__main__":
    create_candidates()
