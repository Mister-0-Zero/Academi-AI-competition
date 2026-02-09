# pyright: ignore
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


DataFrameLike = Any
SeriesLike = Any
TimestampLike = Any

READ_EVENT = 2
WISHLIST_EVENT = 1
DEFAULT_LANGUAGE_ID = 119


@dataclass
class EtlConfig:
    valid_days: int = 30
    gap_days: int = 1
    negative_ratio: float = 1.0
    sample_train_negatives: bool = True
    keep_users_without_positives: bool = False
    user_clusters: int = 75
    author_clusters: int = 45
    random_state: int = 42


def build_datasets(
    interactions: DataFrameLike,
    editions: DataFrameLike,
    users: DataFrameLike,
    book_genres: DataFrameLike,
    candidates: DataFrameLike,
    make_valid: bool = True,
    config: Optional[EtlConfig] = None,
) -> Tuple[DataFrameLike, Optional[DataFrameLike], DataFrameLike]:
    if config is None:
        config = EtlConfig()

    interactions = interactions.copy()
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])

    max_ts = interactions["event_ts"].max()
    if make_valid and config.valid_days and config.valid_days > 0:
        valid_end = max_ts
        valid_start = max_ts - pd.Timedelta(days=config.valid_days)
        history_end = valid_start - pd.Timedelta(days=config.gap_days)
    else:
        valid_start = None
        valid_end = None
        history_end = max_ts

    base = _prepare_base_tables(interactions, editions, users, book_genres)

    history = base.interactions[base.interactions["event_ts"] <= history_end].copy()
    history_enriched = _build_history_enriched(
        history,
        base.editions_base,
        base.users_base,
        base.genres_by_book,
    )

    feature_state = _build_feature_state(history_enriched, base)
    feature_df = _build_feature_frame(
        candidates,
        feature_state,
        config,
    )

    train_labels = _label_candidates(
        interactions,
        window_start=interactions["event_ts"].min(),
        window_end=history_end,
    )
    train_df = feature_df.merge(train_labels, on=["user_id", "edition_id"], how="left")
    train_df = _fill_missing_labels(train_df)
    if config.sample_train_negatives:
        train_df = _sample_negatives(
            train_df,
            negative_ratio=config.negative_ratio,
            keep_users_without_positives=config.keep_users_without_positives,
            random_state=config.random_state,
        )

    valid_df: Optional[pd.DataFrame]
    if valid_start is None or valid_end is None:
        valid_df = None
    else:
        valid_labels = _label_candidates(
            interactions,
            window_start=valid_start,
            window_end=valid_end,
        )
        valid_df = feature_df.merge(valid_labels, on=["user_id", "edition_id"], how="left")
        valid_df = _fill_missing_labels(valid_df)

    test_df = feature_df.copy()

    return train_df, valid_df, test_df


def build_datasets_xy(
    interactions: DataFrameLike,
    editions: DataFrameLike,
    users: DataFrameLike,
    book_genres: DataFrameLike,
    candidates: DataFrameLike,
    make_valid: bool = True,
    config: Optional[EtlConfig] = None,
    target: str | list[str] = "rel",
) -> Tuple[
    DataFrameLike,
    SeriesLike,
    Optional[DataFrameLike],
    Optional[SeriesLike],
    DataFrameLike,
]:
    train_df, valid_df, test_df = build_datasets(
        interactions=interactions,
        editions=editions,
        users=users,
        book_genres=book_genres,
        candidates=candidates,
        make_valid=make_valid,
        config=config,
    )

    label_cols = ["rel", "y_read", "y_wishlist"]
    x_train = train_df.drop(columns=[c for c in label_cols if c in train_df.columns])
    y_train = train_df[target] if isinstance(target, str) else train_df[target]

    if valid_df is None:
        x_valid = None
        y_valid = None
    else:
        x_valid = valid_df.drop(columns=[c for c in label_cols if c in valid_df.columns])
        y_valid = valid_df[target] if isinstance(target, str) else valid_df[target]

    x_test = test_df.copy()
    return x_train, y_train, x_valid, y_valid, x_test


def build_rolling_train(
    interactions: DataFrameLike,
    editions: DataFrameLike,
    users: DataFrameLike,
    book_genres: DataFrameLike,
    candidates: DataFrameLike,
    window_days: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    step_days: Optional[int] = None,
    config: Optional[EtlConfig] = None,
) -> DataFrameLike:
    if config is None:
        config = EtlConfig()
    if step_days is None:
        step_days = window_days

    interactions = interactions.copy()
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])

    min_ts = interactions["event_ts"].min()
    max_ts = interactions["event_ts"].max()

    start_ts = pd.to_datetime(start_date) if start_date else min_ts
    end_ts = pd.to_datetime(end_date) if end_date else max_ts

    base = _prepare_base_tables(interactions, editions, users, book_genres)
    windows = _make_windows(start_ts, end_ts, window_days, step_days)

    all_parts = []
    for window_start, window_end in windows:
        history_end = window_start - pd.Timedelta(days=config.gap_days)
        history = base.interactions[base.interactions["event_ts"] <= history_end].copy()
        history_enriched = _build_history_enriched(
            history,
            base.editions_base,
            base.users_base,
            base.genres_by_book,
        )
        feature_state = _build_feature_state(history_enriched, base)
        feature_df = _build_feature_frame(candidates, feature_state, config)

        labels = _label_candidates(
            interactions,
            window_start=window_start,
            window_end=window_end,
        )
        train_df = feature_df.merge(labels, on=["user_id", "edition_id"], how="left")
        train_df = _fill_missing_labels(train_df)
        if config.sample_train_negatives:
            train_df = _sample_negatives(
                train_df,
                negative_ratio=config.negative_ratio,
                keep_users_without_positives=config.keep_users_without_positives,
                random_state=config.random_state,
            )
        if not train_df.empty:
            all_parts.append(train_df)

    if not all_parts:
        return pd.DataFrame()
    return pd.concat(all_parts, ignore_index=True)


@dataclass
class _BaseTables:
    interactions: DataFrameLike
    editions_base: DataFrameLike
    users_base: DataFrameLike
    genres_by_book: DataFrameLike


@dataclass
class _FeatureState:
    item_features: DataFrameLike
    user_features: DataFrameLike
    user_genre_cnt: SeriesLike
    user_author_cnt: SeriesLike
    user_vec: DataFrameLike
    author_vec: DataFrameLike


def _prepare_base_tables(
    interactions: DataFrameLike,
    editions: DataFrameLike,
    users: DataFrameLike,
    book_genres: DataFrameLike,
) -> _BaseTables:
    editions_base = editions[
        [
            "edition_id",
            "book_id",
            "author_id",
            "publication_year",
            "age_restriction",
            "language_id",
            "publisher_id",
            "title",
            "description",
        ]
    ].copy()

    users_base = users[["user_id", "gender", "age"]].copy()
    users_base = _fill_user_fields(users_base)

    genres_by_book = (
        book_genres.groupby("book_id")["genre_id"].apply(list).reset_index()
    )

    return _BaseTables(
        interactions=interactions,
        editions_base=editions_base,
        users_base=users_base,
        genres_by_book=genres_by_book,
    )


def _fill_user_fields(users: DataFrameLike) -> DataFrameLike:
    users = users.copy()
    gender_mode = users["gender"].mode(dropna=True)
    gender_mode = gender_mode.iloc[0] if not gender_mode.empty else 0
    age_mean = users["age"].mean()
    age_mean = int(round(age_mean)) if not np.isnan(age_mean) else 0
    users["gender"] = users["gender"].fillna(gender_mode).astype(int)
    users["age"] = users["age"].fillna(age_mean).astype(int)
    return users


def _build_history_enriched(
    history: DataFrameLike,
    editions_base: DataFrameLike,
    users_base: DataFrameLike,
    genres_by_book: DataFrameLike,
) -> DataFrameLike:
    history = history.merge(editions_base, on="edition_id", how="left")
    history = history.merge(users_base, on="user_id", how="left")
    history = history.merge(genres_by_book, on="book_id", how="left")
    history["genre_id"] = history["genre_id"].apply(_ensure_list)
    history["gender"] = history["gender"].fillna(0).astype(int)
    history["age"] = history["age"].fillna(0).astype(int)
    return history


def _build_feature_state(
    history_enriched: DataFrameLike,
    base: _BaseTables,
) -> _FeatureState:
    year_rank_map = _build_year_rank_map(history_enriched)
    age_rank_map = _build_age_rank_map(history_enriched)
    author_rank_map = _build_rank_map(
        history_enriched,
        key="author_id",
        score_mass=_score_mass_double(),
    )
    book_rank_map = _build_rank_map(
        history_enriched,
        key="book_id",
        score_mass=_score_mass_pow_175(),
    )

    top_genres = (
        history_enriched.explode("genre_id")["genre_id"]
        .value_counts()
        .sort_values(ascending=False)
    )

    item_features = _build_item_features(
        base.editions_base,
        base.genres_by_book,
        year_rank_map,
        author_rank_map,
        book_rank_map,
        top_genres,
    )

    user_features = _build_user_features(
        base.users_base,
        history_enriched,
        age_rank_map,
    )

    user_genre_cnt = (
        history_enriched.explode("genre_id")
        .groupby(["user_id", "genre_id"])
        .size()
    )
    user_author_cnt = (
        history_enriched.groupby(["user_id", "author_id"]).size()
    )

    user_multi_lang_flag = (
        history_enriched.groupby("user_id")["language_id"].nunique().gt(1).astype(int)
    )
    history_enriched["user_multi_lang_flag"] = history_enriched["user_id"].map(
        user_multi_lang_flag
    )

    history_enriched = history_enriched.merge(
        item_features[
            [
                "edition_id",
                "year_rank",
                "author_rank",
                "book_rank",
                "n_genres",
                "genre_popularity_mean",
            ]
        ],
        on="edition_id",
        how="left",
    )

    history_enriched["total_rank"] = (
        history_enriched[["author_rank", "book_rank", "year_rank"]]
        .fillna(0)
        .sum(axis=1)
    )

    user_vec = history_enriched.groupby("user_id").agg(
        genre_pop_mean=("genre_popularity_mean", "mean"),
        language_multi_flag=("user_multi_lang_flag", "mean"),
        rating_mean=("rating", "mean"),
        author_rank_mean=("author_rank", "mean"),
        book_rank_mean=("book_rank", "mean"),
        year_rank_mean=("year_rank", "mean"),
        total_rank_mean=("total_rank", "mean"),
        n_genres_mean=("n_genres", "mean"),
    )
    author_vec = history_enriched.groupby("author_id").agg(
        genre_pop_mean=("genre_popularity_mean", "mean"),
        year_rank_mean=("year_rank", "mean"),
        book_rank_mean=("book_rank", "mean"),
        n_genres_mean=("n_genres", "mean"),
        rating_mean=("rating", "mean"),
    )

    user_vec = _normalize_df(user_vec)
    author_vec = _normalize_df(author_vec)

    return _FeatureState(
        item_features=item_features,
        user_features=user_features,
        user_genre_cnt=user_genre_cnt,
        user_author_cnt=user_author_cnt,
        user_vec=user_vec,
        author_vec=author_vec,
    )


def _build_item_features(
    editions_base: DataFrameLike,
    genres_by_book: DataFrameLike,
    year_rank_map: dict,
    author_rank_map: dict,
    book_rank_map: dict,
    top_genres: SeriesLike,
) -> DataFrameLike:
    item_features = editions_base.merge(genres_by_book, on="book_id", how="left")
    item_features["genre_id"] = item_features["genre_id"].apply(_ensure_list)

    item_features["n_genres"] = item_features["genre_id"].apply(len)
    item_features["main_genre"] = item_features["genre_id"].apply(
        lambda xs: _main_genre(xs, top_genres)
    )
    item_features["genre_popularity_sum"] = item_features["genre_id"].apply(
        lambda xs: sum(top_genres.get(g, 0) for g in xs)
    )
    item_features["genre_popularity_mean"] = item_features.apply(
        lambda r: r["genre_popularity_sum"] / r["n_genres"]
        if r["n_genres"] > 0
        else 0,
        axis=1,
    )
    item_features["genre_popularity_sum"] = np.log1p(
        item_features["genre_popularity_sum"]
    )
    item_features["genre_popularity_mean"] = np.log1p(
        item_features["genre_popularity_mean"]
    )

    item_features["is_multigenre"] = (item_features["n_genres"] > 1).astype(int)
    item_features["year_rank"] = (
        item_features["publication_year"].map(year_rank_map).fillna(1).astype(int)
    )
    item_features["author_rank"] = (
        item_features["author_id"].map(author_rank_map).fillna(1).astype(int)
    )
    item_features["book_rank"] = (
        item_features["book_id"].map(book_rank_map).fillna(1).astype(int)
    )
    item_features["language_flag"] = (
        item_features["language_id"] == DEFAULT_LANGUAGE_ID
    ).astype(int)

    return item_features


def _build_user_features(
    users_base: DataFrameLike,
    history_enriched: DataFrameLike,
    age_rank_map: dict,
) -> DataFrameLike:
    user_features = users_base.copy()
    user_features["age_rank"] = (
        user_features["age"].map(age_rank_map).fillna(1).astype(int)
    )

    user_multi_lang_flag = (
        history_enriched.groupby("user_id")["language_id"].nunique().gt(1).astype(int)
    )
    user_features["user_multi_lang_flag"] = user_features["user_id"].map(
        user_multi_lang_flag
    )

    user_rating_mean = history_enriched.groupby("user_id")["rating"].mean()
    user_rating_count = history_enriched.groupby("user_id")["rating"].count()
    user_features["user_rating_mean"] = user_features["user_id"].map(
        user_rating_mean
    )
    user_features["user_rating_count"] = user_features["user_id"].map(
        user_rating_count
    )
    return user_features


def _build_feature_frame(
    candidates: DataFrameLike,
    state: _FeatureState,
    config: EtlConfig,
) -> DataFrameLike:
    df = candidates.merge(
        state.item_features,
        on="edition_id",
        how="left",
    )
    df = df.merge(
        state.user_features,
        on="user_id",
        how="left",
    )

    df["user_multi_lang_flag"] = df["user_multi_lang_flag"].fillna(0).astype(int)
    rating_mean_default = state.user_features["user_rating_mean"].mean()
    if np.isnan(rating_mean_default):
        rating_mean_default = 0.0
    df["user_rating_mean"] = df["user_rating_mean"].fillna(rating_mean_default)
    df["user_rating_count"] = df["user_rating_count"].fillna(0).astype(int)

    df["user_author_hits"] = _map_user_author_hits(df, state.user_author_cnt)
    df["user_genre_cnt"] = _map_user_genre_cnt(df, state.user_genre_cnt)

    df["total_rank"] = (
        df[["author_rank", "book_rank", "year_rank"]].fillna(0).sum(axis=1)
    )

    user_vec_base = state.user_vec.copy()
    author_vec_base = state.author_vec.copy()
    user_vec_clusters = user_vec_base.copy()
    author_vec_clusters = author_vec_base.copy()

    user_clusters = _fit_clusters(
        user_vec_clusters,
        n_clusters=config.user_clusters,
        random_state=config.random_state,
    )
    author_clusters = _fit_clusters(
        author_vec_clusters,
        n_clusters=config.author_clusters,
        random_state=config.random_state,
    )
    user_vec_clusters["user_cluster"] = user_clusters
    author_vec_clusters["author_cluster"] = author_clusters

    df = df.merge(
        user_vec_clusters[["user_cluster"]],
        left_on="user_id",
        right_index=True,
        how="left",
    )
    df = df.merge(
        author_vec_clusters[["author_cluster"]],
        left_on="author_id",
        right_index=True,
        how="left",
    )

    df["user_author_sim"] = _user_author_sim(df, user_vec_base, author_vec_base)

    df["author_cluster"] = df["author_cluster"].fillna(-1).astype(int)
    df["user_cluster"] = df["user_cluster"].fillna(-1).astype(int)

    df = df.drop(
        columns=[
            "genre_id",
            "book_id",
            "author_id",
            "publisher_id",
            "title",
            "description",
            "age_restriction",
        ],
        errors="ignore",
    )
    df = _fill_feature_missing(df)
    return df


def _fill_feature_missing(df: DataFrameLike) -> DataFrameLike:
    df = df.copy()
    for col in df.columns:
        if col in {"user_id", "edition_id"}:
            continue
        if df[col].dtype.kind in {"i", "u"}:
            df[col] = df[col].fillna(0)
        else:
            mean_val = df[col].mean()
            if np.isnan(mean_val):
                mean_val = 0.0
            df[col] = df[col].fillna(mean_val)
    return df


def _label_candidates(
    interactions: DataFrameLike,
    window_start: TimestampLike,
    window_end: TimestampLike,
) -> DataFrameLike:
    labels = interactions[
        (interactions["event_ts"] >= window_start)
        & (interactions["event_ts"] <= window_end)
    ].copy()

    labels["y_read"] = (labels["event_type"] == READ_EVENT).astype(int)
    labels["y_wishlist"] = (labels["event_type"] == WISHLIST_EVENT).astype(int)
    labels["rel"] = labels["event_type"].map({READ_EVENT: 3, WISHLIST_EVENT: 1}).fillna(0)

    agg = labels.groupby(["user_id", "edition_id"]).agg(
        rel=("rel", "max"),
        y_read=("y_read", "max"),
        y_wishlist=("y_wishlist", "max"),
    )
    return agg.reset_index()


def _fill_missing_labels(df: DataFrameLike) -> DataFrameLike:
    df = df.copy()
    for col in ["rel", "y_read", "y_wishlist"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


def _sample_negatives(
    df: DataFrameLike,
    negative_ratio: float,
    keep_users_without_positives: bool,
    random_state: int,
) -> DataFrameLike:
    if negative_ratio is None or negative_ratio <= 0:
        return df

    positives = df[df["rel"] > 0]
    negatives = df[df["rel"] == 0]
    if positives.empty:
        return df if keep_users_without_positives else positives

    sampled_negatives = []
    neg_groups = dict(tuple(negatives.groupby("user_id")))
    for user_id, pos_group in positives.groupby("user_id"):
        neg_group = neg_groups.get(user_id)
        if neg_group is None or neg_group.empty:
            continue
        n_pos = len(pos_group)
        n_neg = min(len(neg_group), int(round(n_pos * negative_ratio)))
        if n_neg <= 0:
            continue
        sampled_negatives.append(
            neg_group.sample(n=n_neg, random_state=random_state)
        )

    if keep_users_without_positives:
        users_with_pos = set(positives["user_id"].unique())
        extra_neg = negatives[~negatives["user_id"].isin(users_with_pos)]
        sampled_negatives.append(extra_neg)

    if sampled_negatives:
        negatives = pd.concat(sampled_negatives, ignore_index=True)
    else:
        negatives = negatives.iloc[0:0]

    out = pd.concat([positives, negatives], ignore_index=True)
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


def _build_year_rank_map(history: DataFrameLike) -> dict:
    if history.empty:
        return {}
    counts = (
        history.groupby("publication_year").size().sort_values(ascending=False)
    )
    rank_map = {}
    for i, y in enumerate(counts.index):
        if y in {2025, 2026}:
            rank_map[y] = 10
        else:
            rank_map[y] = max(1, 10 - i // 2)
    return rank_map


def _build_age_rank_map(history: DataFrameLike) -> dict:
    if history.empty:
        return {}
    counts = history.groupby("age").size().sort_values(ascending=False)
    rank_map = {}
    for i, y in enumerate(counts.index):
        rank_map[y] = max(1, 10 - i // 5)
    return rank_map


def _build_rank_map(history: DataFrameLike, key: str, score_mass: list[int]) -> dict:
    if history.empty:
        return {}
    counts = history.groupby(key).size().sort_values(ascending=False)
    rank_map = {}
    for i, val in enumerate(counts.index):
        for rank_inv, threshold in enumerate(score_mass):
            if i < threshold:
                rank_map[val] = 10 - rank_inv
                break
        else:
            rank_map[val] = 1
    return rank_map


def _score_mass_double() -> list[int]:
    score_mass = []
    score = 10
    for _ in range(3, 12):
        score_mass.append(score)
        score *= 2
    return score_mass


def _score_mass_pow_175() -> list[int]:
    score_mass = []
    score = 10
    for _ in range(3, 12):
        score_mass.append(score)
        score = int(round(score * 1.75))
    return score_mass


def _main_genre(genres: list, top_genres: SeriesLike) -> int:
    if not genres:
        return -1
    return max(genres, key=lambda g: top_genres.get(g, 0))


def _ensure_list(value) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _map_user_genre_cnt(df: DataFrameLike, user_genre_cnt: SeriesLike) -> SeriesLike:
    temp = df[["user_id", "genre_id"]].copy()
    temp["row_id"] = np.arange(len(temp))
    temp = temp.explode("genre_id")

    idx = pd.MultiIndex.from_arrays([temp["user_id"], temp["genre_id"]])
    temp["ug_cnt"] = user_genre_cnt.reindex(idx, fill_value=0).values
    return (
        temp.groupby("row_id")["ug_cnt"]
        .sum()
        .reindex(range(len(df)), fill_value=0)
        .values
    )


def _map_user_author_hits(df: DataFrameLike, user_author_cnt: SeriesLike) -> SeriesLike:
    idx = pd.MultiIndex.from_arrays([df["user_id"], df["author_id"]])
    return user_author_cnt.reindex(idx, fill_value=0).values


def _normalize_df(df: DataFrameLike) -> DataFrameLike:
    if df.empty:
        return df
    values = df.values
    values = np.nan_to_num(values, nan=0.0)
    values = normalize(values)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def _fit_clusters(
    vec: DataFrameLike,
    n_clusters: int,
    random_state: int,
) -> SeriesLike:
    if vec.empty:
        return pd.Series([], dtype=int, index=vec.index)
    n_clusters = min(n_clusters, len(vec))
    if n_clusters <= 1:
        return pd.Series(np.zeros(len(vec), dtype=int), index=vec.index)
    return pd.Series(
        KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(vec),
        index=vec.index,
    )


def _user_author_sim(
    df: DataFrameLike,
    user_vec: DataFrameLike,
    author_vec: DataFrameLike,
) -> SeriesLike:
    common_feats = list(set(user_vec.columns) & set(author_vec.columns))
    if not common_feats:
        return pd.Series(np.zeros(len(df)), index=df.index)

    u = user_vec[common_feats]
    a = author_vec[common_feats]

    u = u.rename(columns={c: f"u_{c}" for c in common_feats})
    a = a.rename(columns={c: f"a_{c}" for c in common_feats})

    temp = df[["user_id", "author_id"]].merge(
        u, left_on="user_id", right_index=True, how="left"
    )
    temp = temp.merge(
        a, left_on="author_id", right_index=True, how="left"
    )

    u_cols = [f"u_{c}" for c in common_feats]
    a_cols = [f"a_{c}" for c in common_feats]
    temp[u_cols] = temp[u_cols].fillna(0.0)
    temp[a_cols] = temp[a_cols].fillna(0.0)
    return (temp[u_cols].values * temp[a_cols].values).sum(axis=1)


def _make_windows(
    start_ts: TimestampLike,
    end_ts: TimestampLike,
    window_days: int,
    step_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    cursor = start_ts
    while cursor <= end_ts:
        window_start = cursor
        window_end = cursor + pd.Timedelta(days=window_days)
        if window_end > end_ts:
            window_end = end_ts
        windows.append((window_start, window_end))
        cursor = cursor + pd.Timedelta(days=step_days)
    return windows
