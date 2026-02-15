import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def prosessing_NaN(df: DataFrame) -> DataFrame:
    df["rating"] = df["rating"].fillna(round(df["rating"].mean())).astype("int64")
    df["description"] = df["description"].fillna("")
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0]).astype("int64")
    df["age"] = df["age"].fillna(round(df["age"].mean())).astype("int64")

    return df

def aggr_by_publ_year(df: DataFrame, kol_in_group: int = 2) -> DataFrame:
    read_in_publ = df.groupby("publication_year").size().reset_index(name="counts").sort_values(by="counts", ascending=False)
    read_in_publ = read_in_publ.reset_index(drop=True)

    rank_map = {}
    for i, y in enumerate(read_in_publ["publication_year"]):
        if y == 2025 or y == 2026:
            rank_map[y] = 10
        else:
            rank_map[y] = max(1, 10 - i // kol_in_group)

    df["year_rank"] = df["publication_year"].map(rank_map)

    return df

def aggr_by_number_years_users(df: DataFrame, kol_in_group: int = 5) -> DataFrame:
    age_rank_ = df.groupby("age").size().reset_index(name="counts").sort_values(by="counts", ascending=False).reset_index(drop=True)

    age_rank_map = {}
    for i, y in enumerate(age_rank_["age"]):
        age_rank_map[y] = max(1, 10 - i // kol_in_group)

    df["age_rank"] = df["age"].map(age_rank_map)

    return df

def aggr_by_authors(df: DataFrame, start_val: int = 10) -> DataFrame:
    score_mass = []
    score = start_val
    for i in range(3, 12):
        score_mass.append(score)
        score *= 2

    number_month = len(df["event_ts"].dt.month.unique().tolist())
    div = np.sqrt(6 / number_month)
    score_mass = [score / div for score in score_mass]


    authors_rank_ = df.groupby("author_id").size().reset_index(name="counts").sort_values(by="counts", ascending=False).reset_index(drop=True)

    authors_rank_map = {}
    for i, y in enumerate(authors_rank_["author_id"]):
        for rank_inv, val in enumerate(score_mass):
            if i < val:
                authors_rank_map[y] = start_val - rank_inv
                break
        else:
            authors_rank_map[y] =  1

    df["author_rank"] = df["author_id"].map(authors_rank_map)

    return df

def aggr_by_lang(df: DataFrame) -> DataFrame:
    df["language_flag"] = df["language_id"].apply(lambda x: 1 if x == 119 else 0)

    user_multi_lang_flag = df.groupby("user_id")["language_id"].apply(set).apply(len).apply(lambda x: 1 if x > 1 else 0)
    df["user_multi_lang_flag"] = df["user_id"].map(user_multi_lang_flag)

    return df

def aggr_by_books(df: DataFrame, start_val: int = 10) -> DataFrame:
    score_mass = []
    score = start_val
    for i in range(3, 12):
        score_mass.append(score)
        score = round(score * 1.75)

    number_month = len(df["event_ts"].dt.month.unique().tolist())
    div = np.sqrt(6 / number_month)
    score_mass = [score / div for score in score_mass]

    book_rank_ = df.groupby("book_id").size().reset_index(name="counts").sort_values(by="counts", ascending=False).reset_index(drop=True)

    book_rank_map = {}
    for i, y in enumerate(book_rank_["book_id"]):
        for rank_inv, val in enumerate(score_mass):
            if i < val:
                book_rank_map[y] = start_val - rank_inv
                break
        else:
            book_rank_map[y] =  1

    df["book_rank"] = df["book_id"].map(book_rank_map)

    return df


def aggr_by_genre(df: DataFrame) -> DataFrame:
    df["n_genres"] = df["genre_id"].apply(len)

    top_genres = df["genre_id"].explode().value_counts()

    df["main_genre"] = df["genre_id"].apply(
        lambda xs: max(xs, key=lambda g: top_genres.get(g, 0))
    )

    df["genre_popularity_sum"] = df["genre_id"].apply(
        lambda xs: sum(top_genres.get(g, 0) for g in xs)
    )
    df["genre_popularity_mean"] = (
        df["genre_popularity_sum"] / df["n_genres"]
    )

    df["is_multigenre"] = (df["n_genres"] > 1).astype(int)

    return df

def aggr_user_genre(df: DataFrame) -> DataFrame:
    user_genre_cnt = df.explode("genre_id").groupby(["user_id", "genre_id"]).size()
    df["user_genre_cnt"] = df.apply(lambda x: sum(user_genre_cnt.get((x.user_id, g) , 0 ) for g in x.genre_id), axis=1)

    return df

def aggr_user_author_cnt(df: DataFrame) -> DataFrame:
    user_author_cnt = df.groupby(["user_id", "author_id"]).size()
    df["user_author_hits"] = df.apply(
        lambda r: user_author_cnt.get((r.user_id, r.author_id), 0),
        axis=1
    )

    return df

def log1p_feature(df: DataFrame) -> DataFrame:
    df["genre_popularity_sum"] = np.log1p(df["genre_popularity_sum"])
    df["genre_popularity_mean"] = np.log1p(df["genre_popularity_mean"])

    return df

def total_rank(df: DataFrame) -> DataFrame:
    df["total_rank"] = df[["author_rank", "rating", "book_rank", "year_rank"]].agg(sum, axis=1)
    return df

def add_clusters_and_dis(df: DataFrame, kol_user_clusters: int = 75, kol_author_clusters: int = 45) -> DataFrame:
    user_vec = df.groupby("user_id").agg(
        genre_pop_mean= ("genre_popularity_mean", "mean"),
        language_multi_flag= ("user_multi_lang_flag", "mean"),
        rating_mean= ("rating", "mean"),
        author_rank_mean= ("author_rank", "mean"),
        book_rank_mean= ("book_rank", "mean"),
        year_rank_mean= ("year_rank", "mean"),
        total_rank_mean= ("total_rank", "mean"),
        n_genres_mean=("n_genres", "mean")
    )
    user_ind = user_vec.index
    user_vec = normalize(user_vec.values)

    user_vec = pd.DataFrame(user_vec, index=user_ind)


    author_vec = (
        df
        .groupby("author_id")
        .agg(
            genre_pop_mean=("genre_popularity_mean", "mean"),
            year_rank_mean=("year_rank", "mean"),
            book_rank_mean=("book_rank", "mean"),
            n_genres_mean=("n_genres", "mean"),
            rating_mean=("rating", "mean"),
        )
    )
    author_ind = author_vec.index
    author_vec= normalize(author_vec.values)

    author_vec = pd.DataFrame(author_vec, index=author_ind)


    user_clusters = KMeans(n_clusters=kol_user_clusters, random_state=42).fit_predict(user_vec)
    author_clusters = KMeans(n_clusters=kol_author_clusters, random_state=42).fit_predict(author_vec)

    user_vec["user_cluster"] = user_clusters
    author_vec["author_cluster"] = author_clusters

    df = df.merge(user_vec["user_cluster"], on="user_id", how="left")
    df = df.merge(author_vec["author_cluster"], on="author_id", how="left")

    COMMON_FEATS = list(
        set(user_vec.columns)
        & set(author_vec.columns)
    )
    COMMON_FEATS

    u = user_vec.reset_index()[["user_id", *COMMON_FEATS]].rename(
        columns={c: f"{c}_u" for c in COMMON_FEATS}
    )
    a = author_vec.reset_index()[["author_id", *COMMON_FEATS]].rename(
        columns={c: f"{c}_a" for c in COMMON_FEATS}
    )

    df_ = df.merge(u, on="user_id", how="left")
    df_ = df_.merge(a, on="author_id", how="left")

    u_mat = df_[[f"{c}_u" for c in COMMON_FEATS]].to_numpy()
    a_mat = df_[[f"{c}_a" for c in COMMON_FEATS]].to_numpy()

    num = (u_mat * a_mat).sum(axis=1)
    den = (np.linalg.norm(u_mat, axis=1) * np.linalg.norm(a_mat, axis=1))
    df["user_author_sim"] = num / den

    return df

