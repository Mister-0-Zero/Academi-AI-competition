import numpy as np

import pandas as pd
from pandas import DataFrame
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from agregation import prosessing_NaN, aggr_by_publ_year, aggr_by_number_years_users,\
                       aggr_by_authors, aggr_by_lang, aggr_by_books, aggr_by_genre,\
                       aggr_user_genre, aggr_user_author_cnt, log1p_feature,\
                       total_rank, add_clusters_and_dis


date_split = [pd.Timestamp("2024-11-14"), pd.Timestamp("2024-12-14"), pd.Timestamp("2025-01-14"),\
              pd.Timestamp("2025-02-14"), pd.Timestamp("2025-03-14")]

def Extract(path_dir_data=r"./data/data", path_dir_submissions=r"./data/submit", num_months=None) -> DataFrame:
    #Считывание данных
    book_genres = pd.read_csv(path_dir_data + "/book_genres.csv")
    editions = pd.read_csv(path_dir_data + "/editions.csv")
    interactions = pd.read_csv(path_dir_data + "/interactions.csv")
    users = pd.read_csv(path_dir_data + "/users.csv")

    #Ограничение по времени
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])
    if num_months:
        interactions = interactions[interactions["event_ts"] < date_split[num_months - 1]]

    genres_mas = (book_genres.groupby("book_id")["genre_id"].apply(list).reset_index())
    genres_mas

    df = interactions.merge(editions, on="edition_id", how="left")
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(genres_mas, on="book_id", how="left")

    return df

def Transform(df: DataFrame) -> DataFrame:
    #Обработка NaN
    df = prosessing_NaN(df)

    #Удаляем лишние столбцы
    df.drop(columns=["description", "title", "age_restriction"], inplace=True)

    #Агрегация по годам публикации
    df = aggr_by_publ_year(df, kol_in_group=2)

    #Агрегация по количеству лет пользователей(угруппы пользователей по 5 лет)
    df = aggr_by_number_years_users(df, kol_in_group=5)

    #Агрегация по авторам
    df = aggr_by_authors(df, start_val=10)

    #Агрегации по языкам(основной ли в книге язык и на рзных ли языках книги пользователь читал)
    df = aggr_by_lang(df)

    #Агрегации по книгам(даю ранги книгам в зависимости от частоты их использования)
    df = aggr_by_books(df, start_val=10)

    #Агрегация польователь\жанры(сколько различных жанров прочитал пользователь)
    df = aggr_user_genre(df)

    #Агрегация сколько раз пользователь взаимодействовал с данным автором
    df = aggr_user_author_cnt(df)

    #Логарифмирования больших величин
    df = log1p_feature(df)

    #Присваивание общего ранга
    df = total_rank(df)

    #Добавление векторного представление пользователя и автора + их близость
    df = add_clusters_and_dis(df, kol_user_clusters=75, kol_author_clusters=45)

    #Последние удаление лишних фич
    df.drop(columns=["publisher_id", "author_id", "book_id"], inplace=True)

    return df










