import pandas as pd
from pandas import DataFrame
from os import path

from agregation import prosessing_NaN, aggr_by_publ_year, aggr_by_number_years_users,\
                       aggr_by_authors, aggr_by_lang, aggr_by_books, aggr_by_genre,\
                       aggr_user_genre, aggr_user_author_cnt, log1p_feature,\
                       total_rank, add_clusters_and_dis


date_split = [pd.Timestamp("2024-11-14"), pd.Timestamp("2024-12-14"), pd.Timestamp("2025-01-14"),\
              pd.Timestamp("2025-02-14"), pd.Timestamp("2025-03-14"), None]

def Extract(path_dir_data: path, num_months: int|None) -> DataFrame:
    #Считывание данных
    book_genres = pd.read_csv(path_dir_data + "/book_genres.csv")
    editions = pd.read_csv(path_dir_data + "/editions.csv")
    interactions = pd.read_csv(path_dir_data + "/interactions.csv")
    users = pd.read_csv(path_dir_data + "/users.csv")

    #Ограничение по времени
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])
    if num_months is not None and date_split[num_months - 1] is not None:
        interactions = interactions[interactions["event_ts"] < date_split[num_months - 1]]

    genres_mas = (book_genres.groupby("book_id")["genre_id"].apply(list).reset_index())
    genres_mas

    df = interactions.merge(editions, on="edition_id", how="left")
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(genres_mas, on="book_id", how="left")

    return df

def Transform(df: DataFrame, kol_in_group_publ_year: int, kol_in_group_users_year: int,
              start_val_authors: int, start_val_books: int,
              kol_user_clusters: int, kol_author_clusters: int) -> DataFrame:
    #Обработка NaN
    df = prosessing_NaN(df)

    #Удаляем лишние столбцы
    df.drop(columns=["description", "title", "age_restriction", "event_type", "rating"], inplace=True)

    #Агрегация по годам публикации
    df = aggr_by_publ_year(df, kol_in_group_publ_year)

    #Агрегация по количеству лет пользователей(угруппы пользователей по 5 лет)
    df = aggr_by_number_years_users(df, kol_in_group_users_year)

    #Агрегация по авторам
    df = aggr_by_authors(df, start_val_authors)

    #Агрегации по языкам(основной ли в книге язык и на рзных ли языках книги пользователь читал)
    df = aggr_by_lang(df)

    #Агрегации по книгам(даю ранги книгам в зависимости от частоты их использования)
    df = aggr_by_books(df, start_val_books)

    #Агрегации по жанрам
    df = aggr_by_genre(df)

    #Агрегация польователь\жанры(сколько различных жанров прочитал пользователь)
    df = aggr_user_genre(df)

    #Агрегация сколько раз пользователь взаимодействовал с данным автором
    df = aggr_user_author_cnt(df)

    #Логарифмирования больших величин
    df = log1p_feature(df)

    #Присваивание общего ранга
    df = total_rank(df)

    #Добавление векторного представление пользователя и автора + их близость
    df = add_clusters_and_dis(df, kol_user_clusters, kol_author_clusters)

    #Последние удаление лишних фич
    df.drop(columns=["publisher_id", "author_id", "book_id", "event_ts"], inplace=True)

    return df

def Load(df: DataFrame, path_save: path, expansion: str) -> None:
    if expansion == "csv":
        df.to_csv(path_save)
    elif expansion == "xlsx":
        df.to_excel(path_save)
    else:
        print(f"Ваш формат сохранения: {expansion}")
        raise "Формат сохранения не поддерживается, поддерживается только: csv, xlsx"

def ETL_function(path_dir_data: path =r"./data/data", num_months: int|None =None,
        kol_in_group_publ_year: int =2, kol_in_group_users_year: int =5,
        start_val_authors: int =10, start_val_books: int=10,
        kol_user_clusters: int =75, kol_author_clusters: int=45,
        path_save: path =r"data/after_transform_csv/dataset.csv",
        cyclecally_by_month: bool =False):

    if cyclecally_by_month:
        for ind, _ in enumerate(date_split):
            print(f"Формируем датафрейм длиной логов в {ind + 1} месяц")
            print("Запуск Extract")
            df = Extract(path_dir_data, num_months=ind + 1)
            print("Датафрейм сформировался", "\n")

            print("Запуск Transform")
            df = Transform(df, kol_in_group_publ_year, kol_in_group_users_year,
                           start_val_authors, start_val_books,
                           kol_user_clusters, kol_author_clusters)
            print("Трансформации выполнены", "\n")

            print("Запуск Load")
            path_save_ = path_save.split(".")[0] + f"{ind + 1}" + "." + path_save.split(".")[1]
            expansion = path_save.split(".")[1]
            Load(df, path_save_, expansion)
            print(f"Датафрейм сохранен по пути: {path_save_}", "\n\n")
    else:
        print("Запуск Extract")
        df = Extract(path_dir_data, num_months)
        print("Датафрейм сформировался", "\n")

        print("Запуск Transform")
        df = Transform(df, kol_in_group_publ_year, kol_in_group_users_year,
                       start_val_authors, start_val_books,
                       kol_user_clusters, kol_author_clusters)
        print("Трансформации выполнены", "\n")

        print("Запуск Load")
        expansion = path_save.split(".")[1]
        Load(df, path_save, expansion)
        print(f"Датафрейм сохранен по пути: {path_save}")