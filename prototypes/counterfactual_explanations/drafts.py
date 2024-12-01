import pandas as pd

from part2_utils.predict_user_rating import transform_csv_dataframe_to_user_movies_matrix

user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(pd.read_csv('data/ml-latest-small/ratings.csv'))

# user_movie_matrix_without_user_column = user_movie_matrix.drop(columns=['userId'])
#
# movie_rating_counts = user_movie_matrix_without_user_column.notna().sum()
#
# print(movie_rating_counts)
#
# p90 = movie_rating_counts.quantile(0.50)
# print(f"P90 of movie rating counts: {p90}")
#
# print(len(movie_rating_counts[movie_rating_counts < p90])/len(movie_rating_counts))

users = [269, 120, 278, 163, 161, 383, 238, 392, 188, 35]

preferences = [1, 3, 5, 6, 7, 9, 11, 12, 21, 25, 32, 39, 48, 50, 52, 60, 62, 63, 74, 76, 79, 81, 95]


def get_users_rated_movie(
        users,
        movies,
        user_movie_matrix,
):
    user_movie_matrix_limited = user_movie_matrix[user_movie_matrix['userId'].isin(users)]
    user_movie_matrix_limited = user_movie_matrix_limited[["userId"] + movies]

    def users_rated_movie(movie):
        return list(user_movie_matrix_limited[user_movie_matrix_limited[movie].notna()]["userId"])

    return users_rated_movie


def movies_rated_for_user(
        users,
        movies,
        user_movie_matrix,
):
    user_movie_matrix_limited = user_movie_matrix[user_movie_matrix['userId'].isin(users)]
    user_movie_matrix_limited = user_movie_matrix_limited[["userId"] + movies]

    def movies_rated_for_user(user):
        return list(
            user_movie_matrix_limited[user_movie_matrix_limited["userId"] == user].drop(
                columns=["userId"]
            ).dropna(axis=1, how='any').columns
        )

    return movies_rated_for_user

users_rated_movie = get_users_rated_movie(users, preferences, user_movie_matrix)
# print(users_rated_movie(5))

movies_rated_for_user = movies_rated_for_user(users, preferences, user_movie_matrix)
# print(movies_rated_for_user(269))



def get_movies_not_rated_by_group(users, user_movie_matrix):
    all_movie_ids = set(user_movie_matrix.drop('userId', axis=1).columns)
    rated_movie_ids = set(
        user_movie_matrix[user_movie_matrix["userId"].isin(users)].dropna(axis=1, how='all').drop('userId', axis=1).columns
    )
    return list(all_movie_ids - rated_movie_ids)


get_movies_not_rated_by_group(users, user_movie_matrix)

