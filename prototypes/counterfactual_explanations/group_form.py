from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from part2_utils.predict_user_rating import transform_csv_dataframe_to_user_movies_matrix


def get_k_users_with_max_common_movies_sampled(user_movie_matrix: pd.DataFrame, k: int, sample_size: int = 100):
    # Step 1: Randomly sample 100 users
    sampled_users = user_movie_matrix.sample(n=sample_size,
                                             random_state=1).index  # Set random_state for reproducibility
    sampled_matrix = user_movie_matrix.loc[sampled_users]

    # Step 2: Create a binary matrix where 1 indicates the movie was rated (watched)
    watched_matrix = sampled_matrix.notna().astype(int)

    # Step 3: Calculate common watched movies for each pair of sampled users
    common_counts = {}

    for user1, user2 in tqdm(list(combinations(sampled_users, 2)),
                             desc="Testing random combinations to match a good group"):
        # Count common watched movies between user1 and user2
        common_count = np.dot(watched_matrix.loc[user1], watched_matrix.loc[user2])
        common_counts[(user1, user2)] = common_count

    # Step 4: Find the top `k` users with the maximum common watched movies
    user_common_totals = {}
    for (user1, user2), count in common_counts.items():
        user_common_totals[user1] = user_common_totals.get(user1, 0) + count
        user_common_totals[user2] = user_common_totals.get(user2, 0) + count

    # Sort users by the total count of common watched movies and select top `k`
    top_k_users = sorted(user_common_totals, key=user_common_totals.get, reverse=True)[:k]

    # Return the rows of the user-movie matrix corresponding to the top `k` users
    return user_movie_matrix.loc[top_k_users].reset_index(drop=True)


user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(pd.read_csv('data/ml-latest-small/ratings.csv'))

group = get_k_users_with_max_common_movies_sampled(user_movie_matrix, k=10, sample_size=200)
print(group)
