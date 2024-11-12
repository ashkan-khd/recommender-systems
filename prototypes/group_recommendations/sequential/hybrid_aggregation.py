import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load MovieLens 100K dataset
data = pd.read_csv('ratings.csv')  # Replace with actual path if needed
data.columns = ['userId', 'movieId', 'rating', 'timestamp']

# Group data by user for easy access to each user's ratings
user_ratings = data.groupby('userId')[['movieId', 'rating']].apply(
    lambda x: dict(zip(x['movieId'], x['rating']))).to_dict()
movies = data['movieId'].unique()


def preference_score(user_id, movie_id, user_ratings):
    """
    Calculate preference score for a user on a specific movie.
    If the user has rated the movie, return the rating; otherwise, return a predicted score.
    """
    if movie_id in user_ratings[user_id]:
        return user_ratings[user_id][movie_id]
    # Predicting based on the user's average rating if unseen
    return np.mean(list(user_ratings[user_id].values()))


def calculate_user_satisfaction(user_id, group_rec, user_ratings, top_k=10):
    """
    Calculate user satisfaction based on the group recommendations.
    """
    top_user_prefs = sorted(user_ratings[user_id].items(), key=lambda x: x[1], reverse=True)[:top_k]
    user_ideal_satisfaction = sum([score for _, score in top_user_prefs])
    group_satisfaction = sum([preference_score(user_id, movie, user_ratings) for movie in group_rec])
    return group_satisfaction / user_ideal_satisfaction if user_ideal_satisfaction > 0 else 0


def calculate_group_satisfaction(group, group_rec, user_ratings):
    """
    Calculate average satisfaction across all users in the group.
    """
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings) for user in group]
    return np.mean(individual_satisfactions)


def calculate_group_disagreement(group, group_rec, user_ratings):
    """
    Calculate disagreement within the group as the difference between max and min satisfaction.
    """
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings) for user in group]
    return max(individual_satisfactions) - min(individual_satisfactions)


def avgScore(group, movie, user_ratings):
    """Calculate the average preference score for the movie among the group members."""
    return np.mean([preference_score(user, movie, user_ratings) for user in group])

def leastScore(group, movie, user_ratings):
    """Calculate the minimum preference score for the movie among the group members."""
    return min([preference_score(user, movie, user_ratings) for user in group])


def generate_sequential_recommendations(group, user_ratings, movies, iterations=5, top_k=10, alpha=0.0):
    group_recommendations = []
    user_satisfaction_history = defaultdict(list)
    movie_scores = defaultdict(float)  # Persist movie scores across iterations for dynamic weighting

    for iteration in range(iterations):
        # Calculate movie scores based on the new weighted combination of avgScore and leastScore
        for movie in movies:
            avg_score = avgScore(group, movie, user_ratings)
            least_score = leastScore(group, movie, user_ratings)
            movie_scores[movie] = alpha * least_score + (1 - alpha) * avg_score

        # Sort movies by the adjusted group preference scores and select the top recommendations
        top_movies = sorted(movie_scores, key=movie_scores.get, reverse=True)[:top_k]
        group_recommendations.append(top_movies)

        # Calculate and log satisfaction and disagreement metrics
        group_sat = calculate_group_satisfaction(group, top_movies, user_ratings)
        group_dis = calculate_group_disagreement(group, top_movies, user_ratings)
        alpha = group_dis

        print(f"Iteration {iteration + 1}")
        print(f"Top Movies: {top_movies}")
        print(f"Group Satisfaction: {group_sat:.3f}")
        print(f"Group Disagreement: {group_dis:.3f}\n")

        # Record user satisfaction scores for dynamic adjustment in the next iteration
        for user in group:
            user_satisfaction = calculate_user_satisfaction(user, top_movies, user_ratings)
            user_satisfaction_history[user].append(user_satisfaction)

    return group_recommendations



# Define a sample group of users (you may choose any subset of user IDs from the dataset)
# sample_group = [414, 509, 160]  # Example user IDs
# sample_group = [274, 391, 483, 561]  # Example user IDs
# sample_group = [597, 217, 66, 177]  # Example user IDs
sample_group = [597, 217, 66, 177, 274, 391, 483, 561, 414, 509, 160]  # Example user IDs

# Run the sequential group recommendation process
recommendations = generate_sequential_recommendations(sample_group, user_ratings, movies)
