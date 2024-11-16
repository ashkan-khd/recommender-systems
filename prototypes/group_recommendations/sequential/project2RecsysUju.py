import random
import typing
from time import sleep
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from typing import *
from pandas.core.frame import DataFrame, Series

# Constants and Dataset
RATING_DATASET = r'C:\Users\user\Desktop\ratings.csv'
# Load Data
raw_df = pd.read_csv(RATING_DATASET)

# Transform CSV DataFrame to User-Movies Matrix
def transform_csv_dataframe_to_user_movies_matrix(csv_df: DataFrame) -> DataFrame:
    user_movie_matrix = csv_df.pivot(index='userId', columns='movieId', values='rating')
    user_movie_matrix.reset_index(inplace=True)
    user_movie_matrix.columns.name = None
    return user_movie_matrix

user_movie_matrix = transform_csv_dataframe_to_user_movies_matrix(raw_df)

def predict_user_rating(user_movie_matrix: DataFrame, user_id: int, movie_id: int) -> float:
    """
    Predict the rating for a user based on collaborative filtering (user-based).
    Uses similarity of the user to other users who have rated the same movie.
    """
    # Get all users who have rated the movie
    movie_ratings = user_movie_matrix[movie_id]
    rated_users = movie_ratings[~movie_ratings.isna()].index.tolist()

    # If the user has already rated the movie, return the rating
    if user_id in rated_users:
        return movie_ratings[user_id]
    
    # Calculate similarity between the target user and others
    similarities = []
    target_ratings = user_movie_matrix.loc[user_id]
    
    for other_user in rated_users:
        other_ratings = user_movie_matrix.loc[other_user]
        
        # Calculate similarity (cosine similarity)
        common_movies = target_ratings.dropna().index.intersection(other_ratings.dropna().index)
        if len(common_movies) == 0:
            continue
        
        target_common_ratings = target_ratings[common_movies]
        other_common_ratings = other_ratings[common_movies]
        
        similarity = np.dot(target_common_ratings - np.mean(target_common_ratings), 
                            other_common_ratings - np.mean(other_common_ratings)) / (
                            np.linalg.norm(target_common_ratings - np.mean(target_common_ratings)) *
                            np.linalg.norm(other_common_ratings - np.mean(other_common_ratings)))
        
        similarities.append((similarity, other_user))
    
    # Predict the rating as a weighted average of other ratings based on similarity
    if similarities:
        weighted_ratings = sum(sim * user_movie_matrix.loc[other_user, movie_id] for sim, other_user in similarities)
        total_similarity = sum(abs(sim) for sim, _ in similarities)
        return weighted_ratings / total_similarity if total_similarity != 0 else 0
    else:
        return 0  # Default to 0 if no similar users are found

# Satisfaction and Disagreement Functions
def calculate_user_satisfaction(user_id, group_rec, user_ratings, user_satisfactions):
    """
    Calculate user satisfaction based on the group recommendations.
    """
    user_ideal_satisfaction = user_satisfactions[user_id]
    group_satisfaction = sum([predict_user_rating(user_movie_matrix, user_id, movie) for movie in group_rec])
    return group_satisfaction / user_ideal_satisfaction if user_ideal_satisfaction > 0 else 0

def calculate_group_satisfaction(group, group_rec, user_ratings, user_satisfactions):
    """
    Calculate average satisfaction across all users in the group.
    """
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings, user_satisfactions) for user in group]
    return np.mean(individual_satisfactions)

def calculate_group_disagreement(group, group_rec, user_ratings, user_satisfactions):
    """
    Calculate disagreement within the group as the difference between max and min satisfaction.
    """
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings, user_satisfactions) for user in group]
    return max(individual_satisfactions) - min(individual_satisfactions)

# Score functions for hybrid aggregation
def avgScore(group, movie, user_ratings):
    """Calculate the average preference score for the movie among the group members."""
    return np.mean([predict_user_rating(user_movie_matrix, user, movie) for user in group])

def leastScore(group, movie, user_ratings):
    """Calculate the minimum preference score for the movie among the group members."""
    return min([predict_user_rating(user_movie_matrix, user, movie) for user in group])
# Content-based filtering score (use a placeholder function here for illustration)
def content_based_score(movie, movie_features, group_preferences):
    """Calculate content-based score based on movie features and group preferences"""
    feature_vector = movie_features.get(movie, np.zeros(len(group_preferences[0])))  # Assume 0 if no features
    scores = [np.dot(group_pref, feature_vector) for group_pref in group_preferences]
    return np.mean(scores)

# Generate sequential recommendations with dynamic alpha adjustment
def generate_sequential_recommendations(
    group, 
    user_ratings, 
    movies, 
    movie_features,  # Added: movie features for content-based filtering
    group_preferences,  # Added: group preferences for CBF
    iterations=5, 
    top_k=10, 
    alpha=0.5,  # Initial value for alpha
):
    group_recommendations = []
    movie_scores = defaultdict(float)  # Persist movie scores across iterations for dynamic weighting
    user_satisfactions = defaultdict(float)  # Persist user satisfaction scores across iterations for dynamic weighting
    
    # Initial setup of user satisfaction based on top K preferences
    for user in group:
        top_user_prefs = sorted(user_ratings[user].items(), key=lambda x: x[1], reverse=True)[:top_k]
        user_satisfactions[user] = sum([score for _, score in top_user_prefs])

    for iteration in range(iterations):
        for movie in tqdm(movies, desc=f"Calculating movie scores for iteration {iteration}"):
            # Collaborative Filtering (CF) Score
            cf_score = avgScore(group, movie, user_ratings)
            
            # Content-Based Filtering (CBF) Score
            cbf_score = content_based_score(movie, movie_features, group_preferences)
            
            # Hybrid score: Combine CF and CBF scores with dynamic alpha
            movie_scores[movie] = alpha * cf_score + (1 - alpha) * cbf_score

        # Get top K movies based on hybrid scores
        top_movies = sorted(movie_scores, key=movie_scores.get, reverse=True)[:top_k]
        group_recommendations.append(top_movies)

        # Calculate group satisfaction and disagreement
        group_sat = calculate_group_satisfaction(group, top_movies, user_ratings, user_satisfactions)
        group_dis = calculate_group_disagreement(group, top_movies, user_ratings, user_satisfactions)

        # Dynamically adjust alpha based on group satisfaction and disagreement
        if group_sat + group_dis > 0:
            alpha = group_sat / (group_sat + group_dis)  # Increase alpha if satisfaction is high
        else:
            alpha = 0.5  # Default alpha when there's no significant trend

        # Print iteration results
        print(f"Iteration {iteration + 1}")
        print(f"Top Movies: {top_movies}")
        print(f"Group Satisfaction: {group_sat:.3f}")
        print(f"Group Disagreement: {group_dis:.3f}")
        print(f"Updated Alpha: {alpha:.3f}\n")

        sleep(0.001)  # Optional: Simulate real-time processing

    return group_recommendations

# Sample usage
sample_group = [597, 217, 66, 177, 274, 391, 483, 561, 414, 509, 160]

# User ratings
user_ratings = raw_df[raw_df["userId"].isin(sample_group)].groupby('userId')[['movieId', 'rating']].apply(
    lambda x: dict(zip(x['movieId'], x['rating']))).to_dict()

# Movies already rated by the group
group_users_movies = user_movie_matrix[user_movie_matrix['userId'].isin(sample_group)].dropna(axis=1, how='all').columns.tolist()[1:]

# Randomly sample some movies that the group hasn't rated
movies = random.sample([m for m in raw_df['movieId'].unique() if m not in group_users_movies], 100)

# Movie features (for content-based filtering)
movie_features = {movie: [random.random() for _ in range(10)] for movie in raw_df['movieId'].unique()}

# Group preferences (this can be computed based on individual preferences or a predefined strategy)
group_preferences = [np.random.random(10) for _ in range(len(sample_group))]  # Placeholder preferences for each user

# Run the sequential group recommendation process
recommendations = generate_sequential_recommendations(sample_group, user_ratings, movies, movie_features, group_preferences)