import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from tqdm import tqdm

# Load MovieLens 100K dataset
data = pd.read_csv('ratings.csv')  # Replace with actual path if needed
data.columns = ['userId', 'movieId', 'rating', 'timestamp']

# Group data by user for easy access to each user's ratings
user_ratings = data.groupby('userId')[['movieId', 'rating']].apply(
    lambda x: dict(zip(x['movieId'], x['rating']))).to_dict()
movies = data['movieId'].unique()


def preference_score(user_id, movie_id, user_ratings):
    if movie_id in user_ratings[user_id]:
        return user_ratings[user_id][movie_id]
    return np.mean(list(user_ratings[user_id].values()))


def calculate_user_satisfaction(user_id, group_rec, user_ratings, top_k=10):
    top_user_prefs = sorted(user_ratings[user_id].items(), key=lambda x: x[1], reverse=True)[:top_k]
    user_ideal_satisfaction = sum([score for _, score in top_user_prefs])
    group_satisfaction = sum([preference_score(user_id, movie, user_ratings) for movie in group_rec])
    return group_satisfaction / user_ideal_satisfaction if user_ideal_satisfaction > 0 else 0


def calculate_group_satisfaction(group, group_rec, user_ratings):
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings) for user in group]
    return np.mean(individual_satisfactions)


def calculate_group_disagreement(group, group_rec, user_ratings):
    individual_satisfactions = [calculate_user_satisfaction(user, group_rec, user_ratings) for user in group]
    return max(individual_satisfactions) - min(individual_satisfactions)


def avgScore(group, movie, user_ratings):
    return np.mean([preference_score(user, movie, user_ratings) for user in group])


def leastScore(group, movie, user_ratings):
    return min([preference_score(user, movie, user_ratings) for user in group])


class GroupRecommendationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GroupRecommendationRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out


def generate_sequential_recommendations_rnn(group, user_ratings, movies, iterations=5, top_k=10, alpha=0.3):
    group_recommendations = []
    input_size, hidden_size, output_size = len(movies) + 2, 8, len(movies)

    # Initialize RNN model
    model = GroupRecommendationRNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for iteration in range(iterations):
        # Calculate initial movie scores using avgScore and leastScore
        movie_scores = []
        for movie in movies:
            avg_score = avgScore(group, movie, user_ratings)
            least_score = leastScore(group, movie, user_ratings)
            combined_score = alpha * least_score + (1 - alpha) * avg_score
            movie_scores.append(combined_score)

        # Convert movie_scores to a PyTorch tensor for inclusion in the RNN input
        movie_scores_tensor = torch.tensor(movie_scores, dtype=torch.float32).unsqueeze(0)

        # Calculate group satisfaction and disagreement based on the previous recommendations
        if iteration == 0:
            # Initial values for the first iteration (neutral start)
            group_sat, group_dis = 0.5, 0.5
        else:
            # Calculate based on the previous recommendation
            group_sat = calculate_group_satisfaction(group, top_movies, user_ratings)
            group_dis = calculate_group_disagreement(group, top_movies, user_ratings)

        # Prepare RNN input by combining movie_scores, group_sat, and group_dis
        group_sat_tensor = torch.tensor([group_sat], dtype=torch.float32)
        group_dis_tensor = torch.tensor([group_dis], dtype=torch.float32)
        rnn_input = torch.cat([movie_scores_tensor, group_sat_tensor, group_dis_tensor], dim=-1).unsqueeze(0)

        # Forward pass through RNN to get predicted movie scores
        rnn_output = model(rnn_input)
        rnn_scores = rnn_output.detach().numpy().flatten()

        # Sort movies based on RNN scores and select the top recommendations
        top_movie_indices = np.argsort(rnn_scores)[-top_k:][::-1]
        top_movies = [movies[i] for i in top_movie_indices]
        group_recommendations.append(top_movies)

        # Calculate updated group satisfaction and disagreement based on RNN recommendations
        group_sat = torch.tensor(calculate_group_satisfaction(group, top_movies, user_ratings), requires_grad=True)
        group_dis = torch.tensor(calculate_group_disagreement(group, top_movies, user_ratings), requires_grad=True)

        # Define the loss to maximize satisfaction and minimize disagreement
        loss = (1 - group_sat) + group_dis

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Iteration {iteration + 1}")
        print(f"Top Movies: {top_movies}")
        print(f"Group Satisfaction: {group_sat.item():.3f}")
        print(f"Group Disagreement: {group_dis.item():.3f}")
        print(f"Loss: {loss.item():.4f}\n")

    return group_recommendations


# Define a sample group of users
sample_group = [597, 217, 66, 177, 274, 391, 483, 561, 414, 509, 160]

# Run the sequential group recommendation process with RNN
recommendations = generate_sequential_recommendations_rnn(sample_group, user_ratings, movies)
