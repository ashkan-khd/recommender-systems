import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def calculate_user_satisfaction(user_id, group_rec, user_movie_tensor, top_k=10):
    with torch.no_grad():  # These indices don't need gradients
        top_user_prefs = torch.topk(user_movie_tensor[user_id], top_k).indices
    user_ideal_satisfaction = torch.sum(user_movie_tensor[user_id][top_user_prefs])

    top_group_rec = torch.topk(group_rec, top_k).indices
    group_satisfaction = torch.sum(user_movie_tensor[user_id][top_group_rec])
    return group_satisfaction / user_ideal_satisfaction if user_ideal_satisfaction > 0 else 0


def calculate_group_satisfaction(group, group_rec, user_ratings):
    individual_satisfactions = torch.tensor([
        calculate_user_satisfaction(user, group_rec, user_ratings) for user in range(len(group))
    ], dtype=torch.float32, requires_grad=True)
    return torch.mean(individual_satisfactions)


def calculate_group_disagreement(group, group_rec, user_ratings):
    individual_satisfactions = torch.tensor([
        calculate_user_satisfaction(user, group_rec, user_ratings) for user in range(len(group))
    ], dtype=torch.float32, requires_grad=True)
    return torch.max(individual_satisfactions) - torch.min(individual_satisfactions)


class ModifiedReLU05(nn.Module):
    def forward(self, x):
        return torch.minimum(F.relu(x), torch.tensor(5.0))


class GroupRecommendationRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.activation = ModifiedReLU05()

    def forward(self, embedded, hidden=None):
        # x: (batch_size, sequence_length)
        # embedded = self.embedding(x)
        # embedded: (batch_size, sequence_length, embedding_dim)
        prediction, hidden = self.rnn(embedded, hidden)
        # output: (batch_size, sequence_length, hidden_dim)
        # hidden: (1, batch_size, hidden_dim)
        output = self.fc(hidden)
        # hidden: (1, batch_size, vocab_size)
        output = self.activation(output)
        return output.squeeze(0).squeeze(0), hidden


def generate_sequential_recommendations_rnn(group, user_movie_matrix, movies, iterations=5):
    group_recommendations = []
    user_movie_tensor = torch.tensor(user_movie_matrix.iloc[:, 1:].values, dtype=torch.float32)

    model = GroupRecommendationRNN(len(movies), len(movies), 2 * len(movies))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    hidden = None
    for iteration in range(iterations):
        optimizer.zero_grad()
        group_rec, hidden = model(user_movie_tensor.unsqueeze(0), hidden)

        gs = calculate_group_satisfaction(group, group_rec, user_movie_tensor)
        gd = calculate_group_disagreement(group, group_rec, user_movie_tensor)
        loss = (1 - gs) + gd
        loss.backward()
        optimizer.step()
        print(loss)
        print(group_rec)
        print()

    return group_recommendations



user_movie_df = pd.read_csv("user_movie_matrix_group.csv")
movies = user_movie_df.columns[1:].tolist()

# Define a sample group of users
sample_group = [597, 217, 66, 177, 274, 391, 483, 414, 509, 160]

# Run the sequential group recommendation process with RNN
recommendations = generate_sequential_recommendations_rnn(sample_group, user_movie_df, movies)
