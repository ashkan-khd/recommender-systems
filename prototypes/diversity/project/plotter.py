from collections import defaultdict
from typing import *

from matplotlib import pyplot as plt


class Plotter:

    def __init__(self, years: List[int]):
        self.__data = dict(
            group_satisfaction=[],
            group_disagreement=[],
            group_diversity=[],
            users=defaultdict(lambda: dict(
                satisfaction=[],
                overall_satisfaction=[],
            )),
            group_overall_satisfaction=[],
            diversity_from_last_recommendation=[],
        )
        self.__initialized = True
        self.__years = years

    def add_group_satisfaction(self, satisfaction: float):
        self.__data['group_satisfaction'].append(satisfaction)

    def add_group_disagreement(self, disagreement: float):
        self.__data['group_disagreement'].append(disagreement)

    def add_group_diversity(self, diversity: float):
        self.__data['group_diversity'].append(diversity)

    def add_user_satisfaction(self, user_id: int, satisfaction: float):
        self.__data['users'][user_id]['satisfaction'].append(satisfaction)

    def add_user_overall_satisfaction(self, user_id: int, overall_satisfaction: float):
        self.__data['users'][user_id]['overall_satisfaction'].append(overall_satisfaction)

    def add_group_overall_satisfaction(self, overall_satisfaction: float):
        self.__data['group_overall_satisfaction'].append(overall_satisfaction)

    def add_diversity_from_last_recommendation(self, diversity: float):
        self.__data['diversity_from_last_recommendation'].append(diversity)

    def plot_group_satisfaction(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__years, self.__data['group_satisfaction'], marker='o')
        plt.xlabel('Year')
        plt.ylabel('Group Satisfaction')
        plt.title('Group Satisfaction Over Time')
        plt.xticks(self.__years)
        plt.grid(True)
        plt.show()

    def plot_group_disagreement(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__years, self.__data['group_disagreement'], marker='o')
        plt.xlabel('Year')
        plt.ylabel('Group Disagreement')
        plt.title('Group Disagreement Over Time')
        plt.xticks(self.__years)
        plt.grid(True)
        plt.show()

    def plot_group_diversity(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__years, self.__data['group_diversity'], marker='o')
        plt.xlabel('Year')
        plt.ylabel('Group Diversity')
        plt.title('Group Diversity Over Time')
        plt.xticks(self.__years)
        plt.grid(True)
        plt.show()

    def plot_users_satisfaction(self):
        plt.figure(figsize=(10, 6))
        for user_id, data in self.__data['users'].items():
            plt.plot(self.__years, data['satisfaction'], marker='o', label=f'User {user_id}')
        plt.xlabel('Year')
        plt.ylabel('User Satisfaction')
        plt.title('User Satisfaction Over Time')
        plt.xticks(self.__years)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_users_overall_satisfaction(self):
        plt.figure(figsize=(10, 6))
        for user_id, data in self.__data['users'].items():
            plt.plot(self.__years, data['overall_satisfaction'], marker='o', label=f'User {user_id}')
        plt.xlabel('Year')
        plt.ylabel('User Overall Satisfaction')
        plt.title('User Overall Satisfaction Over Time')
        plt.xticks(self.__years)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_group_overall_satisfaction(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__years, self.__data['group_overall_satisfaction'], marker='o')
        plt.xlabel('Year')
        plt.ylabel('Group Overall Satisfaction')
        plt.title('Group Overall Satisfaction Over Time')
        plt.xticks(self.__years)
        plt.grid(True)
        plt.show()

    def plot_diversity_from_last_recommendation(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__years[1:], self.__data['diversity_from_last_recommendation'], marker='o')
        plt.xlabel('Year')
        plt.ylabel('Average Diversity From Last Recommendation')
        plt.title('Average Diversity From Last Recommendation Over Time')
        plt.xticks(self.__years[1:])
        plt.grid(True)
        plt.show()

    def plot(self):
        self.plot_group_satisfaction()
        self.plot_group_disagreement()
        self.plot_group_diversity()
        self.plot_users_satisfaction()
        self.plot_users_overall_satisfaction()
        self.plot_group_overall_satisfaction()
        self.plot_diversity_from_last_recommendation()
