from dataclasses import dataclass
from typing import List, Callable

from prototypes.counterfactual_explanations.project.constants import UserIDType, MovieIDType


@dataclass
class Group:
    users: List[UserIDType]

    def get_item_intensity(
            self,
            movie: MovieIDType,
            users_rated_movie: Callable[[UserIDType], List[MovieIDType]],
    ) -> float:
        return len([user for user in self.users if user in users_rated_movie(movie)])

    def __len__(self):
        return len(self.users)

    def __iter__(self):
        return iter(self.users)


@dataclass
class Explanation:
    movies: List[MovieIDType]

    def get_user_intensity(
            self,
            user: UserIDType,
            movies_rated_for_user: Callable[[UserIDType], List[MovieIDType]],
    ) -> float:
        return len([movie for movie in self.movies if movie in movies_rated_for_user(user)])

    def __len__(self):
        return len(self.movies)

    def __iter__(self):
        return iter(self.movies)
