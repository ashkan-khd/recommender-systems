from typing import *

from prototypes.diversity.project.constants import *


class Diversifier:

    def F(self, movie_list: List[MovieIDType]):
        return (
                ((self.k - 1) * (1 - self.landa) * self.sim(movie_list))
                +
                (2 * self.landa * self.div(movie_list))
        )

    def __init__(
            self,
            k: int, landa: float,
            sim: Callable[[List[MovieIDType], ], float],
            sim_individual: Callable[[MovieIDType, ], float],
            div: Callable[[List[MovieIDType], ], float],
    ):
        self.k = k
        self.landa = landa
        self.sim = sim
        self.sim_individual = sim_individual
        self.div = div

    def diversify(self, S: List[MovieIDType]) -> List[MovieIDType]:
        R = S[:self.k]
        S = S[self.k:]
        for movie in S:
            R_prime = R.copy()
            for selected_movie in R:
                new_R = R.copy()
                new_R.remove(selected_movie)
                new_R.append(movie)
                if self.F(new_R) > self.F(R_prime):
                    R_prime = new_R
            if self.F(R_prime) > self.F(R):
                R = R_prime
        return R
