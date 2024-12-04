from functools import cached_property
from typing import List, Union, Callable, Optional

from pandas import DataFrame
from tqdm import tqdm

from prototypes.counterfactual_explanations.project.constants import MovieIDType, UserIDType
from prototypes.counterfactual_explanations.project.data import Group
from prototypes.counterfactual_explanations.project.movie_clusters import MoviesClustering


class MoviePreferencesFilterInterface:

    def __call__(self, preferences: List[MovieIDType]) -> List[MovieIDType]:
        pass


class MoveGlobalPopularityFilter(MoviePreferencesFilterInterface):

    def __init__(self, user_movie_matrix: DataFrame, percentile_threshold: float = 0.3):
        self.user_movie_matrix = user_movie_matrix
        self.percentile_threshold = percentile_threshold

    @cached_property
    def movie_rating_counts(self):
        user_movie_matrix_without_user_column = self.user_movie_matrix.drop(columns=['userId'])
        movie_rating_counts = user_movie_matrix_without_user_column.notna().sum()
        return movie_rating_counts

    @cached_property
    def popularity_percentile_value(self):
        percentile_value = self.movie_rating_counts.quantile(self.percentile_threshold)
        return percentile_value

    def __call__(self, preferences: List[MovieIDType]) -> List[MovieIDType]:
        return [movie for movie in preferences if self.movie_rating_counts[movie] > self.popularity_percentile_value]


class MovieGroupIntensityFilter(MoviePreferencesFilterInterface):

    def __init__(
            self,
            group: Group,
            users_rated_movie: Callable[[UserIDType], List[MovieIDType]],
            intensity_threshold: Union[float, int] = 0.5
    ):
        self.group = group
        self.users_rated_movie = users_rated_movie
        if isinstance(intensity_threshold, float):
            self.intensity_threshold = len(group) * intensity_threshold
        else:
            self.intensity_threshold = intensity_threshold

    def __call__(self, preferences: List[MovieIDType]) -> List[MovieIDType]:
        return [
            movie
            for movie in preferences
            if self.group.get_item_intensity(movie, self.users_rated_movie) > self.intensity_threshold
        ]


class MovieQueryClusterFilter(MoviePreferencesFilterInterface):

    def __init__(self, query: MovieIDType, cluster_threshold: Optional[Union[float, int]] = None):
        self.query = query
        self.cluster_threshold = cluster_threshold

    def get_cluster_threshold(self, clustering_movies: List[MovieIDType]) -> int:
        if self.cluster_threshold is None or isinstance(self.cluster_threshold, int):
            return self.cluster_threshold
        return int(len(clustering_movies) * self.cluster_threshold)

    def get_clustering(self, clustering_movies: List[MovieIDType]) -> MoviesClustering:
        return MoviesClustering(clustering_movies, self.get_cluster_threshold(clustering_movies))

    def __call__(self, preferences: List[MovieIDType]) -> List[MovieIDType]:
        clustering_movies = preferences
        if not self.query in clustering_movies:
            clustering_movies.append(self.query)

        clustering = self.get_clustering(clustering_movies)
        clusters, _ = clustering.cluster_movies(important_index=clustering_movies.index(self.query))
        query_cluster = []
        for cluster in clusters:
            if self.query in cluster:
                query_cluster = cluster
                break

        assert query_cluster, "Don't know why the query cluster is empty"

        return [movie for movie in preferences if movie in query_cluster]


class CompositeFilter(MoviePreferencesFilterInterface):

    def __init__(self, filters: List[MoviePreferencesFilterInterface]):
        self.filters = filters

    def apply_filter(
            self, preferences: List[MovieIDType], filter_: MoviePreferencesFilterInterface
    ) -> List[MovieIDType]:
        return filter_(preferences)

    def __call__(self, preferences: List[MovieIDType]) -> List[MovieIDType]:
        for filter_ in tqdm(self.filters, desc="Applying filters"):
            preferences = self.apply_filter(preferences, filter_)
        return preferences


class CompositeFilterWithCondition(CompositeFilter):

    def __init__(self, filters: List[MoviePreferencesFilterInterface], condition: Callable[[List[MovieIDType]], bool]):
        super().__init__(filters)
        self.condition = condition

    def apply_filter(
            self, preferences: List[MovieIDType], filter_: MoviePreferencesFilterInterface
    ) -> List[MovieIDType]:
        filtered_preferences = super().apply_filter(preferences, filter_)
        if self.condition(filtered_preferences):
            return filtered_preferences
        return preferences
