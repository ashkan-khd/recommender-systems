import heapq
from collections import defaultdict


def threshold_algorithm(sorted_user_recommendations, random_access_user_recommendations, K):
    # Initialize data structures
    item_group_score = defaultdict(float)
    item_seen = defaultdict(lambda: False)
    heap = []

    # Do sorted access until we get the top K items with scores above the threshold
    for i in range(len(sorted_user_recommendations[0])):  # Loop over items by ranked position
        sum_last_scores = 0.0
        # Do sorted access for each user
        for user_idx, user_list in enumerate(sorted_user_recommendations):
            item, score = user_list[i]
            sum_last_scores += score  # Update the last seen score for threshold

            if item not in item_group_score:
                for u in range(len(random_access_user_recommendations)):
                    item_group_score[item] += random_access_user_recommendations[u][item]

            if not item_seen[item]:
                group_score = item_group_score[item]

                # Insert into heap as a candidate for top-K
                if len(heap) < K:
                    heapq.heappush(heap, (group_score, item))
                else:
                    if group_score > heap[0][0]:
                        heapq.heappushpop(heap, (group_score, item))

                item_seen[item] = True

        # Calculate threshold as the sum of the last scores seen in each user list
        threshold = sum_last_scores

        # Check if we can stop
        if len(heap) == K and heap[0][0] >= threshold:
            break

    # Extract top K items from the heap
    top_k_items = sorted(heap, key=lambda x: -x[0])
    return [(item, score) for score, item in top_k_items]


def example_1():
    # Test input
    user_recommendations = [
        [("A", 10), ("B", 8), ("C", 6), ("D", 4), ("E", 2)],
        [("B", 9), ("A", 7), ("D", 5), ("E", 3), ("C", 1)],
        [("C", 8), ("A", 6), ("E", 4), ("B", 2), ("D", 1)]
    ]
    user_recommendations_for_random_access = [
        {"A": 10, "B": 8, "C": 6, "D": 4, "E": 2},
        {"A": 7, "B": 9, "C": 1, "D": 5, "E": 3},
        {"A": 6, "B": 2, "C": 8, "D": 1, "E": 4},
    ],
    K = 3

    # Running the algorithm
    top_k_recommendations = threshold_algorithm(
        user_recommendations,
        user_recommendations_for_random_access,
        K
    )
    print(f"Top-{K} recommendations:", top_k_recommendations)


def example_2():

    # Test input
    user_recommendations = [
        [("X1", 1.0), ("X2", 0.8), ("X3", 0.5), ("X4", 0.3), ("X5", 0.1)],
        [("X2", 0.8), ("X3", 0.7), ("X1", 0.3), ("X4", 0.2), ("X5", 0.1)],
        [("X4", 0.8), ("X3", 0.6), ("X1", 0.2), ("X5", 0.1), ("X2", 0.0)]
    ]
    user_recommendations_for_random_access = [
        {"X1": 1.0, "X2": 0.8, "X3": 0.5, "X4": 0.3, "X5": 0.1},
        {"X1": 0.3, "X2": 0.8, "X3": 0.7, "X4": 0.2, "X5": 0.1},
        {"X1": 0.2, "X2": 0.0, "X3": 0.6, "X4": 0.8, "X5": 0.1},
    ]
    K = 2

    # Running the algorithm
    top_k_recommendations = threshold_algorithm(
        user_recommendations,
        user_recommendations_for_random_access,
        K
    )
    print(f"Top-{K} recommendations:", top_k_recommendations)



example_2()