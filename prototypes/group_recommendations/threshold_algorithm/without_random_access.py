import heapq
from collections import defaultdict


def threshold_algorithm(user_recommendations, K):
    # Initialize data structures
    item_scores = defaultdict(lambda: [None] * len(user_recommendations))
    item_counts = defaultdict(int)
    heap = []

    # Track the last score seen in each user list to compute the threshold
    last_scores = [None] * len(user_recommendations)

    # Do sorted access until we get the top K items with scores above the threshold
    for i in range(len(user_recommendations[0])):  # Loop over items by ranked position
        # Do sorted access for each user
        for user_idx, user_list in enumerate(user_recommendations):
            item, score = user_list[i]
            last_scores[user_idx] = score  # Update the last seen score for threshold

            # Update scores for the item
            if item_scores[item][user_idx] is None:  # If we haven't seen this item for this user
                item_scores[item][user_idx] = score
                item_counts[item] += 1

            # If we've collected scores from all users, compute the group score
            if item_counts[item] == len(user_recommendations):
                group_score = sum(filter(None, item_scores[item]))

                # Insert into heap as a candidate for top-K
                if len(heap) < K:
                    heapq.heappush(heap, (group_score, item))
                else:
                    if group_score > heap[0][0]:
                        heapq.heappushpop(heap, (group_score, item))

        # Calculate threshold as the sum of the last scores seen in each user list
        threshold = sum(filter(None, last_scores))

        # Check if we can stop
        if len(heap) == K and heap[0][0] >= threshold:
            break

    # Extract top K items from the heap
    top_k_items = sorted(heap, key=lambda x: -x[0])
    return [(item, score) for score, item in top_k_items]


# Test input
user_recommendations = [
    [("A", 10), ("B", 8), ("C", 6), ("D", 4), ("E", 2)],
    [("B", 9), ("A", 7), ("D", 5), ("E", 3), ("C", 1)],
    [("C", 8), ("A", 6), ("E", 4), ("B", 2), ("D", 1)]
]
K = 3

# Running the algorithm
top_k_recommendations = threshold_algorithm(user_recommendations, K)
print("Top-K recommendations:", top_k_recommendations)
