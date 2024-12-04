import time

from black_box.model_v2 import GroupRecommendationModel

if __name__ == '__main__':
    group = [269, 120, 278, 163, 161, 383, 238, 392, 188, 35]
    start_time = time.time()
    model = GroupRecommendationModel(group)
    recommendation = model.recommend()
    resulted_in = time.time() - start_time
    print(recommendation)
    print(f"in {resulted_in} seconds")
