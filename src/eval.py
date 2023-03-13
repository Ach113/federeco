"""
based on code taken from: https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
"""
import math
import heapq
import numpy as np

# TODO!: Seems very slow, perhaps can be optimized


def get_metrics(rank_list, item):
    if item not in rank_list:
        return 0, 0
    return 1, math.log(2) / math.log(rank_list.index(item) + 2)


def evaluate_model(model, test_ratings, test_negatives, k: int, n_threads: int = 1):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    hits, ndcgs = list(), list()
    users, items = zip(*test_ratings)
    for i, user in enumerate(users):
        item = items[i]
        negatives = test_negatives[i] + [item]
        pred = model.predict([np.full(len(negatives), user, dtype='int32'), np.array(negatives)],
                             batch_size=100, verbose=0, workers=4)
        map_item_score = dict(zip(negatives, pred))
        rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hr, ndcg = get_metrics(rank_list, item)
        hits.append(hr)
        ndcgs.append(ndcg)

    return np.array(hits).mean(), np.array(ndcgs).mean()
