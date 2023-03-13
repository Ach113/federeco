"""
based on code taken from: https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
"""
import math
import heapq
import numpy as np
import tensorflow as tf
from typing import Tuple, List

# TODO!: Seems very slow, perhaps can be optimized


def get_metrics(rank_list: List, item: int) -> Tuple[int, float]:
    if item not in rank_list:
        return 0, 0
    return 1, math.log(2) / math.log(rank_list.index(item) + 2)


def evaluate_model(model: tf.keras.Model,
                   users: List[int], items: List[int], negatives: List[List[int]],
                   k: int, n_workers: int = 1) -> Tuple[float, float]:
    """
    calculates hit rate and normalized discounted cumulative gain for each user across each item in `negatives`
    returns average of top-k list of hit rates and ndcgs
    """
    hits, ndcgs = list(), list()
    for i, user in enumerate(users):
        item = items[i]
        negatives = negatives[i] + [item]
        pred = model.predict([np.full(len(negatives), user, dtype='int32'), np.array(negatives)],
                             batch_size=100, verbose=0, workers=n_workers)
        map_item_score = dict(zip(negatives, pred))
        rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hr, ndcg = get_metrics(rank_list, item)
        hits.append(hr)
        ndcgs.append(ndcg)

    return np.array(hits).mean(), np.array(ndcgs).mean()
