import math
import torch
import heapq
import numpy as np
from typing import Tuple, List

from config import DEVICE


# TODO!: Seems very slow, perhaps can be optimized


def get_metrics(rank_list: List, item: int) -> Tuple[int, float]:
    """
    Used for calculating hit rate & normalized discounted cumulative gain (ndcg)
    :param rank_list: Top-k list of recommendations
    :param item: item we are trying to match with `rank_list`
    :return: tuple containing 1/0 indicating hit/no hit & ndcg
    """
    if item not in rank_list:
        return 0, 0
    return 1, math.log(2) / math.log(rank_list.index(item) + 2)


def evaluate_model(model: torch.nn.Module,
                   users: List[int], items: List[int], negatives: List[List[int]],
                   k: int) -> Tuple[float, float]:
    """
    calculates hit rate and normalized discounted cumulative gain for each user across each item in `negatives`
    returns average of top-k list of hit rates and ndcgs
    """

    # TODO: for some reason this function fails for pinterest dataset

    hits, ndcgs = list(), list()
    for i, user in enumerate(users):
        item = items[i]

        with torch.no_grad():
            item_input = torch.tensor(np.array(negatives[i] + [item]), dtype=torch.int, device=DEVICE)
            user_input = torch.tensor(np.full(len(item_input), user, dtype='int32'), dtype=torch.int, device=DEVICE)
            pred, _ = model(user_input, item_input)

        map_item_score = dict(zip(item_input, pred))
        rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hr, ndcg = get_metrics(rank_list, item)
        hits.append(hr)
        ndcgs.append(ndcg)

    return np.array(hits).mean(), np.array(ndcgs).mean()
