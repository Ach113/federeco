from typing import Tuple, List
from federeco.config import DEVICE
import numpy as np
import heapq
import torch
import math


# TODO: Seems very slow, perhaps can be optimized


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
                   k: int,
                   access_counters: List[int]) -> Tuple[float, float]:
    """
    calculates hit rate and normalized discounted cumulative gain for each user across each item in `negatives`
    returns average of top-k list of hit rates and ndcgs
    """

    # access_counters = np.array(access_counters)
    # blacklist = set(np.where(access_counters == 0)[0].tolist())
    #
    # print(f'clients with 0 access: {int(100 * len(blacklist)/len(access_counters))}%')

    hits, ndcgs = list(), list()
    for i, user in enumerate(users):

        # if user in blacklist:
        #     continue

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
