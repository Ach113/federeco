from typing import Tuple, List
import numpy as np
import heapq
import torch
import math

from federeco.config import DEVICE


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
    calculates hit rate and normalized discounted cumulative gain for each user.
    - generates prediction of top-k recommendations using FedNCF model
        FedNCF takes two inputs:
            1. vector of user ids
            2. item vector which contains negative samples plus single positive sample (item rated by user)
    - hit rate is calculated based on whether the top-k recommendation list contains the positive item
    - ndcg is calculated based on the position of the element in the top-k list

    :param model: FedNCF model for generating recommendations
    :param users: user ids in test dataset
    :param items: items rated by users in test dataset
    :param negatives: items not rated by the users in the test dataset
    :param k: number of top recommendations to use when calculating hr/ndcg
    :return: average hit rates and ndcgs in top-k recommendations
    """

    hits, ndcgs = list(), list()
    for user, item, neg in zip(users, items, negatives):

        item_input = neg + [item]

        with torch.no_grad():
            item_input_gpu = torch.tensor(np.array(item_input), dtype=torch.int, device=DEVICE)
            user_input = torch.tensor(np.full(len(item_input), user, dtype='int32'), dtype=torch.int, device=DEVICE)
            pred, _ = model(user_input, item_input_gpu)
            pred = pred.cpu().numpy().tolist()

        map_item_score = dict(zip(item_input, pred))
        rank_list = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hr, ndcg = get_metrics(rank_list, item)
        hits.append(hr)
        ndcgs.append(ndcg)

    return np.array(hits).mean(), np.array(ndcgs).mean()
