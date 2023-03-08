import math
import heapq
import numpy as np

# TODO: optimize


def evaluate_model(model, test_data, test_negatives, k):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    hits, ndcgs = [[] for _ in range(k)], [[] for _ in range(k)]
    for idx in range(len(test_data)):
        rating = test_data[idx]
        items = test_negatives[idx]
        user_id = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), user_id, dtype='int32')
        predictions = model.predict([users, np.array(items)],
                                         batch_size=100, verbose=0)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()
        # Evaluate top rank list
        ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        if gtItem in ranklist:
            p = ranklist.index(gtItem)
            for i in range(p):
                hits[i].append(0)
                ndcgs[i].append(0)
            for i in range(p, k):
                hits[i].append(1)
                ndcgs[i].append(math.log(2) / math.log(ranklist.index(gtItem) + 2))
        else:
            for i in range(k):
                hits[i].append(0)
                ndcgs[i].append(0)
    hits = [np.array(hits[i]).mean() for i in range(k)]
    ndcgs = [np.array(ndcgs[i]).mean() for i in range(k)]
    return hits, ndcgs