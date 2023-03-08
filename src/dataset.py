import random
import numpy as np
import pandas as pd

from config import *


def load_movie_data(limit: int = None) -> pd.DataFrame:
    df = pd.read_csv(TRAIN_DATA_PATH, header=None, sep='\t')
    df.columns = ['user_id', 'item_id', 'rating', 'ts']
    df = df.drop(columns=['ts'])

    if limit:
        return df.sample(limit)

    return df


def get_train_instances(df):
    users, items = df['user_id'].tolist(), df['item_id'].tolist()
    num_items = len(items)

    user_input, item_input, labels = [], [], []
    zipped = set(zip(users, items))

    for (u, i) in zip(users, items):
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        for t in range(NUM_NEGATIVES):
            j = np.random.randint(num_items)
            while (u, j) in zipped:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input, labels
