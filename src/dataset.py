from typing import List, Tuple
import pandas as pd
import scipy as sp
import numpy as np
import sys
import os

from federeco.config import NUM_NEGATIVES


class Dataset:

    def __init__(self, dataset: str):
        if dataset not in ['movielens', 'pinterest', 'yelp']:
            print(f'Error: unknown dataset {dataset}')
            sys.exit(-1)

        columns = ['user_id', 'item_id', 'rating']
        self.train_df = pd.read_csv(os.path.join('data', dataset + '-train.csv'), header=None, names=columns)
        self.test_df = pd.read_csv(os.path.join('data', dataset + '-test.csv'), names=columns)
        self.neg_path = os.path.join('data', dataset + '-neg.csv')
        self.num_users, self.num_items = self.get_matrix_dim()
        print(f'Loaded `{dataset}` dataset: \nNumber of users: {self.num_users}, Number of items: {self.num_items}')

    def get_matrix_dim(self) -> Tuple[int, int]:
        """
        returns number of unique users and unique items in the dataset
        :return: (number of users, number of items)
        """
        num_users = max(self.train_df['user_id']) + 1
        num_items = max(self.train_df['item_id']) + 1
        return num_users, num_items

    def load_client_train_data(self) -> List[List]:
        """
        Creates a matrix of client data for training the model in form of [[user_id, item_id, label], ...].
        Each row in the matrix corresponds to an interaction between a user and an item.
        Label 1 indicates user has interacted with the item, 0 indicates no interaction.
        Number of negative samples (label 0) per positive sample (label 1) is defined in `config.py` as NUM_NEGATIVES
        :return: training data in form of 2-dimensional list
        """
        mat = sp.sparse.dok_matrix((self.num_users+1, self.num_items+1), dtype=np.float32)

        for user, item, rating in self.train_df.values:
            if rating > 0:
                mat[user, item] = 1.0

        client_datas = [[[], [], []] for _ in range(self.num_users)]

        for (usr, item) in mat.keys():
            client_datas[usr][0].append(usr)
            client_datas[usr][1].append(item)
            client_datas[usr][2].append(1)
            for t in range(NUM_NEGATIVES):
                neg = np.random.randint(self.num_items)
                while (usr, neg) in mat.keys():
                    neg = np.random.randint(self.num_items)
                client_datas[usr][0].append(usr)
                client_datas[usr][1].append(neg)
                client_datas[usr][2].append(0)

        return client_datas

    def load_test_file(self) -> List[List[int]]:
        """
        test file provides single item interaction per unique client in the dataset
        used for evaluating the trained model
        :return: list in the form of [[user_id, item_id], ...]
        """
        return [[user, item] for user, item, _ in self.test_df.values]

    def load_negative_file(self) -> List[List[int]]:
        """
        negative file provides items ids that have not been interacted with (for each unique client)
        file has the form of (user_id) [item_0, item_1, ...]
        :return: matrix where each row contains items not interacted with by the user in given index
        """
        negative_list = []
        with open(self.neg_path, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negative_list.append(negatives)
                line = f.readline()
        return negative_list

    @staticmethod
    def generate_negatives(user_ids,
                           item_ids,
                           n: int):
        """
        generates `n` samples of negatives per user id
        :return:
        """
        return
