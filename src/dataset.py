from typing import List
import pandas as pd
import scipy as sp
import numpy as np
import sys
import os

from federeco.config import NUM_NEGATIVES


class Dataset:

    def __init__(self, data: str):

        if data == 'movielens':
            s = 'ml-1m'
            self.num_users = 6040
            self.num_items = 3706
        elif data == 'pinterest':
            s = 'pinterest-20'
            self.num_users = 55187
            self.num_items = 9916
        elif data == 'yelp':
            s = 'yelp-2018'
            self.num_users = 1326101
            self.num_items = 174567
        else:
            print(f'Error: unknown dataset {data}')
            sys.exit(-1)

        columns = ['user_id', 'item_id', 'rating']
        self.train_df = pd.read_csv(os.path.join('data', s + '-train.csv'), header=None, names=columns)
        self.test_df = pd.read_csv(os.path.join('data', s + '-test.csv'), names=columns)
        self.neg_path = os.path.join('data', s + '-neg.csv')

    def load_client_train_data(self) -> List:
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
        return [[user, item] for user, item, _ in self.test_df.values]

    def load_negative_file(self) -> List[List[int]]:
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
    def get_movie_names(movie_ids: List[int]) -> List[str]:
        movie_names = list()
        with open('data/movies.dat', 'r') as f:
            lines = f.readlines()

        for line in lines:
            _, movie_name, _ = line.split('::')
            movie_names.append(movie_name)

        return [movie_names[i] for i in movie_ids]

    @staticmethod
    def generate_negatives(user_ids,
                           item_ids,
                           n: int):
        """
        generates `n` samples of negatives per user id
        :return:
        """
        return
