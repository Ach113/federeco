import scipy as sp
import numpy as np
import sys
import os

from config import NUM_NEGATIVES

# TODO: convert to pandas


class Dataset:

    def __init__(self, data: str):
        if data == 'movielens':
            s = 'ml-1m'
            self.num_users = 6040
            self.num_items = 3706
        elif data == 'pinterest':
            s = 'pinterest-20'
            self.num_users = 55187
            self.num_items = 9915

        else:
            print(f'Error: unknown dataset {data}')
            sys.exit(-1)

        self.train_path = os.path.join('data', s + '.train.rating')
        self.test_path = os.path.join('data', s + '.test.rating')
        self.neg_path = os.path.join('data', s + '.test.negative')

    def load_client_train_data(self):
        mat = sp.sparse.dok_matrix((self.num_users+1, self.num_items+1), dtype=np.float32)
        with open(self.train_path, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()

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

    def load_test_file(self):
        rating_list = []
        with open(self.test_path, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                rating_list.append([user, item])
                line = f.readline()
        return rating_list

    def load_negative_file(self):
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
