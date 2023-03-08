import scipy as sp
import numpy as np

from config import *

# TODO: convert to pandas


def load_client_train_data():
    mat = sp.sparse.dok_matrix((NUM_USERS, NUM_ITEMS), dtype=np.float32)
    with open(TRAIN_DATA_PATH, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            # print("usr:{} item:{} score:{}".format(user,item,rating))
            if rating > 0:
                mat[user, item] = 1.0
            line = f.readline()

    client_datas = [[[], [], []] for _ in range(NUM_USERS)]
    with open(TRAIN_DATA_PATH, "r") as f:
        for (usr, item) in mat.keys():
            client_datas[usr][0].append(usr)
            client_datas[usr][1].append(item)
            client_datas[usr][2].append(1)
            line = f.readline()
            for t in range(4):
                nega_item = np.random.randint(NUM_ITEMS)
                while (usr, nega_item) in mat.keys():
                    nega_item = np.random.randint(NUM_ITEMS)
            client_datas[usr][0].append(usr)
            client_datas[usr][1].append(nega_item)
            client_datas[usr][2].append(0)
    return client_datas


def load_test_file():
    filename = TEST_DATA_PATH
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file():
    filename = NEGATIVE_DATA_PATH
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList