import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional

from config import *


class Client:

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.client_data = None

    def set_client_data(self, data_array: List[np.ndarray]):
        self.client_data = pd.DataFrame({
            'user_id': data_array[0],
            'item_id': data_array[1],
            'label': data_array[2]
        })

    def train(self, server_model: tf.keras.models.Model) -> np.ndarray:
        user_input, item_input = self.client_data['user_id'], self.client_data['item_id']
        labels = self.client_data['label']

        server_model.fit([user_input, item_input], labels,
                         batch_size=BATCH_SIZE, epochs=LOCAL_EPOCHS, verbose=0, shuffle=True)

        weights = np.array(server_model.get_weights(), dtype='object')

        return weights

    def generate_recommendation(self, server_model: tf.keras.Model, k: Optional[int] = 5) -> List[int]:
        """
        :param server_model: server model which will be used to generate predictions
        :param k: number of recommendations to generate
        :return: list of `k` movie recommendations
        """
        # get movies that user has not yet interacted with
        movies = list(set(range(NUM_ITEMS)).difference(set(self.client_data['item_id'].tolist())))
        movies = np.array(movies)
        client_id = np.array([self.client_id for _ in range(len(movies))])
        # obtain predictions in terms of logit per movie
        logits = server_model.predict([client_id, movies])

        rec_dict = {movie: p for movie, p in zip(movies, logits.squeeze().tolist())}
        # select top k recommendations
        top_k = sorted(rec_dict.items(), key=lambda x: -x[1])[:k]
        rec, _ = zip(*top_k)

        return rec
