import numpy as np
import pandas as pd
# from model import collaborative_filtering_model

from config import *


class Client:

    def __init__(self, client_id):
        self.client_id = client_id
        self.client_data = None

    def set_client_data(self, data_array):
        self.client_data = pd.DataFrame({
            'user_id': data_array[0],
            'item_id': data_array[1],
            'label': data_array[2]
        })

    def train(self, server_model):

        user_input, item_input = self.client_data['user_id'], self.client_data['item_id']
        labels = self.client_data['label']

        server_model.fit([user_input, item_input], labels,
                         batch_size=BATCH_SIZE, epochs=LOCAL_EPOCHS, verbose=0, shuffle=True)

        weights = np.array(server_model.get_weights(), dtype='object')

        return weights
