import numpy as np
from model import collaborative_filtering_model

from config import *


class Client:

    def __init__(self, client_id):
        self.client_id = client_id
        self.client_data = {}

    def set_client_data(self, df):
        self.client_data = df

    def train(self, server_model, client_id, server_weights):
        server_model.set_weights(server_weights)

        user_input, item_input = self.client_data['user_id'], self.client_data['item_id']
        labels = self.client_data['rating']

        server_model.fit([user_input, item_input], labels,
                         batch_size=BATCH_SIZE, epochs=LOCAL_EPOCHS, verbose=0, shuffle=True)

        weights = np.array(server_model.get_weights())

        return weights
