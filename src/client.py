import torch
import numpy as np
import collections
import pandas as pd
from typing import List
from torch import Tensor

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

    def train(self, server_model: torch.nn.Module) -> collections.OrderedDict:
        user_input, item_input = self.client_data['user_id'], self.client_data['item_id']
        labels = self.client_data['label']

        user_input = torch.tensor(user_input, dtype=torch.int)
        item_input = torch.tensor(item_input, dtype=torch.int)
        labels = torch.tensor(labels, dtype=torch.int)

        optimizer = torch.optim.AdamW(server_model.parameters(), lr=LEARNING_RATE)
        for _ in range(LOCAL_EPOCHS):
            logits, loss = server_model(user_input, item_input, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        return server_model.state_dict()
