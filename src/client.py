from federeco.config import BATCH_SIZE, DEVICE
from typing import List, Optional, Any, Tuple
from torch import Tensor
import pandas as pd
import numpy as np
import federeco
import torch


class Client(federeco.client.Client):

    def __init__(self, client_id: int):
        super().__init__(client_id)
        self.client_id = client_id
        self.client_data = None

    def set_client_data(self, data_array: List[np.ndarray]):
        self.client_data = pd.DataFrame({
            'user_id': data_array[0],
            'item_id': data_array[1],
            'label': data_array[2]
        })

    def train(self, server_model: torch.nn.Module, local_epochs: int, learning_rate: float) -> Tuple[dict[str, Any], Tensor]:
        """
        single round of local training for client
        :param server_model: pytorch model that can be trained on user data
        :param local_epochs: number of local training epochs per global epoch
        :param learning_rate: learning rate for the neural network
        :return: weights of the server model, training loss
        """
        user_input, item_input = self.client_data['user_id'], self.client_data['item_id']
        labels = self.client_data['label']

        user_input = torch.tensor(user_input, dtype=torch.int, device=DEVICE)
        item_input = torch.tensor(item_input, dtype=torch.int, device=DEVICE)
        labels = torch.tensor(labels, dtype=torch.int, device=DEVICE)

        # utilize dataloader to train in batches
        dataset = torch.utils.data.TensorDataset(user_input, item_input, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        # perturb the learning rate
        # learning_rate = abs(learning_rate + np.random.normal(0, .01, 1).squeeze())
        optimizer = torch.optim.AdamW(server_model.parameters(), lr=learning_rate)
        loss = None
        for _ in range(local_epochs):
            for _, (u, i, l) in enumerate(dataloader):
                logits, loss = server_model(u, i, l)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(server_model.parameters(), 0.5)
                optimizer.step()

        return server_model.state_dict(), loss

    def generate_recommendation(self, server_model: torch.nn.Module,
                                num_items: int,  k: Optional[int] = 5) -> List[int]:
        """
        :param server_model: server model which will be used to generate predictions
        :param num_items: total number of unique items in dataset
        :param k: number of recommendations to generate
        :return: list of `k` movie recommendations
        """
        # get movies that user has not yet interacted with
        movies = set(range(num_items)).difference(set(self.client_data['item_id'].tolist()))
        movies = torch.tensor(list(movies), dtype=torch.int, device=DEVICE)
        client_id = torch.tensor([self.client_id for _ in range(len(movies))], dtype=torch.int, device=DEVICE)
        # obtain predictions in terms of logit per movie
        with torch.no_grad():
            logits, _ = server_model(client_id, movies)

        rec_dict = {movie: p for movie, p in zip(movies.tolist(), logits.squeeze().tolist())}
        # select top k recommendations
        top_k = sorted(rec_dict.items(), key=lambda x: -x[1])[:k]
        rec, _ = zip(*top_k)

        return rec
