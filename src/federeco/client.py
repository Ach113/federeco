from typing import Tuple, Any, Optional, List
from abc import ABC, abstractmethod
from torch import Tensor
import torch


"""
abstract Client class which is used by `federeco.train.single_training_round()` to train clients locally
`federeco` models must be trained by data wrapped by `Client` objects
"""


class Client(ABC):

    @abstractmethod
    def __init__(self, client_id: int):
        self.client_id = client_id

    @abstractmethod
    def train(self, server_model: torch.nn.Module) -> Tuple[dict[str, Any], Tensor]:
        """
        single round of local training for client
        :param server_model: pytorch model that can be trained on user data
        :return: weights of the server model, training loss
        """
        pass

    @abstractmethod
    def generate_recommendation(self, server_model: torch.nn.Module,
                                num_items: int,  k: Optional[int] = 5) -> List[int]:
        """
        :param server_model: server model which will be used to generate predictions
        :param k: number of recommendations to generate
        :param num_items: total number of unique items in dataset
        :return: list of `k` movie recommendations
        """
        pass
