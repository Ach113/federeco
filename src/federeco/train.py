from typing import List, Any
import collections
import numpy as np
import torch
import copy
import tqdm


def sample_clients(clients: List, num_clients: int) -> List:
    """
    :param clients: list of all available clients
    :param num_clients: number of clients to sample

    sample `num_clients` clients and return along with their respective data
    """
    return np.random.choice(clients, size=num_clients, replace=False).tolist()


def training_process(clients: List,
                     server_model: torch.nn.Module,
                     num_clients: int,
                     num_rounds: int) -> dict[str, Any]:
    """
    :param clients: list of all clients in the system
    :param server_model: server model which is used for training
    :param num_clients: number of clients to sample during single training iteration
    :param num_rounds: total number of training rounds

    per single training round:
        1. samples `num_clients` clients
        2. trains each client locally `LOCAL_EPOCHS` number of times
        3. aggregates weights across `num_clients` clients and sets them to server model
    returns weights of a trained model
    """

    for _ in tqdm.tqdm(range(num_rounds)):
        clients = sample_clients(clients, num_clients)
        w = single_train_round(server_model, clients)
        updated_server_weights = federated_averaging(w)
        server_model.load_state_dict(updated_server_weights)

    return server_model.state_dict()


def single_train_round(server_model: torch.nn.Module,
                       clients: List) -> List[collections.OrderedDict]:
    """
    single round of federated training.
    Trains all clients locally on their respective datasets
    Returns weights of client models as a list
    """
    client_weights = list()
    for client in clients:
        weights = client.train(server_model)
        client_weights.append(weights)
    return client_weights


def federated_averaging(client_weights: List[collections.OrderedDict]) -> collections.OrderedDict:
    """
    calculates the average of client weights
    """
    keys = client_weights[0].keys()
    averages = copy.deepcopy(client_weights[0])

    for w in client_weights[1:]:
        for key in keys:
            averages[key] += w[key]

    for key in keys:
        averages[key] /= len(client_weights)
    return averages
