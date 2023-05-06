from typing import List, Any, Tuple
import collections
import random
import torch
import copy
import tqdm

from federeco.dataset import Dataset
from federeco.client import Client
from federeco.eval import evaluate_model

import pickle


def sample_clients(clients: List[Client], num_clients: int) -> Tuple[List[Client], List[Client]]:
    """
    :param clients: list of all available clients
    :param num_clients: number of clients to sample

    sample `num_clients` clients and return along with their respective data
    """
    sample = clients[:num_clients]
    # rotate the list by `num_clients`
    clients = clients[num_clients:] + clients[:num_clients]

    return sample, clients


def training_process(server_model: torch.nn.Module,
                     all_clients: List[Client],
                     num_clients: int,
                     epochs: int,
                     dataset: Dataset) -> dict[str, Any]:
    """
    :param server_model: server model which is used for training
    :param all_clients: list of all clients in the system
    :param num_clients: number of clients to sample during single training iteration
    :param epochs: total number of training rounds
    :return: weights of a trained model

    per single training round:
        1. samples `num_clients` clients
        2. trains each client locally `LOCAL_EPOCHS` number of times
        3. aggregates weights across `num_clients` clients and sets them to server model
    """

    random.shuffle(all_clients)

    test_data, negatives = dataset.load_test_file(), dataset.load_negative_file()
    users, items = zip(*test_data)
    hrs, ndcgs = list(), list()

    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        # sample `num_clients` clients for training
        clients, all_clients = sample_clients(all_clients, num_clients)
        # apply single round of training
        w, loss = single_train_round(server_model, clients)
        # aggregate weights
        updated_server_weights = federated_averaging(w)
        # set aggregated weights to server model
        server_model.load_state_dict(updated_server_weights)
        # display progress bar with epochs & mean loss of single training round
        pbar.set_description(f'epoch: {epoch+1}, loss: {loss:.2f}')
        if epoch % 5 == 0:
            hr, ndcg = evaluate_model(server_model, users, items, negatives, k=10)
            hrs.append(hr)
            ndcgs.append(ndcg)

    with open('hr_dump.pkl', 'wb') as f:
        pickle.dump(hrs, f)

    with open('ndcg_dump.pkl', 'wb') as f:
        pickle.dump(ndcgs, f)

    return server_model.state_dict()


def single_train_round(server_model: torch.nn.Module,
                       clients: List[Client]) -> Tuple[List[collections.OrderedDict], float]:
    """
    :param server_model: server model to train
    :param clients: list of `Client` objects, `Client` must implement `train()` method
    :return: weights of each client models as a list

    single round of federated training, trains all clients in `clients` locally
    """
    client_weights = list()
    mean_loss = 0
    for client in clients:
        server_model_copy = copy.deepcopy(server_model)
        weights, loss = client.train(server_model_copy)
        mean_loss += float(loss.cpu().detach().numpy())
        client_weights.append(weights)
    return client_weights, mean_loss / len(client_weights)


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
