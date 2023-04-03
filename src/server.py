from model import NeuralCollaborativeFiltering as NCF
from eval import evaluate_model
from dataset import Dataset
from config import DEVICE
from client import Client
from typing import List
import collections
import numpy as np
import torch.nn
import time
import tqdm
import copy
import os


def run_server(dataset: Dataset, num_clients: int, num_rounds: int, path: str) -> torch.nn.Module:
    """
    defines server side ncf model and initiates the training process
    saves the trained model at indicated path
    """
    # define server side model
    server_model = NCF(dataset.num_users+1, dataset.num_items+1)
    server_model.to(DEVICE)

    # if pretrained model already exists, loads its weights
    # if not, initiates the training process
    trained_weights = torch.load(path) if os.path.exists(path) \
        else training_process(dataset, server_model, num_clients, num_rounds)

    torch.save(trained_weights, path)
    # load server model's weights to generate recommendations
    server_model.load_state_dict(trained_weights)

    return server_model


def sample_clients(dataset: Dataset, num_clients: int) -> List[Client]:
    """
    initialize `num_clients` clients along with their respective data
    """
    clients = list()
    # load entire client dataset
    client_dataset = dataset.load_client_train_data()
    # sample random client ids
    client_ids = np.random.choice(range(dataset.num_users), size=num_clients, replace=False)
    for client_id in client_ids:
        c = Client(client_id)
        c.set_client_data(client_dataset[client_id])
        clients.append(c)

    return clients


def training_process(dataset: Dataset,
                     server_model: torch.nn.Module,
                     num_clients: int,
                     num_rounds: int) -> collections.OrderedDict:
    """
    per single training round:
        1. samples `num_clients` clients
        2. trains each client locally `LOCAL_EPOCHS` number of times
        3. aggregates weights across `num_clients` clients and sets them to server model
    returns weights of a trained model
    """
    test_data, negatives = dataset.load_test_file(), dataset.load_negative_file()

    for _ in tqdm.tqdm(range(num_rounds)):
        clients = sample_clients(dataset, num_clients)
        w = single_train_round(server_model, clients)
        updated_server_weights = federated_averaging(w)
        server_model.load_state_dict(updated_server_weights)

    t = time.time()
    users, items = zip(*test_data)
    hr, ndcg = evaluate_model(server_model, users, items, negatives, k=10)
    print(f'hit rate: {hr:.2f}, normalized discounted cumulative gain: {ndcg:.2f} [{time.time() - t:.2f}]s')

    return server_model.state_dict()


def single_train_round(server_model: torch.nn.Module,
                       clients: List[Client]) -> List[collections.OrderedDict]:
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
