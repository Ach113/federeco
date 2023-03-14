import time
import copy
import tqdm
import collections
from typing import List

from dataset import *
from client import Client
from eval import evaluate_model
from model import NeuralCollaborativeFiltering as NCF

from config import *


def run_server(num_clients: int, num_rounds: int, save: bool):
    """
    defines server side ncf model and initiates the training process
    saves the trained model if appropriate parameter is set
    """
    # define server side model
    server_model = NCF(NUM_USERS, NUM_ITEMS)
    server_model.to(DEVICE)
    # train
    trained_model = training_process(server_model, num_clients, num_rounds)

    if save:
        torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)


def sample_clients(num_clients: int) -> List[Client]:
    """
    initialize `num_clients` clients along with their respective data
    """
    clients = list()
    # load entire client dataset
    client_dataset = load_client_train_data()
    # sample random client ids
    client_ids = np.random.choice(range(NUM_USERS), size=num_clients, replace=False)
    for client_id in client_ids:
        c = Client(client_id)
        c.set_client_data(client_dataset[client_id])
        clients.append(c)

    return clients


def training_process(server_model: torch.nn.Module,
                     num_clients: int,
                     num_rounds: int) -> torch.nn.Module:
    """
    per single training round:
        1. samples `num_clients` clients
        2. trains each client locally `LOCAL_EPOCHS` number of times
        3. aggregates weights across `num_clients` clients and sets them to server model
    returns trained keras model
    """
    test_data, negatives = load_test_file(), load_negative_file()

    for _ in tqdm.tqdm(range(num_rounds)):
        clients = sample_clients(num_clients)
        w = single_train_round(server_model, clients)
        updated_server_weights = federated_averaging(w)
        server_model.load_state_dict(updated_server_weights)

    t = time.time()
    users, items = zip(*test_data)
    hr, ndcg = evaluate_model(server_model, users, items, negatives, k=10)
    print(f'hit rate: {hr:.2f}, normalized discounted cumulative gain: {ndcg:.2f} [{time.time() - t:.2f}]s')

    return server_model


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

