from federeco.models import NeuralCollaborativeFiltering as NCF
from federeco.train import training_process
from federeco.eval import evaluate_model
from federeco.config import DEVICE
from dataset import Dataset
from client import Client
from typing import List
import torch.nn
import time
import os


def run_server(dataset: Dataset, num_clients: int, epochs: int, path: str, save: bool) -> torch.nn.Module:
    """
    defines server side ncf model and initiates the training process
    saves the trained model at indicated path
    """
    # define server side model
    server_model = NCF(dataset.num_users, dataset.num_items)
    server_model.to(DEVICE)

    # if pretrained model already exists, loads its weights
    # if not, initiates the training process
    if os.path.exists(path):
        trained_weights = torch.load(path)
    else:
        clients = initialize_clients(dataset)
        trained_weights = training_process(server_model, clients, num_clients, epochs)

    if save:
        torch.save(trained_weights, path)
    # load server model's weights to generate recommendations
    server_model.load_state_dict(trained_weights)

    # evaluate the model
    test_data, negatives = dataset.load_test_file(), dataset.load_negative_file()
    t = time.time()
    users, items = zip(*test_data)
    hr, ndcg = evaluate_model(server_model, users, items, negatives, k=10)
    print(f'hit rate: {hr:.2f}, normalized discounted cumulative gain: {ndcg:.2f} [{time.time() - t:.2f}]s')

    return server_model


def initialize_clients(dataset: Dataset) -> List[Client]:
    """
    creates `Client` instance for each `client_id` in dataset
    returns the list of clients
    """
    clients = list()
    client_dataset = dataset.load_client_train_data()

    for client_id in range(dataset.num_users):
        c = Client(client_id)
        c.set_client_data(client_dataset[client_id])
        clients.append(c)
    return clients
