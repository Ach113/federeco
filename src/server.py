import time
import tqdm
import tensorflow as tf
from typing import List
from tensorflow.keras.optimizers import Adam

from dataset import *
from client import Client
from eval import evaluate_model
from model import collaborative_filtering_model

from config import *


def run_server(num_clients: int, num_rounds: int, save: bool):
    """
    defines server side ncf model and initiates the training process
    saves the trained model if appropriate parameter is set
    """
    # define server side model
    server_model = collaborative_filtering_model(NUM_USERS, NUM_ITEMS)
    server_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=0.5), loss='binary_crossentropy')

    # train
    model = training_process(server_model, num_clients, num_rounds)

    if save:
        model.save(MODEL_SAVE_PATH)


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


def training_process(server_model: tf.keras.models.Model,
                     num_clients: int,
                     num_rounds: int) -> tf.keras.models.Model:
    """
    per single training round:
        1. samples `num_clients` clients
        2. trains each client locally `LOCAL_EPOCHS` number of times
        3. aggregates weights across `num_clients` clients and sets them to server model
    returns trained keras model
    """
    test_data, negative_data = load_test_file(), load_negative_file()

    for _ in tqdm.tqdm(range(num_rounds)):
        clients = sample_clients(num_clients)
        w = single_train_round(server_model, clients)
        updated_server_weights = federated_averaging(w)
        server_model.set_weights(updated_server_weights)

    t = time.time()
    hr, ndcg = evaluate_model(server_model, test_data, negative_data, 10)
    print(f'hit rate: {hr:.2f}, normalized discounted cumulative gain: {ndcg:.2f} [{time.time() - t:.2f}]s')

    return server_model


def single_train_round(server_model: tf.keras.models.Model,
                       clients: List[Client]) -> List[np.ndarray]:
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


def federated_averaging(client_weights: List[np.ndarray]) -> np.ndarray:
    """
    calculates the average of client weights
    """
    return np.sum(client_weights, axis=0) / len(client_weights)

