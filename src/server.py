import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from dataset import *
from client import Client
from eval import evaluate_model
from model import collaborative_filtering_model

from config import *


def run_server(num_clients: int, num_rounds: int, save: bool):
    # define server side model
    server_model = collaborative_filtering_model(NUM_USERS, NUM_ITEMS)
    server_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=0.5), loss='binary_crossentropy',
                         metrics=['accuracy'])
    # train
    model = training_process(server_model, num_clients, num_rounds)

    if save:
        model.save(MODEL_SAVE_PATH)


def sample_clients(n_clients: int):
    clients = list()
    # load entire client dataset
    client_dataset = load_client_train_data()
    # sample random client ids
    client_ids = np.random.choice(range(NUM_USERS), size=n_clients, replace=False)
    for client_id in client_ids:
        c = Client(client_id)
        c.set_client_data(client_dataset[client_id])
        clients.append(c)

    return clients


def training_process(server_model: tf.keras.models.Model, num_clients: int, num_rounds: int):
    test_data, negative_data = load_test_file(), load_negative_file()
    for _ in tqdm.tqdm(range(num_rounds)):
        clients = sample_clients(num_clients)
        w = single_train_round(server_model, clients)
        updated_server_weights = federated_averaging(w)
        server_model.set_weights(updated_server_weights)

    hr, ndcg = evaluate_model(server_model, test_data, negative_data, 10)
    print(f'hit rage: {hr[-1]:.2f}, normalized discounted cumulative gain: {ndcg[-1]:.2f}')

    return server_model


def single_train_round(server_model, clients):
    client_weights = list()
    for client in clients:
        weights = client.train(server_model)
        client_weights.append(weights)
    return client_weights


def federated_averaging(client_weights):
    client_num = len(client_weights)
    assert client_num != 0
    w = client_weights[0]
    for i in range(1, client_num):
        w += client_weights[i]
    w = w / client_num

    return w
