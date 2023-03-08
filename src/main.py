import time
from tensorflow.keras.optimizers import Adam

from model import collaborative_filtering_model
from dataset import load_movie_data, get_train_instances

from config import *


def main():
    # import entire dataset
    t0 = time.time()
    train_data = load_movie_data()
    # convert dataset to appropriate format
    user_input, item_input, labels = get_train_instances(train_data)
    print(f"data load complete [{time.time() - t0:.2f}s].")
    # define and compile the NCF model
    t1 = time.time()
    server_model = collaborative_filtering_model(NUM_USERS, NUM_ITEMS)
    server_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=0.5), loss='binary_crossentropy')
    print(f"model construction complete [{time.time() - t1:.2f}s].")


if __name__ == '__main__':
    main()
