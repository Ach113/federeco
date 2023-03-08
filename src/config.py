TRAIN_DATA_PATH = 'data/ml-1m.train.rating'
TEST_DATA_PATH = 'data/ml-1m.test.rating'
NEGATIVE_DATA_PATH = 'data/ml-1m.test.negative'
MODEL_SAVE_PATH = 'pretrained/ncf.h5'


NUM_USERS = 6040
NUM_ITEMS = 3706
NUM_NEGATIVES = 4

MODEL_PARAMETERS = {
    'default_model': {
        'mf_dim': 8,
        'layers': [64, 32, 16, 8],
        'reg_layers': [0, 0, 0, 0],
        'reg_mf': 0
    },
    'FedNCF': {
        'mf_dim': 16,
        'layers': [64, 32, 16, 8],
        'reg_layers': [0, 0, 0, 0],
        'reg_mf': 0
    },
}

BATCH_SIZE = 32
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.001
