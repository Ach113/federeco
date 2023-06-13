import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_NEGATIVES = 4
