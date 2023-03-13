import torch
from torch import Tensor
from torch.nn import functional as F

from config import *


# TODO: add proper parameters to the model


class NeuralCollaborativeFiltering(torch.nn.Module):

    def __init__(self, num_users: int, num_items: int):
        super().__init__()
        params = MODEL_PARAMETERS['FedNCF']
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = params['mf_dim']
        self.layers = params['layers']
        self.reg_layers = params['reg_layers']
        self.reg_mf = params['reg_mf']

        self.mf_embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.mf_dim)
        self.mf_embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.mf_dim)

        self.mlp_embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=int(self.layers[0] / 2))
        self.mlp_embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=int(self.layers[0] / 2))


    def forward(self, user_input: Tensor, item_input: Tensor, target: Tensor):
        # matrix factorization
        mf_user_latent = torch.nn.Flatten()(self.mf_embedding_user(user_input))
        mf_item_latent = torch.nn.Flatten()(self.mf_embedding_item(item_input))
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)

        # mlp
        mlp_user_latent = torch.nn.Flatten()(self.mlp_embedding_user(user_input))
        mlp_item_latent = torch.nn.Flatten()(self.mlp_embedding_item(item_input))
        mlp_vector = torch.mul(mlp_user_latent, mlp_item_latent)

        for idx in range(1, len(self.layers)):
            layer = torch.nn.Linear(mlp_vector.shape[1], self.layers[idx])
            mlp_vector = torch.nn.ReLU()(layer(mlp_vector))

        predict_vector = torch.cat([mf_vector, mlp_vector], axis=1)
        logits = torch.nn.Linear(in_features=predict_vector.shape[1], out_features=1)(predict_vector)
        logits = torch.nn.Sigmoid()(logits)

        target = target.view(target.shape[0], 1).to(torch.float32)
        loss = F.binary_cross_entropy(logits, target)

        return logits, loss
