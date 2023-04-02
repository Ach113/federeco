from torch.nn import functional as F
from typing import Optional, Tuple
from torch import Tensor
import torch

from config import MODEL_PARAMETERS, DEVICE


class NeuralCollaborativeFiltering(torch.nn.Module):

    def __init__(self, num_users: int, num_items: int):
        super().__init__()
        params = MODEL_PARAMETERS['FedNCF']
        layers = params['layers']
        mf_dim = params['mf_dim']
        mlp_dim = int(layers[0] / 2)

        self.mf_embedding_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim, device=DEVICE)
        self.mf_embedding_item = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim, device=DEVICE)

        self.mlp_embedding_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=mlp_dim, device=DEVICE)
        self.mlp_embedding_item = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=mlp_dim, device=DEVICE)

        self.mlp = torch.nn.ModuleList()
        current_dim = 32
        for idx in range(1, len(layers)):
            self.mlp.append(torch.nn.Linear(current_dim, layers[idx]))
            current_dim = layers[idx]
            self.mlp.append(torch.nn.ReLU())
        self.output_layer = torch.nn.Linear(in_features=24, out_features=1, device=DEVICE)

    def forward(self, user_input: Tensor,
                item_input: Tensor,
                target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[float]]:
        # matrix factorization
        mf_user_latent = torch.nn.Flatten()(self.mf_embedding_user(user_input))
        mf_item_latent = torch.nn.Flatten()(self.mf_embedding_user(item_input))
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)
        # mlp
        mlp_user_latent = torch.nn.Flatten()(self.mf_embedding_user(user_input))
        mlp_item_latent = torch.nn.Flatten()(self.mf_embedding_item(item_input))
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=1)

        for layer in self.mlp:
            mlp_vector = layer(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=1)
        logits = self.output_layer(predict_vector)

        loss = None
        if target is not None:
            target = target.view(target.shape[0], 1).to(torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, target)

        logits = torch.nn.Sigmoid()(logits)

        return logits, loss
