# federeco
Neural collaborative filtering recommendation system utilizing federated averaging for data privacy.
Implementation is done using both Tensorflow/Keras and PyTorch frameworks. 
Main branch includes Pytorch implementation, with Tensorflow implementation on `tensorflow` branch.
# Requirements
numpy 1.24.0 \
torch 2.0.0 \
pandas 2.0.0 \
tqdm 4.64.0 \
scipy 1.9.3 
# Install
Clone the repository:\
`git clone https://github.com/Ach113/federeco` 

Install the dependencies: \
`pip install -r requirements.txt` 

Note: `requirements.txt` does not include `torch`, you must install it separately. \
You can find the appropriate installation command [here](https://pytorch.org/get-started/locally/).

# Usage
Package itself is located in the `src` folder as `src/federeco`. \
All the other files are drivers to instantiate clients and run the training process from a server.
```python
from federeco.models import NeuralCollaborativeFiltering as NCF
from federeco.train import training_process
from federeco.eval import evaluate_model

# define client class
class Client:
    def __init__(self, client_id):
        pass
    
    def train(self):
        # method that trains client locally
        pass

model = NCF(num_users, num_items)
clients =  # list of Client objects #
state_dict = training_process(
    model,  # server-side model
    clients,  # list of all clients in the system
    num_clients,  # number of clients to sample per epoch
    epochs  # number of training epochs
) 
```
    
# Dataset
The program uses [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
and [Pinterest](https://paperswithcode.com/dataset/pinterest) datasets. \
Processed versions of the datasets are provided in `data` folder.

# Parameters
Training parameters, such as learning rate, number of global & local epochs, number of negative samples are all taken from 
[Federated Neural Collaborative Filtering paper](https://arxiv.org/abs/2106.04405). \
Parameters, along with file paths and other configurations are provided in `config.py` file.

# Planned features
1. way to add new users/items into the system
2. add other types of models beside NCF

# References
[[1]](https://arxiv.org/pdf/1708.05031.pdf) X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural Collaborative
Filtering,” Proceedings of the 26th International Conference on World Wide Web -
WWW ’17, 2017, doi: 10.1145/3038912.3052569 \
[[2]](https://arxiv.org/pdf/2106.04405.pdf) V. Perifanis and P. S. Efraimidis, “Federated Neural Collaborative Filtering,” Knowledge-Based Systems, vol. 242, p. 108441, Apr. 2022